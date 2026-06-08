import csv
import logging
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import hydra
import open3d as o3d
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from kirurc_vision.builder import (
    build_model,
    build_synthetic_dataset,
    build_validation_dataset,
)
from kirurc_vision.transforms.normalize_scale import unnormalize_batch
from kirurc_vision.utils.o3d_utils import data_to_pointcloud

log = logging.getLogger("__main__")

HEADER = [
    "Run ID",
    "Tag",
    "mIoU",
    "Target mIoU",
]


def results_to_row(results):
    return [
        results["miou"],
        results["target_miou"],
    ]


def results_to_row_formatted(results):
    return [
        f"{results['miou']:.3f}",
        f"{results['target_miou']:.3f}",
    ]


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate_pretrained")
def main(config: DictConfig) -> None:

    # build dataset
    dataset_cfg = config["test_data"]
    if dataset_cfg["type"] == "synthetic":
        dataset = build_synthetic_dataset(dataset_cfg)
    elif dataset_cfg["type"] == "real":
        dataset = build_validation_dataset(dataset_cfg)
    else:
        raise ValueError(f"Unknown dataset type '{dataset_cfg['type']}'")

    if (start_at := config.get("start_at")) is not None:
        if isinstance(start_at, int):
            split_index = start_at
        elif isinstance(start_at, float):
            n_samples = len(dataset)
            split_index = int(n_samples * start_at)
        dataset = dataset[split_index:]

    # determine i/o modes and create output folders
    save_csv = config.save_csv
    csv_path = Path(save_csv.path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_pkl = config.get("save_pkl", {})
    if to_pcd := config.get("to_pcd", {}):
        save_pcd = to_pcd.get("save_pcd", {})
    else:
        save_pcd = False

    # build dataloader
    if save_pkl or save_pcd:
        batch_size = 1  # can only show only point cloud at a time
        num_workers = 0  # serial data loading
    else:
        batch_size = config["batch_size"]
        num_workers = config["num_workers"]

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, batch_size),
    )

    # build metrics
    metrics = list(instantiate(config["metrics"]).values())
    device = torch.device(
        config["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    metrics = [metric.to(device) for metric in metrics]

    # get model artifact
    entity = config.wandb_entity
    project = config.wandb_project
    api = wandb.Api()

    # we get all runs that match the tags and filters at once
    filters = {
        "tags": {"$all": list(config.tags_all), "$nin": list(config.tags_not_in)},
    }

    # open to file in write mode once to ensure old results are overwritten
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)

    for i, run in enumerate(api.runs(f"{entity}/{project}", filters=filters)):

        # we assume two tags, and the name is the one that is not "final"
        name = next(tag for tag in run.tags if tag != "final")

        model_artifacts = [a for a in run.logged_artifacts() if a.type == "model"]

        if len(model_artifacts) == 1:
            model_artifact = model_artifacts[0]
        else:
            model_artifacts = sorted(model_artifacts, key=lambda a: a.version)
            model_artifact = model_artifacts[-1]
            log.debug(
                f"Run {run.id} has {len(model_artifacts)} logged model artifacts. Taking the latest version: {model_artifact.name}"
            )

        # get config used to train the model and build the model
        training_cfg = run.config
        model_config = training_cfg["model"]
        model_config["device"] = config["device"]
        model, device = build_model(model_config)

        # download the artifact (model state) and load it
        model_dir = Path(f"artifacts/{model_artifact.name}")
        model_artifact.download(root=str(model_dir))
        model_state_dict = torch.load(model_dir / "model.pt", device)
        model.load_state_dict(model_state_dict)

        # NOTE: fine-tuned models perform better in eval mode, while pretrained models perform
        # SIGNIFICANTLY better in train mode.
        # model.eval()

        results = evaluate_model(data_loader, model, metrics, device, config)
        log.info(f"Results from run {run.id} (method '{name}'): {results}")

        with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([run.id, name] + results_to_row(results))


def evaluate_model(dataloader, model, metrics, device, config):
    if save_pkl := config.get("save_pkl", {}):
        pkl_folder = Path(save_pkl.folder)
        pkl_folder.mkdir(parents=True, exist_ok=True)
    if to_pcd := config.get("to_pcd", {}):
        if save_pcd := to_pcd.get("save_pcd", {}):
            pcd_folder = Path(save_pcd.folder)
            pcd_folder.mkdir(parents=True, exist_ok=True)
            pcd_extension = save_pcd.get("extension", ".ply")
            if pcd_extension[0] != ".":
                pcd_extension = "." + pcd_extension
    else:
        save_pcd = False

    accumulated_results = defaultdict(list)

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            prediction = model(data)

        data["prediction"] = prediction
        data["predicted_labels"] = prediction.argmax(dim=-1)
        data = unnormalize_batch(data)

        # compute metrics
        batch_log_data = {"filenames": [path.name for path in data["file"]]}
        batch_log_data["sizes"] = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for metric in metrics:
            results = metric(data)
            batch_log_data |= results
            for key, value in results.items():
                accumulated_results[key].append(value)

        log.debug(batch_log_data)
        data = data.cpu()

        # save prediction to file as torch Data object
        if save_pkl:
            file = data["file"][0]  # assume batch size of 1
            pkl_file = pkl_folder / (file.stem + "_prediction.pkl")
            log.info(f"Saving pkl to {pkl_file}")
            torch.save(data, pkl_file)

        # convert to open3d pointcloud for saving or rendering
        if to_pcd:
            file = data["file"][0]  # assume batch size of 1

            if keys_to_skip := to_pcd.get("skip_keys", []):
                for key in keys_to_skip:
                    data.pop(key, None)

            pcd = data_to_pointcloud(
                data,
                colors_key=to_pcd.get("colors_key", "predicted_labels"),
                color_mapping=config["color_mapping"],
                n_spline_points=to_pcd.get("n_spline_points", 100),
            )

            # save prediction to file as ply pointcloud
            if save_pcd:
                pcd_file = pcd_folder / (file.stem + "_prediction" + pcd_extension)
                log.info(f"Saving pcd to {pcd_file}")
                o3d.io.write_point_cloud(
                    str(pcd_file), pcd, write_ascii=True, compressed=False
                )

    accumulated_results = {
        key: torch.tensor(value).nanmean().item()
        for key, value in accumulated_results.items()
    }

    return accumulated_results


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
