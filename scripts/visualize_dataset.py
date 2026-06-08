import logging
import multiprocessing as mp
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import open3d as o3d
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from kirurc_vision.builder import (
    build_prediction_dataset,
    build_synthetic_dataset,
    build_validation_dataset,
)
from kirurc_vision.transforms.normalize_scale import unnormalize_batch
from kirurc_vision.utils.o3d_utils import data_to_pointcloud
from kirurc_vision.utils.regression import clean_regression_data

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="visualize_dataset")
def main(config: DictConfig) -> None:
    dataset_cfg = config["dataset"]
    if dataset_cfg["type"] == "synthetic":
        dataset = build_synthetic_dataset(dataset_cfg)
    elif dataset_cfg["type"] == "real":
        dataset = build_validation_dataset(dataset_cfg)
    elif dataset_cfg["type"] == "prediction":
        dataset = build_prediction_dataset(dataset_cfg)
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
    if to_pcd := config.get("to_pcd", {}):
        if save_pcd := to_pcd.get("save_pcd", {}):
            pcd_folder = Path(save_pcd.folder)
            pcd_folder.mkdir(parents=True, exist_ok=True)
            pcd_extension = save_pcd.get("extension", ".ply")
            if pcd_extension[0] != ".":
                pcd_extension = "." + pcd_extension
        show = to_pcd.get("show", False)
    else:
        save_pcd = show = False

    train_loader = DataLoader(
        dataset,
        batch_size=1,  # can only show/save only point cloud at a time
        shuffle=False,
        num_workers=0,  # serial data loading
    )

    for data in train_loader:
        data = clean_regression_data(data)
        data = unnormalize_batch(data)

        # convert to open3d pointcloud for saving or rendering
        if to_pcd:
            file = data["file"][0]  # assume batch size of 1

            if keys_to_skip := to_pcd.get("skip_keys", []):
                for key in keys_to_skip:
                    data.pop(key, None)

            pcd = data_to_pointcloud(
                data,
                colors_key=to_pcd.get("colors_key", "y"),
                color_mapping=config["color_mapping"],
                n_spline_points=to_pcd.get("n_spline_points", 100),
            )

            # save prediction to file as ply pointcloud
            if save_pcd:
                pcd_file = pcd_folder / (file.stem + pcd_extension)
                log.info(f"Saving pcd to {pcd_file}")
                o3d.io.write_point_cloud(
                    str(pcd_file), pcd, write_ascii=True, compressed=False
                )

            # render prediction
            if show:
                log.info(f"Rendering prediction from {file.name}")
                o3d.visualization.draw([pcd], **config.get("open3d_draw", {}))

        elif ("rgb" in data) and ("depth" in data):
            figure = plt.figure()
            figure.add_subplot(1, 2, 1)
            plt.imshow(data["rgb"])
            figure.add_subplot(1, 2, 2)
            plt.imshow(data["depth"])
            plt.show()

        elif ("rgb" in data) or ("depth" in data):
            image = data.get("rgb", None) or data.get("depth", None)
            plt.imshow(image)
            plt.show()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
