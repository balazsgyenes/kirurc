import logging
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np  # import numpy before torch for performance (don't ask me)
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader

from kirurc_vision.builder import build_model, build_synthetic_dataset
from kirurc_vision.runner import Runner

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="pretrain")
def main(config: DictConfig) -> None:
    # build synthetic training dataset and dataloader
    log.info("Dataset path is %s", config["synthetic_dataset"]["path"])
    dataset = build_synthetic_dataset(config["synthetic_dataset"])

    n_samples = len(dataset)
    split_index = int(n_samples * (1 - config["hold_out_ratio"]))
    dataset, validation_dataset = (dataset[:split_index], dataset[split_index:])

    # disable data augmentation for validation dataset
    validation_dataset.transform = None

    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=min(config["num_workers"], config["batch_size"]),
    )

    test_loader = DataLoader(
        validation_dataset,
        batch_size=config["test_batch_size"],
        shuffle=False,
        num_workers=min(config["num_workers"], config["test_batch_size"]),
    )

    # remove wandb config from config before saving config to wandb
    with open_dict(config):
        wandb_config = config.pop("wandb", {})
        tags = wandb_config.pop("tags", None)
        notes = wandb_config.pop("notes", None)
        name = wandb_config.pop("name", None)

    if isinstance(tags, str):
        tags = [tags]

    wandb.init(
        entity="kirurc",
        project="kirurc",
        config=OmegaConf.to_container(config, resolve=True),
        save_code=True,  # save script used to start training, git commit, and patch
        job_type="pretrain",
        name=name,
        tags=tags,
        notes=notes,
    )

    wandb.config["synthetic_dataset"]["size"] = len(dataset)
    wandb.config["synthetic_dataset"]["train_size"] = len(dataset)
    wandb.config["synthetic_dataset"]["validation_size"] = len(validation_dataset)

    # build model
    model, device = build_model(config["model"])

    # build optimizer
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    # build LR scheduler
    base = config["scheduler"]["lr_epoch_base"]
    lr_lambda = lambda epoch: base**epoch
    scheduler = LambdaLR(optimizer, lr_lambda)

    n_classes = config["synthetic_dataset"]["n_classes"]

    if (weights := config["class_weights"]) is None:
        counts = dataset.class_counts
        counts = counts.float() / counts.sum()
        weights = 1 / counts
        wandb.config.update({"class_weights": weights.tolist()}, allow_val_change=True)
    else:
        weights = torch.tensor(weights, dtype=torch.float32)

    if len(weights) != n_classes:
        raise ValueError(
            "Size of class weights doesn't match number of classes in dataset!"
        )

    # build loss function
    weights /= torch.sum(weights)  # normalize
    loss = instantiate(
        config["loss"]["args"],
        weight=weights,
    )

    # build metrics
    train_metrics = [
        f for f in instantiate(config["train_metrics"]).values() if f is not None
    ]
    eval_metrics = [
        f for f in instantiate(config["eval_metrics"]).values() if f is not None
    ]

    # set model store directory
    if wandb.run.disabled:
        model_store_dir = Path(f"models/{datetime.now().strftime('%Y-%m-%d_%H-%M')}")
    else:
        model_store_dir = Path(wandb.run.dir)
    model_store_dir.mkdir(parents=True, exist_ok=True)

    # build runner
    runner = Runner(
        model,
        device,
        loss,
        optimizer,
        train_metrics,
        eval_metrics,
        scheduler,
        train_loader,
        test_loader,
        model_store_dir,
        epochs=config["epochs"],
        log_every_n_updates=config["log_every_n_updates"],
        test_every_n_epochs=config["test_every_n_epochs"],
        save_every_n_epochs=config["save_every_n_epochs"],
    )

    model_store_path = runner.run()

    artifact = wandb.Artifact(f"pretrained_{wandb.run.id}", "model")
    artifact.add_file(model_store_path)
    wandb.run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
