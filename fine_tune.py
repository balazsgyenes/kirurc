from datetime import datetime
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader

from kirurc_vision.builder import build_model, build_validation_dataset
from kirurc_vision.runner import Runner


@hydra.main(version_base=None, config_path="conf", config_name="fine_tune")
def main(config: DictConfig) -> None:
    # build training dataset and dataloader
    num_workers = min(config["num_workers"], config["batch_size"])

    training_dataset = build_validation_dataset(config["training_data"])
    training_loader = DataLoader(
        training_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # build test dataset and dataloader
    test_dataset = build_validation_dataset(config["test_data"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # remove wandb config from config before saving config to wandb
    with open_dict(config):
        wandb_config = config.pop("wandb", {})
        notes = wandb_config.pop("notes", None)
        name = wandb_config.pop("name", None)

    if isinstance(tags, str):
        tags = [tags]

    wandb.init(
        entity="gyenes",
        project="fine-tuning",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        save_code=True,  # save script used to start training, git commit, and patch
        job_type="fine-tune",
        name=name,
        tags=tags,
        notes=notes,
    )

    # write relevant info to config
    wandb.config["training_data"]["size"] = len(training_dataset)

    # load pretrained model artifact
    identifier = config["model_artifact"]
    if identifier is not None:
        model_artifact = wandb.run.use_artifact(f"kirurc/kirurc/{identifier}:latest")

        # build model
        pretrain_cfg = model_artifact.logged_by().config
        model_config = pretrain_cfg["model"]
        model_config["device"] = config["device"]
        model, device = build_model(model_config)
        wandb.config["model"] = model_config

        # add pretraining config to fine-tuning config
        del pretrain_cfg["model"]
        pretrain_cfg.pop("validation_dataset", None)
        wandb.config["pretraining"] = pretrain_cfg

        # load saved model state
        model_dir = Path(model_artifact.download())
        model_state_dict = torch.load(model_dir / "model.pt", device)
        model.load_state_dict(model_state_dict)

        # freeze certain parts of the model
        if trainable_cfg := config.get("trainable_layers"):
            # freeze initial layers of transformer, only unfreezing final layers
            model.requires_grad_(False)

            train_mlp = trainable_cfg["mlp_head"] in ("train", "reinit")
            reset_mlp = trainable_cfg["mlp_head"] == "reinit"
            model.freeze_mlp_head(
                freeze=not train_mlp,
                reset_weights=reset_mlp,
            )

            if (n := trainable_cfg.get("n_final_up_layers", 0)) > 0:
                train_upsample = trainable_cfg["final_up_layers"] in ("train", "reinit")
                reset_upsample = trainable_cfg["final_up_layers"] == "reinit"
                model.freeze_last_n_upsample_layers(
                    n=n,
                    freeze=not train_upsample,
                    reset_weights=reset_upsample,
                )

    else:
        # initialize new model based on config
        model_config = config["model"]
        model, device = build_model(model_config)

    wandb.config["model_params"] = sum(p.numel() for p in model.parameters())
    wandb.config["learnable_model_params"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # build optimizer
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    # build LR scheduler
    base = config["scheduler"]["lr_epoch_base"]
    lr_lambda = lambda epoch: base**epoch
    scheduler = LambdaLR(optimizer, lr_lambda)

    # build loss function and metric
    n_classes = config["training_data"]["n_classes"]

    if (weights := config["class_weights"]) is None:
        counts = training_dataset.class_counts
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
    # TODO: implement saving best performance
    runner = Runner(
        model,
        device,
        loss,
        optimizer,
        train_metrics,
        eval_metrics,
        scheduler,
        training_loader,
        test_loader,
        model_store_dir,
        epochs=config["epochs"],
        log_every_n_updates=config["log_every_n_updates"],
        test_every_n_epochs=config["test_every_n_epochs"],
        save_every_n_epochs=config["save_every_n_epochs"],
    )

    model_store_path = runner.run()

    artifact = wandb.Artifact(f"model_{wandb.run.id}", "model")
    artifact.add_file(model_store_path)
    wandb.run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
