from __future__ import annotations

import gc
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import torch
import wandb
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.loader import DataLoader

from kirurc_vision.metrics import Metric

log = logging.getLogger(__name__)


class Runner:
    def __init__(
        self,
        model: Module,
        device: torch.device,
        loss: Callable,
        optimizer: Optimizer,
        train_metrics: Sequence[Metric],
        eval_metrics: Sequence[Metric],
        scheduler: _LRScheduler,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_store_dir: Path,
        epochs: int,
        log_every_n_updates: int,
        test_every_n_epochs: int,
        save_every_n_epochs: int,
        garbage_collect_immediately: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.loss = loss.to(device)
        self.optimizer = optimizer
        self.train_metrics = [metric.to(device) for metric in train_metrics]
        self.eval_metrics = [metric.to(device) for metric in eval_metrics]
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.log_every_n_updates = log_every_n_updates
        self.test_every_n_epochs = test_every_n_epochs
        self.save_every_n_epochs = save_every_n_epochs
        self.gc_immediately = garbage_collect_immediately

        if not model_store_dir.is_dir():
            raise ValueError(
                "Please specify an existing directory for storing model snapshots."
            )
        self.model_store_path = model_store_dir / "model.pt"

    def run(self):
        wandb.save(
            str(self.model_store_path),
            base_path=str(self.model_store_path.parent),
        )
        wandb.save(
            str(self.model_store_path.with_name("*.backup")),
            base_path=str(self.model_store_path.parent),
        )

        updates = 0
        batch_losses = []

        for epoch in range(self.epochs):
            log.info(f"{epoch=} ")
            self.model.train()
            metrics: dict[str, list[float]] = defaultdict(list)

            for i, data in enumerate(self.train_loader):
                data = data.to(self.device)
                prediction = self.model(data)

                loss = self.loss(prediction, data.y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_losses.append(loss.detach().cpu())

                with torch.no_grad():
                    data["prediction"] = prediction.detach()
                    data["predicted_labels"] = prediction.argmax(dim=-1)

                    for metric in self.train_metrics:
                        for key, value in metric(data).items():
                            metrics[key].append(value)

                if self.gc_immediately:
                    # try to free up as much memory as possible
                    for var in "data", "prediction", "loss":
                        del locals()[var]
                    gc.collect()
                    torch.cuda.empty_cache()

                updates += 1
                if updates % self.log_every_n_updates == 0:
                    log_data = {
                        "updates": updates,
                        "epochs": float(epoch + (i + 1) / len(self.train_loader)),
                        "lr": self.scheduler.get_last_lr()[0],
                        "loss": torch.tensor(batch_losses).mean().item(),
                    }
                    log_data |= {
                        key: torch.tensor(value).mean().item()
                        for key, value in metrics.items()
                    }
                    log.info(log_data)
                    wandb.log(log_data)

                    batch_losses.clear()
                    metrics.clear()

            self.scheduler.step()

            if (epoch + 1) % self.test_every_n_epochs == 0:
                self._test()

            if (epoch + 1) % self.save_every_n_epochs == 0:
                torch.save(self.model.state_dict(), self.model_store_path)

                backup_path = self.model_store_path.with_suffix(f".{epoch}.backup")
                torch.save(self.model.state_dict(), backup_path)

        log_data = {
            "updates": updates,
            "epochs": float(epoch + 1),
            "lr": self.scheduler.get_last_lr()[0],
        }
        if len(batch_losses):
            # flush any eval data that hasn't been logged yet
            log_data["loss"] = torch.tensor(batch_losses).mean().item()
            log_data |= {
                key: torch.tensor(value).mean().item() for key, value in metrics.items()
            }
        log.info(log_data)
        wandb.log(log_data)

        torch.save(self.model.state_dict(), self.model_store_path)

        return self.model_store_path

    @torch.no_grad()
    def _test(self):
        log.info("Starting evaluation...")
        self.model.eval()
        metrics: dict[str, list[float]] = defaultdict(list)
        for data in self.test_loader:
            data = data.to(self.device)
            prediction = self.model(data)

            data["prediction"] = prediction
            data["predicted_labels"] = prediction.argmax(dim=-1)

            for metric in self.eval_metrics:
                for key, value in metric(data).items():
                    metrics[key].append(value)

        log_data = {
            f"validation_{key}": torch.tensor(value).nanmean().item()
            for key, value in metrics.items()
        }

        log.info(log_data)
        wandb.log(log_data, commit=False)
