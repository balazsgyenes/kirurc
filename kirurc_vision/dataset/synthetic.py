import json
import logging
import os.path as osp
from pathlib import Path

import torch
from torch_geometric.data import Data

from kirurc_vision.utils.exr import read_exr

from .base import KirurcDataset

log = logging.getLogger(__name__)


class SyntheticExrDataset(KirurcDataset):
    def _verify_files(self) -> None:
        if osp.exists(self.raw_dir):
            raw_dir = Path(self.raw_dir)
            self.target_files = sorted(raw_dir.glob("*target.json"))
            self.image_files = sorted(raw_dir.glob("*image.exr"))
            self.cam_params_file = raw_dir / "camera.json"

            if not self.cam_params_file.exists():
                raise RuntimeError(
                    f"Missing camera parameters file at {self.cam_params_file}!"
                )

            log.debug(
                f"Found {len(self.target_files)} samples in the synthetic dataset."
            )

            if self.take_first_n is not None:
                self.target_files = self.target_files[: self.take_first_n]
                self.image_files = self.image_files[: self.take_first_n]
                log.debug(
                    f"Using only first {len(self.image_files)} samples of dataset"
                )

            if len(self.target_files) != len(self.image_files):
                raise RuntimeError(
                    f"Found {len(self.target_files)} target files but "
                    f"{len(self.image_files)} image files at {self.root}!"
                )

            self._processed_file_names = [
                file.stem.removesuffix("_image") + ".data" for file in self.image_files
            ]

        elif osp.exists(self.processed_dir):
            processed_dir = Path(self.processed_dir)
            self._processed_file_names = sorted(
                file.name for file in processed_dir.glob("*.data")
            )
            if len(self) == 0:
                raise RuntimeError(f"No processed data found at {processed_dir}")
            log.info(
                f"Found no raw synthetic data, so using {len(self)} "
                "processed data points instead."
            )
        else:
            raise FileNotFoundError(
                f"Found neither raw nor processed data at {self.root}"
            )

    def _load_one(self, idx: int) -> Data:
        rgb, depth = read_exr(self.image_files[idx])

        with open(self.target_files[idx], "r") as f:
            targets = json.load(f)

        targets = {
            name: torch.tensor(point, dtype=torch.float32)
            for name, point in targets.items()
        }

        sample = Data(
            rgb=rgb,
            depth=depth,
            targets=targets,
            file=self.image_files[idx],
        )
        return sample
