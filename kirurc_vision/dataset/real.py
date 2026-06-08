import logging
import os.path as osp
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch_geometric.data import Data

from .base import KirurcDataset

log = logging.getLogger(__name__)


class RealPlyDataset(KirurcDataset):
    def _verify_files(self) -> None:
        if osp.exists(self.raw_dir):
            raw_dir = Path(self.raw_dir)
            target1_files = list(raw_dir.glob("*_target1.ply"))
            target2_files = list(raw_dir.glob("*_target2.ply"))
            self.ply_files = sorted(
                file
                for file in raw_dir.glob("*.ply")
                if file not in target1_files and file not in target2_files
            )

            log.debug(f"Found {len(self.ply_files)} samples in the real dataset.")

            if self.take_first_n is not None:
                self.ply_files = self.ply_files[: self.take_first_n]
                target1_files = target1_files[: self.take_first_n]
                target2_files = target2_files[: self.take_first_n]
                log.debug(
                    f"Using only first {len(self.ply_files)} samples of real dataset"
                )

            if not (len(self.ply_files) == len(target1_files) == len(target2_files)):
                raise RuntimeError(
                    "Expected 2 target ply files for each raw ply file, but "
                    f"received {len(self.ply_files)} raw files, {len(target1_files)} "
                    f"files for target 1 and {len(target2_files)} files for target 2."
                )

            for raw_file in self.ply_files:
                if raw_file.with_stem(raw_file.stem + "_target1") not in target1_files:
                    raise RuntimeError(
                        f"Could not find {raw_file.with_stem(raw_file.stem + '_target1')} among target1 files"
                    )
                if raw_file.with_stem(raw_file.stem + "_target2") not in target2_files:
                    raise RuntimeError(
                        f"Could not find {raw_file.with_stem(raw_file.stem + '_target2')} among target2 files"
                    )

            self._processed_file_names = [
                ply_file.stem + ".data" for ply_file in self.ply_files
            ]

        elif osp.exists(self.processed_dir):
            processed_dir = Path(self.processed_dir)
            self._processed_file_names = sorted(
                file.name for file in processed_dir.glob("*.data")
            )
            if len(self) == 0:
                raise RuntimeError(f"No processed data found at {processed_dir}")
            log.info(
                f"Found no raw validation data, so using {len(self)} "
                "processed data points instead."
            )
        else:
            raise FileNotFoundError(
                f"Found neither raw nor processed data at {self.root}"
            )

    def _load_one(self, idx: int) -> Data:
        raw_file = self.ply_files[idx]
        target1 = raw_file.with_stem(raw_file.stem + "_target1")
        target2 = raw_file.with_stem(raw_file.stem + "_target2")
        pcd = o3d.io.read_point_cloud(str(raw_file))
        target_segments = {
            1: o3d.io.read_point_cloud(str(target1)),
            2: o3d.io.read_point_cloud(str(target2)),
        }

        pos = torch.from_numpy(np.asarray(pcd.points).astype(np.float32))
        target_segments = {
            name: torch.from_numpy(np.asarray(segment.points).astype(np.float32))
            for name, segment in target_segments.items()
        }

        sample = Data(
            pos=pos,
            target_segments=target_segments,
            y=torch.zeros(pos.shape[:1], dtype=torch.long),
            file=raw_file,
        )
        return sample
