from __future__ import annotations

import logging
import os.path as osp
from pathlib import Path
from typing import Any, Callable

import numpy as np
import open3d as o3d
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn

from .base import KirurcDataset
from .real import RealPlyDataset

log = logging.getLogger(__name__)


class SavedPredictionsDataset(KirurcDataset):
    def __init__(
        self,
        root: Path,
        labeled_dataset: RealPlyDataset,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        pre_filter: Callable[..., Any] | None = None,
        take_first_n: int | None = None,
        num_workers: int | None = None,
        force_reprocess: bool = False,
        debug_transforms: bool = False,
    ) -> None:

        self.labels = labeled_dataset

        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            take_first_n,
            num_workers,
            force_reprocess,
            debug_transforms,
        )

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "predictions")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed_predictions")

    def _verify_files(self) -> None:
        assert osp.exists(self.raw_dir)
        prediction_files = sorted(Path(self.raw_dir).glob("*.ply"))

        raw_files = self.labels.ply_files
        log.debug(f"Found {len(raw_files)} labeled samples in the real dataset.")

        # sort all files by filename
        # because the files are named by timestamp, this sorts all files in chronological order
        files = sorted(prediction_files + raw_files, key=lambda file: file.name)

        # for each raw point cloud, append the file right after it to the list of predictions
        self.prediction_plys = []
        for i, file in enumerate(files):
            if not file.name.endswith("_raw.ply"):
                continue

            prediction_ply = files[i + 1]
            assert prediction_ply.name.endswith("_prediction.ply")
            self.prediction_plys.append(prediction_ply)

        if len(self.prediction_plys) != len(self.labels):
            raise RuntimeError(
                f"Found {len(self.prediction_plys)} prediction files but "
                f"{len(self.labels)} labels at {self.root}!"
            )

        log.debug(
            f"Found {len(self.prediction_plys)} saved predictions in the saved predictions dataset."
        )

        self._processed_file_names = [
            ply_file.stem + ".data" for ply_file in self.prediction_plys
        ]

    def _load_one(self, idx: int) -> Data:

        # indexing applies any transform function in the dataset, but the experiments dataset
        # doesn't have any
        labels = self.labels[idx]

        # undo normalization to enable matching points
        labels_pos = labels.pos
        if "scale" in labels:
            labels_pos /= labels.scale
        if "center" in labels:
            labels_pos += labels.center

        raw_file = self.prediction_plys[idx]
        pcd = o3d.io.read_point_cloud(str(raw_file))
        pos = torch.from_numpy(np.asarray(pcd.points).astype(np.float32))
        colors = torch.from_numpy(np.asarray(pcd.colors).astype(np.float32))
        colors = (colors * 255).to(torch.uint8)

        # translate saved pointcloud with iterative closest point to minimize chamfer
        # distance to labeled pointcloud
        # also return correspondences for computing labels of saved pointcloud
        pos, indices_labels = icp(
            source=pos, target=labels_pos, n_iterations=2, filename=raw_file
        )

        sample = Data(
            pos=pos,
            colors=colors,
            y=labels.y[indices_labels],
            labels_y=labels.y,
            labels_pos=labels_pos,
            file=raw_file,
        )

        for field in ("t1_spline", "t2_spline", "t1_skeleton", "t2_skeleton"):
            if field in labels:
                sample[field] = labels[field]

        return sample


def icp(
    source, target, n_iterations: int, filename: str
) -> tuple[torch.Tensor, torch.Tensor]:

    assert n_iterations >= 0
    while True:
        # for each point in source, find nearest point in the target pointcloud
        indices_source, indices_target = knn(x=target, y=source, k=1)
        nearest_target = target[indices_target]

        # compute the mean offset from source to target in each dimension
        offset = (nearest_target - source).mean(dim=0)
        mean_offset = torch.linalg.norm(offset)

        if n_iterations <= 0:
            log.debug(
                f"Prediction {filename}: mean offset to labeled point cloud is {mean_offset * 1e3:.3f}mm."
            )
            break

        source += offset
        n_iterations -= 1

        offset_str = ", ".join(
            f"{dim}={val:.3f}" for dim, val in zip(("x", "y", "z"), offset * 1000)
        )
        log.debug(
            f"Prediction {filename}: mean offset to labeled point cloud is {mean_offset * 1e3:.3f}mm. "
            f"Adding offset of ({offset_str})mm"
        )

    return source, indices_target
