import logging
from typing import Sequence

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from kirurc_vision.utils.splines import (
    LaplacianConfig,
    NoSpline,
    SplineConfig,
    compute_spline_params,
    skeletonize_point_cloud,
)

log = logging.getLogger(__name__)


class ComputeReferenceSpline(BaseTransform):
    def __init__(
        self,
        target_classes: Sequence[int],
        laplacian_config: LaplacianConfig,
        spline_config: SplineConfig,
        cleanup: bool = True,
    ) -> None:
        self.target_classes = target_classes
        self.laplacian_config = laplacian_config
        self.spline_config = spline_config
        self.cleanup = cleanup

    def __call__(self, data: Data) -> Data:
        for label in self.target_classes:
            points = data.pos[data.y == label]

            log.debug(f"{data.file.name}: Target {label} has {len(points)} points.")

            # skeleton_points may be none if skeletonization fails
            skeleton_points = skeletonize_point_cloud(
                points.cpu().numpy(),
                self.laplacian_config,
            )
            if skeleton_points is None:
                # if you assign data[key] = None, it just tries to delete the key
                # instead, we need to use a singleton that acts like None
                # omitting the entry messes up batching, so we need to assign something
                data[f"t{label}_spline"] = NoSpline
                continue

            log.debug(
                f"{data.file.name}: Skeleton of Target {label} has {len(skeleton_points)} points."
            )

            if not self.cleanup:
                data[f"t{label}_skeleton"] = [torch.from_numpy(skeleton_points)]

            spline = compute_spline_params(skeleton_points, self.spline_config)

            data[f"t{label}_spline"] = spline

        return data
