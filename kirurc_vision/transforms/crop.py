from __future__ import annotations

from typing import Sequence

import numpy as np
import open3d as o3d
from open3d.geometry import PointCloud
from torch_geometric.data import Data

from .open3d_transform import Open3DTransform


class Crop(Open3DTransform):
    def __init__(
        self,
        center: Sequence | np.ndarray,
        extent: Sequence | np.ndarray,
        rotation: Sequence | np.ndarray | None = None,
    ) -> None:
        center = np.array(center, dtype=np.float64)
        extent = np.array(extent, dtype=np.float64)
        rotation = (
            np.array(rotation, dtype=np.float64) if rotation is not None else None
        )

        if center.shape != (3,):
            raise ValueError(f"Expected center of shape (3,), not {center.shape}")
        if extent.shape != (3,):
            raise ValueError(f"Expected extent of shape (3,), not {extent.shape}")
        if rotation is not None and rotation.shape != (3, 3):
            raise ValueError(f"Expected rotation of shape (3, 3), not {rotation.shape}")

        self.center = center
        self.extent = extent
        self.rotation = rotation

    def _transform_pcd(self, pcd: PointCloud, data: Data) -> tuple[PointCloud, Data]:
        # crop the point cloud
        if self.rotation is None:
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(self.center - self.extent / 2),
                max_bound=(self.center + self.extent / 2),
            )
        else:
            bounding_box = o3d.geometry.OrientedBoundingBox(
                center=self.center,
                R=self.rotation,
                extent=self.extent,
            )

        pcd = pcd.crop(bounding_box)
        return pcd, data
