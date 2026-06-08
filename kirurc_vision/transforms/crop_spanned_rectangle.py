from __future__ import annotations

from typing import Sequence

import numpy as np
import open3d as o3d
from open3d.geometry import PointCloud
from torch_geometric.data import Data

from .open3d_transform import Open3DTransform


class CropSpannedRectangle(Open3DTransform):
    def __init__(
        self,
        spanning_points: Sequence[Sequence] | Sequence[np.ndarray] | np.ndarray,
        z_extent: float,
    ) -> None:
        spanning_points = np.array(spanning_points, dtype=np.float64)
        bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(
            points=o3d.utility.Vector3dVector(spanning_points),
            robust=True,
        )
        self.center = np.array(bounding_box.center)
        self.rotation = np.array(bounding_box.R)
        self.extent = np.array(bounding_box.extent)
        self.extent[np.argmin(self.extent)] = z_extent

    def _transform_pcd(self, pcd: PointCloud, data: Data) -> tuple[PointCloud, Data]:
        # crop the point cloud
        bounding_box = o3d.geometry.OrientedBoundingBox(
            center=self.center,
            R=self.rotation,
            extent=self.extent,
        )
        pcd = pcd.crop(bounding_box)
        return pcd, data
