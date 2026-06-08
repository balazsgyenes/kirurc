from typing import Tuple

from open3d.geometry import PointCloud
from torch_geometric.data import Data

from .open3d_transform import Open3DTransform


class RemovePlane(Open3DTransform):
    def __init__(self,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations

    def _transform_pcd(self, pcd: PointCloud, data: Data) -> Tuple[PointCloud, Data]:
        # segment plane in the scene
        plane_parameters, inliers = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
        )
        pcd = pcd.select_by_index(indices=inliers, invert=True)
        return pcd, data
