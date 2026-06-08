from typing import Tuple

from open3d.geometry import PointCloud
from torch_geometric.data import Data

from .open3d_transform import Open3DTransform


class RemoveStatisticalOutliers(Open3DTransform):
    def __init__(self,
        nb_neighbors: int,
        std_ratio: float,
    ) -> None:
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def _transform_pcd(self, pcd: PointCloud, data: Data) -> Tuple[PointCloud, Data]:
        # remove statistical outliers
        pcd, inliers = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors,
            std_ratio=self.std_ratio,
        )
        return pcd, data
