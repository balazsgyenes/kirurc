from typing import Tuple

import numpy as np
import open3d as o3d
from open3d.geometry import PointCloud
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class Open3DTransform(BaseTransform):
    def __call__(self, data: Data) -> Data:
        points = o3d.utility.Vector3dVector(data["pos"].numpy())
        pcd = o3d.geometry.PointCloud(points)
        
        # reinterpret integer label as float color
        colors = data["y"].unsqueeze(-1).expand(-1, 3).to(torch.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors.numpy())

        assert np.asarray(pcd.colors).max() == data["y"].max(), "labels were clipped"

        pcd, data = self._transform_pcd(pcd, data)

        # read fields back into Data object
        data["pos"] = torch.from_numpy(np.asarray(pcd.points).astype(np.float32))
        colors = torch.from_numpy(np.asarray(pcd.colors))
        data["y"] = colors[:, 0].to(torch.long)
        return data

    def _transform_pcd(self, pcd: PointCloud, data: Data) -> Tuple[PointCloud, Data]:
        raise NotImplementedError
