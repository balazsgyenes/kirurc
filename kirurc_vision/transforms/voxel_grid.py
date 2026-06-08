import torch
from torch_cluster import grid_cluster
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MaxAggregation, MeanAggregation
from torch_geometric.transforms import BaseTransform


class VoxelGrid(BaseTransform):
    def __init__(self, size: float):
        self.size = torch.tensor([size, size, size])
        self.mean_aggr = MeanAggregation()
        self.max_aggr = MaxAggregation()

    def __call__(self, data: Data) -> Data:
        # assign voxel index to each point
        cluster_indices = grid_cluster(data.pos, self.size)
        # calculate mean position for each voxel
        pos = self.mean_aggr(data.pos, cluster_indices)

        # calculate the mean value for the label and round it to the nearest integer (half-to-even)
        # Data objects always have a "y" attribute (initially set to None), so we don't need to use hasattr()
        if data.y is not None:
            y = self.mean_aggr(data.y, cluster_indices)

        # mask out all [0, 0, 0] points (voxels without any points within)
        mask = pos.sum(dim=1) != 0
        data.pos = pos[mask]  # type: ignore
        if data.y is not None:
            data.y = y[mask]  # type: ignore

        return data
