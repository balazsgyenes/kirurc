from typing import List

import torch
from torch_geometric.nn import (PointNetConv, fps, global_max_pool,
                                knn_interpolate, radius)

import wandb


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn: torch.nn.Module):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class SAModule(torch.nn.Module):
    def __init__(self,
        ratio: float,
        radius: float,
        nn: torch.nn.Module,
        max_num_neighbors: int = 64,
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos,
            pos[idx],
            self.radius,
            batch,
            batch[idx],
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class SAModuleMSG(torch.nn.Module):
    def __init__(
        self,
        ratio: float,
        radius_list: List[float],
        max_num_neighbors_list: List[int],
        local_nn_list,
        global_nn=None,
        device=None,
    ):
        super().__init__()
        if any(
            (
                len(radius_list) != len(max_num_neighbors_list),
                len(radius_list) != len(local_nn_list),
            )
        ):
            raise ValueError("Lists must have the same length.")

        self.global_nn = global_nn
        self.ratio = ratio
        self.radius_list = radius_list
        self.max_num_neighbors_list = max_num_neighbors_list
        self.conv_layers = []
        device = torch.device(
            "cuda"
            if not wandb.config["cpu_only"] and torch.cuda.is_available()
            else "cpu"
        )
        for local_nn in local_nn_list:
            conv = PointNetConv(local_nn=local_nn, add_self_loops=False).to(device)

            self.conv_layers.append(conv)

    def forward(self, x, pos, batch):
        global_out = None
        idx = fps(pos, batch, ratio=self.ratio)
        for rad, max_num_neighbors, conv in zip(
            self.radius_list,
            self.max_num_neighbors_list,
            self.conv_layers,
        ):
            row, col = radius(
                pos,
                pos[idx],
                rad,
                batch,
                batch[idx],
                max_num_neighbors=max_num_neighbors,
            )
            edge_index = torch.stack([col, row], dim=0)
            x_dst = None if x is None else x[idx]
            local_out = conv((x, x_dst), (pos, pos[idx]), edge_index)
            global_out = (
                torch.cat((global_out, local_out), dim=1)
                if global_out is not None
                else local_out
            )

        pos, batch = pos[idx], batch[idx]
        if self.global_nn is None:
            return global_out, pos, batch
        else:
            assert isinstance(global_out, torch.Tensor)
            global_out = global_max_pool(global_out, batch)
            return global_out


class FPModule(torch.nn.Module):
    def __init__(self, k: int, nn: torch.nn.Module):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip
