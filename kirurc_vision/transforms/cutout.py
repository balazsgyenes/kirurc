from __future__ import annotations

import re

import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.nn import knn
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import index_to_mask, mask_to_index

_sentinel = object()


@functional_transform("cutout")
class Cutout(BaseTransform):
    def __init__(
        self,
        min_cuts: int = 0,
        max_cuts: int = 30,
        min_points_per_cut: int = 0,
        max_points_per_cut: int = 200,
        classes_to_cut_from: list[int] | object = _sentinel,
    ):
        self.min_n_seeds = min_cuts
        self.max_n_seeds = max_cuts
        self.min_knn = min_points_per_cut
        self.max_knn = max_points_per_cut

        if classes_to_cut_from is _sentinel:
            # maintains compatibility with previous version
            classes_to_cut_from = [0]
        if classes_to_cut_from is not None:
            if not isinstance(classes_to_cut_from, list):
                classes_to_cut_from = [classes_to_cut_from]
            classes_to_cut_from = torch.tensor(classes_to_cut_from, dtype=torch.long)
        self.classes = classes_to_cut_from

    def __call__(self, data: Data) -> Data:
        # randomly select number and size of cuts
        n_seeds = torch.randint(self.min_n_seeds, self.max_n_seeds, ())
        k = torch.randint(self.min_knn, self.max_knn, ())
        if n_seeds == 0 or k == 0:
            return data

        n_points = len(data.y)
        if (n_targets := len(data.pos) - n_points) > 0:
            targets, points = data.pos[-n_targets:], data.pos[:-n_targets]
        else:
            points = data.pos

        # get points with class labels that are eligible for cutting
        if self.classes is None:
            cutable_mask = torch.tensor(True)
            cutable_points = points
            cutable_indices = torch.arange(n_points)
        else:
            # ([N, 1] == [M]) -> [N, M] -> [N]
            cutable_mask = (data.y.unsqueeze(dim=-1) == self.classes).any(dim=-1)
            cutable_points = points[cutable_mask]
            cutable_indices = mask_to_index(cutable_mask)

        # choose seeds from cutable points and get k nearest neighbors
        # TODO: is there a way to randomly sample n integers without replacement?
        seeds = cutable_points[torch.randperm(len(cutable_points))[:n_seeds]]
        _, cutable_indices_to_cut = knn(cutable_points, seeds, k)

        # compute the indices of the original point cloud to keep after cutting
        cutable_cut_mask = index_to_mask(cutable_indices_to_cut, len(cutable_points))
        cutable_indices_to_keep = cutable_indices[~cutable_cut_mask]

        # join the indices that were kept after cutting with those excluded
        # from cutting
        choice = index_to_mask(cutable_indices_to_keep, n_points) | (~cutable_mask)

        # may need to reattach targets to the end
        result = (
            torch.concat((points[choice], targets)) if n_targets > 0 else points[choice]
        )

        # apply choice to data fields
        # copied and modified from pyg's FixedPoints transform
        for key, item in data:
            if key == "num_nodes":
                data.num_nodes = choice.size(0)
            elif key == "pos":
                data[key] = result
            elif bool(re.search("edge", key)):
                continue
            elif (
                torch.is_tensor(item) and item.size(0) == n_points and item.size(0) != 1
            ):
                data[key] = item[choice]

        return data
