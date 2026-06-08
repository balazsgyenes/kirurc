from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class Relabel(BaseTransform):
    def __init__(self, mapping: dict[int, int]) -> None:
        # create mapping where the index of a color corresponds to the label.
        # labels with no given color are mapped to black (zeros)
        self.from_label = torch.tensor(tuple(mapping.keys()), dtype=torch.long)
        self.to_label = torch.tensor(tuple(mapping.values()), dtype=torch.long)

    def __call__(self, data: Data) -> Data:
        labels = data.y

        # ([N, 1] == [M]) -> [N, M]
        # compare N points to M possible mappings
        matches = labels.unsqueeze(-1) == self.from_label

        # returns max value and location of max value in a tuple
        match_found, indices = torch.max(matches, dim=1)

        # wherever a match was found, assign the corresponding label by indexing the to_labels
        labels[match_found] = self.to_label[indices[match_found]]

        return data
