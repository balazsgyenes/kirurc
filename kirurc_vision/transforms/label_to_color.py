from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from kirurc_vision import RGB_COLOR_MAPPING


class LabelToColor(BaseTransform):
    def __init__(
        self,
        color_mapping: RGB_COLOR_MAPPING,
        in_key: str = "y",
        out_key: str = "colors",
        cleanup_labels: bool = True,
    ) -> None:
        # create mapping where indexing with a label returns the corresponding
        # color. labels with no given color are mapped to black (zeros)
        int_map = {
            key: value for key, value in color_mapping.items() if isinstance(key, int)
        }
        self.int_map = torch.zeros(
            (max(int_map.keys()) + 1, 3),
            dtype=torch.uint8,
        )
        for label, color in int_map.items():
            color = torch.tensor(color, dtype=torch.uint8)
            self.int_map[label] = color

        self.in_key = in_key
        self.out_key = out_key
        self.cleanup = cleanup_labels

    def __call__(self, data: Data) -> Data:
        getter = data.pop if self.cleanup else data.get
        labels = getter(self.in_key)

        # generate tensor with same length as labels by combining elements
        # from the color map
        data[self.out_key] = self.int_map[labels]

        return data
