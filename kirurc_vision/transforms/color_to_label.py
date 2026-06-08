from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from kirurc_vision import RGB_COLOR_MAPPING


class ColorToLabel(BaseTransform):
    def __init__(
        self,
        color_mapping: RGB_COLOR_MAPPING,
        default_label: int | None = None,
        in_key: str = "colors",
        out_key: str = "y",
        cleanup_colors: bool = True,
    ) -> None:
        # create mapping where the index of a color corresponds to the label.
        # labels with no given color are mapped to black (zeros)
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

        self.default_label = default_label
        self.in_key = in_key
        self.out_key = out_key
        self.cleanup = cleanup_colors

    def __call__(self, data: Data) -> Data:
        getter = data.pop if self.cleanup else data.get
        colors = getter(self.in_key)

        # ([N, 1, 3] == [M, 3]) -> [N, M, 3] -> [N, M]
        # compare N points to M colors in the map with 3 values each
        # results in an [N, M] tensor giving the equality of point n with color m
        matches = (colors.unsqueeze(1) == self.int_map).all(dim=-1)

        # returns max value and location of max value in a tuple
        match_found, labels = torch.max(matches, dim=1)

        if self.default_label is None:
            if not torch.all(match_found):
                raise RuntimeError(
                    f"No label could be found for {(~match_found).sum()} points."
                )
        else:
            labels[~match_found] = self.default_label

        data[self.out_key] = labels

        return data
