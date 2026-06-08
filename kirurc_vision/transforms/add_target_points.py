import logging

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from kirurc_vision.utils.splines import NoSpline

log = logging.getLogger(__name__)


class AddTargetPoints(BaseTransform):
    def __init__(
        self,
        spline_coordinates: dict[str, list[float]],
        move_targets_to_y: bool = False,
        cleanup: bool = True,
    ) -> None:
        self.spline_coordinates = {
            label: np.asarray(coordinates)
            for label, coordinates in spline_coordinates.items()
        }
        self.move_targets_to_y = move_targets_to_y
        self.cleanup = cleanup

    def __call__(self, data: Data) -> Data:
        target_points = []
        for label, coordinates in self.spline_coordinates.items():

            getter = data.pop if self.cleanup else data.get
            spline = getter(label)

            if spline is NoSpline:
                # if no reference spline exists, fill with nans
                # we always need to output the same number of points so that
                # batching is preserved
                # any predicted points for this duct should just be ignored
                # by chamfer distance loss
                points = np.full((len(coordinates), 3), np.nan)
            else:
                points = spline(coordinates)
            target_points.append(torch.from_numpy(points).to(torch.float32))

        if self.move_targets_to_y:
            data.y = torch.concat(target_points)
        else:
            data.pos = torch.concat([data.pos] + target_points)

        return data


class MoveTargetPointsToY(BaseTransform):
    def __call__(self, data: Data) -> Data:
        n_targets = len(data.pos) - len(data.y)
        data.y = data.pos[-n_targets:]
        data.pos = data.pos[:-n_targets]
        return data
