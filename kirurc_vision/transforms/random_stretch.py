from typing import Union

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform, LinearTransformation


@functional_transform('random_stretch')
class RandomStretch(BaseTransform):
    r"""Scales node positions by randomly sampled factors :math:`s` in each
    dimension within a given interval, *e.g.*, resulting in the transformation
    matrix (functional name: :obj:`random_stretch`)

    .. math::
        \begin{bmatrix}
            s_{x} & 0 & 0 \\
            0 & s_{y} & 0 \\
            0 & 0 & s_{z} \\
        \end{bmatrix}

    for three-dimensional positions.

    Args:
        factor (float): maximum scaling factor defining the range
        :math:`(1-\mathrm{factor}, 1+\mathrm{factor})` to sample from.
    """
    def __init__(self, factor: float):
        assert factor < 1
        self.factor = factor

    def __call__(self, data: Data) -> Data:
        dim = data.pos.size(-1)

        diag = data.pos.new_empty(dim).uniform_(1-self.factor, 1+self.factor)
        matrix = torch.diag(diag)

        return LinearTransformation(matrix)(data)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.factor})'
