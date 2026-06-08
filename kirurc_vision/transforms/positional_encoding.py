import torch
from torch_geometric.transforms import BaseTransform


class PositionalEncoding(BaseTransform):
    r"""Centers both node positions :obj:`pos` around the origin of the first."""

    def __call__(self, data):
        data_collection = []
        f = lambda a, exp: [torch.sin(2 ** exp * a), torch.cos(2 ** exp * a)]
        for i in range(-4, 6):
            data_collection += f(data.pos, i)

        data.x = torch.cat(data_collection, dim=-1)

        return data
