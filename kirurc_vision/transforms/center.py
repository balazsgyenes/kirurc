from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class Center(BaseTransform):
    r"""Centers both node positions :obj:`pos` around the origin of the first."""

    def __call__(self, data: Data) -> Data:
        mid = data.pos.mean(dim=-2, keepdim=True)
        data.mid_offset = mid
        data.pos = data.pos - mid  # type: ignore
        data.target = data.target - mid  # type: ignore

        return data
