from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class NormalizeScale(BaseTransform):
    r"""Centers and normalizes both nodes positions to the interval :math:`(-1, 1)` relative to the first."""

    def __call__(self, data: Data) -> Data:
        center = data.pos.mean(dim=-2, keepdim=True)
        data.pos = data.pos - center
        data["center"] = center

        scale = (1 / data.pos.abs().max()) * 0.999999
        data.pos = data.pos * scale  # type: ignore
        data["scale"] = scale

        return data


def unnormalize_batch(data: Data) -> Data:
    ptr = data.ptr
    batch_size = len(ptr) - 1

    scale = data.pop("scale", None)
    center = data.pop("center", None)

    for b in range(batch_size):
        points = data.pos[ptr[b] : ptr[b + 1]]
        if scale is not None:
            points[...] = points / scale[b]

        if center is not None:
            points[...] = points + center[b]

    return data
