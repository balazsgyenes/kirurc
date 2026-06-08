import torch
from torch_geometric.data import Data
from torch_geometric.nn import nearest
from torch_geometric.transforms import BaseTransform


class MatchTargetSegments(BaseTransform):
    def __init__(self,
        cleanup_targets: bool = True,
    ) -> None:
        self.cleanup = cleanup_targets

    def __call__(self, data: Data) -> Data:
        getter = data.pop if self.cleanup else data.get
        target_segments = getter("target_segments")

        for name, segment in target_segments.items():

            matching = nearest(segment, data.pos)

            matchin_pos = data.pos[matching]
            if not torch.allclose(matchin_pos, segment):
                raise RuntimeError(
                    "At least one point in the target segment was not found "
                    f"in the source point cloud in file '{data['file']}'."
                )

            data.y[matching] = name

        return data
