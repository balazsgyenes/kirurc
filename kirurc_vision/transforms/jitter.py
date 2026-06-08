import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomJitter as TorchRandomJitter


class RandomJitter(TorchRandomJitter):
    def __call__(self, data: Data) -> Data:
        # save values of target points and restore them after jitter
        n_targets = len(data.pos) - len(data.y)
        if n_targets > 0:
            target_points_backup = data.pos[-n_targets:].clone()
        super().__call__(data)
        if n_targets > 0:
            data.pos[-n_targets:] = target_points_backup
        return data


class VariableJitter(RandomJitter):
    def __init__(self, translate):
        self.max_translate = translate
        super().__init__(translate)

    def __call__(self, data):
        self.translate = torch.rand(1).item() * self.max_translate
        return super().__call__(data)
