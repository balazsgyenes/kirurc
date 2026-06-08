import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import index_to_mask


class LabelTargets(BaseTransform):
    def __init__(
        self,
        radius: float,
        target_label_mapping: dict[str, tuple[int, int]],
        append_true_targets: bool = False,
        cleanup_targets: bool = True,
    ) -> None:
        self.radius = radius
        self.target_label_mapping = target_label_mapping
        self.append_true_targets = append_true_targets
        self.cleanup = cleanup_targets

    def __call__(self, data: Data) -> Data:
        getter = data.pop if self.cleanup else data.get
        targets: dict[str, torch.Tensor] = getter("targets")

        target_positions = torch.stack(tuple(targets.values()), dim=0)
        target_names = tuple(targets.keys())

        target_indices, point_indices = radius(
            data["pos"],
            target_positions,
            self.radius,
            max_num_neighbors=data["pos"].shape[0],  # find all neighbors
        )

        labels = data["y"]

        for i, name in enumerate(target_names):
            # get label to apply mapping to, based on name
            from_label, to_label = self.target_label_mapping[name]

            # get indices within radius of this target point, convert to mask
            radius_indices = point_indices[target_indices == i]
            radius_mask = index_to_mask(radius_indices, size=len(labels))

            # set those points that are within radius and belong to the correct
            # class
            labels[radius_mask & (labels == from_label)] = to_label

        if self.append_true_targets:
            data["pos"] = torch.cat((data["pos"], target_positions), dim=0)

            target_labels = torch.tensor(
                [self.target_label_mapping[name][1] for name in target_names],
                dtype=torch.uint8,
            )
            data["y"] = torch.cat((labels, target_labels), dim=0)

        return data
