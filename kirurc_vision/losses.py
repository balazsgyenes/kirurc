from typing import Literal, Sequence

import torch
from torch import Tensor


class ProbMergedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(
        self,
        *args,
        classes_to_merge: Sequence[int],
        classes_to_keep: Sequence[int],
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.merge_indices = torch.tensor(classes_to_merge, dtype=torch.long)
        self.keep_indices = torch.tensor(classes_to_keep, dtype=torch.long)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        merge = input[:, self.merge_indices]
        keep = input[:, self.keep_indices]
        # sum after exponentiation, which keeps sum(exp()) of input constant
        # subtracting maximum ensures that exp sees values <=0, and that log
        # sees values >=1, avoiding infinities
        merged = merge.logsumexp(dim=-1, keepdim=True)
        new_input = torch.cat((merged, keep), dim=-1)
        return super().forward(new_input, target)


class LogitMergedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """Merges classes by summing logits for those classes."""

    def __init__(
        self,
        *args,
        classes_to_merge: Sequence[int],
        classes_to_keep: Sequence[int],
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.merge_indices = torch.tensor(classes_to_merge, dtype=torch.long)
        self.keep_indices = torch.tensor(classes_to_keep, dtype=torch.long)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # merge by summing logits
        merged = input[:, self.merge_indices].sum(dim=-1, keepdim=True)
        other = input[:, self.keep_indices]
        new_input = torch.cat((merged, other), dim=-1)
        return super().forward(new_input, target)


class TargetChamferDistance(torch.nn.Module):
    def __init__(
        self,
        symmetric: bool,
        point_reduction: Literal["mean", "sum"] = "mean",
        weight: None = None,
    ) -> None:
        super().__init__()
        self.symmetric = symmetric
        assert point_reduction in ("mean", "sum")
        self.reducer = "nan" + point_reduction
        # ignore weight, as it's just a placeholder

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        B = prediction.shape[0]
        target = target.view(B, 6, 3)

        # insert singleton dimension so that all predicted points are compared
        # against all target points
        target = target.unsqueeze(dim=2)
        prediction = prediction.unsqueeze(dim=1)

        # we must identify nans in the target tensor and remove them
        missing_targets = target.isnan().any(dim=-1)
        # the value to replace with is arbitrary, but must be finite
        # otherwise, the pow operation results in nan grads
        target = torch.nan_to_num(target, 0)

        # compute distance matrix where d_ij is distance between target i and prediction j
        distances = (prediction - target).pow(2).sum(dim=-1).sqrt()

        # compute distance from each target to the nearest prediction and sum over target
        # nans resulting due to missing targets are converted to inf so that min
        # operation will ignore them
        nearest_target = (
            torch.where(missing_targets, torch.inf, distances).min(dim=1).values
        )
        nearest_target = getattr(nearest_target, self.reducer)(dim=-1)
        chamfer = nearest_target

        if self.symmetric:
            # compute distance from each prediction to nearest target and sum over predictions
            # add this existing loss so that the resulting chamfer distance is symmetric

            # nans resulting due to missing targets are converted back to nan so that nansum
            # operation will ignore them
            nearest_pred = (
                torch.where(missing_targets, torch.nan, distances).min(dim=2).values
            )
            nearest_pred = getattr(nearest_pred, self.reducer)(dim=-1)
            chamfer = nearest_target + nearest_pred

        # take the mean over all batch elements
        # nanmean filters out the case where a datapoint has no targets at all
        return chamfer.nanmean()
