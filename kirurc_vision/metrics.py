from __future__ import annotations

import logging
from typing import Literal, Protocol, Sequence

import numpy as np
import torch
from scipy.cluster.vq import vq
from torch_geometric.data import Data
from torchmetrics.classification import MulticlassJaccardIndex

from kirurc_vision.transforms.normalize_scale import unnormalize_batch
from kirurc_vision.utils.splines import (
    LaplacianConfig,
    NoSpline,
    SplineConfig,
    compute_chamfer_distance,
    compute_spline_params,
    evaluate_spline_points,
    skeletonize_point_cloud,
)

log = logging.getLogger(__name__)


class Metric(Protocol):
    def __call__(self, data: Data, /) -> dict[str, float]: ...

    def to(self, device: torch.device, /) -> Metric:
        return self


class TargetMIoU(MulticlassJaccardIndex):
    """Calculates MulticlassJaccardIndex (also called intersection over union)
    over all classes, and additionally calculates `target_miou` over only the
    provided `target_classes`.
    This is used for pre-training, where we are most interested in the accuracy
    of the target classes than the other classes.
    """

    def __init__(
        self,
        *args,
        target_classes: Sequence[int],
        average: Literal["micro", "macro", "weighted", "none"] | None = "none",
        **kwargs,
    ) -> None:
        super().__init__(*args, average=average, **kwargs)
        self.mask = torch.tensor(target_classes)

    def forward(self, data: Data) -> dict[str, float]:
        predicted_labels = data["predicted_labels"]
        ious = super().forward(predicted_labels, data.y)
        return {
            "miou": ious.nanmean().cpu().item(),
            "target_miou": ious[self.mask].nanmean().cpu().item(),
        }


class SquashedMIoU(MulticlassJaccardIndex):
    """Calculates MulticlassJaccardIndex (also called intersection over
    union) after setting all predicted classes >= `lowest_class_to_squash`
    to 0.
    This is used for fine-tuning, when the dataset only has classes 0, 1, 2
    but the network predicts 7 classes.
    """

    def __init__(
        self,
        *args,
        target_classes: Sequence[int],
        lowest_class_to_squash: int,
        average: Literal["micro", "macro", "weighted", "none"] | None = "none",
        **kwargs,
    ) -> None:
        super().__init__(*args, average=average, **kwargs)
        self.mask = torch.tensor(target_classes)
        self.cutoff = lowest_class_to_squash

    def forward(self, data: Data) -> dict[str, float]:
        predicted_labels = data["predicted_labels"].detach().clone()
        predicted_labels[predicted_labels >= self.cutoff] = 0
        ious = super().forward(predicted_labels, data.y)
        return {
            "miou": ious[: self.cutoff].nanmean().cpu().item(),
            "target_miou": ious[self.mask].nanmean().cpu().item(),
        }


class Unnormalize(Metric):
    def __call__(self, data: Data) -> dict[str, float]:
        data = unnormalize_batch(data)
        return {}


class ClipRegionsChamferDistance(Metric):
    """Computes the path distance between the reference spline and the
    predicted spline for each target class."""

    def __init__(
        self,
        target_classes: Sequence[int],
        laplacian_config: LaplacianConfig,
        spline_config: SplineConfig,
        trim_reference_ends: float | None = None,
        clip_sections: list[list[int]] | None = None,
        chamfer_distance_threshold: float | None = None,
        n_points_ref: int = 100,
        n_points_pred: int = 100,
        cleanup_skeletons: bool = True,
        cleanup_splines: bool = True,
    ) -> None:
        self.target_classes = target_classes
        self.laplacian_config = laplacian_config
        self.spline_config = spline_config
        self.trim_reference_ends = trim_reference_ends
        self.clip_sections = clip_sections
        self.chamfer_distance_threshold = chamfer_distance_threshold
        self.n_points_ref = n_points_ref
        self.n_points_pred = n_points_pred
        self.cleanup_skeletons = cleanup_skeletons
        self.cleanup_splines = cleanup_splines

    def __call__(self, data: Data) -> dict[str, float]:
        ptr = data.ptr
        batch_size = len(ptr) - 1

        n_distances = self.spline_sim_kwargs["n_test_points"]
        spline_distances = torch.full(
            (batch_size, len(self.target_classes), n_distances),
            torch.nan,
        )

        predicted_splines = {
            f"t{label}_pred_spline": [NoSpline] * batch_size
            for label in self.target_classes
        }

        predicted_skeletons = {
            f"t{label}_pred_skeleton": [[]] * batch_size
            for label in self.target_classes
        }

        # loop over each element in the batch
        for b in range(batch_size):

            predicted_labels = data["predicted_labels"][ptr[b] : ptr[b + 1]]
            points = data.pos[ptr[b] : ptr[b + 1]]

            for i, label in enumerate(self.target_classes):

                ref_spline = data[f"t{label}_spline"][b]

                if ref_spline is NoSpline:
                    continue

                pred_points = points[predicted_labels == label]

                skeleton_points = skeletonize_point_cloud(
                    pred_points.cpu().numpy(),
                    self.laplacian_config,
                )

                if skeleton_points is None:
                    continue

                pred_spline = compute_spline_params(skeleton_points, self.spline_config)
                predicted_skeletons[f"t{label}_pred_skeleton"][b] = torch.from_numpy(
                    skeleton_points
                )
                predicted_splines[f"t{label}_pred_spline"][b] = pred_spline

                log.debug(
                    f"Predicted spline for target {label}:\nknots: {pred_spline.t}\ncoeffs: {pred_spline.c}"
                )

                distances = compute_chamfer_distance(
                    ref_spline,
                    pred_spline,
                    n_test_points=self.n_points_ref,
                    n_search_points=self.n_points_pred,
                )
                spline_distances[b, i] = torch.from_numpy(distances)

        # populate with metrics for each batch element
        # average over targets and points first, then average over batch
        metrics = {}

        if self.trim_reference_ends is not None:
            start = int(n_distances * self.trim_reference_ends)
            end = int(n_distances * (1 - self.trim_reference_ends))
            valid_section = slice(start, end)
        else:
            valid_section = slice(None)

        for i, label in enumerate(self.target_classes):
            # for each label, evaluate as true if all distance are not nan
            metrics[f"t{label}_spline_found"] = (
                spline_distances[:, i].isnan().any(dim=-1).logical_not().float()
            )

            # for each label, compute average distance across all points
            distances = spline_distances[:, i, valid_section]
            metrics[f"t{label}_spline_sim"] = distances.nanmean(dim=-1)

        # compute average distance across all points across all labels
        metrics["spline_sim"] = spline_distances[..., valid_section].nanmean(dim=(1, 2))

        # evaluate true if all distances are not nan across all labels
        metrics["splines_found"] = (
            spline_distances.isnan().any(dim=-1).logical_not().float().mean(dim=-1)
        )

        if self.chamfer_distance_threshold is not None:
            # for each clip, evaluate as true if the minimum distance within that section is less than the threshold
            min_distances = torch.zeros(
                (batch_size, len(self.target_classes), len(self.clip_sections))
            )
            for i, (start, end) in enumerate(self.clip_sections):
                distances = spline_distances[..., slice(start, end)]
                min_distances[..., i], _ = distances.min(dim=-1)
            clip_successes = min_distances < self.chamfer_distance_threshold
            metrics["success"] = clip_successes.float().mean()

        # average each metric over batch elements and convert to Python float
        for key, value in metrics.items():
            metrics[key] = value.nanmean().item()

        if not self.cleanup_skeletons:
            for key, skeletons in predicted_skeletons.items():
                data[key] = skeletons

        if not self.cleanup_splines:
            for key, splines in predicted_splines.items():
                data[key] = splines

        return metrics


class SplineChamferDistance(Metric):
    """Computes the path distance between the reference spline and the
    predicted spline for each target class."""

    def __init__(
        self,
        target_classes: Sequence[int],
        laplacian_config: LaplacianConfig,
        spline_config: SplineConfig,
        symmetric: bool = True,
        trim_reference_ends: float | None = None,
        trim_predicted_ends: bool = False,
        chamfer_distance_threshold: float | None = None,
        n_points_ref: int = 100,
        n_points_pred: int = 100,
        cleanup_skeletons: bool = True,
        cleanup_splines: bool = True,
    ) -> None:
        self.target_classes = target_classes
        self.laplacian_config = laplacian_config
        self.spline_config = spline_config
        self.symmetric = symmetric
        self.trim_reference_ends = trim_reference_ends
        self.trim_predicted_ends = trim_predicted_ends
        self.chamfer_distance_threshold = chamfer_distance_threshold
        self.n_points_ref = n_points_ref
        self.n_points_pred = n_points_pred
        self.cleanup_skeletons = cleanup_skeletons
        self.cleanup_splines = cleanup_splines

    def __call__(self, data: Data) -> dict[str, float]:
        ptr = data.ptr
        batch_size = len(ptr) - 1

        spline_distances = torch.full(
            (batch_size, len(self.target_classes)),
            torch.nan,
        )

        predicted_splines = {
            f"t{label}_pred_spline": [NoSpline] * batch_size
            for label in self.target_classes
        }

        predicted_skeletons = {
            f"t{label}_pred_skeleton": [[]] * batch_size
            for label in self.target_classes
        }

        # loop over each element in the batch
        for b in range(batch_size):

            predicted_labels = data["predicted_labels"][ptr[b] : ptr[b + 1]]
            points = data.pos[ptr[b] : ptr[b + 1]]

            for i, label in enumerate(self.target_classes):

                ref_spline = data[f"t{label}_spline"][b]

                if ref_spline is NoSpline:
                    continue

                pred_points = points[predicted_labels == label]

                skeleton_points = skeletonize_point_cloud(
                    pred_points.cpu().numpy(),
                    self.laplacian_config,
                )

                if skeleton_points is None:
                    continue

                predicted_skeletons[f"t{label}_pred_skeleton"][b] = torch.from_numpy(
                    skeleton_points
                )

                pred_spline = compute_spline_params(skeleton_points, self.spline_config)
                predicted_splines[f"t{label}_pred_spline"][b] = pred_spline

                log.debug(
                    f"Predicted spline for target {label}:\nknots: {pred_spline.t}\ncoeffs: {pred_spline.c}"
                )

                points_ref = evaluate_spline_points(ref_spline, self.n_points_ref)
                points_pred = evaluate_spline_points(pred_spline, self.n_points_pred)

                if trim := self.trim_reference_ends:
                    assert trim > 0 and trim < 0.5

                    start = int(len(points_ref) * trim)
                    end = int(len(points_ref) * (1 - trim))
                    points_ref = points_ref[start:end]

                if self.trim_predicted_ends:
                    end_points_ref = points_ref[:: (len(points_ref) - 1)]
                    nearest_pred, _ = vq(obs=end_points_ref, code_book=points_pred)
                    t_min, t_max = np.sort(nearest_pred)
                    points_pred = points_pred[t_min : t_max + 1]

                chamfer_distance = vq(obs=points_ref, code_book=points_pred)[1].mean()

                if self.symmetric:
                    chamfer_distance += vq(obs=points_pred, code_book=points_ref)[
                        1
                    ].mean()
                    chamfer_distance /= 2.0

                spline_distances[b, i] = chamfer_distance

        # populate with metrics for each batch element
        # average over targets and points first, then average over batch
        metrics = {}

        for i, label in enumerate(self.target_classes):
            # for each label, evaluate as true if all distance are not nan
            metrics[f"t{label}_spline_found"] = (
                spline_distances[:, i].isnan().logical_not().float()
            )

            metrics[f"t{label}_spline_sim"] = spline_distances[:, i]

        # compute average distance across across all labels
        metrics["spline_sim"] = spline_distances.nanmean(dim=-1)

        # evaluate true if all distances are not nan across all labels
        metrics["splines_found"] = (
            spline_distances.isnan().logical_not().float().mean(dim=-1)
        )

        if self.chamfer_distance_threshold is not None:

            # comparing against nan always results in False, i.e. failure
            metrics["success"] = (
                (spline_distances < self.chamfer_distance_threshold)
                .float()
                .mean(dim=-1)
            )

        # average each metric over batch elements and convert to Python float
        for key, value in metrics.items():
            metrics[key] = value.nanmean().item()

        if not self.cleanup_skeletons:
            for key, skeletons in predicted_skeletons.items():
                data[key] = skeletons

        if not self.cleanup_splines:
            for key, splines in predicted_splines.items():
                data[key] = splines

        return metrics


class DistanceFromTarget(Metric):
    def __init__(self, distance_threshold: float | None = None) -> None:
        self.threshold = distance_threshold

    def __call__(self, data: Data) -> dict[str, float]:
        prediction, target = data["prediction"], data.y

        B = prediction.shape[0]
        target = target.view(B, 6, 3)

        # insert singleton dimension so that all predicted points are compared
        # against all target points
        target = target.unsqueeze(dim=2)
        prediction = prediction.unsqueeze(dim=1)

        # compute distance matrix where d_ij is distance between target i and prediction j
        distances = (prediction - target).pow(2).sum(dim=-1).sqrt()

        # compute distance from each target to the nearest prediction
        nearest_pred = distances.min(dim=2).values

        metrics = {"mean_distance": nearest_pred.mean()}

        if self.threshold is not None:
            metrics["success"] = (nearest_pred < self.threshold).float().mean()

        return metrics
