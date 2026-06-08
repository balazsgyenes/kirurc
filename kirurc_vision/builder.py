from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Mapping

import numpy as np  # import numpy before torch for performance (don't ask me)
import torch
from hydra.utils import instantiate
from torch_geometric.data import Dataset
from torch_geometric.transforms import Compose, RandomRotate, RandomShear

from kirurc_vision.dataset import (
    RealPlyDataset,
    SavedPredictionsDataset,
    SyntheticExrDataset,
)

log = logging.getLogger(__name__)


def resolve_path(str_path: str) -> Path:
    # resolve any environment variables in the first path element
    elements = str_path.split("/")
    # if first character of first element is "$"
    if (root := elements[0]).startswith("$"):
        elements[0] = os.environ[root[1:]]
    str_path = "/".join(elements)

    # resolve ~ to the user's home directory
    path = Path(str_path).expanduser()
    return path


def build_transform(transform_cfg: Mapping | None) -> Callable | None:
    transform_cfg = transform_cfg or {}
    transforms = []

    if pointwolf_cfg := transform_cfg.get("pointwolf"):
        from kirurc_vision.transforms.pointwolf import PointWOLF

        transforms.append(
            PointWOLF(
                num_anchor=pointwolf_cfg["num_anchor"],
                sample_type=pointwolf_cfg["sample_type"],
                sigma=pointwolf_cfg["sigma"],
                r_max=pointwolf_cfg["r_max"],
                s_max=pointwolf_cfg["s_max"],
                t_max=pointwolf_cfg["t_max"],
            )
        )

    if cutout_cfg := transform_cfg.get("cutout"):
        from kirurc_vision.transforms.cutout import Cutout

        transforms.append(
            Cutout(
                min_cuts=cutout_cfg["min_cuts"],
                max_cuts=cutout_cfg["max_cuts"],
                min_points_per_cut=cutout_cfg["min_points_per_cut"],
                max_points_per_cut=cutout_cfg["max_points_per_cut"],
                classes_to_cut_from=cutout_cfg["classes_to_cut_from"],
            )
        )

    if degrees := transform_cfg.get("random_rotate_deg"):
        transforms.extend(
            [
                RandomRotate(degrees, 0),
                RandomRotate(degrees, 1),
                RandomRotate(degrees, 2),
            ]
        )

    if shear := transform_cfg.get("random_shear"):
        transforms.append(RandomShear(shear))

    if stretch := transform_cfg.get("random_stretch"):
        from kirurc_vision.transforms.random_stretch import RandomStretch

        transforms.append(RandomStretch(stretch))

    if jitter := transform_cfg.get("jitter"):
        from kirurc_vision.transforms.jitter import RandomJitter, VariableJitter

        JitterCls = (
            VariableJitter if transform_cfg.get("vary_jitter", False) else RandomJitter
        )
        transforms.append(JitterCls(jitter))

    if transform_cfg.get("move_targets_to_y"):
        from kirurc_vision.transforms.add_target_points import MoveTargetPointsToY

        transforms.append(MoveTargetPointsToY())

    transform = Compose(transforms) if len(transforms) else None
    return transform


def build_synthetic_dataset(dataset_cfg: Mapping) -> Dataset:
    assert dataset_cfg["type"] == "synthetic"

    dataset_path = resolve_path(dataset_cfg["path"])

    preprocess_cfg = dataset_cfg["preprocess"]

    overrides = {}
    if "to_pointcloud" in preprocess_cfg:
        overrides["to_pointcloud"] = {
            "camera_params_file": dataset_path / "raw/camera.json"
        }

    pre_transforms = instantiate(preprocess_cfg, **overrides)
    pre_transform = (
        Compose([f for f in pre_transforms.values() if f is not None])
        if pre_transforms
        else None
    )

    transform_cfg = dataset_cfg.get("augmentation", {})
    transform = build_transform(transform_cfg)

    return SyntheticExrDataset(
        root=dataset_path,
        transform=transform,
        pre_transform=pre_transform,
        take_first_n=dataset_cfg.get("take_first_n", None),
        num_workers=dataset_cfg["num_workers"],
        force_reprocess=dataset_cfg.get("force_reprocess", False),
        debug_transforms=dataset_cfg.get("debug_transforms", False),
    )


def build_validation_dataset(dataset_cfg: Mapping) -> Dataset:
    assert dataset_cfg["type"] == "real"

    dataset_path = resolve_path(dataset_cfg["path"])

    preprocess_cfg = dataset_cfg["preprocess"]
    pre_transforms = instantiate(preprocess_cfg)
    pre_transform = (
        Compose([f for f in pre_transforms.values() if f is not None])
        if pre_transforms
        else None
    )

    transform_cfg = dataset_cfg.get("augmentation")
    transform = build_transform(transform_cfg)

    dataset = RealPlyDataset(
        root=dataset_path,
        transform=transform,
        pre_transform=pre_transform,
        take_first_n=dataset_cfg.get("take_first_n", None),
        num_workers=dataset_cfg["num_workers"],
        force_reprocess=dataset_cfg.get("force_reprocess", False),
        debug_transforms=dataset_cfg.get("debug_transforms", False),
    )

    return dataset


def build_prediction_dataset(dataset_cfg: Mapping) -> Dataset:
    assert dataset_cfg["type"] == "prediction"

    labeled_cfg = dataset_cfg["labeled_dataset"]
    if "shuffle_seed" in labeled_cfg:
        labeled_cfg["shuffle_seed"] = None  # disable shuffle
    labeled_dataset = build_validation_dataset(labeled_cfg)

    dataset_path = resolve_path(dataset_cfg["path"])

    preprocess_cfg = dataset_cfg["preprocess"]
    pre_transforms = instantiate(preprocess_cfg)
    pre_transform = (
        Compose([f for f in pre_transforms.values() if f is not None])
        if pre_transforms
        else None
    )

    transform_cfg = dataset_cfg.get("augmentation")
    transform = build_transform(transform_cfg)

    dataset = SavedPredictionsDataset(
        root=dataset_path,
        labeled_dataset=labeled_dataset,
        transform=transform,
        pre_transform=pre_transform,
        take_first_n=dataset_cfg.get("take_first_n", None),
        num_workers=dataset_cfg["num_workers"],
        force_reprocess=dataset_cfg.get("force_reprocess", False),
        debug_transforms=dataset_cfg.get("debug_transforms", False),
    )

    return dataset


def build_model(model_cfg: Mapping) -> tuple[torch.nn.Module, torch.device]:
    device = model_cfg["device"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    try:
        model = instantiate(model_cfg["model"])
    except KeyError as e:
        log.warning(
            "Building model with hydra instantiate failed with the following exception:",
            exc_info=True,
        )
        log.warning("Attempting to build model assuming legacy config format...")
        return build_model_legacy(model_cfg)

    return model.to(device), device


def build_model_legacy(model_cfg: Mapping) -> tuple[torch.nn.Module, torch.device]:
    device = model_cfg["device"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if model_cfg["class"] == "PointTransformer":
        from kirurc_vision.models.point_transformer import PointTransformer

        target = PointTransformer
        kwargs = model_cfg["kwargs"]
        if "in_channels" not in kwargs:
            kwargs["in_channels"] = 0  # no color
        if "out_channels" not in kwargs:
            kwargs["out_channels"] = model_cfg["n_classes"]

    else:
        raise ValueError(f"Unsupported model class {model_cfg['class']}")

    model = target(**kwargs)
    return model.to(device), device
