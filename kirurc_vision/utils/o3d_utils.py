from __future__ import annotations

from os import PathLike

import numpy as np
import open3d as o3d
import torch
from torch_geometric.data import Data

from kirurc_vision import COLOR_MAPPING, RGB_COLOR_MAPPING
from kirurc_vision.transforms import LabelToColor
from kirurc_vision.utils.splines import NoSpline, evaluate_spline_points


def save_as_pcd(
    filepath: PathLike,
    data: Data,
    write_ascii: bool = True,
    compressed: bool = False,
) -> None:
    pcd = data_to_pointcloud(data)
    o3d.io.write_point_cloud(
        str(filepath),
        pcd,
        write_ascii=write_ascii,
        compressed=compressed,
    )


def data_to_pointcloud(
    data: Data,
    colors_key: str | None = None,
    color_mapping: COLOR_MAPPING | None = None,
    n_spline_points: int = 50,
) -> o3d.geometry.PointCloud:
    points = o3d.utility.Vector3dVector(data["pos"].numpy())
    pcd = o3d.geometry.PointCloud(points)

    rgb_mapping = (
        to_rgb_color_mapping(color_mapping) if color_mapping is not None else None
    )

    # convert labels to color, if labels exist
    # labels take precedence over existing color, if both exist
    if colors_key is not None:
        if colors_key not in data:
            raise ValueError(f"Field {colors_key} not in data")
    elif data.y is not None:
        colors_key = "y"
    elif "colors" in data:
        colors_key = "colors"

    if colors_key is not None:
        colors = data.get(colors_key)
        if colors.dtype == torch.int64:
            if rgb_mapping is None:
                raise ValueError("Please provide a mapping from labels to colors.")
            data = LabelToColor(rgb_mapping, in_key=colors_key, cleanup_labels=False)(
                data
            )
            colors = data["colors"]

        if colors.dtype == torch.uint8:
            colors = colors.float().numpy() / 255
            pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        color = np.zeros((3,))
        pcd.paint_uniform_color(color[:, None])

    # draw any splines
    keys = ["t1_spline", "t2_spline", "t1_pred_spline", "t2_pred_spline"]
    for key in keys:
        if key not in data:
            continue

        spline = data[key][0]  # assume batch size of 1

        if spline is NoSpline:
            continue

        spline = evaluate_spline_points(spline, n_spline_points)
        spline = o3d.utility.Vector3dVector(spline)
        spline = o3d.geometry.PointCloud(spline)

        if rgb_mapping is not None and key in rgb_mapping:
            color = np.asarray(rgb_mapping[key]) / 255
        else:
            color = np.zeros((3,))
        spline.paint_uniform_color(color[:, None])

        pcd += spline

    # draw any skeletons
    keys = ["t1_skeleton", "t2_skeleton", "t1_pred_skeleton", "t2_pred_skeleton"]
    for key in keys:
        if key not in data:
            continue

        points = data[key][0]  # assume batch size of 1

        if not len(points):
            continue

        points = o3d.utility.Vector3dVector(points.numpy())
        skeleton = o3d.geometry.PointCloud(points)

        if rgb_mapping is not None and key in rgb_mapping:
            color = np.asarray(rgb_mapping[key]) / 255
        else:
            color = np.zeros((3,))
        skeleton.paint_uniform_color(color[:, None])

        pcd += skeleton

    return pcd


def to_rgb_color_mapping(mapping: COLOR_MAPPING) -> RGB_COLOR_MAPPING:
    rgb_mapping = {}
    for key, value in mapping.items():
        rgb_mapping[key] = to_rgb_color(value) if isinstance(value, str) else value

    return rgb_mapping


def to_rgb_color(color_code: str) -> list[int]:
    assert color_code[0] == "#"
    assert len(color_code) == 7
    return [
        int(color_code[1:3], 16),
        int(color_code[3:5], 16),
        int(color_code[5:7], 16),
    ]
