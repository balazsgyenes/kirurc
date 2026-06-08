from __future__ import annotations

import logging
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    from scipy.interpolate import BSpline

log = logging.getLogger(__name__)


class LaplacianConfig(TypedDict):
    MAX_LAPLACE_CONTRACTION_WEIGHT: float
    MAX_POSITIONAL_WEIGHT: float
    INIT_LAPLACIAN_SCALE: float


class SplineConfig(TypedDict):
    smoothing_factor: float
    spline_degree: int


class NoSpline:
    """Represents the absence of a spline, since we cannot use None (Data objects do not allow
    data["foo"] = None). We use a class instead of an object to ensure that unpickled data
    refers to the exact same instance, which doesn't work for e.g. NoSpline = object().
    """

    pass


def skeletonize_point_cloud(
    points: np.ndarray,
    laplacian_config: LaplacianConfig,
) -> np.ndarray | None:
    if len(points) < 15:
        log.warning(
            f"Skeletonization skipped for a point cloud with {len(points)} points"
        )
        return

    import open3d as o3d
    from pc_skeletor import skeletor

    pcd = o3d.utility.Vector3dVector(points)
    pcd = o3d.geometry.PointCloud(pcd)

    try:
        with redirect_stdout(None):
            # "disable" voxel downsampling by setting grid size to 0.0005, which is smaller
            # than grid size we use for voxel downsampling
            sk = skeletor.Skeletonizer(point_cloud=pcd, down_sample=0.0005, debug=False)
            _, skeleton_points = sk.extract(method="Laplacian", config=laplacian_config)
    except RuntimeError as e:
        # skeletonization will fail for point clouds smaller than 10 points (hard-coded)
        # pc_skeletor/skeletor.py:L164
        log.warning(
            f"Skeletonization failed for a point cloud with {len(points)} points",
            exc_info=True,
        )
        return
    except AssertionError as e:
        # dgl/geometry/fps.py:L64
        log.warning(
            f"Skeletonization failed during fps for a point cloud with {len(points)} points (assert failed)",
            exc_info=True,
        )
        return

    skeleton_points = np.asarray(skeleton_points.points)
    return skeleton_points


def compute_spline_params(
    points: np.ndarray,
    spline_config: SplineConfig,
) -> BSpline:
    from scipy.interpolate import BSpline, splprep

    """
    We use splprep instead of the newer `make_interp_spline` because `make_interp_spline` doesn't
    support smoothing, and `make_smoothing_spline` only supports cubic splines and 1D inputs. In
    addition, the representations given by splprep seem to be smaller.
    """

    # the points in the graph should be sorted in order for the interpolation to work as expected
    # TODO: I am not sure if ordering by y always works
    dtype = [("x", float), ("y", float), ("z", float)]
    points = points.view(dtype)
    points = np.sort(points, order="y", axis=0)
    points = points.view(dtype=np.float64)

    # spline_tuple = (t,c,k) with t: knots, c: coefficients, k: degree
    (t, c, k), _ = splprep(
        points.transpose(),  # expects list of coordinates in each dimension
        s=spline_config["smoothing_factor"],
        k=spline_config["spline_degree"],
    )
    t, c = np.asarray(t), np.asarray(c)
    spline = BSpline(t, c.transpose(), k)

    return spline


def evaluate_spline_points(spline: BSpline, n_points: int) -> np.ndarray:
    x = np.linspace(0, 1, n_points)
    return spline(x)


def compute_chamfer_distance(
    ref_spline: BSpline,
    pred_spline: BSpline,
    n_test_points: int = 100,
    n_search_points: int = 100,
) -> np.ndarray:
    from scipy.cluster.vq import vq

    # test the reference spline at N sample points
    points_ref = evaluate_spline_points(ref_spline, n_test_points)

    # discretize the predicted spline into M points to search through
    points_pred = evaluate_spline_points(pred_spline, n_search_points)

    # for each point in the reference spline, we search for the closest point in the predicted spline
    nearest_pred, distances = vq(obs=points_ref, code_book=points_pred)
    return distances


def spline_to_ndarrays(spline: BSpline, prefix: str = "") -> dict[str, np.ndarray]:
    if prefix:
        prefix += "_"
    return {
        f"{prefix}t": spline.t,
        f"{prefix}c": spline.c,
        f"{prefix}k": np.array(spline.k),
    }


def ndarrays_to_spline(spline: dict[str, np.ndarray], prefix: str = "") -> BSpline:
    from scipy.interpolate import BSpline

    if prefix:
        prefix += "_"
    return BSpline(
        t=spline.pop(f"{prefix}t"),
        c=spline.pop(f"{prefix}c"),
        k=int(spline.pop(f"{prefix}k")),
    )


# def clean_splines(data: Data) -> Data:
#     spline_names = []
#     splines = {}
#     for key, value in data.items():
#         if isinstance(value, BSpline):
#             splines.update(spline_to_ndarrays(value, prefix=key))
#             spline_names.append(key)
#     if splines:
#         [data.pop(key) for key in spline_names]
#         data.update(splines)
#         data["splines"] = np.array(",".join(spline_names))

#     return data


# def restore_splines(data: Data) -> Data:
#     if spline_names := data.pop("splines", None):
#         spline_names = str(spline_names).split(",")
#         splines = {}
#         for spline_name in spline_names:
#             splines[spline_name] = ndarrays_to_spline(data, prefix=spline_name)

#     if spline_names:
#         data.update(splines)

#     return data
