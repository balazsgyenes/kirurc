from pathlib import Path
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import open3d as o3d
from open3d.geometry import PointCloud


# define the center of the scene around which we rotate and crop
center = np.array([-192.200699, -58.527771, 1361.188477], dtype=np.float64)

point1 = np.array([-370.402008, -215.416687, 1274.299927], dtype=np.float64)
point2 = np.array([-4.597171, -226.049423, 1317.325073], dtype=np.float64)
point3 = np.array([-366.388824, 23.156744, 1427.353760], dtype=np.float64)
point4 = np.array([-57.295292, 27.720861, 1470.593628], dtype=np.float64)


# # define 3 points on the table to define its plane
# table1 = np.array([
#     -333.790771,
#     -209.139633,
#     1284.722900,
# ], dtype=np.float64)
# table2 = np.array([
#     -357.150970,
#     148.955460,
#     1508.369751,
# ], dtype=np.float64)
# table3 = np.array([
#     103.413376,
#     -249.877533,
#     1317.087036,
# ], dtype=np.float64)

# liver_left = np.array([
#     -90.505775,
#     -64.464836,
#     1366.719727,
# ], dtype=np.float64)
# liver_right = np.array([
#     -317.005371,
#     -62.174416,
#     1320.220947,
# ], dtype=np.float64)


def compute_rotation_to_align(vec: np.ndarray, align_to: np.ndarray) -> np.ndarray:

    vec /= np.linalg.norm(vec)
    align_to /= np.linalg.norm(align_to)

    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = np.cross(vec, align_to)
    s = np.linalg.norm(v)
    c = np.dot(vec, align_to)
    ssc = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ], dtype=np.float64)
    rotation = np.eye(3) + ssc + ((1 - c) / s ** 2) * np.linalg.matrix_power(ssc, 2)

    return rotation


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:

    raw_folder = Path("/home/balazs/data/kirurc/real_2023-03-15/raw")
    for_labelling_folder = raw_folder.parent / "for_labelling"
    for_labelling_folder.mkdir(parents=True, exist_ok=True)

    for file in raw_folder.glob("*.ply"):

        pcd = o3d.io.read_point_cloud(str(file))

        # # rotate to align the table with the xy plane
        # table_normal = np.cross(
        #     table2 - table1,
        #     table3 - table1,
        # )
        # table_normal /= np.linalg.norm(table_normal)
        # upwards = np.array([0, 0, 1], dtype=np.float64)
        # rotation = compute_rotation_to_align(table_normal, upwards)
        # pcd = pcd.rotate(rotation, center=center)

        # initial crop to remove most empty volume in the scene
        extent = np.array([
            600,
            600,
            600,
        ], dtype=np.float64)

        rough_bb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(center - extent / 2),
            max_bound=(center + extent / 2),
        )
        pcd = pcd.crop(rough_bb)

        # remove spurious points
        # we do this before removing the table, as this represents a large
        # number of non-noisy points
        n_points_before = len(np.asarray(pcd.points))
        pcd, inliers = pcd.remove_statistical_outlier(
            nb_neighbors=50,
            std_ratio=2,
        )
        print(f"Removed {n_points_before - len(inliers)} statistical outliers")


        # output_path = for_labelling_folder / (file.stem + "_orig" + file.suffix)
        # o3d.io.write_point_cloud(str(output_path), pcd)

        # segment table in the scene
        plane_parameters, inliers = pcd.segment_plane(
            distance_threshold=10.0,
            # remaining arguments default from C++ source:
            # https://github.com/isl-org/Open3D/blob/master/cpp/open3d/geometry/PointCloud.h#L347
            ransac_n=10,
            num_iterations=1000,
        )
        pcd = pcd.select_by_index(indices=inliers, invert=True)


        spanning_points = np.stack([
            point1, point2, point3, point4,
        ])
        fine_bb = o3d.geometry.OrientedBoundingBox.create_from_points(
            points = o3d.utility.Vector3dVector(spanning_points),
            robust=True,
        )
        extent = np.array(fine_bb.extent)
        extent[np.argmin(extent)] = 200
        fine_bb.extent = extent
        pcd = pcd.crop(fine_bb)



        # plane_normal = plane_parameters[:3]
        # upwards = np.array([0, 0, 1], dtype=np.float64)
        # rotation = compute_rotation_to_align(plane_normal, upwards)

        # extent = np.array([
        #     400,
        #     300,
        #     200,
        # ], dtype=np.float64)

        # fine_bb = o3d.geometry.OrientedBoundingBox(
        #     center=center,
        #     R=rotation.transpose(),  # apply inverse rotation to bounding box
        #     extent=extent,
        # )
        # pcd = pcd.crop(fine_bb)



        # pcd = pcd.rotate(rotation, center=center)
        # fine_bb = o3d.geometry.AxisAlignedBoundingBox(
        #     min_bound=(center - extent / 2),
        #     max_bound=(center + extent / 2),
        # )
        # pcd = pcd.crop(fine_bb)
        # pcd = pcd.rotate(rotation.transpose(), center=center)


        # o3d.visualization.draw_geometries([pcd])

        output_path = for_labelling_folder / file.name
        o3d.io.write_point_cloud(str(output_path), pcd)


if __name__ == "__main__":
    main()
