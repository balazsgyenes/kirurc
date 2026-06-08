import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class ToPointCloud(BaseTransform):
    def __init__(self,
        camera_params_file: Path,
        cleanup_rgbd: bool = True,
    ) -> None:

        try:
            with open(camera_params_file, "r") as f:
                camera_parameters = json.load(f)
            # rename a few parameters
            camera_parameters["width"] = camera_parameters.pop("image_width")
            camera_parameters["height"] = camera_parameters.pop("image_height")
            self.camera_params = camera_parameters
        except FileNotFoundError:
            # maybe this transform is never called, in which case we don't
            # want to raise an error
            self.camera_params = None

        self.cleanup = cleanup_rgbd

    def __call__(self, data: Data) -> Data:
        getter = data.pop if self.cleanup else data.get
        rgb = getter("rgb")
        depth = getter("depth")

        rgb = o3d.geometry.Image(rgb.numpy())
        depth = o3d.geometry.Image(depth.numpy())
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=65000,
            convert_rgb_to_intensity=False,
        )

        camera = o3d.camera.PinholeCameraIntrinsic(**self.camera_params)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)

        xyz = np.asarray(pcd.points).astype(np.float32)
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

        xyz, colors = torch.from_numpy(xyz), torch.from_numpy(colors)

        # flip y and z axes because of Blender's coordinate system
        # -z is defined as the ray of the camera
        # y is defined as going downwards in image space
        # with these transforms, the point cloud matches the depth image
        xyz[:, 1:] = -xyz[:, 1:]

        data["pos"] = xyz
        data["colors"] = colors

        return data
