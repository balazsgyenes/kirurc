"""
@origin : PointWOLF.py by {Sanghyeok Lee, Sihyeon Kim}
@Contact: {cat0626, sh_bs15}@korea.ac.kr
@Time: 2021.09.30
"""

from typing import Literal

import torch
from torch_geometric.data import Data
from torch_geometric.nn import fps
from torch_geometric.transforms import BaseTransform


class PointWOLF(BaseTransform):
    """PointWOLF: Point Cloud Augmentation with Weighted Local Transformation
    https://arxiv.org/abs/2110.05379
    This transform applies a weighted local transformation to the point cloud data.
    Args:
        num_anchor (int): Number of anchor points to sample.
        sample_type (str): Type of sampling for anchor points, either "random" or "fps".
        sigma (float): Standard deviation for the Gaussian kernel used combining local transforms.
        r_max (float): Maximum rotation angle in degrees.
        s_max (float): Maximum scaling factor.
        t_max (float): Maximum translation distance.
    """

    def __init__(
        self,
        num_anchor: int = 4,
        sample_type: Literal["random", "fps"] = "fps",
        sigma: float = 0.5,
        r_max: float = 10,
        s_max: float = 3,
        t_max: float = 0.25,
    ):
        self.num_anchor = num_anchor
        self.sample_type = sample_type
        self.sigma = sigma

        self.r_range = (-abs(r_max), abs(r_max))
        self.s_range = (1.0, s_max)
        self.t_range = (-abs(t_max), abs(t_max))

    def __call__(self, data: Data) -> Data:

        pos = data.pos
        assert pos is not None
        N, _ = pos.shape
        M = self.num_anchor

        if self.sample_type == "random":
            idx = torch.randint(0, N, (M,))
        elif self.sample_type == "fps":
            ratio = M / N
            idx = fps(pos, ratio=ratio)  # (M)

        pos_anchor = pos[idx]  # (M, 3) anchor point

        # Move to canonical space
        pos_normalize = pos.unsqueeze(dim=0) - pos_anchor.unsqueeze(dim=1)  # (M, N, 3)

        # Local transformation at anchor point
        pos_transformed = self.local_transformation(pos_normalize)  # (M, N, 3)

        # Move to origin space
        pos_transformed = pos_transformed + pos_anchor.unsqueeze(dim=1)  # (M, N, 3)

        pos_new = self.kernel_regression(pos, pos_anchor, pos_transformed)

        pos[...] = pos_new
        return data

    def kernel_regression(self, pos, pos_anchor, pos_transformed):
        """
        input :
            pos([N,3])
            pos_anchor([M,3])
            pos_transformed([M,N,3])

        output :
            pos_new([N,3]) : Pointcloud after weighted local transformation
        """
        M, N, _ = pos_transformed.shape
        device = pos.device

        # Distance between anchor points & entire points
        sub = pos_anchor.unsqueeze(1).repeat(1, N, 1) - pos.unsqueeze(0).repeat(
            M, 1, 1
        )  # (M, N, 3), d

        project_axis = self.get_random_axis(1, device=device)

        projection = torch.unsqueeze(project_axis, 1) * torch.eye(
            3, device=device
        )  # (1, 3, 3)

        # Project distance
        sub = sub @ projection  # (M, N, 3)
        sub = torch.sqrt(((sub) ** 2).sum(2))  # (M, N)

        # Kernel regression
        weight = torch.exp(-0.5 * (sub**2) / (self.sigma**2))  # (M, N)
        pos_new = (torch.unsqueeze(weight, 2).repeat(1, 1, 3) * pos_transformed).sum(
            0
        )  # (N, 3)
        pos_new = pos_new / weight.sum(0, keepdim=True).T  # normalize by weight
        return pos_new

    def local_transformation(self, pos):
        M, N, _ = pos.shape
        device = pos.device

        transformation_dropout = torch.bernoulli(
            torch.full((M, 3), 0.5, device=device)
        )  # (M, 3)
        transformation_axis = self.get_random_axis(M, device=device)  # (M, 3)

        degree = (
            torch.pi
            * torch.empty(M, 3, device=device).uniform_(*self.r_range)
            / 180.0
            * transformation_dropout[:, 0:1]
        )  # (M, 3)

        scale = (
            torch.empty(M, 3, device=device).uniform_(*self.s_range)
            * transformation_dropout[:, 1:2]
        )  # (M, 3)
        scale = scale * transformation_axis
        scale[scale == 0] = 1.0  # Scaling factor must be larger than 1

        trl = (
            torch.empty(M, 3, device=device).uniform_(*self.t_range)
            * transformation_dropout[:, 2:3]
        )  # (M, 3)
        trl = trl * transformation_axis

        # Scaling Matrix
        S = torch.diag_embed(scale)  # (M, 3, 3)

        # Rotation Matrix
        sin = torch.sin(degree)
        cos = torch.cos(degree)
        sx, sy, sz = sin[:, 0], sin[:, 1], sin[:, 2]
        cx, cy, cz = cos[:, 0], cos[:, 1], cos[:, 2]

        # Euler ZYX rotation
        R = torch.stack(
            [
                cz * cy,
                cz * sy * sx - sz * cx,
                cz * sy * cx + sz * sx,
                sz * cy,
                sz * sy * sx + cz * cy,
                sz * sy * cx - cz * sx,
                -sy,
                cy * sx,
                cy * cx,
            ],
            dim=1,
        ).reshape(M, 3, 3)

        pos = pos @ R @ S + trl.view(M, 1, 3)
        return pos

    def get_random_axis(self, n_axis, device="cpu"):
        """
        input :
            n_axis(int)

        output :
            axis([n_axis,3]) : projection axis
        """
        # 1(001):z, 2(010):y, 3(011):yz, 4(100):x, 5(101):xz, 6(110):xy, 7(111):xyz
        axis_int = torch.randint(1, 8, (n_axis,), device=device)

        # Use bitwise operations to get axis flags
        axis = axis_int.unsqueeze(1) & (1 << torch.arange(3, device=device))
        axis = axis.clamp(max=1)
        return axis
