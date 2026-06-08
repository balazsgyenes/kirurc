import logging

import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)

TARGET_LABEL = 7


def clean_regression_data(data: Data) -> Data:
    # if data.pos includes extra target points
    if len(data.pos) - len(data.y) > 0:
        missing_targets = data.pos.isnan().any(dim=-1)
        log.info(f"Sample has {missing_targets.sum().item()} missing targets")

        # remove missing targets from point cloud
        data.pos = data.pos[~missing_targets]

        # append target label to y field
        n_valid_targets = len(data.pos) - len(data.y)
        data.y = torch.concat((data.y, torch.full((n_valid_targets,), TARGET_LABEL)))

    return data
