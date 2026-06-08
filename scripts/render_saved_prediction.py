import logging
from pathlib import Path

import hydra
import open3d as o3d
import torch
from omegaconf import DictConfig

from kirurc_vision.utils.o3d_utils import data_to_pointcloud

log = logging.getLogger("render_saved_prediction")


@hydra.main(
    version_base=None, config_path="../conf", config_name="render_saved_prediction"
)
def main(config: DictConfig) -> None:

    pkl_path = Path(config["pkl_path"])
    data = torch.load(pkl_path)

    # determine i/o modes and create output folders
    if save_pcd := config.get("save_pcd", {}):
        pcd_folder = Path(save_pcd.folder)
        pcd_folder.mkdir(parents=True, exist_ok=True)
        pcd_extension = save_pcd.get("extension", ".ply")
        if pcd_extension[0] != ".":
            pcd_extension = "." + pcd_extension
    show = config.get("show", False)

    file = data["file"][0]  # assume batch size of 1

    if keys_to_skip := config.get("skip_keys", []):
        for key in keys_to_skip:
            data.pop(key, None)

    pcd = data_to_pointcloud(
        data,
        colors_key=config.get("colors_key", "predicted_labels"),
        color_mapping=config["color_mapping"],
        n_spline_points=config.get("n_spline_points", 100),
    )

    # save prediction to file as ply pointcloud
    if save_pcd:
        pcd_file = pcd_folder / (file.stem + "_prediction" + pcd_extension)
        log.info(f"Saving pcd to {pcd_file}")
        o3d.io.write_point_cloud(str(pcd_file), pcd, write_ascii=True, compressed=False)

    # render prediction
    if show:
        log.info(f"Rendering prediction from {file.name}")
        o3d.visualization.draw([pcd], **config.get("open3d_draw", {}))


if __name__ == "__main__":
    main()
