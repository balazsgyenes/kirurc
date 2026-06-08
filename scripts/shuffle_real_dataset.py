import json
from pathlib import Path
import shutil

import hydra
import numpy as np
from omegaconf import DictConfig

from kirurc_vision.utils.misc import bunched


@hydra.main(version_base=None, config_path="../conf", config_name="shuffle_real_dataset")
def main(config: DictConfig) -> None:
    
    root = Path(config["dataset"]["path"])
    source = root / "raw_unshuffled"
    destination = root / "raw"

    target1_files = list(source.glob("*_target1.ply"))
    target2_files = list(source.glob("*_target2.ply"))
    raw_files = sorted(
        file for file in source.glob("*.ply")
        if file not in target1_files and file not in target2_files
    )

    for raw_file in raw_files:
        assert raw_file.with_stem(raw_file.stem + "_target1") in target1_files
        assert raw_file.with_stem(raw_file.stem + "_target2") in target2_files

    with open(source / "shuffle.json", "r") as f:
        shuffle_cfg = json.load(f)

    shuffled_indices = list(range(len(raw_files)))
    rng = np.random.default_rng(seed=shuffle_cfg["seed"])
    rng.shuffle(shuffled_indices)

    destination.mkdir(parents=True, exist_ok=True)
    for raw_file, shuffled_index in zip(raw_files, shuffled_indices):
        for file in (
            raw_file,
            raw_file.with_stem(raw_file.stem + "_target1"),
            raw_file.with_stem(raw_file.stem + "_target2"),
        ):
            shutil.copy(file, destination / f"{shuffled_index:02d}_{file.name}")


if __name__ == "__main__":
    main()
