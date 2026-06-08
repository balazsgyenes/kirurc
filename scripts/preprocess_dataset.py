import hydra
from omegaconf import DictConfig

from kirurc_vision.builder import build_synthetic_dataset, build_validation_dataset


@hydra.main(version_base=None, config_path="../conf", config_name="preprocess_dataset")
def main(config: DictConfig) -> None:

    dataset_cfg = config["dataset"]
    if dataset_cfg["type"] == "synthetic":
        build_synthetic_dataset(dataset_cfg)
    elif dataset_cfg["type"] == "real":
        build_validation_dataset(dataset_cfg)
    else:
        raise ValueError(f"Unknown dataset type '{dataset_cfg['type']}'")


if __name__ == "__main__":
    main()
