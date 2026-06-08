import math
import os
from os import PathLike
from pathlib import Path
import signal
import subprocess

import hydra
from omegaconf import DictConfig


BPY_SCRIPTS_DIR = (Path(__file__) / "../../blender/bpy_scripts").resolve()
SCRIPT_SAVE_CAMERA_PARAMS = BPY_SCRIPTS_DIR / "save_camera_params.py"
SCRIPT_CREATE_SAMPLES = BPY_SCRIPTS_DIR / "create_samples.py"


def save_camera_parameters(blender_file: PathLike, output_path: PathLike) -> None:
    blender_command = ["blender", blender_file, "--background", "--python", SCRIPT_SAVE_CAMERA_PARAMS]
    blender_command += ["--", output_path]
    blender_command = [str(arg) for arg in blender_command]
    subprocess.run(blender_command)


def create_samples(
    blender_file: PathLike,
    randomize_script: PathLike,
    output_path: PathLike,
    num_workers: int,
    samples_per_worker: int,
) -> None:

    # Create blender instances to generated the dataset
    processes = []
    for i in range(num_workers):
        from_index = i * samples_per_worker
        to_index = (i + 1) * samples_per_worker

        blender_command = ["blender", blender_file, "--background", "--python", SCRIPT_CREATE_SAMPLES]
        blender_command += ["--", from_index, to_index, randomize_script, output_path]
        blender_command = [str(arg) for arg in blender_command]

        processes.append(subprocess.Popen(blender_command))

    # repeat an interrupt to all workers processes, so that a single interrupt
    # stops all workers
    # https://stackoverflow.com/a/4791612
    def exit_handler(sig, frame):
        """ Kill all processes if Ctrl+C is pressed """
        for proc in processes:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    
    signal.signal(signal.SIGINT, exit_handler)
    
    # Wait for all blender processes to finish
    for process in processes:
        process.wait()


@hydra.main(version_base=None, config_path="../conf", config_name="generate_synthetic")
def main(config: DictConfig) -> None:
    # parse config file
    scenes_base = Path(config["scenes_base"])

    blender_file = scenes_base / str(config["blender_file"])

    if (python_script := config["python_script"]) is None:
        python_script = blender_file.with_suffix(".py")
    else:
        python_script = scenes_base / python_script

    output_base = Path(config["output_base"])
    if (output_folder := config["output_folder"]) is None:
        output_folder = output_base / blender_file.stem
    else:
        output_folder = output_base / output_folder

    dataset_size = config["dataset_size"]
    num_workers = config["num_workers"]

    # calculate number of samples per blender process
    samples_per_worker = math.ceil(dataset_size / num_workers)

    # make output directory
    output_folder = output_folder / "raw"
    output_folder.mkdir(parents=True, exist_ok=True)

    save_camera_parameters(
        blender_file=blender_file,
        output_path=output_folder / "camera.json",
    )

    create_samples(
        blender_file=blender_file,
        randomize_script=python_script,
        output_path=output_folder,
        num_workers=num_workers,
        samples_per_worker=samples_per_worker,
    )


if __name__ == "__main__":
    main()
