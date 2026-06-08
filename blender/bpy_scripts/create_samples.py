import importlib.util
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np


def save_sample_data(index: int, output_path: Path):

    # save rendering as exr file
    image_path = output_path / f"{index:07d}_image.exr"
    print(f"Saving: {image_path}")
    bpy.context.scene.render.filepath = str(image_path)
    bpy.ops.render.render(write_still=True)

    # write positions of target points to a json file
    obj_camera = bpy.context.scene.camera
    camera_matrix = obj_camera.matrix_world.inverted()

    targets = {}
    for i in range(2):
        # get object representing target point in scene
        target_object = bpy.data.objects[f"TargetCenterDuct{i+1}"]

        # convert position from world to camera coordinates
        target = target_object.matrix_world.translation
        target = camera_matrix @ Vector((*target, 1.0))
        target = tuple(target)[:3]

        targets[f"Target{i+1}"] = target

    target_info_path = output_path / f"{index:07d}_target.json"
    with open(target_info_path, "w") as f:
        json.dump(targets, f, indent=4)


if __name__ == "__main__":

    argv = sys.argv[sys.argv.index("--") + 1:]
    from_index = int(argv[0])
    to_index = int(argv[1])
    randomize_script = Path(argv[2])
    output_path = Path(argv[3])
    
    # load the scene-specific randomization script
    spec = importlib.util.spec_from_file_location(randomize_script.stem, randomize_script)
    randomize = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(randomize)

    # fix random seed for reproducibility
    rng = np.random.default_rng(from_index)

    for index in range(from_index, to_index):
        bpy.context.view_layer.update()
        randomize.randomize_scene(rng)
        bpy.context.view_layer.update()
        save_sample_data(index, output_path)
