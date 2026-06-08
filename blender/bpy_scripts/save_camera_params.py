import json
from pathlib import Path
import sys

import bpy


if __name__ == "__main__":

    argv = sys.argv[sys.argv.index("--") + 1:]
    output_path = Path(argv[0])

    camera_settings = bpy.context.scene.camera.data
    render_settings = bpy.context.scene.render

    image_width = render_settings.resolution_x
    image_height = render_settings.resolution_y

    # conversion factor from mm to pixels
    # warning: this may need to be adjusted based on sensor_fit to use
    # image_height and sensor_height
    pixels_per_mm = image_width / camera_settings.sensor_width

    # focal length is given by Blender in mm
    # we assume that fx=fy
    focal_length = camera_settings.lens * pixels_per_mm

    # offsets are given in Blender in mm relative to center of sensor
    cx = camera_settings.shift_x * pixels_per_mm + image_width / 2
    cy = camera_settings.shift_y * pixels_per_mm + image_height / 2

    camera_parameters = {
        "image_width": image_width,
        "image_height": image_height,
        "fx": float(focal_length),
        "fy": float(focal_length),
        "cx": float(cx),
        "cy": float(cy),
    }

    with open(output_path, "w") as f:
        json.dump(camera_parameters, f)
