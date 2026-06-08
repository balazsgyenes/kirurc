import math

import bpy
from mathutils import Vector
import numpy as np


def randomize_scene(rng: np.random.Generator):

    random_deg2rad = lambda min_deg, max_deg: math.radians(rng.uniform(min_deg, max_deg))

    for _ in range(3):

        z_random = random_deg2rad(0, 90)
        x_random = random_deg2rad(-10, 10) + (z_random / math.radians(90)) * random_deg2rad(-70, 70)
        y_random = random_deg2rad(-60, 60)
        bpy.data.objects["0Armature"].pose.bones["Bone.001"].rotation_euler[0] = x_random
        bpy.data.objects["0Armature"].pose.bones["Bone.001"].rotation_euler[1] = y_random
        bpy.data.objects["0Armature"].pose.bones["Bone.001"].rotation_euler[2] = z_random
        
        bpy.data.objects["Gallbladder_randomization_control_object"].location[0] = rng.uniform(-1000, 1000)
        bpy.data.objects["Gallbladder_randomization_control_object"].location[1] = rng.uniform(-1000, 1000)
        bpy.data.objects["Gallbladder_randomization_control_object"].location[2] = rng.uniform(-1000, 1000)

    distance = rng.uniform(300, 400)
    inclination = random_deg2rad(30, 70)
    direction = random_deg2rad(-180, 180)

    cam_x =  math.cos(inclination) * math.sin(direction) * distance
    cam_y = -math.cos(inclination) * math.cos(direction) * distance
    cam_z =  math.sin(inclination)                       * distance

    bpy.data.objects["Camera"].location = Vector((cam_x, cam_y, cam_z))

    bpy.data.objects["0Lookat"].location[0] = rng.uniform(-1.0, 1.0) * 5
    bpy.data.objects["0Lookat"].location[1] = rng.uniform(-1.0, 1.0) * 5
    bpy.data.objects["0Lookat"].location[2] = rng.uniform(-1.0, 1.0) * 5

    bpy.data.objects["Camera"].rotation_euler[0] += rng.uniform(-180.0, 180.0)


def reset_scene():
    bpy.data.objects["0Armature"].pose.bones["Bone.001"].rotation_euler[0] = 0
    bpy.data.objects["0Armature"].pose.bones["Bone.001"].rotation_euler[1] = 0
    bpy.data.objects["0Armature"].pose.bones["Bone.001"].rotation_euler[2] = 0

    bpy.data.objects["Gallbladder_randomization_control_object"].location[0] = 0
    bpy.data.objects["Gallbladder_randomization_control_object"].location[1] = 0
    bpy.data.objects["Gallbladder_randomization_control_object"].location[2] = 0
