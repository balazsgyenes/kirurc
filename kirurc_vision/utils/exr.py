from os import PathLike

import numpy as np
import torch


COLORS = ("R", "G", "B")


def read_exr(filepath: PathLike):
    return read_openexr(filepath)


def read_openexr(filepath: PathLike):
    # https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b

    import OpenEXR as exr

    # read input file
    file = exr.InputFile(str(filepath))
    header = file.header()

    # get image dimensions from header
    window = header["dataWindow"]
    height, width = window.max.y - window.min.y + 1, window.max.x - window.min.x + 1

    # get datatype of color channels from header
    rgb_datatypes = [file.header()["channels"][color].type for color in COLORS]
    assert (datatype == rgb_datatypes[0] for datatype in rgb_datatypes[1:])
    if (rgb_datatype := str(rgb_datatypes[0])) == "HALF":
        rgb_datatype = np.float16
    elif rgb_datatype == "FLOAT":
        rgb_datatype = np.float32
    else:
        raise TypeError(f"Found RGB channels of type {rgb_datatype}")

    # get color channels
    rgb = [np.frombuffer(file.channel(color), dtype=rgb_datatype, count=height * width) for color in COLORS]
    for channel in rgb:
        channel.shape = (height, width)
    rgb = np.stack(rgb, axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    # get depth channel
    depth = np.frombuffer(file.channel("Z"), dtype=np.float32, count=height * width)
    depth.shape = (height, width)

    file.close()

    # copy depth to make it writeable and suppress pytorch warning
    rgb, depth = torch.from_numpy(rgb), torch.from_numpy(depth.copy())

    return rgb, depth


def read_minexr(filepath: PathLike):
    # minexr is much simpler to use than OpenEXR, but cannot handle exr files
    # with compression
    
    import minexr

    with open(filepath, "rb") as file_handle:
        reader = minexr.load(file_handle)
        rgb = reader.select(["R", "G", "B"])
        depth = reader.select(["Z"])    

    # make contiguous because channels are out of order
    # open3d constructors require contiguous arrays
    rgb = np.ascontiguousarray(rgb)
    depth = np.ascontiguousarray(depth)
    
    rgb = (rgb * 255).astype(np.uint8)

    # copy depth to make it writeable and suppress pytorch warning
    rgb, depth = torch.from_numpy(rgb), torch.from_numpy(depth.copy())

    return rgb, depth
