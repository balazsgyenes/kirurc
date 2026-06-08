from typing import Sequence, Union

RGB_COLOR = Sequence[int]  # e.g. [255, 0, 0]
COLOR_CODE = str  # e.g. "#FFFF00"
LABEL = Union[int, str]  # e.g. 4 or "t1_spline"
COLOR_MAPPING = dict[LABEL, Union[RGB_COLOR, COLOR_CODE]]
RGB_COLOR_MAPPING = dict[LABEL, RGB_COLOR]
