'''V4L2 RGB camera driver와 role별 camera rig를 public API로 노출한다.'''

from .processing import fit_square_image
from .rig import CameraRig
from .v4l2 import V4L2RGBCamera

__all__: list[str] = [
    'CameraRig',
    'V4L2RGBCamera',
    'fit_square_image',
]
