'''Camera RGB frame을 policy 입력용 정사각형 image로 변환한다.'''

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import CameraFitMode


def fit_square_image(
    image: NDArray[np.uint8],
    fit_mode: CameraFitMode,
    output_size: int,
) -> NDArray[np.uint8]:
    '''Shape `(H, W, 3)` image를 center crop 또는 zero pad 후 shape `(N, N, 3)`으로 resize한다.'''
    if image.ndim != 3:
        raise ValueError('image must have shape (H, W, C)')
    if output_size <= 0:
        raise ValueError('output_size must be positive')
    height: int = int(image.shape[0])
    width: int = int(image.shape[1])
    square: NDArray[np.uint8]
    if fit_mode == 'center_crop':
        # 긴 축의 양 끝을 잘라 shape `(H, W, C)`를 shape `(S, S, C)`로 변환
        side: int = min(height, width)
        top: int = (height - side) // 2
        left: int = (width - side) // 2
        square = image[top:top + side, left:left + side]
    elif fit_mode == 'zero_pad':
        # 짧은 축의 양 끝에 0을 채워 shape `(H, W, C)`를 shape `(S, S, C)`로 변환
        side = max(height, width)
        top = (side - height) // 2
        left = (side - width) // 2
        square = np.zeros((side, side, image.shape[2]), dtype=image.dtype)
        square[top:top + height, left:left + width] = image
    else:
        raise ValueError(f'unsupported camera fit_mode: {fit_mode}')
    # 축소는 aliasing이 적은 area interpolation을 사용
    interpolation: int = cv2.INTER_AREA if side >= output_size else cv2.INTER_LINEAR
    # Shape `(S, S, C)`를 shape `(N, N, C)`로 resize
    resized: NDArray[np.uint8] = cv2.resize(
        np.ascontiguousarray(square),
        (output_size, output_size),
        interpolation=interpolation,
    )
    return resized
