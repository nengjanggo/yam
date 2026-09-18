'''OpenCV V4L2 backend로 UVC RGB stream을 읽는 yam-abc camera driver를 제공한다.'''

from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from yam_abc_reproduce.camera.interface import CameraFrame, CameraMode

from ..config import CameraDeviceConfig
from .processing import fit_square_image

Clock = Callable[[], float]
IMAGE_KEY: str = 'rgb'


class V4L2RGBCamera:
    '''UVC RGB frame을 정사각형 output으로 변환해 yam-abc CameraFrame으로 반환한다.'''

    mode: CameraMode = CameraMode.MONO

    def __init__(
        self,
        config: CameraDeviceConfig,
        clock: Clock = time.time,
    ) -> None:
        '''V4L2 device를 열고 요청한 pixel format, 해상도와 fps가 적용됐는지 확인한다.'''
        self.name: str = config.role
        self.role: str = config.role
        self._config: CameraDeviceConfig = config
        self._clock: Clock = clock
        self._capture: cv2.VideoCapture | None = cv2.VideoCapture(config.device_path, cv2.CAP_V4L2)
        if not self._capture.isOpened():
            self._capture = None
            raise RuntimeError(f'failed to open camera {self.role!r} at {config.device_path}')
        requested_fourcc: int = cv2.VideoWriter_fourcc(*config.pixel_format)
        self._capture.set(cv2.CAP_PROP_FOURCC, requested_fourcc)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_height)
        self._capture.set(cv2.CAP_PROP_FPS, config.capture_fps)
        # 지원하지 않는 조합은 driver가 다른 값으로 조용히 바꾸므로 실제 적용값을 확인
        applied: tuple[int, int, int, int] = (
            int(self._capture.get(cv2.CAP_PROP_FOURCC)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            round(self._capture.get(cv2.CAP_PROP_FPS)),
        )
        requested: tuple[int, int, int, int] = (
            requested_fourcc,
            config.capture_width,
            config.capture_height,
            config.capture_fps,
        )
        if applied != requested:
            self.stop()
            raise RuntimeError(
                f'camera {self.role!r} does not support '
                f'{config.pixel_format} {config.capture_width}x{config.capture_height}@{config.capture_fps}, '
                f'driver applied {applied[1]}x{applied[2]}@{applied[3]}'
            )

    def image_keys(
        self,
    ) -> list[str]:
        '''Mono RGB camera의 image key를 반환한다.'''
        return [IMAGE_KEY]

    def read(
        self,
    ) -> CameraFrame:
        '''다음 frame을 읽어 shape `(N, N, 3)` RGB image와 capture 시각을 반환한다.'''
        if self._capture is None:
            raise RuntimeError(f'camera {self.role!r} is not open')
        success: bool
        bgr: NDArray[np.uint8]
        success, bgr = self._capture.read()
        timestamp_ms: float = self._clock() * 1000.0
        if not success:
            raise RuntimeError(f'camera {self.role!r} read failed')
        # Shape `(H, W, 3)` BGR을 shape `(H, W, 3)` RGB로 변환
        rgb: NDArray[np.uint8] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Shape `(H, W, 3)`을 shape `(N, N, 3)` policy image로 변환
        image: NDArray[np.uint8] = fit_square_image(
            rgb,
            fit_mode=self._config.fit_mode,
            output_size=self._config.output_size,
        )
        return CameraFrame(
            images={IMAGE_KEY: image},
            timestamp_ms=timestamp_ms,
        )

    def stop(
        self,
    ) -> None:
        '''V4L2 device를 해제한다.'''
        if self._capture is not None:
            self._capture.release()
        self._capture = None
