'''설정된 camera들을 background worker로 읽고 role별 최신 frame을 제공한다.'''

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps
from yam_abc_reproduce.camera.interface import CameraDriver, CameraFrame
from yam_abc_reproduce.camera.worker import CameraWorker

from ..config import CameraConfig, CameraDeviceConfig
from .v4l2 import V4L2RGBCamera

CameraDriverFactory = Callable[[CameraDeviceConfig], CameraDriver]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]


class CameraRig:
    '''Camera마다 yam-abc CameraWorker를 실행하고 신선한 최신 frame만 반환한다.'''

    def __init__(
        self,
        config: CameraConfig,
        driver_factory: CameraDriverFactory = V4L2RGBCamera,
        clock: Clock = time.time,
        sleeper: Sleeper = time.sleep,
        warmup_timeout_s: float = 5.0,
        stream_role: str | None = None,
        stream_frame_path: str | None = None,
        stream_width: int = 640,
        stream_height: int = 480,
        stream_fps: float = 15.0,
        stream_jpeg_quality: int = 70,
    ) -> None:
        '''Camera configuration과 optional Quest stream 출력을 저장한다.'''
        if warmup_timeout_s <= 0.0:
            raise ValueError('warmup_timeout_s must be positive')
        if stream_role is not None and stream_role not in config.roles:
            raise ValueError(f'stream_role {stream_role!r} is not configured')
        if stream_role is not None and not stream_frame_path:
            raise ValueError('stream_frame_path is required when stream_role is set')
        if stream_width <= 0 or stream_height <= 0 or stream_fps <= 0.0:
            raise ValueError('stream dimensions and fps must be positive')
        if not 1 <= stream_jpeg_quality <= 95:
            raise ValueError('stream_jpeg_quality must be in [1, 95]')
        self._config: CameraConfig = config
        self._driver_factory: CameraDriverFactory = driver_factory
        self._clock: Clock = clock
        self._sleeper: Sleeper = sleeper
        self._warmup_timeout_s: float = warmup_timeout_s
        self._stream_role: str | None = stream_role
        self._stream_frame_path: Path | None = (
            Path(stream_frame_path)
            if stream_role is not None and stream_frame_path is not None
            else None
        )
        self._stream_width: int = stream_width
        self._stream_height: int = stream_height
        self._stream_period_s: float = 1.0 / stream_fps
        self._stream_jpeg_quality: int = stream_jpeg_quality
        self._last_stream_timestamp_s: float | None = None
        self._workers: dict[str, CameraWorker] = {}

    @property
    def config(
        self,
    ) -> CameraConfig:
        '''Camera configuration을 반환한다.'''
        return self._config

    def connect(
        self,
    ) -> None:
        '''모든 camera를 열고 첫 frame 도착 후 자동 노출이 안정될 때까지 기다린다.'''
        if self._workers:
            return
        self._last_stream_timestamp_s = None
        device: CameraDeviceConfig
        try:
            for device in self._config.devices:
                worker: CameraWorker = CameraWorker(self._driver_factory(device))
                worker.start(warmup_timeout=self._warmup_timeout_s)
                self._workers[device.role] = worker
            # 연결 직후 frame은 자동 노출이 잡히기 전이라 어두우므로 안정화 시간만큼 대기
            self._sleeper(self._config.settle_time_s)
        except BaseException:
            # 일부 camera만 열린 상태로 남지 않도록 이미 연 camera를 해제
            self.close()
            raise

    def _publish_stream_frame(
        self,
        frames: dict[str, CameraFrame],
    ) -> None:
        '''선택한 role의 RGB frame을 설정 FPS로 atomic JPEG file에 기록한다.'''
        if self._stream_role is None or self._stream_frame_path is None:
            return
        timestamp_s: float = self._clock()
        if (
            self._last_stream_timestamp_s is not None
            and timestamp_s - self._last_stream_timestamp_s < self._stream_period_s
        ):
            return
        camera_frame: CameraFrame = frames[self._stream_role]
        # Shape `(I_h, I_w, 3)` RGB frame을 shape `(stream_height, stream_width, 3)` letterbox image로 변환
        rgb_frame: NDArray[np.uint8] = np.asarray(camera_frame.images['rgb'], dtype=np.uint8)
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError(f'stream RGB frame must have shape (I_h, I_w, 3), got {rgb_frame.shape}')
        output_image: Image.Image = ImageOps.pad(
            Image.fromarray(rgb_frame),
            (self._stream_width, self._stream_height),
            method=Image.Resampling.LANCZOS,
            color=(0, 0, 0),
        )
        self._stream_frame_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path = self._stream_frame_path.with_name(
            f'.{self._stream_frame_path.name}.tmp'
        )
        output_image.save(
            temporary_path,
            format='JPEG',
            quality=self._stream_jpeg_quality,
        )
        os.replace(temporary_path, self._stream_frame_path)
        self._last_stream_timestamp_s = timestamp_s

    def read_frames(
        self,
    ) -> dict[str, CameraFrame]:
        '''Role별 최신 CameraFrame을 반환하고 오래된 frame이 있으면 RuntimeError를 발생시킨다.'''
        if len(self._workers) != len(self._config.devices):
            raise RuntimeError('CameraRig is not connected')
        now_ms: float = self._clock() * 1000.0
        max_frame_age_ms: float = self._config.max_frame_age_s * 1000.0
        frames: dict[str, CameraFrame] = {}
        role: str
        worker: CameraWorker
        for role, worker in self._workers.items():
            frame: CameraFrame | None = worker.read()
            if frame is None:
                raise RuntimeError(f'camera {role!r} has no frame')
            frame_age_ms: float = now_ms - frame.timestamp_ms
            if frame_age_ms > max_frame_age_ms:
                raise RuntimeError(f'camera {role!r} frame is stale: {frame_age_ms:.0f} ms old')
            frames[role] = frame
        self._publish_stream_frame(frames)
        return frames

    def close(
        self,
    ) -> None:
        '''모든 camera worker와 device를 해제한다.'''
        workers: list[CameraWorker] = list(self._workers.values())
        self._workers = {}
        worker: CameraWorker
        for worker in workers:
            worker.stop()
        self._last_stream_timestamp_s = None
