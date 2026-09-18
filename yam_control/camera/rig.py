'''설정된 camera들을 background worker로 읽고 role별 최신 frame을 제공한다.'''

from __future__ import annotations

import time
from collections.abc import Callable

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
    ) -> None:
        '''Camera configuration, driver factory와 frame age 기준 clock을 저장한다.'''
        if warmup_timeout_s <= 0.0:
            raise ValueError('warmup_timeout_s must be positive')
        self._config: CameraConfig = config
        self._driver_factory: CameraDriverFactory = driver_factory
        self._clock: Clock = clock
        self._sleeper: Sleeper = sleeper
        self._warmup_timeout_s: float = warmup_timeout_s
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
