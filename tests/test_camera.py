'''Camera image 정사각형 변환과 CameraRig lifecycle을 hardware 없이 검증한다.'''

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
# Notebook과 같은 방식으로 vendored yam-abc-reproduce를 import path에 추가
sys.path.insert(0, str(PROJECT_ROOT / 'third_party' / 'yam-abc-reproduce'))

from yam_abc_reproduce.camera.interface import CameraFrame, CameraMode  # noqa: E402

from yam_control.camera import CameraRig, fit_square_image  # noqa: E402
from yam_control.config import CameraConfig, CameraDeviceConfig  # noqa: E402


def _banded_image(
) -> NDArray[np.uint8]:
    '''좌우 210 px은 빨강, 가운데 540 px은 초록인 shape `(540, 960, 3)` image를 반환한다.'''
    image: NDArray[np.uint8] = np.zeros((540, 960, 3), dtype=np.uint8)
    image[:, :210] = (255, 0, 0)
    image[:, 210:750] = (0, 255, 0)
    image[:, 750:] = (255, 0, 0)
    return image


class FakeClock:
    '''Test에서 직접 바꿀 수 있는 현재 시각을 제공한다.'''

    def __init__(
        self,
        now_s: float,
    ) -> None:
        '''초기 현재 시각을 저장한다.'''
        self.now_s: float = now_s

    def __call__(
        self,
    ) -> float:
        '''현재 시각을 반환한다.'''
        return self.now_s


class FakeCameraDriver:
    '''고정 timestamp의 shape `(224, 224, 3)` frame을 반환하는 camera driver이다.'''

    mode: CameraMode = CameraMode.MONO

    def __init__(
        self,
        config: CameraDeviceConfig,
        timestamp_ms: float,
    ) -> None:
        '''Role과 frame timestamp를 저장한다.'''
        self.name: str = config.role
        self.role: str = config.role
        self._timestamp_ms: float = timestamp_ms
        self.stopped: bool = False

    def image_keys(
        self,
    ) -> list[str]:
        '''Mono RGB image key를 반환한다.'''
        return ['rgb']

    def read(
        self,
    ) -> CameraFrame:
        '''고정 frame을 반환한다.'''
        return CameraFrame(
            images={'rgb': np.zeros((224, 224, 3), dtype=np.uint8)},
            timestamp_ms=self._timestamp_ms,
        )

    def stop(
        self,
    ) -> None:
        '''Stop 상태를 기록한다.'''
        self.stopped = True


def _camera_config(
) -> CameraConfig:
    '''Top과 wrist camera를 포함한 CameraConfig를 반환한다.'''
    return CameraConfig(
        devices=(
            CameraDeviceConfig(role='top', device_path='/dev/fake-top'),
            CameraDeviceConfig(role='wrist', device_path='/dev/fake-wrist'),
        ),
        settle_time_s=1.0,
        max_frame_age_s=0.5,
    )


def _no_sleep(
    duration_s: float,
) -> None:
    '''Test에서 camera 안정화 대기를 생략한다.'''
    del duration_s


class FitSquareImageTest(unittest.TestCase):
    '''Center crop과 zero pad 이후 224x224 resize를 검증한다.'''

    def test_center_crop_removes_left_and_right_edges(
        self,
    ) -> None:
        '''16:9 image의 좌우를 잘라 가운데 초록 영역만 남기는지 검증한다.'''
        output: NDArray[np.uint8] = fit_square_image(_banded_image(), 'center_crop', 224)

        self.assertEqual(output.shape, (224, 224, 3))
        self.assertTrue(np.all(output[..., 0] == 0))
        self.assertTrue(np.all(output[..., 1] == 255))

    def test_zero_pad_adds_black_rows_above_and_below(
        self,
    ) -> None:
        '''16:9 image 위아래에 검은 영역을 채우고 좌우 영역은 유지하는지 검증한다.'''
        output: NDArray[np.uint8] = fit_square_image(_banded_image(), 'zero_pad', 224)

        self.assertEqual(output.shape, (224, 224, 3))
        # 960 중 210 px 위아래 padding은 224 기준 약 49 px
        self.assertTrue(np.all(output[:45] == 0))
        self.assertTrue(np.all(output[-45:] == 0))
        self.assertTrue(np.all(output[112, :40, 0] == 255))
        self.assertTrue(np.all(output[112, 60:160, 1] == 255))

    def test_unknown_fit_mode_is_rejected(
        self,
    ) -> None:
        '''지원하지 않는 fit mode를 거부하는지 검증한다.'''
        with self.assertRaises(ValueError):
            fit_square_image(_banded_image(), 'stretch', 224)  # type: ignore[arg-type]


class CameraConfigTest(unittest.TestCase):
    '''Camera configuration validation을 검증한다.'''

    def test_duplicate_roles_are_rejected(
        self,
    ) -> None:
        '''같은 role의 camera 두 개를 거부하는지 검증한다.'''
        with self.assertRaises(ValueError):
            CameraConfig(
                devices=(
                    CameraDeviceConfig(role='top', device_path='/dev/a'),
                    CameraDeviceConfig(role='top', device_path='/dev/b'),
                ),
            )

    def test_roles_follow_device_order(
        self,
    ) -> None:
        '''Roles가 device 순서를 따르는지 검증한다.'''
        self.assertEqual(_camera_config().roles, ('top', 'wrist'))


class CameraRigTest(unittest.TestCase):
    '''Fake driver로 CameraRig frame 신선도와 해제 흐름을 검증한다.'''

    def test_read_frames_returns_latest_frame_per_role(
        self,
    ) -> None:
        '''Top과 wrist camera의 최신 frame을 role별로 반환하는지 검증한다.'''
        drivers: list[FakeCameraDriver] = []

        def driver_factory(
            config: CameraDeviceConfig,
        ) -> FakeCameraDriver:
            '''Fake driver를 생성하고 기록한다.'''
            driver: FakeCameraDriver = FakeCameraDriver(config, timestamp_ms=10_000.0)
            drivers.append(driver)
            return driver

        sleeps: list[float] = []
        rig: CameraRig = CameraRig(
            _camera_config(),
            driver_factory=driver_factory,
            clock=FakeClock(10.1),
            sleeper=sleeps.append,
        )
        rig.connect()
        frames: dict[str, CameraFrame] = rig.read_frames()
        rig.close()

        self.assertEqual(sleeps, [1.0])
        self.assertEqual(set(frames), {'top', 'wrist'})
        self.assertEqual(frames['top'].images['rgb'].shape, (224, 224, 3))
        self.assertTrue(all(driver.stopped for driver in drivers))

    def test_stale_frame_raises(
        self,
    ) -> None:
        '''Max frame age보다 오래된 frame을 거부하는지 검증한다.'''
        rig: CameraRig = CameraRig(
            _camera_config(),
            driver_factory=lambda config: FakeCameraDriver(config, timestamp_ms=10_000.0),
            clock=FakeClock(11.0),
            sleeper=_no_sleep,
        )
        rig.connect()
        try:
            with self.assertRaises(RuntimeError):
                rig.read_frames()
        finally:
            rig.close()

    def test_wrist_frame_is_published_for_quest_stream(
        self,
    ) -> None:
        '''Wrist RGB frame을 설정한 크기의 atomic JPEG file로 기록하는지 검증한다.'''
        with tempfile.TemporaryDirectory() as temporary_directory:
            stream_frame_path: Path = Path(temporary_directory) / 'wrist-frame.jpg'
            rig: CameraRig = CameraRig(
                _camera_config(),
                driver_factory=lambda config: FakeCameraDriver(config, timestamp_ms=10_000.0),
                clock=FakeClock(10.1),
                sleeper=_no_sleep,
                stream_role='wrist',
                stream_frame_path=str(stream_frame_path),
                stream_width=640,
                stream_height=480,
                stream_fps=15.0,
            )

            rig.connect()
            rig.read_frames()
            rig.close()

            with Image.open(stream_frame_path) as stream_image:
                self.assertEqual(stream_image.size, (640, 480))

    def test_failed_camera_closes_already_opened_cameras(
        self,
    ) -> None:
        '''두 번째 camera 생성 실패 시 이미 연 camera를 해제하는지 검증한다.'''
        drivers: list[FakeCameraDriver] = []

        def driver_factory(
            config: CameraDeviceConfig,
        ) -> FakeCameraDriver:
            '''Wrist camera에서만 실패한다.'''
            if config.role == 'wrist':
                raise RuntimeError('wrist camera missing')
            driver: FakeCameraDriver = FakeCameraDriver(config, timestamp_ms=10_000.0)
            drivers.append(driver)
            return driver

        rig: CameraRig = CameraRig(
            _camera_config(),
            driver_factory=driver_factory,
            clock=FakeClock(10.0),
            sleeper=_no_sleep,
        )

        with self.assertRaises(RuntimeError):
            rig.connect()
        self.assertTrue(drivers[0].stopped)
        with self.assertRaises(RuntimeError):
            rig.read_frames()


if __name__ == '__main__':
    unittest.main()
