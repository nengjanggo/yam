'''I2RT MuJoCo YAM state를 passive viewer에 표시한다.'''

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray
from PIL import Image


class PassiveViewerHandle(Protocol):
    '''MuJoCo passive viewer의 최소 lifecycle interface를 정의한다.'''

    def sync(
        self,
    ) -> None:
        '''Model state를 viewer에 반영한다.'''
        ...

    def close(
        self,
    ) -> None:
        '''Viewer window를 종료한다.'''
        ...


class MujocoRobotViewer:
    '''Shape `(7,)` YAM command state를 독립적인 MuJoCo viewer model에 반영한다.'''

    def __init__(
        self,
        xml_path: str,
        stream_frame_path: str | None = None,
        stream_width: int = 640,
        stream_height: int = 480,
        stream_fps: float = 15.0,
        stream_jpeg_quality: int = 70,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        '''YAM XML, passive viewer와 optional Quest stream 설정을 저장한다.'''
        if stream_width <= 0 or stream_height <= 0:
            raise ValueError('stream dimensions must be positive')
        if stream_fps <= 0.0:
            raise ValueError('stream_fps must be positive')
        if not 1 <= stream_jpeg_quality <= 95:
            raise ValueError('stream_jpeg_quality must be in [1, 95]')
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_path(xml_path)
        self._data: mujoco.MjData = mujoco.MjData(self._model)
        self._viewer: PassiveViewerHandle | None = None
        self._stream_frame_path: Path | None = (
            None if stream_frame_path is None else Path(stream_frame_path)
        )
        self._stream_width: int = stream_width
        self._stream_height: int = stream_height
        self._stream_period_s: float = 1.0 / stream_fps
        self._stream_jpeg_quality: int = stream_jpeg_quality
        self._clock: Callable[[], float] = clock
        self._last_stream_timestamp_s: float | None = None
        self._renderer: mujoco.Renderer | None = None
        self._stream_camera: mujoco.MjvCamera | None = None

    def _set_joint_position(
        self,
        joint_name: str,
        position: float,
    ) -> None:
        '''Joint name에 해당하는 scalar qpos를 설정한다.'''
        joint_id: int = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            return
        qpos_address: int = int(self._model.jnt_qposadr[joint_id])
        self._data.qpos[qpos_address] = position

    def _apply_state(
        self,
        state: tuple[float, ...],
    ) -> None:
        '''Shape `(7,)` YAM state를 MuJoCo arm과 gripper qpos에 반영한다.'''
        if len(state) != 7:
            raise ValueError('YAM viewer state must have shape (7,)')
        joint_index: int
        for joint_index in range(6):
            self._set_joint_position(
                f'joint{joint_index + 1}',
                state[joint_index],
            )
        normalized_gripper: float = min(1.0, max(0.0, state[6]))
        gripper_joint_name: str
        for gripper_joint_name in ('joint7', 'joint8'):
            joint_id: int = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_JOINT,
                gripper_joint_name,
            )
            if joint_id >= 0:
                lower: float = float(self._model.jnt_range[joint_id, 0])
                upper: float = float(self._model.jnt_range[joint_id, 1])
                self._set_joint_position(
                    gripper_joint_name,
                    lower + normalized_gripper * (upper - lower),
                )
        mujoco.mj_forward(self._model, self._data)

    def _publish_stream_frame(
        self,
        force: bool = False,
    ) -> None:
        '''MuJoCo RGB frame을 FPS 제한에 맞춰 atomic JPEG file로 갱신한다.'''
        if self._stream_frame_path is None or self._renderer is None:
            return
        timestamp_s: float = self._clock()
        if (
            not force
            and self._last_stream_timestamp_s is not None
            and timestamp_s - self._last_stream_timestamp_s < self._stream_period_s
        ):
            return
        self._renderer.update_scene(
            self._data,
            camera=self._stream_camera,
        )
        # Shape `(stream_height, stream_width, 3)` RGB frame을 render
        rgb_frame: NDArray[np.uint8] = self._renderer.render()
        self._stream_frame_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path = self._stream_frame_path.with_name(
            f'.{self._stream_frame_path.name}.tmp'
        )
        Image.fromarray(rgb_frame).save(
            temporary_path,
            format='JPEG',
            quality=self._stream_jpeg_quality,
        )
        os.replace(temporary_path, self._stream_frame_path)
        self._last_stream_timestamp_s = timestamp_s

    def _connect_stream_renderer(
        self,
    ) -> None:
        '''Quest stream을 위한 offscreen renderer와 adaptive free camera를 생성한다.'''
        if self._stream_frame_path is None:
            return
        self._renderer = mujoco.Renderer(
            self._model,
            height=self._stream_height,
            width=self._stream_width,
        )
        self._stream_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._stream_camera)
        # Shape `(ngeom, 3)` geometry position에서 현재 robot의 shape `(3,)` 중심을 계산
        geometry_minimum: NDArray[np.float64] = np.min(self._data.geom_xpos, axis=0)
        geometry_maximum: NDArray[np.float64] = np.max(self._data.geom_xpos, axis=0)
        self._stream_camera.lookat[:] = (geometry_minimum + geometry_maximum) / 2.0
        self._stream_camera.distance = 2.0 * self._model.stat.extent
        # 작업자 정면과 robot의 +x 진행 방향을 일치시켜 뒤쪽에서 렌더링
        self._stream_camera.azimuth = 0.0
        self._stream_camera.elevation = -30.0
        self._publish_stream_frame(force=True)

    def connect(
        self,
        initial_state: tuple[float, ...],
    ) -> None:
        '''Initial YAM state를 적용하고 Quest stream 또는 passive viewer를 연다.'''
        self._apply_state(initial_state)
        if self._stream_frame_path is not None:
            self._connect_stream_renderer()
            return
        self._viewer = cast(
            PassiveViewerHandle,
            mujoco.viewer.launch_passive(self._model, self._data),
        )
        self._viewer.sync()

    def sync(
        self,
        state: tuple[float, ...],
    ) -> None:
        '''최신 shape `(7,)` YAM state를 viewer에 반영한다.'''
        self._apply_state(state)
        if self._viewer is not None:
            self._viewer.sync()
        self._publish_stream_frame()

    def close(
        self,
    ) -> None:
        '''Passive viewer window를 종료한다.'''
        if self._viewer is not None:
            self._viewer.close()
        if self._renderer is not None:
            self._renderer.close()
        self._viewer = None
        self._renderer = None
        self._stream_camera = None
