'''공통 observation/action을 yam-abc-reproduce 단일 arm raw episode로 저장하는 recorder를 생성한다.'''

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from yam_abc_reproduce.camera.interface import CameraMode
from yam_abc_reproduce.data import codec
from yam_abc_reproduce.config import CameraConfig as YamABCCameraConfig
from yam_abc_reproduce.config import RobotConfig as YamABCRobotConfig
from yam_abc_reproduce.config import StationConfig
from yam_abc_reproduce.data.recorder import EpisodeRecorder
from yam_abc_reproduce.data.schema import EpisodeMeta

from ..config import CameraDeviceConfig, RunConfig
from ..robot.kinematics import GRASP_SITE_NAME, YamEndEffectorKinematics
from ..types import RobotAction, RobotObservation
from .recorder import EncodedStep, YamABCRecorderAdapter

# yam-abc 단일 arm schema의 기본 arm prefix
ARM_NAME: str = 'left'
ARM_DOF: int = 6
ROBOT_STATE_DIMENSION: int = 7
IMAGE_KEY: str = 'rgb'
CAMERA_TYPE: str = 'v4l2_rgb'
VIDEO_ENCODER_ENV: str = 'YAM_ABC_VIDEO_ENCODER'
VIDEO_ENCODER: str = 'libx264'
# yam-abc schema 명명 규칙을 따른 measured와 commanded EE pose key
EE_POSE_KEY: str = f'{ARM_NAME}-ee_pose'
ACTION_EE_POSE_KEY: str = f'action-{ARM_NAME}-ee_pose'
EE_POSE_OBSERVATION_KEYS: dict[str, str] = {
    EE_POSE_KEY: 'ee_pose',
    ACTION_EE_POSE_KEY: 'action_ee_pose',
}
EE_POSE_METADATA: dict[str, object] = {
    'keys': [EE_POSE_KEY, ACTION_EE_POSE_KEY],
    'site': GRASP_SITE_NAME,
    'frame': 'robot base (MuJoCo world): +x forward, +y left, +z up',
    'layout': ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'],
    'position_unit': 'm',
    'quaternion_sign': 'continuous within episode, first frame qw >= 0',
}


def _pin_video_encoder(
) -> None:
    '''yam-abc episode video encoder를 GIL을 오래 잡지 않는 libx264로 고정한다.'''
    # NVENC open은 수백 ms 동안 GIL을 잡아 I2RT motor thread의 400 ms timeout을 유발
    os.environ[VIDEO_ENCODER_ENV] = VIDEO_ENCODER
    # yam-abc는 process마다 encoder를 한 번만 선택하므로 이전 선택 결과를 비움
    codec.encoder.cache_clear()


@dataclass(frozen=True)
class RecordedCamera:
    '''yam-abc EpisodeRecorder가 schema key와 metadata에 사용하는 camera 정보를 보관한다.'''

    name: str
    role: str
    mode: CameraMode = CameraMode.MONO
    keys: tuple[str, ...] = field(default=(IMAGE_KEY,))

    def image_keys(
        self,
    ) -> list[str]:
        '''CameraFrame.images에 포함되는 image key를 반환한다.'''
        return list(self.keys)


class _EndEffectorMetadataWriter:
    '''yam-abc episode writer에 EE pose key 설명을 metadata extra로 추가한다.'''

    def __init__(
        self,
        writer: object,
    ) -> None:
        '''원래 yam-abc episode writer를 저장한다.'''
        self._writer: object = writer

    def write_episode(
        self,
        episode_dir: str | Path,
        meta: EpisodeMeta,
        buffers: dict[str, object],
    ) -> None:
        '''EE pose metadata를 추가한 뒤 원래 writer로 episode를 저장한다.'''
        meta.extra['ee_pose'] = dict(EE_POSE_METADATA)
        getattr(self._writer, 'write_episode')(episode_dir, meta, buffers)


class EndEffectorEpisodeRecorder(EpisodeRecorder):
    '''yam-abc EpisodeRecorder에 measured와 commanded grasp_site EE pose를 추가로 저장한다.'''

    def __init__(
        self,
        *args: object,
        **kwargs: object,
    ) -> None:
        '''yam-abc recorder를 생성하고 writer에 EE pose metadata를 연결한다.'''
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.writer = _EndEffectorMetadataWriter(self.writer)
        self._previous_quaternions: dict[str, NDArray[np.float64] | None] = {}

    def start(
        self,
        task_name: str,
    ) -> Path:
        '''Episode를 시작하고 EE pose buffer와 quaternion 부호 기준을 초기화한다.'''
        episode_dir: Path = super().start(task_name)
        with self._lock:
            key: str
            for key in EE_POSE_OBSERVATION_KEYS:
                self._buf[key] = []  # type: ignore[index]
            self._previous_quaternions = {key: None for key in EE_POSE_OBSERVATION_KEYS}
        return episode_dir

    def _continuous_pose(
        self,
        key: str,
        pose: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        '''Shape `(7,)` pose의 quaternion 부호를 이전 frame과 이어지도록 맞춰 반환한다.'''
        continuous_pose: NDArray[np.float64] = np.asarray(pose, dtype=np.float64).copy()
        previous_quaternion: NDArray[np.float64] | None = self._previous_quaternions[key]
        # q와 -q는 같은 회전이므로 부호가 튀지 않게 첫 frame은 qw >= 0, 이후는 이전 frame 기준으로 맞춤
        if previous_quaternion is None:
            should_flip: bool = bool(continuous_pose[6] < 0.0)
        else:
            should_flip = bool(np.dot(continuous_pose[3:], previous_quaternion) < 0.0)
        if should_flip:
            continuous_pose[3:] = -continuous_pose[3:]
        self._previous_quaternions[key] = continuous_pose[3:].copy()
        return continuous_pose

    def tick(
        self,
        actions: dict[str, NDArray[np.float64]],
        obs: dict[str, dict[str, NDArray[np.float64]]],
        frames: dict[str, object],
    ) -> None:
        '''yam-abc 기본 key를 저장한 뒤 같은 step의 EE pose를 저장한다.'''
        super().tick(actions, obs, frames)  # type: ignore[arg-type]
        with self._lock:
            if not self.is_recording or self._buf is None:
                return
            key: str
            observation_key: str
            for key, observation_key in EE_POSE_OBSERVATION_KEYS.items():
                self._buf[key].append(self._continuous_pose(key, obs[ARM_NAME][observation_key]))


def encode_single_arm_step(
    observation: RobotObservation,
    action: RobotAction,
    camera_roles: tuple[str, ...],
    kinematics: YamEndEffectorKinematics,
) -> EncodedStep:
    '''Shape `(7,)` state와 action, EE pose, role별 camera frame을 yam-abc tick 입력으로 변환한다.'''
    if len(observation.state) != ROBOT_STATE_DIMENSION:
        raise ValueError(f'observation state must have shape (7,), got ({len(observation.state)},)')
    if len(action.values) != ROBOT_STATE_DIMENSION:
        raise ValueError(f'action must have shape (7,), got ({len(action.values)},)')
    # Shape `(7,)` state를 shape `(6,)` joint와 shape `(1,)` gripper로 분리
    state: NDArray[np.float64] = np.asarray(observation.state, dtype=np.float64)
    action_values: NDArray[np.float64] = np.asarray(action.values, dtype=np.float64)
    obs: dict[str, dict[str, NDArray[np.float64]]] = {
        ARM_NAME: {
            'joint_pos': state[:ARM_DOF],
            'gripper_pos': state[ARM_DOF:],
            # Shape `(6,)` measured와 commanded joint를 shape `(7,)` grasp_site pose로 변환
            'ee_pose': kinematics.pose(state[:ARM_DOF]),
            'action_ee_pose': kinematics.pose(action_values[:ARM_DOF]),
        },
    }
    actions: dict[str, NDArray[np.float64]] = {
        ARM_NAME: action_values,
    }
    frames: dict[str, object] = {}
    role: str
    for role in camera_roles:
        if role not in observation.images:
            raise ValueError(f'observation is missing camera {role!r}')
        # RecordedCamera name과 role이 같으므로 role을 frame key로 사용
        frames[role] = observation.images[role]
    return actions, obs, frames


def create_yam_abc_recorder(
    config: RunConfig,
) -> YamABCRecorderAdapter:
    '''RunConfig의 단일 arm과 camera 설정으로 yam-abc EpisodeRecorder adapter를 생성한다.'''
    devices: tuple[CameraDeviceConfig, ...] = config.common.camera.devices
    if not devices:
        raise ValueError('save_teleop_data requires at least one camera in CameraConfig.devices')
    _pin_video_encoder()
    # Frame은 control step마다 저장되므로 video fps는 control frequency와 같아야 함
    video_fps: int = round(config.common.control_hz)
    station: StationConfig = StationConfig(
        robot=YamABCRobotConfig(
            arm_name=ARM_NAME,
            num_arm_joints=ARM_DOF,
            gripper_type=config.common.robot.gripper_type,
        ),
        cameras=[
            YamABCCameraConfig(
                name=device.role,
                type=CAMERA_TYPE,
                role=device.role,
                width=device.output_size,
                height=device.output_size,
                fps=video_fps,
            )
            for device in devices
        ],
        control_hz=config.common.control_hz,
        save_root=config.data_root,
        task_name=config.task_prompt,
    )
    recorder: EndEffectorEpisodeRecorder = EndEffectorEpisodeRecorder(
        save_root=config.data_root,
        station=station,
        cameras=[RecordedCamera(name=device.role, role=device.role) for device in devices],
        arm_names=[ARM_NAME],
    )
    return YamABCRecorderAdapter(
        recorder=recorder,
        step_encoder=partial(
            encode_single_arm_step,
            camera_roles=config.common.camera.roles,
            kinematics=YamEndEffectorKinematics(config.common.robot.gripper_type),
        ),
    )
