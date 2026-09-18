'''실행 mode별 configuration과 조합 validation을 정의한다.'''

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias

RunMode: TypeAlias = Literal['teleop', 'inference']
TeleopSourceType: TypeAlias = Literal['leader', 'quest3']
VLAType: TypeAlias = Literal['pi0', 'pi0.5', 'groot']
ExecutionTarget: TypeAlias = Literal['real', 'mujoco']
QuestDisplayMode: TypeAlias = Literal['none', 'status', 'robot_camera']
QuestControllerHand: TypeAlias = Literal['left', 'right']
CameraRole: TypeAlias = Literal['top', 'wrist']
CameraFitMode: TypeAlias = Literal['center_crop', 'zero_pad']


def _validate_pose(
    pose: tuple[float, ...],
    field_name: str,
) -> None:
    '''단일 YAM pose가 arm joint 6개와 gripper 1개인지 검증한다.'''
    pose_dimension: int = 7
    if len(pose) != pose_dimension:
        raise ValueError(f'{field_name} must have shape (7,), got ({len(pose)},)')


@dataclass(frozen=True)
class RobotConfig:
    '''실제 또는 MuJoCo YAM 연결과 episode 시작 pose를 정의한다.'''

    follower_can_channel: str = 'can0'
    leader_can_channel: str = 'can1'
    gripper_type: str = 'linear_4310'
    episode_initial_pose: tuple[float, ...] = (0.0, 1.2, 0.9, 0.0, 0.0, 0.0, 1.0)
    human_reset_pose: tuple[float, ...] | None = None

    def __post_init__(
        self,
    ) -> None:
        '''Episode initial pose와 optional human reset pose shape을 검증한다.'''
        _validate_pose(self.episode_initial_pose, 'episode_initial_pose')
        if self.human_reset_pose is not None:
            _validate_pose(self.human_reset_pose, 'human_reset_pose')

    def resolved_human_reset_pose(
        self,
    ) -> tuple[float, ...]:
        '''Human reset pose가 없으면 episode initial pose를 반환한다.'''
        if self.human_reset_pose is None:
            return self.episode_initial_pose
        return self.human_reset_pose


@dataclass(frozen=True)
class QuestConfig:
    '''Quest 3 relay와 YAM retargeting configuration을 정의한다.'''

    relay_host: str = '127.0.0.1'
    relay_port: int = 8443
    display_mode: QuestDisplayMode = 'none'
    controller_hand: QuestControllerHand = 'right'
    translation_scale: float = 0.5
    rotation_scale: float = 0.5
    position_reach_limit_m: float = 0.10
    rotation_reach_limit_rad: float = 0.35
    max_joint_delta_rad: float = 0.04
    ik_orientation_cost: float = 0.3
    max_frame_age_s: float = 0.25
    stream_frame_path: str = '/tmp/yam-mujoco-frame.jpg'
    stream_width: int = 640
    stream_height: int = 480
    stream_fps: float = 15.0
    stream_jpeg_quality: int = 70
    r_calib: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, -1.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )

    def __post_init__(
        self,
    ) -> None:
        '''Quest relay, controller와 retargeting 제한이 유효한지 검증한다.'''
        if self.display_mode not in ('none', 'status', 'robot_camera'):
            raise ValueError(f'unsupported Quest display_mode: {self.display_mode}')
        if self.controller_hand not in ('left', 'right'):
            raise ValueError(f'unsupported controller_hand: {self.controller_hand}')
        if not 1 <= self.relay_port <= 65535:
            raise ValueError('relay_port must be in [1, 65535]')
        positive_values: tuple[tuple[str, float], ...] = (
            ('translation_scale', self.translation_scale),
            ('rotation_scale', self.rotation_scale),
            ('position_reach_limit_m', self.position_reach_limit_m),
            ('rotation_reach_limit_rad', self.rotation_reach_limit_rad),
            ('max_joint_delta_rad', self.max_joint_delta_rad),
            ('ik_orientation_cost', self.ik_orientation_cost),
            ('max_frame_age_s', self.max_frame_age_s),
        )
        field_name: str
        value: float
        for field_name, value in positive_values:
            if value <= 0.0:
                raise ValueError(f'{field_name} must be positive')
        if len(self.r_calib) != 3 or any(len(row) != 3 for row in self.r_calib):
            raise ValueError('r_calib must have shape (3, 3)')
        if not self.stream_frame_path:
            raise ValueError('stream_frame_path must not be empty')
        if self.stream_width <= 0 or self.stream_height <= 0:
            raise ValueError('stream dimensions must be positive')
        if self.stream_fps <= 0.0:
            raise ValueError('stream_fps must be positive')
        if not 1 <= self.stream_jpeg_quality <= 95:
            raise ValueError('stream_jpeg_quality must be in [1, 95]')

    @property
    def websocket_url(
        self,
    ) -> str:
        '''Quest 3 relay의 secure WebSocket endpoint를 반환한다.'''
        return f'wss://{self.relay_host}:{self.relay_port}/ws'


@dataclass(frozen=True)
class CameraDeviceConfig:
    '''V4L2 RGB camera 하나의 capture 설정과 정사각형 output 변환 방식을 정의한다.'''

    role: CameraRole
    device_path: str
    capture_width: int = 960
    capture_height: int = 540
    capture_fps: int = 30
    pixel_format: str = 'YUYV'
    fit_mode: CameraFitMode = 'center_crop'
    output_size: int = 224

    def __post_init__(
        self,
    ) -> None:
        '''Camera role, device path, capture 값과 output 변환 방식을 검증한다.'''
        if self.role not in ('top', 'wrist'):
            raise ValueError(f'unsupported camera role: {self.role}')
        if not self.device_path:
            raise ValueError('device_path must not be empty')
        positive_values: tuple[tuple[str, int], ...] = (
            ('capture_width', self.capture_width),
            ('capture_height', self.capture_height),
            ('capture_fps', self.capture_fps),
            ('output_size', self.output_size),
        )
        field_name: str
        value: int
        for field_name, value in positive_values:
            if value <= 0:
                raise ValueError(f'{field_name} must be positive')
        if len(self.pixel_format) != 4:
            raise ValueError('pixel_format must be a four-character code')
        if self.fit_mode not in ('center_crop', 'zero_pad'):
            raise ValueError(f'unsupported camera fit_mode: {self.fit_mode}')


@dataclass(frozen=True)
class CameraConfig:
    '''Observation과 recorder가 사용하는 camera 목록, 노출 안정화 시간과 frame 신선도 제한을 정의한다.'''

    devices: tuple[CameraDeviceConfig, ...] = ()
    settle_time_s: float = 1.0
    max_frame_age_s: float = 0.5

    def __post_init__(
        self,
    ) -> None:
        '''Camera role이 중복되지 않고 시간 값이 유효한지 검증한다.'''
        if len(set(self.roles)) != len(self.roles):
            raise ValueError(f'camera roles must be unique, got {self.roles}')
        if self.settle_time_s < 0.0:
            raise ValueError('settle_time_s must be non-negative')
        if self.max_frame_age_s <= 0.0:
            raise ValueError('max_frame_age_s must be positive')

    @property
    def roles(
        self,
    ) -> tuple[CameraRole, ...]:
        '''설정된 camera role을 device 순서대로 반환한다.'''
        return tuple(device.role for device in self.devices)


@dataclass(frozen=True)
class CommonConfig:
    '''Teleoperation과 inference가 공유하는 execution configuration을 정의한다.'''

    execution_target: ExecutionTarget = 'mujoco'
    use_safety_gate: bool = True
    control_hz: float = 30.0
    robot: RobotConfig = field(default_factory=RobotConfig)
    quest: QuestConfig = field(default_factory=QuestConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)

    def __post_init__(
        self,
    ) -> None:
        '''Control frequency가 양수인지 검증한다.'''
        if self.execution_target not in ('real', 'mujoco'):
            raise ValueError(f'unsupported execution_target: {self.execution_target}')
        if self.control_hz <= 0.0:
            raise ValueError('control_hz must be positive')


@dataclass(frozen=True)
class TeleopRunConfig:
    '''Leader Arm 또는 Quest 3 teleoperation 설정을 정의한다.'''

    mode: Literal['teleop'] = 'teleop'
    teleop_source: TeleopSourceType = 'quest3'
    save_teleop_data: bool = False

    def __post_init__(
        self,
    ) -> None:
        '''Teleoperation source가 지원되는 값인지 검증한다.'''
        if self.teleop_source not in ('leader', 'quest3'):
            raise ValueError(f'unsupported teleop_source: {self.teleop_source}')


@dataclass(frozen=True)
class InferenceRunConfig:
    '''VLA checkpoint inference와 RTC 사용 여부를 정의한다.'''

    mode: Literal['inference'] = 'inference'
    vla_type: VLAType = 'pi0.5'
    checkpoint_uri: str = ''
    checkpoint_revision: str | None = None
    policy_config_name: str = ''
    use_rtc: bool = False

    def __post_init__(
        self,
    ) -> None:
        '''Inference에 필요한 checkpoint와 policy config 값을 검증한다.'''
        if self.vla_type not in ('pi0', 'pi0.5', 'groot'):
            raise ValueError(f'unsupported vla_type: {self.vla_type}')
        if self.vla_type != 'groot' and not self.checkpoint_uri:
            raise ValueError('checkpoint_uri is required for π inference')
        if self.vla_type != 'groot' and not self.policy_config_name:
            raise ValueError('policy_config_name is required for π inference')


ModeConfig: TypeAlias = TeleopRunConfig | InferenceRunConfig


@dataclass(frozen=True)
class RunConfig:
    '''Mode-specific configuration과 공통 configuration을 결합한다.'''

    mode_config: ModeConfig
    common: CommonConfig
    task_prompt: str
    data_root: str = 'data/episodes'

    @property
    def mode(
        self,
    ) -> RunMode:
        '''현재 실행 mode를 반환한다.'''
        return self.mode_config.mode

    @property
    def rtc_enabled(
        self,
    ) -> bool:
        '''Inference일 때만 RTC 활성화 값을 반환하고 teleoperation에서는 False를 반환한다.'''
        if isinstance(self.mode_config, InferenceRunConfig):
            return self.mode_config.use_rtc
        return False


def build_run_config(
    mode: RunMode,
    teleop_source: TeleopSourceType,
    save_teleop_data: bool,
    vla_type: VLAType,
    checkpoint_uri: str,
    checkpoint_revision: str | None,
    policy_config_name: str,
    use_rtc: bool,
    execution_target: ExecutionTarget,
    use_safety_gate: bool,
    control_hz: float,
    task_prompt: str,
    data_root: str,
    robot: RobotConfig,
    quest: QuestConfig,
    camera: CameraConfig,
) -> RunConfig:
    '''Notebook 값을 mode별 configuration으로 정규화하여 RunConfig를 반환한다.'''
    if mode not in ('teleop', 'inference'):
        raise ValueError(f'unsupported mode: {mode}')
    common: CommonConfig = CommonConfig(
        execution_target=execution_target,
        use_safety_gate=use_safety_gate,
        control_hz=control_hz,
        robot=robot,
        quest=quest,
        camera=camera,
    )
    mode_config: ModeConfig
    if mode == 'teleop':
        # Teleoperation에서는 전역 USE_RTC 값을 의도적으로 무시
        # MuJoCo joint와 실제 camera image가 섞이지 않도록 MuJoCo에서는 SAVE_TELEOP_DATA를 무시
        mode_config = TeleopRunConfig(
            teleop_source=teleop_source,
            save_teleop_data=save_teleop_data and execution_target == 'real',
        )
    else:
        mode_config = InferenceRunConfig(
            vla_type=vla_type,
            checkpoint_uri=checkpoint_uri,
            checkpoint_revision=checkpoint_revision,
            policy_config_name=policy_config_name,
            use_rtc=use_rtc,
        )
    return RunConfig(
        mode_config=mode_config,
        common=common,
        task_prompt=task_prompt,
        data_root=data_root,
    )


def with_camera_config(
    config: RunConfig,
    camera: CameraConfig,
) -> RunConfig:
    '''RunConfig의 camera configuration만 교체한 새 RunConfig를 반환한다.'''
    return replace(
        config,
        common=replace(config.common, camera=camera),
    )
