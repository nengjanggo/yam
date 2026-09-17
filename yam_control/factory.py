'''RunConfig를 서로 독립적인 runtime component 조합으로 변환한다.'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from types import ModuleType
from typing import cast

from .config import ExecutionTarget, InferenceRunConfig, QuestConfig, RobotConfig, RunConfig, TeleopRunConfig
from .data import NullRecorder
from .interfaces import ActionProducer, EpisodeRecorder, RobotBackend, SafetyGate
from .robot.i2rt_adapter import I2RTRobot, I2RTRobotBackend, RobotVisualizer
from .safety import PassThroughSafetyGate
from .session import RunSession
from .types import RobotAction, RobotObservation, SafetyDecision

RobotFactory = Callable[[RunConfig], RobotBackend]
ActionProducerFactory = Callable[[RunConfig], ActionProducer]
SafetyGateFactory = Callable[[RunConfig], SafetyGate]
RecorderFactory = Callable[[RunConfig], EpisodeRecorder]


def _numpy_vector(
    values: tuple[float, ...],
) -> object:
    '''Tuple을 I2RT가 사용하는 NumPy shape `(S,)` array로 변환한다.'''
    numpy_module: ModuleType = import_module('numpy')
    asarray: Callable[..., object] = getattr(numpy_module, 'asarray')
    return asarray(values, dtype=float)


def _load_i2rt_robot(
    robot_config: RobotConfig,
    execution_target: ExecutionTarget,
) -> I2RTRobot:
    '''Vendored I2RT에서 real 또는 MuJoCo YAM Robot을 지연 생성한다.'''
    get_robot_module: ModuleType = import_module('i2rt.robots.get_robot')
    utils_module: ModuleType = import_module('i2rt.robots.utils')
    get_yam_robot: Callable[..., object] = getattr(get_robot_module, 'get_yam_robot')
    gripper_type_class: object = getattr(utils_module, 'GripperType')
    from_string_name: Callable[[str], object] = getattr(gripper_type_class, 'from_string_name')
    gripper_type: object = from_string_name(robot_config.gripper_type)
    robot: object = get_yam_robot(
        channel=robot_config.follower_can_channel,
        gripper_type=gripper_type,
        sim=execution_target == 'mujoco',
    )
    return cast(I2RTRobot, robot)


def _create_default_robot(
    config: RunConfig,
) -> RobotBackend:
    '''RunConfig의 execution target에 맞는 I2RTRobotBackend를 생성한다.'''
    visualizer_factory: Callable[[I2RTRobot], RobotVisualizer] | None = None
    if config.common.execution_target == 'mujoco':
        visualizer_factory = partial(
            _create_mujoco_visualizer,
            quest_config=config.common.quest,
        )
    return I2RTRobotBackend(
        config=config.common.robot,
        execution_target=config.common.execution_target,
        loader=_load_i2rt_robot,
        vector_converter=_numpy_vector,
        visualizer_factory=visualizer_factory,
    )


def _create_mujoco_visualizer(
    robot: I2RTRobot,
    quest_config: QuestConfig,
) -> RobotVisualizer:
    '''I2RT SimRobot의 XML을 사용하는 passive MuJoCo viewer를 생성한다.'''
    from .robot.mujoco_viewer import MujocoRobotViewer

    stream_frame_path: str | None = (
        quest_config.stream_frame_path
        if quest_config.display_mode == 'robot_camera'
        else None
    )
    return MujocoRobotViewer(
        xml_path=robot.xml_path,
        stream_frame_path=stream_frame_path,
        stream_width=quest_config.stream_width,
        stream_height=quest_config.stream_height,
        stream_fps=quest_config.stream_fps,
        stream_jpeg_quality=quest_config.stream_jpeg_quality,
    )


def _create_default_quest3_action_producer(
    config: RunConfig,
) -> ActionProducer:
    '''Quest relay reader와 YAM IK retargeter를 결합한다.'''
    from .teleop.quest3 import Quest3ActionProducer, WebSocketQuestFrameReader
    from .teleop.yam_retargeter import YamQuestRetargeter

    return Quest3ActionProducer(
        reader=WebSocketQuestFrameReader(
            websocket_url=config.common.quest.websocket_url,
            controller_hand=config.common.quest.controller_hand,
        ),
        retargeter=YamQuestRetargeter(
            robot_config=config.common.robot,
            quest_config=config.common.quest,
        ),
        max_frame_age_s=config.common.quest.max_frame_age_s,
    )


class _UnavailableActionProducer:
    '''아직 주입되지 않은 Quest 3 또는 VLA adapter의 실행을 차단한다.'''

    def __init__(
        self,
        reason: str,
    ) -> None:
        '''Connect 시 표시할 구체적인 누락 사유를 저장한다.'''
        self._reason: str = reason

    def connect(
        self,
    ) -> None:
        '''누락된 adapter 사유를 RuntimeError로 반환한다.'''
        raise RuntimeError(self._reason)

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''연결되지 않은 producer는 episode reset을 수행할 수 없다.'''
        del observation

    def next_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''연결되지 않은 producer는 action을 생성할 수 없다.'''
        del observation
        raise RuntimeError(self._reason)

    def close(
        self,
    ) -> None:
        '''생성된 resource가 없으므로 아무 작업도 하지 않는다.'''


class _UnavailableSafetyGate:
    '''MuJoCo collision validator가 주입되기 전 안전 실행을 차단한다.'''

    def __init__(
        self,
        reason: str,
    ) -> None:
        '''SafetyGate 실행 시 표시할 누락 사유를 저장한다.'''
        self._reason: str = reason

    def reset(
        self,
    ) -> None:
        '''누락된 validator에는 reset할 state가 없다.'''

    def evaluate(
        self,
        observation: RobotObservation,
        action: RobotAction,
    ) -> SafetyDecision:
        '''Validator 없이 안전하다고 간주하지 않고 실행을 차단한다.'''
        del observation
        del action
        raise RuntimeError(self._reason)


class _UnavailableRecorder:
    '''Camera-aware yam-abc recorder가 주입되기 전 recording을 차단한다.'''

    def __init__(
        self,
        reason: str,
    ) -> None:
        '''Recording 시작 시 표시할 누락 사유를 저장한다.'''
        self._reason: str = reason

    def start(
        self,
        task_prompt: str,
    ) -> None:
        '''Recorder adapter가 없으면 raw episode 생성을 차단한다.'''
        del task_prompt
        raise RuntimeError(self._reason)

    def record(
        self,
        observation: RobotObservation,
        action: RobotAction,
    ) -> None:
        '''Recorder가 없으므로 control step을 저장하지 않는다.'''
        del observation
        del action

    def finish(
        self,
    ) -> None:
        '''Recorder가 없으므로 finalize할 episode가 없다.'''

    def abort(
        self,
    ) -> None:
        '''Recorder가 없으므로 폐기할 episode가 없다.'''


@dataclass(frozen=True)
class RuntimeDependencies:
    '''각 configuration 축의 구현체를 독립적으로 교체하는 factory 집합이다.'''

    robot_factory: RobotFactory = _create_default_robot
    quest3_factory: ActionProducerFactory | None = _create_default_quest3_action_producer
    pi_factory: ActionProducerFactory | None = None
    rtc_factory: ActionProducerFactory | None = None
    safety_gate_factory: SafetyGateFactory | None = None
    recorder_factory: RecorderFactory | None = None


def _select_action_producer(
    config: RunConfig,
    dependencies: RuntimeDependencies,
) -> ActionProducer:
    '''Mode별 teleoperation source 또는 VLA producer를 선택한다.'''
    mode_config: TeleopRunConfig | InferenceRunConfig = config.mode_config
    if isinstance(mode_config, TeleopRunConfig):
        if mode_config.teleop_source == 'leader':
            raise NotImplementedError(
                'Leader Arm teleoperation is not implemented because no Leader Arm is available'
            )
        if dependencies.quest3_factory is None:
            return _UnavailableActionProducer(
                'Quest 3 YAM IK retargeter와 WebXR reader를 RuntimeDependencies에 주입해야 합니다.'
            )
        return dependencies.quest3_factory(config)
    if mode_config.vla_type == 'groot':
        raise NotImplementedError('GR00T inference is not implemented yet')
    if mode_config.use_rtc:
        if dependencies.rtc_factory is None:
            return _UnavailableActionProducer(
                'π RTC upstream implementation adapter를 RuntimeDependencies에 주입해야 합니다.'
            )
        return dependencies.rtc_factory(config)
    if dependencies.pi_factory is None:
        return _UnavailableActionProducer(
            'π0 또는 π0.5 checkpoint loader를 RuntimeDependencies에 주입해야 합니다.'
        )
    return dependencies.pi_factory(config)


def _select_safety_gate(
    config: RunConfig,
    dependencies: RuntimeDependencies,
) -> SafetyGate:
    '''SafetyGate boolean에 따라 pass-through 또는 collision gate를 선택한다.'''
    if not config.common.use_safety_gate:
        return PassThroughSafetyGate()
    if dependencies.safety_gate_factory is None:
        return _UnavailableSafetyGate(
            'MuJoCo model 기반 collision validator를 RuntimeDependencies에 주입해야 합니다.'
        )
    return dependencies.safety_gate_factory(config)


def _select_recorder(
    config: RunConfig,
    dependencies: RuntimeDependencies,
) -> EpisodeRecorder:
    '''Teleoperation recording boolean에 따라 NullRecorder 또는 external recorder를 선택한다.'''
    mode_config: TeleopRunConfig | InferenceRunConfig = config.mode_config
    should_record: bool = (
        isinstance(mode_config, TeleopRunConfig)
        and mode_config.save_teleop_data
    )
    if not should_record:
        return NullRecorder()
    if dependencies.recorder_factory is None:
        return _UnavailableRecorder(
            'Camera 기종 확정 후 yam-abc recorder factory를 RuntimeDependencies에 주입해야 합니다.'
        )
    return dependencies.recorder_factory(config)


def create_session(
    config: RunConfig,
    dependencies: RuntimeDependencies | None = None,
) -> RunSession:
    '''RunConfig의 직교 configuration 축을 concrete component 조합으로 변환한다.'''
    resolved_dependencies: RuntimeDependencies = (
        RuntimeDependencies() if dependencies is None else dependencies
    )
    robot: RobotBackend = resolved_dependencies.robot_factory(config)
    action_producer: ActionProducer = _select_action_producer(config, resolved_dependencies)
    safety_gate: SafetyGate = _select_safety_gate(config, resolved_dependencies)
    recorder: EpisodeRecorder = _select_recorder(config, resolved_dependencies)
    return RunSession(
        config=config,
        robot=robot,
        action_producer=action_producer,
        safety_gate=safety_gate,
        recorder=recorder,
    )
