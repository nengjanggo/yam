'''I2RT real YAM과 MuJoCo YAM을 공통 RobotBackend로 변환한다.'''

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol

from ..config import ExecutionTarget, RobotConfig
from ..types import RobotAction, RobotObservation


class I2RTRobot(Protocol):
    '''I2RT Robot에서 사용하는 최소 joint control interface를 정의한다.'''

    xml_path: str

    def get_observations(
        self,
    ) -> Mapping[str, object]:
        '''Joint와 gripper observation mapping을 반환한다.'''
        ...

    def command_joint_pos(
        self,
        joint_pos: object,
    ) -> None:
        '''Shape `(S,)` joint position command를 실행한다.'''
        ...

    def close(
        self,
    ) -> None:
        '''I2RT resource를 해제한다.'''
        ...


I2RTRobotLoader = Callable[[RobotConfig, ExecutionTarget], I2RTRobot]


class RobotVisualizer(Protocol):
    '''Robot state를 GUI에 표시하는 optional visualizer interface를 정의한다.'''

    def connect(
        self,
        initial_state: tuple[float, ...],
    ) -> None:
        '''Initial robot state로 visualizer resource를 생성한다.'''
        ...

    def sync(
        self,
        state: tuple[float, ...],
    ) -> None:
        '''최신 robot state를 visualizer에 반영한다.'''
        ...

    def close(
        self,
    ) -> None:
        '''Visualizer resource를 해제한다.'''
        ...


RobotVisualizerFactory = Callable[[I2RTRobot], RobotVisualizer]
EnvironmentResetter = Callable[[tuple[float, ...]], None]
VectorConverter = Callable[[tuple[float, ...]], object]
ImageProvider = Callable[[], Mapping[str, object]]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]


def _to_float_tuple(
    values: object,
) -> tuple[float, ...]:
    '''Array-like object를 immutable float tuple로 변환한다.'''
    iterable_values: Iterable[object] = values  # type: ignore[assignment]
    return tuple(float(value) for value in iterable_values)


class I2RTRobotBackend:
    '''I2RT의 real 또는 sim Robot을 지연 생성하는 RobotBackend이다.'''

    def __init__(
        self,
        config: RobotConfig,
        execution_target: ExecutionTarget,
        loader: I2RTRobotLoader,
        vector_converter: VectorConverter = tuple,
        visualizer_factory: RobotVisualizerFactory | None = None,
        environment_resetter: EnvironmentResetter | None = None,
        image_provider: ImageProvider | None = None,
        clock: Clock = time.monotonic,
        move_duration_s: float = 3.0,
        move_hz: float = 50.0,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        '''Robot configuration과 외부 I2RT loader를 저장한다.'''
        if move_duration_s <= 0.0:
            raise ValueError('move_duration_s must be positive')
        if move_hz <= 0.0:
            raise ValueError('move_hz must be positive')
        self._config: RobotConfig = config
        self._execution_target: ExecutionTarget = execution_target
        self._loader: I2RTRobotLoader = loader
        self._vector_converter: VectorConverter = vector_converter
        self._visualizer_factory: RobotVisualizerFactory | None = visualizer_factory
        self._environment_resetter: EnvironmentResetter | None = environment_resetter
        self._image_provider: ImageProvider | None = image_provider
        self._clock: Clock = clock
        self._move_duration_s: float = move_duration_s
        self._move_hz: float = move_hz
        self._sleeper: Sleeper = sleeper
        self._robot: I2RTRobot | None = None
        self._visualizer: RobotVisualizer | None = None

    def _require_robot(
        self,
    ) -> I2RTRobot:
        '''연결된 I2RT Robot을 반환한다.'''
        if self._robot is None:
            raise RuntimeError('I2RTRobotBackend is not connected')
        return self._robot

    def connect(
        self,
    ) -> None:
        '''Configuration에 맞는 I2RT real 또는 sim Robot을 생성한다.'''
        self._robot = self._loader(self._config, self._execution_target)
        if self._visualizer_factory is not None:
            self._visualizer = self._visualizer_factory(self._robot)
            initial_state: tuple[float, ...] = self.get_observation().state
            self._visualizer.connect(initial_state)

    def get_observation(
        self,
    ) -> RobotObservation:
        '''I2RT observation을 shape `(S,)` state와 camera mapping으로 변환한다.'''
        robot: I2RTRobot = self._require_robot()
        raw_observation: Mapping[str, object] = robot.get_observations()
        arm: tuple[float, ...] = _to_float_tuple(raw_observation['joint_pos'])
        gripper: tuple[float, ...] = _to_float_tuple(raw_observation['gripper_pos'])
        # Shape `(6,)` arm과 shape `(1,)` gripper를 shape `(7,)` state로 결합
        state: tuple[float, ...] = (*arm, *gripper)
        images: Mapping[str, object] = {} if self._image_provider is None else self._image_provider()
        return RobotObservation(
            state=state,
            images=images,
            timestamp_s=self._clock(),
        )

    def execute(
        self,
        action: RobotAction,
    ) -> None:
        '''Shape `(S,)` action을 I2RT joint position command로 전달한다.'''
        robot: I2RTRobot = self._require_robot()
        command: object = self._vector_converter(action.values)
        robot.command_joint_pos(command)
        if self._visualizer is not None:
            state: tuple[float, ...] = self.get_observation().state
            self._visualizer.sync(state)

    def move_to_pose(
        self,
        pose: tuple[float, ...],
    ) -> None:
        '''현재 state에서 지정한 shape `(S,)` pose까지 joint-space로 천천히 이동한다.'''
        start: tuple[float, ...] = self.get_observation().state
        if len(start) != len(pose):
            raise ValueError('current state and target pose dimensions must match')
        step_count: int = max(1, round(self._move_duration_s * self._move_hz))
        step_period_s: float = 1.0 / self._move_hz
        step_index: int
        for step_index in range(1, step_count + 1):
            alpha: float = step_index / step_count
            # Shape `(S,)` start에서 shape `(S,)` target까지 선형 보간
            interpolated_pose: tuple[float, ...] = tuple(
                start_value + alpha * (target_value - start_value)
                for start_value, target_value in zip(start, pose, strict=True)
            )
            self.execute(RobotAction(values=interpolated_pose))
            self._sleeper(step_period_s)

    def reset_environment(
        self,
        initial_pose: tuple[float, ...],
    ) -> None:
        '''MuJoCo environment callback을 실행하고 robot을 initial pose로 보낸다.'''
        if self._execution_target != 'mujoco':
            raise RuntimeError('reset_environment is available only for MuJoCo')
        if self._environment_resetter is not None:
            self._environment_resetter(initial_pose)
        self.move_to_pose(initial_pose)

    def hold(
        self,
    ) -> None:
        '''현재 measured state를 다시 command하여 robot pose를 유지한다.'''
        observation: RobotObservation = self.get_observation()
        self.execute(RobotAction(values=observation.state))

    def close(
        self,
    ) -> None:
        '''I2RT Robot resource를 해제하고 reference를 제거한다.'''
        if self._visualizer is not None:
            self._visualizer.close()
        if self._robot is not None:
            self._robot.close()
        self._visualizer = None
        self._robot = None
