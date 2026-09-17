'''I2RT backend의 optional MuJoCo visualizer 연결을 검증한다.'''

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

from yam_control.config import ExecutionTarget, RobotConfig
from yam_control.robot.i2rt_adapter import I2RTRobot, I2RTRobotBackend, RobotVisualizer
from yam_control.robot.mujoco_viewer import MujocoRobotViewer
from yam_control.types import RobotAction

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
YAM_XML_PATH: Path = (
    PROJECT_ROOT
    / 'third_party'
    / 'yam-abc-reproduce'
    / 'third_party'
    / 'i2rt'
    / 'i2rt'
    / 'robot_models'
    / 'arm'
    / 'yam'
    / 'yam.xml'
)


class FakeI2RTRobot:
    '''Hardware 없이 I2RT joint state를 저장한다.'''

    xml_path: str = '/tmp/fake.xml'

    def __init__(
        self,
    ) -> None:
        '''Initial shape `(7,)` state를 생성한다.'''
        self.state: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self.closed: bool = False

    def get_observations(
        self,
    ) -> Mapping[str, object]:
        '''Arm shape `(6,)`과 gripper shape `(1,)` observation을 반환한다.'''
        return {
            'joint_pos': self.state[:6],
            'gripper_pos': self.state[6:],
        }

    def command_joint_pos(
        self,
        joint_pos: object,
    ) -> None:
        '''Shape `(7,)` tuple command를 state로 저장한다.'''
        self.state = tuple(float(value) for value in joint_pos)  # type: ignore[union-attr]

    def close(
        self,
    ) -> None:
        '''Close 상태를 기록한다.'''
        self.closed = True


class FakeVisualizer:
    '''GUI 없이 visualizer lifecycle state를 기록한다.'''

    def __init__(
        self,
    ) -> None:
        '''Sync state와 close 상태를 초기화한다.'''
        self.states: list[tuple[float, ...]] = []
        self.closed: bool = False

    def connect(
        self,
        initial_state: tuple[float, ...],
    ) -> None:
        '''Initial state를 기록한다.'''
        self.states.append(initial_state)

    def sync(
        self,
        state: tuple[float, ...],
    ) -> None:
        '''Command 후 state를 기록한다.'''
        self.states.append(state)

    def close(
        self,
    ) -> None:
        '''Close 상태를 기록한다.'''
        self.closed = True


def _load_fake_robot(
    robot_config: RobotConfig,
    execution_target: ExecutionTarget,
) -> I2RTRobot:
    '''Configuration을 검증하고 fake I2RT robot을 반환한다.'''
    del robot_config
    if execution_target != 'mujoco':
        raise ValueError('test loader supports only MuJoCo')
    return FakeI2RTRobot()


def _identity_vector(
    values: tuple[float, ...],
) -> object:
    '''Shape `(7,)` tuple을 변환 없이 반환한다.'''
    return values


class VisualizerFactory:
    '''동일한 fake visualizer instance를 반환한다.'''

    def __init__(
        self,
        visualizer: FakeVisualizer,
    ) -> None:
        '''반환할 visualizer를 저장한다.'''
        self._visualizer: FakeVisualizer = visualizer

    def __call__(
        self,
        robot: I2RTRobot,
    ) -> RobotVisualizer:
        '''Robot XML path를 확인하고 fake visualizer를 반환한다.'''
        if not robot.xml_path:
            raise ValueError('xml_path is required')
        return self._visualizer


def test_backend_syncs_visualizer_after_command(
) -> None:
    '''Backend connect, execute, close가 visualizer lifecycle과 동기화되는지 검증한다.'''
    visualizer: FakeVisualizer = FakeVisualizer()
    backend: I2RTRobotBackend = I2RTRobotBackend(
        config=RobotConfig(),
        execution_target='mujoco',
        loader=_load_fake_robot,
        vector_converter=_identity_vector,
        visualizer_factory=VisualizerFactory(visualizer),
    )
    action: RobotAction = RobotAction(values=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0))

    backend.connect()
    backend.execute(action)
    backend.close()

    assert visualizer.states == [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        action.values,
    ]
    assert visualizer.closed


def test_quest_stream_skips_passive_viewer(
) -> None:
    '''Quest stream을 사용하면 충돌 가능한 desktop passive viewer를 열지 않는지 검증한다.'''
    viewer: MujocoRobotViewer = MujocoRobotViewer(
        xml_path=str(YAM_XML_PATH),
        stream_frame_path='/tmp/yam-mujoco-frame.jpg',
    )
    initial_state: tuple[float, ...] = (0.0, 0.7854, 1.5708, 0.0, 0.0, 0.0, 1.0)

    with (
        patch.object(viewer, '_connect_stream_renderer') as connect_stream_renderer,
        patch('yam_control.robot.mujoco_viewer.mujoco.viewer.launch_passive') as launch_passive,
    ):
        viewer.connect(initial_state)

    connect_stream_renderer.assert_called_once_with()
    launch_passive.assert_not_called()
