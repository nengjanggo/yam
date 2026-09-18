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


class FakeCamera:
    '''Hardware 없이 camera lifecycle event와 고정 frame을 제공한다.'''

    def __init__(
        self,
        events: list[str],
    ) -> None:
        '''Robot과 공유하는 event list를 저장한다.'''
        self.events: list[str] = events

    def connect(
        self,
    ) -> None:
        '''Camera connect event를 기록한다.'''
        self.events.append('camera.connect')

    def read_frames(
        self,
    ) -> Mapping[str, object]:
        '''Top role frame placeholder를 반환한다.'''
        return {'top': 'frame'}

    def close(
        self,
    ) -> None:
        '''Camera close event를 기록한다.'''
        self.events.append('camera.close')


def test_backend_connects_camera_before_robot_and_adds_frames(
) -> None:
    '''Camera를 robot보다 먼저 연결하고 observation image에 frame을 넣는지 검증한다.'''
    events: list[str] = []

    def load_robot(
        robot_config: RobotConfig,
        execution_target: ExecutionTarget,
    ) -> I2RTRobot:
        '''Robot load event를 기록하고 fake robot을 반환한다.'''
        events.append('robot.load')
        return _load_fake_robot(robot_config, execution_target)

    backend: I2RTRobotBackend = I2RTRobotBackend(
        config=RobotConfig(),
        execution_target='mujoco',
        loader=load_robot,
        vector_converter=_identity_vector,
        camera=FakeCamera(events),
    )

    backend.connect()
    images: Mapping[str, object] = backend.get_observation().images
    backend.close()

    assert events == ['camera.connect', 'robot.load', 'camera.close']
    assert images == {'top': 'frame'}


def test_backend_closes_camera_when_robot_load_fails(
) -> None:
    '''Robot 생성 실패 시 이미 연결한 camera를 해제하는지 검증한다.'''
    events: list[str] = []
    backend: I2RTRobotBackend = I2RTRobotBackend(
        config=RobotConfig(),
        execution_target='real',
        loader=_load_fake_robot,
        vector_converter=_identity_vector,
        camera=FakeCamera(events),
    )

    try:
        backend.connect()
    except ValueError:
        pass
    else:
        raise AssertionError('robot load failure must propagate')

    assert events == ['camera.connect', 'camera.close']


class FakeMotorChain:
    '''I2RT motor chain의 running flag만 흉내 낸다.'''

    def __init__(
        self,
    ) -> None:
        '''Running 상태로 시작한다.'''
        self.running: bool = True


class FakeRealI2RTRobot(FakeI2RTRobot):
    '''Motor chain과 server thread 상태를 가진 real robot을 흉내 낸다.'''

    def __init__(
        self,
    ) -> None:
        '''Running motor chain과 살아 있는 server thread 상태를 생성한다.'''
        super().__init__()
        self.motor_chain: FakeMotorChain = FakeMotorChain()
        self._server_thread: FakeThread = FakeThread()


class FakeThread:
    '''Thread의 is_alive 상태만 흉내 낸다.'''

    def __init__(
        self,
    ) -> None:
        '''살아 있는 상태로 시작한다.'''
        self.alive: bool = True

    def is_alive(
        self,
    ) -> bool:
        '''현재 alive 상태를 반환한다.'''
        return self.alive


def _build_real_backend(
    robot: FakeRealI2RTRobot,
) -> I2RTRobotBackend:
    '''주어진 fake real robot을 반환하는 backend를 생성한다.'''
    return I2RTRobotBackend(
        config=RobotConfig(),
        execution_target='real',
        loader=lambda robot_config, execution_target: robot,
        vector_converter=_identity_vector,
    )


def test_backend_rejects_stopped_motor_chain(
) -> None:
    '''I2RT motor chain loop가 멈추면 observation과 command를 거부하고 close는 허용하는지 검증한다.'''
    robot: FakeRealI2RTRobot = FakeRealI2RTRobot()
    backend: I2RTRobotBackend = _build_real_backend(robot)
    backend.connect()
    backend.get_observation()

    robot.motor_chain.running = False

    for call in (
        backend.get_observation,
        lambda: backend.execute(RobotAction(values=robot.state)),
    ):
        try:
            call()
        except RuntimeError as error:
            assert 'motor chain' in str(error)
        else:
            raise AssertionError('stopped motor chain must be rejected')
    backend.close()
    assert robot.closed


def test_backend_rejects_stopped_server_thread(
) -> None:
    '''I2RT robot server thread가 멈추면 command를 거부하는지 검증한다.'''
    robot: FakeRealI2RTRobot = FakeRealI2RTRobot()
    backend: I2RTRobotBackend = _build_real_backend(robot)
    backend.connect()

    robot._server_thread.alive = False

    try:
        backend.execute(RobotAction(values=robot.state))
    except RuntimeError as error:
        assert 'server thread' in str(error)
    else:
        raise AssertionError('stopped server thread must be rejected')
