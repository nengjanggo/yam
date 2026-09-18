'''Fake component로 episode state와 initial pose 흐름을 검증한다.'''

from __future__ import annotations

import unittest
import time
from collections.abc import Callable, Iterator

from yam_control.config import (
    CameraConfig,
    ExecutionTarget,
    QuestConfig,
    RobotConfig,
    RunConfig,
    build_run_config,
)
from yam_control.session import RunSession
from yam_control.types import EpisodeState, RobotAction, RobotObservation, SafetyDecision


class FakeRobot:
    '''Hardware command 없이 RobotBackend event를 기록한다.'''

    def __init__(
        self,
    ) -> None:
        '''초기 shape `(7,)` state와 빈 event를 생성한다.'''
        self.state: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.events: list[tuple[str, tuple[float, ...] | None]] = []

    def connect(
        self,
    ) -> None:
        '''Fake connect event를 기록한다.'''
        self.events.append(('connect', None))

    def get_observation(
        self,
    ) -> RobotObservation:
        '''현재 fake state를 observation으로 반환한다.'''
        return RobotObservation(state=self.state)

    def execute(
        self,
        action: RobotAction,
    ) -> None:
        '''Fake state를 action으로 갱신하고 event를 기록한다.'''
        self.state = action.values
        self.events.append(('execute', action.values))

    def move_to_pose(
        self,
        pose: tuple[float, ...],
    ) -> None:
        '''Fake state를 pose로 이동하고 event를 기록한다.'''
        self.state = pose
        self.events.append(('move_to_pose', pose))

    def reset_environment(
        self,
        initial_pose: tuple[float, ...],
    ) -> None:
        '''Fake environment와 state를 initial pose로 reset한다.'''
        self.state = initial_pose
        self.events.append(('reset_environment', initial_pose))

    def hold(
        self,
    ) -> None:
        '''현재 fake state hold event를 기록한다.'''
        self.events.append(('hold', self.state))

    def close(
        self,
    ) -> None:
        '''Fake close event를 기록한다.'''
        self.events.append(('close', None))


class FakeActionProducer:
    '''고정 action을 생성하는 hardware-free action source이다.'''

    def __init__(
        self,
        action: RobotAction,
    ) -> None:
        '''반환할 고정 action과 reset count를 저장한다.'''
        self._action: RobotAction = action
        self.reset_count: int = 0

    def connect(
        self,
    ) -> None:
        '''Fake producer는 외부 resource를 연결하지 않는다.'''

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''Episode reset count를 증가시킨다.'''
        del observation
        self.reset_count += 1

    def next_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''고정 shape `(7,)` action을 반환한다.'''
        del observation
        return self._action

    def close(
        self,
    ) -> None:
        '''Fake producer는 해제할 resource가 없다.'''


class FakeSafetyGate:
    '''설정된 accept 값에 따라 action을 허용하거나 거부한다.'''

    def __init__(
        self,
        accepted: bool,
    ) -> None:
        '''Safety decision의 고정 accepted 값을 저장한다.'''
        self._accepted: bool = accepted

    def reset(
        self,
    ) -> None:
        '''Fake gate에는 reset할 state가 없다.'''

    def evaluate(
        self,
        observation: RobotObservation,
        action: RobotAction,
    ) -> SafetyDecision:
        '''고정 accepted 값과 입력 action으로 decision을 반환한다.'''
        return SafetyDecision(
            accepted=self._accepted,
            action=action if self._accepted else RobotAction(values=observation.state),
            reason=None if self._accepted else 'unsafe',
        )


class FakeRecorder:
    '''Disk write 없이 recorder lifecycle count를 기록한다.'''

    def __init__(
        self,
    ) -> None:
        '''모든 recorder lifecycle count를 0으로 초기화한다.'''
        self.start_count: int = 0
        self.record_count: int = 0
        self.finish_count: int = 0
        self.abort_count: int = 0

    def start(
        self,
        task_prompt: str,
    ) -> None:
        '''Start count를 증가시킨다.'''
        del task_prompt
        self.start_count += 1

    def record(
        self,
        observation: RobotObservation,
        action: RobotAction,
    ) -> None:
        '''Record count를 증가시킨다.'''
        del observation
        del action
        self.record_count += 1

    def finish(
        self,
    ) -> None:
        '''Finish count를 증가시킨다.'''
        self.finish_count += 1

    def abort(
        self,
    ) -> None:
        '''Abort count를 증가시킨다.'''
        self.abort_count += 1


class FakeClock:
    '''제어 step 경계에 결정적인 시각을 제공한다.'''

    def __init__(
        self,
        times_s: tuple[float, ...],
    ) -> None:
        '''호출 순서대로 반환할 시각을 저장한다.'''
        self._times_s: Iterator[float] = iter(times_s)

    def __call__(
        self,
    ) -> float:
        '''다음 시각을 반환한다.'''
        return next(self._times_s)


def _no_sleep(
    duration_s: float,
) -> None:
    '''Test control loop에서 실제 sleep을 생략한다.'''
    del duration_s


def _build_config(
    execution_target: ExecutionTarget,
) -> RunConfig:
    '''Session test용 RunConfig를 생성한다.'''
    initial_pose: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0)
    reset_pose: tuple[float, ...] = (-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0)
    return build_run_config(
        mode='teleop',
        teleop_source='leader',
        save_teleop_data=False,
        vla_type='pi0',
        checkpoint_uri='',
        checkpoint_revision=None,
        policy_config_name='',
        use_rtc=False,
        execution_target=execution_target,
        use_safety_gate=False,
        control_hz=30.0,
        task_prompt='pick up the object',
        data_root='data/episodes',
        robot=RobotConfig(
            episode_initial_pose=initial_pose,
            human_reset_pose=reset_pose,
        ),
        quest=QuestConfig(),
        camera=CameraConfig(),
    )


def _build_session(
    config: RunConfig,
    robot: FakeRobot,
    safety_gate: FakeSafetyGate,
    recorder: FakeRecorder,
    clock: Callable[[], float] = time.perf_counter,
) -> RunSession:
    '''주어진 fake component로 hardware-free RunSession을 생성한다.'''
    action: RobotAction = RobotAction(values=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0))
    producer: FakeActionProducer = FakeActionProducer(action=action)
    return RunSession(
        config=config,
        robot=robot,
        action_producer=producer,
        safety_gate=safety_gate,
        recorder=recorder,
        sleeper=_no_sleep,
        clock=clock,
    )


class SessionTest(unittest.TestCase):
    '''Initial pose와 episode lifecycle을 fake component로 검증한다.'''

    def test_real_episode_waits_at_reset_pose_then_moves_to_initial_pose(
        self,
    ) -> None:
        '''Real flow가 human reset pose 이후 initial pose에서 episode를 시작하는지 검증한다.'''
        config: RunConfig = _build_config(execution_target='real')
        robot: FakeRobot = FakeRobot()
        recorder: FakeRecorder = FakeRecorder()
        session: RunSession = _build_session(config, robot, FakeSafetyGate(True), recorder)

        session.connect()
        session.prepare_episode()
        prepared_event: tuple[str, tuple[float, ...] | None] = robot.events[1]
        state: EpisodeState = session.run_prepared_episode(max_steps=2)

        self.assertEqual(prepared_event, ('move_to_pose', config.common.robot.human_reset_pose))
        self.assertIn(('move_to_pose', config.common.robot.episode_initial_pose), robot.events)
        self.assertEqual(state, EpisodeState.FINISHED)
        self.assertEqual(recorder.record_count, 2)

    def test_mujoco_episodes_reset_automatically(
        self,
    ) -> None:
        '''MuJoCo 연속 episode가 episode마다 environment를 reset하는지 검증한다.'''
        config: RunConfig = _build_config(execution_target='mujoco')
        robot: FakeRobot = FakeRobot()
        recorder: FakeRecorder = FakeRecorder()
        session: RunSession = _build_session(config, robot, FakeSafetyGate(True), recorder)

        session.connect()
        states: tuple[EpisodeState, ...] = session.run_simulation_episodes(
            episode_count=3,
            max_steps=1,
        )
        reset_count: int = sum(event[0] == 'reset_environment' for event in robot.events)

        self.assertEqual(states, (EpisodeState.FINISHED,) * 3)
        self.assertEqual(reset_count, 3)
        self.assertEqual(recorder.finish_count, 3)
        self.assertEqual(session.action_producer_diagnostics(), {})

    def test_unsafe_action_aborts_without_execution(
        self,
    ) -> None:
        '''SafetyGate 거부 시 candidate action을 실행하지 않고 episode를 abort하는지 검증한다.'''
        config: RunConfig = _build_config(execution_target='mujoco')
        robot: FakeRobot = FakeRobot()
        recorder: FakeRecorder = FakeRecorder()
        session: RunSession = _build_session(config, robot, FakeSafetyGate(False), recorder)

        session.connect()
        session.prepare_episode()
        state: EpisodeState = session.run_prepared_episode(max_steps=1)
        execute_count: int = sum(event[0] == 'execute' for event in robot.events)

        self.assertEqual(state, EpisodeState.ABORTED)
        self.assertEqual(execute_count, 0)
        self.assertEqual(recorder.abort_count, 1)
        self.assertEqual(session.control_loop_diagnostics()['completed_step_count'], 0)
        self.assertIsNone(session.control_loop_diagnostics()['actual_control_hz'])

    def test_control_loop_diagnostics_exclude_setup_and_reset_per_episode(
        self,
    ) -> None:
        '''완료된 step의 처리·sleep 포함 주기만 집계하고 다음 episode에서 초기화한다.'''
        config: RunConfig = _build_config(execution_target='mujoco')
        robot: FakeRobot = FakeRobot()
        recorder: FakeRecorder = FakeRecorder()
        clock: FakeClock = FakeClock((0.0, 0.01, 0.04, 0.04, 0.07, 0.10, 1.0, 1.005, 1.025))
        session: RunSession = _build_session(config, robot, FakeSafetyGate(True), recorder, clock)

        session.connect()
        session.prepare_episode()
        session.run_prepared_episode(max_steps=2)
        first: dict[str, int | float | None] = session.control_loop_diagnostics()

        self.assertEqual(first['completed_step_count'], 2)
        self.assertAlmostEqual(first['mean_processing_ms'], 20.0)
        self.assertAlmostEqual(first['max_processing_ms'], 30.0)
        self.assertAlmostEqual(first['mean_tick_period_ms'], 50.0)
        self.assertAlmostEqual(first['max_tick_period_ms'], 60.0)
        self.assertAlmostEqual(first['actual_control_hz'], 20.0)

        session.prepare_episode()
        session.run_prepared_episode(max_steps=1)
        second: dict[str, int | float | None] = session.control_loop_diagnostics()

        self.assertEqual(second['completed_step_count'], 1)
        self.assertAlmostEqual(second['mean_processing_ms'], 5.0)
        self.assertAlmostEqual(second['mean_tick_period_ms'], 25.0)
        self.assertAlmostEqual(second['actual_control_hz'], 40.0)


if __name__ == '__main__':
    unittest.main()
