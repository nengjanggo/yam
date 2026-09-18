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
from yam_control.types import EpisodeOutcome, EpisodeState, RobotAction, RobotObservation, SafetyDecision


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


class HoldingActionProducer(FakeActionProducer):
    '''정해진 step에서 clutch release hold 사유를 노출하는 action source이다.'''

    def __init__(
        self,
        action: RobotAction,
        holding_steps: tuple[bool, ...],
    ) -> None:
        '''Step별 hold 여부와 고정 action을 저장한다.'''
        super().__init__(action=action)
        self._holding_steps: Iterator[bool] = iter(holding_steps)
        self.last_hold_reason: str | None = None

    def next_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''Hold step이면 현재 state를, 아니면 고정 action을 반환한다.'''
        if next(self._holding_steps):
            self.last_hold_reason = 'clutch released'
            return RobotAction(values=observation.state)
        self.last_hold_reason = None
        return super().next_action(observation)


class OutcomeActionProducer(FakeActionProducer):
    '''지정한 step에서 작업자의 episode 종료 입력을 노출하는 action source이다.'''

    def __init__(
        self,
        action: RobotAction,
        outcome: EpisodeOutcome,
        outcome_at_step: int,
    ) -> None:
        '''종료 입력 종류와 입력이 발생할 step index를 저장한다.'''
        super().__init__(action=action)
        self._outcome: EpisodeOutcome = outcome
        self._outcome_at_step: int = outcome_at_step
        self._step_index: int = 0
        self.episode_outcome: EpisodeOutcome | None = None

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''Episode 종료 입력과 step index를 초기화한다.'''
        super().reset(observation)
        self._step_index = 0
        self.episode_outcome = None

    def next_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''지정한 step부터 episode 종료 입력을 기록한다.'''
        if self._step_index == self._outcome_at_step:
            self.episode_outcome = self._outcome
        self._step_index += 1
        return super().next_action(observation)


class FailingActionProducer(FakeActionProducer):
    '''지정한 step에서 예외를 발생시키는 action source이다.'''

    def __init__(
        self,
        action: RobotAction,
        fail_at_step: int,
    ) -> None:
        '''예외를 발생시킬 step index를 저장한다.'''
        super().__init__(action=action)
        self._fail_at_step: int = fail_at_step
        self._step_index: int = 0

    def next_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''지정한 step에서 RuntimeError를 발생시킨다.'''
        step_index: int = self._step_index
        self._step_index += 1
        if step_index == self._fail_at_step:
            raise RuntimeError('controller disconnected')
        return super().next_action(observation)


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


FIXED_ACTION: RobotAction = RobotAction(values=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0))


def _build_session(
    config: RunConfig,
    robot: FakeRobot,
    safety_gate: FakeSafetyGate,
    recorder: FakeRecorder,
    clock: Callable[[], float] = time.perf_counter,
    producer: FakeActionProducer | None = None,
) -> RunSession:
    '''주어진 fake component로 hardware-free RunSession을 생성한다.'''
    if producer is None:
        producer = FakeActionProducer(action=FIXED_ACTION)
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


class SessionRecordingTest(unittest.TestCase):
    '''Hold step recording 제외와 예외 시 recorder 정리를 검증한다.'''

    def test_holding_steps_are_executed_but_not_recorded(
        self,
    ) -> None:
        '''Clutch release hold step은 robot에 실행하되 recording에서 제외하는지 검증한다.'''
        config: RunConfig = _build_config(execution_target='mujoco')
        robot: FakeRobot = FakeRobot()
        recorder: FakeRecorder = FakeRecorder()
        producer: HoldingActionProducer = HoldingActionProducer(
            action=FIXED_ACTION,
            holding_steps=(True, False, False, True, False),
        )
        session: RunSession = _build_session(
            config,
            robot,
            FakeSafetyGate(True),
            recorder,
            producer=producer,
        )

        session.connect()
        session.prepare_episode()
        state: EpisodeState = session.run_prepared_episode(max_steps=5)
        diagnostics: dict[str, int | float | None] = session.control_loop_diagnostics()
        execute_count: int = sum(event[0] == 'execute' for event in robot.events)

        self.assertEqual(state, EpisodeState.FINISHED)
        self.assertEqual(recorder.record_count, 3)
        self.assertEqual(recorder.finish_count, 1)
        self.assertEqual(diagnostics['completed_step_count'], 5)
        self.assertEqual(diagnostics['recorded_step_count'], 3)
        self.assertGreaterEqual(execute_count, 5)

    def test_exception_aborts_recording_and_allows_next_episode(
        self,
    ) -> None:
        '''Control loop 예외 시 recorder를 abort하고 예외를 다시 발생시키는지 검증한다.'''
        config: RunConfig = _build_config(execution_target='mujoco')
        robot: FakeRobot = FakeRobot()
        recorder: FakeRecorder = FakeRecorder()
        producer: FailingActionProducer = FailingActionProducer(action=FIXED_ACTION, fail_at_step=1)
        session: RunSession = _build_session(
            config,
            robot,
            FakeSafetyGate(True),
            recorder,
            producer=producer,
        )

        session.connect()
        session.prepare_episode()
        with self.assertRaisesRegex(RuntimeError, 'controller disconnected'):
            session.run_prepared_episode(max_steps=3)

        self.assertEqual(session.state, EpisodeState.ABORTED)
        self.assertEqual(recorder.record_count, 1)
        self.assertEqual(recorder.abort_count, 1)
        self.assertEqual(recorder.finish_count, 0)

        session.prepare_episode()
        state: EpisodeState = session.run_prepared_episode(max_steps=1)

        self.assertEqual(state, EpisodeState.FINISHED)
        self.assertEqual(recorder.start_count, 2)

    def test_success_button_finishes_and_saves_early(
        self,
    ) -> None:
        '''성공 입력 step은 실행하지 않고 그 전까지를 저장하며 SUCCEEDED로 끝나는지 검증한다.'''
        config: RunConfig = _build_config(execution_target='mujoco')
        robot: FakeRobot = FakeRobot()
        recorder: FakeRecorder = FakeRecorder()
        producer: OutcomeActionProducer = OutcomeActionProducer(
            action=FIXED_ACTION,
            outcome=EpisodeOutcome.SUCCESS,
            outcome_at_step=2,
        )
        session: RunSession = _build_session(config, robot, FakeSafetyGate(True), recorder, producer=producer)

        session.connect()
        session.prepare_episode()
        execute_count_before: int = sum(event[0] == 'execute' for event in robot.events)
        state: EpisodeState = session.run_prepared_episode(max_steps=10)
        episode_events: list[tuple[str, tuple[float, ...] | None]] = robot.events[len(robot.events) - 3:]

        self.assertEqual(state, EpisodeState.SUCCEEDED)
        self.assertEqual(recorder.record_count, 2)
        self.assertEqual(recorder.finish_count, 1)
        self.assertEqual(recorder.abort_count, 0)
        self.assertEqual(sum(event[0] == 'execute' for event in robot.events) - execute_count_before, 2)
        self.assertEqual(episode_events[-1][0], 'hold')
        self.assertEqual(session.control_loop_diagnostics()['completed_step_count'], 2)

    def test_failure_button_discards_episode(
        self,
    ) -> None:
        '''실패 입력 시 현재 step을 실행하지 않고 저장 없이 DISCARDED로 끝나는지 검증한다.'''
        config: RunConfig = _build_config(execution_target='mujoco')
        robot: FakeRobot = FakeRobot()
        recorder: FakeRecorder = FakeRecorder()
        producer: OutcomeActionProducer = OutcomeActionProducer(
            action=FIXED_ACTION,
            outcome=EpisodeOutcome.FAILURE,
            outcome_at_step=3,
        )
        session: RunSession = _build_session(config, robot, FakeSafetyGate(True), recorder, producer=producer)

        session.connect()
        session.prepare_episode()
        state: EpisodeState = session.run_prepared_episode(max_steps=10)

        self.assertEqual(state, EpisodeState.DISCARDED)
        self.assertEqual(recorder.record_count, 3)
        self.assertEqual(recorder.abort_count, 1)
        self.assertEqual(recorder.finish_count, 0)
        self.assertEqual(robot.events[-1][0], 'hold')

        session.prepare_episode()
        next_state: EpisodeState = session.run_prepared_episode(max_steps=2)

        self.assertEqual(next_state, EpisodeState.FINISHED)
        self.assertEqual(recorder.start_count, 2)

    def test_sleep_subtracts_processing_time_to_keep_control_hz(
        self,
    ) -> None:
        '''처리 시간을 뺀 나머지만 sleep하고 한 주기 이상 밀리면 기준 시각을 재설정하는지 검증한다.'''
        config: RunConfig = _build_config(execution_target='mujoco')
        period_s: float = 1.0 / config.common.control_hz
        # Step마다 (step 시작, 처리 종료, sleep 종료) 순서로 clock을 읽음
        clock: FakeClock = FakeClock((
            0.0, 0.005, period_s,
            period_s, period_s + 0.010, 2.0 * period_s,
            2.0 * period_s, 2.0 * period_s + 0.100, 2.0 * period_s + 0.100,
            2.0 * period_s + 0.100, 2.0 * period_s + 0.110, 3.0 * period_s + 0.100,
        ))
        sleeps: list[float] = []
        session: RunSession = RunSession(
            config=config,
            robot=FakeRobot(),
            action_producer=FakeActionProducer(action=FIXED_ACTION),
            safety_gate=FakeSafetyGate(True),
            recorder=FakeRecorder(),
            sleeper=sleeps.append,
            clock=clock,
        )

        session.connect()
        session.prepare_episode()
        session.run_prepared_episode(max_steps=4)

        # 세 번째 step은 100 ms 처리로 deadline을 넘겨 sleep하지 않고, 네 번째 step은 새 기준으로 sleep
        self.assertEqual(len(sleeps), 3)
        self.assertAlmostEqual(sleeps[0], period_s - 0.005)
        self.assertAlmostEqual(sleeps[1], period_s - 0.010)
        self.assertAlmostEqual(sleeps[2], period_s - 0.010)

    def test_keyboard_interrupt_aborts_recording(
        self,
    ) -> None:
        '''Notebook interrupt도 recorder를 abort하는지 검증한다.'''
        config: RunConfig = _build_config(execution_target='mujoco')
        robot: FakeRobot = FakeRobot()
        recorder: FakeRecorder = FakeRecorder()

        def interrupting_sleep(
            duration_s: float,
        ) -> None:
            '''첫 sleep에서 KeyboardInterrupt를 발생시킨다.'''
            del duration_s
            raise KeyboardInterrupt

        session: RunSession = RunSession(
            config=config,
            robot=robot,
            action_producer=FakeActionProducer(action=FIXED_ACTION),
            safety_gate=FakeSafetyGate(True),
            recorder=recorder,
            sleeper=interrupting_sleep,
        )

        session.connect()
        session.prepare_episode()
        with self.assertRaises(KeyboardInterrupt):
            session.run_prepared_episode(max_steps=3)

        self.assertEqual(session.state, EpisodeState.ABORTED)
        self.assertEqual(recorder.abort_count, 1)


if __name__ == '__main__':
    unittest.main()
