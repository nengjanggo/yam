'''Episode initial pose와 실행 lifecycle을 관리한다.'''

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from .config import RunConfig
from .interfaces import ActionProducer, EpisodeRecorder, RobotBackend, SafetyGate
from .types import EpisodeState, RobotAction, RobotObservation, SafetyDecision

Sleeper = Callable[[float], None]
Clock = Callable[[], float]


class RunSession:
    '''RobotBackend와 action source를 mode에 무관한 episode loop로 조합한다.'''

    def __init__(
        self,
        config: RunConfig,
        robot: RobotBackend,
        action_producer: ActionProducer,
        safety_gate: SafetyGate,
        recorder: EpisodeRecorder,
        sleeper: Sleeper = time.sleep,
        clock: Clock = time.perf_counter,
    ) -> None:
        '''Configuration과 독립 component를 저장하고 IDLE state로 시작한다.'''
        self.config: RunConfig = config
        self._robot: RobotBackend = robot
        self._action_producer: ActionProducer = action_producer
        self._safety_gate: SafetyGate = safety_gate
        self._recorder: EpisodeRecorder = recorder
        self._sleeper: Sleeper = sleeper
        self._clock: Clock = clock
        self.state: EpisodeState = EpisodeState.IDLE
        self._connected: bool = False
        self._stop_requested: bool = False
        self._completed_step_count: int = 0
        self._total_processing_s: float = 0.0
        self._max_processing_s: float = 0.0
        self._total_tick_period_s: float = 0.0
        self._max_tick_period_s: float = 0.0

    def connect(
        self,
    ) -> None:
        '''RobotBackend와 action source resource를 연결한다.'''
        self._robot.connect()
        self._action_producer.connect()
        self._connected = True

    def prepare_episode(
        self,
    ) -> None:
        '''Real robot은 human reset pose로, MuJoCo는 자동 initial state로 준비한다.'''
        if not self._connected:
            raise RuntimeError('connect must run before prepare_episode')
        initial_pose: tuple[float, ...] = self.config.common.robot.episode_initial_pose
        if self.config.common.execution_target == 'real':
            human_reset_pose: tuple[float, ...] = self.config.common.robot.resolved_human_reset_pose()
            self._robot.move_to_pose(human_reset_pose)
            self._robot.hold()
        else:
            self._robot.reset_environment(initial_pose)
        self._stop_requested = False
        self.state = EpisodeState.PREPARED

    def request_stop(
        self,
    ) -> None:
        '''현재 episode를 다음 control boundary에서 정상 종료하도록 요청한다.'''
        self._stop_requested = True

    def action_producer_diagnostics(
        self,
    ) -> dict[str, object]:
        '''ActionProducer가 제공하는 optional episode 진단값을 반환한다.'''
        diagnostics_method: object = getattr(self._action_producer, 'diagnostics', None)
        if not callable(diagnostics_method):
            return {}
        diagnostics: object = diagnostics_method()
        if not isinstance(diagnostics, Mapping):
            raise TypeError('ActionProducer diagnostics must be a mapping')
        return dict(diagnostics)

    def control_loop_diagnostics(
        self,
    ) -> dict[str, int | float | None]:
        '''완료된 step의 처리 시간, 실제 tick 주기와 제어 주파수를 반환한다.'''
        completed_step_count: int = self._completed_step_count
        if completed_step_count == 0:
            return {
                'completed_step_count': 0,
                'mean_processing_ms': None,
                'max_processing_ms': None,
                'mean_tick_period_ms': None,
                'max_tick_period_ms': None,
                'actual_control_hz': None,
            }
        return {
            'completed_step_count': completed_step_count,
            'mean_processing_ms': self._total_processing_s * 1000.0 / completed_step_count,
            'max_processing_ms': self._max_processing_s * 1000.0,
            'mean_tick_period_ms': self._total_tick_period_s * 1000.0 / completed_step_count,
            'max_tick_period_ms': self._max_tick_period_s * 1000.0,
            'actual_control_hz': completed_step_count / self._total_tick_period_s
            if self._total_tick_period_s > 0.0 else None,
        }

    def run_prepared_episode(
        self,
        max_steps: int,
    ) -> EpisodeState:
        '''Prepared episode를 최대 max_steps 동안 실행하고 종료 state를 반환한다.'''
        if self.state != EpisodeState.PREPARED:
            raise RuntimeError('prepare_episode must run before run_prepared_episode')
        if max_steps <= 0:
            raise ValueError('max_steps must be positive')
        # 새 episode의 완료된 control step만 집계한다.
        self._completed_step_count = 0
        self._total_processing_s = 0.0
        self._max_processing_s = 0.0
        self._total_tick_period_s = 0.0
        self._max_tick_period_s = 0.0
        if self.config.common.execution_target == 'real':
            initial_pose: tuple[float, ...] = self.config.common.robot.episode_initial_pose
            self._robot.move_to_pose(initial_pose)
        observation: RobotObservation = self._robot.get_observation()
        self._action_producer.reset(observation)
        self._safety_gate.reset()
        self._recorder.start(self.config.task_prompt)
        self.state = EpisodeState.RUNNING
        control_period_s: float = 1.0 / self.config.common.control_hz
        step_index: int
        for step_index in range(max_steps):
            del step_index
            if self._stop_requested:
                break
            step_start_s: float = self._clock()
            observation = self._robot.get_observation()
            candidate_action: RobotAction = self._action_producer.next_action(observation)
            decision: SafetyDecision = self._safety_gate.evaluate(observation, candidate_action)
            if not decision.accepted:
                self._robot.hold()
                self._recorder.abort()
                self.state = EpisodeState.ABORTED
                return self.state
            self._robot.execute(decision.action)
            self._recorder.record(observation, decision.action)
            processing_s: float = self._clock() - step_start_s
            self._sleeper(control_period_s)
            tick_period_s: float = self._clock() - step_start_s
            self._completed_step_count += 1
            self._total_processing_s += processing_s
            self._max_processing_s = max(self._max_processing_s, processing_s)
            self._total_tick_period_s += tick_period_s
            self._max_tick_period_s = max(self._max_tick_period_s, tick_period_s)
        self._robot.hold()
        self._recorder.finish()
        self.state = EpisodeState.FINISHED
        return self.state

    def run_simulation_episodes(
        self,
        episode_count: int,
        max_steps: int,
    ) -> tuple[EpisodeState, ...]:
        '''MuJoCo episode를 자동 reset하면서 연속 실행하고 각 종료 state를 반환한다.'''
        if self.config.common.execution_target != 'mujoco':
            raise RuntimeError('run_simulation_episodes is available only for MuJoCo')
        if episode_count <= 0:
            raise ValueError('episode_count must be positive')
        states: list[EpisodeState] = []
        episode_index: int
        for episode_index in range(episode_count):
            if episode_index > 0 or self.state != EpisodeState.PREPARED:
                self.prepare_episode()
            episode_state: EpisodeState = self.run_prepared_episode(max_steps)
            states.append(episode_state)
        return tuple(states)

    def close(
        self,
    ) -> None:
        '''Action source와 RobotBackend resource를 순서대로 해제한다.'''
        self._action_producer.close()
        self._robot.close()
        self._connected = False
        self.state = EpisodeState.IDLE
