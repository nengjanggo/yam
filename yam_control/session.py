'''Episode initial pose와 실행 lifecycle을 관리한다.'''

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from .config import RunConfig
from .interfaces import ActionProducer, EpisodeOperator, EpisodeRecorder, RobotBackend, SafetyGate
from .types import (
    EpisodeOutcome,
    EpisodePhase,
    EpisodeState,
    EpisodeStatus,
    OperatorButton,
    RobotAction,
    RobotObservation,
    SafetyDecision,
)

Sleeper = Callable[[float], None]
Clock = Callable[[], float]

# 작업자 화면의 경과 시간 표시 갱신 주파수
STATUS_PUBLISH_HZ: float = 5.0


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
        self._recorded_step_count: int = 0
        self._total_processing_s: float = 0.0
        self._max_processing_s: float = 0.0
        self._total_tick_period_s: float = 0.0
        self._max_tick_period_s: float = 0.0
        self._last_episode_state: EpisodeState | None = None
        self._last_episode_saved: bool | None = None
        self._saved_episode_count: int | None = None

    def connect(
        self,
    ) -> None:
        '''RobotBackend와 action source resource를 연결한다.'''
        self._robot.connect()
        self._action_producer.connect()
        self._connected = True

    def get_observation(
        self,
    ) -> RobotObservation:
        '''연결된 RobotBackend의 최신 measured state와 camera frame을 반환한다.'''
        if not self._connected:
            raise RuntimeError('connect must run before get_observation')
        return self._robot.get_observation()

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

    def _action_producer_is_holding(
        self,
    ) -> bool:
        '''ActionProducer가 optional hold 사유를 제공하고 현재 step에서 hold 중이면 True를 반환한다.'''
        hold_reason: object = getattr(self._action_producer, 'last_hold_reason', None)
        return hold_reason is not None

    def _action_producer_episode_outcome(
        self,
    ) -> EpisodeOutcome | None:
        '''ActionProducer가 optional로 제공하는 작업자의 episode 조기 종료 입력을 반환한다.'''
        episode_outcome: object = getattr(self._action_producer, 'episode_outcome', None)
        if episode_outcome is None:
            return None
        if not isinstance(episode_outcome, EpisodeOutcome):
            raise TypeError('ActionProducer episode_outcome must be an EpisodeOutcome')
        return episode_outcome

    def _episode_operator(
        self,
    ) -> EpisodeOperator | None:
        '''ActionProducer가 작업자 button 입력과 상태 표시를 제공하면 반환한다.'''
        if isinstance(self._action_producer, EpisodeOperator):
            return self._action_producer
        return None

    def _refresh_saved_episode_count(
        self,
    ) -> int | None:
        '''Recorder가 optional로 제공하는 저장 완료 episode 수를 갱신하여 반환한다.'''
        count_method: object = getattr(self._recorder, 'saved_episode_count', None)
        if not callable(count_method):
            self._saved_episode_count = None
            return None
        saved_episode_count: object = count_method()
        if not isinstance(saved_episode_count, int):
            raise TypeError('EpisodeRecorder saved_episode_count must return an int')
        self._saved_episode_count = saved_episode_count
        return saved_episode_count

    def _publish_status(
        self,
        phase: EpisodePhase,
        elapsed_s: float | None = None,
        limit_s: float | None = None,
    ) -> None:
        '''작업자 화면이 있으면 현재 단계와 직전 episode 결과를 표시한다.'''
        operator: EpisodeOperator | None = self._episode_operator()
        if operator is None:
            return
        operator.publish_episode_status(
            EpisodeStatus(
                phase=phase,
                elapsed_s=elapsed_s,
                limit_s=limit_s,
                last_state=self._last_episode_state,
                last_episode_saved=self._last_episode_saved,
                saved_episode_count=self._saved_episode_count,
            )
        )

    def _end_episode(
        self,
        state: EpisodeState,
        saved_episode_count_before: int | None,
    ) -> EpisodeState:
        '''종료 state와 실제 저장 여부를 기록하고 작업자 화면에 결과를 표시한다.'''
        self.state = state
        saved_episode_count: int | None = self._refresh_saved_episode_count()
        # 성공 입력이어도 기록된 step이 없으면 recorder가 폴더를 지우므로 저장 수 증가로 판단
        self._last_episode_state = state
        self._last_episode_saved = (
            saved_episode_count is not None
            and saved_episode_count_before is not None
            and saved_episode_count > saved_episode_count_before
        )
        self._publish_status(EpisodePhase.ENDED)
        return state

    def _wait_for_button(
        self,
        operator: EpisodeOperator,
        phase: EpisodePhase,
        accepted_buttons: frozenset[OperatorButton],
        elapsed_s: float | None = None,
        limit_s: float | None = None,
    ) -> OperatorButton:
        '''Robot을 멈춘 채 상태를 표시하며 accepted button이 새로 눌릴 때까지 기다린다.'''
        control_period_s: float = 1.0 / self.config.common.control_hz
        status_interval_ticks: int = max(1, round(self.config.common.control_hz / STATUS_PUBLISH_HZ))
        tick_index: int = 0
        while True:
            # 대기 중에도 wrist camera stream이 멈추지 않도록 observation을 계속 읽음
            self._robot.get_observation()
            if tick_index % status_interval_ticks == 0:
                self._publish_status(phase, elapsed_s, limit_s)
            button: OperatorButton | None = operator.poll_button()
            if button is not None and button in accepted_buttons:
                return button
            self._sleeper(control_period_s)
            tick_index += 1

    def control_loop_diagnostics(
        self,
    ) -> dict[str, int | float | None]:
        '''완료된 step의 처리 시간, 실제 tick 주기와 제어 주파수를 반환한다.'''
        completed_step_count: int = self._completed_step_count
        if completed_step_count == 0:
            return {
                'completed_step_count': 0,
                'recorded_step_count': 0,
                'mean_processing_ms': None,
                'max_processing_ms': None,
                'mean_tick_period_ms': None,
                'max_tick_period_ms': None,
                'actual_control_hz': None,
            }
        return {
            'completed_step_count': completed_step_count,
            'recorded_step_count': self._recorded_step_count,
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
        self._recorded_step_count = 0
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
        saved_episode_count_before: int | None = self._refresh_saved_episode_count()
        self._recorder.start(self.config.task_prompt)
        self.state = EpisodeState.RUNNING
        self._last_episode_state = None
        self._last_episode_saved = None
        operator: EpisodeOperator | None = self._episode_operator()
        control_hz: float = self.config.common.control_hz
        control_period_s: float = 1.0 / control_hz
        limit_s: float = max_steps / control_hz
        status_interval_steps: int = max(1, round(control_hz / STATUS_PUBLISH_HZ))
        step_index: int
        next_tick_s: float | None = None
        succeeded: bool = False
        try:
            for step_index in range(max_steps):
                if self._stop_requested:
                    break
                if step_index % status_interval_steps == 0:
                    self._publish_status(EpisodePhase.RUNNING, step_index / control_hz, limit_s)
                step_start_s: float = self._clock()
                if next_tick_s is None:
                    next_tick_s = step_start_s + control_period_s
                observation = self._robot.get_observation()
                candidate_action: RobotAction = self._action_producer.next_action(observation)
                episode_outcome: EpisodeOutcome | None = self._action_producer_episode_outcome()
                if episode_outcome is EpisodeOutcome.FAILURE:
                    # 작업자가 실패로 표시한 episode는 현재 step을 실행하지 않고 저장 없이 종료
                    self._robot.hold()
                    self._recorder.abort()
                    return self._end_episode(EpisodeState.DISCARDED, saved_episode_count_before)
                if episode_outcome is EpisodeOutcome.SUCCESS:
                    # 작업자가 성공으로 표시한 episode는 현재 step을 실행하지 않고 저장하며 종료
                    succeeded = True
                    break
                decision: SafetyDecision = self._safety_gate.evaluate(observation, candidate_action)
                if not decision.accepted:
                    self._robot.hold()
                    self._recorder.abort()
                    return self._end_episode(EpisodeState.ABORTED, saved_episode_count_before)
                self._robot.execute(decision.action)
                # Clutch release 등으로 action source가 hold 중인 step은 저장하지 않음
                if not self._action_producer_is_holding():
                    self._recorder.record(observation, decision.action)
                    self._recorded_step_count += 1
                processing_end_s: float = self._clock()
                processing_s: float = processing_end_s - step_start_s
                # 처리 시간을 뺀 다음 tick 시각까지만 sleep해 control_hz와 video fps를 일치시킴
                sleep_s: float = next_tick_s - processing_end_s
                if sleep_s > 0.0:
                    self._sleeper(sleep_s)
                tick_end_s: float = self._clock()
                tick_period_s: float = tick_end_s - step_start_s
                next_tick_s += control_period_s
                # 한 주기 이상 밀리면 밀린 tick을 몰아서 실행하지 않도록 기준 시각을 재설정
                if tick_end_s > next_tick_s:
                    next_tick_s = tick_end_s + control_period_s
                self._completed_step_count += 1
                self._total_processing_s += processing_s
                self._max_processing_s = max(self._max_processing_s, processing_s)
                self._total_tick_period_s += tick_period_s
                self._max_tick_period_s = max(self._max_tick_period_s, tick_period_s)
            else:
                if operator is not None:
                    # 제한 시간에 도달하면 robot을 멈추고 작업자가 A(성공) 또는 B(실패)를 고를 때까지 대기
                    self._robot.hold()
                    outcome_button: OperatorButton = self._wait_for_button(
                        operator,
                        EpisodePhase.AWAITING_OUTCOME,
                        frozenset({OperatorButton.PRIMARY, OperatorButton.SECONDARY}),
                        elapsed_s=limit_s,
                        limit_s=limit_s,
                    )
                    if outcome_button is OperatorButton.SECONDARY:
                        self._recorder.abort()
                        return self._end_episode(EpisodeState.DISCARDED, saved_episode_count_before)
                    succeeded = True
        except BaseException:
            # 예외나 KeyboardInterrupt로 중단된 episode는 완료 flag 없이 폐기하고 다음 start를 허용
            self._recorder.abort()
            self._end_episode(EpisodeState.ABORTED, saved_episode_count_before)
            raise
        self._robot.hold()
        # Video encoding 동안 이전 안내를 보고 button을 다시 누르지 않도록 저장 중임을 표시
        self._publish_status(EpisodePhase.SAVING)
        self._recorder.finish()
        return self._end_episode(
            EpisodeState.SUCCEEDED if succeeded else EpisodeState.FINISHED,
            saved_episode_count_before,
        )

    def run_operator_episodes(
        self,
        max_steps: int,
    ) -> tuple[EpisodeState, ...]:
        '''작업자 A button으로 episode 초기화와 시작을 반복하고 notebook interrupt 시 종료 state들을 반환한다.'''
        operator: EpisodeOperator | None = self._episode_operator()
        if operator is None:
            raise RuntimeError('run_operator_episodes requires an ActionProducer with operator button input')
        if max_steps <= 0:
            raise ValueError('max_steps must be positive')
        states: list[EpisodeState] = []
        start_button: frozenset[OperatorButton] = frozenset({OperatorButton.PRIMARY})
        self._refresh_saved_episode_count()
        try:
            while True:
                if self.state != EpisodeState.PREPARED:
                    self._wait_for_button(operator, EpisodePhase.WAITING_PREPARE, start_button)
                    self._publish_status(EpisodePhase.PREPARING)
                    self.prepare_episode()
                    self._wait_for_button(operator, EpisodePhase.WAITING_START, start_button)
                    self._publish_status(EpisodePhase.STARTING)
                try:
                    states.append(self.run_prepared_episode(max_steps))
                except KeyboardInterrupt:
                    # Episode 도중 interrupt는 저장 없이 ABORTED로 끝난 episode로 기록
                    states.append(self.state)
                    raise
        except KeyboardInterrupt:
            # Notebook 정지 버튼을 연속 실행의 정상 종료로 사용
            pass
        finally:
            self._publish_status(EpisodePhase.IDLE)
        return tuple(states)

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
