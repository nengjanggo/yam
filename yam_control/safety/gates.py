'''SafetyGate의 pass-through와 swept-path 구현을 제공한다.'''

from __future__ import annotations

from typing import Protocol

from ..types import RobotAction, RobotObservation, SafetyDecision


class ConfigurationValidator(Protocol):
    '''Shape `(S,)` robot configuration의 collision과 limit을 검사한다.'''

    def is_safe(
        self,
        configuration: tuple[float, ...],
    ) -> bool:
        '''주어진 configuration이 안전하면 True를 반환한다.'''
        ...


class PassThroughSafetyGate:
    '''SafetyGate 비활성화 시 후보 action을 수정 없이 통과시킨다.'''

    def reset(
        self,
    ) -> None:
        '''Pass-through gate에는 reset할 상태가 없으므로 아무 작업도 하지 않는다.'''

    def evaluate(
        self,
        observation: RobotObservation,
        action: RobotAction,
    ) -> SafetyDecision:
        '''후보 action을 항상 accepted 상태로 반환한다.'''
        return SafetyDecision(
            accepted=True,
            action=action,
        )


class SweptPathSafetyGate:
    '''현재 state와 목표 action 사이를 보간하여 모든 configuration을 검사한다.'''

    def __init__(
        self,
        validator: ConfigurationValidator,
        substeps: int = 20,
    ) -> None:
        '''Configuration validator와 양수 substep 수를 저장한다.'''
        if substeps <= 0:
            raise ValueError('substeps must be positive')
        self._validator: ConfigurationValidator = validator
        self._substeps: int = substeps
        self._latched: bool = False

    def reset(
        self,
    ) -> None:
        '''새 episode 시작 시 latched stop을 해제한다.'''
        self._latched = False

    def _hold_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''현재 measured state를 hold action으로 반환한다.'''
        return RobotAction(values=observation.state)

    def evaluate(
        self,
        observation: RobotObservation,
        action: RobotAction,
    ) -> SafetyDecision:
        '''Shape `(S,)` state에서 action까지의 모든 보간 configuration을 검사한다.'''
        if len(observation.state) != len(action.values):
            raise ValueError('observation and action dimensions must match')
        if self._latched:
            return SafetyDecision(
                accepted=False,
                action=self._hold_action(observation),
                reason='safety stop is latched',
            )
        step_index: int
        for step_index in range(1, self._substeps + 1):
            alpha: float = step_index / self._substeps
            # Shape `(S,)`에서 shape `(S,)`로 joint-space 선형 보간
            configuration: tuple[float, ...] = tuple(
                current + alpha * (target - current)
                for current, target in zip(observation.state, action.values, strict=True)
            )
            if not self._validator.is_safe(configuration):
                self._latched = True
                return SafetyDecision(
                    accepted=False,
                    action=self._hold_action(observation),
                    reason=f'unsafe swept-path configuration at substep {step_index}',
                )
        return SafetyDecision(
            accepted=True,
            action=action,
        )
