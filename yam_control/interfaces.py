'''Runtime 구현체가 따라야 하는 공통 Protocol을 정의한다.'''

from __future__ import annotations

from typing import Protocol

from .types import ActionChunk, RobotAction, RobotObservation, SafetyDecision


class RobotBackend(Protocol):
    '''실제 YAM과 MuJoCo YAM이 공유하는 실행 interface를 정의한다.'''

    def connect(
        self,
    ) -> None:
        '''Backend resource를 연결한다.'''
        ...

    def get_observation(
        self,
    ) -> RobotObservation:
        '''현재 shape `(S,)` robot state와 image를 반환한다.'''
        ...

    def execute(
        self,
        action: RobotAction,
    ) -> None:
        '''SafetyGate를 통과한 shape `(S,)` action을 실행한다.'''
        ...

    def move_to_pose(
        self,
        pose: tuple[float, ...],
    ) -> None:
        '''Recording 없이 robot을 지정한 shape `(S,)` pose로 이동한다.'''
        ...

    def reset_environment(
        self,
        initial_pose: tuple[float, ...],
    ) -> None:
        '''Simulator 환경과 robot을 episode 시작 상태로 reset한다.'''
        ...

    def hold(
        self,
    ) -> None:
        '''현재 pose를 유지한다.'''
        ...

    def close(
        self,
    ) -> None:
        '''Backend resource를 해제한다.'''
        ...


class ActionProducer(Protocol):
    '''Teleoperation 또는 inference에서 단일 RobotAction을 생성한다.'''

    def connect(
        self,
    ) -> None:
        '''Controller 또는 policy resource를 연결한다.'''
        ...

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''새 episode의 measured state를 기준으로 내부 상태를 reset한다.'''
        ...

    def next_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''현재 observation에 대한 shape `(S,)` action을 반환한다.'''
        ...

    def close(
        self,
    ) -> None:
        '''Controller 또는 policy resource를 해제한다.'''
        ...


class VLABackend(Protocol):
    '''VLA observation을 shape `(H, S)` ActionChunk로 변환한다.'''

    def connect(
        self,
    ) -> None:
        '''Checkpoint 또는 inference server resource를 연결한다.'''
        ...

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''새 episode의 measured state를 기준으로 policy state를 reset한다.'''
        ...

    def predict_chunk(
        self,
        observation: RobotObservation,
    ) -> ActionChunk:
        '''현재 observation으로 action chunk를 예측한다.'''
        ...

    def close(
        self,
    ) -> None:
        '''Checkpoint 또는 inference server resource를 해제한다.'''
        ...


class SafetyGate(Protocol):
    '''후보 action을 검사하여 실제 실행 가능한 action을 반환한다.'''

    def reset(
        self,
    ) -> None:
        '''새 episode에서 latched stop 상태를 해제한다.'''
        ...

    def evaluate(
        self,
        observation: RobotObservation,
        action: RobotAction,
    ) -> SafetyDecision:
        '''현재 state에서 후보 action까지의 안전성을 검사한다.'''
        ...


class EpisodeRecorder(Protocol):
    '''Episode boundary와 실행 action을 저장하는 interface를 정의한다.'''

    def start(
        self,
        task_prompt: str,
    ) -> None:
        '''Episode recording을 시작한다.'''
        ...

    def record(
        self,
        observation: RobotObservation,
        action: RobotAction,
    ) -> None:
        '''한 control step의 observation과 실제 실행 action을 저장한다.'''
        ...

    def finish(
        self,
    ) -> None:
        '''정상 종료한 episode를 finalize한다.'''
        ...

    def abort(
        self,
    ) -> None:
        '''안전 문제로 중단된 episode를 폐기한다.'''
        ...
