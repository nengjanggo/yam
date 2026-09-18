'''Runtime component가 공유하는 observation, action과 episode type을 정의한다.'''

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class EpisodeState(Enum):
    '''Episode lifecycle 상태를 나타낸다.'''

    IDLE = 'idle'
    PREPARED = 'prepared'
    RUNNING = 'running'
    FINISHED = 'finished'
    SUCCEEDED = 'succeeded'
    DISCARDED = 'discarded'
    ABORTED = 'aborted'


class EpisodeOutcome(Enum):
    '''작업자가 episode 도중 입력한 조기 종료 결과를 나타낸다.'''

    SUCCESS = 'success'
    FAILURE = 'failure'


@dataclass(frozen=True)
class RobotObservation:
    '''Shape `(S,)` state와 camera image mapping을 하나의 observation으로 보관한다.'''

    state: tuple[float, ...]
    images: Mapping[str, object] = field(default_factory=dict)
    timestamp_s: float = 0.0


@dataclass(frozen=True)
class RobotAction:
    '''Shape `(S,)`의 단일 control-step robot action을 보관한다.'''

    values: tuple[float, ...]


@dataclass(frozen=True)
class ActionChunk:
    '''Shape `(H, S)`의 VLA action chunk를 보관한다.'''

    values: tuple[tuple[float, ...], ...]

    @property
    def horizon(
        self,
    ) -> int:
        '''Action chunk의 prediction horizon `H`를 반환한다.'''
        return len(self.values)


@dataclass(frozen=True)
class QuestPose:
    '''Shape `(3,)` position과 shape `(4,)` quaternion을 보관한다.'''

    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class QuestFrame:
    '''착용한 HMD와 controller pose 및 analog input을 한 frame으로 보관한다.'''

    controller_pose: QuestPose
    hmd_pose: QuestPose
    trigger: float
    clutch_pressed: bool
    timestamp_s: float
    primary_button_pressed: bool = False
    secondary_button_pressed: bool = False


@dataclass(frozen=True)
class SafetyDecision:
    '''SafetyGate 통과 여부와 실제 실행할 action 및 거부 이유를 보관한다.'''

    accepted: bool
    action: RobotAction
    reason: str | None = None
