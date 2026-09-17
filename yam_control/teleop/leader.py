'''YAM Leader Arm state를 single-arm RobotAction으로 변환한다.'''

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

from ..types import RobotAction, RobotObservation


class LeaderReader(Protocol):
    '''Leader Arm joint와 teaching handle 입력을 읽는 interface를 정의한다.'''

    def get_state(
        self,
    ) -> tuple[Iterable[float], float, list[bool]]:
        '''Shape `(6,)` arm joint, gripper scalar와 button 상태를 반환한다.'''
        ...


LeaderLoader = Callable[[], LeaderReader]


class LeaderActionProducer:
    '''Leader Arm joint와 gripper를 shape `(7,)` follower action으로 변환한다.'''

    def __init__(
        self,
        loader: LeaderLoader,
    ) -> None:
        '''Leader Arm 연결을 지연시키는 loader를 저장한다.'''
        self._loader: LeaderLoader = loader
        self._reader: LeaderReader | None = None

    def connect(
        self,
    ) -> None:
        '''yam-abc-reproduce의 YAM Leader Arm reader를 생성한다.'''
        self._reader = self._loader()

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''Leader mapping은 absolute joint mapping이므로 별도 상태를 reset하지 않는다.'''

    def next_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''Leader joint와 normalized gripper를 shape `(7,)` action으로 반환한다.'''
        if self._reader is None:
            raise RuntimeError('LeaderActionProducer is not connected')
        arm_values: Iterable[float]
        gripper_value: float
        button_states: list[bool]
        arm_values, gripper_value, button_states = self._reader.get_state()
        del button_states
        # Shape `(6,)` arm joint를 immutable tuple로 변환
        arm: tuple[float, ...] = tuple(float(value) for value in arm_values)
        return RobotAction(values=(*arm, float(gripper_value)))

    def close(
        self,
    ) -> None:
        '''Leader reader reference를 해제한다.'''
        self._reader = None
