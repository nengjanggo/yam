'''Recording 비활성화용 recorder와 external recorder adapter 경계를 제공한다.'''

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ..types import RobotAction, RobotObservation


class NullRecorder:
    '''Recording 비활성화 시 동일한 lifecycle을 유지하는 no-op recorder이다.'''

    def start(
        self,
        task_prompt: str,
    ) -> None:
        '''Episode 시작 요청을 받고 아무 데이터도 만들지 않는다.'''

    def record(
        self,
        observation: RobotObservation,
        action: RobotAction,
    ) -> None:
        '''Control step을 받고 아무 데이터도 저장하지 않는다.'''

    def finish(
        self,
    ) -> None:
        '''Episode 정상 종료 요청을 받고 아무 작업도 하지 않는다.'''

    def abort(
        self,
    ) -> None:
        '''Episode 중단 요청을 받고 아무 작업도 하지 않는다.'''


class YamABCRecorder(Protocol):
    '''yam-abc-reproduce EpisodeRecorder의 lifecycle을 정의한다.'''

    def start(
        self,
        task_name: str,
    ) -> object:
        '''새 raw episode recording을 시작한다.'''
        ...

    def tick(
        self,
        actions: object,
        obs: object,
        frames: object,
    ) -> None:
        '''한 control step의 encoded data를 저장한다.'''
        ...

    def stop(
        self,
    ) -> object:
        '''Episode를 yam-abc raw format으로 finalize한다.'''
        ...

    def abort(
        self,
    ) -> object:
        '''완료되지 않은 raw episode를 폐기한다.'''
        ...


EncodedStep = tuple[object, object, object]
StepEncoder = Callable[[RobotObservation, RobotAction], EncodedStep]


class YamABCRecorderAdapter:
    '''공통 observation/action을 yam-abc-reproduce recorder 입력으로 변환한다.'''

    def __init__(
        self,
        recorder: YamABCRecorder,
        step_encoder: StepEncoder,
        episode_directory: Path,
        complete_flag_name: str,
    ) -> None:
        '''External recorder, camera-aware step encoder와 task episode 폴더를 저장한다.'''
        self._recorder: YamABCRecorder = recorder
        self._step_encoder: StepEncoder = step_encoder
        self._episode_directory: Path = episode_directory
        self._complete_flag_name: str = complete_flag_name

    def saved_episode_count(
        self,
    ) -> int:
        '''현재 task 폴더에서 저장이 끝까지 완료된 episode 수를 반환한다.'''
        if not self._episode_directory.is_dir():
            return 0
        return sum(
            (episode_path / self._complete_flag_name).is_file()
            for episode_path in self._episode_directory.iterdir()
        )

    def start(
        self,
        task_prompt: str,
    ) -> None:
        '''Task prompt를 yam-abc episode task name으로 전달한다.'''
        self._recorder.start(task_prompt)

    def record(
        self,
        observation: RobotObservation,
        action: RobotAction,
    ) -> None:
        '''Shape `(S,)` data와 camera frame을 yam-abc schema로 encoding해 저장한다.'''
        actions: object
        obs: object
        frames: object
        actions, obs, frames = self._step_encoder(observation, action)
        self._recorder.tick(actions, obs, frames)

    def finish(
        self,
    ) -> None:
        '''Raw episode를 완료하여 LeRobot v3 변환 가능한 상태로 만든다.'''
        self._recorder.stop()

    def abort(
        self,
    ) -> None:
        '''완료되지 않은 yam-abc raw episode를 폐기한다.'''
        self._recorder.abort()
