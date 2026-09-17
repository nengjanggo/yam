'''π0와 π0.5 checkpoint inference backend의 지연 연결 경계를 제공한다.'''

from __future__ import annotations

from typing import Callable

from ..config import InferenceRunConfig
from ..interfaces import VLABackend
from ..types import ActionChunk, RobotObservation

PolicyLoader = Callable[[InferenceRunConfig], VLABackend]


class PiBackend:
    '''Checkpoint loader를 connect 시점까지 호출하지 않는 π backend이다.'''

    def __init__(
        self,
        config: InferenceRunConfig,
        loader: PolicyLoader,
    ) -> None:
        '''π configuration과 external openpi loader를 저장한다.'''
        if config.vla_type not in ('pi0', 'pi0.5'):
            raise ValueError('PiBackend supports only pi0 and pi0.5')
        self._config: InferenceRunConfig = config
        self._loader: PolicyLoader = loader
        self._backend: VLABackend | None = None

    def connect(
        self,
    ) -> None:
        '''Configured checkpoint를 external openpi loader로 로드한다.'''
        self._backend = self._loader(self._config)
        self._backend.connect()

    def predict_chunk(
        self,
        observation: RobotObservation,
    ) -> ActionChunk:
        '''로드된 π policy로 shape `(H, S)` action chunk를 예측한다.'''
        if self._backend is None:
            raise RuntimeError('PiBackend is not connected')
        return self._backend.predict_chunk(observation)

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''로드된 π policy의 episode state를 measured state 기준으로 reset한다.'''
        if self._backend is None:
            raise RuntimeError('PiBackend is not connected')
        self._backend.reset(observation)

    def close(
        self,
    ) -> None:
        '''로드된 π policy resource를 해제한다.'''
        if self._backend is not None:
            self._backend.close()
        self._backend = None
