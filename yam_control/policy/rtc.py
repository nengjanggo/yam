'''Open-loop chunk 실행과 upstream RTC adapter를 제공한다.'''

from __future__ import annotations

from ..interfaces import ActionProducer, VLABackend
from ..types import ActionChunk, RobotAction, RobotObservation


class OpenLoopChunkExecutor:
    '''VLA action chunk를 앞에서부터 순서대로 실행하고 소진 시 다시 예측한다.'''

    def __init__(
        self,
        backend: VLABackend,
    ) -> None:
        '''VLA backend와 비어 있는 action chunk 상태를 저장한다.'''
        self._backend: VLABackend = backend
        self._chunk: ActionChunk | None = None
        self._index: int = 0

    def connect(
        self,
    ) -> None:
        '''VLA checkpoint 또는 inference server resource를 연결한다.'''
        self._backend.connect()

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''새 episode에서 backend와 cached action chunk를 reset한다.'''
        self._backend.reset(observation)
        self._chunk = None
        self._index = 0

    def next_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''현재 chunk의 다음 shape `(S,)` action을 반환한다.'''
        if self._chunk is None or self._index >= self._chunk.horizon:
            self._chunk = self._backend.predict_chunk(observation)
            self._index = 0
        if self._chunk.horizon == 0:
            raise ValueError('VLA returned an empty action chunk')
        values: tuple[float, ...] = self._chunk.values[self._index]
        self._index += 1
        return RobotAction(values=values)

    def close(
        self,
    ) -> None:
        '''VLA resource와 cached chunk를 해제한다.'''
        self._backend.close()
        self._chunk = None
        self._index = 0


class RTCActionProducerAdapter:
    '''Upstream RTC implementation을 공통 ActionProducer로 노출하는 얇은 adapter이다.'''

    def __init__(
        self,
        upstream_producer: ActionProducer,
    ) -> None:
        '''Physical Intelligence reference를 적용한 upstream producer를 저장한다.'''
        self._upstream_producer: ActionProducer = upstream_producer

    def connect(
        self,
    ) -> None:
        '''Upstream RTC resource를 연결한다.'''
        self._upstream_producer.connect()

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''Upstream RTC episode state를 reset한다.'''
        self._upstream_producer.reset(observation)

    def next_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''Upstream RTC가 생성한 현재 control-step action을 반환한다.'''
        return self._upstream_producer.next_action(observation)

    def close(
        self,
    ) -> None:
        '''Upstream RTC resource를 해제한다.'''
        self._upstream_producer.close()
