'''Open-loop chunk 실행과 upstream RTC adapter를 제공한다.'''

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

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


class RealTimeChunkExecutor:
    '''RTC action prefix로 다음 chunk를 비동기 예측하고 이어서 실행한다.'''

    def __init__(
        self,
        backend: VLABackend,
        prefix_length: int,
        lead_steps: int,
    ) -> None:
        '''Backend와 prefix 길이, 비동기 요청 lead step을 저장한다.'''
        if prefix_length <= 0 or lead_steps < prefix_length:
            raise ValueError('RTC requires 0 < prefix_length <= lead_steps')
        self._backend: VLABackend = backend
        self._prefix_length: int = prefix_length
        self._lead_steps: int = lead_steps
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[ActionChunk] | None = None
        self._chunk: ActionChunk | None = None
        self._index: int = 0

    def connect(
        self,
    ) -> None:
        '''Policy server에 연결하고 RTC inference worker를 생성한다.'''
        self._backend.connect()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''이전 요청을 완료한 뒤 새 episode의 chunk state를 비운다.'''
        if self._future is not None:
            self._future.result()
        self._backend.reset(observation)
        self._future = None
        self._chunk = None
        self._index = 0

    def next_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''현재 chunk를 실행하고 prefix-conditioned 다음 chunk를 미리 요청한다.'''
        if self._executor is None:
            raise RuntimeError('RTC executor is not connected')
        if self._chunk is None:
            self._chunk = self._backend.predict_chunk(observation)
            if self._chunk.horizon <= self._prefix_length:
                raise ValueError('RTC action horizon must exceed prefix_length')
        if self._index >= self._chunk.horizon:
            if self._future is None or not self._future.done():
                # 다음 chunk가 늦으면 마지막 target을 유지하고 inference를 기다린다.
                return RobotAction(values=self._chunk.values[-1])
            self._chunk = self._future.result()
            self._future = None
            if self._chunk.horizon <= self._prefix_length:
                raise ValueError('RTC continuation must exceed prefix_length')
            # 새 chunk의 prefix는 이전 chunk의 마지막 action과 같으므로 실행에서 제외한다.
            self._index = self._prefix_length
        values: tuple[float, ...] = self._chunk.values[self._index]
        self._index += 1
        if self._future is None and self._index >= self._chunk.horizon - self._lead_steps:
            prefix: ActionChunk = ActionChunk(values=self._chunk.values[-self._prefix_length:])
            self._future = self._executor.submit(self._backend.predict_chunk, observation, prefix)
        return RobotAction(values=values)

    def close(
        self,
    ) -> None:
        '''진행 중인 inference를 마친 뒤 worker와 server 연결을 해제한다.'''
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._backend.close()
        self._executor = None
        self._future = None
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
