'''Open-loop chunk 실행과 upstream RTC adapter를 제공한다.'''

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

from ..interfaces import ActionProducer, VLABackend
from ..types import ActionChunk, RobotAction, RobotObservation, RtcGuidance


class OpenLoopChunkExecutor:
    '''VLA action chunk를 앞에서부터 순서대로 실행하고 execute_steps만큼 실행하면 다시 예측한다.'''

    def __init__(
        self,
        backend: VLABackend,
        execute_steps: int | None = None,
    ) -> None:
        '''VLA backend와 chunk당 실행할 action 수, 비어 있는 action chunk 상태를 저장한다.'''
        if execute_steps is not None and execute_steps <= 0:
            raise ValueError('execute_steps must be positive')
        self._backend: VLABackend = backend
        # None이면 chunk 전체를 실행한다.
        self._execute_steps: int | None = execute_steps
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
        if self._chunk is None or self._index >= self._chunk_limit():
            self._chunk = self._backend.predict_chunk(observation)
            self._index = 0
        if self._chunk.horizon == 0:
            raise ValueError('VLA returned an empty action chunk')
        values: tuple[float, ...] = self._chunk.values[self._index]
        self._index += 1
        return RobotAction(values=values)

    def _chunk_limit(
        self,
    ) -> int:
        '''현재 chunk에서 실행할 action 수를 반환한다.'''
        if self._chunk is None:
            return 0
        if self._execute_steps is None:
            return self._chunk.horizon
        return min(self._execute_steps, self._chunk.horizon)

    def close(
        self,
    ) -> None:
        '''VLA resource와 cached chunk를 해제한다.'''
        self._backend.close()
        self._chunk = None
        self._index = 0


class RealTimeChunkExecutor:
    '''RTC soft masking guidance로 다음 chunk를 비동기 예측하고 이어서 실행한다.

    Chunk 정렬은 https://github.com/Physical-Intelligence/real-time-chunking-kinetix 의
    `src/eval_flow.py`를 따른다. 요청 시점부터 남은 이전 chunk를 guidance로 전달하고, 새 chunk가
    도착하면 그 사이 이전 chunk로 실행한 step 수만큼 새 chunk 앞부분을 건너뛴다.
    '''

    def __init__(
        self,
        backend: VLABackend,
        inference_delay: int,
        execute_horizon: int,
    ) -> None:
        '''Backend와 예상 inference delay, chunk 요청 간격을 저장한다.'''
        if not 0 <= inference_delay <= execute_horizon or execute_horizon <= 0:
            raise ValueError('RTC requires 0 <= inference_delay <= execute_horizon and execute_horizon > 0')
        self._backend: VLABackend = backend
        self._inference_delay: int = inference_delay
        self._execute_horizon: int = execute_horizon
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[ActionChunk] | None = None
        self._chunk: ActionChunk | None = None
        self._index: int = 0
        self._request_index: int = 0

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
        self._request_index = 0

    def next_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''현재 chunk를 실행하고 execute_horizon마다 guidance를 붙인 다음 chunk를 미리 요청한다.'''
        if self._executor is None:
            raise RuntimeError('RTC executor is not connected')
        if self._chunk is None:
            self._chunk = self._backend.predict_chunk(observation)
            if self._chunk.horizon <= self._execute_horizon:
                raise ValueError('RTC action horizon must exceed execute_horizon')
            self._index = 0
        if self._future is not None and self._future.done():
            # 새 chunk의 index 0은 요청 시점 action이므로 그 뒤 이전 chunk로 실행한 step 수만큼 건너뛴다.
            next_chunk: ActionChunk = self._future.result()
            self._future = None
            if next_chunk.horizon != self._chunk.horizon:
                raise ValueError('RTC continuation must keep the action horizon')
            self._index = min(self._index - self._request_index, next_chunk.horizon - 1)
            self._chunk = next_chunk
        if self._index >= self._chunk.horizon:
            # 다음 chunk가 늦으면 마지막 target을 유지하고 inference를 기다린다.
            # 새 chunk를 요청 시점 기준 경과 step에 맞추도록 대기한 step도 index에 센다.
            self._index += 1
            return RobotAction(values=self._chunk.values[-1])
        if self._future is None and self._index >= self._execute_horizon:
            # 추론이 늦어 execute_horizon보다 뒤에서 요청하면 남은 이전 chunk 길이까지만 참고한다.
            prefix_attention_horizon: int = self._chunk.horizon - max(self._execute_horizon, self._index)
            # 현재 step action부터 남은 이전 chunk를 요청 시점 observation에 정렬해 guidance로 전달한다.
            rtc: RtcGuidance = RtcGuidance(
                prev_action_chunk=ActionChunk(values=self._chunk.values[self._index:]),
                inference_delay=min(self._inference_delay, prefix_attention_horizon),
                prefix_attention_horizon=prefix_attention_horizon,
            )
            self._request_index = self._index
            self._future = self._executor.submit(self._backend.predict_chunk, observation, rtc)
        values: tuple[float, ...] = self._chunk.values[self._index]
        self._index += 1
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
        self._request_index = 0


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
