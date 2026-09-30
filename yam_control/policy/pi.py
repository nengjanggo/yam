'''π0와 π0.5 checkpoint inference backend의 지연 연결 경계를 제공한다.'''

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from ..config import InferenceRunConfig, RunConfig
from ..interfaces import ActionProducer, VLABackend
from .rtc import OpenLoopChunkExecutor, RealTimeChunkExecutor
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
        action_prefix: ActionChunk | None = None,
    ) -> ActionChunk:
        '''로드된 π policy로 shape `(H, S)` action chunk를 예측한다.'''
        if self._backend is None:
            raise RuntimeError('PiBackend is not connected')
        if action_prefix is None:
            return self._backend.predict_chunk(observation)
        return self._backend.predict_chunk(observation, action_prefix)

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


def _rgb_image(
    image: object,
) -> np.ndarray:
    '''RGB array 또는 CameraFrame을 받아 shape `(224, 224, 3)` uint8 image를 반환한다.'''
    # 실제 camera의 CameraFrame에서는 RGB image만 꺼낸다.
    frame_images: object = getattr(image, 'images', None)
    if isinstance(frame_images, Mapping):
        image = frame_images['rgb']
    rgb_image: np.ndarray = np.asarray(image)
    if rgb_image.shape != (224, 224, 3) or rgb_image.dtype != np.uint8:
        raise ValueError('policy image must have shape (224, 224, 3) and dtype uint8')
    return rgb_image


class WebsocketPiBackend:
    '''YAM observation을 OpenPI WebSocket policy 입력으로 변환한다.'''

    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
    ) -> None:
        '''서버 주소와 task prompt를 저장하고 연결은 connect까지 미룬다.'''
        self._host: str = host
        self._port: int = port
        self._prompt: str = prompt
        self._client: WebsocketClientPolicy | None = None

    def connect(
        self,
    ) -> None:
        '''기존 OpenPI WebSocket client로 policy server에 연결한다.'''
        self._client = WebsocketClientPolicy(host=self._host, port=self._port)

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''새 episode에서 client policy 상태를 초기화한다.'''
        del observation
        if self._client is None:
            raise RuntimeError('OpenPI client is not connected')
        self._client.reset()

    def predict_chunk(
        self,
        observation: RobotObservation,
        action_prefix: ActionChunk | None = None,
    ) -> ActionChunk:
        '''측정 state, camera frame과 optional prefix로 `(H, 7)` action chunk를 요청한다.'''
        if self._client is None:
            raise RuntimeError('OpenPI client is not connected')
        # Shape `(7,)` measured state와 두 shape `(224, 224, 3)` RGB image를 전송한다.
        state: np.ndarray = np.asarray(observation.state, dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError('policy state must be finite and have shape (7,)')
        top_image: np.ndarray = _rgb_image(observation.images['top'])
        wrist_image: np.ndarray = _rgb_image(observation.images['wrist'])
        policy_input: dict[str, object] = {
            'observation/state': state,
            'observation/image': top_image,
            'observation/left_wrist': wrist_image,
            'prompt': self._prompt,
        }
        if action_prefix is not None:
            # Shape `(prefix_length, 7)`의 절대 joint target을 OpenPI RTC 입력으로 전달한다.
            prefix_values: np.ndarray = np.asarray(action_prefix.values, dtype=np.float32)
            if prefix_values.ndim != 2 or prefix_values.shape[1] != 7 or not np.isfinite(prefix_values).all():
                raise ValueError('action_prefix must be finite and have shape (prefix_length, 7)')
            policy_input['action_prefix'] = prefix_values
        policy_output: dict[str, object] = self._client.infer(policy_input)
        # Shape `(H, 7)`의 절대 joint target을 YAM action chunk로 변환한다.
        actions: np.ndarray = np.asarray(policy_output['actions'], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError('policy actions must be finite and have shape (H, 7) with H > 0')
        values: tuple[tuple[float, ...], ...] = tuple(tuple(float(value) for value in action) for action in actions)
        return ActionChunk(values=values)

    def close(
        self,
    ) -> None:
        '''WebSocket 연결을 닫고 client 상태를 해제한다.'''
        if self._client is not None:
            self._client.close()
        self._client = None


def create_pi_action_producer(
    config: RunConfig,
    host: str,
    port: int,
    rtc_prefix_length: int = 4,
    rtc_lead_steps: int = 8,
) -> ActionProducer:
    '''Inference RunConfig와 server 주소로 선택한 chunk executor를 생성한다.'''
    if not isinstance(config.mode_config, InferenceRunConfig):
        raise ValueError('Pi action producer requires inference mode')

    def load_policy(
        inference_config: InferenceRunConfig,
    ) -> VLABackend:
        '''Inference config를 받아 YAM WebSocket backend를 생성한다.'''
        del inference_config
        return WebsocketPiBackend(host=host, port=port, prompt=config.task_prompt)

    backend: PiBackend = PiBackend(config=config.mode_config, loader=load_policy)
    if config.mode_config.use_rtc:
        return RealTimeChunkExecutor(
            backend=backend,
            prefix_length=rtc_prefix_length,
            lead_steps=rtc_lead_steps,
        )
    return OpenLoopChunkExecutor(backend=backend)
