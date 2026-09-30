'''YAM WebSocket policy adapter와 실제 robot action 경계를 검증한다.'''

from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from yam_control.policy.pi import WebsocketPiBackend
from yam_control.policy.rtc import RealTimeChunkExecutor
from yam_control.safety import JointStepSafetyGate
from yam_control.types import ActionChunk, RobotAction, RobotObservation


class FakePolicyClient:
    '''WebSocket 없이 OpenPI client의 최소 inference contract를 제공한다.'''

    def __init__(
        self,
        host: str,
        port: int,
    ) -> None:
        '''서버 주소를 저장하고 observation capture를 준비한다.'''
        self.host: str = host
        self.port: int = port
        self.observation: dict[str, object] | None = None
        self.closed: bool = False

    def infer(
        self,
        observation: dict[str, object],
    ) -> dict[str, object]:
        '''OpenPI 입력을 저장하고 shape `(2, 7)` action을 반환한다.'''
        self.observation = observation
        return {'actions': np.zeros((2, 7), dtype=np.float32)}

    def reset(
        self,
    ) -> None:
        '''Stateless client의 episode reset을 수행한다.'''

    def close(
        self,
    ) -> None:
        '''Test client 연결 종료를 기록한다.'''
        self.closed = True


def test_websocket_pi_backend_maps_observation_and_actions(
) -> None:
    '''YAM state와 camera가 OpenPI 키로 전송되고 action chunk가 7차원인지 검증한다.'''
    with patch('yam_control.policy.pi.WebsocketClientPolicy', FakePolicyClient):
        backend: WebsocketPiBackend = WebsocketPiBackend(host='127.0.0.1', port=8000, prompt='cable')
        backend.connect()
        image: np.ndarray = np.zeros((224, 224, 3), dtype=np.uint8)
        camera_frame: SimpleNamespace = SimpleNamespace(images={'rgb': image})
        observation: RobotObservation = RobotObservation(
            state=(0.0,) * 7,
            images={'top': camera_frame, 'wrist': camera_frame},
        )
        backend.reset(observation)
        actions: tuple[tuple[float, ...], ...] = backend.predict_chunk(observation).values
        client: FakePolicyClient = backend._client  # type: ignore[assignment]
        assert client.observation is not None
        assert set(client.observation) == {'observation/state', 'observation/image', 'observation/left_wrist', 'prompt'}
        assert np.asarray(client.observation['observation/state']).shape == (7,)
        assert np.asarray(client.observation['observation/image']).shape == (224, 224, 3)
        assert len(actions) == 2 and all(len(action) == 7 for action in actions)
        action_prefix: ActionChunk = ActionChunk(values=((0.1,) * 7, (0.2,) * 7))
        backend.predict_chunk(observation, action_prefix)
        assert client.observation is not None
        np.testing.assert_allclose(client.observation['action_prefix'], action_prefix.values)
        backend.close()
        assert client.closed


def test_joint_step_safety_gate_rejects_large_action(
) -> None:
    '''실제 YAM에 전달할 큰 joint step과 gripper 범위 오류를 거부한다.'''
    gate: JointStepSafetyGate = JointStepSafetyGate(max_joint_step_rad=0.1)
    observation: RobotObservation = RobotObservation(state=(0.0,) * 6 + (0.5,))
    assert gate.evaluate(observation, RobotAction(values=(0.05,) * 6 + (0.5,))).accepted
    assert not gate.evaluate(observation, RobotAction(values=(0.2,) + (0.0,) * 5 + (0.5,))).accepted
    assert not gate.evaluate(observation, RobotAction(values=(0.0,) * 6 + (1.2,))).accepted
    assert not gate.evaluate(observation, RobotAction(values=(float('nan'),) + (0.0,) * 5 + (0.5,))).accepted


class FakeRtcBackend:
    '''RTC prefix를 확인하고 다음 action chunk를 지연 반환한다.'''

    def __init__(
        self,
    ) -> None:
        '''다음 chunk를 허용하는 event와 prefix 기록을 준비한다.'''
        self.release: Event = Event()
        self.prefix: ActionChunk | None = None

    def connect(
        self,
    ) -> None:
        '''Test backend 연결 상태를 유지한다.'''

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''새 episode의 observation을 받는다.'''
        del observation

    def predict_chunk(
        self,
        observation: RobotObservation,
        action_prefix: ActionChunk | None = None,
    ) -> ActionChunk:
        '''첫 chunk 또는 prefix-conditioned continuation을 반환한다.'''
        del observation
        if action_prefix is None:
            return ActionChunk(values=tuple((float(index),) * 7 for index in range(6)))
        self.prefix = action_prefix
        assert self.release.wait(timeout=2)
        return ActionChunk(values=action_prefix.values + tuple((float(index),) * 7 for index in range(6, 10)))

    def close(
        self,
    ) -> None:
        '''Test backend를 해제한다.'''


def test_rtc_executor_prefetches_prefix_and_holds_on_underrun(
) -> None:
    '''RTC가 prefix를 전달하고 inference 지연 중 마지막 target을 유지하는지 검증한다.'''
    backend: FakeRtcBackend = FakeRtcBackend()
    executor: RealTimeChunkExecutor = RealTimeChunkExecutor(backend=backend, prefix_length=2, lead_steps=2)
    observation: RobotObservation = RobotObservation(state=(0.0,) * 7)
    executor.connect()
    executor.reset(observation)
    values: tuple[float, ...] = tuple(executor.next_action(observation).values[0] for _ in range(6))
    assert values == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert executor.next_action(observation).values[0] == 5.0
    backend.release.set()
    assert executor._future is not None
    executor._future.result(timeout=2)
    assert backend.prefix is not None
    assert backend.prefix.values == ((4.0,) * 7, (5.0,) * 7)
    assert executor.next_action(observation).values[0] == 6.0
    executor.close()
