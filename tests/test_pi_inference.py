'''YAM WebSocket policy adapter와 실제 robot action 경계를 검증한다.'''

from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from yam_control.policy.pi import WebsocketPiBackend
from yam_control.policy.rtc import OpenLoopChunkExecutor, RealTimeChunkExecutor
from yam_control.safety import JointStepSafetyGate
from yam_control.types import ActionChunk, RobotAction, RobotObservation, RtcGuidance


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
        rtc: RtcGuidance = RtcGuidance(
            prev_action_chunk=ActionChunk(values=((0.1,) * 7, (0.2,) * 7)),
            inference_delay=1,
            prefix_attention_horizon=2,
        )
        backend.predict_chunk(observation, rtc)
        assert client.observation is not None
        np.testing.assert_allclose(client.observation['rtc_prev_action_chunk'], rtc.prev_action_chunk.values)
        assert client.observation['rtc_inference_delay'] == 1
        assert client.observation['rtc_prefix_attention_horizon'] == 2
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
    '''RTC guidance를 기록하고 다음 action chunk를 지연 반환한다.'''

    def __init__(
        self,
    ) -> None:
        '''다음 chunk를 허용하는 event와 guidance 기록을 준비한다.'''
        self.release: Event = Event()
        self.rtc: RtcGuidance | None = None

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
        rtc: RtcGuidance | None = None,
    ) -> ActionChunk:
        '''첫 chunk 0~5 또는 요청 시점에 정렬된 continuation 100~105를 반환한다.'''
        del observation
        if rtc is None:
            return ActionChunk(values=tuple((float(index),) * 7 for index in range(6)))
        self.rtc = rtc
        assert self.release.wait(timeout=2)
        return ActionChunk(values=tuple((float(100 + index),) * 7 for index in range(6)))

    def close(
        self,
    ) -> None:
        '''Test backend를 해제한다.'''


def test_rtc_executor_aligns_guidance_and_skips_executed_steps(
) -> None:
    '''RTC가 남은 이전 chunk를 guidance로 보내고 도착 시 실행한 step만큼 건너뛰는지 검증한다.'''
    backend: FakeRtcBackend = FakeRtcBackend()
    executor: RealTimeChunkExecutor = RealTimeChunkExecutor(backend=backend, inference_delay=1, execute_horizon=2)
    observation: RobotObservation = RobotObservation(state=(0.0,) * 7)
    executor.connect()
    executor.reset(observation)
    # Index 2에서 요청하고 inference가 끝날 때까지 이전 chunk를 계속 실행한다.
    values: tuple[float, ...] = tuple(executor.next_action(observation).values[0] for _ in range(4))
    assert values == (0.0, 1.0, 2.0, 3.0)
    assert backend.rtc is not None
    assert backend.rtc.prev_action_chunk.values == tuple((float(index),) * 7 for index in range(2, 6))
    assert backend.rtc.inference_delay == 1
    assert backend.rtc.prefix_attention_horizon == 4
    backend.release.set()
    assert executor._future is not None
    executor._future.result(timeout=2)
    # 요청 후 이전 chunk로 2 step(2, 3)을 실행했으므로 새 chunk는 index 2부터 이어진다.
    assert executor.next_action(observation).values[0] == 102.0
    executor.close()


def test_rtc_executor_holds_last_target_on_underrun(
) -> None:
    '''다음 chunk가 늦어 이전 chunk를 모두 실행하면 마지막 target을 유지하는지 검증한다.'''
    backend: FakeRtcBackend = FakeRtcBackend()
    executor: RealTimeChunkExecutor = RealTimeChunkExecutor(backend=backend, inference_delay=1, execute_horizon=2)
    observation: RobotObservation = RobotObservation(state=(0.0,) * 7)
    executor.connect()
    executor.reset(observation)
    values: tuple[float, ...] = tuple(executor.next_action(observation).values[0] for _ in range(8))
    assert values == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0)
    backend.release.set()
    assert executor._future is not None
    executor._future.result(timeout=2)
    # 요청 후 6 step이 지나 horizon을 넘었으므로 새 chunk의 마지막 action부터 이어진다.
    assert executor.next_action(observation).values[0] == 105.0
    executor.close()


class CountingChunkBackend:
    '''predict_chunk 호출 횟수를 첫 action 값으로 담은 chunk를 반환한다.'''

    def __init__(
        self,
        horizon: int,
    ) -> None:
        '''반환할 chunk 길이와 호출 횟수를 준비한다.'''
        self.horizon: int = horizon
        self.calls: int = 0

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
        '''호출 순번 * 100 + chunk 내 index 값을 가진 chunk를 반환한다.'''
        del observation, action_prefix
        self.calls += 1
        return ActionChunk(values=tuple((float(self.calls * 100 + index),) * 7 for index in range(self.horizon)))

    def close(
        self,
    ) -> None:
        '''Test backend를 해제한다.'''


def test_open_loop_executor_executes_only_first_steps_of_each_chunk(
) -> None:
    '''execute_steps만큼 실행한 뒤 남은 action을 버리고 다시 예측하는지 검증한다.'''
    backend: CountingChunkBackend = CountingChunkBackend(horizon=5)
    executor: OpenLoopChunkExecutor = OpenLoopChunkExecutor(backend=backend, execute_steps=2)
    observation: RobotObservation = RobotObservation(state=(0.0,) * 7)
    executor.connect()
    executor.reset(observation)
    values: tuple[float, ...] = tuple(executor.next_action(observation).values[0] for _ in range(5))
    assert values == (100.0, 101.0, 200.0, 201.0, 300.0)
    assert backend.calls == 3


def test_open_loop_executor_defaults_to_full_chunk_and_caps_at_horizon(
) -> None:
    '''execute_steps가 없거나 horizon보다 크면 chunk 전체를 실행하는지 검증한다.'''
    observation: RobotObservation = RobotObservation(state=(0.0,) * 7)
    for execute_steps in (None, 10):
        backend: CountingChunkBackend = CountingChunkBackend(horizon=3)
        executor: OpenLoopChunkExecutor = OpenLoopChunkExecutor(backend=backend, execute_steps=execute_steps)
        executor.reset(observation)
        values: tuple[float, ...] = tuple(executor.next_action(observation).values[0] for _ in range(4))
        assert values == (100.0, 101.0, 102.0, 200.0)
