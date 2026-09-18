'''착용한 Quest 3의 controller frame을 안전한 YAM action으로 변환한다.'''

from __future__ import annotations

import json
import ssl
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Protocol, cast

from ..config import QuestControllerHand
from ..types import QuestFrame, QuestPose, RobotAction, RobotObservation


class WebSocketConnection(Protocol):
    '''Quest relay reader가 사용하는 최소 synchronous WebSocket interface를 정의한다.'''

    def __iter__(
        self,
    ) -> Iterator[str | bytes]:
        '''WebSocket message iterator를 반환한다.'''
        ...

    def close(
        self,
    ) -> None:
        '''WebSocket connection을 종료한다.'''
        ...


WebSocketConnector = Callable[[str, ssl.SSLContext | None, float], WebSocketConnection]


def create_websocket_ssl_context(
    websocket_url: str,
) -> ssl.SSLContext | None:
    '''Local relay self-signed certificate를 허용하는 SSL context를 반환한다.'''
    if not websocket_url.startswith('wss://'):
        return None
    if not websocket_url.startswith(('wss://127.0.0.1:', 'wss://localhost:')):
        return ssl.create_default_context()
    ssl_context: ssl.SSLContext = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context


def _connect_websocket(
    websocket_url: str,
    ssl_context: ssl.SSLContext | None,
    open_timeout_s: float,
) -> WebSocketConnection:
    '''websockets package로 synchronous Quest relay connection을 생성한다.'''
    from websockets.sync.client import connect

    return cast(
        WebSocketConnection,
        connect(
            websocket_url,
            ssl=ssl_context,
            open_timeout=open_timeout_s,
        ),
    )


def _parse_vector(
    value: object,
    dimension: int,
    field_name: str,
) -> tuple[float, ...]:
    '''JSON array를 지정한 dimension의 float tuple로 변환하여 반환한다.'''
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f'{field_name} must have shape ({dimension},)')
    return tuple(float(item) for item in value)


def _parse_pose(
    value: object,
    field_name: str,
) -> QuestPose:
    '''JSON pose mapping을 shape `(3,)` position과 shape `(4,)` quaternion으로 변환한다.'''
    if not isinstance(value, Mapping):
        raise ValueError(f'{field_name} must be an object')
    position: tuple[float, float, float] = cast(
        tuple[float, float, float],
        _parse_vector(value.get('position'), 3, f'{field_name}.position'),
    )
    orientation_xyzw: tuple[float, float, float, float] = cast(
        tuple[float, float, float, float],
        _parse_vector(value.get('orientation'), 4, f'{field_name}.orientation'),
    )
    return QuestPose(
        position=position,
        orientation_xyzw=orientation_xyzw,
    )


def parse_quest_frame_payload(
    payload: object,
    controller_hand: QuestControllerHand,
    received_timestamp_s: float,
) -> QuestFrame:
    '''Relay의 xr_frame payload에서 선택한 controller와 HMD pose를 QuestFrame으로 변환한다.'''
    if not isinstance(payload, Mapping) or payload.get('type') != 'xr_frame':
        raise ValueError('payload must be an xr_frame object')
    if controller_hand not in ('left', 'right'):
        raise ValueError(f'unsupported controller_hand: {controller_hand}')
    controllers: object = payload.get('controllers')
    if not isinstance(controllers, Mapping):
        raise ValueError('controllers must be an object')
    controller: object = controllers.get(controller_hand)
    if not isinstance(controller, Mapping):
        raise ValueError(f'{controller_hand} controller is missing')
    buttons: object = controller.get('buttons')
    if not isinstance(buttons, list) or len(buttons) < 2:
        raise ValueError('controller buttons must include trigger and clutch')
    trigger_button: object = buttons[0]
    clutch_button: object = buttons[1]
    if not isinstance(trigger_button, Mapping) or not isinstance(clutch_button, Mapping):
        raise ValueError('controller button entries must be objects')
    trigger: float = float(trigger_button.get('v', 0.0))
    if not 0.0 <= trigger <= 1.0:
        raise ValueError('trigger must be in [0, 1]')
    return QuestFrame(
        controller_pose=_parse_pose(controller, f'controllers.{controller_hand}'),
        hmd_pose=_parse_pose(payload.get('viewer'), 'viewer'),
        trigger=trigger,
        clutch_pressed=bool(clutch_button.get('p', False)),
        timestamp_s=received_timestamp_s,
    )


def _has_required_tracking_data(
    payload: object,
    controller_hand: QuestControllerHand,
) -> bool:
    '''xr_frame이 선택한 controller, HMD pose와 두 button을 포함하는지 반환한다.'''
    if not isinstance(payload, Mapping) or payload.get('type') != 'xr_frame':
        return False
    controllers: object = payload.get('controllers')
    if not isinstance(controllers, Mapping):
        return False
    controller: object = controllers.get(controller_hand)
    if not isinstance(controller, Mapping):
        return False
    buttons: object = controller.get('buttons')
    return isinstance(buttons, list) and len(buttons) >= 2 and isinstance(payload.get('viewer'), Mapping)


class QuestFrameReader(Protocol):
    '''WebXR relay에서 최신 Quest controller와 HMD frame을 읽는다.'''

    def connect(
        self,
    ) -> None:
        '''WebXR relay에 연결한다.'''
        ...

    def read(
        self,
    ) -> QuestFrame:
        '''최신 QuestFrame을 반환한다.'''
        ...

    def diagnostics(
        self,
    ) -> dict[str, object]:
        '''Quest frame 수신 상태 진단값을 반환한다.'''
        ...

    def close(
        self,
    ) -> None:
        '''WebXR relay 연결을 해제한다.'''
        ...


class QuestRetargeter(Protocol):
    '''Quest reference-space controller pose를 YAM shape `(7,)` action으로 retarget한다.'''

    def rebase(
        self,
        frame: QuestFrame,
        observation: RobotObservation,
    ) -> None:
        '''Clutch engage 시 controller pose, HMD yaw와 robot 기준 pose를 저장한다.'''
        ...

    def retarget(
        self,
        frame: QuestFrame,
        observation: RobotObservation,
    ) -> RobotAction:
        '''현재 controller pose를 YAM joint action으로 변환한다.'''
        ...


class WebSocketQuestFrameReader:
    '''Quest relay stream을 background에서 읽고 가장 최근 QuestFrame을 제공한다.'''

    def __init__(
        self,
        websocket_url: str,
        controller_hand: QuestControllerHand,
        connector: WebSocketConnector = _connect_websocket,
        clock: Callable[[], float] = time.monotonic,
        open_timeout_s: float = 10.0,
    ) -> None:
        '''Relay endpoint, controller hand와 connection dependency를 저장한다.'''
        if not websocket_url.startswith(('ws://', 'wss://')):
            raise ValueError('websocket_url must start with ws:// or wss://')
        if open_timeout_s <= 0.0:
            raise ValueError('open_timeout_s must be positive')
        self._websocket_url: str = websocket_url
        self._controller_hand: QuestControllerHand = controller_hand
        self._connector: WebSocketConnector = connector
        self._clock: Callable[[], float] = clock
        self._open_timeout_s: float = open_timeout_s
        self._condition: threading.Condition = threading.Condition()
        self._stop_event: threading.Event = threading.Event()
        self._connection: WebSocketConnection | None = None
        self._thread: threading.Thread | None = None
        self._latest_frame: QuestFrame | None = None
        self._received_frame_count: int = 0
        self._skipped_tracking_frame_count: int = 0

    def _receive_frames(
        self,
    ) -> None:
        '''Connection message를 읽어 가장 최근 xr_frame으로 교체한다.'''
        connection: WebSocketConnection = self._require_connection()
        raw_message: str | bytes
        for raw_message in connection:
            if self._stop_event.is_set():
                return
            message: str = raw_message.decode('utf-8') if isinstance(raw_message, bytes) else raw_message
            payload: object = json.loads(message)
            if isinstance(payload, Mapping) and payload.get('type') == 'xr_frame':
                # 일시적인 controller 또는 HMD tracking 누락 frame만 건너뜀
                if not _has_required_tracking_data(payload, self._controller_hand):
                    with self._condition:
                        self._skipped_tracking_frame_count += 1
                    continue
                frame: QuestFrame = parse_quest_frame_payload(
                    payload=payload,
                    controller_hand=self._controller_hand,
                    received_timestamp_s=self._clock(),
                )
                with self._condition:
                    self._latest_frame = frame
                    self._received_frame_count += 1
                    self._condition.notify_all()

    def _require_connection(
        self,
    ) -> WebSocketConnection:
        '''연결된 WebSocket을 반환한다.'''
        if self._connection is None:
            raise RuntimeError('WebSocketQuestFrameReader is not connected')
        return self._connection

    def connect(
        self,
    ) -> None:
        '''Quest relay에 연결하고 background receiver를 시작한다.'''
        ssl_context: ssl.SSLContext | None = create_websocket_ssl_context(self._websocket_url)
        self._connection = self._connector(
            self._websocket_url,
            ssl_context,
            self._open_timeout_s,
        )
        self._stop_event.clear()
        self._latest_frame = None
        self._received_frame_count = 0
        self._skipped_tracking_frame_count = 0
        self._thread = threading.Thread(
            target=self._receive_frames,
            daemon=True,
            name='quest3-frame-reader',
        )
        self._thread.start()

    def read(
        self,
    ) -> QuestFrame:
        '''첫 xr_frame을 기다린 뒤 가장 최근 QuestFrame을 반환한다.'''
        with self._condition:
            frame_received: bool = self._condition.wait_for(
                lambda: self._latest_frame is not None,
                timeout=self._open_timeout_s,
            )
            if not frame_received or self._latest_frame is None:
                raise RuntimeError('Quest relay did not provide an xr_frame')
            return self._latest_frame

    def diagnostics(
        self,
    ) -> dict[str, object]:
        '''수신 thread, 유효 frame 수, tracking 누락 수와 최신 frame age를 반환한다.'''
        with self._condition:
            latest_frame: QuestFrame | None = self._latest_frame
            received_frame_count: int = self._received_frame_count
            skipped_tracking_frame_count: int = self._skipped_tracking_frame_count
        thread: threading.Thread | None = self._thread
        latest_frame_age_s: float | None = (
            None if latest_frame is None else self._clock() - latest_frame.timestamp_s
        )
        return {
            'receiver_thread_alive': thread is not None and thread.is_alive(),
            'received_frame_count': received_frame_count,
            'skipped_tracking_frame_count': skipped_tracking_frame_count,
            'latest_frame_age_s': latest_frame_age_s,
        }

    def close(
        self,
    ) -> None:
        '''Background receiver와 Quest relay connection을 종료한다.'''
        self._stop_event.set()
        if self._connection is not None:
            self._connection.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._connection = None
        self._thread = None
        self._latest_frame = None


class Quest3ActionProducer:
    '''Clutch-relative mapping과 stale-frame safety를 적용한다.'''

    def __init__(
        self,
        reader: QuestFrameReader,
        retargeter: QuestRetargeter,
        clock: Callable[[], float] = time.monotonic,
        max_frame_age_s: float = 0.25,
    ) -> None:
        '''Quest reader, YAM retargeter와 stale-frame 제한을 저장한다.'''
        if max_frame_age_s <= 0.0:
            raise ValueError('max_frame_age_s must be positive')
        self._reader: QuestFrameReader = reader
        self._retargeter: QuestRetargeter = retargeter
        self._clock: Callable[[], float] = clock
        self._max_frame_age_s: float = max_frame_age_s
        self._engaged: bool = False
        self._reengage_required: bool = False
        self.last_hold_reason: str | None = None
        self._hold_reason_counts: dict[str, int] = {}
        self._clutch_control_step_count: int = 0
        self._changed_action_step_count: int = 0
        self._max_action_delta: float = 0.0

    def connect(
        self,
    ) -> None:
        '''Quest Browser WebXR relay에 연결하고 첫 유효 QuestFrame을 확인한다.'''
        self._reader.connect()
        self._reader.read()

    def reset(
        self,
        observation: RobotObservation,
    ) -> None:
        '''Episode마다 clutch 상태와 optional retargeter 진단값을 초기화한다.'''
        reset_diagnostics: object = getattr(self._retargeter, 'reset_diagnostics', None)
        if callable(reset_diagnostics):
            reset_diagnostics()
        self._engaged = False
        self._reengage_required = False
        self.last_hold_reason = None
        self._hold_reason_counts = {}
        self._clutch_control_step_count = 0
        self._changed_action_step_count = 0
        self._max_action_delta = 0.0

    def _hold(
        self,
        observation: RobotObservation,
        reason: str,
    ) -> RobotAction:
        '''현재 measured shape `(S,)` state를 hold action으로 반환한다.'''
        self.last_hold_reason = reason
        self._hold_reason_counts[reason] = self._hold_reason_counts.get(reason, 0) + 1
        return RobotAction(values=observation.state)

    def next_action(
        self,
        observation: RobotObservation,
    ) -> RobotAction:
        '''최신 Quest frame을 clutch-relative YAM action 또는 hold action으로 변환한다.'''
        frame: QuestFrame = self._reader.read()
        frame_age_s: float = self._clock() - frame.timestamp_s
        if frame_age_s > self._max_frame_age_s:
            self._engaged = False
            self._reengage_required = True
            return self._hold(observation, 'stale Quest frame')
        if not frame.clutch_pressed:
            self._engaged = False
            self._reengage_required = False
            return self._hold(observation, 'clutch released')
        if self._reengage_required:
            return self._hold(observation, 'release clutch before re-engaging')
        if not self._engaged:
            self._retargeter.rebase(frame, observation)
            self._engaged = True
        self.last_hold_reason = None
        self._clutch_control_step_count += 1
        action: RobotAction = self._retargeter.retarget(frame, observation)
        action_delta: float = max(
            abs(action_value - state_value)
            for action_value, state_value in zip(action.values, observation.state, strict=True)
        )
        self._max_action_delta = max(self._max_action_delta, action_delta)
        if action_delta > 1e-6:
            self._changed_action_step_count += 1
        return action

    def diagnostics(
        self,
    ) -> dict[str, object]:
        '''현재 episode의 clutch, reader와 optional IK 진단값을 반환한다.'''
        diagnostics_method: object = getattr(self._retargeter, 'diagnostics', None)
        retargeter_diagnostics: object = diagnostics_method() if callable(diagnostics_method) else {}
        if not isinstance(retargeter_diagnostics, Mapping):
            raise TypeError('Quest retargeter diagnostics must be a mapping')
        return {
            'clutch_control_step_count': self._clutch_control_step_count,
            'changed_action_step_count': self._changed_action_step_count,
            'max_action_delta': self._max_action_delta,
            'hold_reason_counts': dict(self._hold_reason_counts),
            'last_hold_reason': self.last_hold_reason,
            'reader': self._reader.diagnostics(),
            'retargeter': dict(retargeter_diagnostics),
        }

    def close(
        self,
    ) -> None:
        '''Quest relay 연결을 해제하고 clutch 상태를 초기화한다.'''
        self._reader.close()
        self._engaged = False
        self._reengage_required = False
