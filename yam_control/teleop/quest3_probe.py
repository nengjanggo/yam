'''Quest 3 xr_frame stream을 robot 연결 없이 검증하는 command-line probe이다.'''

from __future__ import annotations

import json
import ssl
import sys
import time
from collections.abc import Mapping
from typing import cast

from .quest3 import create_websocket_ssl_context, parse_quest_frame_payload
from ..config import QuestControllerHand
from ..types import QuestFrame


def _summarize_frames(
    frames: list[QuestFrame],
    advertised_camera_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    '''QuestFrame 입력과 optional relay camera 광고 상태를 요약한다.'''
    if not frames:
        raise ValueError('at least one QuestFrame is required')
    controller_motion_span_m: list[float] = []
    hmd_motion_span_m: list[float] = []
    axis_index: int
    for axis_index in range(3):
        controller_axis: list[float] = [frame.controller_pose.position[axis_index] for frame in frames]
        hmd_axis: list[float] = [frame.hmd_pose.position[axis_index] for frame in frames]
        controller_motion_span_m.append(max(controller_axis) - min(controller_axis))
        hmd_motion_span_m.append(max(hmd_axis) - min(hmd_axis))
    trigger_values: list[float] = [frame.trigger for frame in frames]
    clutch_pressed_count: int = sum(frame.clutch_pressed for frame in frames)
    checks: dict[str, bool] = {
        'controller_motion_seen': max(controller_motion_span_m) >= 0.01,
        'trigger_motion_seen': max(trigger_values) - min(trigger_values) >= 0.1,
        'clutch_pressed_seen': clutch_pressed_count > 0,
        'clutch_released_seen': clutch_pressed_count < len(frames),
    }
    if advertised_camera_ids is not None:
        checks['mujoco_camera_advertised'] = 'top' in advertised_camera_ids
    return {
        'sample_count': len(frames),
        'controller_motion_span_m': controller_motion_span_m,
        'hmd_motion_span_m': hmd_motion_span_m,
        'trigger_range': [min(trigger_values), max(trigger_values)],
        'clutch_pressed_count': clutch_pressed_count,
        'advertised_camera_ids': advertised_camera_ids,
        'checks': checks,
        'all_checks_passed': all(checks.values()),
    }


def probe_quest3(
    websocket_url: str,
    controller_hand: QuestControllerHand,
    sample_count: int,
) -> dict[str, object]:
    '''Relay camera 목록과 sample_count개의 xr_frame을 수신해 진단 요약을 반환한다.'''
    from websockets.sync.client import ClientConnection, connect

    if not websocket_url.startswith(('ws://', 'wss://')):
        raise ValueError('websocket_url must start with ws:// or wss://')
    if sample_count <= 0:
        raise ValueError('sample_count must be positive')
    frames: list[QuestFrame] = []
    advertised_camera_ids: tuple[str, ...] = ()
    ssl_context: ssl.SSLContext | None = create_websocket_ssl_context(websocket_url)
    with connect(
        websocket_url,
        ssl=ssl_context,
        open_timeout=10.0,
    ) as websocket:
        connection: ClientConnection = websocket
        while len(frames) < sample_count:
            raw_message: str | bytes = connection.recv(timeout=10.0)
            message: str = raw_message.decode('utf-8') if isinstance(raw_message, bytes) else raw_message
            payload: object = json.loads(message)
            if isinstance(payload, Mapping) and payload.get('type') == 'camera_list':
                cameras: object = payload.get('cameras')
                if isinstance(cameras, list):
                    advertised_camera_ids = tuple(
                        str(camera['id'])
                        for camera in cameras
                        if isinstance(camera, Mapping) and 'id' in camera
                    )
            if isinstance(payload, dict) and payload.get('type') == 'xr_frame':
                frame: QuestFrame = parse_quest_frame_payload(
                    payload=payload,
                    controller_hand=controller_hand,
                    received_timestamp_s=time.monotonic(),
                )
                frames.append(frame)
    return _summarize_frames(
        frames=frames,
        advertised_camera_ids=advertised_camera_ids,
    )


def main(
) -> None:
    '''Command-line argument로 Quest 3 probe를 실행하고 JSON summary를 stdout으로 반환한다.'''
    websocket_url: str = sys.argv[1] if len(sys.argv) > 1 else 'wss://127.0.0.1:8443/ws'
    controller_hand_value: str = sys.argv[2] if len(sys.argv) > 2 else 'right'
    if controller_hand_value not in ('left', 'right'):
        raise ValueError(f'unsupported controller_hand: {controller_hand_value}')
    controller_hand: QuestControllerHand = cast(QuestControllerHand, controller_hand_value)
    sample_count: int = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    summary: dict[str, object] = probe_quest3(
        websocket_url=websocket_url,
        controller_hand=controller_hand,
        sample_count=sample_count,
    )
    sys.stdout.write(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
