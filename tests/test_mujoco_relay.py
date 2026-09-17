'''YAM이 소유하는 MuJoCo WebSocket/WebRTC relay를 검증한다.'''

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import cast

import av
import numpy as np
from fastapi import FastAPI, WebSocket
from PIL import Image

from yam_control.teleop.mujoco_relay import (
    CAMERA_ID,
    WEB_CLIENT_DIRECTORY,
    MujocoCameraTrack,
    MujocoFrameReader,
    MujocoRelay,
    RelayClient,
    create_app,
)


class FakeWebSocket:
    '''Network 없이 relay가 전송한 WebSocket text를 저장한다.'''

    def __init__(
        self,
    ) -> None:
        '''전송 message list를 초기화한다.'''
        self.messages: list[str] = []

    async def send_text(
        self,
        message: str,
    ) -> None:
        '''전송된 text message를 저장한다.'''
        self.messages.append(message)


def _write_frame(
    frame_path: Path,
    color: tuple[int, int, int],
) -> None:
    '''단색 shape `(4, 6, 3)` RGB test frame을 저장한다.'''
    frame: np.ndarray = np.full((4, 6, 3), color, dtype=np.uint8)
    temporary_path: Path = frame_path.with_suffix('.tmp')
    Image.fromarray(frame).save(temporary_path, format='PNG')
    os.replace(temporary_path, frame_path)


def test_frame_reader_refreshes_changed_atomic_image(
    tmp_path: Path,
) -> None:
    '''Image file 변경이 RGB frame cache에 반영되는지 검증한다.'''
    frame_path: Path = tmp_path / 'frame.jpg'
    _write_frame(frame_path, (255, 0, 0))
    reader: MujocoFrameReader = MujocoFrameReader(
        frame_path=frame_path,
        width=6,
        height=4,
    )

    first_frame: np.ndarray | None = reader.latest()
    _write_frame(frame_path, (0, 255, 0))
    second_frame: np.ndarray | None = reader.latest()

    assert first_frame is not None
    assert second_frame is not None
    assert first_frame.shape == (4, 6, 3)
    assert tuple(first_frame[0, 0]) == (255, 0, 0)
    assert tuple(second_frame[0, 0]) == (0, 255, 0)


def test_camera_track_returns_configured_rgb_frame(
    tmp_path: Path,
) -> None:
    '''MuJoCo camera track이 RGB frame shape과 RTP time base를 유지하는지 검증한다.'''
    frame_path: Path = tmp_path / 'frame.jpg'
    _write_frame(frame_path, (10, 20, 30))
    reader: MujocoFrameReader = MujocoFrameReader(
        frame_path=frame_path,
        width=6,
        height=4,
    )
    track: MujocoCameraTrack = MujocoCameraTrack(reader=reader, fps=15)

    video_frame: av.VideoFrame = asyncio.run(track.recv())
    rgb_frame: np.ndarray = video_frame.to_ndarray(format='rgb24')
    track.stop()

    assert rgb_frame.shape == (4, 6, 3)
    assert tuple(rgb_frame[0, 0]) == (10, 20, 30)
    assert video_frame.time_base is not None


def test_relay_echoes_ping_and_toggles_camera(
    tmp_path: Path,
) -> None:
    '''Ping echo와 camera toggle WebSocket protocol을 검증한다.'''
    frame_path: Path = tmp_path / 'frame.jpg'
    _write_frame(frame_path, (10, 20, 30))
    reader: MujocoFrameReader = MujocoFrameReader(
        frame_path=frame_path,
        width=6,
        height=4,
    )
    relay: MujocoRelay = MujocoRelay(frame_reader=reader, fps=15)
    fake_websocket: FakeWebSocket = FakeWebSocket()
    client: RelayClient = RelayClient(
        websocket=cast(WebSocket, fake_websocket),
        camera_track=MujocoCameraTrack(reader=reader, fps=15),
    )

    async def run_protocol_check(
    ) -> None:
        '''Ping과 camera toggle message를 순서대로 처리한다.'''
        await relay.handle_message(
            client,
            json.dumps({'type': 'ping', 't_client': 12.5}),
        )
        await relay.handle_message(
            client,
            json.dumps(
                {
                    'type': 'camera_toggle',
                    'camera_id': CAMERA_ID,
                    'enabled': False,
                }
            ),
        )
        assert client.camera_track is not None
        assert not client.camera_track.enabled
        await relay.close_peer_connection(client)

    asyncio.run(run_protocol_check())
    response: object = json.loads(fake_websocket.messages[0])

    assert isinstance(response, dict)
    assert response['echo'] == {'type': 'ping', 't_client': 12.5}
    assert client.camera_track is None


def test_relay_broadcasts_xr_frame_to_other_client(
    tmp_path: Path,
) -> None:
    '''xr_frame이 sender를 제외한 teleoperation client로 전달되는지 검증한다.'''
    reader: MujocoFrameReader = MujocoFrameReader(
        frame_path=tmp_path / 'frame.jpg',
        width=6,
        height=4,
    )
    relay: MujocoRelay = MujocoRelay(frame_reader=reader, fps=15)
    sender_websocket: FakeWebSocket = FakeWebSocket()
    recipient_websocket: FakeWebSocket = FakeWebSocket()
    sender: RelayClient = RelayClient(
        websocket=cast(WebSocket, sender_websocket),
    )
    recipient: RelayClient = RelayClient(
        websocket=cast(WebSocket, recipient_websocket),
    )
    relay.clients = {
        cast(WebSocket, sender_websocket): sender,
        cast(WebSocket, recipient_websocket): recipient,
    }
    raw_message: str = json.dumps(
        {
            'type': 'xr_frame',
            'controllers': {},
            'viewer': None,
        }
    )

    asyncio.run(relay.handle_message(sender, raw_message))

    assert sender_websocket.messages == []
    assert recipient_websocket.messages == [raw_message]


def test_app_owns_websocket_and_static_routes(
    tmp_path: Path,
) -> None:
    '''Local FastAPI application이 relay와 web asset route를 제공하는지 검증한다.'''
    application: FastAPI = create_app(
        frame_path=tmp_path / 'frame.jpg',
        width=640,
        height=480,
        fps=15,
    )
    route_paths: set[str] = {
        route.path
        for route in application.routes
    }

    assert {'/', '/ws', '/static'} <= route_paths
    assert isinstance(application.state.relay, MujocoRelay)
    index_html: str = (WEB_CLIENT_DIRECTORY / 'index.html').read_text()
    assert '/static/quest_client.js' in index_html
    assert (WEB_CLIENT_DIRECTORY / 'quest_input.js').is_file()
    assert (WEB_CLIENT_DIRECTORY / 'quest_client.js').is_file()
