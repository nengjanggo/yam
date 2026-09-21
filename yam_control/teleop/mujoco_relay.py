'''Quest 3 WebSocket relay와 execution target별 WebRTC video stream을 제공한다.'''

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import AsyncIterator, cast

import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from numpy.typing import NDArray
from PIL import Image

DEFAULT_FRAME_PATH: str = '/tmp/yam-mujoco-frame.jpg'
CAMERA_ID: str = 'top'
CAMERA_LABEL: str = 'MuJoCo'
WRIST_CAMERA_ID: str = 'wrist'
WRIST_CAMERA_LABEL: str = 'Wrist Camera'
VIDEO_CLOCK_RATE: int = 90_000
WEB_CLIENT_DIRECTORY: Path = Path(__file__).with_name('web')
RELAY_MESSAGE_TYPES: frozenset[str] = frozenset(
    {
        'haptic_calibrate',
        'haptic_calibrate_result',
        'ik_state',
        'request_settings',
        'xr_frame',
    }
)


class MujocoFrameReader:
    '''Atomic JPEG file에서 최신 MuJoCo RGB frame을 읽는다.'''

    def __init__(
        self,
        frame_path: Path,
        width: int,
        height: int,
    ) -> None:
        '''JPEG path와 fallback frame shape을 저장한다.'''
        self.frame_path: Path = frame_path
        self.width: int = width
        self.height: int = height
        self._file_identity: tuple[int, int, int] | None = None
        self._frame: NDArray[np.uint8] | None = None

    def latest(
        self,
    ) -> NDArray[np.uint8] | None:
        '''JPEG가 갱신됐을 때만 shape `(height, width, 3)` RGB frame을 반환한다.'''
        if not self.frame_path.exists():
            return self._frame
        file_stat: os.stat_result = self.frame_path.stat()
        file_identity: tuple[int, int, int] = (
            file_stat.st_ino,
            file_stat.st_mtime_ns,
            file_stat.st_size,
        )
        if file_identity == self._file_identity:
            return self._frame
        with Image.open(self.frame_path) as image:
            # Shape `(height, width, 3)` RGB image를 독립적인 uint8 array로 복사
            frame: NDArray[np.uint8] = np.asarray(
                image.convert('RGB').resize((self.width, self.height)),
                dtype=np.uint8,
            ).copy()
        self._file_identity = file_identity
        self._frame = frame
        return frame


class MujocoCameraTrack(VideoStreamTrack):
    '''MuJoCo RGB frame을 지정한 FPS의 WebRTC video track으로 제공한다.'''

    kind: str = 'video'

    def __init__(
        self,
        reader: MujocoFrameReader,
        fps: int,
    ) -> None:
        '''Frame reader와 WebRTC timestamp state를 초기화한다.'''
        super().__init__()
        if fps <= 0:
            raise ValueError('fps must be positive')
        self.reader: MujocoFrameReader = reader
        self.fps: int = fps
        self.enabled: bool = True
        self._frame_interval: int = round(VIDEO_CLOCK_RATE / fps)
        self._timestamp: int = 0
        self._started_at_s: float | None = None
        self._last_frame: NDArray[np.uint8] = np.zeros(
            (reader.height, reader.width, 3),
            dtype=np.uint8,
        )

    async def _next_timestamp(
        self,
    ) -> tuple[int, Fraction]:
        '''설정한 FPS에 맞는 RTP timestamp와 time base를 반환한다.'''
        current_time_s: float = time.monotonic()
        if self._started_at_s is None:
            self._started_at_s = current_time_s
        else:
            self._timestamp += self._frame_interval
            target_time_s: float = self._started_at_s + self._timestamp / VIDEO_CLOCK_RATE
            await asyncio.sleep(max(0.0, target_time_s - current_time_s))
        return self._timestamp, Fraction(1, VIDEO_CLOCK_RATE)

    async def recv(
        self,
    ) -> av.VideoFrame:
        '''최신 RGB frame을 timestamp가 지정된 av.VideoFrame으로 반환한다.'''
        timestamp: int
        time_base: Fraction
        timestamp, time_base = await self._next_timestamp()
        latest_frame: NDArray[np.uint8] | None = self.reader.latest()
        if self.enabled and latest_frame is not None:
            self._last_frame = latest_frame
        # Shape `(height, width, 3)` RGB array를 WebRTC video frame으로 변환
        video_frame: av.VideoFrame = av.VideoFrame.from_ndarray(
            self._last_frame,
            format='rgb24',
        )
        video_frame.pts = timestamp
        video_frame.time_base = time_base
        return video_frame


@dataclass
class RelayClient:
    '''한 WebSocket client의 WebRTC resource를 저장한다.'''

    websocket: WebSocket
    peer_connection: RTCPeerConnection | None = None
    camera_track: MujocoCameraTrack | None = None


class MujocoRelay:
    '''Quest browser와 teleoperation process 사이 message와 선택된 video를 relay한다.'''

    def __init__(
        self,
        frame_reader: MujocoFrameReader,
        fps: int,
    ) -> None:
        '''Frame reader와 연결 client registry를 video 비활성 상태로 초기화한다.'''
        self.frame_reader: MujocoFrameReader = frame_reader
        self.fps: int = fps
        self.clients: dict[WebSocket, RelayClient] = {}
        self.camera_id: str | None = None
        self.camera_label: str | None = None
        self.camera_layout: str | None = None

    def camera_descriptions(
        self,
    ) -> list[dict[str, str]]:
        '''현재 활성화된 Quest video source description을 반환한다.'''
        if self.camera_id is None or self.camera_label is None or self.camera_layout is None:
            return []
        return [
            {
                'id': self.camera_id,
                'label': self.camera_label,
                'layout': self.camera_layout,
            }
        ]

    async def configure_video(
        self,
        message: Mapping[str, object],
        sender: WebSocket,
    ) -> None:
        '''Runtime config로 video source를 교체하고 Quest client에 camera list를 알린다.'''
        config_value: object = message.get('config')
        config: Mapping[str, object] = config_value if isinstance(config_value, Mapping) else {}
        video_value: object = config.get('video')
        video: Mapping[str, object] = video_value if isinstance(video_value, Mapping) else {}
        enabled: bool = bool(video.get('enabled', False))
        camera_id: object = video.get('camera_id')
        camera_label: object = video.get('label')
        camera_layout: object = video.get('layout')
        if enabled and not all(
            isinstance(value, str) and value
            for value in (camera_id, camera_label, camera_layout)
        ):
            raise ValueError('enabled video config requires camera_id, label and layout')
        if enabled and camera_layout not in ('mujoco', 'wrist'):
            raise ValueError(f'unsupported camera layout: {camera_layout}')
        self.camera_id = str(camera_id) if enabled else None
        self.camera_label = str(camera_label) if enabled else None
        self.camera_layout = str(camera_layout) if enabled else None
        clients: tuple[RelayClient, ...] = tuple(
            client
            for websocket, client in self.clients.items()
            if websocket is not sender
        )
        await asyncio.gather(
            *(self.close_peer_connection(client) for client in clients)
        )
        await self.broadcast(
            json.dumps(
                {
                    'type': 'camera_list',
                    'cameras': self.camera_descriptions(),
                }
            ),
            sender,
        )

    async def broadcast(
        self,
        raw_message: str,
        sender: WebSocket,
    ) -> None:
        '''Message를 sender를 제외한 모든 WebSocket client로 전달한다.'''
        recipients: tuple[WebSocket, ...] = tuple(
            websocket
            for websocket in self.clients
            if websocket is not sender
        )
        await asyncio.gather(
            *(websocket.send_text(raw_message) for websocket in recipients),
            return_exceptions=True,
        )

    async def close_peer_connection(
        self,
        client: RelayClient,
    ) -> None:
        '''Client의 WebRTC track과 peer connection을 종료한다.'''
        peer_connection: RTCPeerConnection | None = client.peer_connection
        camera_track: MujocoCameraTrack | None = client.camera_track
        client.peer_connection = None
        client.camera_track = None
        if camera_track is not None:
            camera_track.stop()
        if peer_connection is not None:
            await peer_connection.close()

    def _prefer_h264(
        self,
        peer_connection: RTCPeerConnection,
    ) -> None:
        '''Quest hardware decode를 위해 가능한 경우 H.264 codec을 우선한다.'''
        video_codecs: list[object] = [
            codec
            for codec in RTCRtpSender.getCapabilities('video').codecs
            if codec.mimeType == 'video/H264'
        ]
        if not video_codecs:
            return
        for transceiver in peer_connection.getTransceivers():
            if transceiver.kind == 'video':
                transceiver.setCodecPreferences(video_codecs)  # type: ignore[arg-type]

    async def start_webrtc(
        self,
        client: RelayClient,
        message: Mapping[str, object],
    ) -> None:
        '''활성 video track을 추가하고 WebRTC offer를 WebSocket으로 전송한다.'''
        await self.close_peer_connection(client)
        if self.camera_id is None:
            return
        enabled_cameras_value: object = message.get('enabled_cameras')
        enabled_cameras: set[str] = (
            {self.camera_id}
            if not isinstance(enabled_cameras_value, list)
            else {str(camera_id) for camera_id in enabled_cameras_value}
        )
        peer_connection: RTCPeerConnection = RTCPeerConnection()
        camera_track: MujocoCameraTrack = MujocoCameraTrack(
            reader=self.frame_reader,
            fps=self.fps,
        )
        camera_track.enabled = self.camera_id in enabled_cameras
        sender: RTCRtpSender = peer_connection.addTrack(camera_track)
        # Quest client가 MediaStream id로 camera slot을 연결하므로 현재 camera id를 지정
        sender._stream_id = self.camera_id  # type: ignore[attr-defined]
        client.peer_connection = peer_connection
        client.camera_track = camera_track
        self._prefer_h264(peer_connection)

        @peer_connection.on('iceconnectionstatechange')
        async def close_failed_connection(
        ) -> None:
            '''실패하거나 종료된 WebRTC connection resource를 해제한다.'''
            if peer_connection.iceConnectionState in ('failed', 'closed'):
                await self.close_peer_connection(client)

        offer: RTCSessionDescription = await peer_connection.createOffer()
        await peer_connection.setLocalDescription(offer)
        local_description: RTCSessionDescription | None = peer_connection.localDescription
        if local_description is None:
            raise RuntimeError('WebRTC local description was not created')
        await client.websocket.send_text(
            json.dumps(
                {
                    'type': 'webrtc_offer',
                    'sdp': local_description.sdp,
                    'sdp_type': local_description.type,
                    'cameras': self.camera_descriptions(),
                }
            )
        )

    async def accept_webrtc_answer(
        self,
        client: RelayClient,
        message: Mapping[str, object],
    ) -> None:
        '''Quest browser의 WebRTC answer를 active peer connection에 적용한다.'''
        if client.peer_connection is None:
            return
        sdp: object = message.get('sdp')
        sdp_type: object = message.get('sdp_type')
        if not isinstance(sdp, str) or not isinstance(sdp_type, str):
            raise ValueError('webrtc_answer requires string sdp and sdp_type')
        await client.peer_connection.setRemoteDescription(
            RTCSessionDescription(
                sdp=sdp,
                type=sdp_type,
            )
        )

    async def handle_message(
        self,
        client: RelayClient,
        raw_message: str,
    ) -> None:
        '''WebSocket JSON message를 broadcast 또는 WebRTC handler로 전달한다.'''
        message: object = json.loads(raw_message)
        if not isinstance(message, Mapping):
            raise ValueError('WebSocket message must be a JSON object')
        message_type: object = message.get('type')
        if not isinstance(message_type, str):
            raise ValueError('WebSocket message type must be a string')
        if message_type == 'config_update':
            await self.configure_video(message, client.websocket)
            return
        if message_type in RELAY_MESSAGE_TYPES:
            await self.broadcast(raw_message, client.websocket)
            return
        if message_type == 'webrtc_request':
            await self.start_webrtc(client, message)
            return
        if message_type == 'webrtc_answer':
            await self.accept_webrtc_answer(client, message)
            return
        if message_type == 'camera_toggle':
            if client.camera_track is not None and message.get('camera_id') == self.camera_id:
                client.camera_track.enabled = bool(message.get('enabled', True))
            return
        if message_type in ('ice_candidate', 'latency_report'):
            return
        await client.websocket.send_text(
            json.dumps(
                {
                    'echo': message,
                    'server_time': time.time(),
                }
            )
        )

    async def serve_websocket(
        self,
        websocket: WebSocket,
    ) -> None:
        '''WebSocket client를 등록하고 disconnect까지 message를 처리한다.'''
        await websocket.accept()
        client: RelayClient = RelayClient(websocket=websocket)
        self.clients[websocket] = client
        await websocket.send_text(
            json.dumps(
                {
                    'type': 'camera_list',
                    'cameras': self.camera_descriptions(),
                }
            )
        )
        try:
            while True:
                event: dict[str, object] = await websocket.receive()
                if event.get('type') == 'websocket.disconnect':
                    break
                raw_message: object = event.get('text')
                if isinstance(raw_message, str):
                    await self.handle_message(client, raw_message)
        finally:
            await self.close_peer_connection(client)
            self.clients.pop(websocket, None)

    async def close(
        self,
    ) -> None:
        '''모든 WebRTC peer connection을 종료한다.'''
        await asyncio.gather(
            *(self.close_peer_connection(client) for client in tuple(self.clients.values()))
        )


def create_app(
    frame_path: Path,
    width: int,
    height: int,
    fps: int,
) -> FastAPI:
    '''MuJoCo camera configuration으로 FastAPI relay application을 생성한다.'''
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError('width, height and fps must be positive')
    frame_reader: MujocoFrameReader = MujocoFrameReader(
        frame_path=frame_path,
        width=width,
        height=height,
    )
    relay: MujocoRelay = MujocoRelay(
        frame_reader=frame_reader,
        fps=fps,
    )

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        '''Application 종료 시 WebRTC resource를 해제한다.'''
        del application
        yield
        await relay.close()

    application: FastAPI = FastAPI(lifespan=lifespan)
    application.state.relay = relay

    @application.middleware('http')
    async def disable_web_cache(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        '''Quest browser가 local web asset을 매번 다시 읽도록 cache를 비활성화한다.'''
        response: Response = await call_next(request)
        if request.url.path == '/' or request.url.path.startswith('/static/'):
            response.headers['Cache-Control'] = 'no-store, max-age=0, must-revalidate'
        return response

    @application.get('/')
    async def index(
    ) -> FileResponse:
        '''YAM Quest Web UI entrypoint를 반환한다.'''
        return FileResponse(WEB_CLIENT_DIRECTORY / 'index.html', media_type='text/html')

    @application.websocket('/ws')
    async def websocket_endpoint(
        websocket: WebSocket,
    ) -> None:
        '''Quest browser와 teleoperation process의 WebSocket endpoint를 제공한다.'''
        await relay.serve_websocket(websocket)

    application.mount(
        '/static',
        StaticFiles(directory=WEB_CLIENT_DIRECTORY),
        name='static',
    )
    return application


FRAME_PATH: Path = Path(os.environ.get('YAM_MUJOCO_FRAME_PATH', DEFAULT_FRAME_PATH))
FRAME_WIDTH: int = int(os.environ.get('CAM_WIDTH', '640'))
FRAME_HEIGHT: int = int(os.environ.get('CAM_HEIGHT', '480'))
FRAME_FPS: int = int(os.environ.get('CAM_FPS', '15'))
app: FastAPI = create_app(
    frame_path=FRAME_PATH,
    width=FRAME_WIDTH,
    height=FRAME_HEIGHT,
    fps=FRAME_FPS,
)


def main(
) -> None:
    '''Command-line TLS option으로 MuJoCo relay application을 실행한다.'''
    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser()
    argument_parser.add_argument('--host', default='127.0.0.1')
    argument_parser.add_argument('--port', type=int, default=8443)
    argument_parser.add_argument('--ssl-keyfile', default=None)
    argument_parser.add_argument('--ssl-certfile', default=None)
    arguments: argparse.Namespace = argument_parser.parse_args()
    import uvicorn

    uvicorn.run(
        app,
        host=cast(str, arguments.host),
        port=cast(int, arguments.port),
        ssl_keyfile=cast(str | None, arguments.ssl_keyfile),
        ssl_certfile=cast(str | None, arguments.ssl_certfile),
    )


if __name__ == '__main__':
    main()
