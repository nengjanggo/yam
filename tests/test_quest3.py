'''Quest 3 xr_frame parsing, input probe와 clutch safety 동작을 검증한다.'''

from __future__ import annotations

import unittest
import json
import ssl
from collections.abc import Iterator

from yam_control.teleop.quest3 import (
    Quest3ActionProducer,
    WebSocketConnection,
    WebSocketQuestFrameReader,
    parse_quest_frame_payload,
)
from yam_control.teleop.quest3_probe import _summarize_frames
from yam_control.types import EpisodeOutcome, QuestFrame, QuestPose, RobotAction, RobotObservation


class FakeQuestReader:
    '''외부 Quest 연결 없이 지정된 frame을 순서대로 반환한다.'''

    def __init__(
        self,
        frames: list[QuestFrame],
    ) -> None:
        '''읽을 frame queue와 현재 index를 저장한다.'''
        self._frames: list[QuestFrame] = frames
        self._index: int = 0
        self.connected: bool = False
        self.read_count: int = 0

    def connect(
        self,
    ) -> None:
        '''Fake reader는 외부 resource를 연결하지 않는다.'''
        self.connected = True

    def read(
        self,
    ) -> QuestFrame:
        '''Queue의 다음 QuestFrame을 반환한다.'''
        frame: QuestFrame = self._frames[self._index]
        self._index += 1
        self.read_count += 1
        return frame

    def close(
        self,
    ) -> None:
        '''Fake reader는 해제할 resource가 없다.'''

    def diagnostics(
        self,
    ) -> dict[str, object]:
        '''Fake reader는 빈 수신 진단값을 반환한다.'''
        return {}


class FakeRetargeter:
    '''Rebase count를 기록하고 고정 YAM action을 반환한다.'''

    def __init__(
        self,
    ) -> None:
        '''Rebase count를 0으로 초기화한다.'''
        self.rebase_count: int = 0

    def rebase(
        self,
        frame: QuestFrame,
        observation: RobotObservation,
    ) -> None:
        '''Clutch engage 기준 pose 저장을 count로 대체한다.'''
        del frame
        del observation
        self.rebase_count += 1

    def retarget(
        self,
        frame: QuestFrame,
        observation: RobotObservation,
    ) -> RobotAction:
        '''Test용 shape `(7,)` action을 반환한다.'''
        del frame
        del observation
        return RobotAction(values=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))


class FakeWebSocketConnection:
    '''Network 없이 WebSocket message iterator를 제공한다.'''

    def __init__(
        self,
        messages: list[str],
    ) -> None:
        '''전달할 message와 close 상태를 저장한다.'''
        self._messages: list[str] = messages
        self.sent_messages: list[str] = []
        self.closed: bool = False

    def __iter__(
        self,
    ) -> Iterator[str | bytes]:
        '''저장된 message iterator를 반환한다.'''
        return iter(self._messages)

    def close(
        self,
    ) -> None:
        '''Close 상태를 기록한다.'''
        self.closed = True

    def send(
        self,
        message: str,
    ) -> None:
        '''Reader가 relay로 전송한 text message를 기록한다.'''
        self.sent_messages.append(message)


class FakeWebSocketConnector:
    '''WebSocket URL을 기록하고 fake connection을 반환한다.'''

    def __init__(
        self,
        connection: FakeWebSocketConnection,
    ) -> None:
        '''반환할 fake connection을 저장한다.'''
        self._connection: FakeWebSocketConnection = connection
        self.websocket_url: str | None = None

    def __call__(
        self,
        websocket_url: str,
        ssl_context: ssl.SSLContext | None,
        open_timeout_s: float,
    ) -> WebSocketConnection:
        '''Connector 인자를 검증하고 fake connection을 반환한다.'''
        del ssl_context
        if open_timeout_s <= 0.0:
            raise ValueError('open_timeout_s must be positive')
        self.websocket_url = websocket_url
        return self._connection


def _frame(
    hmd_x_m: float,
    clutch_pressed: bool,
) -> QuestFrame:
    '''지정된 HMD x 위치와 clutch 상태의 QuestFrame을 생성한다.'''
    controller_pose: QuestPose = QuestPose(
        position=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    hmd_pose: QuestPose = QuestPose(
        position=(hmd_x_m, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    return QuestFrame(
        controller_pose=controller_pose,
        hmd_pose=hmd_pose,
        trigger=0.0,
        clutch_pressed=clutch_pressed,
        timestamp_s=10.0,
    )


def _button_frame(
    primary_pressed: bool,
    secondary_pressed: bool,
    clutch_pressed: bool = True,
) -> QuestFrame:
    '''지정된 A/X, B/Y button 상태의 QuestFrame을 생성한다.'''
    frame: QuestFrame = _frame(hmd_x_m=0.0, clutch_pressed=clutch_pressed)
    return QuestFrame(
        controller_pose=frame.controller_pose,
        hmd_pose=frame.hmd_pose,
        trigger=frame.trigger,
        clutch_pressed=frame.clutch_pressed,
        timestamp_s=frame.timestamp_s,
        primary_button_pressed=primary_pressed,
        secondary_button_pressed=secondary_pressed,
    )


def _run_button_frames(
    frames: list[QuestFrame],
) -> list[EpisodeOutcome | None]:
    '''Reset 후 frame마다 next_action을 호출하고 episode_outcome 변화를 반환한다.'''
    producer: Quest3ActionProducer = Quest3ActionProducer(
        reader=FakeQuestReader(frames=frames),
        retargeter=FakeRetargeter(),
        clock=_clock,
    )
    observation: RobotObservation = RobotObservation(state=(0.0,) * 7)
    producer.reset(observation)
    outcomes: list[EpisodeOutcome | None] = []
    frame: QuestFrame
    for frame in frames:
        del frame
        producer.next_action(observation)
        outcomes.append(producer.episode_outcome)
    return outcomes


def _clock(
) -> float:
    '''Frame timestamp와 동일한 monotonic test 시간을 반환한다.'''
    return 10.0


class Quest3ActionProducerTest(unittest.TestCase):
    '''착용한 Quest 3 운용의 clutch와 stale-frame safety를 검증한다.'''

    def test_retargeter_diagnostics_reset_per_episode(
        self,
    ) -> None:
        '''IK 진단값이 producer 출력에 전달되고 clutch 재진입이 아닌 episode reset에 초기화되는지 검증한다.'''
        class DiagnosticRetargeter(FakeRetargeter):
            '''Fake retargeter에 episode 진단값을 추가한다.'''

            def reset_diagnostics(
                self,
            ) -> None:
                '''이전 episode의 rebase 횟수를 초기화한다.'''
                self.rebase_count = 0

            def diagnostics(
                self,
            ) -> dict[str, object]:
                '''현재 episode의 rebase 횟수를 반환한다.'''
                return {'rebase_count': self.rebase_count}

        reader: FakeQuestReader = FakeQuestReader(frames=[
            _frame(hmd_x_m=0.0, clutch_pressed=True),
            _frame(hmd_x_m=0.0, clutch_pressed=False),
            _frame(hmd_x_m=0.0, clutch_pressed=True),
        ])
        producer: Quest3ActionProducer = Quest3ActionProducer(
            reader=reader,
            retargeter=DiagnosticRetargeter(),
            clock=_clock,
        )
        observation: RobotObservation = RobotObservation(state=(0.0,) * 7)
        producer.reset(observation)
        producer.next_action(observation)
        producer.next_action(observation)
        producer.next_action(observation)
        self.assertEqual(producer.diagnostics()['retargeter'], {'rebase_count': 2})
        producer.reset(observation)
        self.assertEqual(producer.diagnostics()['retargeter'], {'rebase_count': 0})

    def test_connect_waits_for_first_valid_frame(
        self,
    ) -> None:
        '''Connect가 첫 QuestFrame을 읽어 실제 WebXR 입력까지 확인하는지 검증한다.'''
        reader: FakeQuestReader = FakeQuestReader(
            frames=[_frame(hmd_x_m=0.0, clutch_pressed=False)],
        )
        producer: Quest3ActionProducer = Quest3ActionProducer(
            reader=reader,
            retargeter=FakeRetargeter(),
            clock=_clock,
        )

        producer.connect()

        self.assertTrue(reader.connected)
        self.assertEqual(reader.read_count, 1)

    def test_hmd_motion_is_allowed_while_clutch_is_pressed(
        self,
    ) -> None:
        '''착용한 HMD가 이동해도 clutch가 눌린 동안 retargeting을 계속하는지 검증한다.'''
        frames: list[QuestFrame] = [
            _frame(hmd_x_m=0.0, clutch_pressed=True),
            _frame(hmd_x_m=0.03, clutch_pressed=True),
        ]
        reader: FakeQuestReader = FakeQuestReader(frames=frames)
        retargeter: FakeRetargeter = FakeRetargeter()
        producer: Quest3ActionProducer = Quest3ActionProducer(
            reader=reader,
            retargeter=retargeter,
            clock=_clock,
        )
        observation: RobotObservation = RobotObservation(
            state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )

        first_action: RobotAction = producer.next_action(observation)
        moved_action: RobotAction = producer.next_action(observation)
        diagnostics: dict[str, object] = producer.diagnostics()

        self.assertNotEqual(first_action.values, observation.state)
        self.assertNotEqual(moved_action.values, observation.state)
        self.assertEqual(retargeter.rebase_count, 1)
        self.assertEqual(diagnostics['clutch_control_step_count'], 2)
        self.assertEqual(diagnostics['changed_action_step_count'], 2)
        self.assertEqual(diagnostics['hold_reason_counts'], {})

    def test_primary_button_press_marks_success(
        self,
    ) -> None:
        '''A/X button을 새로 누르면 clutch와 무관하게 성공 episode로 기록하는지 검증한다.'''
        outcomes: list[EpisodeOutcome | None] = _run_button_frames([
            _button_frame(primary_pressed=False, secondary_pressed=False),
            _button_frame(primary_pressed=True, secondary_pressed=False, clutch_pressed=False),
            _button_frame(primary_pressed=False, secondary_pressed=True),
        ])

        self.assertEqual(outcomes, [None, EpisodeOutcome.SUCCESS, EpisodeOutcome.SUCCESS])

    def test_secondary_button_press_marks_failure(
        self,
    ) -> None:
        '''B/Y button을 새로 누르면 실패로 기록하고 동시 입력에서는 실패를 우선하는지 검증한다.'''
        failure_outcomes: list[EpisodeOutcome | None] = _run_button_frames([
            _button_frame(primary_pressed=False, secondary_pressed=False),
            _button_frame(primary_pressed=False, secondary_pressed=True),
        ])
        simultaneous_outcomes: list[EpisodeOutcome | None] = _run_button_frames([
            _button_frame(primary_pressed=False, secondary_pressed=False),
            _button_frame(primary_pressed=True, secondary_pressed=True),
        ])

        self.assertEqual(failure_outcomes, [None, EpisodeOutcome.FAILURE])
        self.assertEqual(simultaneous_outcomes, [None, EpisodeOutcome.FAILURE])

    def test_button_held_from_episode_start_requires_release(
        self,
    ) -> None:
        '''Episode 시작 전부터 누르고 있던 button은 release 후 다시 눌러야 인정하는지 검증한다.'''
        outcomes: list[EpisodeOutcome | None] = _run_button_frames([
            _button_frame(primary_pressed=True, secondary_pressed=False),
            _button_frame(primary_pressed=True, secondary_pressed=False),
            _button_frame(primary_pressed=False, secondary_pressed=False),
            _button_frame(primary_pressed=True, secondary_pressed=False),
        ])

        self.assertEqual(outcomes, [None, None, None, EpisodeOutcome.SUCCESS])

    def test_stale_frame_button_is_ignored(
        self,
    ) -> None:
        '''Stale frame의 button 입력은 episode 종료로 인정하지 않는지 검증한다.'''
        stale_frame: QuestFrame = _button_frame(primary_pressed=True, secondary_pressed=False)
        stale_frame = QuestFrame(
            controller_pose=stale_frame.controller_pose,
            hmd_pose=stale_frame.hmd_pose,
            trigger=stale_frame.trigger,
            clutch_pressed=stale_frame.clutch_pressed,
            timestamp_s=0.0,
            primary_button_pressed=True,
        )
        outcomes: list[EpisodeOutcome | None] = _run_button_frames([
            _button_frame(primary_pressed=False, secondary_pressed=False),
            stale_frame,
        ])

        self.assertEqual(outcomes, [None, None])

    def test_parse_quest_frame_payload(
        self,
    ) -> None:
        '''Relay xr_frame에서 right controller, trigger, clutch와 HMD pose를 변환하는지 검증한다.'''
        payload: dict[str, object] = {
            'type': 'xr_frame',
            'controllers': {
                'right': {
                    'position': [0.1, 0.2, 0.3],
                    'orientation': [0.0, 0.0, 0.0, 1.0],
                    'buttons': [
                        {'p': True, 't': True, 'v': 0.75},
                        {'p': True, 't': True, 'v': 1.0},
                    ],
                },
            },
            'viewer': {
                'position': [1.0, 2.0, 3.0],
                'orientation': [0.0, 0.0, 0.0, 1.0],
            },
        }

        frame: QuestFrame = parse_quest_frame_payload(
            payload=payload,
            controller_hand='right',
            received_timestamp_s=12.5,
        )

        self.assertEqual(frame.controller_pose.position, (0.1, 0.2, 0.3))
        self.assertEqual(frame.hmd_pose.position, (1.0, 2.0, 3.0))
        self.assertEqual(frame.trigger, 0.75)
        self.assertTrue(frame.clutch_pressed)
        self.assertEqual(frame.timestamp_s, 12.5)
        self.assertFalse(frame.primary_button_pressed)
        self.assertFalse(frame.secondary_button_pressed)

    def test_parse_quest_frame_payload_reads_a_and_b_buttons(
        self,
    ) -> None:
        '''xr-standard button index 4와 5를 A/X와 B/Y 입력으로 변환하는지 검증한다.'''
        released: dict[str, object] = {'p': False, 't': False, 'v': 0.0}
        payload: dict[str, object] = {
            'type': 'xr_frame',
            'controllers': {
                'right': {
                    'position': [0.0, 0.0, 0.0],
                    'orientation': [0.0, 0.0, 0.0, 1.0],
                    'buttons': [
                        released,
                        released,
                        released,
                        released,
                        {'p': True, 't': True, 'v': 1.0},
                        released,
                    ],
                },
            },
            'viewer': {
                'position': [0.0, 0.0, 0.0],
                'orientation': [0.0, 0.0, 0.0, 1.0],
            },
        }

        frame: QuestFrame = parse_quest_frame_payload(
            payload=payload,
            controller_hand='right',
            received_timestamp_s=1.0,
        )

        self.assertTrue(frame.primary_button_pressed)
        self.assertFalse(frame.secondary_button_pressed)

    def test_probe_summary_allows_worn_hmd_motion(
        self,
    ) -> None:
        '''Controller, trigger와 clutch를 관측하면 착용한 HMD가 움직여도 probe가 통과하는지 검증한다.'''
        frames: list[QuestFrame] = [
            QuestFrame(
                controller_pose=QuestPose(
                    position=(0.0, 0.0, 0.0),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                hmd_pose=QuestPose(
                    position=(0.0, 0.0, 0.0),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                trigger=0.0,
                clutch_pressed=False,
                timestamp_s=10.0,
            ),
            QuestFrame(
                controller_pose=QuestPose(
                    position=(0.02, 0.0, 0.0),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                hmd_pose=QuestPose(
                    position=(0.5, 0.0, 0.0),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                trigger=1.0,
                clutch_pressed=True,
                timestamp_s=10.1,
            ),
        ]

        advertised_camera_ids: tuple[str, ...] = ('top',)
        summary: dict[str, object] = _summarize_frames(
            frames=frames,
            advertised_camera_ids=advertised_camera_ids,
        )
        missing_camera_summary: dict[str, object] = _summarize_frames(
            frames=frames,
            advertised_camera_ids=(),
        )

        self.assertTrue(summary['all_checks_passed'])
        self.assertEqual(summary['advertised_camera_ids'], advertised_camera_ids)
        self.assertFalse(missing_camera_summary['all_checks_passed'])

    def test_websocket_reader_returns_latest_xr_frame(
        self,
    ) -> None:
        '''Background reader가 relay message에서 QuestFrame을 생성하는지 검증한다.'''
        payload: dict[str, object] = {
            'type': 'xr_frame',
            'controllers': {
                'right': {
                    'position': [0.1, 0.2, 0.3],
                    'orientation': [0.0, 0.0, 0.0, 1.0],
                    'buttons': [
                        {'v': 0.4},
                        {'p': True},
                    ],
                },
            },
            'viewer': {
                'position': [0.0, 1.6, 0.0],
                'orientation': [0.0, 0.0, 0.0, 1.0],
            },
        }
        connection: FakeWebSocketConnection = FakeWebSocketConnection(
            messages=[json.dumps(payload)],
        )
        connector: FakeWebSocketConnector = FakeWebSocketConnector(connection=connection)
        reader: WebSocketQuestFrameReader = WebSocketQuestFrameReader(
            websocket_url='ws://127.0.0.1:8443/ws',
            controller_hand='right',
            connector=connector,
            clock=_clock,
        )

        reader.connect()
        frame: QuestFrame = reader.read()
        reader.close()

        self.assertEqual(frame.controller_pose.position, (0.1, 0.2, 0.3))
        self.assertEqual(frame.trigger, 0.4)
        self.assertTrue(frame.clutch_pressed)
        self.assertEqual(connector.websocket_url, 'ws://127.0.0.1:8443/ws')
        self.assertTrue(connection.closed)

    def test_websocket_reader_sends_initial_relay_config(
        self,
    ) -> None:
        '''Connect 직후 execution target별 video config를 relay에 전송하는지 검증한다.'''
        connection: FakeWebSocketConnection = FakeWebSocketConnection(messages=[])
        connector: FakeWebSocketConnector = FakeWebSocketConnector(connection=connection)
        initial_message: dict[str, object] = {
            'type': 'config_update',
            'config': {'video': {'enabled': False}},
        }
        reader: WebSocketQuestFrameReader = WebSocketQuestFrameReader(
            websocket_url='ws://127.0.0.1:8443/ws',
            controller_hand='right',
            initial_message=initial_message,
            connector=connector,
        )

        reader.connect()
        reader.close()

        self.assertEqual(connection.sent_messages, [json.dumps(initial_message)])

    def test_websocket_reader_skips_temporary_tracking_loss(
        self,
    ) -> None:
        '''선택한 controller가 누락된 frame 뒤의 유효 frame을 계속 수신하는지 검증한다.'''
        missing_controller_payload: dict[str, object] = {
            'type': 'xr_frame',
            'controllers': {},
            'viewer': None,
        }
        valid_payload: dict[str, object] = {
            'type': 'xr_frame',
            'controllers': {
                'right': {
                    'position': [0.1, 0.2, 0.3],
                    'orientation': [0.0, 0.0, 0.0, 1.0],
                    'buttons': [
                        {'v': 0.4},
                        {'p': True},
                    ],
                },
            },
            'viewer': {
                'position': [0.0, 1.6, 0.0],
                'orientation': [0.0, 0.0, 0.0, 1.0],
            },
        }
        connection: FakeWebSocketConnection = FakeWebSocketConnection(
            messages=[
                json.dumps(missing_controller_payload),
                json.dumps(valid_payload),
            ],
        )
        connector: FakeWebSocketConnector = FakeWebSocketConnector(connection=connection)
        reader: WebSocketQuestFrameReader = WebSocketQuestFrameReader(
            websocket_url='ws://127.0.0.1:8443/ws',
            controller_hand='right',
            connector=connector,
            clock=_clock,
        )

        reader.connect()
        frame: QuestFrame = reader.read()
        diagnostics: dict[str, object] = reader.diagnostics()
        reader.close()

        self.assertEqual(frame.controller_pose.position, (0.1, 0.2, 0.3))
        self.assertEqual(diagnostics['received_frame_count'], 1)
        self.assertEqual(diagnostics['skipped_tracking_frame_count'], 1)


if __name__ == '__main__':
    unittest.main()
