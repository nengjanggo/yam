'''Mode별 configuration 정규화와 validation을 검증한다.'''

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import cast

from yam_control import create_session
from yam_control.config import (
    CameraConfig,
    CameraDeviceConfig,
    ExecutionTarget,
    InferenceRunConfig,
    QuestConfig,
    RobotConfig,
    RunConfig,
    RunMode,
    TeleopRunConfig,
    build_run_config,
    with_camera_config,
)


def _build_config(
    mode: RunMode,
    use_rtc: bool,
    save_teleop_data: bool = False,
    execution_target: ExecutionTarget = 'mujoco',
) -> RunConfig:
    '''Test에 필요한 최소 RunConfig를 생성한다.'''
    return build_run_config(
        mode=mode,
        teleop_source='quest3',
        save_teleop_data=save_teleop_data,
        vla_type='pi0.5',
        checkpoint_uri='checkpoint',
        checkpoint_revision=None,
        policy_config_name='pi05_yam',
        use_rtc=use_rtc,
        execution_target=execution_target,
        use_safety_gate=False,
        control_hz=30.0,
        task_prompt='pick up the object',
        data_root='data/episodes',
        robot=RobotConfig(),
        quest=QuestConfig(),
        camera=CameraConfig(),
    )


class ConfigTest(unittest.TestCase):
    '''Run mode에 따른 configuration 선택을 검증한다.'''

    def test_teleop_ignores_rtc_flag(
        self,
    ) -> None:
        '''Teleoperation에서 USE_RTC가 True여도 RTC를 활성화하지 않는지 검증한다.'''
        config: RunConfig = _build_config(mode='teleop', use_rtc=True)
        mode_config: TeleopRunConfig = cast(TeleopRunConfig, config.mode_config)

        self.assertEqual(mode_config.teleop_source, 'quest3')
        self.assertFalse(config.rtc_enabled)

    def test_inference_keeps_rtc_flag(
        self,
    ) -> None:
        '''Inference에서 USE_RTC 값을 유지하는지 검증한다.'''
        config: RunConfig = _build_config(mode='inference', use_rtc=True)
        mode_config: InferenceRunConfig = cast(InferenceRunConfig, config.mode_config)

        self.assertEqual(mode_config.vla_type, 'pi0.5')
        self.assertTrue(config.rtc_enabled)

    def test_mujoco_ignores_save_teleop_data(
        self,
    ) -> None:
        '''MuJoCo에서는 SAVE_TELEOP_DATA가 True여도 recording을 끄는지 검증한다.'''
        config: RunConfig = _build_config(mode='teleop', use_rtc=False, save_teleop_data=True)
        mode_config: TeleopRunConfig = cast(TeleopRunConfig, config.mode_config)

        self.assertFalse(mode_config.save_teleop_data)

    def test_real_keeps_save_teleop_data(
        self,
    ) -> None:
        '''Real robot에서는 SAVE_TELEOP_DATA 값을 유지하는지 검증한다.'''
        config: RunConfig = _build_config(
            mode='teleop',
            use_rtc=False,
            save_teleop_data=True,
            execution_target='real',
        )
        mode_config: TeleopRunConfig = cast(TeleopRunConfig, config.mode_config)

        self.assertTrue(mode_config.save_teleop_data)

    def test_with_camera_config_replaces_only_camera(
        self,
    ) -> None:
        '''Camera configuration만 교체하고 나머지 configuration은 유지하는지 검증한다.'''
        config: RunConfig = _build_config(mode='teleop', use_rtc=False)
        camera: CameraConfig = CameraConfig(
            devices=(CameraDeviceConfig(role='top', device_path='/dev/fake-top'),),
        )
        updated: RunConfig = with_camera_config(config, camera)

        self.assertEqual(updated.common.camera.roles, ('top',))
        self.assertEqual(updated.common.quest, config.common.quest)
        self.assertEqual(updated.mode_config, config.mode_config)

    def test_pose_requires_seven_values(
        self,
    ) -> None:
        '''Single YAM initial pose가 shape `(7,)`인지 검증한다.'''
        invalid_pose: tuple[float, ...] = (0.0, 1.0)

        with self.assertRaises(ValueError):
            RobotConfig(episode_initial_pose=invalid_pose)

    def test_invalid_runtime_mode_is_rejected(
        self,
    ) -> None:
        '''Literal type을 우회한 잘못된 mode도 runtime에서 거부하는지 검증한다.'''
        invalid_mode: RunMode = cast(RunMode, 'invalid')

        with self.assertRaises(ValueError):
            _build_config(mode=invalid_mode, use_rtc=False)

    def test_invalid_quest_stream_quality_is_rejected(
        self,
    ) -> None:
        '''Quest stream JPEG quality가 허용 범위 밖이면 거부되는지 검증한다.'''
        invalid_quality: int = 100

        with self.assertRaises(ValueError):
            QuestConfig(stream_jpeg_quality=invalid_quality)

    def test_real_wrist_preview_requires_wrist_camera(
        self,
    ) -> None:
        '''Real wrist preview를 켰는데 wrist role camera가 없으면 session 생성을 거부하는지 검증한다.'''
        config: RunConfig = _build_config(
            mode='teleop',
            use_rtc=False,
            execution_target='real',
        )
        config = replace(
            config,
            common=replace(
                config.common,
                quest=QuestConfig(show_real_wrist_camera=True),
            ),
        )

        with self.assertRaisesRegex(ValueError, 'requires a wrist camera'):
            create_session(config)

    def test_inference_ignores_real_wrist_preview(
        self,
    ) -> None:
        '''Inference에서는 Quest wrist preview 설정과 wrist camera 유무를 무시하는지 검증한다.'''
        config: RunConfig = _build_config(
            mode='inference',
            use_rtc=False,
            execution_target='real',
        )
        config = replace(
            config,
            common=replace(
                config.common,
                quest=QuestConfig(show_real_wrist_camera=True),
            ),
        )

        create_session(config)

    def test_groot_selection_is_explicitly_not_implemented(
        self,
    ) -> None:
        '''GR00T가 public type에 존재하지만 session 생성 시 명시적으로 거부되는지 검증한다.'''
        config: RunConfig = build_run_config(
            mode='inference',
            teleop_source='leader',
            save_teleop_data=False,
            vla_type='groot',
            checkpoint_uri='',
            checkpoint_revision=None,
            policy_config_name='',
            use_rtc=False,
            execution_target='mujoco',
            use_safety_gate=False,
            control_hz=30.0,
            task_prompt='pick up the object',
            data_root='data/episodes',
            robot=RobotConfig(),
            quest=QuestConfig(),
            camera=CameraConfig(),
        )

        with self.assertRaises(NotImplementedError):
            create_session(config)

    def test_leader_selection_is_explicitly_not_implemented(
        self,
    ) -> None:
        '''Leader Arm source가 session 생성 시 명시적으로 거부되는지 검증한다.'''
        config: RunConfig = build_run_config(
            mode='teleop',
            teleop_source='leader',
            save_teleop_data=False,
            vla_type='pi0.5',
            checkpoint_uri='',
            checkpoint_revision=None,
            policy_config_name='',
            use_rtc=False,
            execution_target='mujoco',
            use_safety_gate=False,
            control_hz=30.0,
            task_prompt='pick up the object',
            data_root='data/episodes',
            robot=RobotConfig(),
            quest=QuestConfig(),
            camera=CameraConfig(),
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            'Leader Arm teleoperation is not implemented',
        ):
            create_session(config)


if __name__ == '__main__':
    unittest.main()
