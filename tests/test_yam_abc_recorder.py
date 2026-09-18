'''단일 arm, 단일 camera 구성의 yam-abc raw episode 저장을 hardware 없이 검증한다.'''

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
# Notebook과 같은 방식으로 vendored yam-abc-reproduce를 import path에 추가
sys.path.insert(0, str(PROJECT_ROOT / 'third_party' / 'yam-abc-reproduce'))
# GPU 유무와 무관하게 결정적인 software encoder를 사용
os.environ.setdefault('YAM_ABC_VIDEO_ENCODER', 'libx264')

from yam_abc_reproduce.camera.interface import CameraFrame  # noqa: E402
from yam_abc_reproduce.data import codec  # noqa: E402

from yam_control.config import (  # noqa: E402
    CameraConfig,
    CameraDeviceConfig,
    QuestConfig,
    RobotConfig,
    RunConfig,
    build_run_config,
)
from yam_control.data.yam_abc import (  # noqa: E402
    ACTION_EE_POSE_KEY,
    EE_POSE_KEY,
    EndEffectorEpisodeRecorder,
    create_yam_abc_recorder,
    encode_single_arm_step,
)
from yam_control.robot.kinematics import YamEndEffectorKinematics  # noqa: E402
from yam_control.types import RobotAction, RobotObservation  # noqa: E402


def _build_config(
    data_root: str,
    camera: CameraConfig,
) -> RunConfig:
    '''Recording이 켜진 real teleoperation RunConfig를 생성한다.'''
    return build_run_config(
        mode='teleop',
        teleop_source='quest3',
        save_teleop_data=True,
        vla_type='pi0.5',
        checkpoint_uri='',
        checkpoint_revision=None,
        policy_config_name='',
        use_rtc=False,
        execution_target='real',
        use_safety_gate=False,
        control_hz=30.0,
        task_prompt='pick up the object',
        data_root=data_root,
        robot=RobotConfig(),
        quest=QuestConfig(),
        camera=camera,
    )


def _top_camera(
) -> CameraConfig:
    '''Top camera 하나만 있는 CameraConfig를 반환한다.'''
    return CameraConfig(
        devices=(CameraDeviceConfig(role='top', device_path='/dev/fake-top'),),
    )


def _observation(
    step_index: int,
) -> RobotObservation:
    '''Step index를 담은 shape `(7,)` state와 top camera frame을 반환한다.'''
    image: np.ndarray = np.full((224, 224, 3), step_index * 10, dtype=np.uint8)
    return RobotObservation(
        state=(0.1 * step_index, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        images={'top': CameraFrame(images={'rgb': image}, timestamp_ms=1000.0 + step_index)},
    )


KINEMATICS: YamEndEffectorKinematics = YamEndEffectorKinematics('linear_4310')


class YamABCRecorderTest(unittest.TestCase):
    '''Step encoding과 yam-abc default format 저장을 검증한다.'''

    def test_step_encoder_splits_joint_and_gripper(
        self,
    ) -> None:
        '''Shape `(7,)` state와 action을 단일 arm key로 분리하는지 검증한다.'''
        action: RobotAction = RobotAction(values=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5))
        actions, obs, frames = encode_single_arm_step(_observation(1), action, ('top',), KINEMATICS)

        self.assertEqual(obs['left']['joint_pos'].shape, (6,))  # type: ignore[index]
        self.assertEqual(obs['left']['gripper_pos'].shape, (1,))  # type: ignore[index]
        self.assertEqual(actions['left'].shape, (7,))  # type: ignore[index]
        self.assertEqual(set(frames), {'top'})  # type: ignore[arg-type]
        self.assertEqual(obs['left']['ee_pose'].shape, (7,))  # type: ignore[index]
        np.testing.assert_allclose(
            obs['left']['action_ee_pose'],  # type: ignore[index]
            KINEMATICS.pose(np.asarray(action.values[:6])),
        )

    def test_ee_pose_matches_grasp_site_fk(
        self,
    ) -> None:
        '''EPISODE_INITIAL_POSE의 grasp_site가 robot base 기준 앞쪽 41 cm, 높이 27 cm인지 검증한다.'''
        pose: np.ndarray = KINEMATICS.pose(np.array((0.0, 1.2, 0.9, 0.0, 0.0, 0.0)))

        np.testing.assert_allclose(pose[:3], (0.4095, 0.0, 0.2723), atol=1e-3)
        self.assertAlmostEqual(float(np.linalg.norm(pose[3:])), 1.0)

    def test_step_encoder_rejects_missing_camera(
        self,
    ) -> None:
        '''설정된 camera frame이 observation에 없으면 거부하는지 검증한다.'''
        observation: RobotObservation = RobotObservation(state=(0.0,) * 7)

        with self.assertRaises(ValueError):
            encode_single_arm_step(observation, RobotAction(values=(0.0,) * 7), ('top',), KINEMATICS)

    def test_recorder_requires_camera(
        self,
    ) -> None:
        '''Camera 없이 recorder 생성을 거부하는지 검증한다.'''
        with self.assertRaises(ValueError):
            create_yam_abc_recorder(_build_config('/tmp/unused', CameraConfig()))

    def test_recorder_writes_single_arm_single_camera_episode(
        self,
    ) -> None:
        '''단일 arm과 top camera episode가 yam-abc default format으로 저장되는지 검증한다.'''
        with tempfile.TemporaryDirectory() as data_root:
            recorder = create_yam_abc_recorder(_build_config(data_root, _top_camera()))
            recorder.start('pick up the object')
            step_index: int
            for step_index in range(5):
                action: RobotAction = RobotAction(values=(0.1 * step_index, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
                recorder.record(_observation(step_index), action)
            recorder.finish()

            episode_dirs: list[Path] = list((Path(data_root) / 'pick_up_the_object').iterdir())
            self.assertEqual(len(episode_dirs), 1)
            episode_dir: Path = episode_dirs[0]
            metadata: dict[str, object] = json.loads((episode_dir / 'metadata.json').read_text())

            self.assertTrue((episode_dir / 'write_complete.flag').exists())
            self.assertTrue((episode_dir / 'top-images-rgb.mp4').exists())
            self.assertEqual(np.load(episode_dir / 'left-joint_pos.npy').shape, (5, 6))
            self.assertEqual(np.load(episode_dir / 'left-gripper_pos.npy').shape, (5, 1))
            self.assertEqual(np.load(episode_dir / 'action-left-joint.npy').shape, (5, 6))
            self.assertEqual(np.load(episode_dir / 'action-left-gripper.npy').shape, (5, 1))
            self.assertEqual(np.load(episode_dir / 'top-timestamp.npy').shape, (5,))
            ee_pose: np.ndarray = np.load(episode_dir / f'{EE_POSE_KEY}.npy')
            action_ee_pose: np.ndarray = np.load(episode_dir / f'{ACTION_EE_POSE_KEY}.npy')
            self.assertEqual(ee_pose.shape, (5, 7))
            self.assertEqual(action_ee_pose.shape, (5, 7))
            np.testing.assert_allclose(ee_pose[2, :3], KINEMATICS.pose(np.array((0.2, 0, 0, 0, 0, 0)))[:3])
            self.assertEqual(metadata['extra']['ee_pose']['site'], 'grasp_site')  # type: ignore[index]
            self.assertEqual(metadata['arm_names'], ['left'])
            self.assertEqual(metadata['num_frames'], 5)
            camera_metadata: dict[str, object] = metadata['cameras'][0]  # type: ignore[index]
            self.assertEqual(
                (camera_metadata['role'], camera_metadata['width'], camera_metadata['height'], camera_metadata['fps']),
                ('top', 224, 224, 30),
            )

    def test_ee_quaternion_sign_is_continuous(
        self,
    ) -> None:
        '''같은 회전의 부호 반전 quaternion을 이전 frame과 이어지는 부호로 저장하는지 검증한다.'''
        with tempfile.TemporaryDirectory() as data_root:
            recorder = create_yam_abc_recorder(_build_config(data_root, _top_camera()))
            episode_recorder: EndEffectorEpisodeRecorder = recorder._recorder  # type: ignore[assignment]
            episode_recorder.start('pick up the object')
            pose: np.ndarray = np.array((0.4, 0.0, 0.3, 0.1, 0.2, 0.3, -0.927))
            first: np.ndarray = episode_recorder._continuous_pose(EE_POSE_KEY, pose)
            flipped_pose: np.ndarray = pose.copy()
            flipped_pose[3:] = -pose[3:]
            second: np.ndarray = episode_recorder._continuous_pose(EE_POSE_KEY, flipped_pose)
            episode_recorder.abort()

        self.assertGreaterEqual(first[6], 0.0)
        np.testing.assert_allclose(second, first)

    def test_recorder_pins_libx264_encoder(
        self,
    ) -> None:
        '''NVENC가 이미 선택된 process에서도 recorder 생성 시 libx264로 다시 고정하는지 검증한다.'''
        os.environ['YAM_ABC_VIDEO_ENCODER'] = 'auto'
        codec.encoder.cache_clear()

        create_yam_abc_recorder(_build_config('/tmp/unused', _top_camera()))

        self.assertEqual(os.environ['YAM_ABC_VIDEO_ENCODER'], 'libx264')
        self.assertEqual(codec.encoder(), 'libx264')

    def test_abort_discards_episode(
        self,
    ) -> None:
        '''Abort한 episode directory가 남지 않는지 검증한다.'''
        with tempfile.TemporaryDirectory() as data_root:
            recorder = create_yam_abc_recorder(_build_config(data_root, _top_camera()))
            recorder.start('pick up the object')
            recorder.record(_observation(0), RobotAction(values=(0.0,) * 7))
            recorder.abort()

            self.assertEqual(list((Path(data_root) / 'pick_up_the_object').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
