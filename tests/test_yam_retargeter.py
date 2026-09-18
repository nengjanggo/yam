'''YAM Quest retargeter를 headless MuJoCo model로 검증한다.'''

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from yam_control.config import QuestConfig, RobotConfig
from yam_control.teleop.yam_retargeter import YamQuestRetargeter
from yam_control.types import QuestFrame, QuestPose, RobotAction, RobotObservation


class YamQuestRetargeterTest(unittest.TestCase):
    '''실제 robot command 없이 YAM FK/IK retargeting을 검증한다.'''

    def test_stationary_controller_holds_arm_and_maps_trigger(
        self,
    ) -> None:
        '''Clutch engage 직후 arm pose를 유지하고 trigger를 gripper action으로 변환한다.'''
        retargeter: YamQuestRetargeter = YamQuestRetargeter(
            robot_config=RobotConfig(),
            quest_config=QuestConfig(),
        )
        observation: RobotObservation = RobotObservation(
            state=(0.0, 0.7854, 1.5708, 0.0, 0.0, 0.0, 1.0),
        )
        controller_pose: QuestPose = QuestPose(
            position=(0.2, 1.2, -0.3),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        frame: QuestFrame = QuestFrame(
            controller_pose=controller_pose,
            hmd_pose=QuestPose(
                position=(0.0, 1.6, 0.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            trigger=0.2,
            clutch_pressed=True,
            timestamp_s=1.0,
        )

        retargeter.rebase(frame, observation)
        action: RobotAction = retargeter.retarget(frame, observation)

        # Shape `(6,)` arm action이 engage 시점 shape `(6,)` state를 유지하는지 검증
        np.testing.assert_allclose(action.values[:6], observation.state[:6], atol=1e-3)
        self.assertAlmostEqual(action.values[6], 0.8)

        moved_frame: QuestFrame = QuestFrame(
            controller_pose=QuestPose(
                position=(0.22, 1.2, -0.3),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            hmd_pose=frame.hmd_pose,
            trigger=0.2,
            clutch_pressed=True,
            timestamp_s=1.1,
        )
        moved_action: RobotAction = retargeter.retarget(moved_frame, observation)
        # Shape `(6,)` moved action과 current arm state의 joint delta를 계산
        joint_delta: np.ndarray = (
            np.asarray(moved_action.values[:6]) - np.asarray(observation.state[:6])
        )
        self.assertTrue(np.any(np.abs(joint_delta) > 1e-5))
        self.assertTrue(np.all(np.abs(joint_delta) <= 0.040001))

    def test_ik_diagnostics_include_failure_clipping_and_model_fk_residual(
        self,
    ) -> None:
        '''IK 실패와 joint delta clipping을 계수하고 episode reset으로 누적값을 지운다.'''
        retargeter: YamQuestRetargeter = YamQuestRetargeter(
            robot_config=RobotConfig(),
            quest_config=QuestConfig(),
        )
        observation: RobotObservation = RobotObservation(
            state=(0.0, 0.7854, 1.5708, 0.0, 0.0, 0.0, 1.0),
        )
        hmd_pose: QuestPose = QuestPose(
            position=(0.0, 1.6, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        engage_frame: QuestFrame = QuestFrame(
            controller_pose=QuestPose(
                position=(0.2, 1.2, -0.3),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            hmd_pose=hmd_pose,
            trigger=0.0,
            clutch_pressed=True,
            timestamp_s=1.0,
        )
        moved_frame: QuestFrame = QuestFrame(
            controller_pose=QuestPose(
                position=(0.22, 1.2, -0.3),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            hmd_pose=hmd_pose,
            trigger=0.0,
            clutch_pressed=True,
            timestamp_s=1.1,
        )
        self.assertIsNone(retargeter.diagnostics()['ik_solve_mean_ms'])
        retargeter.rebase(engage_frame, observation)
        # Shape `(7,)` 관측 state에서 shape `(nq,)` IK 입력을 생성
        current_qpos: np.ndarray = retargeter._configuration_qpos(observation)
        with patch.object(retargeter._kinematics, 'ik', return_value=(False, current_qpos)), patch(
            'yam_control.teleop.yam_retargeter.time.perf_counter',
            side_effect=(10.0, 10.002),
        ):
            failed_action: RobotAction = retargeter.retarget(moved_frame, observation)
        self.assertEqual(failed_action.values, observation.state)
        failed_diagnostics: dict[str, int | float | None] = retargeter.diagnostics()
        self.assertEqual(failed_diagnostics['ik_attempt_count'], 1)
        self.assertEqual(failed_diagnostics['ik_failure_count'], 1)
        self.assertAlmostEqual(failed_diagnostics['ik_solve_mean_ms'], 2.0)
        self.assertAlmostEqual(failed_diagnostics['target_to_commanded_ee_position_mean_m'], 0.01)
        self.assertAlmostEqual(failed_diagnostics['target_to_commanded_ee_orientation_mean_rad'], 0.0)

        # Shape `(nq,)` IK 결과에서 두 번째 joint를 tick 제한보다 크게 변경
        solved_qpos: np.ndarray = current_qpos.copy()
        solved_qpos[retargeter._arm_qpos_addresses[1]] += 0.2
        with patch.object(retargeter._kinematics, 'ik', return_value=(True, solved_qpos)), patch(
            'yam_control.teleop.yam_retargeter.time.perf_counter',
            side_effect=(20.0, 20.004),
        ):
            clipped_action: RobotAction = retargeter.retarget(moved_frame, observation)
        self.assertAlmostEqual(clipped_action.values[1] - observation.state[1], 0.04)
        diagnostics: dict[str, int | float | None] = retargeter.diagnostics()
        self.assertEqual(diagnostics['ik_attempt_count'], 2)
        self.assertEqual(diagnostics['ik_failure_count'], 1)
        self.assertEqual(diagnostics['joint_delta_clipped_step_count'], 1)
        self.assertAlmostEqual(diagnostics['ik_solve_mean_ms'], 3.0)
        self.assertAlmostEqual(diagnostics['ik_solve_max_ms'], 4.0)
        self.assertGreaterEqual(
            diagnostics['target_to_commanded_ee_position_max_m'],
            diagnostics['target_to_commanded_ee_position_mean_m'],
        )

        retargeter.reset_diagnostics()
        reset_diagnostics: dict[str, int | float | None] = retargeter.diagnostics()
        self.assertEqual(reset_diagnostics['ik_attempt_count'], 0)
        self.assertEqual(reset_diagnostics['ik_failure_count'], 0)
        self.assertEqual(reset_diagnostics['joint_delta_clipped_step_count'], 0)
        self.assertIsNone(reset_diagnostics['ik_solve_mean_ms'])
        self.assertIsNone(reset_diagnostics['target_to_commanded_ee_position_mean_m'])

        # Shape `(4,)` 현재 quaternion과 직교하는 quaternion으로 90도 목표를 구성
        current_position: np.ndarray
        current_quaternion: np.ndarray
        current_position, current_quaternion = retargeter._end_effector_pose(current_qpos)
        orthogonal_quaternion: np.ndarray = np.asarray(
            (-current_quaternion[1], current_quaternion[0], -current_quaternion[3], current_quaternion[2]),
            dtype=np.float64,
        )
        target_quaternion: np.ndarray = (
            np.cos(np.pi / 4.0) * current_quaternion
            + np.sin(np.pi / 4.0) * orthogonal_quaternion
        )
        retargeter._record_target_error(current_qpos, current_position, target_quaternion)
        self.assertAlmostEqual(retargeter._target_to_commanded_ee_orientation_total_rad, np.pi / 2.0)
        # Shape `(4,)` 부호를 반대로 해도 같은 회전이므로 동일한 최단 각도를 기록
        retargeter._record_target_error(current_qpos, current_position, -target_quaternion)
        self.assertAlmostEqual(retargeter._target_to_commanded_ee_orientation_total_rad, np.pi)


if __name__ == '__main__':
    unittest.main()
