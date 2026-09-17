'''YAM Quest retargeter를 headless MuJoCo model로 검증한다.'''

from __future__ import annotations

import unittest

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


if __name__ == '__main__':
    unittest.main()
