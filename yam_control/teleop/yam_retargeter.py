'''Quest 3 controller motion을 I2RT YAM FK/IK로 joint action으로 변환한다.'''

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

from ..config import QuestConfig, RobotConfig
from ..types import QuestFrame, RobotAction, RobotObservation
from .clutch_pose_mapper import ClutchPoseMapper

ARM_DOF: int = 6
ROBOT_STATE_DIMENSION: int = 7
GRASP_SITE_NAME: str = 'grasp_site'


def _yaw_from_quaternion_xyzw(
    quaternion_xyzw: tuple[float, float, float, float],
) -> float:
    '''WebXR quaternion에서 world-up `+Y` 축 yaw radian을 반환한다.'''
    x: float
    y: float
    z: float
    w: float
    x, y, z, w = quaternion_xyzw
    return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


def _rotation_y(
    angle_rad: float,
) -> NDArray[np.float64]:
    '''`+Y` 축 angle_rad rotation matrix shape `(3, 3)`을 반환한다.'''
    cosine: float = float(np.cos(angle_rad))
    sine: float = float(np.sin(angle_rad))
    # Shape `(3, 3)` world-up rotation matrix를 생성
    return np.asarray(
        (
            (cosine, 0.0, sine),
            (0.0, 1.0, 0.0),
            (-sine, 0.0, cosine),
        ),
        dtype=np.float64,
    )


def _quaternion_wxyz(
    quaternion_xyzw: tuple[float, float, float, float],
) -> NDArray[np.float64]:
    '''WebXR shape `(4,)` xyzw quaternion을 MuJoCo wxyz quaternion으로 변환한다.'''
    x: float
    y: float
    z: float
    w: float
    x, y, z, w = quaternion_xyzw
    # Shape `(4,)` xyzw에서 shape `(4,)` wxyz로 순서를 변환
    return np.asarray((w, x, y, z), dtype=np.float64)


class YamQuestRetargeter:
    '''Quest reference-space controller delta를 shape `(7,)` YAM action으로 변환한다.'''

    def __init__(
        self,
        robot_config: RobotConfig,
        quest_config: QuestConfig,
    ) -> None:
        '''I2RT YAM model, Kinematics와 local ClutchPoseMapper를 생성한다.'''
        gripper_type: GripperType = GripperType.from_string_name(robot_config.gripper_type)
        xml_path: str = combine_arm_and_gripper_xml(
            ArmType.YAM,
            gripper_type,
        )
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_path(xml_path)
        self._kinematics: Kinematics = Kinematics(xml_path, GRASP_SITE_NAME)
        # Shape `(3, 3)` Quest world vector를 YAM base vector로 변환하는 calibration
        self._r_calib: NDArray[np.float64] = np.asarray(quest_config.r_calib, dtype=np.float64)
        identity: NDArray[np.float64] = np.eye(3, dtype=np.float64)
        if not np.allclose(self._r_calib.T @ self._r_calib, identity, atol=1e-6):
            raise ValueError('r_calib must be an orthonormal rotation matrix')
        if not np.isclose(np.linalg.det(self._r_calib), 1.0, atol=1e-6):
            raise ValueError('r_calib determinant must be 1')
        self._mapper: ClutchPoseMapper = ClutchPoseMapper(
            rotation=self._r_calib,
            translation_scale=quest_config.translation_scale,
            rotation_scale=quest_config.rotation_scale,
            position_reach_limit_m=quest_config.position_reach_limit_m,
            rotation_reach_limit_rad=quest_config.rotation_reach_limit_rad,
        )
        self._max_joint_delta_rad: float = quest_config.max_joint_delta_rad
        # Shape `(6,)` action order의 MuJoCo qpos address를 저장
        self._arm_qpos_addresses: NDArray[np.int64] = np.asarray(
            [self._joint_qpos_address(f'joint{joint_index}') for joint_index in range(1, ARM_DOF + 1)],
            dtype=np.int64,
        )
        # Shape `(6, 2)` lower/upper joint limit를 저장
        self._arm_joint_limits: NDArray[np.float64] = np.asarray(
            [
                self._model.jnt_range[
                    mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{joint_index}')
                ]
                for joint_index in range(1, ARM_DOF + 1)
            ],
            dtype=np.float64,
        )

    def _joint_qpos_address(
        self,
        joint_name: str,
    ) -> int:
        '''Joint name에 해당하는 MuJoCo qpos address를 반환한다.'''
        joint_id: int = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise ValueError(f'joint is missing from YAM model: {joint_name}')
        return int(self._model.jnt_qposadr[joint_id])

    def _configuration_qpos(
        self,
        observation: RobotObservation,
    ) -> NDArray[np.float64]:
        '''Shape `(7,)` robot state를 Kinematics model shape `(nq,)` qpos로 변환한다.'''
        if len(observation.state) != ROBOT_STATE_DIMENSION:
            raise ValueError('YAM observation state must have shape (7,)')
        # Shape `(7,)` state에서 shape `(nq,)` MuJoCo qpos로 확장
        qpos: NDArray[np.float64] = np.zeros(self._model.nq, dtype=np.float64)
        qpos[self._arm_qpos_addresses] = np.asarray(observation.state[:ARM_DOF], dtype=np.float64)
        normalized_gripper: float = float(np.clip(observation.state[ARM_DOF], 0.0, 1.0))
        joint_name: str
        for joint_name in ('joint7', 'joint8'):
            joint_id: int = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id >= 0:
                qpos_address: int = int(self._model.jnt_qposadr[joint_id])
                lower: float = float(self._model.jnt_range[joint_id, 0])
                upper: float = float(self._model.jnt_range[joint_id, 1])
                qpos[qpos_address] = lower + normalized_gripper * (upper - lower)
        return qpos

    def _end_effector_pose(
        self,
        qpos: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Shape `(nq,)` qpos FK로 position `(3,)`과 quaternion `(4,)`을 반환한다.'''
        # Shape `(nq,)` qpos에서 shape `(4, 4)` end-effector transform으로 변환
        transform: NDArray[np.float64] = np.asarray(self._kinematics.fk(qpos), dtype=np.float64)
        position: NDArray[np.float64] = transform[:3, 3].copy()
        quaternion_wxyz: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            quaternion_wxyz,
            np.ascontiguousarray(transform[:3, :3]).ravel(),
        )
        return position, quaternion_wxyz

    def rebase(
        self,
        frame: QuestFrame,
        observation: RobotObservation,
    ) -> None:
        '''Clutch engage 시 controller, HMD yaw와 현재 YAM end-effector pose를 저장한다.'''
        qpos: NDArray[np.float64] = self._configuration_qpos(observation)
        ee_position: NDArray[np.float64]
        ee_quaternion_wxyz: NDArray[np.float64]
        ee_position, ee_quaternion_wxyz = self._end_effector_pose(qpos)
        hmd_yaw_rad: float = _yaw_from_quaternion_xyzw(frame.hmd_pose.orientation_xyzw)
        # Shape `(3, 3) @ (3, 3)`을 shape `(3, 3)` engage rotation으로 결합
        engage_rotation: NDArray[np.float64] = self._r_calib @ _rotation_y(-hmd_yaw_rad)
        self._mapper.set_rotation(engage_rotation)
        self._mapper.engage(
            np.asarray(frame.controller_pose.position, dtype=np.float64),
            _quaternion_wxyz(frame.controller_pose.orientation_xyzw),
            ee_position,
            ee_quaternion_wxyz,
        )

    def retarget(
        self,
        frame: QuestFrame,
        observation: RobotObservation,
    ) -> RobotAction:
        '''Controller delta target을 IK하고 joint 변화량을 제한한 shape `(7,)` action을 반환한다.'''
        qpos: NDArray[np.float64] = self._configuration_qpos(observation)
        ee_position: NDArray[np.float64]
        ee_quaternion_wxyz: NDArray[np.float64]
        ee_position, ee_quaternion_wxyz = self._end_effector_pose(qpos)
        target: tuple[NDArray[np.float64], NDArray[np.float64]] | None = self._mapper.target(
            np.asarray(frame.controller_pose.position, dtype=np.float64),
            _quaternion_wxyz(frame.controller_pose.orientation_xyzw),
            ee_position,
            ee_quaternion_wxyz,
        )
        if target is None:
            return RobotAction(values=observation.state)
        target_position: NDArray[np.float64]
        target_quaternion_wxyz: NDArray[np.float64]
        target_position, target_quaternion_wxyz = target
        # Shape `(3,)` position과 `(4,)` quaternion으로 shape `(4, 4)` IK target을 생성
        target_transform: NDArray[np.float64] = np.eye(4, dtype=np.float64)
        target_rotation_flat: NDArray[np.float64] = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(target_rotation_flat, target_quaternion_wxyz)
        target_transform[:3, :3] = target_rotation_flat.reshape(3, 3)
        target_transform[:3, 3] = target_position
        success: bool
        solved_qpos: NDArray[np.float64]
        success, solved_qpos = self._kinematics.ik(
            target_pose=target_transform,
            site_name=GRASP_SITE_NAME,
            init_q=qpos,
            pos_threshold=1e-3,
            ori_threshold=1e-3,
            max_iters=100,
        )
        if not success:
            return RobotAction(values=observation.state)
        # Shape `(nq,)` IK result에서 shape `(6,)` arm joint target을 추출
        solved_arm: NDArray[np.float64] = solved_qpos[self._arm_qpos_addresses]
        bounded_arm: NDArray[np.float64] = np.clip(
            solved_arm,
            self._arm_joint_limits[:, 0],
            self._arm_joint_limits[:, 1],
        )
        current_arm: NDArray[np.float64] = np.asarray(observation.state[:ARM_DOF], dtype=np.float64)
        joint_delta: NDArray[np.float64] = np.clip(
            bounded_arm - current_arm,
            -self._max_joint_delta_rad,
            self._max_joint_delta_rad,
        )
        commanded_arm: NDArray[np.float64] = current_arm + joint_delta
        commanded_gripper: float = 1.0 - float(np.clip(frame.trigger, 0.0, 1.0))
        action_values: tuple[float, ...] = (
            *(float(value) for value in commanded_arm),
            commanded_gripper,
        )
        return RobotAction(values=action_values)
