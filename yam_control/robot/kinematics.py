'''YAM MuJoCo model FK로 grasp_site의 robot base 기준 pose를 계산한다.'''

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

ARM_DOF: int = 6
GRASP_SITE_NAME: str = 'grasp_site'
EE_POSE_DIMENSION: int = 7


class YamEndEffectorKinematics:
    '''Shape `(6,)` arm joint를 shape `(7,)` grasp_site pose `(x, y, z, qx, qy, qz, qw)`로 변환한다.'''

    def __init__(
        self,
        gripper_type: str,
    ) -> None:
        '''Gripper 종류에 맞는 YAM model과 I2RT Kinematics를 생성한다.'''
        xml_path: str = combine_arm_and_gripper_xml(
            ArmType.YAM,
            GripperType.from_string_name(gripper_type),
        )
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_path(xml_path)
        self._kinematics: Kinematics = Kinematics(xml_path, GRASP_SITE_NAME)
        # Shape `(6,)` action order의 MuJoCo qpos address를 저장
        self._arm_qpos_addresses: NDArray[np.int64] = np.asarray(
            [
                self._model.jnt_qposadr[
                    mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{joint_index}')
                ]
                for joint_index in range(1, ARM_DOF + 1)
            ],
            dtype=np.int64,
        )

    def pose(
        self,
        arm_joints: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        '''Shape `(6,)` arm joint의 grasp_site position `(3,)`과 xyzw quaternion `(4,)`을 반환한다.'''
        joints: NDArray[np.float64] = np.asarray(arm_joints, dtype=np.float64)
        if joints.shape != (ARM_DOF,):
            raise ValueError(f'arm_joints must have shape (6,), got {joints.shape}')
        # Gripper finger는 grasp_site 위치에 영향을 주지 않으므로 0으로 둠
        qpos: NDArray[np.float64] = np.zeros(self._model.nq, dtype=np.float64)
        qpos[self._arm_qpos_addresses] = joints
        # Shape `(nq,)` qpos에서 shape `(4, 4)` robot base 기준 transform으로 변환
        transform: NDArray[np.float64] = np.asarray(self._kinematics.fk(qpos), dtype=np.float64)
        quaternion_wxyz: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            quaternion_wxyz,
            np.ascontiguousarray(transform[:3, :3]).ravel(),
        )
        # MuJoCo wxyz를 scipy와 ROS가 사용하는 xyzw 순서로 변환
        return np.concatenate((transform[:3, 3], quaternion_wxyz[1:], quaternion_wxyz[:1]))
