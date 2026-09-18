'''Quest 3 controller motion을 I2RT YAM FK/IK로 joint action으로 변환한다.'''

from __future__ import annotations

import time

import mink
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
# I2RT Kinematics.ik와 같은 mink 미분 IK 설정
IK_DT_S: float = 0.01
IK_SOLVER: str = 'quadprog'
IK_DAMPING: float = 1e-4
IK_MAX_ITERS: int = 100
IK_POSITION_THRESHOLD_M: float = 1e-3
IK_ORIENTATION_THRESHOLD_RAD: float = 1e-3
IK_POSITION_COST: float = 1.0
# 팔꿈치가 완전히 펴진 특이점을 넘어 반대 branch로 넘어가지 않도록 IK의 joint3 상한에 두는 여유
ELBOW_SINGULARITY_MARGIN_RAD: float = 0.2
ELBOW_JOINT_NAME: str = 'joint3'


def _elbow_straight_angle(
    model: mujoco.MjModel,
) -> float:
    '''Shoulder(joint2) 축과 wrist(joint5) 축 거리가 최대가 되는 joint3 각도를 반환한다.'''
    data: mujoco.MjData = mujoco.MjData(model)
    shoulder_joint_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint2')
    elbow_joint_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ELBOW_JOINT_NAME)
    wrist_joint_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint5')
    elbow_qpos_address: int = int(model.jnt_qposadr[elbow_joint_id])
    lower: float = float(model.jnt_range[elbow_joint_id, 0])
    upper: float = float(model.jnt_range[elbow_joint_id, 1])
    # Shape `(K,)` joint3 후보마다 shoulder-wrist 거리를 계산
    candidates: NDArray[np.float64] = np.linspace(lower, upper, 629)
    distances: list[float] = []
    candidate: float
    for candidate in candidates:
        data.qpos[:] = 0.0
        data.qpos[elbow_qpos_address] = candidate
        mujoco.mj_kinematics(model, data)
        distances.append(float(np.linalg.norm(data.xanchor[wrist_joint_id] - data.xanchor[shoulder_joint_id])))
    return float(candidates[int(np.argmax(distances))])


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
        # Joint limit 안에서 위치를 우선하도록 orientation cost를 조정한 mink 미분 IK
        self._ik_orientation_cost: float = quest_config.ik_orientation_cost
        # IK 전용 model은 joint3 상한을 팔꿈치 특이점 앞으로 제한
        self._ik_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(xml_path)
        elbow_joint_id: int = mujoco.mj_name2id(self._ik_model, mujoco.mjtObj.mjOBJ_JOINT, ELBOW_JOINT_NAME)
        self._ik_model.jnt_range[elbow_joint_id, 1] = min(
            float(self._ik_model.jnt_range[elbow_joint_id, 1]),
            _elbow_straight_angle(self._ik_model) - ELBOW_SINGULARITY_MARGIN_RAD,
        )
        self._ik_configuration: mink.Configuration = mink.Configuration(self._ik_model)
        self._ik_task: mink.FrameTask = mink.FrameTask(
            frame_name=GRASP_SITE_NAME,
            frame_type='site',
            position_cost=IK_POSITION_COST,
            orientation_cost=self._ik_orientation_cost,
            lm_damping=1.0,
        )
        self._ik_limits: list[mink.Limit] = [mink.ConfigurationLimit(self._ik_model)]
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
        self.reset_diagnostics()

    def reset_diagnostics(
        self,
    ) -> None:
        '''새 episode의 IK 시간, 실패, clipping과 model FK 오차 누적값을 초기화한다.'''
        self._ik_attempt_count: int = 0
        self._ik_failure_count: int = 0
        self._ik_partial_step_count: int = 0
        self._ik_solve_total_ms: float = 0.0
        self._ik_solve_max_ms: float = 0.0
        self._joint_delta_clipped_step_count: int = 0
        self._joint_limit_clipped_step_count: int = 0
        self._target_to_commanded_ee_position_total_m: float = 0.0
        self._target_to_commanded_ee_position_max_m: float = 0.0
        self._target_to_commanded_ee_orientation_total_rad: float = 0.0
        self._target_to_commanded_ee_orientation_max_rad: float = 0.0

    def diagnostics(
        self,
    ) -> dict[str, int | float | None]:
        '''현재 episode의 IK와 명령 EE model FK 진단값을 반환한다.'''
        attempt_count: int = self._ik_attempt_count
        return {
            'ik_attempt_count': attempt_count,
            'ik_failure_count': self._ik_failure_count,
            'ik_partial_step_count': self._ik_partial_step_count,
            'ik_solve_mean_ms': self._ik_solve_total_ms / attempt_count if attempt_count else None,
            'ik_solve_max_ms': self._ik_solve_max_ms if attempt_count else None,
            'joint_delta_clipped_step_count': self._joint_delta_clipped_step_count,
            'joint_limit_clipped_step_count': self._joint_limit_clipped_step_count,
            'target_to_commanded_ee_position_mean_m': (
                self._target_to_commanded_ee_position_total_m / attempt_count if attempt_count else None
            ),
            'target_to_commanded_ee_position_max_m': (
                self._target_to_commanded_ee_position_max_m if attempt_count else None
            ),
            'target_to_commanded_ee_orientation_mean_rad': (
                self._target_to_commanded_ee_orientation_total_rad / attempt_count if attempt_count else None
            ),
            'target_to_commanded_ee_orientation_max_rad': (
                self._target_to_commanded_ee_orientation_max_rad if attempt_count else None
            ),
        }

    def _target_error(
        self,
        qpos: NDArray[np.float64],
        target_position: NDArray[np.float64],
        target_quaternion_wxyz: NDArray[np.float64],
    ) -> tuple[float, float]:
        '''Shape `(nq,)` qpos의 model FK와 목표 pose 사이 위치 오차와 회전각 오차를 반환한다.'''
        position: NDArray[np.float64]
        quaternion_wxyz: NDArray[np.float64]
        position, quaternion_wxyz = self._end_effector_pose(qpos)
        position_error_m: float = float(np.linalg.norm(target_position - position))
        quaternion_dot: float = float(np.dot(target_quaternion_wxyz, quaternion_wxyz))
        orientation_error_rad: float = float(2.0 * np.arccos(np.clip(abs(quaternion_dot), 0.0, 1.0)))
        return position_error_m, orientation_error_rad

    def _weighted_target_error(
        self,
        qpos: NDArray[np.float64],
        target_position: NDArray[np.float64],
        target_quaternion_wxyz: NDArray[np.float64],
    ) -> float:
        '''IK task cost와 같은 비율로 위치와 회전 오차를 결합한 목표 오차를 반환한다.'''
        position_error_m: float
        orientation_error_rad: float
        position_error_m, orientation_error_rad = self._target_error(
            qpos,
            target_position,
            target_quaternion_wxyz,
        )
        return float(np.hypot(
            IK_POSITION_COST * position_error_m,
            self._ik_orientation_cost * orientation_error_rad,
        ))

    def _record_target_error(
        self,
        qpos: NDArray[np.float64],
        target_position: NDArray[np.float64],
        target_quaternion_wxyz: NDArray[np.float64],
    ) -> None:
        '''명령 qpos의 model FK와 controller 목표 pose 사이 오차를 누적한다.'''
        position_error_m: float
        orientation_error_rad: float
        position_error_m, orientation_error_rad = self._target_error(
            qpos,
            target_position,
            target_quaternion_wxyz,
        )
        self._target_to_commanded_ee_position_total_m += position_error_m
        self._target_to_commanded_ee_position_max_m = max(
            self._target_to_commanded_ee_position_max_m,
            position_error_m,
        )
        self._target_to_commanded_ee_orientation_total_rad += orientation_error_rad
        self._target_to_commanded_ee_orientation_max_rad = max(
            self._target_to_commanded_ee_orientation_max_rad,
            orientation_error_rad,
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

    def _solve_ik(
        self,
        qpos: NDArray[np.float64],
        target_transform: NDArray[np.float64],
    ) -> tuple[bool, NDArray[np.float64]]:
        '''Joint limit 안에서 목표에 수렴하면 True와 qpos를, 아니면 False와 마지막 qpos를 반환한다.'''
        self._ik_configuration.update(qpos)
        self._ik_task.set_target(mink.SE3.from_matrix(target_transform))
        iteration_index: int
        for iteration_index in range(IK_MAX_ITERS):
            del iteration_index
            velocity: NDArray[np.float64] = mink.solve_ik(
                self._ik_configuration,
                [self._ik_task],
                IK_DT_S,
                IK_SOLVER,
                damping=IK_DAMPING,
                limits=self._ik_limits,
            )
            self._ik_configuration.integrate_inplace(velocity, IK_DT_S)
            # Shape `(6,)` task error를 position `(3,)`과 orientation `(3,)`으로 분리
            error: NDArray[np.float64] = self._ik_task.compute_error(self._ik_configuration)
            if (
                np.linalg.norm(error[:3]) <= IK_POSITION_THRESHOLD_M
                and np.linalg.norm(error[3:]) <= IK_ORIENTATION_THRESHOLD_RAD
            ):
                return True, self._ik_configuration.q.copy()
        return False, self._ik_configuration.q.copy()

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
        solve_start_s: float = time.perf_counter()
        success, solved_qpos = self._solve_ik(qpos, target_transform)
        solve_ms: float = (time.perf_counter() - solve_start_s) * 1000.0
        self._ik_attempt_count += 1
        self._ik_solve_total_ms += solve_ms
        self._ik_solve_max_ms = max(self._ik_solve_max_ms, solve_ms)
        if not success:
            self._ik_failure_count += 1
            # 수렴하지 못해도 현재 pose보다 목표에 가까워진 해는 사용해 가능한 만큼 따라감
            current_error: float = self._weighted_target_error(qpos, target_position, target_quaternion_wxyz)
            solved_error: float = self._weighted_target_error(solved_qpos, target_position, target_quaternion_wxyz)
            if solved_error >= current_error - 1e-6:
                self._record_target_error(qpos, target_position, target_quaternion_wxyz)
                return RobotAction(values=observation.state)
            self._ik_partial_step_count += 1
        # Shape `(nq,)` IK result에서 shape `(6,)` arm joint target을 추출
        solved_arm: NDArray[np.float64] = solved_qpos[self._arm_qpos_addresses]
        bounded_arm: NDArray[np.float64] = np.clip(
            solved_arm,
            self._arm_joint_limits[:, 0],
            self._arm_joint_limits[:, 1],
        )
        if np.any(np.abs(bounded_arm - solved_arm) > 1e-12):
            self._joint_limit_clipped_step_count += 1
        current_arm: NDArray[np.float64] = np.asarray(observation.state[:ARM_DOF], dtype=np.float64)
        requested_joint_delta: NDArray[np.float64] = bounded_arm - current_arm
        joint_delta: NDArray[np.float64] = np.clip(
            requested_joint_delta,
            -self._max_joint_delta_rad,
            self._max_joint_delta_rad,
        )
        if np.any(np.abs(joint_delta - requested_joint_delta) > 1e-12):
            self._joint_delta_clipped_step_count += 1
        commanded_arm: NDArray[np.float64] = current_arm + joint_delta
        # Shape `(nq,)` current qpos에 shape `(6,)` 명령 arm을 반영해 FK 오차를 계산
        commanded_qpos: NDArray[np.float64] = qpos.copy()
        commanded_qpos[self._arm_qpos_addresses] = commanded_arm
        self._record_target_error(commanded_qpos, target_position, target_quaternion_wxyz)
        commanded_gripper: float = 1.0 - float(np.clip(frame.trigger, 0.0, 1.0))
        action_values: tuple[float, ...] = (
            *(float(value) for value in commanded_arm),
            commanded_gripper,
        )
        return RobotAction(values=action_values)
