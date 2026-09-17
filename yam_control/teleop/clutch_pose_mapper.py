'''Quest controller delta를 clutch-relative end-effector target으로 변환한다.'''

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray


def _as_vector(
    values: NDArray[np.float64],
    dimension: int,
    field_name: str,
) -> NDArray[np.float64]:
    '''입력값을 지정한 shape의 float64 vector로 복사해 반환한다.'''
    vector: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    if vector.shape != (dimension,):
        raise ValueError(f'{field_name} must have shape ({dimension},)')
    return vector.copy()


def _normalize_quaternion(
    quaternion_wxyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    '''Shape `(4,)` wxyz quaternion을 unit quaternion으로 정규화한다.'''
    quaternion: NDArray[np.float64] = _as_vector(
        quaternion_wxyz,
        4,
        'quaternion_wxyz',
    )
    norm: float = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError('quaternion norm must be positive')
    return quaternion / norm


def _quaternion_multiply(
    left_wxyz: NDArray[np.float64],
    right_wxyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    '''두 shape `(4,)` wxyz quaternion을 합성해 shape `(4,)` quaternion을 반환한다.'''
    left: NDArray[np.float64] = _normalize_quaternion(left_wxyz)
    right: NDArray[np.float64] = _normalize_quaternion(right_wxyz)
    # Shape `(4,)` left와 right를 shape `(4,)` 합성 quaternion으로 변환
    product: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(product, left, right)
    return _normalize_quaternion(product)


def _quaternion_conjugate(
    quaternion_wxyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    '''Shape `(4,)` unit quaternion의 shape `(4,)` inverse를 반환한다.'''
    quaternion: NDArray[np.float64] = _normalize_quaternion(quaternion_wxyz)
    return np.asarray(
        (
            quaternion[0],
            -quaternion[1],
            -quaternion[2],
            -quaternion[3],
        ),
        dtype=np.float64,
    )


def _rotation_vector(
    quaternion_wxyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    '''Shape `(4,)` quaternion을 shortest-path shape `(3,)` rotation vector로 변환한다.'''
    quaternion: NDArray[np.float64] = _normalize_quaternion(quaternion_wxyz)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    # Shape `(4,)` quaternion을 shape `(3,)` axis-angle vector로 변환
    rotation_vector: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    mujoco.mju_quat2Vel(rotation_vector, quaternion, 1.0)
    return rotation_vector


def _quaternion_from_rotation_vector(
    rotation_vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    '''Shape `(3,)` rotation vector를 shape `(4,)` wxyz quaternion으로 변환한다.'''
    vector: NDArray[np.float64] = _as_vector(
        rotation_vector,
        3,
        'rotation_vector',
    )
    angle_rad: float = float(np.linalg.norm(vector))
    if angle_rad <= 1e-12:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    axis: NDArray[np.float64] = vector / angle_rad
    half_angle_rad: float = angle_rad / 2.0
    sine: float = float(np.sin(half_angle_rad))
    return np.asarray(
        (
            float(np.cos(half_angle_rad)),
            sine * axis[0],
            sine * axis[1],
            sine * axis[2],
        ),
        dtype=np.float64,
    )


def _rotation_matrix_quaternion(
    rotation: NDArray[np.float64],
) -> NDArray[np.float64]:
    '''Shape `(3, 3)` rotation matrix를 shape `(4,)` wxyz quaternion으로 변환한다.'''
    matrix: NDArray[np.float64] = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError('rotation must have shape (3, 3)')
    identity: NDArray[np.float64] = np.eye(3, dtype=np.float64)
    if not np.allclose(matrix.T @ matrix, identity, atol=1e-6):
        raise ValueError('rotation must be orthonormal')
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        raise ValueError('rotation determinant must be 1')
    # Shape `(3, 3)` matrix를 contiguous shape `(9,)`으로 변경해 MuJoCo에 전달
    quaternion_wxyz: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(
        quaternion_wxyz,
        np.ascontiguousarray(matrix).reshape(9),
    )
    return _normalize_quaternion(quaternion_wxyz)


class ClutchPoseMapper:
    '''Engage 시점 기준 controller delta를 결정적인 end-effector target으로 변환한다.'''

    def __init__(
        self,
        rotation: NDArray[np.float64],
        translation_scale: float,
        rotation_scale: float,
        position_reach_limit_m: float,
        rotation_reach_limit_rad: float,
    ) -> None:
        '''좌표계 rotation, motion scale과 current pose 기준 reach limit을 저장한다.'''
        positive_values: tuple[tuple[str, float], ...] = (
            ('translation_scale', translation_scale),
            ('rotation_scale', rotation_scale),
            ('position_reach_limit_m', position_reach_limit_m),
            ('rotation_reach_limit_rad', rotation_reach_limit_rad),
        )
        field_name: str
        value: float
        for field_name, value in positive_values:
            if value <= 0.0:
                raise ValueError(f'{field_name} must be positive')
        self._translation_scale: float = translation_scale
        self._rotation_scale: float = rotation_scale
        self._position_reach_limit_m: float = position_reach_limit_m
        self._rotation_reach_limit_rad: float = rotation_reach_limit_rad
        self._rotation_quaternion_wxyz: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
        self._rotation_inverse_quaternion_wxyz: NDArray[np.float64] = np.zeros(
            4,
            dtype=np.float64,
        )
        self._controller_engage_position: NDArray[np.float64] | None = None
        self._controller_engage_quaternion_wxyz: NDArray[np.float64] | None = None
        self._ee_engage_position: NDArray[np.float64] | None = None
        self._ee_engage_quaternion_wxyz: NDArray[np.float64] | None = None
        self.set_rotation(rotation)

    def set_rotation(
        self,
        rotation: NDArray[np.float64],
    ) -> None:
        '''Quest world vector를 robot base vector로 변환하는 shape `(3, 3)` rotation을 설정한다.'''
        rotation_quaternion_wxyz: NDArray[np.float64] = _rotation_matrix_quaternion(rotation)
        self._rotation_quaternion_wxyz = rotation_quaternion_wxyz
        self._rotation_inverse_quaternion_wxyz = _quaternion_conjugate(
            rotation_quaternion_wxyz
        )
        self._rotation: NDArray[np.float64] = np.asarray(
            rotation,
            dtype=np.float64,
        ).copy()

    def engage(
        self,
        controller_position: NDArray[np.float64],
        controller_quaternion_wxyz: NDArray[np.float64],
        ee_position: NDArray[np.float64],
        ee_quaternion_wxyz: NDArray[np.float64],
    ) -> None:
        '''Clutch engage 시 controller와 end-effector 기준 pose를 복사해 저장한다.'''
        self._controller_engage_position = _as_vector(
            controller_position,
            3,
            'controller_position',
        )
        self._controller_engage_quaternion_wxyz = _normalize_quaternion(
            controller_quaternion_wxyz
        )
        self._ee_engage_position = _as_vector(
            ee_position,
            3,
            'ee_position',
        )
        self._ee_engage_quaternion_wxyz = _normalize_quaternion(
            ee_quaternion_wxyz
        )

    def target(
        self,
        controller_position: NDArray[np.float64],
        controller_quaternion_wxyz: NDArray[np.float64],
        current_ee_position: NDArray[np.float64],
        current_ee_quaternion_wxyz: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        '''현재 controller pose에서 reach-limited end-effector position과 quaternion을 반환한다.'''
        if (
            self._controller_engage_position is None
            or self._controller_engage_quaternion_wxyz is None
            or self._ee_engage_position is None
            or self._ee_engage_quaternion_wxyz is None
        ):
            return None
        current_controller_position: NDArray[np.float64] = _as_vector(
            controller_position,
            3,
            'controller_position',
        )
        current_controller_quaternion_wxyz: NDArray[np.float64] = _normalize_quaternion(
            controller_quaternion_wxyz
        )
        current_position: NDArray[np.float64] = _as_vector(
            current_ee_position,
            3,
            'current_ee_position',
        )
        current_quaternion_wxyz: NDArray[np.float64] = _normalize_quaternion(
            current_ee_quaternion_wxyz
        )

        # Shape `(3,)` controller delta를 calibration과 scale로 robot base delta로 변환
        controller_delta_position: NDArray[np.float64] = (
            current_controller_position - self._controller_engage_position
        )
        target_position: NDArray[np.float64] = (
            self._ee_engage_position
            + self._rotation @ (self._translation_scale * controller_delta_position)
        )
        position_error: NDArray[np.float64] = target_position - current_position
        position_error_norm_m: float = float(np.linalg.norm(position_error))
        if position_error_norm_m > self._position_reach_limit_m:
            target_position = (
                current_position
                + position_error * (self._position_reach_limit_m / position_error_norm_m)
            )

        # Shape `(4,)` controller rotation delta를 robot base frame으로 conjugation
        controller_delta_quaternion_wxyz: NDArray[np.float64] = _quaternion_multiply(
            current_controller_quaternion_wxyz,
            _quaternion_conjugate(self._controller_engage_quaternion_wxyz),
        )
        mapped_delta_quaternion_wxyz: NDArray[np.float64] = _quaternion_multiply(
            _quaternion_multiply(
                self._rotation_quaternion_wxyz,
                controller_delta_quaternion_wxyz,
            ),
            self._rotation_inverse_quaternion_wxyz,
        )
        scaled_rotation_vector: NDArray[np.float64] = (
            self._rotation_scale * _rotation_vector(mapped_delta_quaternion_wxyz)
        )
        target_quaternion_wxyz: NDArray[np.float64] = _quaternion_multiply(
            _quaternion_from_rotation_vector(scaled_rotation_vector),
            self._ee_engage_quaternion_wxyz,
        )
        rotation_error_quaternion_wxyz: NDArray[np.float64] = _quaternion_multiply(
            target_quaternion_wxyz,
            _quaternion_conjugate(current_quaternion_wxyz),
        )
        rotation_error_vector: NDArray[np.float64] = _rotation_vector(
            rotation_error_quaternion_wxyz
        )
        rotation_error_norm_rad: float = float(np.linalg.norm(rotation_error_vector))
        if rotation_error_norm_rad > self._rotation_reach_limit_rad:
            limited_rotation_vector: NDArray[np.float64] = (
                rotation_error_vector
                * (self._rotation_reach_limit_rad / rotation_error_norm_rad)
            )
            target_quaternion_wxyz = _quaternion_multiply(
                _quaternion_from_rotation_vector(limited_rotation_vector),
                current_quaternion_wxyz,
            )
        return target_position, target_quaternion_wxyz
