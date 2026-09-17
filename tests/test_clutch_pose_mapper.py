'''Local ClutchPoseMapper의 결정적인 clutch-relative mapping을 검증한다.'''

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from yam_control.teleop.clutch_pose_mapper import ClutchPoseMapper


def _quaternion_z(
    angle_rad: float,
) -> NDArray[np.float64]:
    '''Z축 angle_rad rotation을 shape `(4,)` wxyz quaternion으로 반환한다.'''
    half_angle_rad: float = angle_rad / 2.0
    return np.asarray(
        (
            float(np.cos(half_angle_rad)),
            0.0,
            0.0,
            float(np.sin(half_angle_rad)),
        ),
        dtype=np.float64,
    )


def _rotation_angle_rad(
    quaternion_wxyz: NDArray[np.float64],
) -> float:
    '''Shape `(4,)` unit quaternion의 shortest-path rotation angle을 반환한다.'''
    normalized_quaternion: NDArray[np.float64] = (
        quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    )
    return 2.0 * float(
        np.arccos(np.clip(abs(normalized_quaternion[0]), 0.0, 1.0))
    )


def _mapper(
    rotation: NDArray[np.float64] | None = None,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    position_reach_limit_m: float = 10.0,
    rotation_reach_limit_rad: float = 3.0,
) -> ClutchPoseMapper:
    '''Test configuration으로 ClutchPoseMapper를 생성한다.'''
    resolved_rotation: NDArray[np.float64] = (
        np.eye(3, dtype=np.float64)
        if rotation is None
        else rotation
    )
    return ClutchPoseMapper(
        rotation=resolved_rotation,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
        position_reach_limit_m=position_reach_limit_m,
        rotation_reach_limit_rad=rotation_reach_limit_rad,
    )


def _engage_identity(
    mapper: ClutchPoseMapper,
    ee_position: NDArray[np.float64] | None = None,
) -> None:
    '''Identity controller pose와 지정한 end-effector position으로 engage한다.'''
    resolved_ee_position: NDArray[np.float64] = (
        np.zeros(3, dtype=np.float64)
        if ee_position is None
        else ee_position
    )
    mapper.engage(
        controller_position=np.zeros(3, dtype=np.float64),
        controller_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        ee_position=resolved_ee_position,
        ee_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
    )


def test_target_before_engage_is_none(
) -> None:
    '''Engage 기준 pose가 없으면 target을 생성하지 않는지 검증한다.'''
    mapper: ClutchPoseMapper = _mapper()

    target: tuple[NDArray[np.float64], NDArray[np.float64]] | None = mapper.target(
        controller_position=np.zeros(3, dtype=np.float64),
        controller_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        current_ee_position=np.zeros(3, dtype=np.float64),
        current_ee_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
    )

    assert target is None


def test_translation_uses_rotation_and_scale(
) -> None:
    '''Controller translation에 Quest-to-robot rotation과 scale이 적용되는지 검증한다.'''
    # Shape `(3, 3)` rotation은 Quest +X를 robot +Y로 변환
    rotation: NDArray[np.float64] = np.asarray(
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    mapper: ClutchPoseMapper = _mapper(
        rotation=rotation,
        translation_scale=0.5,
    )
    ee_engage_position: NDArray[np.float64] = np.asarray((0.5, 0.0, 0.4), dtype=np.float64)
    _engage_identity(mapper, ee_position=ee_engage_position)

    target: tuple[NDArray[np.float64], NDArray[np.float64]] | None = mapper.target(
        controller_position=np.asarray((0.2, 0.0, 0.0), dtype=np.float64),
        controller_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        current_ee_position=ee_engage_position,
        current_ee_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
    )

    assert target is not None
    target_position: NDArray[np.float64]
    target_quaternion_wxyz: NDArray[np.float64]
    target_position, target_quaternion_wxyz = target
    np.testing.assert_allclose(target_position, (0.5, 0.1, 0.4), atol=1e-8)
    np.testing.assert_allclose(target_quaternion_wxyz, (1.0, 0.0, 0.0, 0.0), atol=1e-8)


def test_rotation_scale_is_absolute_from_engage_pose(
) -> None:
    '''Controller의 절대 90도 delta에 rotation scale 0.5가 항상 45도로 적용되는지 검증한다.'''
    mapper: ClutchPoseMapper = _mapper(rotation_scale=0.5)
    _engage_identity(mapper)
    controller_quaternion_wxyz: NDArray[np.float64] = _quaternion_z(np.pi / 2.0)

    first_target: tuple[NDArray[np.float64], NDArray[np.float64]] | None = mapper.target(
        controller_position=np.zeros(3, dtype=np.float64),
        controller_quaternion_wxyz=controller_quaternion_wxyz,
        current_ee_position=np.zeros(3, dtype=np.float64),
        current_ee_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
    )
    second_target: tuple[NDArray[np.float64], NDArray[np.float64]] | None = mapper.target(
        controller_position=np.zeros(3, dtype=np.float64),
        controller_quaternion_wxyz=controller_quaternion_wxyz,
        current_ee_position=np.zeros(3, dtype=np.float64),
        current_ee_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
    )

    assert first_target is not None
    assert second_target is not None
    np.testing.assert_allclose(first_target[1], _quaternion_z(np.pi / 4.0), atol=1e-8)
    np.testing.assert_allclose(second_target[1], first_target[1], atol=1e-8)


def test_reach_limits_do_not_absorb_controller_delta(
) -> None:
    '''Reach limit 이후 controller가 engage 근처로 돌아오면 절대 mapping이 즉시 복원되는지 검증한다.'''
    mapper: ClutchPoseMapper = _mapper(
        position_reach_limit_m=0.1,
        rotation_reach_limit_rad=0.2,
    )
    _engage_identity(mapper)
    identity_quaternion_wxyz: NDArray[np.float64] = np.asarray(
        (1.0, 0.0, 0.0, 0.0),
        dtype=np.float64,
    )

    limited_target: tuple[NDArray[np.float64], NDArray[np.float64]] | None = mapper.target(
        controller_position=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        controller_quaternion_wxyz=_quaternion_z(1.0),
        current_ee_position=np.zeros(3, dtype=np.float64),
        current_ee_quaternion_wxyz=identity_quaternion_wxyz,
    )
    returned_target: tuple[NDArray[np.float64], NDArray[np.float64]] | None = mapper.target(
        controller_position=np.asarray((0.05, 0.0, 0.0), dtype=np.float64),
        controller_quaternion_wxyz=_quaternion_z(0.1),
        current_ee_position=np.zeros(3, dtype=np.float64),
        current_ee_quaternion_wxyz=identity_quaternion_wxyz,
    )

    assert limited_target is not None
    assert returned_target is not None
    assert np.isclose(np.linalg.norm(limited_target[0]), 0.1)
    assert np.isclose(_rotation_angle_rad(limited_target[1]), 0.2)
    np.testing.assert_allclose(returned_target[0], (0.05, 0.0, 0.0), atol=1e-8)
    assert np.isclose(_rotation_angle_rad(returned_target[1]), 0.1)


def test_reengage_replaces_controller_and_end_effector_origins(
) -> None:
    '''Re-engage가 이전 mapping history 없이 controller와 end-effector 기준 pose를 교체하는지 검증한다.'''
    mapper: ClutchPoseMapper = _mapper()
    _engage_identity(mapper)
    new_ee_position: NDArray[np.float64] = np.asarray((0.4, -0.2, 0.3), dtype=np.float64)
    mapper.engage(
        controller_position=np.asarray((1.0, 2.0, 3.0), dtype=np.float64),
        controller_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        ee_position=new_ee_position,
        ee_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
    )

    target: tuple[NDArray[np.float64], NDArray[np.float64]] | None = mapper.target(
        controller_position=np.asarray((1.0, 2.0, 3.0), dtype=np.float64),
        controller_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        current_ee_position=new_ee_position,
        current_ee_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
    )

    assert target is not None
    np.testing.assert_allclose(target[0], new_ee_position, atol=1e-8)
    np.testing.assert_allclose(target[1], (1.0, 0.0, 0.0, 0.0), atol=1e-8)
