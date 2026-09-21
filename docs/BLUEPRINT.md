# Blueprint

## 범위

이 repository는 Meta Quest 3 teleoperation과 VLA inference를 동일한 runtime contract로 실행하며 `execution_target`으로 실제 YAM과 MuJoCo YAM을 전환한다. 실행 entry point는 [`yam.ipynb`](../yam.ipynb#L1), component 조합 entry point는 [`create_session()`](../yam_control/factory.py#L351)이다.

## Repository 구조

```text
yam/
├── yam.ipynb — configuration과 episode 실행 entry point
├── yam_control/
│   ├── config.py — configuration type과 cross-field validation
│   ├── types.py — observation, action과 episode data contract
│   ├── interfaces.py — runtime component Protocol
│   ├── factory.py — configuration에서 concrete component 조합
│   ├── session.py — episode lifecycle과 control loop orchestration
│   ├── robot/ — 실제 YAM/MuJoCo adapter와 kinematics
│   ├── teleop/ — Quest transport, controller mapping과 IK
│   ├── camera/ — UVC capture, frame processing과 multi-camera lifecycle
│   ├── data/ — episode encoding과 yam-abc recorder adapter
│   ├── policy/ — VLA와 action chunk execution boundary
│   └── safety/ — action validation boundary
├── tests/ — unit, integration과 headless simulation test
└── third_party/yam-abc-reproduce/ — pin된 yam-abc, I2RT와 OpenPI source
```

설치와 실행은 [README](../README.md), 설계 결정은 [Design](./DESIGN.md), 검증 범위는 [Testing](./TESTING.md)을 참조한다.

## Runtime 관계

```text
RobotBackend ── RobotObservation ──► ActionProducer
      ▲                                  │
      │                                  ▼
      └──────── accepted RobotAction ◄─ SafetyGate
                         │
                         ▼
                  EpisodeRecorder
```

[`RunSession`](../yam_control/session.py#L16)이 component lifecycle과 control tick 순서를 소유한다. [`RuntimeDependencies`](../yam_control/factory.py#L276)는 구현체 교체 지점이며, mode와 execution target 선택은 [`factory.py`](../yam_control/factory.py#L54)에만 집중한다.

## 주요 module

| Directory | 책임과 public interface |
|---|---|
| `yam_control/config.py` | [`RunConfig`](../yam_control/config.py#L254), [`build_run_config()`](../yam_control/config.py#L279), [`with_camera_config()`](../yam_control/config.py#L332)가 실행 configuration을 정의한다. |
| `yam_control/types.py` | [`RobotObservation`](../yam_control/types.py#L30), [`RobotAction`](../yam_control/types.py#L39), [`ActionChunk`](../yam_control/types.py#L46)가 component 간 data contract다. |
| `yam_control/interfaces.py` | [`RobotBackend`](../yam_control/interfaces.py#L10), [`ActionProducer`](../yam_control/interfaces.py#L59), [`SafetyGate`](../yam_control/interfaces.py#L119), [`EpisodeRecorder`](../yam_control/interfaces.py#L137)를 정의한다. |
| `yam_control/robot/` | [`I2RTRobotBackend`](../yam_control/robot/i2rt_adapter.py#L103)가 실제 YAM과 MuJoCo YAM을 같은 interface로 감싼다. [`YamEndEffectorKinematics`](../yam_control/robot/kinematics.py#L17)는 recorder와 teleoperation이 공유하는 FK를 제공한다. |
| `yam_control/teleop/` | [`Quest3ActionProducer`](../yam_control/teleop/quest3.py#L362)가 Quest frame lifecycle을 관리하고 [`YamQuestRetargeter`](../yam_control/teleop/yam_retargeter.py#L100)가 controller target을 joint action으로 변환한다. |
| `yam_control/camera/` | [`V4L2RGBCamera`](../yam_control/camera/v4l2.py#L20)가 한 device를 읽고 [`CameraRig`](../yam_control/camera/rig.py#L24)가 role별 worker lifecycle을 관리한다. |
| `yam_control/data/` | [`YamABCRecorderAdapter`](../yam_control/data/recorder.py#L74)가 session recorder contract를 yam-abc episode format에 연결한다. |
| `yam_control/policy/` | [`PiBackend`](../yam_control/policy/pi.py#L14)가 VLA loading boundary를, [`OpenLoopChunkExecutor`](../yam_control/policy/rtc.py#L9)가 chunk 실행 contract를 제공한다. |
| `yam_control/safety/` | [`PassThroughSafetyGate`](../yam_control/safety/gates.py#L21)와 [`SweptPathSafetyGate`](../yam_control/safety/gates.py#L41)가 action 승인 정책을 제공한다. |

## Input/output contract

- `RobotObservation.state`: shape `(S,)`; `S`는 robot state dimension이며 단일 YAM에서 arm joint 6개와 normalized gripper 1개다.
- `RobotObservation.images[role]`: shape `(I_h, I_w, 3)` RGB image; `I_h`와 `I_w`는 image height와 width다.
- `RobotAction.values`: shape `(S,)` target state다.
- `ActionChunk.values`: shape `(H, S)`; `H`는 action prediction horizon이다.
- `QuestPose.position`: shape `(3,)` position vector다.
- `QuestPose.orientation_xyzw`: shape `(4,)` quaternion이다.
- Raw EE pose: shape `(N, 7)`; `N`은 episode frame 수이며 각 row는 position `(x, y, z)`와 quaternion `(qx, qy, qz, qw)`다.

## Lifecycle과 invariant

- [`RunSession.connect()`](../yam_control/session.py#L47) 이후에만 episode를 준비하며 [`RunSession.close()`](../yam_control/session.py#L239)가 모든 resource를 해제한다.
- 실제 robot은 작업자 environment reset과 `episode_initial_pose` 이동을 recording 밖에서 수행한다.
- Recorder는 SafetyGate를 통과해 backend에 전달된 action만 저장한다.
- Hold step, initial pose 이동과 environment reset 구간은 episode sample에 포함하지 않는다.
- Camera role은 episode 전체에서 고정되고 stale frame은 재사용하지 않는다.
- 실제 YAM과 MuJoCo YAM 전환은 `execution_target` 변경만으로 수행하며 downstream component contract는 유지한다.
- Leader Arm과 GR00T는 public configuration에 남아 있지만 실행 시 명시적으로 `NotImplementedError`를 발생시킨다.
