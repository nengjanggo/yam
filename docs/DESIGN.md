# Design

## Configuration 축과 component 조합

Run mode, action source, execution target, SafetyGate, recording과 RTC는 독립된 configuration 축으로 유지한다. [`create_session()`](../yam_control/factory.py#L351)이 이 축을 concrete component로 한 번만 변환하므로 runtime component는 다른 축의 조건문을 반복하지 않는다.

Teleoperation에서는 RTC가 의미 없으므로 `use_rtc`를 무시하고, MuJoCo에서는 실제 camera image와 simulation state가 섞인 dataset을 만들지 않도록 recording을 비활성화한다. 이 normalization은 [`build_run_config()`](../yam_control/config.py#L279)에서 수행한다.

## 실제 YAM과 MuJoCo의 공통 contract

실제 YAM과 I2RT `SimRobot`은 [`I2RTRobotBackend`](../yam_control/robot/i2rt_adapter.py#L103) 뒤에서 같은 observation/action contract를 사용한다. Simulation 전용 동작은 environment reset과 visualization에 제한해 action source, recorder와 session을 target별로 복제하지 않는다.

## Quest Browser 기반 teleoperation

Quest input은 Unity application 대신 WebXR을 사용한다. Browser만으로 배포할 수 있고 Meta WebXR의 `gripSpace`, `Gamepad`와 passthrough를 직접 사용할 수 있기 때문이다. WebSocket relay와 WebRTC stream은 repository 내부의 [`mujoco_relay.py`](../yam_control/teleop/mujoco_relay.py#L1)가 제공해 외부 teleoperation repository에 대한 runtime dependency를 두지 않는다.

Controller pose는 HMD-relative pose가 아니라 WebXR `local-floor` reference space에서 읽는다. Clutch engage 시점의 HMD yaw만 robot frame calibration에 사용하고 이후 head motion을 controller delta에 섞지 않는다. [`ClutchPoseMapper`](../yam_control/teleop/clutch_pose_mapper.py#L126)는 누적 integration 대신 engage pose 기준의 absolute delta를 사용해 같은 controller pose가 같은 target을 만들도록 한다.

## YAM IK 정책

[`YamQuestRetargeter`](../yam_control/teleop/yam_retargeter.py#L100)는 position과 orientation을 함께 최적화하되 position을 우선한다. Wrist orientation이 완전히 도달할 수 없는 구간에서도 grasp point 이동을 유지하기 위한 선택이다. Joint limit과 elbow singularity 회피 제한을 IK 내부에 적용하고, 미수렴 해가 목표 오차를 줄일 때만 partial step을 허용한다.

Tick당 joint delta와 controller reach limit은 물리 hardware 차이를 조정할 수 있는 calibration knob로 유지한다. 제한을 제거하면 tracking spike나 unreachable target이 즉시 큰 robot command로 바뀔 수 있다.

## Safety 전략

SafetyGate는 action producer와 backend 사이에 단일 승인 지점으로 둔다. [`SweptPathSafetyGate`](../yam_control/safety/gates.py#L41)는 현재 configuration뿐 아니라 목표까지의 path를 검사하고, 위험을 감지하면 stop을 latch한다. Validator가 없을 때 안전하다고 추정하지 않고 실행을 차단한다.

`use_safety_gate=False`는 초기 bring-up을 위해 허용하지만 software safety를 대체하지 않는다. 실제 robot에서는 hardware emergency stop과 작업자 workspace 절차가 필수다.

## Concurrency와 timing

Camera capture와 Quest WebSocket 수신은 background worker가 담당하고 control loop는 최신 frame만 소비한다. I/O 지연이 control period를 직접 막지 않도록 하는 대신 stale frame을 명시적으로 거부한다. 구현 경계는 [`CameraRig`](../yam_control/camera/rig.py#L24)와 [`WebSocketQuestFrameReader`](../yam_control/teleop/quest3.py#L225)다.

Control loop는 이전 sleep 시간에 누적해서 기다리지 않고 deadline을 기준으로 다음 tick을 예약한다. 처리가 한 주기보다 길어지면 밀린 tick을 몰아서 실행하지 않는다. 이 정책은 robot command burst를 피하고 recorder timestamp와 control frequency의 의미를 유지한다.

## Quest video 경로

MuJoCo renderer와 실제 wrist camera는 target에 따라 같은 atomic JPEG/WebRTC 경로를 공유한다. 실제 wrist stream은 이미 capture한 `CameraFrame`을 재사용해 camera device를 두 번 열지 않는다. [`factory.py`](../yam_control/factory.py#L54)가 target별 video source를 선택하고 [`quest_client.js`](../yam_control/teleop/web/quest_client.js#L176)가 MuJoCo panel과 wrist panel의 presentation만 구분한다.

## Recording 정책

Dataset에는 action source가 제안한 action이 아니라 SafetyGate를 통과해 실제 backend에 전달된 action을 저장한다. Hold와 environment reset 구간을 제외해 observation/action pair의 의미를 유지한다. 측정 EE pose와 명령 EE pose를 모두 저장해 joint-space와 task-space 분석을 함께 지원한다.

Video encoder는 hardware encoder보다 `libx264`를 사용한다. Hardware encoder initialization이 Python GIL을 오래 점유해 I2RT motor communication timeout을 유발했기 때문이다. Encoder 선택은 [`yam_abc.py`](../yam_control/data/yam_abc.py#L49)에 고정한다.

## VLA와 RTC boundary

π0와 π0.5 loading은 [`PiBackend`](../yam_control/policy/pi.py#L14)의 injection boundary로 남겨 runtime import와 checkpoint dependency를 지연한다. RTC는 YAM-compatible upstream producer가 확정될 때 [`RTCActionProducerAdapter`](../yam_control/policy/rtc.py#L59)에 주입한다. 특정 simulation reference implementation을 YAM runtime으로 재작성하지 않는다.
