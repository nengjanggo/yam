# YAM Control Blueprint

## 범위

이 repository는 Meta Quest 3 WebXR controller를 사용하는 teleoperation과 π0 또는 π0.5 checkpoint inference를 하나의 실행 구조로 제공한다. Leader Arm teleoperation은 사용할 hardware가 없어 선택 시 `NotImplementedError`를 발생시킨다. GR00T inference, zero pose utility, Isaac Sim은 현재 구현 범위에서 보류한다.

## 외부 repository

- `third_party/yam-abc-reproduce` — Leader Arm, episode recording, LeRobot 변환, π policy client/server 코드를 제공한다.
- `third_party/yam-abc-reproduce/third_party/i2rt` — 실제 YAM과 MuJoCo YAM에 동일한 robot API를 제공한다.
- `third_party/yam-abc-reproduce/third_party/policy/openpi` — π0와 π0.5 checkpoint inference를 제공한다.
- `lerobot_yam` — clone하지 않으며 공식 I2RT SDK와 중복되는 hardware 구현을 사용하지 않는다.

현재 clone 상태는 `yam-abc-reproduce` commit `7d9f9d135a2b949de54a349856b65863f81319e1`과 그 repository가 pin한 I2RT submodule commit `5d47b358bafb30c65e397f2ece506550a0db4594`이다. Quest UI와 clutch mapping은 local module로 제공한다.

WebSocket relay, WebRTC signaling, H.264 우선순위와 MuJoCo JPEG camera track은 `yam_control.teleop.mujoco_relay`가 직접 제공한다. WebRTC track은 `CAM_FPS`와 동일한 FPS로 전송해 불필요한 duplicate frame을 줄인다. Local Quest Web UI는 MuJoCo video를 자동 연결하고 WebXR controller frame을 최대 30 Hz로 전송한다. World-locked video panel은 eye-level보다 `0.75 m` 낮게 배치한다. MuJoCo RGB/WebRTC video에는 alpha channel이 없으므로 WebGL shader가 검은 배경을 chroma key로 제거해 Quest passthrough를 표시한다. TLS certificate는 ignore된 `certs/`에 보관한다.

WebXR controller 처리는 Meta 공식 [`webxr-first-steps`](https://github.com/meta-quest/webxr-first-steps)에서 사용하는 WebXR input source 방식과 WebXR 표준 `gripSpace`, `XRFrame.getPose()`, `Gamepad` API를 따른다. Meta repository 전체는 clone하지 않으며 tutorial build system, Three.js와 asset을 dependency로 추가하지 않는다.

## Directory 구조

```text
yam/
├── README.md — 다른 PC의 clone, dependency, TLS, firewall, CAN 설정과 실기 검증 전 주의점을 안내한다.
├── yam.ipynb — configuration, session 생성, episode 준비와 실행을 순서대로 수행한다.
├── blueprint.md — architecture, configuration, module 책임과 사용상 주의점을 기록한다.
├── pyproject.toml — yam-control과 vendored I2RT dependency를 정의한다.
├── uv.lock — Python 3.12 notebook과 test dependency version을 고정한다.
├── .gitignore — local Python environment와 bytecode generated file을 제외한다.
├── certs/ — ignore된 local Quest relay TLS certificate와 private key를 보관한다.
├── configs/
│   └── README.md — robot, camera, Quest 3, SafetyGate configuration을 추가할 위치를 설명한다.
├── yam_control/
│   ├── __init__.py — notebook과 외부 사용자가 호출할 public API를 노출한다.
│   ├── config.py — mode별 discriminated configuration과 조합 validation을 정의한다.
│   ├── types.py — RobotObservation, RobotAction, ActionChunk와 episode 상태 type을 정의한다.
│   ├── interfaces.py — robot, teleop, VLA, SafetyGate, recorder의 공통 Protocol을 정의한다.
│   ├── factory.py — configuration을 실제 구현체 조합으로 한 번만 변환한다.
│   ├── session.py — episode prepare, manual reset 대기, 실행, 종료 state machine을 관리한다.
│   ├── robot/
│   │   ├── __init__.py — robot backend public API를 노출한다.
│   │   ├── i2rt_adapter.py — I2RT 실제 YAM과 MuJoCo YAM을 지연 연결하는 adapter를 제공한다.
│   │   └── mujoco_viewer.py — SimRobot state를 passive viewer와 Quest stream용 offscreen frame에 동기화한다.
│   ├── teleop/
│   │   ├── __init__.py — teleoperation source public API를 노출한다.
│   │   ├── clutch_pose_mapper.py — engage 기준 controller delta를 결정적인 end-effector target으로 변환한다.
│   │   ├── leader.py — 지원하지 않는 Leader Arm 선택을 명시적으로 거부한다.
│   │   ├── mujoco_relay.py — WebSocket relay, WebRTC signaling과 MuJoCo JPEG camera track을 제공한다.
│   │   ├── quest3.py — WebXR background reader와 clutch/stale-frame ActionProducer를 제공한다.
│   │   ├── quest3_probe.py — robot 연결 없이 Quest 3 xr_frame 입력을 검증한다.
│   │   ├── yam_retargeter.py — local clutch mapping과 I2RT YAM FK/IK를 결합한다.
│   │   └── web/ — Quest entrypoint HTML, WebXR client와 controller 입력 module을 제공한다.
│   ├── policy/
│   │   ├── __init__.py — VLA backend public API를 노출한다.
│   │   ├── pi.py — π0와 π0.5 backend의 지연 checkpoint loading 경계를 제공한다.
│   │   ├── groot.py — GR00T 선택 시 명시적인 NotImplementedError를 발생시킨다.
│   │   └── rtc.py — open-loop chunk executor와 upstream RTC ActionProducer adapter를 제공한다.
│   ├── safety/
│   │   ├── __init__.py — SafetyGate public API를 노출한다.
│   │   └── gates.py — PassThroughSafetyGate와 MuJoCo SafetyGate 연결 지점을 제공한다.
│   └── data/
│       ├── __init__.py — recorder public API를 노출한다.
│       └── recorder.py — NullRecorder와 yam-abc-reproduce recorder adapter를 제공한다.
├── tests/
│   ├── test_clutch_pose_mapper.py — translation, rotation, reach limit와 re-engage mapping을 검증한다.
│   ├── test_config.py — mode별 configuration 정규화와 validation을 검증한다.
│   ├── test_i2rt_adapter.py — visualizer lifecycle과 Quest stream의 passive viewer 생략을 검증한다.
│   ├── test_mujoco_relay.py — MuJoCo frame cache, WebRTC track과 relay protocol을 검증한다.
│   ├── test_quest3.py — xr_frame parsing, background reader, input probe와 clutch safety를 검증한다.
│   ├── test_session.py — fake component로 episode state와 initial pose 흐름을 검증한다.
│   ├── test_yam_retargeter.py — headless YAM model로 FK/IK와 gripper mapping을 검증한다.
│   ├── quest_client_self_check.cjs — fake browser로 Quest Web UI의 camera 요청과 xr_frame 전송을 검증한다.
│   └── quest_input_self_check.cjs — fake WebXR object로 xr_frame controller와 viewer payload를 검증한다.
└── third_party/
    └── yam-abc-reproduce/ — I2RT 공식 end-to-end pipeline을 원본 상태로 보관한다.
```

## Configuration type

```text
RunMode = 'teleop' | 'inference'
TeleopSourceType = 'leader' | 'quest3'
VLAType = 'pi0' | 'pi0.5' | 'groot'
ExecutionTarget = 'real' | 'mujoco'

RunConfig
├── TeleopRunConfig
│   ├── mode = 'teleop'
│   ├── teleop_source
│   └── save_teleop_data
└── InferenceRunConfig
    ├── mode = 'inference'
    ├── vla_type
    ├── checkpoint_uri
    ├── checkpoint_revision
    ├── policy_config_name
    └── use_rtc

CommonConfig
├── execution_target
├── use_safety_gate
├── control_hz
├── RobotConfig
│   └── episode_initial_pose = (0.0, 1.2, 0.9, 0.0, 0.0, 0.0, 1.0)
├── QuestConfig
│   ├── relay_host, relay_port, controller_hand, display_mode
│   ├── translation_scale, rotation_scale
│   ├── position_reach_limit_m, rotation_reach_limit_rad
│   ├── max_joint_delta_rad, max_frame_age_s
│   ├── stream_frame_path, stream_width, stream_height
│   ├── stream_fps, stream_jpeg_quality
│   └── r_calib
└── CameraConfig
```

Notebook의 전역 `USE_RTC` 값은 `mode='inference'`일 때만 `InferenceRunConfig`에 반영한다. `mode='teleop'`이면 값을 변경하지 않아도 RTC를 사용하지 않고 정상적으로 teleoperation을 실행한다.

`CAN_CHANNEL`은 Linux SocketCAN network interface 이름이다. 예를 들어 follower가 연결된 adapter가 `can0`이면 `CAN_CHANNEL='can0'`을 사용한다. `execution_target='mujoco'`에서는 follower `CAN_CHANNEL`을 열지 않는다.

`GRIPPER_TYPE`은 YAM end effector hardware 종류이다. 기본값 `linear_4310`은 DM4310 기반 linear gripper를 의미하며 실제 장착된 gripper와 반드시 일치해야 한다.

## 실행 pipeline

```text
ObservationProvider
        │
        ▼
┌────────────────────────────────────┐
│              RunMode               │
│                                    │
│ teleop                             │
│   Quest 3                          │
│       → TeleopActionProducer       │
│                                    │
│ inference                          │
│   π0 / π0.5 / GR00T                │
│       → VLABackend                 │
│       → ChunkExecutor              │
└────────────────┬───────────────────┘
                 │ RobotAction
                 ▼
             SafetyGate
                 │ accepted RobotAction
                 ▼
            ActionExecutor
                 │
          ┌──────┴──────┐
          ▼             ▼
       real YAM      MuJoCo YAM
                 │
                 ▼
              Recorder
```

Recorder는 action source가 제안한 action이 아니라 SafetyGate를 통과해 실제 backend로 전달된 action을 저장한다. Initial pose 이동과 작업자의 환경 reset 시간은 episode recording에 포함하지 않는다.

## 공통 data shape

- `RobotObservation.state` — shape `(S,)`; `S`는 robot action dimension이며 단일 YAM에서는 `7`이다.
- `RobotObservation.images[role]` — shape `(I_h, I_w, 3)`; `I_h`와 `I_w`는 image height와 width이다.
- `RobotAction.values` — shape `(S,)`; 단일 YAM에서는 arm joint 6개와 normalized gripper 1개이다.
- `ActionChunk.values` — shape `(H, S)`; `H`는 prediction horizon의 action step 수이다.
- `QuestPose.position` — shape `(3,)`; Quest reference frame의 controller 위치이다.
- `QuestPose.orientation_xyzw` — shape `(4,)`; Quest reference frame의 controller quaternion이다.

## 모듈별 책임

### RobotBackend

Observation 읽기, `(S,)` action 실행, episode initial pose 이동, environment reset, current pose hold와 resource 해제만 담당한다. 실제 YAM과 MuJoCo YAM 선택은 factory에서 수행하며 나머지 module은 execution target을 알지 않는다.

MuJoCo target은 I2RT `SimRobot`을 action backend로 사용한다. `display_mode='robot_camera'`이면 `MujocoRobotViewer`가 동일 XML을 EGL offscreen renderer로 그려 atomic JPEG file을 갱신하며 desktop passive viewer는 열지 않는다. Offscreen camera는 `episode_initial_pose`가 반영된 현재 robot geometry 중심을 바라보고 `2.0 × model.stat.extent` 거리, `0°` azimuth와 `-30°` elevation에서 전체 robot을 표시한다. `joint1=0`에서 robot이 향하는 MuJoCo `+x` 방향과 작업자 정면이 일치하도록 robot 뒤쪽의 약간 높은 위치에서 바라보는 구도이다. 그 외 mode에서는 passive viewer에 measured state를 동기화한다. EGL과 desktop GL context를 동시에 만들지 않으므로 Notebook process의 context 충돌을 방지한다. Viewer와 JPEG는 visualization만 담당하며 control state의 source가 아니다.

구현 위치: [`RobotBackend`](./yam_control/interfaces.py#L10), [`I2RTRobotBackend`](./yam_control/robot/i2rt_adapter.py#L81), [`MujocoRobotViewer`](./yam_control/robot/mujoco_viewer.py#L34), [`MuJoCo visualizer 선택`](./yam_control/factory.py#L72)

### TeleopSource

매 control tick에 `(S,)` RobotAction 하나를 생성한다. Quest 3 source는 WebXR pose 수신과 YAM IK retargeting을 분리한다. Leader source는 configuration type에는 남겨 두지만 session 생성 시 `NotImplementedError`로 거부한다.

구현 위치: [`ActionProducer`](./yam_control/interfaces.py#L59), [`Leader source 거부`](./yam_control/factory.py#L211), [`Quest3ActionProducer`](./yam_control/teleop/quest3.py#L277)

### Quest 3 운용

작업자는 Quest 3 headset을 착용하고 `immersive-ar` passthrough로 workspace와 controller tracking을 유지한다. MuJoCo teleoperation에서는 world-locked `MuJoCo` video panel로 simulator 화면을 함께 확인한다. `MujocoFrameReader`가 atomic JPEG file을 읽고 `MujocoCameraTrack`이 `top` camera H.264 WebRTC stream으로 Quest에 전송한다.

기본 stream은 `640×480`, `15 FPS`, JPEG quality `70`이다. WebRTC 연결은 PC와 Quest 사이의 LAN peer-to-peer 경로이므로 외부 인터넷 traffic을 발생시키지 않지만, local Wi-Fi bandwidth와 PC의 MuJoCo render/JPEG/H.264 encode 부하는 추가된다. 화질보다 latency가 중요한 검증 단계에서는 이 기본값을 유지하고, 끊김이 있으면 먼저 resolution 또는 FPS를 낮춘다. Local Web UI는 `immersive-ar`만 허용하며, video rendering이 실패해도 controller frame 전송을 먼저 수행한다.

`quest_input.js`는 `XRSession.inputSources`의 left/right controller `gripSpace`를 `XRFrame.getPose()`로 `local-floor` reference space에서 읽고 `Gamepad.buttons`와 `Gamepad.axes`를 기존 `xr_frame` JSON protocol로 직렬화한다. Controller `gripSpace`가 없거나 pose tracking이 유효하지 않은 frame은 전송 대상에서 제외한다. WebXR `local-floor` controller pose는 HMD frame이 아니라 고정된 reference space 기준이다. 따라서 controller pose를 HMD-relative pose로 재변환하지 않고, clutch engage 시점의 HMD yaw만 `r_calib` alignment에 적용한다. 이후에는 local `ClutchPoseMapper`가 engage 시점 기준의 절대 controller delta를 계산하므로 headset을 착용한 작업자의 head motion은 robot action에 섞이지 않는다. Stale WebXR frame을 감지하면 clutch를 한 번 release하기 전까지 stop 상태를 유지한다.

`WebSocketQuestFrameReader`는 relay를 background thread에서 읽고 control loop에는 가장 최근 frame만 제공한다. WebXR tracking이 선택한 controller 또는 HMD pose를 일시적으로 누락한 frame은 건너뛰되 다음 유효 frame을 계속 수신한다. Local `ClutchPoseMapper`는 같은 controller pose가 같은 raw target을 만들도록 translation과 rotation을 engage pose에서 매 frame 다시 계산한다. Position과 rotation reach limit은 current end-effector pose 기준으로 target만 제한하며 mapping 기준을 흡수하거나 누적하지 않는다. `YamQuestRetargeter`는 이 target에 I2RT `Kinematics`를 적용하고 arm joint limit과 tick당 `max_joint_delta_rad`을 적용한다. Quest trigger `0=open, 1=closed`를 I2RT normalized gripper `0=closed, 1=open`에 맞게 `1 - trigger`로 변환한다. `Quest3ActionProducer.diagnostics()`는 episode 종료 후 clutch control step 수, hold 사유, action 변화 step 수와 최대 action delta뿐 아니라 reader thread 상태, 유효 frame 수, tracking 누락 frame 수와 최신 frame age를 제공한다.

`quest3_probe.py`는 일상 실행 Notebook에서 호출하지 않는다. Controller motion, trigger, grip 또는 relay camera 광고를 별도로 진단해야 할 때만 command-line에서 실행한다. 일반 실행에서는 `Quest3ActionProducer.connect()`가 첫 유효 `xr_frame`을 기다리므로 Quest Browser WebXR 입력 연결을 `3. Quest 3 및 MuJoCo 연결` cell에서 확인한다.

[`Quest3ActionProducer.diagnostics()`](./yam_control/teleop/quest3.py#L420)의 `retargeter` 항목은 [`YamQuestRetargeter.diagnostics()`](./yam_control/teleop/yam_retargeter.py#L128)를 전달한다. `ik_attempt_count`와 `ik_failure_count`는 IK 시도·실패 횟수, `ik_solve_mean_ms`와 `ik_solve_max_ms`는 IK 계산 시간이다. `joint_delta_clipped_step_count`와 `joint_limit_clipped_step_count`는 각 제한이 적용된 성공한 IK step 수다. `target_to_commanded_ee_position_mean_m/max_m`와 `target_to_commanded_ee_orientation_mean_rad/max_rad`는 reach limit 이후 IK 목표와 최종 명령의 FK 간 거리·회전각이다. 실패 시 반환하는 hold 명령도 포함하며 실제 로봇의 측정 추종 오차나 reach limit 이전 원래 목표의 오차는 아니다. 진단값은 episode마다 초기화하고 clutch 재진입 시에는 유지한다. IK 시도가 없으면 시간·오차는 `None`이다. 오차 계산은 IK 시도마다 FK 1회를 추가하며 이 비용은 control loop processing 시간에 포함되고 IK 계산 시간에는 포함되지 않는다.

구현 위치: [`QuestConfig`](./yam_control/config.py#L53), [`MuJoCo relay`](./yam_control/teleop/mujoco_relay.py#L54), [`Quest Web UI`](./yam_control/teleop/web/index.html#L1), [`Quest client`](./yam_control/teleop/web/quest_client.js#L1), [`WebXR input module`](./yam_control/teleop/web/quest_input.js#L63), [`WebSocketQuestFrameReader`](./yam_control/teleop/quest3.py#L199), [`Quest3ActionProducer`](./yam_control/teleop/quest3.py#L330), [`ClutchPoseMapper`](./yam_control/teleop/clutch_pose_mapper.py#L126), [`YamQuestRetargeter`](./yam_control/teleop/yam_retargeter.py#L63), [`Quest 3 probe`](./yam_control/teleop/quest3_probe.py#L54), [`ClutchPoseMapper test`](./tests/test_clutch_pose_mapper.py#L79), [`MuJoCo relay test`](./tests/test_mujoco_relay.py#L52), [`Quest 3 test`](./tests/test_quest3.py#L167), [`WebXR input self-check`](./tests/quest_input_self_check.cjs#L10), [`YAM retargeter test`](./tests/test_yam_retargeter.py#L17)

### VLABackend

RobotObservation을 받아 shape `(H, S)` ActionChunk를 반환한다. π0와 π0.5는 checkpoint URI, revision, policy config를 별도로 받는다. GR00T type은 public configuration에 유지하지만 backend 생성 시 NotImplementedError를 발생시킨다.

구현 위치: [`VLABackend`](./yam_control/interfaces.py#L89), [`PiBackend`](./yam_control/policy/pi.py#L14), [`GrootBackend`](./yam_control/policy/groot.py#L4)

### ChunkExecutor

Inference에서 ActionChunk를 control tick별 RobotAction으로 변환한다. `use_rtc=False`이면 open-loop executor를 사용한다. `use_rtc=True`이면 upstream RTC implementation 전체를 `rtc_factory`로 주입하고 이 repository는 `RTCActionProducerAdapter` glue만 사용한다. Teleoperation에서는 전역 `USE_RTC` 값을 무시하고 ChunkExecutor를 만들지 않는다.

Physical Intelligence의 `real-time-chunking-kinetix`는 Kinetix simulation experiment reference이며 YAM용 π0/π0.5 drop-in runtime은 아니다. 따라서 이 repository에는 해당 알고리즘을 재작성하거나 repository를 clone하지 않았다. YAM observation/action contract와 호환되는 upstream RTC producer가 확정되면 `RuntimeDependencies.rtc_factory`에 연결한다.

구현 위치: [`OpenLoopChunkExecutor`와 `RTCActionProducerAdapter`](./yam_control/policy/rtc.py#L9), [`RTC 선택`](./yam_control/factory.py#L211)

### SafetyGate

`use_safety_gate=False`이면 PassThroughSafetyGate를 사용하고 `True`이면 MuJoCo model 기반 `ConfigurationValidator`가 주입된 SweptPathSafetyGate를 사용한다. SafetyGate는 joint limit, self-collision, table geometry와 현재 pose에서 목표 pose까지의 swept path를 확인하며 위험한 action을 backend에 전달하지 않고 stop을 latch한다. Validator가 주입되지 않은 상태에서는 안전하다고 가정하지 않고 실행을 차단한다.

구현 위치: [`SafetyGate`](./yam_control/interfaces.py#L119), [`SweptPathSafetyGate`](./yam_control/safety/gates.py#L41), [`factory 선택`](./yam_control/factory.py#L242)

### Recorder

`save_teleop_data=False`이면 NullRecorder를 사용하고 `True`이면 yam-abc-reproduce EpisodeRecorder adapter를 사용한다. Recorder는 teleoperation episode의 observation과 실제 실행 action만 저장하며 prepare와 manual environment reset 구간은 저장하지 않는다.

Camera 기종이 확정되기 전에는 yam-abc가 요구하는 camera frame encoding을 결정할 수 없으므로 `YamABCRecorderAdapter`에는 `StepEncoder`를 주입한다. Camera-aware recorder가 주입되지 않은 상태에서 `save_teleop_data=True`이면 recording 시작을 차단한다.

구현 위치: [`EpisodeRecorder`](./yam_control/interfaces.py#L137), [`NullRecorder`와 `YamABCRecorderAdapter`](./yam_control/data/recorder.py#L11), [`factory 선택`](./yam_control/factory.py#L256)

### Session

실제 robot은 `prepare_episode()`에서 `human_reset_pose`로 이동한 뒤 작업자의 수동 환경 reset을 기다리고 `run_prepared_episode()`에서 `episode_initial_pose`로 이동한 뒤 episode를 시작한다. MuJoCo는 environment와 robot을 자동 reset하고 episode boundary를 유지하면서 연속 실행할 수 있다.

[`RunSession.control_loop_diagnostics()`](./yam_control/session.py#L88)는 마지막 episode의 `completed_step_count`, `mean_processing_ms`, `max_processing_ms`, `mean_tick_period_ms`, `max_tick_period_ms`, `actual_control_hz`를 반환한다. Processing은 observation부터 recording까지, tick period는 sleep까지 포함한다. 초기 pose 이동과 episode 전후 준비·정리는 제외하며 완료한 step이 없으면 시간·주파수는 `None`이다. 기존 sleep 주기나 action 계산은 바꾸지 않고 측정만 추가한다.

구현 위치: [`RunSession`](./yam_control/session.py#L15), [`configuration`](./yam_control/config.py#L178), [`session test`](./tests/test_session.py#L245)

## Notebook cell 안내

`yam.ipynb`의 최상단 Markdown cell은 프로젝트 루트에서 실행하는 상대경로 기반 LAN HTTPS Quest relay command와 Quest Browser 접속 주소를 안내한다. 이후 각 code cell 앞에 `1. 실행 설정`, `2. Session 생성`, `3. Quest 3 및 MuJoCo 연결`, `4. Episode 초기화`, `5. Teleoperation 실행`, `6. Session 종료` Markdown 이름을 표시한다. 별도 작업자 안내 출력은 생성하지 않으며, `5. Teleoperation 실행`은 episode 결과, `retargeter`를 포함한 action source 진단값과 `control_loop` 시간 통계를 표시한다. MuJoCo 여러 episode 연속 실행 시 진단값은 마지막 episode 기준이다.

`1. 실행 설정` cell은 `MUJOCO_GL='egl'`을 MuJoCo import 전에 설정하고 `QUEST_DISPLAY_MODE='robot_camera'`와 stream parameter를 정의한다. `3. Quest 3 및 MuJoCo 연결` cell에서 Quest용 offscreen stream을 시작하고 첫 유효 Quest input frame을 확인한다. `4. Episode 초기화` cell에서 initial pose와 environment를 reset한 뒤 `5. Teleoperation 실행` cell로 진행한다. `robot_camera` mode에서는 desktop passive viewer를 열지 않는다.

Notebook kernel과 Quest relay는 repository root의 Python 3.12 `.venv`를 사용한다. Root environment는 vendored I2RT를 editable local dependency로 사용한다. `certs/key.pem`과 `certs/cert.pem`은 gitignore 대상이며, 새 PC에서는 HTTPS 접속 주소를 SAN에 포함해 다시 생성해야 한다.

구현 위치: [`yam.ipynb`](./yam.ipynb#L8)

## 현재 구현 경계

- 자동 검증 완료: mode별 configuration, teleoperation에서 `USE_RTC` 무시, factory 조합, Leader source 거부, local WebXR controller/viewer frame 생성, local WebSocket relay와 MuJoCo WebRTC camera track, Quest `xr_frame` parser/input probe/background reader, Quest clutch/stale-frame safety, local absolute clutch mapping과 I2RT FK/IK를 사용한 YAM retargeting, I2RT MuJoCo YAM command, passive viewer state 동기화, EGL offscreen JPEG 생성, episode initial pose lifecycle, SafetyGate 거부 흐름, recorder adapter 경계, 이름이 지정된 notebook cell.
- Concrete adapter 필요: table scene를 포함한 MuJoCo `ConfigurationValidator`, camera driver와 `StepEncoder`, π0/π0.5 checkpoint loader, YAM-compatible upstream RTC producer.
- 명시적으로 보류: Leader Arm teleoperation, GR00T inference, zero pose utility, Isaac Sim, 실제 robot camera view.
- Test 범위: 실제 robot, Quest 3, camera, VLA를 연결하지 않고 fake component와 headless MuJoCo YAM model로 자동 검증한다. 새 Quest Web UI의 hardware input과 video panel은 작업자가 MuJoCo teleoperation으로 확인했다. 실제 robot command는 test에서 실행하지 않는다.

## 주의점

- 실제 robot command가 발생하는 test는 실행하지 않는다.
- 첫 Quest→MuJoCo 검증에서는 `use_safety_gate=False`를 사용한다. Table geometry와 `ConfigurationValidator`가 준비되기 전에는 실제 robot으로 변경하지 않는다.
- `teleop_source='leader'`는 session 생성 시 `NotImplementedError`로 거부된다.
- 실제 robot에서 `use_safety_gate=False`는 허용하지만 hardware emergency stop을 즉시 사용할 수 있어야 한다.
- 작업자가 robot workspace에 들어가는 동안 energized robot이 위험하면 `human_reset_pose`를 `episode_initial_pose`와 분리한다.
- Quest 3가 `immersive-vr` fallback으로 진입해 실제 workspace가 보이지 않으면 teleoperation을 시작하지 않는다.
- MuJoCo 화면이 Quest 3에 보이지 않으면 relay를 `yam_control.teleop.mujoco_relay` module로 실행했는지와 Quest Settings의 `Camera streaming`이 켜져 있는지 확인한다.
- LeRobot v3 변환을 실제 수집에 사용하기 전에 dataset finalization과 multi-episode round-trip을 검증한다.
