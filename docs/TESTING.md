# Testing

## 실행 명령

전체 Python test와 Quest browser self-check는 다음 명령으로 실행한다.

```bash
.venv/bin/python -m pytest -q
node tests/quest_input_self_check.cjs
node tests/quest_client_self_check.cjs
jq empty yam.ipynb
```

실제 robot command가 발생하는 test는 작성하거나 자동 실행하지 않는다.

## Unit test

- Configuration normalization과 지원하지 않는 조합: [`test_config.py`](../tests/test_config.py#L1)
- Image crop/pad, camera lifecycle, stale frame과 wrist stream: [`test_camera.py`](../tests/test_camera.py#L1)
- Clutch translation/rotation과 reach limit: [`test_clutch_pose_mapper.py`](../tests/test_clutch_pose_mapper.py#L1)
- Quest frame parsing, stale-frame safety와 episode button: [`test_quest3.py`](../tests/test_quest3.py#L1)
- YAM FK/IK와 gripper mapping: [`test_yam_retargeter.py`](../tests/test_yam_retargeter.py#L1)
- Recorder schema, EE pose와 encoder 선택: [`test_yam_abc_recorder.py`](../tests/test_yam_abc_recorder.py#L1)

## Integration 및 simulation test

- Session lifecycle, hold exclusion, deadline scheduling과 abort: [`test_session.py`](../tests/test_session.py#L1)
- I2RT adapter의 real/simulation boundary와 resource lifecycle: [`test_i2rt_adapter.py`](../tests/test_i2rt_adapter.py#L1)
- Relay protocol, JPEG cache와 WebRTC track: [`test_mujoco_relay.py`](../tests/test_mujoco_relay.py#L1)
- Fake WebXR/browser 환경의 input payload와 video layout: [`quest_input_self_check.cjs`](../tests/quest_input_self_check.cjs#L1), [`quest_client_self_check.cjs`](../tests/quest_client_self_check.cjs#L1)
- Headless MuJoCo model을 사용한 retargeting과 robot command path는 Python test suite에 포함한다.

## 실제 hardware 검증 현황

- Quest 3 controller로 MuJoCo YAM과 실제 YAM이 움직이는 경로를 확인했다.
- Quest 3에서 MuJoCo video panel과 passthrough를 확인했다.
- 실제 L515 RGB capture, image 변환과 yam-abc episode 저장·decode를 확인했다.
- 실제 YAM에서 initial pose와 episode 반복 workflow를 확인했다.

## Test gap

- 실제 wrist camera의 Quest panel 표시와 장시간 network 안정성
- Top/wrist camera 동시 장시간 recording과 LeRobot v3 multi-episode round trip
- Table geometry를 포함한 실제 `ConfigurationValidator`와 SafetyGate
- π0, π0.5 checkpoint inference와 YAM-compatible RTC
- Leader Arm, GR00T와 Isaac Sim은 미구현 상태이므로 검증하지 않았다.
