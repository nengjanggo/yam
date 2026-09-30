# Testing

## 실행 명령

전체 Python test와 Quest browser self-check는 다음 명령으로 실행한다.

```bash
.venv/bin/python -m pytest -q
node tests/quest_input_self_check.cjs
node tests/quest_client_self_check.cjs
jq empty teleop.ipynb
jq empty inference.ipynb
jq empty finetune.ipynb

cd third_party/yam-abc-reproduce/third_party/policy/openpi
uv run pytest -q \
  src/openpi/policies/yam_policy_test.py \
  src/openpi/training/yam_config_test.py \
  src/openpi/models/model_test.py::test_pi05_stop_gradient_vlm_prefix_preserves_loss_and_action_expert_gradients \
  src/openpi/models/model_test.py::test_pi05_rtc_prefix_is_preserved \
  src/openpi/training/weight_loaders_test.py
```

실제 robot command가 발생하는 test는 작성하거나 자동 실행하지 않는다.

## Unit test

- Configuration normalization과 지원하지 않는 조합: [`test_config.py`](../tests/test_config.py#L1)
- Image crop/pad, camera lifecycle, stale frame과 wrist stream: [`test_camera.py`](../tests/test_camera.py#L1)
- Clutch translation/rotation과 reach limit: [`test_clutch_pose_mapper.py`](../tests/test_clutch_pose_mapper.py#L1)
- Quest frame parsing, stale-frame safety와 episode button: [`test_quest3.py`](../tests/test_quest3.py#L1)
- YAM FK/IK와 gripper mapping: [`test_yam_retargeter.py`](../tests/test_yam_retargeter.py#L1)
- Recorder schema, EE pose와 encoder 선택: [`test_yam_abc_recorder.py`](../tests/test_yam_abc_recorder.py#L1)
- OpenPI WebSocket input/output mapping, RTC prefix handoff·underrun hold와 실제 YAM joint step 제한: [`test_pi_inference.py`](../tests/test_pi_inference.py#L1)
- Single-arm YAM image mask, DiT LoRA parameter filter, LoRA rank/alpha/rsLoRA override·validation, `pi05_base` BF16 direct restore equivalence와 VLM prefix stop-gradient 전후 loss/action expert gradient equivalence: [`yam_policy_test.py`](../third_party/yam-abc-reproduce/third_party/policy/openpi/src/openpi/policies/yam_policy_test.py#L1), [`yam_config_test.py`](../third_party/yam-abc-reproduce/third_party/policy/openpi/src/openpi/training/yam_config_test.py#L1), [`weight_loaders_test.py`](../third_party/yam-abc-reproduce/third_party/policy/openpi/src/openpi/training/weight_loaders_test.py#L1), [`model_test.py`](../third_party/yam-abc-reproduce/third_party/policy/openpi/src/openpi/models/model_test.py#L44)

## Integration 및 simulation test

- Session lifecycle, hold exclusion, deadline scheduling, abort와 작업자 button 기반 연속 episode: [`test_session.py`](../tests/test_session.py#L1)
- I2RT adapter의 real/simulation boundary와 resource lifecycle: [`test_i2rt_adapter.py`](../tests/test_i2rt_adapter.py#L1)
- Relay protocol, JPEG cache와 WebRTC track: [`test_mujoco_relay.py`](../tests/test_mujoco_relay.py#L1)
- Fake WebXR/browser 환경의 input payload, video layout과 episode 상태 알림: [`quest_input_self_check.cjs`](../tests/quest_input_self_check.cjs#L1), [`quest_client_self_check.cjs`](../tests/quest_client_self_check.cjs#L1)
- Headless MuJoCo model을 사용한 retargeting과 robot command path는 Python test suite에 포함한다.
- `pi05_yam_dit_lora` checkpoint step `14999`와 dataset top/wrist frame을 사용해 OpenPI server→YAM adapter→MuJoCo 경로의 open-loop 및 RTC 실행과 measured state 변화를 확인했다. RTC는 첫 chunk 경계를 넘었고, 동일한 server를 유지한 채 episode 셀을 두 번 반복 실행했다. 실제 장치 명령은 사용하지 않았다.

## 실제 hardware 검증 현황

- Quest 3 controller로 MuJoCo YAM과 실제 YAM이 움직이는 경로를 확인했다.
- Quest 3에서 MuJoCo video panel과 passthrough를 확인했다.
- 실제 L515 RGB capture, image 변환과 yam-abc episode 저장·decode를 확인했다.
- 실제 YAM에서 initial pose와 episode 반복 workflow를 확인했다.

## Test gap

- 실제 wrist camera의 Quest panel 표시와 장시간 network 안정성
- Top/wrist camera 동시 장시간 recording과 LeRobot v3 multi-episode round trip
- Table geometry를 포함한 실제 `ConfigurationValidator`와 SafetyGate
- 실제 YAM에서 π0.5 checkpoint inference, 실제 top/wrist camera와 action safety 검증
- 실제 YAM에서 RTC 지연·underrun 빈도와 real-time control 검증
- π0.5 fine-tuning의 재현성과 실제 작업 성공률
- Leader Arm, GR00T와 Isaac Sim은 미구현 상태이므로 검증하지 않았다.
