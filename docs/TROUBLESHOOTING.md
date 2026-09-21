# Troubleshooting

## Quest Browser에 `connection refused`가 표시됨

- 증상: `https://<PC_LAN_IP>:8443` 접속이 거부된다.
- 가능한 원인: Relay가 실행되지 않았거나 URL의 PC LAN IP가 현재 주소와 다르다.
- 확인 방법: PC에서 `ip -br address`, `ss -ltn | grep 8443`, `sudo ufw status verbose`를 확인한다.
- 해결 방법: [README의 relay 명령](../README.md#실행)을 다시 실행하고 현재 PC LAN IP로 접속한다. TCP 8443 firewall rule도 확인한다.

## Quest는 연결되지만 video가 표시되지 않음

- 증상: `Start Teleop`은 실행되지만 MuJoCo 또는 wrist panel이 보이지 않는다.
- 가능한 원인: WebRTC UDP가 차단되었거나 session이 video source를 아직 광고하지 않았다.
- 확인 방법: Relay terminal에서 Quest와 notebook의 WebSocket 연결을 확인하고 UDP firewall rule을 확인한다. MuJoCo는 `QUEST_DISPLAY_MODE`, 실제 robot은 `SHOW_REAL_WRIST_CAMERA_IN_QUEST`와 wrist camera role을 확인한다.
- 해결 방법: [README의 Quest network 설정](../README.md#quest-3-network-설정)을 적용한 뒤 relay, Quest page와 notebook session을 다시 시작한다.

## Controller를 움직여도 robot이 움직이지 않음

- 증상: Quest와 relay는 연결되지만 action이 hold된다.
- 가능한 원인: Clutch가 release되었거나 Quest frame이 stale 상태다.
- 확인 방법: Teleoperation cell 출력의 `last_hold_reason`, `frame_age_s`, `clutch_control_step_count`를 확인한다.
- 해결 방법: Controller tracking을 회복한 뒤 grip button을 완전히 놓았다가 다시 누른다. Passthrough와 controller가 정상 tracking되지 않으면 실제 robot을 실행하지 않는다.

## Camera device를 열 수 없거나 요청한 format이 거부됨

- 증상: Camera 연결 cell이 device open, resolution, FPS 또는 pixel format 오류를 발생시킨다.
- 가능한 원인: `/dev/videoN` 번호가 바뀌었거나 RGB가 아닌 RealSense node를 선택했다.
- 확인 방법: `v4l2-ctl --list-devices`와 `v4l2-ctl --device <path> --list-formats-ext`를 실행한다.
- 해결 방법: Notebook camera path를 `/dev/v4l/by-id/...-video-index0`의 실제 RGB node로 바꾸고 지원되는 format과 resolution을 설정한다.

## `show_real_wrist_camera requires a wrist camera`

- 증상: 실제 robot session 생성 시 wrist camera validation 오류가 발생한다.
- 가능한 원인: `SHOW_REAL_WRIST_CAMERA_IN_QUEST=True`이지만 `camera_devices`에 `role='wrist'`가 없다.
- 확인 방법: Notebook camera configuration의 role과 device path를 확인한다.
- 해결 방법: Wrist camera를 설정하거나 `SHOW_REAL_WRIST_CAMERA_IN_QUEST=False`로 변경한다.

## CAN interface 또는 motor communication 오류

- 증상: YAM 연결이 실패하거나 `loss communication` 이후 command가 실행되지 않는다.
- 가능한 원인: CAN interface가 down 상태이거나 bitrate가 다르거나 I2RT motor thread가 종료되었다.
- 확인 방법: `ip -details link show can0`에서 `UP`과 `bitrate 1000000`을 확인한다.
- 해결 방법: [README의 CAN 설정](../README.md#실제-yam과-camera-설정)을 다시 적용한다. Motor error 이후에는 Session 종료 cell을 실행한 뒤 hardware 상태를 확인하고 새 session으로 다시 연결한다.

## Module import 오류 또는 vendored package 누락

- 증상: `yam_abc_reproduce`, `i2rt` 또는 관련 module을 찾지 못한다.
- 가능한 원인: Git submodule이 초기화되지 않았거나 다른 Python kernel을 사용한다.
- 확인 방법: `git submodule status --recursive`와 notebook kernel path를 확인한다.
- 해결 방법: [README의 설치 명령](../README.md#설치)을 다시 실행하고 `.venv/bin/python` kernel을 선택한다.

## 수정한 code가 notebook에 반영되지 않음

- 증상: Python file을 수정했지만 notebook 동작이 이전과 같다.
- 가능한 원인: Kernel이 import cache의 기존 module을 사용한다.
- 확인 방법: Notebook의 선택된 kernel이 repository `.venv/bin/python`인지 확인한다.
- 해결 방법: Notebook kernel을 restart하고 첫 cell부터 다시 실행한다.
