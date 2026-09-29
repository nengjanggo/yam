# YAM Control

## Prerequisites

- Ubuntu PC와 Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

## 설치

```bash
sudo apt update
sudo apt install git curl openssl build-essential python3-dev \
  linux-headers-$(uname -r) can-utils v4l-utils
```

```bash
git submodule update --init third_party/yam-abc-reproduce
git -C third_party/yam-abc-reproduce config \
  submodule.third_party/i2rt.url https://github.com/i2rt-robotics/i2rt.git
git -C third_party/yam-abc-reproduce submodule update --init --recursive
git submodule status --recursive
uv sync --locked --all-groups
.venv/bin/python --version
```

## π0.5 fine-tuning

OpenPI dependency는 vendored OpenPI directory에서 별도로 설치한다.

```bash
cd third_party/yam-abc-reproduce/third_party/policy/openpi
uv sync --locked --group dev
cd ../../../../..
```

[`finetune.ipynb`](./finetune.ipynb)을 repository root에서 열고 다음 순서로 해당 flag 하나만 `True`로 설정해 실행한다.

1. `RUN_TESTS`
2. `RUN_NORM_STATS`
3. `RUN_SMOKE_TRAINING`
4. `RUN_FULL_TRAINING`

dataset은 `lerobot-v3/nengjanggo/yam_pick_up_the_white_ethernet_cable_and_plug_it_into_the_black_ethernet_port`에 있어야 한다. 첫 실행에서는 `pi05_base` checkpoint가 내려받아질 수 있다. model/data contract는 [Blueprint](./docs/BLUEPRINT.md)를 참조한다.

## Quest 3 network 설정


```bash
ip -br address # PC LAN IP 확인
ROBOT_PC_IP=192.168.1.2 # 확인된 주소로 변경
install -d -m 700 certs
openssl req -x509 -newkey rsa:3072 -sha256 -nodes -days 365 \
  -keyout certs/key.pem -out certs/cert.pem \
  -subj "/CN=${ROBOT_PC_IP}" \
  -addext "subjectAltName=IP:${ROBOT_PC_IP},IP:127.0.0.1,DNS:localhost"
chmod 600 certs/key.pem
```

방화벽이 활성화되어 있다면 Quest IP에서 오는 relay TCP port와 WebRTC UDP port를 허용한다.

```bash
QUEST_IP=192.168.1.23 # 실제 주소로 변경
read -r RTC_UDP_START RTC_UDP_END < /proc/sys/net/ipv4/ip_local_port_range
sudo ufw allow proto tcp from "$QUEST_IP" to any port 8443
sudo ufw allow proto udp from "$QUEST_IP" to any port "${RTC_UDP_START}:${RTC_UDP_END}"
```