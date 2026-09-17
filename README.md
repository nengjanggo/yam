## 1. Dependency 설치

```bash
sudo apt update
sudo apt install git curl openssl build-essential python3-dev linux-headers-$(uname -r)
```

```bash
git submodule update --init third_party/yam-abc-reproduce
git -C third_party/yam-abc-reproduce config submodule.third_party/i2rt.url https://github.com/i2rt-robotics/i2rt.git
git -C third_party/yam-abc-reproduce submodule update --init --recursive
git submodule status --recursive
uv sync --locked --all-groups
.venv/bin/python --version
```

## 2. PC LAN IP 확인 및 TLS certificate 생성

PC의 실제 LAN IP를 확인하고 아래 `192.168.1.2`를 실제 주소로 바꾼다.

```bash
ip -br address
ROBOT_PC_IP=192.168.1.2
install -d -m 700 certs
openssl req -x509 -newkey rsa:3072 -sha256 -nodes -days 365 \
  -keyout certs/key.pem -out certs/cert.pem \
  -subj "/CN=${ROBOT_PC_IP}" \
  -addext "subjectAltName=IP:${ROBOT_PC_IP},IP:127.0.0.1,DNS:localhost"
chmod 600 certs/key.pem
openssl x509 -in certs/cert.pem -noout -dates -ext subjectAltName
```

## 3. 방화벽 설정

Quest 3의 LAN IP를 확인한 다음 UFW 상태를 확인한다. UFW가 활성화되어 있고 inbound가 제한되어 있다면, Quest 3에서 오는 relay TCP port 8443과 WebRTC media UDP port 범위를 허용한다.

```bash
sudo ufw status verbose
QUEST_IP=192.168.1.23  # Quest 3의 실제 LAN IP로 변경
read -r RTC_UDP_START RTC_UDP_END < /proc/sys/net/ipv4/ip_local_port_range
sudo ufw allow proto tcp from "$QUEST_IP" to any port 8443
sudo ufw allow proto udp from "$QUEST_IP" to any port "${RTC_UDP_START}:${RTC_UDP_END}"
sudo ufw status numbered
```

## 4. YAM CAN interface 확인

Follower CAN adapter가 실제로 보이는지 확인하고, I2RT가 요구하는 1 Mbit/s로 interface를 올린다([I2RT CAN 설정](./third_party/yam-abc-reproduce/third_party/i2rt/README.md#L35)). `can0`은 예시다. 실제 interface 이름을 [notebook의 `FOLLOWER_CAN_CHANNEL`](./yam.ipynb#L113)에 설정한다.

```bash
ip -br link
sudo ip link set can0 up type can bitrate 1000000
ip -details link show can0
```