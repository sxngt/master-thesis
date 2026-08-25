# 환경 설정

## 기본 (uv)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
make install            # uv sync → .venv 생성 + uv.lock 기준 설치
make test               # 시뮬레이터 불필요
make smoke              # mock 백엔드 1분 학습 sanity check
```

## 시뮬레이터
- **Isaac Gym** (주 학습): NVIDIA 사이트에서 Preview 4 수동 다운로드 후
  `uv pip install -e <isaacgym>/python`. CUDA GPU 필수. torch보다 먼저 import 됨에 주의
  (backends/isaac.py가 처리).
- **PyBullet** (교차 검증): `make install-sim`
- **Gazebo** (교차 검증 + 하드웨어 스테이징): ROS 2 Humble + gazebo_ros 필요.

## 로봇 에셋
`assets/robots/<name>/README.md` 참조 — URDF는 각 공식 저장소에서 취득 (Git LFS).

## 시크릿
`cp .env.example .env` 후 API 키 입력. `.env`는 절대 커밋 금지.

## Docker (클러스터/재현)
```bash
docker compose -f docker/docker-compose.yml build
```

## Python 버전 주의
- 기본 개발 환경: Python 3.13 (`.python-version`, uv 관리).
- **Isaac Gym Preview 4는 구형 Python(≤3.8)만 공식 지원** — Isaac Gym 학습 머신에서는
  별도 uv 환경(`uv venv --python 3.8`)을 쓰거나, 후속 프레임워크인 **Isaac Lab**
  (Isaac Sim 기반, 최신 Python 지원) 전환을 검토할 것. 백엔드 플러그인 구조상
  `envs/backends/isaac.py`만 교체하면 나머지 코드는 그대로다.
