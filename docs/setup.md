# 환경 설정

## 기본 (uv)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
make install            # uv sync → .venv 생성 + uv.lock 기준 설치
make test               # 시뮬레이터 불필요
make smoke              # mock 백엔드 1분 학습 sanity check
```

## 시뮬레이터
- **Isaac Lab** (주 학습, 검증 완료): 이 머신에는 `~/Dev/IsaacLab` +
  conda env `env_isaaclab`(Python 3.11, Isaac Sim 5.1)로 설치되어 있음.
  Isaac Lab 학습은 반드시 그 환경의 Python으로 실행:
  ```bash
  PYTHONPATH=src ~/anaconda3/envs/env_isaaclab/bin/python scripts/train.py \
      --sim isaaclab --algorithm ppo --robot a1 --terrain stairs --seed 0
  ```
  빠른 실기 스모크(16 envs, ~3분):
  ```bash
  PYTHONPATH=src ~/anaconda3/envs/env_isaaclab/bin/python scripts/train.py \
      --sim isaaclab --algorithm ppo --robot a1 --terrain flat --seed 0 \
      --override sim.num_envs=16 --override sim.episode_length_s=5 \
      --override run.total_timesteps=2000 --override logging.wandb=false
  ```
  로봇 USD 에셋(A1, ANYmal-C)은 첫 실행 시 NVIDIA Nucleus에서 자동 다운로드.
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
- 기본 개발 환경: Python 3.13 (`.python-version`, uv 관리) — 테스트/분석/스크립트.
- Isaac Lab 실행은 `env_isaaclab`의 Python 3.11 사용 (`PYTHONPATH=src` 방식,
  잠금 충돌 없음). legacy Isaac Gym Preview(≤Py3.8)는 사용하지 않는다 —
  주 학습 백엔드는 Isaac Lab(`envs/backends/isaaclab_backend.py`)이다.

## 4080 단일 GPU 최고 효율 구성
실측(TD3 학습 중): GPU util ~70 %, VRAM 3.1/16 GB → **2런 동시 실행이 최적**.

1. CPU 거버너를 performance로 (Isaac 물리 호스트 측 병목 해소, 재부팅 시 초기화):
   ```bash
   sudo cpupower frequency-set -g performance
   ```
2. 배치는 순차 드라이버 대신 병렬 잡 스케줄러로 실행:
   ```bash
   # jobs.txt: 한 줄 = 학습 명령 1개 (matrix_runner의 export_jobs 출력과 호환)
   python3 scripts/run_jobs.py --jobs jobs.txt --parallel 2 --gpus 0
   ```
   - 재개 지원(<jobs>.status.json), 실패 잡 격리, 잡별 로그
   - 멀티 GPU가 생기면 `--parallel 4 --gpus 0,1,2,3`으로 그대로 확장(≈ ÷GPU 수)
3. 특히 off-policy(SAC/TD3) 구간이 배치 시간의 대부분을 차지하므로
   off-policy 잡끼리 병렬 배치하는 것이 효과가 가장 큼 (~1.4-1.6x).
