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

## 원격 서버 dongbeen (4×RTX 4090, Tailscale)
- 접속: `ssh -p 12888 sxngt@100.104.103.77` (Tailscale IP). 공유 호스트,
  sudo 필요 시 비밀번호. Ubuntu 20.04 / driver 550 / 188 GB RAM / `/mnt/sdb1` 644 GB.
- 서버의 기존 환경을 그대로 사용 (재설치 없음):
  `/mnt/sdb1/sxngt/isaac-sim-4.5.0/python.sh` (Python 3.10, Isaac Lab 2.1.1 설치됨).
  `~/.bashrc`가 `env.sh`를 소싱해 `OMNI_KIT_ACCEPT_EULA`, `XDG_CACHE_HOME`을 설정한다.
  Isaac Lab 2.1.1과 로컬 5.1은 백엔드가 쓰는 API가 동일 (RenderCfg 제외, 서버 미사용).
- 1회 설정 (로컬): `ssh-copy-id -i ~/.ssh/id_ed25519 -p 12888 sxngt@100.104.103.77`
  (비대화식 rsync/ssh용. 키는 `~/.ssh/id_ed25519`).
- 동기화 + 실행:
  ```bash
  scripts/remote_sync.sh                                        # rsync (data/, .git 제외; .env는 별도 600)
  scripts/remote_sync.sh --launch data/results/coach_batch/jobs.txt   # 4 GPU × 2런 병렬
  GPUS=3 PARALLEL=2 scripts/remote_sync.sh --launch <jobs>            # 다른 배치와 공유 시
  ```
  잡 파일의 인터프리터는 `$ISAAC_PY`로 결정되므로 로컬(env_isaaclab)과 서버
  (Isaac 4.5 python.sh)에서 같은 jobs.txt를 쓴다. 상태: `<jobs>.status.json`,
  드라이버 로그: `<jobs>.driver.log`.
- 재개: 같은 잡 파일로 `--launch`를 다시 실행하면 done은 건너뛰고 failed만
  재실행한다. 드라이버만 죽이면(`pkill -f "run_jobs.py --jobs <jobs>"`) 자식 학습은
  계속 돌아 로그를 끝까지 쓰므로, 재개 시 `--reconcile`이 로그의 성공 마커로
  이들을 done 처리한다. 자식이 아직 도는 중에 재개하면 `--skip-running`으로
  같은 명령이 이미 실행 중인 잡을 건너뛴다(중복 실행 방지; 나중에 `--reconcile`).
  **서버에 다른 배치가 돌고 있으면 GPU를 나눠 쓸 것** —
  드라이버는 자기 잡만 세어 GPU를 고르므로 남의 잡을 모르고, 4090 24 GB에
  10 GB 잡 2개 + 코치 잡 1개면 OOM.
- Isaac python에 추가로 필요한 패키지: `openai==1.85.0` (설치 완료). 최신 openai는
  aiohttp 전송 계층을 번들하는데 Isaac Sim 4.5의 prebundled aiohttp가 구버전이라
  import 시 `AttributeError: SocketTimeoutError` — 1.86 미만으로 고정할 것.
  pydantic/dotenv/pyyaml은 Isaac에 이미 포함.
- 서버 시스템 `python3`는 3.8 — `scripts/run_jobs.py`, `batch_status.py`처럼 시스템
  python으로 도는 스크립트는 `from __future__ import annotations` 유지.
- 실측 처리량 (PPO 4096 envs): GPU당 1잡 108k steps/s, 2잡 동시 46~66k/잡
  (합계 ~120k). VRAM은 잡당 ~4.5 GB지만 GPU 연산·CPU(20스레드, 70 %)가 병목이라
  GPU당 2잡(`--parallel 8`)이 상한. 40M 스텝 잡 ≈ 10분.
