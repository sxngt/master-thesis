# CLAUDE.md — 사족보행 로봇 험지보행 강화학습 연구

## 프로젝트 개요

석사 졸업논문 연구 저장소. 두 가지 핵심 축:

1. **강화학습 알고리즘 체계적 비교**: PPO, TRPO, A3C, SAC, TD3, DDPG를
   3개 로봇(Unitree A1, ANYmal C, MIT Mini Cheetah) × 12개 험지 시나리오에서
   비교 (3×6×12 매트릭스, 조합당 10 시드 반복).
2. **LLM 통합 피드백 강화학습**: 자연어 피드백을 LLM(Claude/GPT)으로
   수치 보상으로 변환하여 전통적 RL 보상과 하이브리드 결합.
   `R_total = α·R_traditional + β·R_LLM + γ·R_human_preference`

**연구 설계의 핵심: 동일한 알고리즘을 여러 시뮬레이터에서 검증한다.**
학습은 Isaac Lab(Isaac Sim, GPU 병렬), 검증은 PyBullet/Gazebo 교차 평가(sim-to-sim gap).
따라서 시뮬레이터는 `envs/backends/` 플러그인 + `configs/sim/` 설정 그룹으로
완전히 분리되어 있고, 알고리즘/보상/메트릭 코드는 시뮬레이터를 직접 import하지
않는다. 실험 축은 5차원: 알고리즘(6) × 로봇(3) × 지형(12) × 시드(10) × 시뮬레이터.

패키지 관리: **uv** (uv.lock 커밋). 실험 추적: **Weights & Biases**.
하드웨어 검증: Unitree A1.

## 디렉토리 구조

```
configs/               # YAML 설정 (계층적 합성: default + algorithm/robot/terrain/reward/experiment)
  algorithm/           # ppo, trpo, a3c, sac, td3, ddpg — 알고리즘별 하이퍼파라미터 + 탐색 공간
  robot/               # a1, anymal_c, mini_cheetah — 물리/센서/액션 스펙
  terrain/             # 12개 험지 시나리오 × 3단계 난이도
  reward/              # traditional(수치적), naive(코치 복구 실험용), hybrid_llm(LLM 통합)
  coach/               # 보상 코치: llm / random / hillclimb (Phase 3 파일럿)
  experiment/          # phase1_baseline ~ phase4_hardware 실험 정의
src/quadruped_rl/      # 메인 패키지 (uv sync로 editable 설치)
  envs/                # BaseEnv 계약, 지형 생성, 커리큘럼
    backends/          # 시뮬레이터 플러그인: isaaclab_backend(주 학습, 실기 검증됨),
                       #   pybullet_backend/gazebo(교차 검증), mock·mock_vec(CI 전용)
  algorithms/          # RL 알고리즘 구현 (base.py의 Algorithm 인터페이스 준수)
  rewards/             # 보상 컴포넌트 + 하이브리드 결합
  llm_feedback/        # 피드백 수집 → LLM 변환 → 보상 모델 파이프라인
  metrics/             # 운동/안정성/효율성/학습 메트릭 (순수 함수, 테스트 대상)
  analysis/            # ANOVA, Tukey HSD, Cohen's d, 플롯
  harness/             # 실험 하네스: Trainer, Evaluator, MatrixRunner, 시딩, 체크포인트
scripts/               # CLI 진입점 (train.py, evaluate.py, run_matrix.py, sweep.py, ...)
assets/robots/         # URDF/메시 (Git LFS, 저장소에 미포함 시 README 참조)
data/                  # 실험 산출물 (git 미추적; checkpoints/, results/, feedback/ 등)
docs/                  # 실험 프로토콜, 메트릭 정의, IRB/피드백 수집 프로토콜
tests/                 # pytest — 메트릭/보상/통계는 반드시 단위 테스트 유지
paper/                 # 논문 원고 (LaTeX)
```

## 자주 쓰는 명령

```bash
make install           # uv sync (.venv 생성, uv.lock 기준)
make test              # uv run pytest (시뮬레이터 불필요 — 순수 로직만)
make lint              # uv run ruff check + format check
make smoke             # mock 백엔드 1분 sanity 학습

# 개별 학습 (uv run 필수 — 시스템 python 사용 금지)
# Isaac Lab 학습은 env_isaaclab의 Python으로 (uv 환경 아님 — docs/setup.md):
PYTHONPATH=src ~/anaconda3/envs/env_isaaclab/bin/python scripts/train.py \
    --sim isaaclab --algorithm ppo --robot a1 --terrain stairs --seed 0
uv run python scripts/run_matrix.py --experiment phase2_matrix   # 전체 매트릭스 (재개 지원)
python3 scripts/run_jobs.py --jobs jobs.txt --parallel 2         # 4080: 2런 동시 실행이 최적 (docs/setup.md)
scripts/remote_sync.sh --launch data/results/coach_batch/jobs.txt  # 4x4090 서버 dongbeen (docs/setup.md)
uv run python scripts/cross_validate.py \
    --checkpoint data/results/<run_id>/checkpoints/best.pt \
    --sims isaaclab pybullet gazebo                              # 시뮬레이터 교차 검증
uv run python scripts/evaluate.py --checkpoint data/results/<run_id>/checkpoints/best.pt
uv run python scripts/sweep.py --algorithm sac --robot a1 --terrain stairs --trials 50
uv run python scripts/analyze.py --experiment phase2_matrix \
    --metrics success_rate cost_of_transport                     # ANOVA/Tukey + 플롯
```

의존성 추가는 `uv add <pkg>` (dev는 `uv add --dev`), 직접 pyproject 편집 후
`uv lock`도 가능. `pip install` 직접 호출 금지.

## 핵심 규약 (반드시 준수)

### 재현성 — 타협 불가
- 모든 무작위성은 `harness/seeding.py::set_global_seed()` 경유. 새 난수 소스
  (numpy/torch/env)를 추가하면 반드시 여기에 등록.
- 실험 실행 시 해석된(resolved) 전체 설정을 `data/results/<run_id>/config.yaml`로
  저장. 설정 저장 없는 학습 코드는 머지 금지.
- 결과 보고 시 항상 시드 개수·평균±표준편차·95% CI 명시. 단일 시드 결과로
  결론 내리지 않는다.

### 설정 시스템
- 모든 실험 파라미터는 YAML로. 코드에 하이퍼파라미터 하드코딩 금지.
- 합성 순서: `default.yaml` ← **sim** ← algorithm ← robot ← terrain ← reward
  ← coach ← experiment ← CLI 오버라이드 (뒤가 우선). 로더는 `harness/config.py`.
- `configs/coach/` (llm | random | hillclimb): 학습 중 보상 파라미터를 동적으로
  조정하는 보상 코치(`llm_feedback/coach.py`). `--coach`로 활성화. 조정 가능한
  파라미터·범위·가드레일·목적함수 J가 모두 YAML에 있고, 모든 개입은
  `<run_dir>/coach_log.jsonl`에 프롬프트/응답 원문까지 기록된다.

### 시뮬레이터 백엔드 (이 연구의 구조적 핵심)
- 모든 백엔드는 `envs/base_env.py::BaseEnv` 계약(observation_dim, action_dim,
  reset, step의 info dict 스키마)을 구현하고 `@register_env_backend("이름")`으로 등록.
- 알고리즘·보상·메트릭·하네스 코드에서 `import isaaclab`/`import pybullet` 금지 —
  시뮬레이터 의존은 `envs/backends/` 안에만 존재해야 한다.
- 벡터화 백엔드는 `VectorEnv` 계약(torch 텐서, auto-reset, 배치 info dict),
  단일 백엔드는 `BaseEnv` 계약. 벡터화 보상 수정 시 rewards/vectorized.py와
  traditional.py의 수치 동일성 테스트가 깨지지 않아야 한다.
- Isaac Lab 코드는 uv 환경에서 실행 불가 — env_isaaclab의 Python 3.11로 실행하고,
  isaaclab.envs 등 무거운 import는 AppLauncher 실행 이후에만(backend 내부 lazy import).
- 백엔드 네이티브 의존성이 없으면 조용히 스킵됨 (backends/__init__.py의 lazy import).
  mock 백엔드는 항상 사용 가능.
- 새 백엔드 추가: backends/에 파일 생성 + 등록 + `configs/sim/<이름>.yaml` 작성이 전부.
- 교차 검증 결과 해석 시 물리 파라미터 정합(system identification) 여부를 먼저 확인.

### 알고리즘 구현
- 모든 알고리즘은 `algorithms/base.py::Algorithm` ABC 구현
  (`act`, `update`, `save`, `load`, `hyperparameter_space`).
- 새 알고리즘 추가: 파일 생성 → `@register_algorithm("이름")` 데코레이터 →
  `configs/algorithm/<이름>.yaml` 작성 → 등록만으로 매트릭스 러너에 자동 포함.
- On-policy(PPO/TRPO/A3C)와 off-policy(SAC/TD3/DDPG)는 버퍼 인터페이스가 다름
  — `base.py`의 `RolloutBuffer` / `ReplayBuffer` 구분 유지.

### 메트릭
- 메트릭 정의는 `docs/metrics.md`가 단일 진실 공급원(SSOT). 구현 변경 시 문서 동기화.
- CoT(Cost of Transport) = E / (m·g·d) — 논문 전체에서 이 정의 고정.
- 메트릭 함수는 순수 함수로 유지 (환경/시뮬레이터 의존 금지) → tests/에서 검증.

### LLM 피드백
- LLM API 호출은 `llm_feedback/translator.py`에만 위치. API 키는 `.env`
  (절대 커밋 금지, `.env.example` 참조).
- LLM 응답은 구조화된 JSON 스키마(`llm_feedback/schemas.py`)로 검증 후 사용.
  파싱 실패 시 해당 피드백 폐기 + 로그 (보상 오염 방지).
- 인간 피드백 원본 데이터는 `data/feedback/`에 익명화 상태로만 저장 (IRB 규정).

### 코드 스타일
- Python 3.10+, 타입 힌트 필수, ruff 포맷. 실행은 항상 `uv run` 경유.
- 주석/독스트링은 영어 (논문·오픈소스 공개 대비), 커밋 메시지는 영어.
- 사용자와의 대화는 한국어.

## Claude 작업 지침

- **시뮬레이터 의존 코드 주의**: 이 머신에는 Isaac Lab이 `env_isaaclab`
  (conda, Python 3.11)에 설치돼 있음. 대규모 배치는 원격 서버 dongbeen
  (4×4090, Isaac Sim 4.5 / Isaac Lab 2.1.1 — 서버의 기존 환경 사용, 재설치 금지;
  docs/setup.md)에서 실행. isaaclab 의존 코드의 실기 검증은
  `PYTHONPATH=src ~/anaconda3/envs/env_isaaclab/bin/python`으로 실행 (docs/setup.md의
  스모크 명령 참조). 순수 로직은 uv 환경의 `tests/`로 검증.
- **실험 실행은 사용자 확인 후**: 학습 실행은 수 시간~수 일 소요. 임의로
  장시간 학습을 시작하지 말고, 스모크 테스트(`--smoke-test` 플래그, ~1분)로 검증.
- **data/ 는 삭제·수정 금지**: 실험 산출물은 재생성 비용이 큼.
- **통계 검정 우선**: 성능 비교 관련 코드/분석 요청 시 반드시 유의성 검정
  (ANOVA → Tukey HSD)과 효과 크기(Cohen's d)를 포함할 것.
- **논문 표기 일관성**: 알고리즘 명칭은 PPO, TRPO, A3C, SAC, TD3, DDPG로 통일.
  지형 명칭은 `configs/terrain/` 파일명을 정본으로.

## 연구 단계 (32개월)

| Phase | 기간 | 내용 | 상태 |
|-------|------|------|------|
| 1 | 6개월 | 환경 구축 + 기준 알고리즘 (A1, 평지/계단) | **진행 중** |
| 2 | 8개월 | 3×6×12 비교 매트릭스, 조합당 10시드 | 대기 |
| 3 | 6개월 | LLM 통합 시스템 개발·초기 검증 | 대기 |
| 4 | 4개월 | 통합 실험·성능 비교 분석 | 대기 |
| 5 | 6개월 | Unitree A1 하드웨어 검증(sim-to-real) + 논문 | 대기 |

## 학위논문 작성 규정 (인제대학교 대학원 — 절대 준수)

논문 파일: `paper/build_thesis.py`(생성기) → `paper/thesis.docx`. 본문 수정은
반드시 생성기를 통해서 하고 docx를 직접 편집하지 않는다.

- **구성 순서**: 표지 → 표제지 → 인준서 → 목차 → 국문초록 → 영문초록 →
  본문(서론, 연구목적, 연구재료 및 방법, 연구성적, 고찰, 결론) → 참고문헌 → 부록
- **규격**: 19×26cm(4×6배판). 본문 여백 상3/좌3.5/우2.5cm, 하단 중앙 면번호
- **본문 활자**: 휴먼명조/신명조 10~11pt, 줄간격 200%, 1행 국문 25~35자
- **면번호**: 국문초록 첫장~목차·도표목록 끝 = 로마 소문자(ⅰ,ⅱ…),
  본문(서론)부터 = 아라비아 숫자. 모두 하단 표기
- **항목구분번호**: 예시B 채택 — Ⅰ. → 1. → A. → 1) → a. → (1)
- **초록**: 국문·영문 각 2면 내외. 표·그림·인용 금지. 내용 = 연구목표·범위 /
  재료·방법 / 성적 요약 / 결론. 말미에 Key Words 6개 이내(국·영 각각)
- **연구성적**: 현상적 결과만 기술(해석·배경은 고찰에서). 표는 **영문**,
  제목 상단, 명사·형용사 첫글자 대문자, 약자는 하단 풀이. 그림은 글 **영문**,
  제목 하단. 같은 자료를 표·그림 중복 제시 금지. 표/그림 1개당 1면, 본문 해당
  부위에 삽입. 번호는 아라비아 일련번호
- **고찰**: ① 연구방법의 타당성·신뢰성 검토(재료 선택·관찰항목 채택의 문제점,
  비뚤림의 불가피성) ② 성적의 해석·추론 ③ 미해석 부분과 후속 과제 명시
- **결론**: 구체적 숫자·지엽 사항 배제, 연구목적 달성 여부, 후속 제언
- **참고문헌**: 본문 어깨번호(위첨자) 인용 순서대로.
  잡지: `저자(3인 초과 시 3인+et al). 제목. 잡지명, 년, 권：면수.`
  단행본: `저자. 도서명. 판. 발행지, 발행사, 년：면수.`
- **약어**: 최초 사용 시 괄호에 '이하 ○○이라 함' 명시. 숫자는 아라비아
- **학술용어**: 국문 원칙(공인 용어집 준거), 고유명사는 원자 표기
- 표지/표제지/인준서는 예시1~3 레이아웃(글자 크기·간격 포함)을 따르고,
  미확정 정보(학과·성명·지도교수·제출년월)는 ○○○ 플레이스홀더 유지
