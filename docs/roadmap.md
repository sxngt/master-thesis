# 구현 로드맵 (코드 관점)

## 지금 구현되어 있는 것
- 실험 하네스 전체: 설정 합성, 시딩, Trainer/Evaluator/MatrixRunner, 체크포인트,
  W&B+JSONL 로깅, Optuna 스윕, 메타 가중치 최적화
- **Isaac Lab 백엔드 (실기 검증 완료, 2026-09-01)**: DirectRLEnv 기반
  `_QuadrupedEnv` + VectorEnv 어댑터, 지형 매핑(9/12 시나리오), A1/ANYmal-C
  에셋, config 기반 벡터화 보상. RTX 4080에서 16-env 스모크 학습 통과
- 벡터화 학습 경로: VectorEnv 계약, VecRolloutBuffer(GAE 등가성 테스트),
  PPO 벡터 수집, Evaluator 벡터 롤아웃, mock_vec CI 백엔드
- 벡터화 보상(rewards/vectorized.py): traditional.py와 수치 동일성 테스트로 고정
- 알고리즘: PPO(완전 구현, 레퍼런스), SAC(완전 구현, off-policy 템플릿)
- 메트릭 4개 그룹 + 통계 분석(ANOVA/Tukey/Cohen's d) + 플롯
- 지형 12종 생성기 + 커리큘럼, mock 시뮬레이터 백엔드
- LLM 피드백 파이프라인(스키마/변환/수집/선호도 모델) + 보상 오염 가드
- **LLM 보상 코치 (Phase 3 파일럿, 2026-09-04)**: `llm_feedback/coach.py` —
  평가 주기마다 관측 리포트(KPI + 보행 기술자 + 보상 기여도) → 스케줄러
  (llm | random | hillclimb) → 가드레일(부호 고정, 선형 ±30 % / 로그 ×÷3,
  최대 3개) → `env.set_reward_params` + 정책 스냅샷 → 목적함수 J 하락 시 롤백.
  `configs/coach/*.yaml`, `--coach`, `scripts/make_coach_jobs.py`(36 잡),
  원격 실행 `scripts/remote_sync.sh`(docs/setup.md). Isaac Lab 실기 스모크 통과
- 테스트 스위트 (시뮬레이터 불필요)

## Phase 1 완료 전 구현 필요 (우선순위순)
1. ~~Isaac Lab 백엔드~~ ✅ 완료 (잔여: moving_platform/seesaw 동적 지형,
   rough_slope 표면 노이즈, mini_cheetah URDF→USD 변환)
2. ~~`algorithms/td3.py`, `ddpg.py`~~ ✅ 완료 (2026-09-01) — TD3: target policy
   smoothing + delayed update(2:1) + clipped double Q; DDPG: OU 노이즈 및
   adaptive parameter-space 노이즈 두 변형. 스모크 학습·메커니즘 단위 테스트 포함
3. ~~`algorithms/trpo.py`~~ ✅ 완료 (2026-09-01) — CG 기반 natural gradient
   (Fisher-vector product via KL 이중 미분), KL 제약 backtracking line search,
   별도 value 회귀. 거부된 스텝은 파라미터 정확 복원
4. ~~`algorithms/a3c.py`~~ ✅ 완료 (2026-09-01) — 워커 스레드별 독립 env+로컬
   네트워크, n-step 리턴, gradient accumulation, 공유 글로벌 네트워크에 비동기
   적용. VectorEnv 백엔드와는 비호환(설계상 CPU 워커 전제, 명시적 거부).
   LSTM 액터 경로는 미구현
5. ~~SAC/TD3/DDPG + TRPO의 VectorEnv 수집 경로~~ ✅ 완료 (2026-09-01) —
   ReplayBuffer 배치 삽입(랩어라운드), 배치 act(단일=numpy/배치=tensor),
   벡터 OU 노이즈(done 행 리셋), TRPO VecRolloutBuffer 경로.
   Isaac Lab 실기 스모크(SAC/TRPO, 16 envs) 통과. A3C 제외 5개 알고리즘이
   Isaac Lab에서 학습 가능. off-policy는 num_envs 64-256 권장
   (configs/sim/isaaclab.yaml 주석)
6. **실제 수렴 검증 런** — PPO × A1 × flat/stairs, 4096 envs로 보행 학습
   확인 + 보상 가중치 튜닝 (Phase 1 실질 목표)
7. `envs/backends/pybullet_backend.py` — 교차 검증용
8. 커리큘럼-지형 연동 (terrain_level 승급 시 지형 재생성)
9. A3C LSTM 액터 경로 (networks.py recurrent 플래그 연결)

## 이후
- Gazebo 백엔드 + ROS 브리지 (Phase 4-5)
- 시스템 식별로 시뮬레이터 물리 파라미터 정합 (configs/sim/*의 physics 값 갱신)
- LSTM 액터 경로 (networks.py의 recurrent 플래그 실제 연결)
