# 평가 메트릭 정의 (Single Source of Truth)

구현: `src/quadruped_rl/metrics/`. **이 문서와 구현이 어긋나면 버그다** — 변경 시 반드시 동기화.
모든 결과 보고: 시드별 값 → 평균 ± 표준편차 + 95% CI (t-분포), 시드 수 명시.

## 1. 운동 성능 (metrics/locomotion.py)

| 메트릭 | 정의 | 단위 | 방향 |
|---|---|---|---|
| `mean_forward_velocity` | 시작→종료 수평 변위 / 경과 시간 | m/s | ↑ |
| `path_efficiency` | 최단 직선 거리 / 실제 이동 경로 길이, (0,1] | – | ↑ |
| `success_rate` | 목표 도달 에피소드 비율 | – | ↑ |
| `completion_time_s` | 성공 에피소드의 코스 완주 시간 | s | ↓ |

**에피소드 정의 (Isaac Lab)**: 스폰 지점에서 `sim.course_length_m`(5 m) 이상 이동하면
성공이며 그 즉시 에피소드가 종료된다(`sim.terminate_on_goal`, 가치 부트스트랩되는
truncation — 최적 정책은 불변). 넘어짐(몸통 접촉 또는 전도) 또는 `episode_length_s`(20 s)
경과로도 종료. 2026-09-05 이전 런은 목표 도달 후에도 계속 걸어 2×2×8 m 지형 가장자리
(스폰 중심에서 ~6 m)에서 떨어졌으므로 성공 에피소드가 전부 "낙상"으로 끝났다
(`fall_frequency` ≈ 9/min = 6.7 s마다 1회가 그 흔적).

**평가 프로토콜 (harness/evaluator.py)**: N개 병렬 env 중 균등 간격의 `run.eval_episodes`
(Isaac Lab 기본 256)개 env를 고정하고 각 env의 **첫 에피소드**만 집계한다. "먼저 끝난
K개 에피소드"를 모으는 방식(2026-09-05 이전)은 수천 env 중 가장 빨리 넘어진 것부터
수확하므로 실패에 편향되며, 성공률이 평가마다 0↔1로 요동했다(K=20). 동일 체크포인트를
두 방식으로 재평가하면 0.0 → 0.75~0.78(rough-hard), 0.0 → 0.51~0.53(stairs-medium).
리셋마다 초기 상태를 임의화한다(`sim.reset_noise`: 위치 ±0.5 m, 요 ±π, 초기 속도
±0.3 m/s·rad/s, 관절각 ×U(0.8, 1.2)). 이것이 없으면 같은 패치의 env가 모두 동일 궤적을
그려 유효 표본이 패치 수(4)로 줄고 성공률이 0.25 단위로 양자화된다(docs/findings.md 3.6).

## 2. 안정성 (metrics/stability.py)

| 메트릭 | 정의 | 단위 | 방향 |
|---|---|---|---|
| `fall_frequency` | 전복 횟수 / 분 | 1/min | ↓ |
| `attitude_stability` | √(std(roll)² + std(pitch)²) (IMU) | rad | ↓ |
| `contact_force_variance` | 발끝 접촉력 크기의 분산 | N² | ↓ |
| `recovery_time_s` | 외란 후 자세 편차 < 0.1 rad 를 50스텝 유지까지의 시간 | s | ↓ |

## 3. 효율성 (metrics/efficiency.py)

| 메트릭 | 정의 | 단위 | 방향 |
|---|---|---|---|
| `cost_of_transport` | **CoT = E / (m·g·d)**, g = 9.81 | – | ↓ |
| `torque_efficiency` | RMS 관절 토크 / 이동 거리 | N·m/m | ↓ |

에너지 E: 양의 기계적 일률 Σ|τ·q̇| 의 시간 적분.

## 4. 학습 효율 (metrics/learning.py) — 학습 곡선에서 계산

| 메트릭 | 정의 |
|---|---|
| `samples_to_threshold` | 목표 성능 도달까지 필요한 환경 스텝 수 (미도달 = censored 처리, max로 대체 금지) |
| `convergence_step` | 롤링 윈도우가 최종값 ±2% 이내에 들어가는 최초 스텝 |
| `training_stability` | 학습 곡선의 롤링 표준편차 평균 |
| `area_under_curve` | 정규화된 학습 곡선 AUC (속도+점근 성능 결합 점수) |

## 5. 강건성 프로토콜
- 관측 노이즈 주입: σ ∈ {0, 0.01, 0.05}
- 동역학 랜덤화 on/off 비교
- sim-to-sim gap: `scripts/cross_validate.py` — Isaac Lab 학습 정책을 PyBullet/Gazebo에서 평가, 메트릭별 상대 격차 보고
- sim-to-real gap: Phase 5, 실제 A1 대비 동일 메트릭

## 6. 통계 검정 (analysis/statistics.py)
one-way ANOVA(+η²) → Tukey HSD 사후검정 → Cohen's d → 95% CI.
비교 표는 전부 `compare_algorithms()` 경유 (방법론 통일).
