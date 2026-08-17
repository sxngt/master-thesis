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
- sim-to-sim gap: `scripts/cross_validate.py` — Isaac Gym 학습 정책을 PyBullet/Gazebo에서 평가, 메트릭별 상대 격차 보고
- sim-to-real gap: Phase 5, 실제 A1 대비 동일 메트릭

## 6. 통계 검정 (analysis/statistics.py)
one-way ANOVA(+η²) → Tukey HSD 사후검정 → Cohen's d → 95% CI.
비교 표는 전부 `compare_algorithms()` 경유 (방법론 통일).
