# 구현 로드맵 (코드 관점)

## 지금 구현되어 있는 것
- 실험 하네스 전체: 설정 합성, 시딩, Trainer/Evaluator/MatrixRunner, 체크포인트,
  W&B+JSONL 로깅, Optuna 스윕, 메타 가중치 최적화
- 알고리즘: PPO(완전 구현, 레퍼런스), SAC(완전 구현, off-policy 템플릿)
- 메트릭 4개 그룹 + 통계 분석(ANOVA/Tukey/Cohen's d) + 플롯
- 지형 12종 생성기 + 커리큘럼, mock 시뮬레이터 백엔드
- LLM 피드백 파이프라인(스키마/변환/수집/선호도 모델) + 보상 오염 가드
- 테스트 스위트 (시뮬레이터 불필요)

## Phase 1 완료 전 구현 필요 (우선순위순)
1. `envs/backends/isaac.py` — Isaac Gym 백엔드 (벡터화 학습의 핵심)
2. `algorithms/td3.py`, `ddpg.py` — sac.py 구조를 따라 구현
3. `algorithms/trpo.py` — CG natural gradient + line search
4. `algorithms/a3c.py` — 비동기 워커 그룹
5. `envs/backends/pybullet_backend.py` — 교차 검증용
6. 로봇 URDF 에셋 취득 + 센서 파이프라인
7. 벡터화 환경 지원 (현 Trainer 루프는 단일 env 기준 → num_envs 배치화)

## 이후
- Gazebo 백엔드 + ROS 브리지 (Phase 4-5)
- 시스템 식별로 시뮬레이터 물리 파라미터 정합 (configs/sim/*의 physics 값 갱신)
- LSTM 액터 경로 (networks.py의 recurrent 플래그 실제 연결)
