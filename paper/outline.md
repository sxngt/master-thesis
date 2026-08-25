# 논문 구성 (초안)

1. 서론 — 험지보행의 필요성, 기존 모델 기반 제어의 한계, RL 접근, LLM 피드백의 가능성
2. 관련 연구 — 사족보행 RL, 알고리즘 비교 연구, RLHF/LLM-보상 연구
3. 방법론
   3.1 벤치마크 환경 (3 로봇 × 12 지형 × 다중 시뮬레이터)
   3.2 알고리즘 구현 및 튜닝 (Optuna)
   3.3 평가 메트릭 체계 (docs/metrics.md)
   3.4 LLM 통합 하이브리드 보상 (R_total = αR_trad + βR_LLM + γR_pref)
4. 실험 결과
   4.1 알고리즘 비교 (ANOVA/Tukey/Cohen's d)
   4.2 시뮬레이터 교차 검증 (sim-to-sim gap)
   4.3 LLM 통합 효과 (베이스라인 대비)
   4.4 하드웨어 검증 (Unitree A1)
5. 논의 — 알고리즘 선택 가이드라인, 한계
6. 결론

그림/표는 전부 scripts/analyze.py + analysis/plots.py 산출물 사용 (재현 가능).
