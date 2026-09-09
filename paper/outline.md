# 논문 구성 (2026-09-09 개정 — 실제 수행 범위 기준)

인제대학교 규정 순서(CLAUDE.md): 국문초록 → 영문초록 → Ⅰ 서론 → Ⅱ 연구목적 →
Ⅲ 연구재료 및 방법 → Ⅳ 연구성적 → Ⅴ 고찰 → Ⅵ 결론 → 참고문헌 → 부록.
생성기 `paper/build_thesis.py`가 정본; 이 파일은 내용 지도.

제목(제안, 지도교수 확정 전): 「사족보행 로봇의 험지 보행을 위한 강화학습 알고리즘
비교 및 대형 언어 모델 기반 보상 코칭에 관한 연구」

## 범위 (원 계획 대비)
| 축 | 원 계획 | 실제 |
|---|---|---|
| 알고리즘 | 6 | 6 (PPO, TRPO, A3C, SAC, TD3, DDPG) — 제1부 |
| 로봇 | 3 | A1 본실험 + ANYmal C 이식(표 5) |
| 지형 | 12 × 3 난이도 | 평지·계단·요철 × 난이도 스케일링(표 4) |
| 시드 | 10 | 3 (0, 1, 2) |
| LLM 통합 | 하이브리드 보상 + 인간 선호 | **학습 중 보상 코치**(LLM / random / hillclimb / none) — 제2부 |
| 시뮬레이터 교차 검증 | PyBullet/Gazebo | 후속 과제 |
| 하드웨어 | A1 실기 | 후속 과제 |

## Ⅰ 서론 (작성됨)
1. 의의 2. 관련 지견 — 보행 RL, 알고리즘 비교, 재현성, **LLM 보상 설계**(Eureka,
L2R, Text2Reward, Kwon)와 본 연구의 공백(학습 중 조정 + 판단층 + 무정보 대조군)
3. 유도 4. 주제(제1부 비교 → 제2부 코치) 5. 용어(코치, 개입, J, LLM-PPO, 커리큘럼 레버)

## Ⅱ 연구목적 (작성됨) — 실행 목표 1)~5) 제1부, 6)~8) 제2부

## Ⅲ 연구재료 및 방법 (작성됨)
1. 설계 개요 2. 재료(제1부) 3. 방법(제1부)
4. LLM 보상 코치 연구: A 설계 B 장비·예산(4090×4, 40M, 고정 256 env 평가, KL β 유계)
C 코치 구조(보고→제안→검증→판정) D 판단층(잡음 인지 롤백, 원장·거부권·정착 효과,
단계, 복귀 불변식·잠금, headroom) E 로컬 LLM·플레이북 F random/hillclimb
G 자가 진화(사전 등록 채택 기준) H 관찰 항목 I 통계(짝 t, Wilcoxon, CI, d, Fisher)

## Ⅳ 연구성적 (본문 작성됨 — 현상만 기술, 해석 금지)
번호는 build_thesis.py가 등장 순서로 매긴다(본문 참조는 `{Fig:이름}` / `{Table:이름}`).
그림 1(코치 구조도, `coach_architecture`)은 Ⅲ.4.C에 들어가므로 제1부 그림은 2~7.
1. 알고리즘 비교: 표 1~5, 그림 2~7 (paper/tables, paper/figures — 생성됨; 본문 문장은 아직 없음)
2. LLM 보상 코치 (내부 v1~v13을 논문 v1·v2·v3으로 축약 — Ⅲ.4.F; `paper/make_version_figures.py`):
   - A 버전별 목적함수: 표 6 `coach_versions_thesis`, 그림 8 `coach_version_lineage`, 그림 9 `coach_objective_by_version`
   - B v3 설정별·통합: 표 7 `table6_coach_per_setting_evolve4`, 표 8 `table7_coach_pooled_evolve4`
   - C 판단층 개입: 표 9 `coach_intervention_stats`, 그림 10 `coach_decision_activity`, 그림 11 `coach_curriculum_lever`
     (v3 9런에서 원장 거부권·잠금·동결 기각 0건 — 본문에 명시)
   - D 빈약 보상 복구: 그림 12 `coach_naive_recovery`, 표 10 `coach_naive_recipe` (21/17 vs 6/0, Fisher p=0.001)
   - E LLM 없는 코치: 표 11~13 `table8_ablation_per_setting / _contrasts / _escapes`

## Ⅴ 고찰 (골격만)
1. 방법의 타당성·신뢰성(재료·항목·시드 수·단일 시뮬레이터·LLM 비결정성)
2. 해석(알고리즘 역전, 코치 이득의 원천, naive 복구, 계단 병목, 진화 이력)
3. 미해석·후속(커리큘럼 방향 게이트, 목적함수, 시드 확대, 교차 검증, SAC, 실기)

## Ⅵ 결론 (골격만) / 초록 (제2부 반영하여 재작성 필요)

표·그림은 scripts/analyze.py + analysis/plots.py 산출물만 사용(재현 가능),
표는 영문·제목 상단, 그림은 영문·제목 하단, 1개당 1면.
