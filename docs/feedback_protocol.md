# 인간 피드백 수집 프로토콜 (Phase 3)

## 참가자
- 전문가 그룹: 로봇공학 연구자 10명 / 비전문가 그룹: 일반인 20명
- **IRB 승인 후 진행.** 동의서, 보상, 익명화 필수.

## 수집 모드 (`scripts/collect_feedback.py`)
1. **structured**: "로봇이 [상황]에서 [행동]할 때 [평가]" 템플릿
2. **free_form**: 동영상 시청 후 자유 서술
3. **realtime**: 시뮬레이션 관찰 중 즉각 코멘트

## 익명화 규칙
- 저장 필드는 `source_group`(expert/non_expert)뿐, 개인 식별자 저장 금지.
- 원본 데이터는 `data/feedback/`에만 존재 (git 미추적), 접근 통제.

## 파이프라인
feedback.jsonl → `llm_feedback/translator.py` (LLM 구조화 + 채점, 스키마 검증
실패 시 폐기) → `rewards/hybrid.py` (β항) / `llm_feedback/reward_model.py`
(선호도 쌍 학습, γ항).
