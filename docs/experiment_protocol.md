# 실험 프로토콜

## 실험 축 (5차원)
**알고리즘(6) × 로봇(3) × 지형(12) × 시드(10) × 시뮬레이터**
- 학습은 Isaac Lab 고정, 평가는 3개 시뮬레이터 교차 검증.
- 정의는 `configs/experiment/*.yaml`이 정본. 코드에 실험 목록 하드코딩 금지.

## 실행 규칙
1. 모든 학습은 `scripts/train.py` 또는 `scripts/run_matrix.py`로만 실행
   (Trainer가 시딩·설정 저장·평가·체크포인트를 일괄 보장).
2. 매트릭스 실행은 `<experiment>_status.json`으로 중단/재개 지원 —
   실패 셀은 traceback과 함께 기록되고 재실행 시 이어서 진행.
3. 하이퍼파라미터 튜닝(`scripts/sweep.py`)은 본 실험 **이전에** 셀당 예산을 정해
   수행하고, 최종 비교는 튜닝된 고정 설정으로만.
4. 평가 시드(≥1000)는 학습 시드(0–9)와 분리.
5. 스모크 테스트 결과는 절대 보고에 사용하지 않는다.

## 산출물 계약 (run당)
```
data/results/<run_id>/
  config.yaml          # 해석 완료된 전체 설정 (재현의 기준점)
  metrics.jsonl        # train/eval/final 메트릭 시계열
  checkpoints/         # step_*.pt + best.pt (+ .json 메타)
  cross_validation.json  # (선택) 시뮬레이터 교차 평가
```

## 데이터 수집 (thesis 1.2.3)
- 고유수용감각 1000 Hz, 제어 명령 100 Hz → HDF5 (`utils/hdf5_io.py`)
- 대용량 산출물은 git 미추적, W&B + 외부 스토리지 백업
