# 코치 자가 진화 루프 (v5 → v6 → v7 …) 런북

로컬 오픈웨이트 LLM(비용 0)으로 보상 코치를 돌리고, 결과를 다시 LLM에게 보여
다음 코치 **버전**을 쓰게 하는 외부 루프. 설계와 수학은 `docs/coach_versions.md`
(v5)와 `src/quadruped_rl/llm_feedback/evolve.py`의 독스트링 참조.

## 1. GPU 배분 (dongbeen, 4×4090)

| GPU | 용도 | 비고 |
|-----|------|------|
| 0,1,2 | LLM 서버 **풀**: GPU당 독립 ollama 데몬 (`scripts/llm_server.sh pool-start`, 포트 11435/11436/11437) | 장당 qwen3.8:27b-mtp-q4_K_M(18 GB) + 32k 컨텍스트 |
| 3 | 시뮬레이션 (`evolve_coach.py --sim-gpus 3 --parallel 3`) | 4090 1장 = 동시 3런 (비동기 코치라 LLM 대기 없음) |

**왜 풀인가 (2026-09-07 실측).** ollama는 텐서 병렬이 없다: 한 모델을 3장에 나누면
레이어 파이프라인으로 **한 번에 한 장만 계산**한다(GPU 0~2 각 33 % 사용률, 30 tok/s
= 30 GB 가중치 ÷ 4090 대역폭 1 TB/s = 정확히 1장 속도). 게다가 qwen3.5 계열 hybrid
아키텍처는 ollama에서 동시 요청을 지원하지 않아(`n_slots = 1`,
`OLLAMA_NUM_PARALLEL` 무시) 동시 런의 코치 호출이 직렬 큐에 선다. 그 사이 코치가
동기 호출이었으므로 GPU 3의 학습이 멈춰 있었다(런당 호출 20회 × 2~8분). 해결 세 가지:
(1) GPU당 데몬 1개 = 데이터 병렬, 장당 full 대역폭; (2) `-mtp-` 태그 = multi-token
prediction 내장 투기 디코딩(무손실 1.5~2×); (3) `coach.async_llm` = 호출 중
학습 계속, 다음 평가에서 적용(`report_step`, `llm_latency_s`가 로그에 남는다).
**(3)은 풀 스택에서는 끄는 것이 낫다(2026-09-08 측정)**: 런마다 데몬이 있으면 호출이
41~87 s인데 학습 초반 1M-step 평가 간격은 ~20 s라 비동기 제안이 리포트보다 2~4
평가 늦게 적용됐고(첫 target 하향 2.1M → 3~5M, 낡은 리포트로 두 번째 하향 → 롤백),
게다가 적용 시점 기준으로 다음 리포트를 잡는 버그로 개입 주기가 2M → 4M으로
반감했다(rough-hard 탈출 10~11M → 15~16M, 33M 속도 1.3 → 1.1). GPU 한 장에 런 3개가
돌면 동기 대기는 다른 두 런이 흡수하므로 처리량 손실이 없다 → `async_llm: false`.
(주기 버그는 그리드 정렬로 고쳤고 비동기는 런 1개/GPU 상황용으로 남겨 둔다.)
q4_K_M은 q8_0 대비 약간의 정확도 손실이 있으므로 v5를 새 스택에서 **재측정**한 뒤
후보와 짝짓는다(`data/results/evolve2/`, 컨트롤은 심볼릭 링크로 재사용).

시스템 `ollama.service`(다른 사용자, crash-loop)는 건드리지 않는다. 우리 데몬은
`/mnt/sdb1/sxngt/ollama`(바이너리·가중치·PID·로그)에 완전히 분리돼 있다.

**Vulkan 함정 (2026-09-08).** ollama ≥ 0.33은 GPU를 CUDA와 **Vulkan** 두 경로로
탐색하는데 Vulkan은 `CUDA_VISIBLE_DEVICES`를 무시한다. GPU당 데몬이 CUDA 1장 +
Vulkan 4장을 보고 "더 많은 GPU가 보이는" Vulkan 백엔드를 골라 모델을 4장(시뮬 GPU 3
포함)에 펼쳐 놓았다(장당 13~19 GB, 파이프라인 속도로 회귀). `llm_server.sh`는
`OLLAMA_LLM_LIBRARY=cuda_v12`로 백엔드를 고정한다(`LLM_LIBRARY`로 변경). 검증:
데몬당 llama-server 1개가 자기 GPU에만 19.3 GB, GPU 3은 비어 있음. 풀 실측: 데몬당
75 tok/s 동시(3장 합계 ≈ 226 tok/s, 장당 ≈ 400 W) vs 구 3장 파이프라인 29.9 tok/s.

## 2. 모델 (2026-09-07 기준 선택)

| 모델 | 크기 | 왜 |
|------|------|----|
| **qwen3.8:27b-mtp-q4_K_M** (풀 기본) | 18 GB, MTP 헤드 내장 | 4090 1장에 들어가는 가장 강한 dense 모델. MTP로 디코드 1.5~2× |
| qwen3.8:27b-q8_0 / -mtp-q8_0 | 30 GB, dense, thinking, 256k ctx | 2026-08 공개. Artificial Analysis 지능 지수 52 (gpt-oss-120b 24). q8은 사실상 무손실이지만 3장 파이프라인(1장 속도) |
| qwen3.6:35b-a3b | 23 GB, MoE 3B active | 대안: 3~4배 빠름, 지능은 27b dense 아래 |
| gpt-oss:20b | 14 GB, MXFP4 | 폴백. `reasoning_effort` low/medium/high 지원 |

`qwen3.5:122b`(81 GB)와 `gpt-oss:120b`(65 GB, KV 여유 없음)는 72 GB에 맞지 않거나
동시 4슬롯이 불가능해 제외. 모델 교체는 `--override coach.llm.model=<tag>` 또는
`configs/coach/llm_local.yaml`.

## 3. 서버 준비 (GPU 사용 없음)

```bash
# 1) 저장소 동기화 (data/·.env 제외)
scripts/remote_sync.sh
# 2) 바이너리·가중치 설치 — 이미 실행됨: /mnt/sdb1/sxngt/ollama_setup.sh
#    (GPU 없는 데몬으로 pull만 하고 종료). 진행/결과: /mnt/sdb1/sxngt/ollama/setup.log
ssh -p 12888 sxngt@100.104.103.77 'tail -5 /mnt/sdb1/sxngt/ollama/setup.log; \
    OLLAMA_MODELS=/mnt/sdb1/sxngt/ollama/models /mnt/sdb1/sxngt/ollama/bin/ollama list'
```

## 4. 기동 (GPU가 비면)

```bash
ssh -p 12888 sxngt@100.104.103.77
cd /mnt/sdb1/sxngt/workspace/master-thesis
export LLM_HOME=/mnt/sdb1/sxngt/ollama LLM_GPUS=0,1,2
scripts/llm_server.sh pool-start qwen3.8:27b-mtp-q4_K_M   # GPU당 데몬 1개 (serve-<i>.{pid,log})
scripts/llm_server.sh pool-warm  qwen3.8:27b-mtp-q4_K_M   # 셋 동시에 VRAM 적재 + JSON 테스트, tok/s
scripts/llm_server.sh pool-status
# 단일 데몬(3장 파이프라인, q8)이 필요하면: start / warm / status / stop

tmux new -s evolve
export OMNI_KIT_ACCEPT_EULA=YES XDG_CACHE_HOME=/mnt/sdb1/sxngt/.cache
export ISAAC_PY=/mnt/sdb1/sxngt/isaac-sim-4.5.0/python.sh
mkdir -p data/results/evolve
$ISAAC_PY scripts/evolve_coach.py --root data/results/evolve2 --sim-gpus 3 --parallel 3 \
    --llm-model qwen3.8:27b-mtp-q4_K_M \
    --llm-url http://127.0.0.1:11435/v1 http://127.0.0.1:11436/v1 http://127.0.0.1:11437/v1 \
    2>&1 | tee -a data/results/evolve2/driver.log
```

`--llm-url`을 여러 개 주면 잡을 라운드로빈으로 배정한다(동시 런이 서로 다른
데몬을 쓴다); 메타 LLM은 첫 URL을 쓴다. 코치 preset(`llm_local.yaml`)의
`async_llm`은 false(동기)가 기본이다 — §1의 이유. 런 3개가 GPU 하나를 나눠 쓰는 한
동기 대기는 처리량을 깎지 않는다.

드라이버는 시작 전에(그리고 매 세대마다) GPU 3의 여유 메모리(`--min-free-mib`,
기본 16 GB)와 LLM 응답을 확인하며, 부족하면 5분 간격으로 기다린다. 다른 잡을
죽이는 코드는 없다. 미리 켜 두고 GPU가 비는 순간 자동으로 시작하게 해도 된다.

## 5. 한 세대의 흐름

1. **대조군** `--controls none`: 코치 없는 PPO를 같은 설정×시드로 1회 (보고서 기준선).
2. **현직(incumbent)** v5를 설정×시드(기본 3설정×3시드=9런, 40M 스텝)로 측정.
3. **메타 LLM**이 현직의 템플릿·오버라이드·결과표·코치 행동 진단·이력을 읽고
   `v6/`(system.md, user.md, coach.yaml, CHANGELOG.md)를 JSON으로 제안.
   검증(플레이스홀더, 필수 필드, 중괄호 포맷, `TUNABLE` 범위, 길이 ≤2.5×) 실패 시
   오류를 되돌려 최대 3회 재요청. 코드 변경 아이디어는 적용하지 않고
   `proposals.md`에 쌓는다(사람이 검토).
4. **후보** v6를 같은 설정×시드로 측정 → 쌍대 비교(같은 설정·시드끼리 ΔJ).
5. **채택 규칙(사전 등록)**: 쌍 ≥ 6, 합동 평균 ΔJ > 0, 어떤 설정도 평균 ΔJ < −0.05
   가 아님, 단측 대응 t 검정 p ≤ 0.2 (탐색 단계라 관대; Cohen's d·95% CI도 보고).
   채택되면 v6가 현직, 아니면 v5가 유지되고 실패 이유가 다음 제안의 입력이 된다.
6. `report.md`·`state.json` 갱신 후 다음 세대. 이름은 v6, v7, … 자동.

**사람이 쓴 후보(큐잉).** `configs/coach/versions/<이름>/`에 CHANGELOG.md의
`parent: <현직>` 줄과 함께 버전 디렉터리를 두면, 드라이버는 다음 세대에서 메타 LLM에
묻기 전에 그 디렉터리를 후보로 채택해 같은 규칙으로 측정·판정한다(state.json에 아직
없는 디렉터리만; 이름은 `v5p`처럼 숫자가 아닌 접미사를 써서 자동 이름 v6, v7과
충돌하지 않게 한다). 코드가 바뀐 장치(예: 플레이북)를 통제된 쌍대 비교로 검증하는
통로다. 이미 도는 드라이버는 다음 세대의 시작에서만 큐를 보므로, 즉시 채택하려면
드라이버를 재시작한다(§7; 완료된 배치는 건너뛴다). 큐 순서는 CHANGELOG.md가 쓰인
순서(mtime)이며, 후보를 큐에서 빼려면 `parent:` 줄에 괄호로 사유를 덧붙여 정확히 일치하지
않게 한다(v8·v9의 예).

**런 간 플레이북(cross-run playbook, 2026-09-08).** 코치의 증거 절(`_evidence`)은
현재 런만 계산했으므로, 런을 시작하는 코치는 같은 과제의 앞선 런에서 무엇이 통했는지
전혀 몰랐다. 구 스택 v5(n=9)는 컨트롤 대비 ΔJ +0.44 ± 0.41(95 % CI), p = 0.034,
d = 0.70으로 채택 수준이었지만 rough-medium-naive는 시드 0만 바닥을 벗어났다
(0.643 ± 0.902; 시드 1·2는 40M 스텝 내내 J 0.12). 그 과제에서 바닥을 벗어난 앞선
런 16/38은 **모두** 첫 리포트에 {target_ms ↓, energy.weight ↑, action_rate.weight ↑}를
함께 적용했고, 갇힌 시드들은 target만 내린 뒤 갇힌 채로 target을 다시 올리고
stability/LR로 흘러갔다(J가 바닥에서 평평하므로 전부 "kept"). 대응:
`llm_feedback/playbook.py` — 끝난 런들의 색인(바닥 탈출 스텝, 탈출 전 바닥에서 적용한
(파라미터, 방향) 이동과 첫 리포트 적용 수, 갇힌 런만 한 이동, 보행 중 런 간 원장,
최종 target_ms 구간별 최종 J)을 `<root>/playbook.json`으로 저장하고, 코치가
`coach.playbook.enabled`일 때 리포트 끝에 같은 과제(없으면 임의 과제)의 계산된 몇 줄을
붙인다(random/hillclimb 런은 경험으로 치지 않음, 현재 런 제외). 드라이버는 매 배치 전에
`<root>/*/runs` + `--playbook-roots`(구 배치)로 재구축하고 코치 런에
`coach.playbook.path`를 넘긴다; `playbook.enabled`는 메타 LLM의 `TUNABLE` 키이고,
진단(`diagnostics`)에는 바닥을 못 벗어난 런의 이동 순서가 붙는다. 첫 후보는
사람이 쓴 `v5p`(= v5 + 플레이북 + 가중치 문단 하나 + 정착 효과 원장). 수동 구축·열람:
`uv run python scripts/build_playbook.py --show rough-medium-naive <roots…>`.

같은 맥락의 두 번째 증거 보강이 **정착 효과 원장**(`coach.settled_reports`, 기본 0=끔,
`TUNABLE` 0~3)이다. 명령 상향의 즉시 효과(다음 리포트)는 성공률이 먼저 떨어지는 구조
때문에 체계적으로 음수라 코치가 상향을 거부한다(evolve2 v6 s1, docs/coach_versions.md §6).
켜면 유지된 커리큘럼 이동마다 k 리포트 뒤 J를 추세 제거해 두 번째 원장 줄("Settled
effect of curriculum moves")로 붙인다. v5p는 `settled_reports: 2`.
세 번째는 **원장 거부권**(`coach.ledger_veto_obs`, 기본 0, `TUNABLE` 0~6): 보상 가중치
이동의 방향이 런 내 원장에서 이미 유의하게 음수(평균 + 2 sd < 0, 관측 ≥ k)이면
떨어뜨리고 이력 줄에 보인다(v6 stairs s1이 음수 원장 5관측에도 페널티를 여섯 번 강화,
docs/coach_versions.md §6.1). v5p는 `ledger_veto_obs: 3`.
네 번째는 **커리큘럼 잠금**(`coach.curriculum_lock`, 기본 false, `TUNABLE`): 복귀 단계에서
커리큘럼 레버 하향 제안을 떨어뜨린다(메타 LLM v7의 프롬프트 규칙을 가드레일로, §6.3).
v5p·v7p(= v7 + 같은 네 장치, parent v7) 모두 켠다.

한 세대 비용(4090 1장, 4096 envs, 40M 스텝): 코치 없는 런 ≈ 15~20분, 동시 2런
→ 컨트롤 9런 ≈ 1.5시간 (2026-09-07 실측). 동기 코치 + 단일 q8 데몬에서는 코치 런이
≈ 90~100분(LLM 대기가 런 시간의 절반 이상)이었다 — v5 첫 측정. 풀 + MTP + 비동기
코치(§1) 실측(2026-09-08, evolve2 v5, 동시 3런): LLM 호출 41~87 s(구 2~8분), 리포트
2.06M → 적용 5.01M(짧은 에피소드 구간에서 1M ≈ 20 s), GPU 3 사용률 ≈ 74~99 %,
`coach/llm_inflight`로 대기 중 여부가 로그에 남는다. 실효 개입 간격은 interval +
호출 지연 ≈ 3M.

## 6. 산출물

```
data/results/evolve/
  state.json        # 현직, 세대, 버전별 상태/비교/요약, 로그 (재시작 시 이어서)
  report.md         # 대조군·버전별 표(n, mean±sd, 95% CI), 쌍대 비교, 판정
  meta_log.jsonl    # 메타 LLM 프롬프트·응답 원문·소요 시간
  proposals.md      # 코드 수정 아이디어 (자동 적용 안 함)
  driver.log
  control-none/ v5/ v6/ …
    jobs.txt  jobs.txt.status.json  jobs.txt.driver.log  runs/<run_id>/
configs/coach/versions/v6/   # 새 버전 (커밋 대상)
```

로컬로 가져오기: `rsync -avz -e 'ssh -p 12888' sxngt@100.104.103.77:/mnt/sdb1/sxngt/workspace/master-thesis/data/results/evolve/ data/results/evolve/`
(그리고 `configs/coach/versions/`).

### 6.1 실시간 관찰 — `scripts/watch_evolve.py`

W&B 없이 런 디렉터리(`metrics.jsonl`, `coach_log.jsonl`, `state.json`)만 읽는다.
표준 라이브러리만 쓰므로 서버의 시스템 python3(3.8)로도 돌고, 파일은 증분으로만
읽는다.

```bash
# 대시보드(15 s마다 갱신): 세대/현직/판정, GPU·LLM 풀 상태, 런별 step·속도·ETA·
# PPO loss/KL/β·최근 평가 성공률/속도/J·bestJ·코치 마지막 개입, 최근 LLM 응답, 드라이버 로그
python3 scripts/watch_evolve.py data/results/evolve3 --remote        # 로컬에서 ssh로 서버 실행
python3 scripts/watch_evolve.py data/results/evolve3                 # 서버에서 직접

# 이벤트 스트림(tail -f): 평가마다 한 줄, LLM 제안마다 진단·제안/적용 파라미터·예측 ΔJ·
# 신뢰도·지연·토큰, 판정(kept/rolled_back, 추세 보정 효과), 크래시, 드라이버 로그
python3 scripts/watch_evolve.py data/results/evolve3 --remote --follow
python3 scripts/watch_evolve.py data/results/evolve3 --remote --follow --full --prompt  # 응답 원문·프롬프트까지
python3 scripts/watch_evolve.py --run data/results/evolve3/v7/runs/<run_id> --remote    # 런 하나의 전체 이력

# 기타: --history(과거 이벤트 전부 재생) --last N --interval S --no-probe --once --no-color
```

행 표시: `klstop×N`(KL 조기 종료 횟수), `nonfinite×N`(스킵된 미니배치), β ≥ 8 노란색,
|KL| > 0.1 빨간색, `stalled?`(15분간 기록 없음), `CRASH`(잡 로그의 Traceback).
배치 디렉터리(`data/results/coach_v5`)나 런 디렉터리를 직접 줘도 된다.

## 7. 재시작·중단

- 드라이버를 죽이고 다시 실행하면 `state.json`과 `run_jobs`의 status 파일로 완료된
  런을 건너뛰고 이어간다(`--reconcile`). 드라이버(evolve_coach.py) → run_jobs.py →
  train.py 순의 트리이므로 셋을 함께 죽여야 한다. ssh 한 줄로 죽일 때 `pkill -f`
  패턴이 자기 셸의 명령줄과도 일치해 셸이 먼저 죽는 함정이 있다 — 대괄호 트릭을 쓴다:
  `pkill -f "[e]volve_coach.py"; pkill -f "[r]un_jobs.py --jobs .*evolve/"; pkill -f "results_root=[d]ata/results/evolve/"`.
  드라이버만 죽고 run_jobs가 살아 있으면 그 배치가 끝날 때까지 기다렸다가 드라이버를
  다시 띄운다(같은 jobs.txt를 두 run_jobs가 동시에 돌리면 안 된다).
- LLM 데몬만 재시작: `scripts/llm_server.sh pool-stop && scripts/llm_server.sh pool-start <model>`
  (우리 PID 파일의 프로세스만 종료). 단일 데몬은 `stop`/`start`.
- 중간에 LLM이 죽으면 코치는 `api_retries`(5회, 20 s 백오프) 후 그 개입만 건너뛴다.

## 8. 로컬 드라이런 (mock 시뮬레이터, 실제 로컬 LLM)

`mock_vec` 백엔드(`configs/sim/mock_vec.yaml`)와 그 2-파라미터 보상 공간에 맞춘
코치 `llm_local_mock`(결정층은 `llm_local`과 동일)을 쓴다. `mock` 백엔드는 코치
훅(`reward_params`)이 없어 쓸 수 없다. `--versions-dir`로 스크래치 버전 디렉터리를
따로 두면 `configs/coach/versions/`가 더럽혀지지 않는다(v5 복사본이 있어야 한다).
gpt-oss:20b는 `--meta-reasoning medium`이 적당하다(high는 4~5k 토큰 사고 후 답변;
서버의 qwen3.8은 이 옵션을 받지 않는다). 사고가 예산을 다 먹어 빈 답이 오면
드라이버가 다음 시도에서 effort를 한 단계 내린다.

```bash
LLM_HOME=~/ollama LLM_GPUS=0 scripts/llm_server.sh start gpt-oss:20b && scripts/llm_server.sh warm gpt-oss:20b
mkdir -p /tmp/evolve_versions && cp -r configs/coach/versions/v5 /tmp/evolve_versions/
uv run python scripts/evolve_coach.py --root data/results/evolve_dry --sim mock_vec --coach llm_local_mock \
    --python .venv/bin/python --sim-gpus 0 --min-free-mib 2000 --llm-model gpt-oss:20b --meta-reasoning medium \
    --steps 60000 --eval-interval 10000 --seeds 0 1 --settings flat-easy-naive --parallel 2 \
    --generations 1 --min-pairs 2 --versions-dir /tmp/evolve_versions \
    --extra-override coach.interval_steps=20000 --extra-override coach.warmup_steps=20000
```
