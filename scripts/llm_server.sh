#!/usr/bin/env bash
# Local open-weight LLM for the reward coach (zero API cost), served with a
# user-level ollama on the GPUs reserved for it. Runs on dongbeen (or any box):
#   scripts/llm_server.sh start [model]   # daemon on $LLM_GPUS (default 0,1,2) -> $LLM_PORT
#   scripts/llm_server.sh pull  [model]   # download weights (disk only, no GPU)
#   scripts/llm_server.sh warm  [model]   # load into VRAM + one JSON test call, prints tok/s
#   scripts/llm_server.sh status          # daemon, loaded models, GPU memory
#   scripts/llm_server.sh stop            # stops ONLY the daemon started by this script
# Pool mode (one independent daemon per GPU, data parallel): ollama splits a
# model over several GPUs by layers and computes on one GPU at a time (no
# tensor parallelism), and the qwen3.5+ hybrid architecture serves a single
# request at a time, so N concurrent coach calls on one 3-GPU daemon queue up
# at one GPU's speed. N daemons with a model that fits one 4090 (e.g.
# qwen3.8:27b-mtp-q4_K_M, 18 GB, multi-token prediction ~1.5-2x decode) give
# N x the throughput: ports $LLM_PORT, $LLM_PORT+1, ... one per GPU of $LLM_GPUS.
#   scripts/llm_server.sh pool-start [model]   # daemons serve-<i>.{pid,log}
#   scripts/llm_server.sh pool-warm  [model]   # warm every daemon (in parallel)
#   scripts/llm_server.sh pool-status | pool-stop
# The system ollama.service (another user's, crash-looping) is never touched:
# this instance has its own port, model dir and PID file. OpenAI-compatible
# endpoint: http://127.0.0.1:$LLM_PORT/v1  (configs/coach/llm_local.yaml).
set -euo pipefail

LLM_PORT=${LLM_PORT:-11435}
LLM_GPUS=${LLM_GPUS:-0,1,2}
LLM_MODEL=${2:-${LLM_MODEL:-qwen3.8:27b-q8_0}}
LLM_HOME=${LLM_HOME:-$HOME/ollama}           # binary (bin/ollama) if not on PATH; dongbeen: /mnt/sdb1/sxngt/ollama
LLM_MODELS=${LLM_MODELS:-$LLM_HOME/models}   # weights (dongbeen: qwen3.8:27b-q8_0, qwen3.6:35b-a3b, gpt-oss:20b)
LLM_LOG=${LLM_LOG:-$LLM_HOME/serve.log}
PIDFILE=$LLM_HOME/serve.pid
LLM_MAX_MODELS=2
if [ -n "${LLM_INSTANCE:-}" ]; then   # pool member i: its own GPU, port, pid and log
  LLM_MAX_MODELS=1
  IFS=, read -r -a _gpus <<< "$LLM_GPUS"
  LLM_GPUS=${_gpus[$LLM_INSTANCE]}
  LLM_PORT=$((LLM_PORT + LLM_INSTANCE))
  LLM_LOG=$LLM_HOME/serve-$LLM_INSTANCE.log
  PIDFILE=$LLM_HOME/serve-$LLM_INSTANCE.pid
fi
export OLLAMA_HOST=127.0.0.1:$LLM_PORT

pool() {  # run "$0 <cmd> [model]" once per GPU of $LLM_GPUS
  IFS=, read -r -a gpus <<< "$LLM_GPUS"
  for i in "${!gpus[@]}"; do LLM_INSTANCE=$i "$0" "$@"; done
}

bin() {
  if [ -x "$LLM_HOME/bin/ollama" ]; then echo "$LLM_HOME/bin/ollama"; else command -v ollama; fi
}

alive() { curl -sf -m 3 "http://$OLLAMA_HOST/api/version" >/dev/null; }

case "${1:-status}" in
  start)
    mkdir -p "$LLM_MODELS"
    if alive; then echo "[llm] already serving on $OLLAMA_HOST"; exit 0; fi
    # keep_alive -1: never unload between coach calls; num_parallel: one slot per
    # concurrent training run; flash attention + q8 KV cache: 4 x 32k contexts fit.
    # OLLAMA_LLM_LIBRARY pins the CUDA backend: ollama >= 0.33 also discovers the
    # GPUs through Vulkan, which ignores CUDA_VISIBLE_DEVICES, and then picks
    # Vulkan because it "sees more GPUs" -- spreading the model over every card
    # including the simulation GPU (observed 2026-09-08 with the per-GPU pool).
    CUDA_VISIBLE_DEVICES=$LLM_GPUS OLLAMA_LLM_LIBRARY=${LLM_LIBRARY:-cuda_v12} \
      OLLAMA_MODELS=$LLM_MODELS OLLAMA_KEEP_ALIVE=-1 \
      OLLAMA_NUM_PARALLEL=${LLM_PARALLEL:-4} OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
      OLLAMA_CONTEXT_LENGTH=${LLM_CONTEXT:-32768} OLLAMA_MAX_LOADED_MODELS=$LLM_MAX_MODELS \
      nohup "$(bin)" serve >> "$LLM_LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 30); do alive && break; sleep 1; done
    alive && echo "[llm] serving on $OLLAMA_HOST (GPUs $LLM_GPUS, pid $(cat "$PIDFILE"))" \
          || { echo "[llm] failed to start, see $LLM_LOG"; exit 1; }
    ;;
  pull)
    alive || "$0" start
    "$(bin)" pull "$LLM_MODEL"
    ;;
  warm)
    alive || { echo "[llm] not running; run: $0 start"; exit 1; }
    python3 - "$LLM_MODEL" "$OLLAMA_HOST" <<'EOF'
import json, sys, time, urllib.request
model, host = sys.argv[1], sys.argv[2]
body = {
    "model": model,
    "messages": [
        {"role": "system", "content": "Reply with one JSON object only."},
        {"role": "user", "content": 'Return {"ok": true, "sum": 17+25} with the sum computed.'},
    ],
    "response_format": {"type": "json_object"},
    "max_tokens": 2000,
}
req = urllib.request.Request(f"http://{host}/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=1800) as r:
    resp = json.load(r)
dt = time.time() - t0
text = resp["choices"][0]["message"]["content"]
usage = resp.get("usage", {})
print(f"[llm] {model}: {dt:.1f}s, {usage.get('completion_tokens', 0)} completion tokens "
      f"({usage.get('completion_tokens', 0) / max(dt, 1e-6):.0f} tok/s incl. load)")
print("[llm] reply:", text.strip()[:200])
json.loads(text)  # must be valid JSON
EOF
    ;;
  pool-start) pool start "$LLM_MODEL" ;;
  pool-stop) pool stop ;;
  pool-status) pool status ;;
  pool-warm)
    IFS=, read -r -a gpus <<< "$LLM_GPUS"
    for i in "${!gpus[@]}"; do LLM_INSTANCE=$i "$0" warm "$LLM_MODEL" & done
    wait
    ;;
  status)
    if alive; then
      echo "[llm] serving on $OLLAMA_HOST"; "$(bin)" ps 2>/dev/null || true
    else
      echo "[llm] not running"
    fi
    [ -n "${LLM_INSTANCE:-}" ] || \
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || true
    ;;
  stop)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      kill "$(cat "$PIDFILE")" && rm -f "$PIDFILE" && echo "[llm] stopped"
    else
      echo "[llm] no daemon started by this script"
    fi
    ;;
  *)
    sed -n 2,12p "$0"; exit 1 ;;
esac
