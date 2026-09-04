#!/usr/bin/env bash
# Sync this repo to the 4x4090 server (Tailscale host alias `dongbeen`) and
# optionally launch a job file there with the parallel scheduler.
#   scripts/remote_sync.sh                                   # sync only
#   scripts/remote_sync.sh --launch data/results/coach_batch/jobs.txt
#   GPUS=3 PARALLEL=2 scripts/remote_sync.sh --launch <jobs>   # share the box
# Re-launching the same jobs file resumes it: finished jobs are skipped and
# runs that completed after a killed driver are recovered (--reconcile).
# Requires the local key to be authorised on the server once
# (docs/setup.md, "원격 서버" section):
#   ssh-copy-id -i ~/.ssh/id_ed25519 -p 12888 sxngt@100.104.103.77
set -euo pipefail
HOST=${REMOTE_HOST:-sxngt@100.104.103.77}
SSH="ssh -p ${REMOTE_PORT:-12888} -i ${REMOTE_KEY:-$HOME/.ssh/id_ed25519} -o BatchMode=yes"
REMOTE_ROOT=${REMOTE_ROOT:-/mnt/sdb1/sxngt/workspace/master-thesis}
# Isaac Sim 4.5 bundled python (Isaac Lab 2.1.1 installed into it)
REMOTE_PY=${REMOTE_PY:-/mnt/sdb1/sxngt/isaac-sim-4.5.0/python.sh}
cd "$(dirname "$0")/.."

rsync -az --delete -e "$SSH" \
  --exclude .git --exclude .venv --exclude data --exclude docs/media \
  --exclude paper --exclude '__pycache__' --exclude '*.egg-info' --exclude .env \
  ./ "$HOST:$REMOTE_ROOT/"
# API keys: copied separately so --delete never touches it; never committed
if [ -f .env ]; then rsync -az -e "$SSH" --chmod=600 .env "$HOST:$REMOTE_ROOT/.env"; fi
echo "[sync] done -> $HOST:$REMOTE_ROOT"

if [ "${1:-}" = "--launch" ]; then
  JOBS=${2:?jobs file}
  PARALLEL=${PARALLEL:-8}
  GPUS=${GPUS:-0,1,2,3}
  $SSH "$HOST" "mkdir -p $REMOTE_ROOT/$(dirname "$JOBS")"
  rsync -az -e "$SSH" "$JOBS" "$HOST:$REMOTE_ROOT/$JOBS"
  $SSH "$HOST" "cd $REMOTE_ROOT && \
    export ISAAC_PY=$REMOTE_PY OMNI_KIT_ACCEPT_EULA=YES XDG_CACHE_HOME=/mnt/sdb1/sxngt/.cache && \
    nohup python3 scripts/run_jobs.py --jobs $JOBS --parallel $PARALLEL --gpus $GPUS \
      --success-pattern '\"final\"' --reconcile >> $JOBS.driver.log 2>&1 < /dev/null & \
    echo '[launch] driver pid' \$!"
fi
