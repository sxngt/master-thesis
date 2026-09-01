#!/usr/bin/env bash
# Phase 1 convergence runs: all Isaac-Lab-capable algorithms, A1, flat.
# Sequential (one Isaac Sim per process). Logs: data/results/phase1_flat_logs/
set -u
PY=$HOME/anaconda3/envs/env_isaaclab/bin/python
cd "$(dirname "$0")/.."
LOG_DIR=data/results/phase1_flat_logs
mkdir -p "$LOG_DIR"
COMMON="--sim isaaclab --robot a1 --terrain flat --seed 0 --override logging.wandb=false"

run() {
  local name=$1; shift
  echo "[driver] START $name $(date '+%H:%M:%S')" | tee -a "$LOG_DIR/driver.log"
  PYTHONPATH=src $PY scripts/train.py --algorithm "$name" $COMMON "$@" \
      > "$LOG_DIR/$name.log" 2>&1
  local rc=$?
  echo "[driver] END $name rc=$rc $(date '+%H:%M:%S')" | tee -a "$LOG_DIR/driver.log"
  grep -E '"success_rate"|"mean_forward_velocity_ms"|"run_dir"' "$LOG_DIR/$name.log" \
      | tail -3 | tee -a "$LOG_DIR/driver.log"
}

# On-policy: full 4096-env parallelism, 50M steps
ONP="--override run.total_timesteps=50000000 --override run.eval_interval_steps=2000000 \
     --override run.checkpoint_interval_steps=5000000"
run ppo  $ONP
run trpo $ONP

# Off-policy: 128 envs (update-to-sample ratio), 5M steps, 16 updates/sim-step
OFFP="--override sim.num_envs=128 --override run.total_timesteps=5000000 \
      --override run.eval_interval_steps=250000 \
      --override run.checkpoint_interval_steps=1000000 \
      --override algorithm.updates_per_step=16"
run sac  $OFFP
run td3  $OFFP
run ddpg $OFFP

echo "[driver] ALL DONE $(date '+%H:%M:%S')" | tee -a "$LOG_DIR/driver.log"
