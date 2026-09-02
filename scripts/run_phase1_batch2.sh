#!/usr/bin/env bash
# Phase 1 batch 2: (a) DDPG parameter-space noise retry, (b) stairs terrain,
# (c) multi-seed statistics. Sequential; resumes are manual (rerun skips are
# NOT implemented here — check driver.log before rerunning).
set -u
PY=$HOME/anaconda3/envs/env_isaaclab/bin/python
cd "$(dirname "$0")/.."
LOG_DIR=data/results/phase1_batch2_logs
mkdir -p "$LOG_DIR"

ONP="--override run.total_timesteps=50000000 --override run.eval_interval_steps=5000000 \
     --override run.checkpoint_interval_steps=10000000"
OFFP="--override sim.num_envs=128 --override run.total_timesteps=5000000 \
      --override run.eval_interval_steps=500000 \
      --override run.checkpoint_interval_steps=1000000 \
      --override algorithm.updates_per_step=16"

run() {
  local name=$1 terrain=$2 seed=$3; shift 3
  local tag="${name}_${terrain}_s${seed}"
  echo "[driver] START $tag $(date '+%H:%M:%S')" | tee -a "$LOG_DIR/driver.log"
  PYTHONPATH=src $PY scripts/train.py --sim isaaclab --algorithm "$name" \
      --robot a1 --terrain "$terrain" --seed "$seed" \
      --override logging.wandb=false "$@" > "$LOG_DIR/$tag.log" 2>&1
  local rc=$?
  echo "[driver] END $tag rc=$rc $(date '+%H:%M:%S')" | tee -a "$LOG_DIR/driver.log"
  grep -E '"success_rate"|"mean_forward_velocity_ms"' "$LOG_DIR/$tag.log" \
      | tail -2 | tee -a "$LOG_DIR/driver.log"
}

# (a) DDPG with parameter-space noise — does it rescue the OU failure?
run ddpg flat 0 $OFFP --override algorithm.noise_type=parameter_space

# (b) stairs (easy), seed 0 — first rough-terrain results
for algo in ppo trpo; do run $algo stairs 0 $ONP; done
for algo in sac td3; do run $algo stairs 0 $OFFP; done

# (c) multi-seed statistics: flat seeds 1,2
for seed in 1 2; do
  for algo in ppo trpo; do run $algo flat $seed $ONP; done
  for algo in sac td3; do run $algo flat $seed $OFFP; done
done

# (b+c) stairs seeds 1,2
for seed in 1 2; do
  for algo in ppo trpo; do run $algo stairs $seed $ONP; done
  for algo in sac td3; do run $algo stairs $seed $OFFP; done
done

echo "[driver] ALL DONE $(date '+%H:%M:%S')" | tee -a "$LOG_DIR/driver.log"
