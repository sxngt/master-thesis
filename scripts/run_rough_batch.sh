#!/usr/bin/env bash
# Rough-terrain (irregular bumpy ground, medium +-5 cm) training batch:
# PPO/TRPO/SAC/TD3 x seeds 0-2. DDPG excluded (failure mode already
# characterized on flat; see docs/algorithm_analysis.md).
set -u
PY=$HOME/anaconda3/envs/env_isaaclab/bin/python
cd "$(dirname "$0")/.."
LOG_DIR=data/results/rough_batch_logs
mkdir -p "$LOG_DIR"

ONP="--override run.total_timesteps=50000000 --override run.eval_interval_steps=5000000 \
     --override run.checkpoint_interval_steps=10000000"
OFFP="--override sim.num_envs=128 --override run.total_timesteps=5000000 \
      --override run.eval_interval_steps=500000 \
      --override run.checkpoint_interval_steps=1000000 \
      --override algorithm.updates_per_step=16"

run() {
  local name=$1 seed=$2; shift 2
  local tag="${name}_rough_s${seed}"
  echo "[driver] START $tag $(date '+%H:%M:%S')" | tee -a "$LOG_DIR/driver.log"
  PYTHONPATH=src $PY scripts/train.py --sim isaaclab --algorithm "$name" \
      --robot a1 --terrain rough --seed "$seed" \
      --override sim.terrain_level=medium \
      --override logging.wandb=false "$@" > "$LOG_DIR/$tag.log" 2>&1
  echo "[driver] END $tag rc=$? $(date '+%H:%M:%S')" | tee -a "$LOG_DIR/driver.log"
  grep -E '"success_rate"|"mean_forward_velocity_ms"' "$LOG_DIR/$tag.log" \
      | tail -2 | tee -a "$LOG_DIR/driver.log"
}

# seed 0 first for all four (early full comparison), then remaining seeds
for seed in 0 1 2; do
  run ppo  $seed $ONP
  run trpo $seed $ONP
  run sac  $seed $OFFP
  run td3  $seed $OFFP
done
echo "[driver] ALL DONE $(date '+%H:%M:%S')" | tee -a "$LOG_DIR/driver.log"
