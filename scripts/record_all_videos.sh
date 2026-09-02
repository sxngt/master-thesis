#!/usr/bin/env bash
# Record gait videos for every (algorithm, terrain) best checkpoint.
# Input: data/results/video_jobs.txt  ("tag /path/to/ckpt.pt" per line)
set -u
PY=$HOME/anaconda3/envs/env_isaaclab/bin/python
cd "$(dirname "$0")/.."
LOG_DIR=data/results/video_logs
mkdir -p "$LOG_DIR" docs/media

while read -r tag ckpt; do
  [ -z "$tag" ] && continue
  if [ -f "docs/media/$tag.mp4" ]; then
    echo "[videos] SKIP $tag (exists)"; continue
  fi
  echo "[videos] START $tag $(date '+%H:%M:%S')"
  PYTHONPATH=src $PY scripts/record_video.py --checkpoint "$ckpt" \
      --out "docs/media/$tag.mp4" --seconds 10 --gif \
      > "$LOG_DIR/$tag.log" 2>&1
  echo "[videos] END $tag rc=$? $(date '+%H:%M:%S')"
done < data/results/video_jobs.txt
echo "[videos] ALL DONE"
ls -la docs/media/
