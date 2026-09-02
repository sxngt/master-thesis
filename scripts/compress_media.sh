#!/usr/bin/env bash
# Compress capture MP4s for the repo (CRF 26) and rebuild palette GIFs.
set -eu
cd "$(dirname "$0")/.."
FF=$($HOME/anaconda3/envs/env_isaaclab/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
for f in docs/media/*.mp4; do
  tmp="${f%.mp4}.tmp.mp4"
  "$FF" -y -loglevel error -i "$f" -c:v libx264 -crf 26 -preset slow \
        -pix_fmt yuv420p -movflags +faststart "$tmp"
  mv "$tmp" "$f"
  "$FF" -y -loglevel error -t 6 -i "$f" \
        -vf "fps=10,scale=480:-2:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
        -loop 0 "${f%.mp4}.gif"
  echo "$(du -h "$f" | cut -f1) $f | $(du -h "${f%.mp4}.gif" | cut -f1) ${f%.mp4}.gif"
done
