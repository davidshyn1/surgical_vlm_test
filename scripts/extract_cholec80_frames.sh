#!/usr/bin/env bash
# Extract Cholec80 frames with ffmpeg (no OpenCV / no pip numpy conflict).
# Output layout: OUT_ROOT/video41/000000.png  (frame index = annotation Frame column)
#
# Usage:
#   CHOLEC80_ROOT=../data/cholec80 OUT_ROOT=/path/to/frames STRIDE=25 \
#     bash scripts/extract_cholec80_frames.sh
#   # eval videos only (41-80):
#   VID_START=41 VID_END=80 bash scripts/extract_cholec80_frames.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHOLEC80_ROOT="${CHOLEC80_ROOT:-$ROOT/../data/cholec80}"
OUT_ROOT="${OUT_ROOT:-$CHOLEC80_ROOT/frames_stride25}"
STRIDE="${STRIDE:-25}"
VID_START="${VID_START:-41}"
VID_END="${VID_END:-80}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found on PATH" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

for ((vid = VID_START; vid <= VID_END; vid++)); do
  stem=$(printf "video%02d" "$vid")
  mp4="$CHOLEC80_ROOT/videos/${stem}.mp4"
  phase="$CHOLEC80_ROOT/phase_annotations/${stem}-phase.txt"
  outdir="$OUT_ROOT/$stem"
  if [[ ! -f "$mp4" ]]; then
    echo "SKIP missing $mp4" >&2
    continue
  fi
  mkdir -p "$outdir"
  echo "[INFO] $stem -> $outdir (stride=$STRIDE)" >&2
  awk -F'\t' -v stride="$STRIDE" '
    NR == 1 { next }
    { fi = $1 + 0; if (fi % stride == 0) print fi }
  ' "$phase" | while read -r frame_idx; do
    outpng="$outdir/$(printf '%06d.png' "$frame_idx")"
    [[ -f "$outpng" ]] && continue
    ffmpeg -hide_banner -loglevel error -y \
      -i "$mp4" \
      -vf "select=eq(n\\,${frame_idx})" \
      -vsync vfr -frames:v 1 \
      "$outpng" </dev/null
  done
done

echo "Done. Frames under: $OUT_ROOT" >&2
