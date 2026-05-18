#!/usr/bin/env bash
# Build Cholec80 0.1 fps frame evaluation set (fixed: 25 fps video -> 0.1 fps samples).
#
# - Extracts PNGs at native frame indices 0, 250, 500, … (ffmpeg select=eq(n\,IDX))
# - Writes aligned phase annotations: OUT_ROOT/video41/video41-phase.txt
#   (subsampled from 25 fps phase_annotations; Frame column matches PNG names)
#
# Stride 250 = 25 fps / 0.1 fps (see cholec80_data.CHOLEC80_EVAL_FRAME_STRIDE).
#
# Usage:
#   CHOLEC80_ROOT=../data/cholec80 bash scripts/extract_cholec80_frames.sh
#   VID_START=41 VID_END=80 bash scripts/extract_cholec80_frames.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHOLEC80_ROOT="${CHOLEC80_ROOT:-$ROOT/../data/cholec80}"
OUT_ROOT="${OUT_ROOT:-$CHOLEC80_ROOT/frames_0p1fps}"
# Fixed 0.1 fps from 25 fps Cholec80 videos (stride 250; keep in sync with cholec80_data.py).
STRIDE=250
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
  phase_native="$CHOLEC80_ROOT/phase_annotations/${stem}-phase.txt"
  outdir="$OUT_ROOT/$stem"
  phase_out="$outdir/${stem}-phase.txt"

  if [[ ! -f "$mp4" ]]; then
    echo "SKIP missing $mp4" >&2
    continue
  fi
  if [[ ! -f "$phase_native" ]]; then
    echo "SKIP missing $phase_native" >&2
    continue
  fi

  mkdir -p "$outdir"

  awk -F'\t' -v stride="$STRIDE" '
    NR == 1 { print; next }
    { fi = $1 + 0; if (fi % stride == 0) print $0 }
  ' "$phase_native" > "$phase_out"

  n_ann=$(($(wc -l < "$phase_out") - 1))
  echo "[INFO] $stem -> $outdir (0.1 fps, stride=$STRIDE, annotations=$n_ann)" >&2

  awk -F'\t' -v stride="$STRIDE" '
    NR == 1 { next }
    { fi = $1 + 0; if (fi % stride == 0) print fi }
  ' "$phase_native" | while read -r frame_idx; do
    outpng="$outdir/$(printf '%06d.png' "$frame_idx")"
    [[ -f "$outpng" ]] && continue
    ffmpeg -hide_banner -loglevel error -y \
      -i "$mp4" \
      -vf "select=eq(n\\,${frame_idx})" \
      -vsync vfr -frames:v 1 \
      "$outpng" </dev/null
  done
done

echo "Done. 0.1 fps frames + phase manifests under: $OUT_ROOT" >&2
echo "  Example: $OUT_ROOT/video41/000000.png + $OUT_ROOT/video41/video41-phase.txt" >&2
