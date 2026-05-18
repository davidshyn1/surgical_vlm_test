#!/usr/bin/env bash
# Build Cholec80 0.1 fps frame evaluation set (fixed: 25 fps video -> 0.1 fps samples).
#
# One ffmpeg decode pass per video (select every STRIDE-th frame), then rename PNGs to
# native frame indices: 000000.png, 000250.png, … matching phase annotations.
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
  n_png=$(find "$outdir" -maxdepth 1 -type f -regex '.*/[0-9]{6}\.png$' 2>/dev/null | wc -l)
  if [[ "$n_png" -ge "$n_ann" && "$n_ann" -gt 0 ]]; then
    echo "[INFO] $stem: skip extract ($n_png/$n_ann frames present)" >&2
    continue
  fi

  echo "[INFO] $stem -> $outdir (0.1 fps, stride=$STRIDE, 1x ffmpeg decode)" >&2
  rm -f "$outdir"/.extract_*.png

  ffmpeg -hide_banner -loglevel error -y \
    -i "$mp4" \
    -vf "select='not(mod(n\\,${STRIDE}))'" \
    -vsync vfr \
    "$outdir/.extract_%06d.png"

  shopt -s nullglob
  tmp_files=("$outdir"/.extract_*.png)
  if [[ ${#tmp_files[@]} -eq 0 ]]; then
    echo "WARN: $stem: ffmpeg produced no frames" >&2
    continue
  fi
  mapfile -t tmp_sorted < <(printf '%s\n' "${tmp_files[@]}" | sort -V)

  i=0
  for f in "${tmp_sorted[@]}"; do
    frame_idx=$((i * STRIDE))
    dest="$outdir/$(printf '%06d.png' "$frame_idx")"
    if [[ -f "$dest" ]]; then
      rm -f "$f"
    else
      mv "$f" "$dest"
    fi
    i=$((i + 1))
  done
  rm -f "$outdir"/.extract_*.png

  echo "[INFO] $stem: wrote $i frames (expected ~$n_ann)" >&2
done

echo "Done. 0.1 fps frames + phase manifests under: $OUT_ROOT" >&2
echo "  Example: $OUT_ROOT/video41/000000.png + $OUT_ROOT/video41/video41-phase.txt" >&2
