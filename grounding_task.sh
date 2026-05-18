#!/usr/bin/env bash
# SurgVLM-style eval runner (triplet recognition).
# Backend .venv: bash setup_backend.sh [prismatic|cosmos|groot|all]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BACKEND_ROOT_DEFAULT="$(cd "$ROOT/../backend" && pwd)"
VLA_ROOT="${VLA_ROOT_OVERRIDE:-$BACKEND_ROOT_DEFAULT}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$ROOT/../.cache/huggingface}"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

: "${GVL_PRISMATIC_DEFAULT:=$VLA_ROOT/prismatic-vlms/.venv/bin/python}"
: "${GVL_COSMOS_DEFAULT:=$VLA_ROOT/cosmos-reason2/.venv/bin/python}"
: "${GVL_GROOT_DEFAULT:=$VLA_ROOT/GR00T-H/.venv/bin/python}"
: "${CHOLECT50_CHALLENGE_VAL_ROOT:=$ROOT/../eval/cholect50-challenge-val}"
: "${CHOLECT50_VIDEOS_ROOT:=}"
: "${CHOLEC80_ROOT:=$ROOT/../data/Cholec80}"
: "${CHOLEC80_FRAMES_ROOT:=$ROOT/../eval/cholec80/frames_0p1fps}"
: "${ENDOVIS17_VQLA_ROOT:=$ROOT/../eval/EndoVis-17-VQLA}"

guess_conda_python() {
  local env_name="$1"
  local cand
  for cand in \
    "/NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/.conda/envs/${env_name}/bin/python" \
    "/home/skshyn/miniconda3/envs/${env_name}/bin/python" \
    "/home/skshyn/anaconda3/envs/${env_name}/bin/python"
  do
    if [[ -x "$cand" ]]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

backend_repo_dir() {
  case "$1" in
    prismatic) echo "$VLA_ROOT/prismatic-vlms" ;;
    cosmos) echo "$VLA_ROOT/cosmos-reason2" ;;
    groot) echo "$VLA_ROOT/GR00T-H" ;;
    *) echo "" ;;
  esac
}

append_unique_pythonpath() {
  local add_path="$1"
  [[ -z "$add_path" || ! -d "$add_path" ]] && return 0
  if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$add_path"
    return 0
  fi
  case ":$PYTHONPATH:" in
    *":$add_path:"*) ;;
    *) export PYTHONPATH="$add_path:$PYTHONPATH" ;;
  esac
}

ensure_repo_uv_venv_if_needed() {
  local backend="$1"
  case "${GROUNDING_TASK_AUTO_BACKEND_SETUP:-1}" in
    0|false|no|off) return 0 ;;
  esac
  local repo setup
  repo="$(backend_repo_dir "$backend")"
  [[ -z "$repo" || ! -d "$repo" ]] && return 0
  case "$backend" in
    prismatic) [[ -n "${PRISMATIC_PYTHON:-}" ]] && return 0 ;;
    cosmos) [[ -n "${COSMOS_PYTHON:-}" ]] && return 0 ;;
    groot) [[ -n "${GROOT_PYTHON:-}" ]] && return 0 ;;
  esac
  [[ -x "$repo/.venv/bin/python" ]] && return 0
  setup="$ROOT/setup_backend_uv_env.sh"
  if [[ ! -f "$setup" ]]; then
    echo "ERROR: missing $repo/.venv — run: bash $ROOT/setup_backend.sh $backend" >&2
    exit 2
  fi
  echo "[INFO] No .venv at $repo — running: bash $setup $backend" >&2
  bash "$setup" "$backend"
}

resolve_backend_python() {
  local backend="$1" py="" repo
  repo="$(backend_repo_dir "$backend")"
  case "$backend" in
    prismatic)
      py="${PRISMATIC_PYTHON:-}"
      [[ -z "$py" && -n "$repo" && -x "$repo/.venv/bin/python" ]] && py="$repo/.venv/bin/python"
      [[ -z "$py" && -x "$ROOT/.venv-prismatic/bin/python" ]] && py="$ROOT/.venv-prismatic/bin/python"
      [[ -z "$py" ]] && py="$(guess_conda_python prismatic || true)"
      [[ -z "$py" && -x "$GVL_PRISMATIC_DEFAULT" ]] && py="$GVL_PRISMATIC_DEFAULT"
      ;;
    cosmos)
      py="${COSMOS_PYTHON:-}"
      [[ -z "$py" && -n "$repo" && -x "$repo/.venv/bin/python" ]] && py="$repo/.venv/bin/python"
      [[ -z "$py" && -x "$ROOT/.venv-cosmos/bin/python" ]] && py="$ROOT/.venv-cosmos/bin/python"
      [[ -z "$py" && -x "$GVL_COSMOS_DEFAULT" ]] && py="$GVL_COSMOS_DEFAULT"
      ;;
    groot)
      py="${GROOT_PYTHON:-}"
      [[ -z "$py" && -n "$repo" && -x "$repo/.venv/bin/python" ]] && py="$repo/.venv/bin/python"
      [[ -z "$py" && -x "$ROOT/.venv-groot/bin/python" ]] && py="$ROOT/.venv-groot/bin/python"
      [[ -z "$py" && -x "$GVL_GROOT_DEFAULT" ]] && py="$GVL_GROOT_DEFAULT"
      ;;
    *)
      echo "ERROR: unsupported backend '$backend'" >&2
      exit 2
      ;;
  esac
  if [[ -z "$py" || ! -x "$py" ]]; then
    echo "ERROR: Python not found for backend '$backend'" >&2
    exit 2
  fi
  echo "$py"
}

has_flag() {
  local needle="$1" arg
  shift
  for arg in "$@"; do
    [[ "$arg" == "$needle" ]] && return 0
  done
  return 1
}

usage() {
  cat <<'EOF'
Usage:
  BACKEND=<backend> bash grounding_task.sh triplet_recognition_cholect50 [args...]
  BACKEND=<backend> bash grounding_task.sh phase_recognition_cholec80 [args...]
  BACKEND=<backend> bash grounding_task.sh instrument_localization_endovis17 [args...]

Example (EndoVis-17 instrument localization, full 236 queries):
  BACKEND=prismatic DEVICE_VISIBLE=0 \\
    bash grounding_task.sh instrument_localization_endovis17 --max-samples 5

Example (Cholec80 phase recognition, eval videos 41–80):
  BACKEND=prismatic DEVICE_VISIBLE=0 \\
    bash grounding_task.sh phase_recognition_cholec80 --video 41

Example (full eval, one model load, one VLM call per frame — default):
  CHOLECT50_VIDEOS_ROOT=/path/to/CholecT50/videos \
    BACKEND=prismatic DEVICE_VISIBLE=0 \
    bash grounding_task.sh triplet_recognition_cholect50 --prompt-mode mcq

  Do not loop per instrument in shell; omit --instrument to evaluate all annotations
  in a single process (same frame is inferred once, scored per GT triplet).

  # Open vocabulary (no option list), subsampled:
  bash grounding_task.sh triplet_recognition_cholect50 --prompt-mode ov --samples-only --video VID68

Env:
  CHOLECT50_CHALLENGE_VAL_ROOT  default: ../eval/cholect50-challenge-val
  CHOLECT50_VIDEOS_ROOT         frame images (required if not under dataset-root/videos)
  CHOLEC80_ROOT                 default: ../data/Cholec80 (falls back to ../data/cholec80)
  CHOLEC80_FRAMES_ROOT          0.1 fps frames (default: ../eval/cholec80/frames_0p1fps)
  ENDOVIS17_VQLA_ROOT           default: ../eval/EndoVis-17-VQLA
  DEVICE_VISIBLE                -> CUDA_VISIBLE_DEVICES (default 0)
  MODEL_ID                      default --model-id when omitted
  GROUNDING_TASK_AUTO_BACKEND_SETUP=0  skip auto uv install

Setup (first time):
  bash setup_backend.sh prismatic
  cp /path/to/.hf_token $ROOT/.hf_token   # or set --hf-token

Paths (defaults):
  eval labels: $ROOT/../eval/cholect50-challenge-val
  HF cache:    $ROOT/../.cache/huggingface
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

task="$1"
shift

case "$task" in
  triplet_recognition_cholect50) script="triplet_recognition_cholect50.py" ;;
  phase_recognition_cholec80) script="phase_recognition_cholec80.py" ;;
  instrument_localization_endovis17) script="instrument_localization_endovis17.py" ;;
  -h|--help|help) usage; exit 0 ;;
  *)
    echo "ERROR: unknown task '$task'" >&2
    usage
    exit 2
    ;;
esac

backend="${BACKEND:-prismatic}"
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--backend" ]]; then
    j=$((i + 1))
    [[ $j -le $# ]] && backend="${!j}"
    break
  fi
done

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found" >&2
  exit 2
fi

ensure_repo_uv_venv_if_needed "$backend"
python_bin="$(resolve_backend_python "$backend")"
append_unique_pythonpath "$ROOT"
repo="$(backend_repo_dir "$backend")"
[[ -n "$repo" ]] && append_unique_pythonpath "$repo"

set -- "$@" --backend "$backend"
[[ -n "${MODEL_ID:-}" ]] && ! has_flag "--model-id" "$@" && set -- "$@" --model-id "$MODEL_ID"

: "${DEVICE_VISIBLE:=0}"
export CUDA_VISIBLE_DEVICES="$DEVICE_VISIBLE"

if [[ "$task" == "triplet_recognition_cholect50" ]]; then
  ! has_flag "--dataset-root" "$@" && set -- "$@" --dataset-root "$CHOLECT50_CHALLENGE_VAL_ROOT"
  [[ -n "$CHOLECT50_VIDEOS_ROOT" ]] && ! has_flag "--videos-root" "$@" && set -- "$@" --videos-root "$CHOLECT50_VIDEOS_ROOT"
fi

if [[ "$task" == "phase_recognition_cholec80" ]]; then
  ! has_flag "--dataset-root" "$@" && set -- "$@" --dataset-root "$CHOLEC80_ROOT"
  ! has_flag "--split" "$@" && set -- "$@" --split eval
  [[ -n "$CHOLEC80_FRAMES_ROOT" ]] && ! has_flag "--frames-root" "$@" && set -- "$@" --frames-root "$CHOLEC80_FRAMES_ROOT"
fi

if [[ "$task" == "instrument_localization_endovis17" ]]; then
  ! has_flag "--dataset-root" "$@" && set -- "$@" --dataset-root "$ENDOVIS17_VQLA_ROOT"
  ! has_flag "--frames-root" "$@" && set -- "$@" --frames-root "$ENDOVIS17_VQLA_ROOT/left_frames"
  ! has_flag "--annotations-root" "$@" && set -- "$@" --annotations-root "$ENDOVIS17_VQLA_ROOT/vqla"
fi

exec uv run --python "$python_bin" "$script" "$@"
