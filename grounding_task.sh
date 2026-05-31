#!/usr/bin/env bash
# SurgVLM-style eval runner (triplet / phase / localization).
# Prismatic: bash setup_backend.sh prismatic  (../backend/prismatic-vlms)
# HF models (Qwen, InternVL, PaliGemma, Cosmos, GR00T, …): transformers env via HF_PYTHON
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BACKEND_ROOT_DEFAULT="$(cd "$ROOT/../backend" && pwd)"
VLA_ROOT="${VLA_ROOT_OVERRIDE:-$BACKEND_ROOT_DEFAULT}"
# Hub snapshots: <surgical>/.cache/huggingface/hub
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$ROOT/../.cache/huggingface}"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_ROOT/transformers}"

: "${GVL_PRISMATIC_DEFAULT:=$VLA_ROOT/prismatic-vlms/.venv/bin/python}"
: "${CHOLECT50_CHALLENGE_VAL_ROOT:=$ROOT/../eval/cholect50-challenge-val}"
: "${CHOLECT50_VIDEOS_ROOT:=}"
: "${CHOLEC80_ROOT:=$ROOT/../data/Cholec80}"
: "${CHOLEC80_EVAL_ROOT:=$ROOT/../eval/cholec80}"
: "${CHOLEC80_FRAMES_ROOT:=$CHOLEC80_EVAL_ROOT/frames_0p1fps}"
: "${ENDOVIS2017_ROOT:=$ROOT/../eval/endovis2017}"
: "${ENDOVIS18_VQA_ROOT:=$ROOT/../eval/EndoVis-18-VQA}"
: "${ENDOVIS2018_IMAGES_ROOT:=$ROOT/../eval/endovis2018}"
: "${ENDOSCAPES_ROOT:=$ROOT/../eval/endoscapes}"
: "${SARRARP50_ROOT:=$ROOT/../eval/sarrarp50}"
: "${SURGICAL_PROMPTS_JSON:=$ROOT/../eval/prompts/surgical_prompts.json}"

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

# Default HF interpreter: conda env "surgical" (override with export HF_PYTHON=...)
if [[ -z "${HF_PYTHON:-}" ]]; then
  HF_PYTHON="$(guess_conda_python surgical || true)"
  [[ -n "$HF_PYTHON" ]] && export HF_PYTHON
fi

is_prismatic_backend() {
  [[ "${1,,}" == "prismatic" ]]
}

is_api_backend() {
  case "${1,,}" in
    openai|gpt|chatgpt|gemini|google|claude|anthropic) return 0 ;;
    *) return 1 ;;
  esac
}

backend_repo_dir() {
  case "$1" in
    prismatic) echo "$VLA_ROOT/prismatic-vlms" ;;
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
  is_prismatic_backend "$backend" || return 0
  case "${GROUNDING_TASK_AUTO_BACKEND_SETUP:-1}" in
    0|false|no|off) return 0 ;;
  esac
  local repo setup
  repo="$(backend_repo_dir "$backend")"
  [[ -z "$repo" || ! -d "$repo" ]] && return 0
  [[ -n "${PRISMATIC_PYTHON:-}" ]] && return 0
  [[ -x "$repo/.venv/bin/python" ]] && return 0
  setup="$ROOT/setup_backend_uv_env.sh"
  if [[ ! -f "$setup" ]]; then
    echo "ERROR: missing $repo/.venv — run: bash $ROOT/setup_backend.sh prismatic" >&2
    exit 2
  fi
  echo "[INFO] No .venv at $repo — running: bash $setup prismatic" >&2
  bash "$setup" prismatic
}

resolve_backend_python() {
  local backend="$1" py="" repo
  if is_prismatic_backend "$backend"; then
    repo="$(backend_repo_dir prismatic)"
    py="${PRISMATIC_PYTHON:-}"
    [[ -z "$py" && -n "$repo" && -x "$repo/.venv/bin/python" ]] && py="$repo/.venv/bin/python"
    [[ -z "$py" && -x "$ROOT/.venv-prismatic/bin/python" ]] && py="$ROOT/.venv-prismatic/bin/python"
    [[ -z "$py" ]] && py="$(guess_conda_python prismatic || true)"
    [[ -z "$py" && -x "$GVL_PRISMATIC_DEFAULT" ]] && py="$GVL_PRISMATIC_DEFAULT"
  elif is_api_backend "$backend"; then
    py="${API_PYTHON:-${HF_PYTHON:-}}"
    [[ -z "$py" ]] && py="$(guess_conda_python surgical || true)"
    [[ -z "$py" && -x "$(command -v python3)" ]] && py="$(command -v python3)"
  else
    py="${HF_PYTHON:-}"
    [[ -z "$py" ]] && py="$(guess_conda_python surgical || true)"
    [[ -z "$py" && -x "$ROOT/.venv-hf/bin/python" ]] && py="$ROOT/.venv-hf/bin/python"
    [[ -z "$py" && -x "$(command -v python3)" ]] && py="$(command -v python3)"
  fi
  if [[ -z "$py" || ! -x "$py" ]]; then
    echo "ERROR: Python not found for backend '$backend'" >&2
    echo "  prismatic: set PRISMATIC_PYTHON or run setup_backend.sh prismatic" >&2
    echo "  hf/qwen3/internvl/…: conda env surgical or export HF_PYTHON=/path/to/python" >&2
    echo "  openai/gemini/claude: set API_PYTHON (lightweight env; needs requests via stdlib only)" >&2
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
  BACKEND=<backend> bash grounding_task.sh tissue_instrument_recognition_endovis18 [args...]
  BACKEND=<backend> bash grounding_task.sh cvs_evaluation_endoscapes [args...]
  BACKEND=<backend> bash grounding_task.sh language_grounding_surgical_prompts [args...]
  bash grounding_task.sh visual_cross_attention_cholect50 --video VID68 --frame 837 --query-from-gt-crop

Language grounding uses text-only inference (no image). Compatible backends:
  prismatic, hf, qwen3-*, cosmos-*, internvl3.5, paligemma2, groot, openai, gemini, claude

Backends:
  prismatic     TRI-ML prismatic-vlms (local backend package / checkpoint)
  hf            transformers AutoProcessor (set MODEL_ID or --model-id)
  qwen3-4b, qwen3-32b, cosmos-2b, cosmos-32b  size-specific (see backend_registry.py)
  internvl3.5, paligemma2, groot              other HF families
  openai, gpt, chatgpt, gemini, claude        cloud vision APIs (API keys in .openai_api_key etc.)

Examples:
  BACKEND=prismatic DEVICE_VISIBLE=0 \\
    bash grounding_task.sh instrument_localization_endovis17 --max-samples 5

  BACKEND=qwen3-4b HF_PYTHON=/path/to/python \\
    bash grounding_task.sh instrument_localization_endovis17 --max-samples 5

  BACKEND=qwen3-32b DEVICE_VISIBLE=0 \\
    bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol joint --max-samples 3

  BACKEND=cosmos-32b HF_PYTHON=/path/to/python \\
    bash grounding_task.sh phase_recognition_cholec80 --video 41

  BACKEND=prismatic bash grounding_task.sh language_grounding_surgical_prompts --limit 20
  BACKEND=qwen3-4b bash grounding_task.sh language_grounding_surgical_prompts --filter-subtype pit_to_verb
  BACKEND=gpt MODEL_ID=gpt-4o-mini bash grounding_task.sh language_grounding_surgical_prompts --limit 20

Env:
  HF_PYTHON                     Python for local HF backends (default: conda surgical)
  API_PYTHON                    Python for openai/gemini/claude (default: same as HF_PYTHON)
  PRISMATIC_PYTHON              Python for prismatic backend
  MODEL_ID                      default --model-id when omitted
  MODEL_NAME                    default --model-name (output folder slug) when omitted
  GROUNDING_TASK_AUTO_BACKEND_SETUP=0  skip auto uv install (prismatic only)
  SURGICAL_PROMPTS_JSON              default --dataset-json for language_grounding_surgical_prompts
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
  tissue_instrument_recognition_endovis18) script="tissue_instrument_recognition_endovis18.py" ;;
  cvs_evaluation_endoscapes) script="cvs_evaluation_endoscapes.py" ;;
  action_recognition_sarrarp50) script="action_recognition_sarrarp50.py" ;;
  language_grounding_surgical_prompts) script="language_grounding_surgical_prompts.py" ;;
  visual_cross_attention_cholect50) script="visual_cross_attention_cholect50.py" ;;
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
# prismatic: add prismatic-vlms so ``import prismatic`` works (backends.py also sets this).
[[ -n "$repo" ]] && append_unique_pythonpath "$repo"

set -- "$@" --backend "$backend"
[[ -n "${MODEL_ID:-}" ]] && ! has_flag "--model-id" "$@" && set -- "$@" --model-id "$MODEL_ID"
[[ -n "${MODEL_NAME:-}" ]] && ! has_flag "--model-name" "$@" && set -- "$@" --model-name "$MODEL_NAME"
[[ -n "${API_WORKERS:-}" ]] && ! has_flag "--api-workers" "$@" && set -- "$@" --api-workers "$API_WORKERS"

: "${DEVICE_VISIBLE:=0}"
export CUDA_VISIBLE_DEVICES="$DEVICE_VISIBLE"

if [[ "$task" == "triplet_recognition_cholect50" || "$task" == "visual_cross_attention_cholect50" ]]; then
  ! has_flag "--dataset-root" "$@" && set -- "$@" --dataset-root "$CHOLECT50_CHALLENGE_VAL_ROOT"
  [[ -n "$CHOLECT50_VIDEOS_ROOT" ]] && ! has_flag "--videos-root" "$@" && set -- "$@" --videos-root "$CHOLECT50_VIDEOS_ROOT"
fi

if [[ "$task" == "phase_recognition_cholec80" ]]; then
  ! has_flag "--dataset-root" "$@" && set -- "$@" --dataset-root "$CHOLEC80_ROOT"
  ! has_flag "--split" "$@" && set -- "$@" --split eval
  [[ -n "$CHOLEC80_FRAMES_ROOT" ]] && ! has_flag "--frames-root" "$@" && set -- "$@" --frames-root "$CHOLEC80_FRAMES_ROOT"
fi

if [[ "$task" == "instrument_localization_endovis17" ]]; then
  ! has_flag "--dataset-root" "$@" && set -- "$@" --dataset-root "$ENDOVIS2017_ROOT"
fi

if [[ "$task" == "tissue_instrument_recognition_endovis18" ]]; then
  ! has_flag "--vqa-root" "$@" && set -- "$@" --vqa-root "$ENDOVIS18_VQA_ROOT"
  ! has_flag "--images-root" "$@" && set -- "$@" --images-root "$ENDOVIS2018_IMAGES_ROOT"
fi

if [[ "$task" == "cvs_evaluation_endoscapes" ]]; then
  ! has_flag "--dataset-root" "$@" && set -- "$@" --dataset-root "$ENDOSCAPES_ROOT"
fi

if [[ "$task" == "action_recognition_sarrarp50" ]]; then
  ! has_flag "--dataset-root" "$@" && set -- "$@" --dataset-root "$SARRARP50_ROOT"
fi

if [[ "$task" == "language_grounding_surgical_prompts" ]]; then
  ! has_flag "--dataset-json" "$@" && set -- "$@" --dataset-json "$SURGICAL_PROMPTS_JSON"
fi

exec uv run --python "$python_bin" "$script" "$@"
