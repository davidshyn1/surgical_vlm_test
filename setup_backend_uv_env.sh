#!/usr/bin/env bash
# 각 backend 리포지토리 안에 .venv 를 두고, uv 로 생성·설치.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
BACKEND_ROOT_DEFAULT="$(cd "$ROOT/../backend" && pwd)"
VLA_ROOT="${VLA_ROOT_OVERRIDE:-$BACKEND_ROOT_DEFAULT}"
CUDA_131_HOME="${CUDA_131_HOME:-}"
CUDA_13_HOME="${CUDA_13_HOME:-}"
CUDA_130_HOME="${CUDA_130_HOME:-}"
PYTORCH_CUDA_EXTRA_INDEX="${PYTORCH_CUDA_EXTRA_INDEX:-https://download.pytorch.org/whl/cu130}"

PRISMATIC_DIR="${PRISMATIC_DIR:-$VLA_ROOT/prismatic-vlms}"
COSMOS_DIR="${COSMOS_DIR:-$VLA_ROOT/cosmos-reason2}"
GROOT_DIR="${GROOT_DIR:-$VLA_ROOT/GR00T-H}"
INTERVL_DIR="${INTERVL_DIR:-$VLA_ROOT/InternVL}"

usage() {
  cat <<'EOF'
Usage:
  bash setup_backend_uv_env.sh <backend|all>

Backends:
  prismatic  — .venv in prismatic-vlms (torch 2.11 / transformers / flash-attn 2.8.3)
  cosmos     — uv sync --extra cu130 in cosmos-reason2
  groot      — uv sync + editable gr00t + flash-attn in GR00T-H
  internvl   — .venv in InternVL (py3.9, requirements + clip_benchmark)
  all        — 위 순서대로 전부

Repo 경로 기본값: VLA_ROOT 아래
  prismatic-vlms, cosmos-reason2, GR00T-H, InternVL

환경 변수:
  VLA_ROOT_OVERRIDE=/path/to/backend-parent
  PRISMATIC_DIR / COSMOS_DIR / GROOT_DIR / INTERVL_DIR
  CUDA_131_HOME=/usr/local/cuda-13.1   # 권장: CUDA 13.1 toolkit root (nvcc 필수)
  CUDA_13_HOME / CUDA_130_HOME         # 비우면 무시; 있으면 CUDA_131_HOME과 동일 우선순위
  PYTORCH_CUDA_EXTRA_INDEX=...         # 기본 cu130 (공식 휠)

uv 는 PATH 에 있어야 합니다. 각 명령은 해당 리포의 .venv/bin/python 을 명시합니다.
EOF
}

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found on PATH." >&2
  exit 2
fi

_nvcc_output_is_cuda_131() {
  [[ "$1" =~ release[[:space:]]+13\.1[[:space:]]*, ]]
}

ensure_cuda_131() {
  local root cand nvcc_bin nvcc_v
  nvcc_bin=""
  local first="${CUDA_131_HOME:-${CUDA_13_HOME:-${CUDA_130_HOME:-}}}"
  for cand in "${first:-}" /usr/local/cuda-13.1 /usr/local/cuda; do
    [[ -z "$cand" ]] && continue
    [[ -x "$cand/bin/nvcc" ]] || continue
    nvcc_v="$("$cand/bin/nvcc" --version 2>/dev/null || true)"
    if _nvcc_output_is_cuda_131 "$nvcc_v"; then
      root="$cand"
      nvcc_bin="$cand/bin/nvcc"
      break
    fi
  done
  if [[ -z "$nvcc_bin" ]]; then
    echo "ERROR: CUDA 13.1 nvcc not found (need 'release 13.1,' in nvcc --version)." >&2
    echo "       Tried: CUDA_131_HOME/CUDA_13_HOME/CUDA_130_HOME, /usr/local/cuda-13.1, /usr/local/cuda" >&2
    echo "       Set CUDA_131_HOME to the toolkit root (e.g. /usr/local/cuda-13.1)." >&2
    exit 2
  fi
  export CUDA_HOME="$root"
  export CUDACXX="$nvcc_bin"
  export NVCC="$nvcc_bin"
  export CUDA_PATH="$root"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

  nvcc_v="$("$NVCC" --version 2>/dev/null || true)"
  echo "[INFO] nvcc CUDA 13.1: NVCC=$NVCC"
  echo "[INFO] CUDA_HOME=$CUDA_HOME"
}

install_prismatic() {
  echo "[INFO] === prismatic-vlms (.venv) ==="
  ensure_cuda_131
  cd "$PRISMATIC_DIR"
  uv venv --python 3.10 .venv
  local py="$PRISMATIC_DIR/.venv/bin/python"
  uv pip install --python "$py" -e .
  uv pip install --python "$py" packaging ninja
  uv pip uninstall --python "$py" -y torch torchvision transformers 2>/dev/null || true
  uv pip install --python "$py" torch==2.11.0 torchvision==0.26.0 "transformers[torch]" \
    --extra-index-url "$PYTORCH_CUDA_EXTRA_INDEX"
  uv pip install --python "$py" flash-attn==2.8.3 --no-build-isolation
  echo "[DONE] prismatic: $PRISMATIC_DIR/.venv"
}

install_cosmos() {
  echo "[INFO] === cosmos-reason2 (uv sync → .venv) ==="
  cd "$COSMOS_DIR"
  uv sync --extra cu130
  echo "[DONE] cosmos: $COSMOS_DIR/.venv (or uv-managed env in repo)"
}

install_groot() {
  echo "[INFO] === GR00T-H (uv sync + editable) ==="
  ensure_cuda_131
  cd "$GROOT_DIR"
  uv sync --python 3.10
  local py="$GROOT_DIR/.venv/bin/python"
  uv pip install --python "$py" -e .
  uv pip install --python "$py" "flash-attn==2.7.4.post1" --no-build-isolation
  echo "[DONE] groot: $GROOT_DIR/.venv"
}

install_internvl() {
  echo "[INFO] === InternVL (.venv, Python 3.9) ==="
  ensure_cuda_131
  cd "$INTERVL_DIR"
  uv venv --python 3.9 .venv
  local py="$INTERVL_DIR/.venv/bin/python"
  uv pip install --python "$py" -r requirements.txt
  uv pip install --python "$py" "setuptools<82" wheel
  uv pip install --python "$py" -r requirements/clip_benchmark.txt --no-build-isolation
  uv pip install --python "$py" flash-attn==2.3.6 --no-build-isolation
  echo "[DONE] internvl: $INTERVL_DIR/.venv"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

backend="$1"
case "$backend" in
  -h|--help|help)
    usage
    exit 0
    ;;
  prismatic) install_prismatic ;;
  cosmos) install_cosmos ;;
  groot) install_groot ;;
  internvl) install_internvl ;;
  all)
    install_prismatic
    install_cosmos
    install_groot
    install_internvl
    echo "[DONE] all backends (uv) under VLA_ROOT=$VLA_ROOT"
    ;;
  *)
    echo "ERROR: unknown backend '$backend'" >&2
    usage
    exit 2
    ;;
esac

echo "[INFO] 예: BACKEND=prismatic bash $ROOT/grounding_task.sh triplet_recognition_cholect50 --limit 5"
