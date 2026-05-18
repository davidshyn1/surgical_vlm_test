#!/usr/bin/env bash
# 레거시 수동 설치 메모. 권장: bash setup_backend.sh [prismatic|cosmos|groot|all]
#
# 아래는 prismatic-vlms 수동 절차 예시 (CUDA/버전은 setup_backend_uv_env.sh 와 동기화됨).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_ROOT="${VLA_ROOT_OVERRIDE:-$(cd "$ROOT/../backend" && pwd)}"
PRISMATIC_DIR="${PRISMATIC_DIR:-$VLA_ROOT/prismatic-vlms}"

echo "Use: bash $ROOT/setup_backend.sh prismatic" >&2
echo "     (or: bash $ROOT/setup_backend_uv_env.sh all)" >&2
exit 1

# --- reference only (do not run blindly) ---
# cd "$PRISMATIC_DIR"
# uv venv --python 3.10 .venv
# source .venv/bin/activate
# uv pip install -e .
# uv pip install packaging ninja
# uv pip uninstall -y torch torchvision transformers
# uv pip install torch==2.11.0 torchvision==0.26.0 "transformers[torch]" \\
#   --extra-index-url https://download.pytorch.org/whl/cu130
# uv pip install flash-attn==2.8.3 --no-build-isolation
