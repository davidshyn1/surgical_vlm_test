#!/usr/bin/env bash
# 백엔드별 repo 안 .venv 를 uv 로만 생성·설치 (setup_backend_uv_env.sh 위임).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage:
  bash $0 [backend|all]

  all (기본) — prismatic, cosmos, groot, internvl 순서로 uv 설치
  prismatic | cosmos | groot | internvl — 해당 백엔드만

실제 로직: $ROOT/setup_backend_uv_env.sh

환경 변수: VLA_ROOT_OVERRIDE (기본: $ROOT/../backend)
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

exec bash "$ROOT/setup_backend_uv_env.sh" "${1:-all}"
