#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${ISAAC_PYTHON:?Set ISAAC_PYTHON to your separately installed Isaac Sim 4.5 Python}"
exec "$ISAAC_PYTHON" "$@"
