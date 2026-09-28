#!/usr/bin/env bash
# Creates a new environment only; never changes the active/shared Python.
set -euo pipefail
cd "$(dirname "$0")/.."
envdir="${1:-.venv-train}"
if [[ -e "$envdir" ]]; then echo "Refusing to overwrite existing environment: $envdir" >&2; exit 1; fi
"${PYTHON_BASE:-python3.11}" -m venv "$envdir"
py="$envdir/bin/python"
"$py" -m pip install --upgrade pip
"$py" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$py" -m pip install -r requirements/eval.txt -r requirements/train-support.txt
# Historical compatibility runner, not PyPI's unrelated 2.3.1 implementation.
# mjlab metadata expects torch>=2.7 / RSL-RL 5.0.1; see docs/INSTALL.md.
"$py" -m pip install --no-deps mjlab==1.2.0 'rsl-rl-lib @ git+https://github.com/ccrpRepo/AMP_mjlab.git@2d6b0d1b1c00681accfe4bd9f244fad1eaa51940#subdirectory=rsl_rl'
echo "Created $envdir. Read docs/INSTALL.md before selecting an idle training GPU."
