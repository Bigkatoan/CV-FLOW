#!/usr/bin/env bash
# scripts/build_wheel.sh — build a cv-flow wheel and drop it in ~/wheels for
# fully offline reuse from other projects on this (or another) machine.
#
# Usage:
#   scripts/build_wheel.sh [output_dir]   # default output_dir: ~/wheels
#
# Then, from another project's venv:
#   pip install --no-index --find-links ~/wheels "cv-flow[gpu]"
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$HOME/wheels}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUT_DIR"

"$PYTHON" -m pip show build >/dev/null 2>&1 || "$PYTHON" -m pip install --quiet build

cd "$REPO_DIR"
"$PYTHON" -m build --wheel --outdir "$OUT_DIR"

echo ""
echo "Wheel built in: $OUT_DIR"
ls -la "$OUT_DIR"/cv_flow-*.whl
echo ""
echo "Install from another project with:"
echo "  pip install --no-index --find-links \"$OUT_DIR\" \"cv-flow[gpu]\""
