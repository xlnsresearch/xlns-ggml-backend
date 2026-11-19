#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../lib.sh"
lock_experiment dot-product-scaling

setup_build
ensure_models

lib_run python3 "$HERE/run.py" -m "$MODEL_SMOLLM" -m "$MODEL_QWEN" "$@"

write_report "$HERE/report.py" "$HERE/results/REPORT.md"
finish_work "$REPO_ROOT/build/exp-dot"
