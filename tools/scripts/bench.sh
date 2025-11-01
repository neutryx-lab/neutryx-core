#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR/src"

python "$ROOT_DIR/examples/perf/run_mc_vs_analytic.py" "$@"
