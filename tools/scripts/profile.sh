#!/usr/bin/env bash
set -euo pipefail
python -m cProfile -o .prof/profile.out examples/01_bs_vanilla.py
