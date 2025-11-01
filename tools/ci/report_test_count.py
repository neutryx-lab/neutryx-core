#!/usr/bin/env python3
"""Parse pytest collection output and publish the total test count."""
from __future__ import annotations

import os
import sys
from pathlib import Path


IGNORED_PREFIXES = (
    "=",
    "collected",
    "-",
    "________________",
)


def count_tests(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    total = 0
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith(IGNORED_PREFIXES):
            continue
        if line.lower().startswith("no tests collected"):
            return 0
        # pytest -q --collect-only emits `file::test_name`
        # Newer versions may emit `file.py: N` summary lines; treat the trailing
        # integer as a count of collected tests for that module.
        if ":" in line and line.rsplit(":", 1)[-1].strip().isdigit():
            total += int(line.rsplit(":", 1)[-1].strip())
        else:
            total += 1
    return total


def append_summary(message: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(message)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: report_test_count.py <collect-output>", file=sys.stderr)
        return 1

    count = count_tests(Path(argv[1]))
    summary = f"Total pytest items collected: {count}\n"
    print(summary, end="")
    append_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
