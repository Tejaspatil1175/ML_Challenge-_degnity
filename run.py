#!/usr/bin/env python
"""Project Task Runner (similar to npm scripts in Node.js).

Usage:
    python run.py test                 # Run all unit tests (like 'npm test')
    python run.py test -v              # Run tests with verbose output
    python run.py explore              # Run dataset exploration report
    python run.py normalize            # Run text and address normalization
    python run.py block                # Run multi-stage candidate blocking
    python run.py features             # Run feature extraction & labeling
    python run.py all                  # Run full pipeline end-to-end (like 'npm run all')
    python run.py all --sample 5000    # Run on a quick 5,000-row sample
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_tests(extra_args: list[str]) -> int:
    """Executes the complete pytest test suite."""
    print(">> Running test suite (pytest backend/tests/)...")
    cmd = [sys.executable, "-m", "pytest", "backend/tests/"] + extra_args
    return subprocess.run(cmd).returncode


def run_pipeline_stage(subcommand: str, extra_args: list[str]) -> int:
    """Executes a pipeline CLI subcommand."""
    import time
    from datetime import datetime

    start_time = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now_str}] >> Starting pipeline stage: '{subcommand}' with args: {extra_args}")
    cmd = [sys.executable, "-m", "backend.src.pipeline", subcommand] + extra_args
    ret = subprocess.run(cmd).returncode
    elapsed = time.time() - start_time
    if ret == 0:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >> Stage '{subcommand}' completed successfully in {elapsed:.2f}s.\n")
    else:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >> Stage '{subcommand}' exited with code {ret} in {elapsed:.2f}s.\n")
    return ret


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    command = sys.argv[1].lower()
    extra_args = sys.argv[2:]

    if command in ("test", "tests"):
        return run_tests(extra_args)
    elif command in ("all", "run-all", "start"):
        return run_pipeline_stage("run-all", extra_args)
    elif command in ("explore", "normalize", "block", "features", "train", "evaluate", "predict", "package"):
        return run_pipeline_stage(command, extra_args)
    else:
        print(f"[ERROR] Unknown command: '{command}'")
        print("Available commands: test, explore, normalize, block, features, train, evaluate, predict, package, all")
        return 1


if __name__ == "__main__":
    sys.exit(main())
