#!/usr/bin/env python3
"""Pre-flight check for a local run directory and the experiment environment.

    python scripts/check_env.py

`ffcl doctor` checks Python packages, the device, and dataset/model access. This adds a
writability check for the local output directory.
"""
from __future__ import annotations
import os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    rc = 0

    results = Path(os.environ.get("RESULTS_DIR", "./results"))
    print(f"== output directory: {results}")
    try:
        results.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=results):
            pass
    except OSError as exc:
        print(f"  cannot write to {results}: {exc}", file=sys.stderr)
        rc = 1

    print("\n== ffcl doctor")
    rc |= subprocess.run([sys.executable, "-m", "ffcl", "doctor", "--out-dir", str(results)],
                         cwd=ROOT).returncode

    print("\nenvironment ok" if rc == 0 else "\nproblems above; not ready to run",
          file=sys.stderr if rc else sys.stdout)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
