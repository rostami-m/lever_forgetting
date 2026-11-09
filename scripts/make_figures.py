#!/usr/bin/env python3
"""Generate figures from an experiment run directory.

    python scripts/make_figures.py                 # from ./results (your own run)
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=os.environ.get("RESULTS_DIR", "results"),
                    help="experiment run directory")
    a = ap.parse_args()

    env = dict(os.environ, MPLBACKEND="Agg")

    out = a.out_dir
    if not Path(out).is_dir():
        print(f"no run directory at {out}. Run experiments first.", file=sys.stderr)
        return 1
    rc = subprocess.run([sys.executable, "-m", "ffcl", "figures", "--out-dir", out],
                        cwd=ROOT, env=env).returncode
    print(f"\nfigures -> {out}/figures/" if rc == 0 else "\nfigure generation failed")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
