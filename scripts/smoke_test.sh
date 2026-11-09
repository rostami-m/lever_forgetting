#!/usr/bin/env bash
# Prove the pipeline works before committing to ~60 GPU-hours.
#
#     scripts/smoke_test.sh
#
# The smoke run exercises the pipeline on a tiny model.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

hr() { printf '\n%s\n' "------------------------------------------------------------"; }

hr; echo "1/4  invariant tests (GPM projection algebra)"
python tests/test_gpm_projection.py

hr; echo "2/4  experiment inventory"
python -m ffcl list

hr; echo "3/4  environment"
python scripts/check_env.py

hr; echo "4/4  end-to-end pipeline on a tiny model"
python -m ffcl smoke --out-dir ./.smoke
echo "  wrote: ./.smoke  (remove with: rm -rf ./.smoke)"

hr; echo "SMOKE TEST PASSED"
