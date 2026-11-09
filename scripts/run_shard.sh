#!/usr/bin/env bash
# Run shard i of n of the experiment suite.
#
#     scripts/run_shard.sh 0 1     # the whole suite (~60 GPU-hours on one 48 GB GPU)
#     scripts/run_shard.sh 3 10    # one machine's share of it
#
# Sharding here is by EXPERIMENT, not by cell, because the eleven experiments differ by
# more than an order of magnitude in cost and several need a 48 GB card while others run
# on 24 GB. Dealing whole experiments round-robin keeps each shard self-contained: a
# shard either finishes an experiment or leaves it resumable, and no shard needs another
# shard's output. `ffcl` skips cells already present in its output log, so re-running a
# shard costs only what it had not finished.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SHARD="${1:?usage: run_shard.sh <shard-index-0-based> <shard-count> [extra ffcl args]}"
NSHARDS="${2:?usage: run_shard.sh <shard-index-0-based> <shard-count> [extra ffcl args]}"
shift 2

RESULTS_DIR="${RESULTS_DIR:-./results}"
mkdir -p "$RESULTS_DIR"
STAGE="$RESULTS_DIR"

# The eleven experiments, cheapest first so a shard produces something usable early.
EXPERIMENTS="published fullft probe locus geometry sequential capbroad cl_stream crossmodel intensity"

echo "== pre-flight"
python tests/test_gpm_projection.py
python -m ffcl doctor --out-dir "$STAGE"

i=0
MINE=""
for e in $EXPERIMENTS; do
  [ "$((i % NSHARDS))" -eq "$SHARD" ] && MINE="$MINE $e"
  i=$((i + 1))
done

if [ -z "$MINE" ]; then
  echo "shard $SHARD of $NSHARDS has no experiments assigned (more shards than experiments)"
  exit 0
fi

echo "== shard $SHARD of $NSHARDS runs:$MINE"
for e in $MINE; do
  echo "---- $e"
  python -m ffcl run "$e" --out-dir "$STAGE" "$@"
done

echo "shard $SHARD complete. When every shard is done: scripts/collect.sh"
