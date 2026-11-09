#!/usr/bin/env bash
# Merge shard outputs, then turn logs into the paper's numbers and figures.
#
#     scripts/collect.sh
#
# Safe to run while shards are still going, to see partial progress.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESULTS_DIR="${RESULTS_DIR:-./results}"
LOCAL="$RESULTS_DIR"
mkdir -p "$LOCAL"

# ffcl's own merge, for the case where shards wrote to separate directories.
shopt -s nullglob
parts=(./results.shard*)
if [ ${#parts[@]} -gt 0 ]; then
  echo "== merging ${#parts[@]} shard directories"
  python -m ffcl merge "${parts[@]}" --out-dir "$LOCAL"
fi

echo
echo "== logs -> paper numbers"
python -m ffcl analyze --out-dir "$LOCAL" --latex

echo
echo "== logs -> figures"
python -m ffcl figures --out-dir "$LOCAL"

echo
echo
echo "collect complete."
echo "  numbers : $LOCAL/paper_numbers.json, $LOCAL/tables.tex"
echo "  figures : $LOCAL/figures/*.pdf"
