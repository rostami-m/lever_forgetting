#!/usr/bin/env bash
# Slurm job array: one array task per shard.
#
# Submitted by `runner/launch.sh --backend slurm --shards N`, which sets the array
# range and exports SHARDS. Resource directives (--partition, --gres, --time) are
# passed on the sbatch command line rather than hard-coded here, because they are the
# part that differs on every cluster.
#
# You can also submit it directly:
#     sbatch --array=0-39 --gres=gpu:1 --export=ALL,SHARDS=40 runner/slurm_array.sbatch

#SBATCH --job-name=paper-repro
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runner/logs/shard-%a.out
#SBATCH --requeue

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

SHARD="${SLURM_ARRAY_TASK_ID:?not running under a Slurm array}"
NSHARDS="${SHARDS:?launch.sh should have exported SHARDS}"

# Node-local scratch for the model/dataset cache when the cluster provides it. Reading
# tens of GB of weights over a shared filesystem, from every array task at once, is
# slow enough to dominate the run.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export HF_HOME="${HF_HOME:-$SLURM_TMPDIR/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-$SLURM_TMPDIR/torch}"
fi

export RESULTS_DIR="${RESULTS_DIR:-./results}"

echo "shard $SHARD/$NSHARDS on $(hostname)  results -> $RESULTS_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

srun scripts/run_shard.sh "$SHARD" "$NSHARDS" "$@"
