#!/usr/bin/env bash
# Infrastructure-agnostic launcher.
#
# Fans the experiment grid out over N shards on a backend of your choice. The only
# thing a backend has to be able to do is invoke
#
#     scripts/run_shard.sh <shard-index> <shard-count>
#
# so adding a backend this file does not cover (LSF, PBS, Kubernetes, a CI matrix)
# means writing a loop around that one command -- see the `local` backend below for
# the shortest possible example.
#
# Nothing here is specific to any cloud. `--backend ssh` drives machines you have
# already provisioned, wherever they are; storage is a URI (see runner/README.md).
#
# Written for bash 3.2, which is what macOS ships. That rules out `mapfile`, `wait -n`,
# and bare "${arr[@]}" under `set -u`; if you extend this file, keep to that subset or
# it will break for every macOS user.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
REPO_NAME="$(basename "$REPO_ROOT")"

BACKEND=local
SHARDS=1
JOBS=1
GPUS=""
GPUS_PER_TASK=1
PARTITION=""
TIME_LIMIT="24:00:00"
HOSTS_FILE="runner/hosts.txt"
IMAGE="$REPO_NAME:repro"
DRY_RUN=0
EXTRA=()

usage() {
  cat <<EOF
Fan the experiment grid out over N shards on the backend of your choice.

Usage: runner/launch.sh [options] [-- extra args passed through to run_shard.sh]

  --backend {local|docker|slurm|ssh}   where to run              (default: local)
  --shards N                           split the grid N ways     (default: 1)
  -j, --jobs N                         concurrent shards, local/docker backends
  --gpus SPEC                          docker --gpus value, e.g. all or '"device=0"'
  --gpus-per-task N                    slurm --gres=gpu:N        (default: 1)
  --partition NAME                     slurm partition
  --time HH:MM:SS                      slurm time limit          (default: 24:00:00)
  --hosts FILE                         one hostname per line     (default: runner/hosts.txt)
  --image NAME                         docker image tag          (default: $IMAGE)
  --dry-run                            print the commands, run nothing
  -h, --help                           this message

Environment:
  RESULTS_DIR   local directory for results (default: ./results)

Examples:
  runner/launch.sh --backend local  --shards 1
  runner/launch.sh --backend local  --shards 8  -j 2
  runner/launch.sh --backend docker --shards 4  --gpus all
  runner/launch.sh --backend slurm  --shards 40 --partition gpu
  runner/launch.sh --backend ssh    --shards 40 --hosts runner/hosts.txt
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --backend)        BACKEND="$2"; shift 2 ;;
    --shards)         SHARDS="$2"; shift 2 ;;
    -j|--jobs)        JOBS="$2"; shift 2 ;;
    --gpus)           GPUS="$2"; shift 2 ;;
    --gpus-per-task)  GPUS_PER_TASK="$2"; shift 2 ;;
    --partition)      PARTITION="$2"; shift 2 ;;
    --time)           TIME_LIMIT="$2"; shift 2 ;;
    --hosts)          HOSTS_FILE="$2"; shift 2 ;;
    --image)          IMAGE="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; EXTRA=("$@"); break ;;
    *) echo "launch.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
done

export RESULTS_DIR="${RESULTS_DIR:-./results}"

# bash 3.2: expanding an empty array under `set -u` is an error, so guard every use.
extra_args() { if [ ${#EXTRA[@]} -gt 0 ]; then printf '%s\n' "${EXTRA[@]}"; fi; }
EXTRA_STR="$(extra_args | tr '\n' ' ')"

run() {
  if [ "$DRY_RUN" -eq 1 ]; then printf '  +'; printf ' %q' "$@"; printf '\n'
  else "$@"; fi
}

# Block until fewer than $JOBS background jobs remain. Replaces `wait -n`, which
# bash 3.2 does not have.
throttle() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$JOBS" ]; do sleep 2; done
}

echo "backend=$BACKEND shards=$SHARDS jobs=$JOBS RESULTS_DIR=$RESULTS_DIR"
[ "$DRY_RUN" -eq 1 ] && echo "(dry run -- nothing will execute)"

LAST_SHARD=$((SHARDS - 1))

case "$BACKEND" in

  local)
    # The reference path. Every other backend is an optimisation of this loop.
    if [ "$JOBS" -le 1 ] || [ "$DRY_RUN" -eq 1 ]; then
      for i in $(seq 0 $LAST_SHARD); do
        run scripts/run_shard.sh "$i" "$SHARDS" $EXTRA_STR
      done
    else
      fail=0
      for i in $(seq 0 $LAST_SHARD); do
        throttle
        scripts/run_shard.sh "$i" "$SHARDS" $EXTRA_STR &
      done
      # bash 3.2 has no `wait -p`; wait for all and take a non-zero from any child.
      while wait -n 2>/dev/null; do :; done || true
      wait || fail=1
      [ "$fail" -eq 0 ] || { echo "at least one shard failed; see the output above" >&2; exit 1; }
    fi
    ;;

  docker)
    command -v docker >/dev/null || { echo "docker not found on PATH" >&2; exit 1; }
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
      echo "image $IMAGE not present; building it"
      run docker build -t "$IMAGE" -f runner/Dockerfile .
    fi
    for i in $(seq 0 $LAST_SHARD); do
      [ "$JOBS" -gt 1 ] && [ "$DRY_RUN" -eq 0 ] && throttle
      set -- docker run --rm
      [ -n "$GPUS" ] && set -- "$@" --gpus "$GPUS"
      set -- "$@" -e "RESULTS_DIR=$RESULTS_DIR" -e "HF_TOKEN=${HF_TOKEN:-}" \
                  -v "$REPO_ROOT:/work" -v "${HF_HOME:-$HOME/.cache/huggingface}:/cache/huggingface" \
                  -w /work --name "$REPO_NAME-shard$i" \
                  "$IMAGE" scripts/run_shard.sh "$i" "$SHARDS" $EXTRA_STR
      if [ "$JOBS" -le 1 ] || [ "$DRY_RUN" -eq 1 ]; then run "$@"; else "$@" & fi
    done
    [ "$DRY_RUN" -eq 0 ] && wait
    ;;

  slurm)
    command -v sbatch >/dev/null || { echo "sbatch not found on PATH" >&2; exit 1; }
    mkdir -p runner/logs
    set -- sbatch --array="0-$LAST_SHARD" --gres="gpu:$GPUS_PER_TASK" \
                  --time="$TIME_LIMIT" --job-name="$REPO_NAME-repro" \
                  --output="runner/logs/shard-%a.out"
    [ -n "$PARTITION" ] && set -- "$@" --partition="$PARTITION"
    set -- "$@" --export="ALL,SHARDS=$SHARDS,RESULTS_DIR=$RESULTS_DIR" \
                runner/slurm_array.sbatch $EXTRA_STR
    run "$@"
    echo "submitted. watch: squeue --me   logs: runner/logs/shard-*.out"
    ;;

  ssh)
    [ -f "$HOSTS_FILE" ] || { echo "no hosts file at $HOSTS_FILE" >&2; exit 1; }
    HOSTS=""
    NHOSTS=0
    while IFS= read -r line; do
      case "$line" in ''|\#*) continue ;; esac
      HOSTS="$HOSTS $line"
      NHOSTS=$((NHOSTS + 1))
    done < "$HOSTS_FILE"
    [ "$NHOSTS" -gt 0 ] || { echo "$HOSTS_FILE lists no hosts" >&2; exit 1; }
    set -- $HOSTS
    echo "$NHOSTS host(s), $SHARDS shard(s); shards are dealt round-robin"
    mkdir -p runner/logs
    # Ship the repo first. results/ is excluded so a worker never overwrites the
    # merged tree with its own partial view of it.
    for h in $HOSTS; do
      run rsync -az --delete --exclude results --exclude .git --exclude .stage \
        --exclude runner/logs "$REPO_ROOT/" "$h:~/$REPO_NAME/"
    done
    for i in $(seq 0 $LAST_SHARD); do
      eval "h=\${$(( (i % NHOSTS) + 1 ))}"
      run ssh -o BatchMode=yes "$h" \
        "cd ~/$REPO_NAME && RESULTS_DIR='$RESULTS_DIR' nohup scripts/run_shard.sh $i $SHARDS $EXTRA_STR > shard-$i.log 2>&1 & echo started shard $i"
    done
    echo "launched. per-shard logs live on each host as ~/$REPO_NAME/shard-N.log"
    ;;

  *) echo "unknown backend: $BACKEND" >&2; exit 2 ;;
esac

echo "next: scripts/collect.sh"
