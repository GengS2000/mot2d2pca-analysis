#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT_ROOT=${1:-"$ROOT/results/permutation_1000"}
RSCRIPT=${RSCRIPT:-/public/software/VersionHub/R/4.4.2/Rscript}
N_PERM=${N_PERM:-1000}
PERM_SEED=${PERM_SEED:-20260715}
MAX_CONCURRENT=${MAX_CONCURRENT:-12}
PARTITION=${PARTITION:-hebhcnormal02}
ACCOUNT=${ACCOUNT:-}
WORKER_MEM=${WORKER_MEM:-3G}
WORKER_TIME=${WORKER_TIME:-04:00:00}
AGG_MEM=${AGG_MEM:-3G}
AGG_TIME=${AGG_TIME:-04:00:00}
MISSING="$OUTPUT_ROOT/aggregate/missing_permutation_ids.txt"

[[ -f "$MISSING" ]] || { echo "Missing ID file not found: $MISSING" >&2; exit 1; }
N_MISSING=$(grep -cve '^[[:space:]]*$' "$MISSING" || true)
((N_MISSING > 0)) || { echo "No missing permutations to resubmit."; exit 0; }
((MAX_CONCURRENT > N_MISSING)) && MAX_CONCURRENT=$N_MISSING

COMMON=(--nodes=1 --ntasks=1 --cpus-per-task=1 --chdir="$ROOT")
[[ -n "$PARTITION" ]] && COMMON+=(--partition="$PARTITION")
[[ -n "$ACCOUNT" ]] && COMMON+=(--account="$ACCOUNT")
EXPORTS="ALL,BIOLOGY_ROOT=$ROOT,PERM_OUTPUT_ROOT=$OUTPUT_ROOT,RSCRIPT=$RSCRIPT,N_PERM=$N_PERM,CHUNK_SIZE=1,PERM_SEED=$PERM_SEED,PERM_ID_FILE=$MISSING"

ARRAY_JOB=$(sbatch --parsable "${COMMON[@]}" --mem="$WORKER_MEM" --time="$WORKER_TIME" \
  --export="$EXPORTS" --output="$OUTPUT_ROOT/logs/slurm/retry_%A_%a.out" \
  --error="$OUTPUT_ROOT/logs/slurm/retry_%A_%a.err" \
  --array="0-$((N_MISSING - 1))%${MAX_CONCURRENT}" slurm/permutation_array.slurm)

AGG_JOB=$(sbatch --parsable "${COMMON[@]}" --mem="$AGG_MEM" --time="$AGG_TIME" \
  --export="$EXPORTS" --output="$OUTPUT_ROOT/logs/slurm/aggregate_retry_%j.out" \
  --error="$OUTPUT_ROOT/logs/slurm/aggregate_retry_%j.err" \
  --dependency="afterany:${ARRAY_JOB}" slurm/permutation_aggregate.slurm)

echo "Resubmitted $N_MISSING missing permutation IDs: array=$ARRAY_JOB aggregate=$AGG_JOB"
