#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT=${1:-"$ROOT/results/permutation_1000"}
N_PERM=${N_PERM:-1000}
PERM_SEED=${PERM_SEED:-20260715}
CHUNK_SIZE=${CHUNK_SIZE:-10}
MAX_CONCURRENT=${MAX_CONCURRENT:-12}
RSCRIPT=${RSCRIPT:-/public/software/VersionHub/R/4.4.2/Rscript}
PARTITION=${PARTITION:-hebhcnormal02}
ACCOUNT=${ACCOUNT:-}
WORKER_MEM=${WORKER_MEM:-3G}
WORKER_TIME=${WORKER_TIME:-04:00:00}
AGG_MEM=${AGG_MEM:-3G}
AGG_TIME=${AGG_TIME:-04:00:00}

[[ -x "$RSCRIPT" ]] || { echo "Rscript is not executable: $RSCRIPT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/logs/slurm" "$OUTPUT_ROOT/aggregate"

echo "Running server preflight..."
"$RSCRIPT" permutation_slurm/preflight_server.R

TOTAL_IDS=$((N_PERM + 1))  # includes permutation_id=0 observed-label baseline
ARRAY_TASKS=$(((TOTAL_IDS + CHUNK_SIZE - 1) / CHUNK_SIZE))
((MAX_CONCURRENT > ARRAY_TASKS)) && MAX_CONCURRENT=$ARRAY_TASKS

COMMON=(--nodes=1 --ntasks=1 --cpus-per-task=1 --chdir="$ROOT")
[[ -n "$PARTITION" ]] && COMMON+=(--partition="$PARTITION")
[[ -n "$ACCOUNT" ]] && COMMON+=(--account="$ACCOUNT")
EXPORTS="ALL,BIOLOGY_ROOT=$ROOT,PERM_OUTPUT_ROOT=$OUTPUT_ROOT,RSCRIPT=$RSCRIPT,N_PERM=$N_PERM,CHUNK_SIZE=$CHUNK_SIZE,PERM_SEED=$PERM_SEED"

ARRAY_JOB=$(sbatch --parsable "${COMMON[@]}" \
  --mem="$WORKER_MEM" --time="$WORKER_TIME" \
  --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/slurm/worker_%A_%a.out" \
  --error="$OUTPUT_ROOT/logs/slurm/worker_%A_%a.err" \
  --array="0-$((ARRAY_TASKS - 1))%${MAX_CONCURRENT}" \
  slurm/permutation_array.slurm)

AGG_JOB=$(sbatch --parsable "${COMMON[@]}" \
  --mem="$AGG_MEM" --time="$AGG_TIME" \
  --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/slurm/aggregate_%j.out" \
  --error="$OUTPUT_ROOT/logs/slurm/aggregate_%j.err" \
  --dependency="afterany:${ARRAY_JOB}" \
  slurm/permutation_aggregate.slurm)

cat <<EOF
Submitted 1000 null permutations plus observed baseline.
Output:       $OUTPUT_ROOT
Array job:    $ARRAY_JOB
Aggregate:    $AGG_JOB
Array tasks:  $ARRAY_TASKS (chunk size=$CHUNK_SIZE, concurrency=$MAX_CONCURRENT)

Monitor:
  squeue -j $ARRAY_JOB,$AGG_JOB
  tail -f $OUTPUT_ROOT/logs/slurm/worker_${ARRAY_JOB}_0.out
EOF
