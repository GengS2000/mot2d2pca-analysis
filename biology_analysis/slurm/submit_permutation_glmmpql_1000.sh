#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT=${1:-"$ROOT/results/permutation_1000"}
N_PERM=${N_PERM:-1000}
PERM_SEED=${PERM_SEED:-20260715}
CHUNK_SIZE=${CHUNK_SIZE:-2}
RSCRIPT=${RSCRIPT:-/public/software/VersionHub/R/4.4.2/Rscript}
PARTITION=${PARTITION:-hebhcnormal02}
ACCOUNT=${ACCOUNT:-}
WORKER_MEM=${WORKER_MEM:-3G}
WORKER_TIME=${WORKER_TIME:-04:00:00}
AGG_MEM=${AGG_MEM:-3G}
AGG_TIME=${AGG_TIME:-04:00:00}
MAX_SUBMITTED_TASKS=${MAX_SUBMITTED_TASKS:-1000}
MAX_PARALLEL_TASKS=${MAX_PARALLEL_TASKS:-200}
EXISTING_SUBMITTED_TASKS=${EXISTING_SUBMITTED_TASKS:-0}
EXISTING_RUNNING_TASKS=${EXISTING_RUNNING_TASKS:-0}
AGGREGATE_SUBDIR=${AGGREGATE_SUBDIR:-aggregate_glmmpql}

[[ -x "$RSCRIPT" ]] || { echo "Rscript is not executable: $RSCRIPT" >&2; exit 1; }
((CHUNK_SIZE >= 1)) || { echo "CHUNK_SIZE must be at least 1" >&2; exit 2; }
((MAX_SUBMITTED_TASKS > EXISTING_SUBMITTED_TASKS)) || { echo "No submitted-task budget remains" >&2; exit 2; }
((MAX_PARALLEL_TASKS > EXISTING_RUNNING_TASKS)) || { echo "No parallel-task budget remains" >&2; exit 2; }

TOTAL_IDS=$((N_PERM + 1))
ARRAY_TASKS=$(((TOTAL_IDS + CHUNK_SIZE - 1) / CHUNK_SIZE))
NEW_SUBMITTED_TASKS=$((ARRAY_TASKS + 1))
AVAILABLE_SUBMITTED_TASKS=$((MAX_SUBMITTED_TASKS - EXISTING_SUBMITTED_TASKS))
if ((NEW_SUBMITTED_TASKS > AVAILABLE_SUBMITTED_TASKS)); then
  echo "Need $NEW_SUBMITTED_TASKS submitted-task slots, but only $AVAILABLE_SUBMITTED_TASKS remain." >&2
  echo "Increase CHUNK_SIZE or wait for existing jobs to finish." >&2
  exit 2
fi

AVAILABLE_PARALLEL_TASKS=$((MAX_PARALLEL_TASKS - EXISTING_RUNNING_TASKS))
MAX_CONCURRENT=${MAX_CONCURRENT:-$AVAILABLE_PARALLEL_TASKS}
((MAX_CONCURRENT > AVAILABLE_PARALLEL_TASKS)) && MAX_CONCURRENT=$AVAILABLE_PARALLEL_TASKS
((MAX_CONCURRENT > ARRAY_TASKS)) && MAX_CONCURRENT=$ARRAY_TASKS

mkdir -p "$OUTPUT_ROOT/logs/slurm" "$OUTPUT_ROOT/$AGGREGATE_SUBDIR"
echo "Running glmmPQL permutation preflight..."
"$RSCRIPT" permutation_slurm/preflight_server.R --methods glmmpql

COMMON=(--nodes=1 --ntasks=1 --cpus-per-task=1 --chdir="$ROOT")
[[ -n "$PARTITION" ]] && COMMON+=(--partition="$PARTITION")
[[ -n "$ACCOUNT" ]] && COMMON+=(--account="$ACCOUNT")
EXPORTS="ALL,BIOLOGY_ROOT=$ROOT,PERM_OUTPUT_ROOT=$OUTPUT_ROOT,RSCRIPT=$RSCRIPT,N_PERM=$N_PERM,CHUNK_SIZE=$CHUNK_SIZE,PERM_SEED=$PERM_SEED,PERM_METHODS=glmmpql,AGGREGATE_SUBDIR=$AGGREGATE_SUBDIR"

ARRAY_JOB=$(sbatch --parsable "${COMMON[@]}" \
  --mem="$WORKER_MEM" --time="$WORKER_TIME" --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/slurm/glmmpql_worker_%A_%a.out" \
  --error="$OUTPUT_ROOT/logs/slurm/glmmpql_worker_%A_%a.err" \
  --array="0-$((ARRAY_TASKS - 1))%${MAX_CONCURRENT}" \
  slurm/permutation_array.slurm)

AGG_JOB=$(sbatch --parsable "${COMMON[@]}" \
  --mem="$AGG_MEM" --time="$AGG_TIME" --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/slurm/glmmpql_aggregate_%j.out" \
  --error="$OUTPUT_ROOT/logs/slurm/glmmpql_aggregate_%j.err" \
  --dependency="afterany:${ARRAY_JOB}" slurm/permutation_aggregate.slurm)

cat <<EOF
Submitted glmmPQL for permutation IDs 0-$N_PERM.
Existing output root: $OUTPUT_ROOT
Array job:           $ARRAY_JOB
Aggregate job:       $AGG_JOB
Array elements:      $ARRAY_TASKS (chunk size=$CHUNK_SIZE)
Max concurrent:      $MAX_CONCURRENT
New submitted tasks: $NEW_SUBMITTED_TASKS / available $AVAILABLE_SUBMITTED_TASKS
Aggregate output:    $OUTPUT_ROOT/$AGGREGATE_SUBDIR

Monitor:
  squeue -j $ARRAY_JOB,$AGG_JOB
  tail -f "$OUTPUT_ROOT/logs/slurm/glmmpql_worker_${ARRAY_JOB}_0.out"
EOF
