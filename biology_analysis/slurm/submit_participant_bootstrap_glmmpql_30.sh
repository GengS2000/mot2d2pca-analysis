#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT=${1:-"$ROOT/results/participant_bootstrap_30"}
N_BOOTSTRAP=${N_BOOTSTRAP:-30}
RSCRIPT=${RSCRIPT:-/public/software/VersionHub/R/4.4.2/Rscript}
PARTITION=${PARTITION:-hebhcnormal02}
ACCOUNT=${ACCOUNT:-}
STAGE2_TIME=${STAGE2_TIME:-12:00:00}
AGG_TIME=${AGG_TIME:-04:00:00}
MEM_PER_CPU=${MEM_PER_CPU:-3G}
MAX_SUBMITTED_TASKS=${MAX_SUBMITTED_TASKS:-1000}
MAX_PARALLEL_TASKS=${MAX_PARALLEL_TASKS:-200}
EXISTING_SUBMITTED_TASKS=${EXISTING_SUBMITTED_TASKS:-0}
EXISTING_RUNNING_TASKS=${EXISTING_RUNNING_TASKS:-0}
AGGREGATE_SUBDIR=${AGGREGATE_SUBDIR:-aggregate_glmmpql}

[[ -x "$RSCRIPT" ]] || { echo "Rscript is not executable: $RSCRIPT" >&2; exit 1; }
STATUS="$OUTPUT_ROOT/aggregate/stage1_aggregate_status.json"
[[ -f "$STATUS" ]] || { echo "Stage 1 aggregate status is missing: $STATUS" >&2; exit 1; }
grep -q '"complete": true' "$STATUS" || { echo "Stage 1 is incomplete; do not run glmmPQL Stage 2." >&2; exit 1; }

ARRAY_TASKS=$((2 * N_BOOTSTRAP))
NEW_SUBMITTED_TASKS=$((ARRAY_TASKS + 1))
AVAILABLE_SUBMITTED_TASKS=$((MAX_SUBMITTED_TASKS - EXISTING_SUBMITTED_TASKS))
if ((NEW_SUBMITTED_TASKS > AVAILABLE_SUBMITTED_TASKS)); then
  echo "Need $NEW_SUBMITTED_TASKS submitted-task slots, but only $AVAILABLE_SUBMITTED_TASKS remain." >&2
  exit 2
fi
AVAILABLE_PARALLEL_TASKS=$((MAX_PARALLEL_TASKS - EXISTING_RUNNING_TASKS))
((AVAILABLE_PARALLEL_TASKS > 0)) || { echo "No parallel-task budget remains." >&2; exit 2; }
MAX_CONCURRENT=${MAX_CONCURRENT:-$AVAILABLE_PARALLEL_TASKS}
((MAX_CONCURRENT > AVAILABLE_PARALLEL_TASKS)) && MAX_CONCURRENT=$AVAILABLE_PARALLEL_TASKS
((MAX_CONCURRENT > ARRAY_TASKS)) && MAX_CONCURRENT=$ARRAY_TASKS

mkdir -p "$OUTPUT_ROOT/logs/slurm" "$OUTPUT_ROOT/$AGGREGATE_SUBDIR"
echo "Running glmmPQL bootstrap Stage 2 preflight..."
"$RSCRIPT" participant_resampling_slurm/preflight_stage2.R --methods glmmpql

COMMON=(--nodes=1 --ntasks=1 --cpus-per-task=1 --chdir="$ROOT" --partition="$PARTITION")
[[ -n "$ACCOUNT" ]] && COMMON+=(--account="$ACCOUNT")
EXPORTS="ALL,BIOLOGY_ROOT=$ROOT,BOOTSTRAP_OUTPUT_ROOT=$OUTPUT_ROOT,RSCRIPT=$RSCRIPT,N_BOOTSTRAP=$N_BOOTSTRAP,STAGE2_METHODS=glmmpql,AGGREGATE_SUBDIR=$AGGREGATE_SUBDIR"

STAGE2_JOB=$(sbatch --parsable "${COMMON[@]}" \
  --mem-per-cpu="$MEM_PER_CPU" --time="$STAGE2_TIME" --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/slurm/glmmpql_stage2_%A_%a.out" \
  --error="$OUTPUT_ROOT/logs/slurm/glmmpql_stage2_%A_%a.err" \
  --array="0-$((ARRAY_TASKS - 1))%${MAX_CONCURRENT}" \
  slurm/participant_bootstrap_glmmpql.slurm)

AGG_JOB=$(sbatch --parsable "${COMMON[@]}" \
  --mem-per-cpu="$MEM_PER_CPU" --time="$AGG_TIME" --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/slurm/glmmpql_stage2_aggregate_%j.out" \
  --error="$OUTPUT_ROOT/logs/slurm/glmmpql_stage2_aggregate_%j.err" \
  --dependency="afterany:${STAGE2_JOB}" slurm/participant_bootstrap_stage2_aggregate.slurm)

cat <<EOF
Submitted glmmPQL on the existing participant bootstrap Stage 1 outputs.
Existing output root: $OUTPUT_ROOT
Stage 2 array:       $STAGE2_JOB ($ARRAY_TASKS tasks; max concurrent=$MAX_CONCURRENT)
Aggregate:           $AGG_JOB
Aggregate output:    $OUTPUT_ROOT/$AGGREGATE_SUBDIR
Stage 1 was reused and was not resubmitted.

Monitor:
  squeue -j $STAGE2_JOB,$AGG_JOB
  tail -f "$OUTPUT_ROOT/logs/slurm/glmmpql_stage2_${STAGE2_JOB}_0.out"
EOF
