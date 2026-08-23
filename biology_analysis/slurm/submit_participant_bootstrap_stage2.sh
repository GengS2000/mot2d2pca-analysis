#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT=${1:-"$ROOT/results/participant_bootstrap_30"}
N_BOOTSTRAP=${N_BOOTSTRAP:-30}
RSCRIPT=${RSCRIPT:-/public/software/VersionHub/R/4.4.2/Rscript}
PARTITION=${PARTITION:-hebhcnormal02}
ACCOUNT=${ACCOUNT:-}
TOTAL_TASK_BUDGET=${TOTAL_TASK_BUDGET:-60}
EXISTING_TASKS=${EXISTING_TASKS:-12}
DEFAULT_EXPERIMENT_CONCURRENCY=$((TOTAL_TASK_BUDGET - EXISTING_TASKS))
STAGE2_MAX_CONCURRENT=${STAGE2_MAX_CONCURRENT:-$DEFAULT_EXPERIMENT_CONCURRENCY}
STAGE2_TASKS_PER_ARRAY=${STAGE2_TASKS_PER_ARRAY:-2}
STAGE2_TIME=${STAGE2_TIME:-12:00:00}
AGG2_TIME=${AGG2_TIME:-04:00:00}
MEM_PER_CPU=${MEM_PER_CPU:-3G}

((DEFAULT_EXPERIMENT_CONCURRENCY > 0)) || { echo "TOTAL_TASK_BUDGET must exceed EXISTING_TASKS" >&2; exit 2; }
((STAGE2_TASKS_PER_ARRAY >= 1)) || { echo "STAGE2_TASKS_PER_ARRAY must be at least 1" >&2; exit 2; }
[[ -x "$RSCRIPT" ]] || { echo "Rscript is not executable: $RSCRIPT" >&2; exit 1; }
STATUS="$OUTPUT_ROOT/aggregate/stage1_aggregate_status.json"
[[ -f "$STATUS" ]] || { echo "Stage 1 aggregate status is missing: $STATUS" >&2; exit 1; }
grep -q '"complete": true' "$STATUS" || { echo "Stage 1 is incomplete; inspect aggregate/missing_stage1_task_ids.txt first." >&2; exit 1; }

echo "Running Stage 2 preflight..."
"$RSCRIPT" participant_resampling_slurm/preflight_stage2.R

STAGE2_LOGICAL_TASKS=$((4 * N_BOOTSTRAP))
STAGE2_ARRAY_TASKS=$(((STAGE2_LOGICAL_TASKS + STAGE2_TASKS_PER_ARRAY - 1) / STAGE2_TASKS_PER_ARRAY))
((STAGE2_MAX_CONCURRENT > STAGE2_ARRAY_TASKS)) && STAGE2_MAX_CONCURRENT=$STAGE2_ARRAY_TASKS
mkdir -p "$OUTPUT_ROOT/logs/slurm" "$OUTPUT_ROOT/aggregate"

COMMON=(--nodes=1 --ntasks=1 --cpus-per-task=1 --chdir="$ROOT" --partition="$PARTITION")
[[ -n "$ACCOUNT" ]] && COMMON+=(--account="$ACCOUNT")
EXPORTS="ALL,BIOLOGY_ROOT=$ROOT,BOOTSTRAP_OUTPUT_ROOT=$OUTPUT_ROOT,RSCRIPT=$RSCRIPT,N_BOOTSTRAP=$N_BOOTSTRAP,STAGE2_TASKS_PER_ARRAY=$STAGE2_TASKS_PER_ARRAY"

STAGE2_JOB=$(sbatch --parsable "${COMMON[@]}" \
  --mem-per-cpu="$MEM_PER_CPU" --time="$STAGE2_TIME" --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/slurm/stage2_%A_%a.out" \
  --error="$OUTPUT_ROOT/logs/slurm/stage2_%A_%a.err" \
  --array="0-$((STAGE2_ARRAY_TASKS - 1))%${STAGE2_MAX_CONCURRENT}" \
  slurm/participant_bootstrap_stage2.slurm)

AGG2_JOB=$(sbatch --parsable "${COMMON[@]}" \
  --mem-per-cpu="$MEM_PER_CPU" --time="$AGG2_TIME" --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/slurm/stage2_aggregate_%j.out" \
  --error="$OUTPUT_ROOT/logs/slurm/stage2_aggregate_%j.err" \
  --dependency="afterany:${STAGE2_JOB}" slurm/participant_bootstrap_stage2_aggregate.slurm)

echo "Submitted Stage 2: logical tasks=$STAGE2_LOGICAL_TASKS; array elements=$STAGE2_ARRAY_TASKS; max concurrent=$STAGE2_MAX_CONCURRENT"
echo "Stage 2 array job: $STAGE2_JOB"
echo "Stage 2 aggregate: $AGG2_JOB"
