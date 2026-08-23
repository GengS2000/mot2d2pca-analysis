#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT=${1:-"$ROOT/results/participant_bootstrap_30"}
N_BOOTSTRAP=${N_BOOTSTRAP:-30}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-20260716}
PYTHON_BIN=${PYTHON_BIN:-python}
RSCRIPT=${RSCRIPT:-/public/software/VersionHub/R/4.4.2/Rscript}
PARTITION=${PARTITION:-hebhcnormal02}
ACCOUNT=${ACCOUNT:-}
TOTAL_TASK_BUDGET=${TOTAL_TASK_BUDGET:-60}
EXISTING_TASKS=${EXISTING_TASKS:-12}
DEFAULT_EXPERIMENT_CONCURRENCY=$((TOTAL_TASK_BUDGET - EXISTING_TASKS))
((DEFAULT_EXPERIMENT_CONCURRENCY > 0)) || { echo "TOTAL_TASK_BUDGET must exceed EXISTING_TASKS" >&2; exit 2; }
STAGE1_MAX_CONCURRENT=${STAGE1_MAX_CONCURRENT:-$DEFAULT_EXPERIMENT_CONCURRENCY}
STAGE2_MAX_CONCURRENT=${STAGE2_MAX_CONCURRENT:-$DEFAULT_EXPERIMENT_CONCURRENCY}
STAGE1_TIME=${STAGE1_TIME:-06:00:00}
STAGE2_TIME=${STAGE2_TIME:-12:00:00}
AGG1_TIME=${AGG1_TIME:-01:00:00}
AGG2_TIME=${AGG2_TIME:-04:00:00}
MEM_PER_CPU=${MEM_PER_CPU:-3G}
RUN_STAGE2=${RUN_STAGE2:-false}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -x "$RSCRIPT" ]] || { echo "Rscript is not executable: $RSCRIPT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/logs/slurm" "$OUTPUT_ROOT/aggregate"

echo "Running Stage 1 preflight..."
"$PYTHON_BIN" participant_resampling_slurm/preflight.py
if [[ "$RUN_STAGE2" == "true" ]]; then
  echo "Running Stage 2 preflight..."
  "$RSCRIPT" participant_resampling_slurm/preflight_stage2.R
fi

STAGE1_TASKS=$((2 * N_BOOTSTRAP))
STAGE2_LOGICAL_TASKS=$((4 * N_BOOTSTRAP))
STAGE2_TASKS_PER_ARRAY=${STAGE2_TASKS_PER_ARRAY:-2}
((STAGE2_TASKS_PER_ARRAY >= 1)) || { echo "STAGE2_TASKS_PER_ARRAY must be at least 1" >&2; exit 2; }
STAGE2_ARRAY_TASKS=$(((STAGE2_LOGICAL_TASKS + STAGE2_TASKS_PER_ARRAY - 1) / STAGE2_TASKS_PER_ARRAY))
((STAGE1_MAX_CONCURRENT > STAGE1_TASKS)) && STAGE1_MAX_CONCURRENT=$STAGE1_TASKS
((STAGE2_MAX_CONCURRENT > STAGE2_ARRAY_TASKS)) && STAGE2_MAX_CONCURRENT=$STAGE2_ARRAY_TASKS

COMMON=(--nodes=1 --ntasks=1 --cpus-per-task=1 --chdir="$ROOT" --partition="$PARTITION")
[[ -n "$ACCOUNT" ]] && COMMON+=(--account="$ACCOUNT")
EXPORTS="ALL,BIOLOGY_ROOT=$ROOT,BOOTSTRAP_OUTPUT_ROOT=$OUTPUT_ROOT,PYTHON_BIN=$PYTHON_BIN,RSCRIPT=$RSCRIPT,N_BOOTSTRAP=$N_BOOTSTRAP,BOOTSTRAP_SEED=$BOOTSTRAP_SEED,STAGE2_TASKS_PER_ARRAY=$STAGE2_TASKS_PER_ARRAY"

STAGE1_JOB=$(sbatch --parsable "${COMMON[@]}" \
  --mem-per-cpu="$MEM_PER_CPU" --time="$STAGE1_TIME" \
  --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/slurm/stage1_%A_%a.out" \
  --error="$OUTPUT_ROOT/logs/slurm/stage1_%A_%a.err" \
  --array="0-$((STAGE1_TASKS - 1))%${STAGE1_MAX_CONCURRENT}" \
  slurm/participant_bootstrap_stage1.slurm)

AGG1_JOB=$(sbatch --parsable "${COMMON[@]}" \
  --mem-per-cpu="$MEM_PER_CPU" --time="$AGG1_TIME" \
  --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/slurm/stage1_aggregate_%j.out" \
  --error="$OUTPUT_ROOT/logs/slurm/stage1_aggregate_%j.err" \
  --dependency="afterany:${STAGE1_JOB}" \
  slurm/participant_bootstrap_stage1_aggregate.slurm)

if [[ "$RUN_STAGE2" == "true" ]]; then
  STAGE2_JOB=$(sbatch --parsable "${COMMON[@]}" \
    --mem-per-cpu="$MEM_PER_CPU" --time="$STAGE2_TIME" \
    --export="$EXPORTS" \
    --output="$OUTPUT_ROOT/logs/slurm/stage2_%A_%a.out" \
    --error="$OUTPUT_ROOT/logs/slurm/stage2_%A_%a.err" \
    --array="0-$((STAGE2_ARRAY_TASKS - 1))%${STAGE2_MAX_CONCURRENT}" \
    --dependency="afterany:${STAGE1_JOB}" \
    slurm/participant_bootstrap_stage2.slurm)

  AGG2_JOB=$(sbatch --parsable "${COMMON[@]}" \
    --mem-per-cpu="$MEM_PER_CPU" --time="$AGG2_TIME" \
    --export="$EXPORTS" \
    --output="$OUTPUT_ROOT/logs/slurm/stage2_aggregate_%j.out" \
    --error="$OUTPUT_ROOT/logs/slurm/stage2_aggregate_%j.err" \
    --dependency="afterany:${STAGE2_JOB}" \
    slurm/participant_bootstrap_stage2_aggregate.slurm)
else
  STAGE2_JOB="deferred"
  AGG2_JOB="deferred"
fi

cat <<EOF
Submitted participant-block bootstrap experiment.
Output:             $OUTPUT_ROOT
Partition:          $PARTITION
Memory per CPU:     $MEM_PER_CPU
Stage 1 array:      $STAGE1_JOB ($STAGE1_TASKS tasks; max concurrent=$STAGE1_MAX_CONCURRENT)
Stage 1 aggregate:  $AGG1_JOB
Stage 2 array:      $STAGE2_JOB ($STAGE2_LOGICAL_TASKS logical tasks in $STAGE2_ARRAY_TASKS array elements; max concurrent=$STAGE2_MAX_CONCURRENT)
Stage 2 aggregate:  $AGG2_JOB
Concurrency budget: existing=$EXISTING_TASKS + experiment<=${DEFAULT_EXPERIMENT_CONCURRENCY} => total<=${TOTAL_TASK_BUDGET}
Stage 2 is deferred by default to avoid account submit-job limits. After Stage 1 aggregation is complete, run:
  bash slurm/submit_participant_bootstrap_stage2.sh "$OUTPUT_ROOT"

Monitor:
  squeue -j $STAGE1_JOB${STAGE2_JOB:+,$STAGE2_JOB}
  tail -f "$OUTPUT_ROOT/logs/slurm/stage1_${STAGE1_JOB}_0.out"
EOF
