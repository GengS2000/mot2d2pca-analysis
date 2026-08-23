#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT_ROOT=${1:-"$ROOT/results/participant_bootstrap_30"}
RSCRIPT=${RSCRIPT:-/public/software/VersionHub/R/4.4.2/Rscript}
N_BOOTSTRAP=${N_BOOTSTRAP:-30}
PARTITION=${PARTITION:-hebhcnormal02}
ACCOUNT=${ACCOUNT:-}
MEM_PER_CPU=${MEM_PER_CPU:-3G}
MAX_CONCURRENT=${MAX_CONCURRENT:-60}
STAGE2_TIME=${STAGE2_TIME:-12:00:00}
AGG_TIME=${AGG_TIME:-04:00:00}
AGGREGATE_SUBDIR=${AGGREGATE_SUBDIR:-aggregate_glmmpql}
MISSING="$OUTPUT_ROOT/$AGGREGATE_SUBDIR/missing_stage2_task_ids.txt"

[[ -f "$MISSING" ]] || { echo "Missing-task file not found: $MISSING" >&2; exit 1; }
N_MISSING=$(grep -cve '^[[:space:]]*$' "$MISSING" || true)
((N_MISSING > 0)) || { echo "No missing glmmPQL bootstrap tasks."; exit 0; }
((MAX_CONCURRENT > N_MISSING)) && MAX_CONCURRENT=$N_MISSING
ARRAY_SPEC="$(paste -sd, "$MISSING")%${MAX_CONCURRENT}"

COMMON=(--nodes=1 --ntasks=1 --cpus-per-task=1 --chdir="$ROOT" --partition="$PARTITION")
[[ -n "$ACCOUNT" ]] && COMMON+=(--account="$ACCOUNT")
EXPORTS="ALL,BIOLOGY_ROOT=$ROOT,BOOTSTRAP_OUTPUT_ROOT=$OUTPUT_ROOT,RSCRIPT=$RSCRIPT,N_BOOTSTRAP=$N_BOOTSTRAP,STAGE2_METHODS=glmmpql,AGGREGATE_SUBDIR=$AGGREGATE_SUBDIR"

STAGE2_JOB=$(sbatch --parsable "${COMMON[@]}" \
  --mem-per-cpu="$MEM_PER_CPU" --time="$STAGE2_TIME" --export="$EXPORTS" \
  --array="$ARRAY_SPEC" \
  --output="$OUTPUT_ROOT/logs/slurm/glmmpql_stage2_retry_%A_%a.out" \
  --error="$OUTPUT_ROOT/logs/slurm/glmmpql_stage2_retry_%A_%a.err" \
  slurm/participant_bootstrap_glmmpql.slurm)

AGG_JOB=$(sbatch --parsable "${COMMON[@]}" \
  --mem-per-cpu="$MEM_PER_CPU" --time="$AGG_TIME" --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/slurm/glmmpql_stage2_aggregate_retry_%j.out" \
  --error="$OUTPUT_ROOT/logs/slurm/glmmpql_stage2_aggregate_retry_%j.err" \
  --dependency="afterany:${STAGE2_JOB}" slurm/participant_bootstrap_stage2_aggregate.slurm)

echo "Resubmitted $N_MISSING missing glmmPQL bootstrap tasks: array=$STAGE2_JOB aggregate=$AGG_JOB"
