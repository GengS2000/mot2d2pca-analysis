#!/bin/bash
#SBATCH --job-name=p2d2_boot_glmmpql
#SBATCH --partition=hebhcnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=12:00:00
#SBATCH --array=0-0

set -euo pipefail
: "${BIOLOGY_ROOT:?BIOLOGY_ROOT must be exported}"
: "${BOOTSTRAP_OUTPUT_ROOT:?BOOTSTRAP_OUTPUT_ROOT must be exported}"
: "${RSCRIPT:=/public/software/VersionHub/R/4.4.2/Rscript}"
: "${N_BOOTSTRAP:=30}"

cd "$BIOLOGY_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TASK_ID=$SLURM_ARRAY_TASK_ID
TOTAL_TASKS=$((2 * N_BOOTSTRAP))
((TASK_ID >= 0 && TASK_ID < TOTAL_TASKS)) || { echo "Task ID $TASK_ID outside 0-$((TOTAL_TASKS - 1))" >&2; exit 2; }
if ((TASK_ID < N_BOOTSTRAP)); then
  COHORT=ECAM
  REPLICATE=$((TASK_ID + 1))
else
  COHORT=DIABIMMUNE
  REPLICATE=$((TASK_ID - N_BOOTSTRAP + 1))
fi

echo "[$(date '+%F %T')] glmmPQL task=$TASK_ID cohort=$COHORT bootstrap=$REPLICATE"
"$RSCRIPT" participant_resampling_slurm/stage2_inference_worker.R \
  --project-dir "$BIOLOGY_ROOT" \
  --output-root "$BOOTSTRAP_OUTPUT_ROOT" \
  --task-id "$TASK_ID" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --cohort "$COHORT" \
  --method glmmpql \
  --bootstrap-replicate "$REPLICATE" \
  --rscript "$RSCRIPT" \
  --cores 1 --resume true
