#!/bin/bash
#SBATCH --job-name=p2d2_boot_da
#SBATCH --partition=hebhcnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=12:00:00
#SBATCH --array=0-59%48

set -euo pipefail
: "${BIOLOGY_ROOT:?BIOLOGY_ROOT must be exported}"
: "${BOOTSTRAP_OUTPUT_ROOT:?BOOTSTRAP_OUTPUT_ROOT must be exported}"
: "${RSCRIPT:=/public/software/VersionHub/R/4.4.2/Rscript}"
: "${N_BOOTSTRAP:=30}"
: "${STAGE2_TASKS_PER_ARRAY:=2}"
cd "$BIOLOGY_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TOTAL_LOGICAL_TASKS=$((4 * N_BOOTSTRAP))
FAILED=0

if [[ -n "${STAGE2_TASK_ID_FILE:-}" ]]; then
  LINE=$((SLURM_ARRAY_TASK_ID + 1))
  TASK_IDS=("$(sed -n "${LINE}p" "$STAGE2_TASK_ID_FILE")")
  [[ -n "${TASK_IDS[0]}" ]] || { echo "No Stage 2 task ID at line $LINE" >&2; exit 2; }
else
  START=$((SLURM_ARRAY_TASK_ID * STAGE2_TASKS_PER_ARRAY))
  END=$((START + STAGE2_TASKS_PER_ARRAY - 1))
  (( END >= TOTAL_LOGICAL_TASKS )) && END=$((TOTAL_LOGICAL_TASKS - 1))
  (( START < TOTAL_LOGICAL_TASKS )) || exit 0
  TASK_IDS=($(seq "$START" "$END"))
fi

for TASK_ID in "${TASK_IDS[@]}"; do
  echo "[$(date '+%F %T')] logical Stage 2 task=${TASK_ID}"
  if ! "$RSCRIPT" participant_resampling_slurm/stage2_inference_worker.R \
    --project-dir "$BIOLOGY_ROOT" \
    --output-root "$BOOTSTRAP_OUTPUT_ROOT" \
    --task-id "$TASK_ID" \
    --n-bootstrap "$N_BOOTSTRAP" \
    --rscript "$RSCRIPT" \
    --cores 1 --resume true; then
    FAILED=1
  fi
done

exit "$FAILED"
