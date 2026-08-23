#!/bin/bash
#SBATCH --job-name=mot_perm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=04:00:00
#SBATCH --array=0-0

set -euo pipefail
: "${BIOLOGY_ROOT:?BIOLOGY_ROOT must be exported by the submitter}"
: "${PERM_OUTPUT_ROOT:?PERM_OUTPUT_ROOT must be exported by the submitter}"
: "${RSCRIPT:=/public/software/VersionHub/R/4.4.2/Rscript}"
: "${N_PERM:=1000}"
: "${CHUNK_SIZE:=10}"
: "${PERM_SEED:=20260715}"
: "${PERM_METHODS:=linda,maaslin2}"

cd "$BIOLOGY_ROOT"
mkdir -p "$PERM_OUTPUT_ROOT/logs/slurm"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [[ -n "${PERM_ID_FILE:-}" ]]; then
  LINE=$((SLURM_ARRAY_TASK_ID + 1))
  PERM_ID=$(sed -n "${LINE}p" "$PERM_ID_FILE")
  [[ -n "$PERM_ID" ]] || { echo "No permutation ID at line $LINE" >&2; exit 2; }
  RANGE_ARGS=(--perm-ids "$PERM_ID")
else
  START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
  END=$((START + CHUNK_SIZE - 1))
  (( END > N_PERM )) && END=$N_PERM
  (( START <= N_PERM )) || exit 0
  RANGE_ARGS=(--perm-start "$START" --perm-end "$END")
fi

echo "[$(date '+%F %T')] array=${SLURM_ARRAY_TASK_ID} args=${RANGE_ARGS[*]}"
"$RSCRIPT" permutation_slurm/permutation_worker.R \
  "${RANGE_ARGS[@]}" \
  --project-dir "$BIOLOGY_ROOT" \
  --out-dir "$PERM_OUTPUT_ROOT" \
  --rscript "$RSCRIPT" \
  --cohorts ECAM,DIABIMMUNE \
  --methods "$PERM_METHODS" \
  --seed "$PERM_SEED" \
  --cores 1 \
  --compact true \
  --resume true
