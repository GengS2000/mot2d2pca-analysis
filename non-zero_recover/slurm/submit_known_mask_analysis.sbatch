#!/usr/bin/env bash
#SBATCH --job-name=known_mask_analysis
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=known_mask_analysis_%j.out
#SBATCH --error=known_mask_analysis_%j.err

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTDIR="${OUTDIR:-outputs_known_mask}"

echo "[slurm] analysis job=${SLURM_JOB_ID} host=$(hostname)"

"${PYTHON_BIN}" scripts/analyze_repeat_level_inference.py \
  --root "${OUTDIR}" \
  --outdir "${OUTDIR}/analysis" \
  --replacement-mode known_mask \
  --baseline no_imputation

echo "[slurm] analysis complete"
