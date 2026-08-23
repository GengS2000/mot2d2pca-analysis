#!/usr/bin/env bash
#SBATCH --job-name=exp3_merge
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=750M
#SBATCH --time=08:00:00
#SBATCH --output=exp3_merge_%j.out
#SBATCH --error=exp3_merge_%j.err

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTDIR="${OUTDIR:-outputs_known_mask}"
WRITE_COMBINED_LONG="${WRITE_COMBINED_LONG:-0}"

merge_args=(
  scripts/merge_experiment3_trajectory_chunks.py
  --expdir "${OUTDIR}/experiment3_trajectory"
  --clean-existing
)

if [[ "${WRITE_COMBINED_LONG}" == "1" ]]; then
  merge_args+=(--write-combined-long)
fi

"${PYTHON_BIN}" "${merge_args[@]}"

"${PYTHON_BIN}" scripts/analyze_repeat_level_inference.py \
  --root "${OUTDIR}" \
  --outdir "${OUTDIR}/analysis" \
  --replacement-mode known_mask \
  --baseline no_imputation

echo "[slurm-exp3] merge and analysis complete"
