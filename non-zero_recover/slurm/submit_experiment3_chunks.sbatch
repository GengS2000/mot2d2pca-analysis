#!/usr/bin/env bash
#SBATCH --job-name=exp3_chunk
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=750M
#SBATCH --time=06:00:00
#SBATCH --output=exp3_chunk_%A_%a.out
#SBATCH --error=exp3_chunk_%A_%a.err

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTDIR="${OUTDIR:-outputs_known_mask}"
TAX_PATH="${TAX_PATH:-hmp2_mgx_taxonomy.tsv}"
META_PATH="${META_PATH:-metagenomics.xlsx}"
REPEATS="${REPEATS:-10}"
FRACS="${FRACS:-0.05,0.10,0.20,0.30}"
MECHANISMS="${MECHANISMS:-random_entry,low_depth_sample,time_point,subject_specific_dropout,consecutive_time_window_dropout,low_abundance_biased}"
METHODS="${METHODS:-penalized_2d2_pca,no_imputation}"
RANK_D="${RANK_D:-10}"
RANK_Q="${RANK_Q:-10}"
LAM="${LAM:-0.7}"
SIGMA="${SIGMA:-8.0}"
T0="${T0:-30}"
T="${T:-30}"
DTW_TRAJECTORIES="${DTW_TRAJECTORIES:-25}"

echo "[slurm-exp3] job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"

"${PYTHON_BIN}" scripts/run_experiment3_trajectory_chunk.py \
  --chunk-index "${SLURM_ARRAY_TASK_ID}" \
  --chunk-base 1 \
  --tax-path "${TAX_PATH}" \
  --meta-path "${META_PATH}" \
  --outdir "${OUTDIR}" \
  --repeats "${REPEATS}" \
  --fracs "${FRACS}" \
  --mechanisms "${MECHANISMS}" \
  --methods "${METHODS}" \
  --replacement-modes known_mask \
  --rank-d "${RANK_D}" \
  --rank-q "${RANK_Q}" \
  --lam "${LAM}" \
  --sigma "${SIGMA}" \
  --T0 "${T0}" \
  --T "${T}" \
  --dtw-trajectories "${DTW_TRAJECTORIES}" \
  --skip-preprocessing-diagnostics \
  --overwrite

echo "[slurm-exp3] chunk ${SLURM_ARRAY_TASK_ID} complete"
