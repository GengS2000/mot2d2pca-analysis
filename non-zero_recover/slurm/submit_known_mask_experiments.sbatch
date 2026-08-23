#!/usr/bin/env bash
#SBATCH --job-name=known_mask_exp
#SBATCH --array=1-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=known_mask_exp_%A_%a.out
#SBATCH --error=known_mask_exp_%A_%a.err

set -euo pipefail

# Configure these with environment variables at submission time if needed:
#   OUTDIR=outputs_known_mask REPEATS=10 sbatch slurm/submit_known_mask_experiments.sbatch
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
TOP_VARIABLE_TAXA="${TOP_VARIABLE_TAXA:-40}"

echo "[slurm] job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
echo "[slurm] running experiment ${SLURM_ARRAY_TASK_ID} into ${OUTDIR}"

"${PYTHON_BIN}" scripts/run_known_mask_experiments.py \
  --experiments "${SLURM_ARRAY_TASK_ID}" \
  --max-workers 1 \
  --python "${PYTHON_BIN}" \
  --tax-path "${TAX_PATH}" \
  --meta-path "${META_PATH}" \
  --outdir "${OUTDIR}" \
  --repeats "${REPEATS}" \
  --fracs "${FRACS}" \
  --mechanisms "${MECHANISMS}" \
  --methods "${METHODS}" \
  --rank-d "${RANK_D}" \
  --rank-q "${RANK_Q}" \
  --lam "${LAM}" \
  --sigma "${SIGMA}" \
  --T0 "${T0}" \
  --T "${T}" \
  --top-variable-taxa "${TOP_VARIABLE_TAXA}" \
  --overwrite

echo "[slurm] experiment ${SLURM_ARRAY_TASK_ID} complete"
