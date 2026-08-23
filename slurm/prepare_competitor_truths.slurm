#!/bin/bash
#SBATCH --job-name=prepare_comp_truth
#SBATCH --partition=hebhctest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
BENCHMARK_ENV="${BENCHMARK_ENV:-micro3dimp}"
TAX_PATH="${TAX_PATH:-sim2-compare/hmp2_mgx_taxonomy.tsv}"
META_PATH="${META_PATH:-sim2-compare/metagenomics.xlsx}"
cd "${PROJECT_DIR}"
mkdir -p logs artifacts/competitor_truths

if [[ "${CONDA_DEFAULT_ENV:-}" != "${BENCHMARK_ENV}" ]]; then
  # QIIME2's libxml2 deactivate hook reads an unset variable.  Conda hooks are
  # therefore run with nounset temporarily disabled.
  set +u
  source "${HOME}/miniconda3/bin/activate"
  conda activate "${BENCHMARK_ENV}"
  set -u
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ "${SKIP_TRUTH_BUILD:-0}" == "1" ]]; then
  for SIM in sim1 sim2 sim3; do
    TRUTH_PATH="artifacts/competitor_truths/${SIM}.nc"
    if [[ ! -f "${TRUTH_PATH}" ]]; then
      echo "Requested truth reuse, but file is missing: ${TRUTH_PATH}" >&2
      exit 2
    fi
  done
  echo "Reusing existing sim1/sim2/sim3 truth tensors."
  exit 0
fi

python sim1_compare/sim1_methodcompare.py \
  --tax-path "${TAX_PATH}" \
  --meta-path "${META_PATH}" \
  --outdir artifacts/competitor_truths \
  --truth-path artifacts/competitor_truths/sim1.nc \
  --build-truth-only \
  --methods mean \
  --seed-base 20260322 \
  --truth-observed-mode positive_only \
  --eps 1e-3 --d 1 --q 1 --lam 0.1 --sigma 4.0 --t0 30 --t 30

python sim2-compare/ctf_reconstruct_compare.py \
  --tax_path "${TAX_PATH}" \
  --meta_path "${META_PATH}" \
  --outdir artifacts/competitor_truths/sim2_build \
  --truth_path artifacts/competitor_truths/sim2.nc \
  --build_truth_only \
  --methods mean \
  --eps 1e-3 --ctf_rank 3 --ctf_method fallback --ctf_seed 0

python sim3_compare/sim3_compare.py \
  --tax_path "${TAX_PATH}" \
  --meta_path "${META_PATH}" \
  --outdir artifacts/competitor_truths/sim3_build \
  --truth_path artifacts/competitor_truths/sim3.nc \
  --build_truth_only \
  --methods mean \
  --eps 1e-3 --ctf_rank 3 --ctf_method fallback --ctf_seed 0 \
  --enable_effect --effect_seed 2026 --n_cd 65 --n_noncd 65 \
  --diff_frac 0.10 --group_effect_mean 0.8 --group_effect_sd 0.2 \
  --interaction_mean 0.15 --interaction_sd 0.05 \
  --random_slope_sd 0.05 --noise_sd 0.10
