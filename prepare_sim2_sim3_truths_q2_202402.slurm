#!/bin/bash
#SBATCH --job-name=sim23_truth
#SBATCH --output=server_runs/logs/%x_%j.out
#SBATCH --error=server_runs/logs/%x_%j.err
#SBATCH --partition=hebhcnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G

set -eo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p server_runs/logs

: "${SIM2_TRUTH_PATH:?Export SIM2_TRUTH_PATH}"
: "${SIM3_TRUTH_PATH:?Export SIM3_TRUTH_PATH}"
[[ "$SIM2_TRUTH_PATH" != "$SIM3_TRUTH_PATH" ]] || {
  echo "SIM2_TRUTH_PATH and SIM3_TRUTH_PATH must be different files" >&2
  exit 1
}

source ~/miniconda3/bin/activate
conda activate q2-202402
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$(dirname "$SIM2_TRUTH_PATH")" "$(dirname "$SIM3_TRUTH_PATH")"

if [[ ! -f "$SIM2_TRUTH_PATH" || "${REBUILD_SIM2_TRUTH:-0}" == "1" ]]; then
  python sim2-compare/ctf_reconstruct_compare.py \
    --tax_path sim2-compare/hmp2_mgx_taxonomy.tsv \
    --meta_path sim2-compare/metagenomics.xlsx \
    --outdir "$(dirname "$SIM2_TRUTH_PATH")" \
    --truth_path "$SIM2_TRUTH_PATH" \
    --build_truth_only \
    --force_rebuild_truth \
    --methods mean \
    --eps 1e-3 \
    --ctf_rank 3 \
    --ctf_method fallback \
    --ctf_seed 0 \
    --quiet
else
  echo "[truth] reuse frozen Sim2 CP-ALS truth: $SIM2_TRUTH_PATH"
fi

if [[ ! -f "$SIM3_TRUTH_PATH" || "${REBUILD_SIM3_TRUTH:-0}" == "1" ]]; then
  python sim3_compare/sim3_compare.py \
    --tax-path sim3_compare/hmp2_mgx_taxonomy.tsv \
    --meta-path sim3_compare/metagenomics.xlsx \
    --outdir "$(dirname "$SIM3_TRUTH_PATH")" \
    --base-truth-path "$SIM2_TRUTH_PATH" \
    --truth-path "$SIM3_TRUTH_PATH" \
    --build_truth_only \
    --force_rebuild_truth \
    --methods mean \
    --eps 1e-3 \
    --ctf_rank 3 \
    --ctf_method fallback \
    --ctf_seed 0 \
    --enable-effect \
    --effect_seed 2026 \
    --n_cd 65 \
    --n_noncd 65 \
    --diff_frac 0.10 \
    --group_effect_mean 0.8 \
    --group_effect_sd 0.2 \
    --interaction_mean 0.15 \
    --interaction_sd 0.05 \
    --random_slope_sd 0.05 \
    --noise_sd 0.10 \
    --quiet
else
  echo "[truth] reuse frozen Sim3 legacy-effect truth: $SIM3_TRUTH_PATH"
fi

test -f "$SIM2_TRUTH_PATH"
test -f "$SIM3_TRUTH_PATH"
echo "[done] Sim2 and Sim3 truths are frozen and ready"
