#!/bin/bash
#SBATCH --job-name=sim12_full
#SBATCH --output=server_runs/logs/%x_%A_%a.out
#SBATCH --error=server_runs/logs/%x_%A_%a.err
#SBATCH --array=0-499%180
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=hebhctest

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p server_runs/logs server_runs/final/sim1 server_runs/final/sim2

: "${SIM1_TRUTH_PATH:?Export SIM1_TRUTH_PATH to the selected frozen Sim1 truth}"
: "${SIM2_TRUTH_PATH:?Export SIM2_TRUTH_PATH to the selected frozen Sim2 truth}"

source ~/miniconda3/bin/activate
conda activate q2-2024
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

fracs=(0.5 0.6 0.7 0.8 0.9)
frac_seed_offsets=(500000 600000 700000 800000 900000)
task_id="${SLURM_ARRAY_TASK_ID}"
if (( task_id % 2 == 0 )); then
  local_id="$((task_id / 2))"
  frac="${fracs[$((local_id / 50))]}"
  repeat="$((local_id % 50))"
  python sim1_compare/sim1_methodcompare.py \
    --tax-path sim1_compare/hmp2_mgx_taxonomy.tsv \
    --meta-path sim1_compare/metagenomics.xlsx \
    --outdir server_runs/final/sim1 \
    --truth-path "$SIM1_TRUTH_PATH" \
    --frac "$frac" --repeat "$repeat" --seed-base 20260322 \
    --methods penalized_2d2_pca ctf mean nearest_positive temporal_knn \
    --eps 1e-3 \
    --d "${SIM1_D:-5}" --q "${SIM1_Q:-3}" \
    --lam "${SIM1_LAM:-0.03}" --sigma "${SIM1_SIGMA:-4}" \
    --t0 30 --t 30 --temporal-knn-k 3 --temporal-knn-bandwidth 4
else
  local_id="$((task_id / 2))"
  frac_idx="$((local_id / 50))"
  frac="${fracs[$frac_idx]}"
  repeat="$((local_id % 50))"
  mask_seed="$((20260322 + frac_seed_offsets[$frac_idx] + repeat))"
  algo_seed="$((mask_seed + 10000))"
  python sim2-compare/ctf_reconstruct_compare.py \
    --tax_path sim2-compare/hmp2_mgx_taxonomy.tsv \
    --meta_path sim2-compare/metagenomics.xlsx \
    --outdir server_runs/final/sim2 \
    --truth_path "$SIM2_TRUTH_PATH" \
    --frac "$frac" --repeat "$repeat" --mask_seed "$mask_seed" --algo_seed "$algo_seed" \
    --methods penalized_2d2_pca ctf mean nearest_positive temporal_knn \
    --eps 1e-3 --defer_merge \
    --d "${SIM2_D:-5}" --q "${SIM2_Q:-3}" \
    --lam "${SIM2_LAM:-0.03}" --sigma "${SIM2_SIGMA:-4}" \
    --T0 30 --T 30 --ctf_method gemelli \
    --temporal-knn-k 3 --temporal-knn-sigma 4 --quiet
fi
