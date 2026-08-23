#!/bin/bash
#SBATCH --job-name=sim3_compare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=45G
#SBATCH --partition=hebhctest

set -euo pipefail
mkdir -p logs

source ~/miniconda3/bin/activate
conda activate micro3dimp

which python
python --version
hostname

echo "[START] $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"
echo "N_WORKERS=16"
echo "ENABLE_EFFECT=1"

# Each repeat is a process. Keep numerical libraries single-threaded to avoid
# N_WORKERS x BLAS-thread oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python sim3_compare.py \
  --tax-path hmp2_mgx_taxonomy.tsv \
  --meta-path metagenomics.xlsx \
  --outdir results_sim3_compare_effect_parallel \
  --n-workers 16 \
  --n-repeats 50 \
  --fracs 0.5,0.6,0.7,0.8,0.9 \
  --mask-seed 20260322 \
  --algo-seed 20260322 \
  --methods \
    penalized_2d2_pca \
    mean \
    median \
    last \
    equal \
    weighted_time_average \
    temporal_knn \
    weighted_avg_legacy \
    knn_epanechnikov_legacy \
    spline_legacy \
    glv_mse_legacy \
    glv_ridge_legacy \
  --eps 1e-3 \
  --d 1 \
  --q 1 \
  --lam 0.1 \
  --sigma 4.0 \
  --T0 30 \
  --T 30 \
  --temporal-knn-k 3 \
  --temporal-knn-bandwidth 4.0 \
  --legacy-knn-k 3 \
  --spline-min-points 4 \
  --equal-value 0.0 \
  --dbn-dense-alpha 1e-3 \
  --dbn-sparse-alpha 1e-2 \
  --glv-alpha 1e-3 \
  --glv-ridge-lambda-M 1e-3 \
  --glv-ridge-lambda-mue 1e-3 \
  --glv-ridge-lambda-E 1e-3 \
  --limits-error-threshold 0.05 \
  --limits-repeats 20 \
  --limits-med-or-avg 0 \
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
  --verbose

echo "[END] $(date)"
