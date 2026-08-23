#!/bin/bash
#SBATCH --job-name=ctf_compare
#SBATCH --partition=hebhctest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=45G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs artifacts/method_compare

source ~/miniconda3/bin/activate
conda activate micro3dimp

which python
python --version
hostname

echo "[START] $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python sim3_compare.py \
  --tax_path hmp2_mgx_taxonomy.tsv \
  --meta_path metagenomics.xlsx \
  --outdir artifacts/sim3_effect \
  --fracs 0.5,0.7,0.8,0.9 \
  --n_repeats 4 \
  --mask_seed 123 \
  --algo_seed 0 \
  --eps 1e-3 \
  --min_pos_cells 10 \
  --min_participants 5 \
  --methods penalized_2d2_pca,mean,median,last,weighted_time_average,temporal_knn,spline,dbn_dense,dbn_sparse,glv,equal \
  --d 1 \
  --q 1 \
  --lam 0.1 \
  --sigma 4.0 \
  --T0 30 \
  --T 30 \
  --ctf_rank 3 \
  --ctf_method fallback \
  --ctf_seed 0 \
  --enable_effect \
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
  --force_rebuild_truth \
  --temporal_knn_k 3 \
  --temporal_knn_sigma 4.0 \
  --spline_min_points 4 \
  --equal_fill_value 0.0 \
  --dbn_dense_alpha 1.0 \
  --dbn_sparse_alpha 0.05 \
  --dbn_sparse_n_iter 150 \
  --glv_alpha 0.5

echo "[END] $(date)"
