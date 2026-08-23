#!/bin/bash
#SBATCH --job-name=sim1_only
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
conda activate q2-2024

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

python sim1_methodcompare.py \
  --tax-path hmp2_mgx_taxonomy.tsv \
  --meta-path metagenomics.xlsx \
  --outdir results_sim1_compare_pilot \
  --n-repeats 4 \
  --fracs 0.5 0.6 0.7 0.8 0.9 \
  --seed-base 20260322 \
  --truth-observed-mode positive_only \
  --second-observed-mode positive_only_except_replace \
  --mask-eligible-mode positive_only \
  --methods penalized_2d2_pca ctf mean nearest_positive temporal_knn \
  --eps 1e-3 \
  --d 5 \
  --q 3 \
  --lam 0.03 \
  --sigma 4.0 \
  --t0 30 \
  --t 30 \
  --temporal-knn-k 3 \
  --temporal-knn-bandwidth 4.0 \
  --verbose
echo "[END] $(date)"
