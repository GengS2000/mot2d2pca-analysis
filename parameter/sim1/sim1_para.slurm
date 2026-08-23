#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --job-name=coarse_lam_sigma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=45G
#SBATCH --partition=hebhctest

set -euo pipefail

# Create the log directory
mkdir -p logs

# ====== Environment: adjust to the local cluster ======
source ~/miniconda3/bin/activate
conda activate micro3dimp

python preprocess_only_positive.py \
  --tax-path hmp2_mgx_taxonomy.tsv \
  --meta-path metagenomics.xlsx \
  --outdir results_sim1_coarse_stage1 \
  --n-repeats 4 \
  --fracs 0.8 0.9 \
  --seed-base 20260331 \
  --truth-observed-mode all_finite_except_replace \
  --second-observed-mode all_finite_except_replace \
  --mask-eligible-mode all_finite \
  --eps 1e-3 \
  --truth-d 2 \
  --truth-q 2 \
  --truth-lam 0.1 \
  --truth-sigma 4.0 \
  --grid-d 1 \
  --grid-q 1 \
  --grid-lam 0.5 0.1 0.2 0.5 \
  --grid-sigma 1 2 3 4 \
  --t0 30 \
  --t 30 \
  --verbose