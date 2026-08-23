#!/bin/bash
#SBATCH --job-name=truth_tune
#SBATCH --output=server_runs/logs/%x_%A_%a.out
#SBATCH --error=server_runs/logs/%x_%A_%a.err
#SBATCH --array=0-23%24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=hebhctest

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p server_runs/logs server_runs/truth_sweep

source ~/miniconda3/bin/activate
conda activate q2-2024

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

task_id="${SLURM_ARRAY_TASK_ID}"
d_values=(5 8 16 32)
q_values=(3 5 10 20)
lam_values=(0 0.005 0.005 0.015 0.015 0.03)
sigma_values=(4 2 4 2 4 4)
capacity_idx=$((task_id / 6))
variant_idx=$((task_id % 6))
d="${d_values[$capacity_idx]}"
q="${q_values[$capacity_idx]}"
lam="${lam_values[$variant_idx]}"
sigma="${sigma_values[$variant_idx]}"
candidate="d${d}_q${q}_lam${lam}_sigma${sigma}"
outdir="server_runs/truth_sweep/sim1/${candidate}"
truth="${outdir}/truth_tensor_sim1.nc"
mkdir -p "$outdir"
python sim1_compare/sim1_methodcompare.py \
  --tax-path sim1_compare/hmp2_mgx_taxonomy.tsv \
  --meta-path sim1_compare/metagenomics.xlsx \
  --outdir "$outdir" \
  --truth-path "$truth" \
  --build-truth-only --force-rebuild-truth \
  --methods mean \
  --d "$d" --q "$q" --lam "$lam" --sigma "$sigma" \
  --t0 30 --t 30 --eps 1e-3 --truth-observed-mode positive_only
python truth_diagnostics.py \
  --truth "$truth" --output "${outdir}/truth_diagnostics.csv" \
  --simulation sim1 --candidate "$candidate" --eps 1e-3 \
  --d "$d" --q "$q" --lam "$lam" --sigma "$sigma"
