#!/bin/bash
#SBATCH --job-name=sim1_truth
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

: "${SIM1_TRUTH_PATH:?Export SIM1_TRUTH_PATH to the new frozen Sim1 truth file}"

SIM1_D="${SIM1_D:-5}"
SIM1_Q="${SIM1_Q:-3}"
SIM1_LAM="${SIM1_LAM:-0.03}"
SIM1_SIGMA="${SIM1_SIGMA:-4}"
SIM1_T0="${SIM1_T0:-30}"
SIM1_T="${SIM1_T:-30}"
SIM1_SEED_BASE="${SIM1_SEED_BASE:-20260322}"

source ~/miniconda3/bin/activate
conda activate q2-202402
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$(dirname "$SIM1_TRUTH_PATH")"

echo "[Sim1 truth] path=$SIM1_TRUTH_PATH"
echo "[Sim1 truth] d=$SIM1_D q=$SIM1_Q lam=$SIM1_LAM sigma=$SIM1_SIGMA T0=$SIM1_T0 T=$SIM1_T"
echo "[Sim1 truth] mode=hybrid (raw observed cells kept, MOT-(2D)^2PCA fills unobserved)"

python sim1_compare/sim1_methodcompare.py \
  --tax-path sim1_compare/hmp2_mgx_taxonomy.tsv \
  --meta-path sim1_compare/metagenomics.xlsx \
  --outdir "$(dirname "$SIM1_TRUTH_PATH")" \
  --truth-path "$SIM1_TRUTH_PATH" \
  --build-truth-only \
  --force-rebuild-truth \
  --truth-hybrid \
  --methods mean \
  --eps 1e-3 \
  --d "$SIM1_D" \
  --q "$SIM1_Q" \
  --lam "$SIM1_LAM" \
  --sigma "$SIM1_SIGMA" \
  --t0 "$SIM1_T0" \
  --t "$SIM1_T" \
  --seed-base "$SIM1_SEED_BASE"

test -s "$SIM1_TRUTH_PATH"
echo "[done] frozen Sim1 truth: $SIM1_TRUTH_PATH"
echo "[check] diagnostics: $(dirname "$SIM1_TRUTH_PATH")/truth_diagnostics_sim1.csv"
