#!/bin/bash
#SBATCH --job-name=sim123_merge
#SBATCH --output=server_runs/logs/%x_%j.out
#SBATCH --error=server_runs/logs/%x_%j.err
#SBATCH --partition=hebhcnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
# This partition grants roughly 2 GiB per allocated CPU. The merge itself is
# mostly single-process, but 16 allocated CPUs are required to obtain 32 GiB.
#SBATCH --cpus-per-task=16
# The abundance-space merge loads several large CSV families into pandas at
# once. Historical merged Sim1 CSVs alone exceed 1 GiB on disk, so 3 GiB is
# prone to Slurm OOM kills after pandas object/string expansion.
#SBATCH --mem-per-cpu=2G

set -eo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
: "${RUN_ROOT:?Export RUN_ROOT}"

source ~/miniconda3/bin/activate
conda activate q2-202402

# Check the exact expected (fraction, repeat) set, basic readability, Sim3
# completion markers, and the imputation parameters recorded in the manifest.
python validate_sim123_run.py "$RUN_ROOT"
echo "[manifest] $RUN_ROOT/manifest.json"
cat "$RUN_ROOT/manifest.json"

set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Merge commands request only a built-in method so dependency checks do not
# rerun Gemelli.  They only read completed per-shard files.
python sim1_compare/sim1_methodcompare.py \
  --outdir "$RUN_ROOT/sim1" --merge-only --methods mean
python sim2-compare/ctf_reconstruct_compare.py \
  --outdir "$RUN_ROOT/sim2" --merge_only --methods mean
python sim3_compare/sim3_compare.py \
  --outdir "$RUN_ROOT/sim3" --merge_only --methods mean

echo "[done] merged Sim1, Sim2, and Sim3 under $RUN_ROOT"
