#!/bin/bash
#SBATCH --job-name=sim123_full
#SBATCH --output=server_runs/logs/%x_%A_%a.out
#SBATCH --error=server_runs/logs/%x_%A_%a.err
#SBATCH --array=0-749%200
#SBATCH --partition=hebhcnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G

set -eo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"

: "${SIM1_TRUTH_PATH:?Export SIM1_TRUTH_PATH}"
: "${SIM2_TRUTH_PATH:?Export SIM2_TRUTH_PATH}"
: "${SIM3_TRUTH_PATH:?Export SIM3_TRUTH_PATH}"
: "${RUN_ROOT:?Export RUN_ROOT}"
test -f "$SIM1_TRUTH_PATH"
test -f "$SIM2_TRUTH_PATH"
test -f "$SIM3_TRUTH_PATH"
mkdir -p server_runs/logs "$RUN_ROOT/sim1" "$RUN_ROOT/sim2" "$RUN_ROOT/sim3"

# Imputation-stage MOT-(2D)^2PCA parameters.  These apply to the penalized_2d2_pca
# method inside the comparison shards only; the frozen truths are untouched.
# Defaults are the tuned best config (d=1, q=1, lam=2, sigma=16).  Override by
# exporting IMPUTE_D/IMPUTE_Q/IMPUTE_LAM/IMPUTE_SIGMA/IMPUTE_T0/IMPUTE_T/IMPUTE_EPS.
IMPUTE_D="${IMPUTE_D:-1}"
IMPUTE_Q="${IMPUTE_Q:-1}"
IMPUTE_LAM="${IMPUTE_LAM:-2}"
IMPUTE_SIGMA="${IMPUTE_SIGMA:-16}"
IMPUTE_T0="${IMPUTE_T0:-30}"
IMPUTE_T="${IMPUTE_T:-30}"
IMPUTE_EPS="${IMPUTE_EPS:-1e-3}"

source ~/miniconda3/bin/activate
conda activate q2-202402
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

fracs=(0.5 0.6 0.7 0.8 0.9)
frac_seed_offsets=(500000 600000 700000 800000 900000)
task_id="${SLURM_ARRAY_TASK_ID}"
simulation_id="$((task_id % 3))"
local_id="$((task_id / 3))"
frac_idx="$((local_id / 50))"
repeat="$((local_id % 50))"
frac="${fracs[$frac_idx]}"
mask_seed="$((20260322 + frac_seed_offsets[$frac_idx] + repeat))"
algo_seed="$((mask_seed + 10000))"

echo "[shard] run_root=$RUN_ROOT simulation_id=$simulation_id frac=$frac repeat=$repeat mask_seed=$mask_seed algo_seed=$algo_seed"
echo "[shard] impute params: d=$IMPUTE_D q=$IMPUTE_Q lam=$IMPUTE_LAM sigma=$IMPUTE_SIGMA T0=$IMPUTE_T0 T=$IMPUTE_T eps=$IMPUTE_EPS"

if (( simulation_id == 0 )); then
  python sim1_compare/sim1_methodcompare.py \
    --tax-path sim1_compare/hmp2_mgx_taxonomy.tsv \
    --meta-path sim1_compare/metagenomics.xlsx \
    --outdir "$RUN_ROOT/sim1" \
    --truth-path "$SIM1_TRUTH_PATH" \
    --require-existing-truth \
    --frac "$frac" --repeat "$repeat" --seed-base 20260322 \
    --methods penalized_2d2_pca ctf mean nearest_positive temporal_knn \
    --eps "$IMPUTE_EPS" --d "$IMPUTE_D" --q "$IMPUTE_Q" --lam "$IMPUTE_LAM" --sigma "$IMPUTE_SIGMA" \
    --t0 "$IMPUTE_T0" --t "$IMPUTE_T" \
    --competitor-ctf-rank 3 \
    --competitor-ctf-initializations 10 \
    --competitor-ctf-max-iterations 50 \
    --temporal-knn-k 3 --temporal-knn-bandwidth 4 \
    --compact-output
elif (( simulation_id == 1 )); then
  python sim2-compare/ctf_reconstruct_compare.py \
    --tax_path sim2-compare/hmp2_mgx_taxonomy.tsv \
    --meta_path sim2-compare/metagenomics.xlsx \
    --outdir "$RUN_ROOT/sim2" \
    --truth_path "$SIM2_TRUTH_PATH" \
    --require-existing-truth \
    --frac "$frac" --repeat "$repeat" \
    --mask_seed "$mask_seed" --algo_seed "$algo_seed" \
    --methods penalized_2d2_pca ctf mean nearest_positive temporal_knn \
    --eps "$IMPUTE_EPS" --defer_merge --compact-output \
    --d "$IMPUTE_D" --q "$IMPUTE_Q" --lam "$IMPUTE_LAM" --sigma "$IMPUTE_SIGMA" --T0 "$IMPUTE_T0" --T "$IMPUTE_T" \
    --competitor-ctf-rank 3 \
    --competitor-ctf-initializations 10 \
    --competitor-ctf-max-iterations 50 \
    --temporal-knn-k 3 --temporal-knn-sigma 4 --quiet
else
  python sim3_compare/sim3_compare.py \
    --tax-path sim3_compare/hmp2_mgx_taxonomy.tsv \
    --meta-path sim3_compare/metagenomics.xlsx \
    --outdir "$RUN_ROOT/sim3" \
    --truth-path "$SIM3_TRUTH_PATH" \
    --require-existing-truth --enable-effect \
    --frac "$frac" --repeat "$repeat" \
    --mask-seed "$mask_seed" --algo-seed "$algo_seed" \
    --methods penalized_2d2_pca ctf mean nearest_positive temporal_knn \
    --eps "$IMPUTE_EPS" --defer-merge --compact-output \
    --d "$IMPUTE_D" --q "$IMPUTE_Q" --lam "$IMPUTE_LAM" --sigma "$IMPUTE_SIGMA" --T0 "$IMPUTE_T0" --T "$IMPUTE_T" \
    --competitor-ctf-rank 3 \
    --competitor-ctf-initializations 10 \
    --competitor-ctf-max-iterations 50 \
    --temporal-knn-k 3 --temporal-knn-sigma 4 --quiet
fi
