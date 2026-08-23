#!/bin/bash
set -euo pipefail

: "${SIM1_TRUTH_PATH:?Export SIM1_TRUTH_PATH to the selected frozen Sim1 truth}"
: "${SIM2_TRUTH_PATH:?Export SIM2_TRUTH_PATH to the existing frozen Sim2 CP-ALS truth}"
: "${SIM3_TRUTH_PATH:?Export SIM3_TRUTH_PATH to the existing frozen Sim3 effect truth}"

for truth_path in "$SIM1_TRUTH_PATH" "$SIM2_TRUTH_PATH" "$SIM3_TRUTH_PATH"; do
  if [[ ! -s "$truth_path" ]]; then
    echo "Required frozen truth is missing or empty: $truth_path" >&2
    echo "Build Sim1 with build_sim1_truth_q2_202402.slurm; upload and reuse the existing Sim2/Sim3 truths." >&2
    exit 1
  fi
done

RUN_ROOT="${RUN_ROOT:-server_runs/sim123_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT
RUN_LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$RUN_ROOT"

# Imputation-stage MOT-(2D)^2PCA parameters forwarded to the array shards.
# Defaults are the tuned best config (d=1, q=1, lam=2, sigma=16); override by
# exporting IMPUTE_D/IMPUTE_Q/IMPUTE_LAM/IMPUTE_SIGMA/IMPUTE_T0/IMPUTE_T/IMPUTE_EPS.
IMPUTE_D="${IMPUTE_D:-1}"
IMPUTE_Q="${IMPUTE_Q:-1}"
IMPUTE_LAM="${IMPUTE_LAM:-2}"
IMPUTE_SIGMA="${IMPUTE_SIGMA:-16}"
IMPUTE_T0="${IMPUTE_T0:-30}"
IMPUTE_T="${IMPUTE_T:-30}"
IMPUTE_EPS="${IMPUTE_EPS:-1e-3}"

# Never silently mix experiments: refuse to write into a RUN_ROOT that already
# contains files unless ALLOW_RUN_ROOT_REUSE=1 is set explicitly.
if [[ -n "$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  if [[ "${ALLOW_RUN_ROOT_REUSE:-0}" != "1" ]]; then
    echo "Refusing to reuse non-empty RUN_ROOT: $RUN_ROOT" >&2
    echo "It already contains results from a previous submission (run \"ls $RUN_ROOT\" to confirm)." >&2
    echo "Export ALLOW_RUN_ROOT_REUSE=1 to intentionally overwrite/rerun into it." >&2
    exit 1
  fi
  echo "[warn] ALLOW_RUN_ROOT_REUSE=1: reusing non-empty RUN_ROOT ${RUN_ROOT} (existing files may be overwritten)"
fi
mkdir -p "$RUN_LOG_DIR"

# Record what this submission is, so a directory can be identified later.
cat > "$RUN_ROOT/manifest.json" <<EOF
{
  "submit_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "submit_script": "$(basename "$0")",
  "run_root": "$RUN_ROOT",
  "sim1_truth": "$SIM1_TRUTH_PATH",
  "sim2_truth": "$SIM2_TRUTH_PATH",
  "sim3_truth": "$SIM3_TRUTH_PATH",
  "impute_params": {
    "d": "${IMPUTE_D}", "q": "${IMPUTE_Q}", "lam": "${IMPUTE_LAM}",
    "sigma": "${IMPUTE_SIGMA}", "T0": "${IMPUTE_T0}", "T": "${IMPUTE_T}",
    "eps": "${IMPUTE_EPS}"
  },
  "array_spec": "${ARRAY_SPEC:-0-749%200}",
  "skip_merge": "${SKIP_MERGE:-0}"
}
EOF

common_export="ALL,SIM1_TRUTH_PATH=${SIM1_TRUTH_PATH},SIM2_TRUTH_PATH=${SIM2_TRUTH_PATH},SIM3_TRUTH_PATH=${SIM3_TRUTH_PATH},RUN_ROOT=${RUN_ROOT},IMPUTE_D=${IMPUTE_D},IMPUTE_Q=${IMPUTE_Q},IMPUTE_LAM=${IMPUTE_LAM},IMPUTE_SIGMA=${IMPUTE_SIGMA},IMPUTE_T0=${IMPUTE_T0},IMPUTE_T=${IMPUTE_T},IMPUTE_EPS=${IMPUTE_EPS}"

preflight_job="$(sbatch --parsable \
  --output="$RUN_LOG_DIR/%x_%j.out" --error="$RUN_LOG_DIR/%x_%j.err" \
  --export="$common_export" preflight_sim123_q2_202402.slurm)"
preflight_job="${preflight_job%%;*}"
array_spec="${ARRAY_SPEC:-0-749%200}"
array_job="$(sbatch --parsable --dependency="afterok:${preflight_job}" \
  --output="$RUN_LOG_DIR/%x_%A_%a.out" --error="$RUN_LOG_DIR/%x_%A_%a.err" \
  --array="$array_spec" --export="$common_export" run_sim1_sim2_sim3_q2_202402.slurm)"
array_job="${array_job%%;*}"

echo "preflight job: ${preflight_job}"
echo "Sim1+Sim2+Sim3 array: ${array_job} (array=${array_spec})"
if [[ "${SKIP_MERGE:-0}" == "1" ]]; then
  echo "merge job: skipped by SKIP_MERGE=1"
else
  merge_job="$(sbatch --parsable --dependency="afterok:${array_job}" \
    --output="$RUN_LOG_DIR/%x_%j.out" --error="$RUN_LOG_DIR/%x_%j.err" \
    --export="$common_export" merge_sim1_sim2_sim3_q2_202402.slurm)"
  merge_job="${merge_job%%;*}"
  echo "merge job: ${merge_job} (afterok:${array_job})"
fi
echo "result root: ${RUN_ROOT}"
echo "manifest: ${RUN_ROOT}/manifest.json"
