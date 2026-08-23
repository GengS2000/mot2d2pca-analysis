#!/bin/bash
# Chunked submit driver for Sim1+Sim2+Sim3 (750 shards) under a group submit
# limit of 200 (AssocGrpSubmitJobsLimit).  The array `%180` throttle does NOT
# reduce the submit-time count, so a single 0-749 array is rejected.  This
# driver submits 4 sequential waves (188/188/188/186 tasks, <=200 submitted at
# any moment, 180 concurrent), waits for each wave to fully COMPLETE, and
# finally submits the merge job.
set -euo pipefail

: "${SIM1_TRUTH_PATH:?Export SIM1_TRUTH_PATH}"
: "${SIM2_TRUTH_PATH:?Export SIM2_TRUTH_PATH}"
: "${SIM3_TRUTH_PATH:?Export SIM3_TRUTH_PATH}"
: "${RUN_ROOT:?Export RUN_ROOT}"

for truth_path in "$SIM1_TRUTH_PATH" "$SIM2_TRUTH_PATH" "$SIM3_TRUTH_PATH"; do
  if [[ ! -s "$truth_path" ]]; then
    echo "Required frozen truth is missing or empty: $truth_path" >&2
    echo "Build Sim1 with build_sim1_truth_q2_202402.slurm; upload and reuse the existing Sim2/Sim3 truths." >&2
    exit 1
  fi
done

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
  "waves": "0-187%180,188-375%180,376-563%180,564-749%180",
  "skip_merge": "${SKIP_MERGE:-0}"
}
EOF

common_export="ALL,SIM1_TRUTH_PATH=${SIM1_TRUTH_PATH},SIM2_TRUTH_PATH=${SIM2_TRUTH_PATH},SIM3_TRUTH_PATH=${SIM3_TRUTH_PATH},RUN_ROOT=${RUN_ROOT},IMPUTE_D=${IMPUTE_D},IMPUTE_Q=${IMPUTE_Q},IMPUTE_LAM=${IMPUTE_LAM},IMPUTE_SIGMA=${IMPUTE_SIGMA},IMPUTE_T0=${IMPUTE_T0},IMPUTE_T=${IMPUTE_T},IMPUTE_EPS=${IMPUTE_EPS}"

wait_for_job_completed() {
  local job="$1" label="$2" expected_tasks="$3"
  local waited=0
  while squeue -j "$job" -h | grep -q .; do
    sleep 60
    waited=$((waited + 60))
    if (( waited % 600 == 0 )); then
      echo "[chunked] ${label} still running (${waited}s elapsed), queued now:"
      squeue -j "$job" -h -o "%.10i %.2t" | sort | uniq -c | sort -rn | head -3
    fi
  done
  local records="" record_count=0 accounting_wait=0
  while (( accounting_wait < 300 )); do
    records="$(sacct -j "$job" -n -P --format=JobIDRaw,State 2>/dev/null || true)"
    if (( expected_tasks == 1 )); then
      record_count="$(awk -F'|' -v j="$job" '$1 == j {n++} END {print n+0}' <<<"$records")"
    else
      record_count="$(awk -F'|' -v j="$job" '$1 ~ ("^" j "_[0-9]+$") {n++} END {print n+0}' <<<"$records")"
    fi
    (( record_count >= expected_tasks )) && break
    sleep 10
    accounting_wait=$((accounting_wait + 10))
  done
  if (( record_count < expected_tasks )); then
    echo "[chunked] ${label}: accounting has ${record_count}/${expected_tasks} expected records after 300s" >&2
    exit 1
  fi
  local bad
  bad="$(
    awk -F'|' -v j="$job" -v n="$expected_tasks" \
      '($1 == j && n == 1) || ($1 ~ ("^" j "_[0-9]+$") && n > 1) {
         if ($2 !~ /^COMPLETED/) print $1 "=" $2
       }' <<<"$records"
  )"
  if [[ -n "$bad" ]]; then
    echo "[chunked] ${label} has non-COMPLETED tasks:" >&2
    echo "$bad" >&2
    echo "[chunked] resubmit the failed task ids with the same RUN_ROOT, then run merge manually." >&2
    exit 1
  fi
  echo "[chunked] ${label} done (all COMPLETED)"
}

# 1) preflight (Gemelli API + three frozen truths + Sim2/Sim3 coordinate check)
preflight_job="$(sbatch --parsable \
  --output="$RUN_LOG_DIR/%x_%j.out" --error="$RUN_LOG_DIR/%x_%j.err" \
  --export="$common_export" preflight_sim123_q2_202402.slurm)"
echo "[chunked] preflight job: ${preflight_job}"
preflight_job="${preflight_job%%;*}"
wait_for_job_completed "$preflight_job" "preflight" 1

# 2) four sequential waves covering task ids 0-749
waves=("0-187%180" "188-375%180" "376-563%180" "564-749%180")
wave_sizes=(188 188 188 186)
for wave_idx in "${!waves[@]}"; do
  spec="${waves[$wave_idx]}"
  wave_job="$(sbatch --parsable \
    --output="$RUN_LOG_DIR/%x_%A_%a.out" --error="$RUN_LOG_DIR/%x_%A_%a.err" \
    --array="$spec" --export="$common_export" run_sim1_sim2_sim3_q2_202402.slurm)"
  wave_job="${wave_job%%;*}"
  echo "[chunked] wave job: ${wave_job} (array=${spec})"
  wait_for_job_completed "$wave_job" "wave ${spec%%%*}" "${wave_sizes[$wave_idx]}"
done

# 3) merge (validates all 250 shard sentinels per simulation before merging)
if [[ "${SKIP_MERGE:-0}" == "1" ]]; then
  echo "[chunked] merge job: skipped by SKIP_MERGE=1"
else
  merge_job="$(sbatch --parsable \
    --output="$RUN_LOG_DIR/%x_%j.out" --error="$RUN_LOG_DIR/%x_%j.err" \
    --export="$common_export" merge_sim1_sim2_sim3_q2_202402.slurm)"
  echo "[chunked] merge job: ${merge_job}"
fi

echo "[chunked] all waves submitted and completed. result root: ${RUN_ROOT}"
echo "[chunked] manifest: ${RUN_ROOT}/manifest.json"
