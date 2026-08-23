#!/usr/bin/env bash
#SBATCH --job-name=exp3_submit_next
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128M
#SBATCH --time=00:15:00
#SBATCH --output=exp3_submit_next_%j.out
#SBATCH --error=exp3_submit_next_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

NEXT_CHUNK="${NEXT_CHUNK:?NEXT_CHUNK is required}"
CHUNK_COUNT="${CHUNK_COUNT:?CHUNK_COUNT is required}"
CHUNKS_PER_SUBMISSION="${CHUNKS_PER_SUBMISSION:-160}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-40}"
SUBMIT_JOB_LIMIT="${SUBMIT_JOB_LIMIT:-200}"
CHUNK_CPUS="${CHUNK_CPUS:-16}"
CHUNK_MEM_PER_CPU="${CHUNK_MEM_PER_CPU:-750M}"
CONTINUE_CPUS="${CONTINUE_CPUS:-1}"
CONTINUE_MEM_PER_CPU="${CONTINUE_MEM_PER_CPU:-128M}"
MERGE_CPUS="${MERGE_CPUS:-12}"
MERGE_MEM_PER_CPU="${MERGE_MEM_PER_CPU:-750M}"

if (( NEXT_CHUNK > CHUNK_COUNT )); then
  echo "[slurm-exp3] no remaining chunks: NEXT_CHUNK=${NEXT_CHUNK}, CHUNK_COUNT=${CHUNK_COUNT}"
  exit 0
fi

if (( CHUNKS_PER_SUBMISSION > SUBMIT_JOB_LIMIT - 2 )); then
  echo "CHUNKS_PER_SUBMISSION=${CHUNKS_PER_SUBMISSION} is too high for SUBMIT_JOB_LIMIT=${SUBMIT_JOB_LIMIT}." >&2
  echo "Use at most SUBMIT_JOB_LIMIT-2 so this continuation job can submit the next dependency safely." >&2
  exit 2
fi

batch_end=$(( NEXT_CHUNK + CHUNKS_PER_SUBMISSION - 1 ))
if (( batch_end > CHUNK_COUNT )); then
  batch_end="${CHUNK_COUNT}"
fi

echo "[slurm-exp3] submitting chunk batch ${NEXT_CHUNK}-${batch_end} of ${CHUNK_COUNT}"
echo "[slurm-exp3] max concurrent chunks=${ARRAY_CONCURRENCY}; batch size cap=${CHUNKS_PER_SUBMISSION}"
echo "[slurm-exp3] chunk resources: cpus=${CHUNK_CPUS}, mem-per-cpu=${CHUNK_MEM_PER_CPU}"
echo "[slurm-exp3] continue resources: cpus=${CONTINUE_CPUS}, mem-per-cpu=${CONTINUE_MEM_PER_CPU}"
echo "[slurm-exp3] merge resources: cpus=${MERGE_CPUS}, mem-per-cpu=${MERGE_MEM_PER_CPU}"

export OUTDIR REPEATS FRACS MECHANISMS METHODS
export PYTHON_BIN TAX_PATH META_PATH RANK_D RANK_Q LAM SIGMA T0 T DTW_TRAJECTORIES WRITE_COMBINED_LONG ARRAY_CONCURRENCY
export CHUNK_COUNT CHUNKS_PER_SUBMISSION SUBMIT_JOB_LIMIT
export CHUNK_CPUS CHUNK_MEM_PER_CPU CONTINUE_CPUS CONTINUE_MEM_PER_CPU MERGE_CPUS MERGE_MEM_PER_CPU

chunk_job_id="$(
  sbatch --parsable \
    --cpus-per-task="${CHUNK_CPUS}" \
    --mem-per-cpu="${CHUNK_MEM_PER_CPU}" \
    --array=${NEXT_CHUNK}-${batch_end}%${ARRAY_CONCURRENCY} \
    slurm/submit_experiment3_chunks.sbatch
)"
chunk_dep_id="${chunk_job_id%%;*}"
echo "[slurm-exp3] submitted chunk batch ${NEXT_CHUNK}-${batch_end}: ${chunk_job_id}"

if (( batch_end < CHUNK_COUNT )); then
  following_chunk=$(( batch_end + 1 ))
  continue_job_id="$(
    sbatch --parsable \
      --cpus-per-task="${CONTINUE_CPUS}" \
      --mem-per-cpu="${CONTINUE_MEM_PER_CPU}" \
      --dependency=afterok:${chunk_dep_id} \
      --export=ALL,NEXT_CHUNK=${following_chunk},CHUNK_COUNT=${CHUNK_COUNT},CHUNKS_PER_SUBMISSION=${CHUNKS_PER_SUBMISSION} \
      slurm/submit_experiment3_chunks_continue.sbatch
  )"
  echo "[slurm-exp3] submitted continuation job for chunks ${following_chunk}-${CHUNK_COUNT}: ${continue_job_id}"
else
  merge_job_id="$(
    sbatch --parsable \
      --cpus-per-task="${MERGE_CPUS}" \
      --mem-per-cpu="${MERGE_MEM_PER_CPU}" \
      --dependency=afterok:${chunk_dep_id} \
      slurm/submit_experiment3_chunks_merge.sbatch
  )"
  echo "[slurm-exp3] submitted dependent Experiment 3 merge+analysis job: ${merge_job_id}"
fi
