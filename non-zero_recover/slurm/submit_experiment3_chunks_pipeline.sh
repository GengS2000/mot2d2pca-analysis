#!/usr/bin/env bash
set -euo pipefail

# Submit only Experiment 3 in chunked form.
#
# Run from the project root:
#   OUTDIR=outputs_known_mask PYTHON_BIN=/path/to/python bash slurm/submit_experiment3_chunks_pipeline.sh
#
# Defaults:
#   mechanisms = 6
#   fracs      = 4
#   repeats    = 10
#   methods    = penalized_2d2_pca,no_imputation
#   chunks     = mechanisms * fracs * repeats = 240
#
# Some clusters count every array task against AssocGrpSubmitJobsLimit. For
# example, a 240-task array can be rejected under a 200-job submit limit even
# when ARRAY_CONCURRENCY is only 40. This wrapper submits chunks in staged
# batches and lets a tiny continuation job submit the next batch after the
# previous batch has completed.

OUTDIR="${OUTDIR:-outputs_known_mask}"
REPEATS="${REPEATS:-10}"
FRACS="${FRACS:-0.05,0.10,0.20,0.30}"
MECHANISMS="${MECHANISMS:-random_entry,low_depth_sample,time_point,subject_specific_dropout,consecutive_time_window_dropout,low_abundance_biased}"
METHODS="${METHODS:-penalized_2d2_pca,no_imputation}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-40}"
CHUNKS_PER_SUBMISSION="${CHUNKS_PER_SUBMISSION:-160}"
SUBMIT_JOB_LIMIT="${SUBMIT_JOB_LIMIT:-200}"
CHUNK_CPUS="${CHUNK_CPUS:-16}"
CHUNK_MEM_PER_CPU="${CHUNK_MEM_PER_CPU:-750M}"
CONTINUE_CPUS="${CONTINUE_CPUS:-1}"
CONTINUE_MEM_PER_CPU="${CONTINUE_MEM_PER_CPU:-128M}"
MERGE_CPUS="${MERGE_CPUS:-12}"
MERGE_MEM_PER_CPU="${MERGE_MEM_PER_CPU:-750M}"

IFS=',' read -r -a frac_arr <<< "${FRACS}"
IFS=',' read -r -a mech_arr <<< "${MECHANISMS}"
chunk_count=$(( ${#frac_arr[@]} * ${#mech_arr[@]} * REPEATS ))

if (( ARRAY_CONCURRENCY < 1 )); then
  echo "ARRAY_CONCURRENCY must be positive; got ${ARRAY_CONCURRENCY}" >&2
  exit 2
fi

if (( CHUNKS_PER_SUBMISSION < 1 )); then
  echo "CHUNKS_PER_SUBMISSION must be positive; got ${CHUNKS_PER_SUBMISSION}" >&2
  exit 2
fi

if (( chunk_count > CHUNKS_PER_SUBMISSION && CHUNKS_PER_SUBMISSION > SUBMIT_JOB_LIMIT - 2 )); then
  echo "CHUNKS_PER_SUBMISSION=${CHUNKS_PER_SUBMISSION} is too high for SUBMIT_JOB_LIMIT=${SUBMIT_JOB_LIMIT}." >&2
  echo "Use at most SUBMIT_JOB_LIMIT-2 so the continuation job can submit the next batch safely." >&2
  exit 2
fi

first_end="${CHUNKS_PER_SUBMISSION}"
if (( first_end > chunk_count )); then
  first_end="${chunk_count}"
fi

echo "Submitting Experiment 3 chunk pipeline with ${chunk_count} chunks"
echo "First batch: 1-${first_end}; max concurrent ${ARRAY_CONCURRENCY}; batch size cap ${CHUNKS_PER_SUBMISSION}"
echo "OUTDIR=${OUTDIR}"
echo "METHODS=${METHODS}"
echo "CHUNK resources: cpus=${CHUNK_CPUS}, mem-per-cpu=${CHUNK_MEM_PER_CPU}"
echo "CONTINUE resources: cpus=${CONTINUE_CPUS}, mem-per-cpu=${CONTINUE_MEM_PER_CPU}"
echo "MERGE resources: cpus=${MERGE_CPUS}, mem-per-cpu=${MERGE_MEM_PER_CPU}"

export OUTDIR REPEATS FRACS MECHANISMS METHODS
export PYTHON_BIN TAX_PATH META_PATH RANK_D RANK_Q LAM SIGMA T0 T DTW_TRAJECTORIES WRITE_COMBINED_LONG ARRAY_CONCURRENCY
export CHUNK_COUNT="${chunk_count}" CHUNKS_PER_SUBMISSION SUBMIT_JOB_LIMIT
export CHUNK_CPUS CHUNK_MEM_PER_CPU CONTINUE_CPUS CONTINUE_MEM_PER_CPU MERGE_CPUS MERGE_MEM_PER_CPU

chunk_job_id="$(
  sbatch --parsable \
    --cpus-per-task="${CHUNK_CPUS}" \
    --mem-per-cpu="${CHUNK_MEM_PER_CPU}" \
    --array=1-${first_end}%${ARRAY_CONCURRENCY} \
    slurm/submit_experiment3_chunks.sbatch
)"
chunk_dep_id="${chunk_job_id%%;*}"
echo "Submitted Experiment 3 chunk batch 1-${first_end}: ${chunk_job_id}"

if (( first_end < chunk_count )); then
  next_chunk=$(( first_end + 1 ))
  continue_job_id="$(
    sbatch --parsable \
      --cpus-per-task="${CONTINUE_CPUS}" \
      --mem-per-cpu="${CONTINUE_MEM_PER_CPU}" \
      --dependency=afterok:${chunk_dep_id} \
      --export=ALL,NEXT_CHUNK=${next_chunk},CHUNK_COUNT=${chunk_count},CHUNKS_PER_SUBMISSION=${CHUNKS_PER_SUBMISSION} \
      slurm/submit_experiment3_chunks_continue.sbatch
  )"
  echo "Submitted continuation job for chunks ${next_chunk}-${chunk_count}: ${continue_job_id}"
else
  merge_job_id="$(
    sbatch --parsable \
      --cpus-per-task="${MERGE_CPUS}" \
      --mem-per-cpu="${MERGE_MEM_PER_CPU}" \
      --dependency=afterok:${chunk_dep_id} \
      slurm/submit_experiment3_chunks_merge.sbatch
  )"
  echo "Submitted dependent Experiment 3 merge+analysis job: ${merge_job_id}"
fi
