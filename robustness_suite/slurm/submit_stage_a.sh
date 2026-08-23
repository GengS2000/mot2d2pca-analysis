#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source ~/miniconda3/bin/activate
conda activate micro3dimp
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

DATASET=$(python -c "import yaml; print(yaml.safe_load(open('configs/stage_a_tuning.yaml'))['dataset'])")
python scripts/prepare_data.py --dataset "$DATASET"
python scripts/generate_masks.py --dataset "$DATASET"
python scripts/build_stage_a_manifest.py
N=$(( $(wc -l < manifests/stage_a.csv) - 1 ))
MAX=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml'))['array_concurrency'])")
WORKERS=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml')).get('max_array_tasks', 200))")
SUBMIT_LIMIT=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml')).get('submit_job_limit', 200))")
PARTITION=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml'))['partition'])")
ACCOUNT=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml')).get('account', ''))")
CPUS=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml'))['cpus_per_task'])")
MEM=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml'))['mem'])")
WALLTIME=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml'))['time'])")
OUTPUT_CONFIG=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml')).get('output_dir', 'results'))")
TASK_RETRIES=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml')).get('task_retries', 2))")
if [[ "$OUTPUT_CONFIG" = /* ]]; then OUTPUT_ROOT="$OUTPUT_CONFIG"; else OUTPUT_ROOT="$ROOT/$OUTPUT_CONFIG"; fi
SBATCH_COMMON=(--partition="$PARTITION" --cpus-per-task="$CPUS" --mem="$MEM" --time="$WALLTIME" --chdir="$ROOT")
if [[ -n "$ACCOUNT" ]]; then SBATCH_COMMON+=(--account="$ACCOUNT"); fi
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/stage_a/tasks" "$OUTPUT_ROOT/aggregate"
if [[ ! -w "$OUTPUT_ROOT" ]]; then echo "Output directory is not writable: $OUTPUT_ROOT" >&2; exit 1; fi
CURRENT_JOBS=$(squeue --array --noheader --user="$USER" 2>/dev/null | wc -l)
AVAILABLE_WORKERS=$(( SUBMIT_LIMIT - CURRENT_JOBS - 1 ))
if (( AVAILABLE_WORKERS < 1 )); then
  echo "No submission slots are available: current=${CURRENT_JOBS}, limit=${SUBMIT_LIMIT}; one slot is also required for aggregation." >&2
  echo "Wait for existing jobs to finish or cancel the obsolete mot2d2_stage_a array, then rerun this script." >&2
  exit 1
fi
if (( WORKERS > AVAILABLE_WORKERS )); then WORKERS=$AVAILABLE_WORKERS; fi
if (( WORKERS > N + 1 )); then WORKERS=$(( N + 1 )); fi
if (( MAX > WORKERS )); then MAX=$WORKERS; fi
EXPORTS="ALL,ROBUSTNESS_SUITE_ROOT=$ROOT,ROBUSTNESS_OUTPUT_ROOT=$OUTPUT_ROOT,TASK_RETRIES=$TASK_RETRIES"
JOB_A=$(sbatch --parsable "${SBATCH_COMMON[@]}" --export="$EXPORTS" --output="$OUTPUT_ROOT/logs/stage_a_%A_%a.out" --error="$OUTPUT_ROOT/logs/stage_a_%A_%a.err" --array="0-$((WORKERS - 1))%${MAX}" slurm/10_stage_a_array.slurm)
JOB_AGG=$(sbatch --parsable "${SBATCH_COMMON[@]}" --export="$EXPORTS" --output="$OUTPUT_ROOT/logs/stage_a_aggregate_%j.out" --error="$OUTPUT_ROOT/logs/stage_a_aggregate_%j.err" --dependency="afterok:${JOB_A}" slurm/11_stage_a_aggregate.slurm)
echo "Stage A: ${N} manifest tasks distributed across ${WORKERS} Slurm workers."
echo "Stage A array: ${JOB_A}"
echo "Stage A aggregation: ${JOB_AGG}"
echo "Results and logs: ${OUTPUT_ROOT}"
