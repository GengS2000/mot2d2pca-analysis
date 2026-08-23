#!/bin/bash
set -euo pipefail
STAGE=${1:?usage: submit_experiment.sh stage_b|stage_c}; [[ "$STAGE" =~ ^stage_[bc]$ ]] || exit 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
# Some legacy conda `bin/activate` scripts interpret this script's positional
# arguments as an environment name.  Preserve STAGE above, then clear `$1`.
set --
source ~/miniconda3/bin/activate; conda activate micro3dimp; export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
OUTCFG=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml')).get('output_dir', 'results'))")
if [[ "$OUTCFG" = /* ]]; then OUT="$OUTCFG"; else OUT="$ROOT/$OUTCFG"; fi
export ROBUSTNESS_OUTPUT_ROOT="$OUT"
case "$STAGE" in
  stage_b) STAGE_CONFIG="configs/stage_b_sensitivity.yaml" ;;
  stage_c) STAGE_CONFIG="configs/stage_c_ablation.yaml" ;;
esac
DATASET=$(python -c "import yaml; print(yaml.safe_load(open('$STAGE_CONFIG'))['dataset'])")
python scripts/prepare_data.py --dataset "$DATASET"
python "scripts/build_${STAGE}_manifest.py"
N=$(( $(wc -l < "manifests/${STAGE}.csv") - 1 ))
read_cfg(){ python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml'))['$1'])"; }
PARTITION=$(read_cfg partition); CPUS=$(read_cfg cpus_per_task); MEM=$(read_cfg mem); WALLTIME=$(read_cfg time); MAX=$(read_cfg array_concurrency); WORKERS=$(read_cfg max_array_tasks); LIMIT=$(read_cfg submit_job_limit); RETRIES=$(read_cfg task_retries)
mkdir -p "$OUT/logs" "$OUT/$STAGE/tasks" "$OUT/aggregate/$STAGE"
CURRENT=$(squeue --array --noheader --user="$USER" 2>/dev/null | wc -l); AVAILABLE=$((LIMIT-CURRENT-1)); ((AVAILABLE>=1)) || { echo "No submission slot available" >&2; exit 1; }
((WORKERS>AVAILABLE)) && WORKERS=$AVAILABLE; ((WORKERS>N)) && WORKERS=$N; ((MAX>WORKERS)) && MAX=$WORKERS
COMMON=(--partition="$PARTITION" --cpus-per-task="$CPUS" --mem="$MEM" --time="$WALLTIME" --chdir="$ROOT")
EXPORTS="ALL,ROBUSTNESS_SUITE_ROOT=$ROOT,ROBUSTNESS_OUTPUT_ROOT=$OUT,STAGE=$STAGE,TASK_RETRIES=$RETRIES"
JOB=$(sbatch --parsable "${COMMON[@]}" --export="$EXPORTS" --output="$OUT/logs/${STAGE}_%A_%a.out" --error="$OUT/logs/${STAGE}_%A_%a.err" --array="0-$((WORKERS-1))%${MAX}" slurm/run_experiment_array.slurm)
AGG=$(sbatch --parsable "${COMMON[@]}" --export="$EXPORTS" --output="$OUT/logs/${STAGE}_aggregate_%j.out" --error="$OUT/logs/${STAGE}_aggregate_%j.err" --dependency="afterok:$JOB" slurm/aggregate_experiment.slurm)
echo "$STAGE: $N tasks across $WORKERS workers; array=$JOB aggregation=$AGG"
