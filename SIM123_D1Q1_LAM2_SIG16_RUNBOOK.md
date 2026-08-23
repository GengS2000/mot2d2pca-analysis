# Sim1/Sim2/Sim3 rerun: d=1, q=1, lambda=2, sigma=16

This rerun changes only the `penalized_2d2_pca` imputation parameters in the three
comparison experiments. Do **not** rebuild the frozen truth of Sim1/Sim2/Sim3 with these
parameters; all methods must keep reading the same existing truth tensors, otherwise the
comparison is not matched.

## 1. Set up the variables in the project root on the cluster

Adjust the three truth filenames below to their actual locations on the cluster. Each
must be a non-empty file.

```bash
cd /path/to/penalized_2d2_pca

export SIM1_TRUTH_PATH="$PWD/server_truths/sim1/truth_mot_d5_q3_lam003_sig4_hybrid.nc"
export SIM2_TRUTH_PATH="$PWD/server_truths/sim2/truth_cpals.nc"
export SIM3_TRUTH_PATH="$PWD/server_truths/sim3/truth_legacy_effect.nc"

test -s "$SIM1_TRUTH_PATH"
test -s "$SIM2_TRUTH_PATH"
test -s "$SIM3_TRUTH_PATH"

export IMPUTE_D=1
export IMPUTE_Q=1
export IMPUTE_LAM=2
export IMPUTE_SIGMA=16
export IMPUTE_T0=30
export IMPUTE_T=30
export IMPUTE_EPS=1e-3

# Must be a directory that has never been used; do not place any file in it beforehand.
export RUN_ROOT="$PWD/server_runs/sim123_d1q1_lam2_sig16_$(date +%Y%m%d_%H%M%S)"
test ! -e "$RUN_ROOT"
unset ALLOW_RUN_ROOT_REUSE SKIP_MERGE
```

## 2. Submit all 750 array tasks at once

The cluster currently allows 1,000 submitted jobs with at most 200 running
concurrently, so `0-749%200` can be submitted in one go. The preflight job, the 750
array elements and the merge job total 752 jobs, below the submission cap, and `%200`
limits concurrency to 200. The submit script only calls `sbatch` and returns within
seconds, so `nohup` is unnecessary:

```bash
export ARRAY_SPEC='0-749%200'
bash submit_sim1_sim2_sim3_q2_202402.sh | tee "submit_d1q1_lam2_sig16_$(date +%Y%m%d_%H%M%S).log"
```

The dependency order is preflight -> the `0-749%200` array -> merge. The array does not
run if the preflight fails, and the `afterok` merge does not run if any array task
fails. Slurm logs for each run are under `$RUN_ROOT/logs`, results under
`$RUN_ROOT/sim1`, `sim2` and `sim3`, and the parameters are recorded in
`$RUN_ROOT/manifest.json`. Switch to the chunked script only if the cluster submission
cap drops below 752 again.

## 3. Check status

```bash
squeue -u "$USER"
cat "$RUN_ROOT/manifest.json"
```

The manifest must show:

```json
"impute_params": {"d": "1", "q": "1", "lam": "2", "sigma": "16"}
```

Any shard log should likewise contain:

```text
[shard] impute params: d=1 q=1 lam=2 sigma=16 ...
```

## 4. Checks around the merge

The merge job runs strict checks automatically: it verifies the manifest parameters and
the exact 250 `(frac, repeat)` shard and completion markers for each of the three
simulations. Missing files, empty files, unreadable CSVs or error markers all block the
merge. The same checks can be run manually:

```bash
source ~/miniconda3/bin/activate
conda activate q2-202402
python validate_sim123_run.py "$RUN_ROOT"
```

After a successful merge these files must exist:

```bash
test -s "$RUN_ROOT/sim1/summary_with_errorbar_sim1.csv"
test -s "$RUN_ROOT/sim2/errorbar_summary.csv"
test -s "$RUN_ROOT/sim3/errorbar_summary.csv"
```

## 5. Manual resubmission, only when the merge failed for resource reasons

First confirm that the 750 shard checks pass, then resubmit the merge alone. Do not
rerun the array, and do not set `ALLOW_RUN_ROOT_REUSE=1`:

```bash
python validate_sim123_run.py "$RUN_ROOT"
sbatch --parsable \
  --cpus-per-task=16 \
  --mem-per-cpu=2G \
  --output="$RUN_ROOT/logs/%x_%j.out" \
  --error="$RUN_ROOT/logs/%x_%j.err" \
  --export=ALL,RUN_ROOT="$RUN_ROOT",IMPUTE_D=1,IMPUTE_Q=1,IMPUTE_LAM=2,IMPUTE_SIGMA=16 \
  merge_sim1_sim2_sim3_q2_202402.slurm
```

The merge requests 16 cores at 2 GB per core, 32 GB in total. The 16 cores are mainly
there to satisfy the partition's memory-per-core quota; the merge program still
restricts BLAS to a single thread. The taxon-level merged CSVs are large, and the
original single-core 3 GB configuration tended to trigger a Slurm `OUT_OF_MEMORY`. If
the strict checks report missing shards, rerun the corresponding array tasks first
rather than forcing the merge past the checks.
