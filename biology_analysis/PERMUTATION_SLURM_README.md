# ECAM / DIABIMMUNE: Slurm plan for 1,000 permutations

## Files

- `permutation_slurm/permutation_worker.R`: supports `perm_start/perm_end`, `perm_ids`
  and compact cleanup.
- `permutation_slurm/aggregate_permutations.R`: streaming aggregation, without writing a
  giant per-row intermediate CSV.
- `permutation_slurm/preflight_server.R`: checks the R packages, `linda()` and the input
  files.
- `slurm/permutation_array.slurm`: Slurm array worker.
- `slurm/permutation_aggregate.slurm`: aggregation job.
- `slurm/submit_permutation_1000.sh`: submits the 1,000 permutations plus the
  true-label baseline.
- `slurm/resubmit_missing_permutations.sh`: reruns only the missing permutation IDs.

## Compact file contract

Only these files are kept for each `(cohort, method, permutation_id)`:

1. `results/<cohort>/<method>/perm_XXXX.rds`: compressed per-taxon results, covering both
   observed and completed;
2. `logs/<cohort>/<method>/perm_XXXX.log`: model log;
3. `status/<cohort>/<method>/perm_XXXX.rds`: status, elapsed time, exit code and cleanup
   state;
4. `labels/<cohort>/perm_XXXX.csv.gz`: the participant labels actually used.

The MaAsLin2 `features/`, `fits/`, residuals and other large outputs are written only
under `work/`, then recursively cleaned immediately after each task is validated and its
compact RDS saved. Failed tasks also clean their temporary directories; the errors
remain in the log and status files.

## Cluster deployment

Sync the whole `biology_analysis` directory to the cluster, for example:

```text
/public/home/USER/biology_analysis
```

Confirm that the NetCDF files, metadata, scripts and the following directory structure
are all synced:

```text
biology_analysis/
  ECAM_data/
  DIABIMMUNE_data/
  permutation_validation_small.R
  permutation_slurm/
  slurm/
```

Enter the project directory:

```bash
cd /public/home/USER/biology_analysis
chmod +x slurm/*.sh
```

## Environment check

```bash
/path/to/R/4.4.2/Rscript permutation_slurm/preflight_server.R
```

This must print:

```text
PREFLIGHT_OK
```

If the administrator installed the packages into a personal library, set this before
submitting, for example:

```bash
export R_LIBS_USER=/path/to/your/R/library
```

The submit scripts use `--export=ALL`, which passes the variable to the array and
aggregation jobs.

## Run a compact test on the cluster first

Do not submit all 1,000 permutations on the first attempt. Run `0-1` directly first,
that is the true-label baseline plus one permutation:

```bash
/path/to/R/4.4.2/Rscript \
  permutation_slurm/permutation_worker.R \
  --perm-start 0 --perm-end 1 \
  --project-dir "$PWD" \
  --out-dir "$PWD/results/permutation_server_smoke" \
  --rscript /path/to/R/4.4.2/Rscript \
  --compact true --resume true --cores 1
```

Aggregation test:

```bash
/path/to/R/4.4.2/Rscript \
  permutation_slurm/aggregate_permutations.R \
  --project-dir "$PWD" \
  --out-dir "$PWD/results/permutation_server_smoke" \
  --n-perm 1
```

Check:

```bash
cat results/permutation_server_smoke/aggregate/completeness_report.csv
du -sh results/permutation_server_smoke
find results/permutation_server_smoke/work -type f | head
```

Completeness should be `complete=TRUE`, and no model intermediates should remain under
`work/`.

## Submit the 1,000 permutations

Defaults: 10 permutation IDs per array task, at most 12 concurrent workers, 1 CPU / 3 GB
/ 4 hours per worker. The aggregation job is also 1 CPU / 3 GB and reads in a streaming
fashion.

```bash
cd /public/home/USER/biology_analysis
bash slurm/submit_permutation_1000.sh "$PWD/results/permutation_1000"
```

The default partition is `hebhcnormal02`. To override the partition or set an account:

```bash
PARTITION=YOUR_PARTITION ACCOUNT=YOUR_ACCOUNT \
MAX_CONCURRENT=12 CHUNK_SIZE=10 \
bash slurm/submit_permutation_1000.sh "$PWD/results/permutation_1000"
```

If resources are tight or single tasks time out, reduce the chunk size:

```bash
CHUNK_SIZE=5 MAX_CONCURRENT=12 WORKER_TIME=03:00:00 \
bash slurm/submit_permutation_1000.sh "$PWD/results/permutation_1000"
```

Do not raise `MAX_CONCURRENT` and the per-model `cores` at the same time. This plan
parallelizes at the outer array level, fixes `cores=1` inside each model, and sets the
BLAS/OpenMP thread counts to 1.

## Monitoring

The submit script prints the array and aggregate job IDs:

```bash
squeue -u "$USER"
tail -f results/permutation_1000/logs/slurm/worker_JOBID_0.out
```

Progress estimate:

```bash
find results/permutation_1000/results -name 'perm_*.rds' | wc -l
```

The final count should be:

```text
(1000 + 1) x 2 cohorts x 2 methods = 4004 compact result files
```

## Aggregated output

The aggregation job uses `afterany`, so a completeness report is produced even if some
workers failed:

- `aggregate/completeness_report.csv`
- `aggregate/missing_or_failed_tasks.csv`
- `aggregate/missing_permutation_ids.txt`
- `aggregate/permutation_run_summary.csv`
- `aggregate/null_calibration_summary.csv`
- `aggregate/paired_inflation_completed_vs_observed.csv`
- `aggregate/empirical_permutation_results.csv`
- `aggregate/compact_result_manifest.csv`
- `aggregate/label_manifest.csv`
- `aggregate/sessionInfo.txt`

Of these, `null_calibration_summary.csv` and
`paired_inflation_completed_vs_observed.csv` are the main results of the 1,000-permutation
false-positive inflation check. `empirical_permutation_results.csv` is produced at the
same time, but the smallest empirical p value from 1,000 permutations is about 0.001,
which gives limited resolution once BH is applied across 480/839 taxa; it should not be
used on its own as the final per-taxon significance criterion.

If tasks are missing:

```bash
bash slurm/resubmit_missing_permutations.sh "$PWD/results/permutation_1000"
```

On a rerun, cohort/method combinations that already succeeded within the same
permutation ID are skipped automatically by `resume=TRUE`, and only the missing results
are recomputed.

## Manual shard invocation

Contiguous range:

```bash
Rscript permutation_slurm/permutation_worker.R \
  --perm-start 101 --perm-end 120 \
  --project-dir "$PWD" --out-dir "$PWD/results/manual" --compact true
```

Specific IDs:

```bash
Rscript permutation_slurm/permutation_worker.R \
  --perm-ids 7,19,203 \
  --project-dir "$PWD" --out-dir "$PWD/results/manual" --compact true
```
