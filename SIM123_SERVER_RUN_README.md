# Sim1/Sim2/Sim3 cluster run

## Truth strategy for this round

- **Sim1**: the **hybrid truth** (`--truth-hybrid`) is rebuilt and frozen on the cluster.
  The model part is still produced by a MOT-(2D)^2 PCA fit (d=5, q=3, lambda=0.03,
  sigma=4, fixed seed), but **observed cells keep their original observed values**; only
  the unobserved cells (about 93% of this dataset) are filled with the fitted
  reconstruction. Rationale: a pure fitted truth retains only about 0.1% temporal
  variance, because the low-rank model flattens the temporal dynamics on sparse data,
  whereas the hybrid truth preserves the real temporal dynamics at the observed cells
  (within-trajectory sd about 0.3-0.42 log10).
- **Sim2**: reuses the earlier observed-only CP-ALS truth that produced the current
  downstream results; not rebuilt.
- **Sim3**: reuses the truth carrying the earlier artificial effects; not rebuilt.
- The comparator set is the same across all three: MOT-(2D)^2 PCA, Gemelli CTF, Mean,
  Nearest Positive Observation and Temporal-KNN.
- The production array jobs only read frozen truths. If any truth is missing the job
  fails immediately rather than rebuilding it concurrently.

## Imputation parameters (decoupled from the truth-construction parameters)

- **Truth-construction parameters** are fixed at d=5, q=3, lambda=0.03, sigma=4. They
  affect only the already-frozen truth, so rerunning the imputation does not require
  rebuilding it.
- **Imputation-stage MOT-(2D)^2 PCA parameters** are passed through the environment
  variables `IMPUTE_D / IMPUTE_Q / IMPUTE_LAM / IMPUTE_SIGMA / IMPUTE_T0 / IMPUTE_T /
  IMPUTE_EPS`. The defaults are the tuned configuration **d=1, q=1, lambda=2, sigma=16**
  (T0/T=30, eps=1e-3). `run_sim1_sim2_sim3_q2_202402.slurm`,
  `submit_sim1_sim2_sim3_q2_202402.sh` and
  `submit_sim1_sim2_sim3_q2_202402_chunked.sh` all support them; to change parameters,
  `export IMPUTE_D=...` before submitting rather than editing the scripts. Shard logs
  print `[shard] impute params: ...` for traceability.
- These affect only the `penalized_2d2_pca` method. CTF takes its own parameters such as
  `--competitor-ctf-rank 3`, and Mean / Nearest / Temporal-KNN have no model parameters.
- **A rerun must use a new `RUN_ROOT`** (for example
  `server_runs/sim123_v2_d1q1_lam2_sig16`). Reusing an old directory lets the merge mix
  shards from two parameter settings.

## Directory layout on the cluster

Prepare the following in the project root:

```text
penalized_2d2_pca/
├── competitor_methods.py
├── build_sim1_truth_q2_202402.slurm
├── preflight_sim123_q2_202402.slurm
├── run_sim1_sim2_sim3_q2_202402.slurm
├── merge_sim1_sim2_sim3_q2_202402.slurm
├── submit_sim1_sim2_sim3_q2_202402.sh
├── sim1_compare/
│   ├── sim1_methodcompare.py
│   ├── hmp2_mgx_taxonomy.tsv
│   └── metagenomics.xlsx
├── sim2-compare/
│   ├── ctf_reconstruct_compare.py
│   ├── hmp2_mgx_taxonomy.tsv
│   └── metagenomics.xlsx
├── sim3_compare/
│   ├── sim3_compare.py
│   ├── hmp2_mgx_taxonomy.tsv
│   └── metagenomics.xlsx
├── server_truths/
│   ├── sim1/
│   ├── sim2/truth_cpals.nc
│   └── sim3/truth_legacy_effect.nc
└── server_runs/logs/
```

The Sim1 truth is produced by a job. The Sim2 and Sim3 `.nc` files must be uploaded
beforehand.

## Step 1: build the Sim1 truth

```bash
cd /path/to/penalized_2d2_pca
mkdir -p server_truths/{sim1,sim2,sim3} server_runs/logs

export SIM1_TRUTH_PATH="$PWD/server_truths/sim1/truth_mot_d5_q3_lam003_sig4_hybrid.nc"
export SIM1_D=5
export SIM1_Q=3
export SIM1_LAM=0.03
export SIM1_SIGMA=4

SIM1_TRUTH_JOB=$(sbatch --parsable \
  --export=ALL,SIM1_TRUTH_PATH="$SIM1_TRUTH_PATH",SIM1_D="$SIM1_D",SIM1_Q="$SIM1_Q",SIM1_LAM="$SIM1_LAM",SIM1_SIGMA="$SIM1_SIGMA" \
  build_sim1_truth_q2_202402.slurm)
echo "$SIM1_TRUTH_JOB"
```

`build_sim1_truth_q2_202402.slurm` already sets `--truth-hybrid` (observed values kept,
unobserved cells filled by MOT-(2D)^2 PCA).

Once the job finishes, check:

```bash
sacct -j "$SIM1_TRUTH_JOB" --format=JobIDRaw,State,Elapsed,MaxRSS
test -s "$SIM1_TRUTH_PATH"
cat server_truths/sim1/truth_diagnostics_sim1.csv
```

Diagnostic targets for the hybrid truth, verified locally:

- `log_time_variance_fraction` about **0.02-0.03** (whole-tensor scope, diluted by the
  93% of smoothly filled cells; this is expected)
- `log_time_variance_fraction_observed_only` about **0.15-0.20** (share of temporal
  variance in the real dynamics at observed cells)
- `observed_trajectory_sd_mean` about **0.3-0.4** (log10 units, within-trajectory
  fluctuation)
- `truth_hybrid` = **1**
- Proceed to the production comparison only when the state is `COMPLETED` and the
  metrics above fall inside these ranges.

> Note: the earlier target of about 30% came from computing the metric under the
> floor-filled convention on the raw data. It is unreachable on a smoothly filled truth;
> the hybrid truth reports the observed-cell metric instead, which reflects the real
> temporal dynamics.

## Step 2: submit the method comparison

```bash
export SIM1_TRUTH_PATH="$PWD/server_truths/sim1/truth_mot_d5_q3_lam003_sig4_hybrid.nc"
export SIM2_TRUTH_PATH="$PWD/server_truths/sim2/truth_cpals.nc"
export SIM3_TRUTH_PATH="$PWD/server_truths/sim3/truth_legacy_effect.nc"
export RUN_ROOT="$PWD/server_runs/sim123_final_v1"

unset ARRAY_SPEC SKIP_MERGE BUILD_TRUTHS REBUILD_SIM2_TRUTH REBUILD_SIM3_TRUTH
bash submit_sim1_sim2_sim3_q2_202402.sh
```

The submit entry point creates, in order: the preflight job, the `0-749%180` array job,
and a merge job that depends on the array completing successfully. Maximum concurrency
is 180, below the cluster's 200-task limit.

> **Caution**: `AssocGrpSubmitJobsLimit` counts the full task count of the array **at
> submission time**; `%180` only limits concurrency. If the group submission limit is
> 200, submitting `0-749` in one go is rejected with
> `group max submit job limit exceeded 200`. In that case use the chunked driver script
> (waves of at most 188 tasks, 180 concurrent, run sequentially, with a single merge at
> the end), or ask the administrator to raise `GrpSubmitJobs` to at least 800 and keep
> using the original entry point.

### Fallback: chunked submission when the group limit is 200

```bash
export SIM1_TRUTH_PATH="$PWD/server_truths/sim1/truth_mot_d5_q3_lam003_sig4_hybrid.nc"
export SIM2_TRUTH_PATH="$PWD/server_truths/sim2/truth_cpals.nc"
export SIM3_TRUTH_PATH="$PWD/server_truths/sim3/truth_legacy_effect.nc"
export RUN_ROOT="$PWD/server_runs/sim123_final_v1"

unset SKIP_MERGE
chmod +x submit_sim1_sim2_sim3_q2_202402_chunked.sh
# nohup guards against an SSH disconnect; the driver polls each wave before
# submitting the next one.
nohup bash submit_sim1_sim2_sim3_q2_202402_chunked.sh \
  > server_runs/logs/chunked_driver_$(date +%Y%m%d_%H%M%S).log 2>&1 &

tail -f server_runs/logs/chunked_driver_*.log
```

Chunked flow: preflight -> wave 1 (0-187) -> wave 2 (188-375) -> wave 3 (376-563) ->
wave 4 (564-749) -> merge. A wave containing any non-`COMPLETED` task aborts the driver
and lists the failed task IDs; those shards can be resubmitted individually and the
merge run by hand afterwards.

## Optional: pilot run at a single masking fraction

Tasks 0-149 correspond to 50 repeats of each of the three simulations at `frac=0.5`:

```bash
export ARRAY_SPEC='0-149%150'
export SKIP_MERGE=1
export RUN_ROOT="$PWD/server_runs/sim123_pilot_frac05"
bash submit_sim1_sim2_sim3_q2_202402.sh
```

Submit the full run under a new `RUN_ROOT` afterwards; pilot shards must not be mixed
into the production directory.

## Monitoring

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobIDRaw,State,Elapsed,MaxRSS,AllocCPUS
```

Results are written to `$RUN_ROOT/sim1`, `$RUN_ROOT/sim2` and `$RUN_ROOT/sim3`.
