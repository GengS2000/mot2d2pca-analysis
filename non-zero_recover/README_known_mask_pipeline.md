# Known-mask experiment pipeline

This pipeline treats the imputation mask as provided by the experiment design. It does not perform zero detection. The legacy `oracle_mask` name is normalized to `known_mask` for compatibility, but new runs should use `known_mask`.

## Local parallel run

Run all four experiments concurrently:

```bash
python scripts/run_known_mask_experiments.py \
  --outdir outputs_known_mask \
  --max-workers 4
```

The runner always passes `--replacement-modes known_mask` to worker experiments. It also passes `--skip-preprocessing-diagnostics` by default to avoid parallel writes to shared preprocessing diagnostic files. Experiment result CSVs are still written in full.

For a full benchmark, the runner defaults to:

- mechanisms: `random_entry,low_depth_sample,time_point,subject_specific_dropout,consecutive_time_window_dropout,low_abundance_biased`
- fractions: `0.05,0.10,0.20,0.30`
- repeats: `10`
- methods: `penalized_2d2_pca,no_imputation`

Run repeat-level inference after the experiments finish:

```bash
python scripts/analyze_repeat_level_inference.py \
  --root outputs_known_mask \
  --outdir outputs_known_mask/analysis \
  --replacement-mode known_mask \
  --baseline no_imputation
```

## SLURM run

Submit the experiment array and dependent analysis job:

```bash
bash slurm/submit_known_mask_pipeline.sh
```

Optional overrides:

```bash
OUTDIR=outputs_known_mask REPEATS=20 PYTHON_BIN=/path/to/python bash slurm/submit_known_mask_pipeline.sh
```

The experiment array has four tasks:

- task 1: experiment 1 zero recovery
- task 2: experiment 2 non-zero distribution recovery
- task 3: experiment 3 longitudinal trajectory recovery
- task 4: experiment 4 dynamic correlation/network recovery

## Recommended rerun for Experiment 3

Experiment 3 can produce very large trajectory-level tables. The recommended server workflow is now the chunked array runner:

```bash
OUTDIR=outputs_known_mask \
PYTHON_BIN=/path/to/python \
bash slurm/submit_experiment3_chunks_pipeline.sh
```

By default the wrapper limits the array to 40 concurrent chunks to avoid excessive shared-filesystem reads. Override if needed:

```bash
ARRAY_CONCURRENCY=20 OUTDIR=outputs_known_mask PYTHON_BIN=/path/to/python bash slurm/submit_experiment3_chunks_pipeline.sh
```

Some clusters count every array element against the submit-job quota. If your account has a 200-job submit limit, the default 240 chunks are automatically submitted in staged batches. The default batch size is 160:

```bash
CHUNKS_PER_SUBMISSION=160 OUTDIR=outputs_known_mask PYTHON_BIN=/path/to/python bash slurm/submit_experiment3_chunks_pipeline.sh
```

For stricter quotas, reduce the staged batch size, for example:

```bash
CHUNKS_PER_SUBMISSION=120 ARRAY_CONCURRENCY=40 OUTDIR=outputs_known_mask PYTHON_BIN=/path/to/python bash slurm/submit_experiment3_chunks_pipeline.sh
```

Some clusters also reject jobs whose requested memory per CPU exceeds `DefMemPerCPU`. The chunk pipeline requests memory with `--mem-per-cpu` by default. Override these settings if your cluster uses a different limit:

```bash
CHUNK_CPUS=16 \
CHUNK_MEM_PER_CPU=750M \
CONTINUE_CPUS=1 \
CONTINUE_MEM_PER_CPU=128M \
MERGE_CPUS=12 \
MERGE_MEM_PER_CPU=750M \
OUTDIR=outputs_known_mask \
PYTHON_BIN=/path/to/python \
bash slurm/submit_experiment3_chunks_pipeline.sh
```

This submits one SLURM array task per:

```text
mechanism x frac x repeat
```

With the defaults, this is:

```text
6 mechanisms x 4 fractions x 10 repeats = 240 chunks
```

Each chunk compares only:

```text
penalized_2d2_pca,no_imputation
```

Chunk outputs are kept under:

```text
outputs_known_mask/experiment3_trajectory/chunks/
```

The dependent merge job writes:

- `outputs_known_mask/experiment3_trajectory/subject_level_metrics.csv`
- `outputs_known_mask/experiment3_trajectory/taxon_clustering_metrics.csv`
- `outputs_known_mask/experiment3_trajectory/trajectory_metrics_repeat_level.csv`
- `outputs_known_mask/experiment3_trajectory/trajectory_metrics_summary.csv`
- `outputs_known_mask/experiment3_trajectory/failed_runs.csv`
- `outputs_known_mask/experiment3_trajectory/chunk_manifest.csv`
- `outputs_known_mask/analysis/repeat_level_inference_stats.csv`

By default, the merge job does not create the very large combined `trajectory_metrics_long.csv`; all partial long CSVs remain preserved in `chunks/`. If a combined long table is required, submit with:

```bash
WRITE_COMBINED_LONG=1 OUTDIR=outputs_known_mask bash slurm/submit_experiment3_chunks_pipeline.sh
```

## Preserved output CSVs

Experiment 1:

- `outputs_known_mask/experiment1_zero_recovery/config.json`
- `outputs_known_mask/experiment1_zero_recovery/metrics_long.csv`
- `outputs_known_mask/experiment1_zero_recovery/metrics_summary.csv`
- `outputs_known_mask/experiment1_zero_recovery/failed_runs.csv`

Experiment 2:

- `outputs_known_mask/experiment2_distribution/config.json`
- `outputs_known_mask/experiment2_distribution/distribution_metrics_long.csv`
- `outputs_known_mask/experiment2_distribution/distribution_metrics_summary.csv`
- `outputs_known_mask/experiment2_distribution/raw_nonzero_da_log_distribution_summary.csv`
- `outputs_known_mask/experiment2_distribution/raw_nonzero_da_log_distribution_quantiles.csv`
- `outputs_known_mask/experiment2_distribution/raw_nonzero_vs_imputed_at_masked_distribution_metrics.csv`
- `outputs_known_mask/experiment2_distribution/raw_nonzero_vs_imputed_at_masked_distribution_summary.csv`
- `outputs_known_mask/experiment2_distribution/raw_nonzero_vs_imputed_at_masked_distribution_quantiles.csv`
- `outputs_known_mask/experiment2_distribution/failed_runs.csv`

Experiment 3:

- `outputs_known_mask/experiment3_trajectory/config.json`
- `outputs_known_mask/experiment3_trajectory/chunks/trajectory_metrics_long_part_*.csv`
- `outputs_known_mask/experiment3_trajectory/trajectory_metrics_long.csv` only if `WRITE_COMBINED_LONG=1`
- `outputs_known_mask/experiment3_trajectory/trajectory_metrics_repeat_level.csv`
- `outputs_known_mask/experiment3_trajectory/trajectory_metrics_summary.csv`
- `outputs_known_mask/experiment3_trajectory/subject_level_metrics.csv`
- `outputs_known_mask/experiment3_trajectory/taxon_clustering_metrics.csv`
- `outputs_known_mask/experiment3_trajectory/failed_runs.csv`

Experiment 4:

- `outputs_known_mask/experiment4_dynamic_correlation/config.json`
- `outputs_known_mask/experiment4_dynamic_correlation/static_correlation_metrics.csv`
- `outputs_known_mask/experiment4_dynamic_correlation/lagged_correlation_metrics.csv`
- `outputs_known_mask/experiment4_dynamic_correlation/network_edge_overlap.csv`
- `outputs_known_mask/experiment4_dynamic_correlation/hub_taxa_overlap.csv`
- `outputs_known_mask/experiment4_dynamic_correlation/failed_runs.csv`

Analysis:

- `outputs_known_mask/analysis/repeat_level_inference_stats.csv`
- `outputs_known_mask/analysis/method_rankings.csv`
- `outputs_known_mask/analysis/analysis_manifest.csv`

## Statistical unit

The primary inference unit is:

```text
mechanism x frac x repeat x replacement_mode
```

Metrics with finer rows, such as taxon-, participant-, trajectory-, or network-edge-level rows, are first averaged within each primary unit and method. This avoids treating correlated taxa, subjects, or trajectories as independent replicates.
