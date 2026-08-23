# CTF reference baseline

CTF (the official Gemelli implementation) is the only external tensor-factorization
baseline in this benchmark. The other three comparators (Mean, Nearest Positive
Observation and Temporal-KNN) are implemented in
`sim1_compare/sim1_methodcompare.py`.

## Fairness constraints enforced by the adapter

- All three simulations reuse their own existing `replace_mask` and random seeds; no
  mask is resampled for the comparator.
- The comparator only ever sees the masked input. The mask used to fit CTF explicitly
  excludes every cell in `replace_mask`.
- The adapter writes back only at cells where `replace_mask` is `True`. Every run
  additionally verifies element-wise that off-mask values are bit-for-bit unchanged.
- Input and output stay on the `log10(percent + eps)` scale, with `eps=1e-3` by default.
- Each method, masking fraction and repeat is an independent task. A failure writes
  `competitor_failed_*.json` and does not affect the other tasks.

CTF uses Gemelli's `tensor_rclr`, `tenals` and `construct_tensor`. Its reconstructed
scores are used only to distribute the residual relative-abundance mass of the masked
taxa within each participant x bin slice; unmasked taxa keep their original values.

Pinned upstream source:

- Gemelli/CTF: <https://github.com/biocore/gemelli>, pinned at commit
  `c53c9ee958948683bc4216cf5fa447e2c3ac4806`.

## Environment

No R packages are required. Install Gemelli into the existing benchmark environment
(called `micro3dimp` below):

```bash
python -m pip install -r requirements_competitors.txt
```

Pre-run check:

```bash
conda activate micro3dimp
python -c "import gemelli; print(gemelli.__version__)"
```

## Full SLURM run

```bash
export PROJECT_DIR="$PWD"
export BENCHMARK_ENV=micro3dimp
bash slurm/submit_ctf_only.sh
```

If the three **identical** truth tensors used by the earlier methods already exist on
the cluster, place them under `artifacts/competitor_truths/{sim1,sim2,sim3}.nc` and skip
rebuilding them:

```bash
SKIP_TRUTH_PREP=1 bash slurm/submit_ctf_only.sh
```

The default design is 3 simulations x 5 masking fractions (0.5-0.9) x 50 repeats, that
is 750 logical tasks. To stay within `AssocGrpSubmitJobsLimit=200`, the CTF SLURM array
has only 197 scheduling elements; each element runs about 4 logical tasks serially over
a strided index, covering all 750 simulation x fraction x repeat combinations. At most
144 CTF tasks run concurrently, and no GPU is required. The merge program audits the
750 CTF logical tasks only.

If the truth tensors were already produced by the original experiments, they can be
placed directly at:

```text
artifacts/competitor_truths/sim1.nc
artifacts/competitor_truths/sim2.nc
artifacts/competitor_truths/sim3.nc
```

and the array can then be submitted without the prepare job.

## Smoke test and re-runs

Single-task example:

```bash
python competitor_task.py \
  --simulation sim2 --method ctf --frac 0.5 --repeat 0 \
  --truth-path artifacts/competitor_truths/sim2.nc \
  --outdir artifacts/competitor_smoke
```

A task that already succeeded is skipped on re-execution based on its
`competitor_complete_*.json`; a failed task can simply be resubmitted at the
corresponding array index. Pass `--force` to `competitor_task.py` to overwrite a
success marker deliberately.

Manual merge and completeness audit:

```bash
python merge_competitor_results.py \
  --root artifacts/competitor_results \
  --n-repeats 50 --fracs 0.5 0.6 0.7 0.8 0.9
```

The merged results are written to `artifacts/competitor_results/merged/{sim1,sim2,sim3}/`.
`competitor_task_audit.csv` reports the expected, successful, failed and missing task
counts per simulation x method. The merge program exits with a non-zero status while any
task is still missing, so that an incomplete run is never mistaken for a final result.

## Scope of the comparator set

The benchmark reported in the manuscript comprises CTF plus the three trajectory fills
listed above. Two further families were considered and are not part of it:

- **Count-likelihood imputation methods** (mbImpute and related negative-binomial or
  zero-inflated count models) require the original read counts. The four cohorts
  analysed here were obtained as relative-abundance profiles without recoverable counts,
  so these methods cannot be applied to the inputs used here.
- **DeepMicroGen and softImpute** were prototyped against this benchmark's mask
  interface but were not run to completion and are not reported. Their reference
  implementations require external runtimes that are not part of this repository's
  environment specification: a legacy TensorFlow 1.x checkout for DeepMicroGen, and a
  separate R installation and bridge script for softImpute. The prototype adapters have
  been removed from this repository rather than shipped in a state that cannot be
  executed or reproduced.
