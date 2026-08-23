#!/usr/bin/env python3
"""Aggregate and validate participant-bootstrap P2D2PCA outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COHORTS = ("ECAM", "DIABIMMUNE")


def task_id(cohort: str, replicate: int, n_bootstrap: int) -> int:
    return COHORTS.index(cohort) * n_bootstrap + replicate - 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=30)
    args = parser.parse_args()
    root = args.output_root.resolve()
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    metric_rows = []
    taxon_tables = []
    missing_ids = []
    for cohort in COHORTS:
        for replicate in range(1, args.n_bootstrap + 1):
            run_dir = root / "stage1" / cohort / f"bootstrap_{replicate:03d}"
            success_path = run_dir / "SUCCESS.json"
            required = [
                run_dir / "tensor_base_raw.nc",
                run_dir / "tensor_p2d2pca_imputed.nc",
                run_dir / "participant_order.csv",
                run_dir / "subject_groups_long.csv",
                run_dir / "taxon_imputation_summary.csv.gz",
            ]
            complete = success_path.is_file() and all(path.is_file() and path.stat().st_size > 0 for path in required)
            reason = ""
            metrics = {}
            if complete:
                try:
                    metrics = json.loads(success_path.read_text(encoding="utf-8"))
                    complete = metrics.get("status") == "completed"
                    if not complete:
                        reason = "SUCCESS marker status is not completed"
                except Exception as exc:  # pragma: no cover - server corruption path
                    complete = False
                    reason = f"invalid SUCCESS.json: {exc}"
            else:
                reason = "missing one or more required files"
            tid = task_id(cohort, replicate, args.n_bootstrap)
            manifest_rows.append(
                {
                    "task_id": tid,
                    "cohort": cohort,
                    "bootstrap_replicate": replicate,
                    "complete": complete,
                    "reason": reason,
                    "run_dir": str(run_dir),
                }
            )
            if not complete:
                missing_ids.append(tid)
                continue
            metric_rows.append(metrics)
            table = pd.read_csv(run_dir / "taxon_imputation_summary.csv.gz")
            taxon_tables.append(table)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(aggregate_dir / "stage1_completeness_report.csv", index=False)
    (aggregate_dir / "missing_stage1_task_ids.txt").write_text(
        "".join(f"{value}\n" for value in missing_ids), encoding="utf-8"
    )
    metrics_df = pd.json_normalize(metric_rows)
    metrics_df.to_csv(aggregate_dir / "imputation_run_metrics.csv", index=False)

    if not metrics_df.empty:
        numeric = [
            "n_unique_source_participants", "duplicate_draw_fraction", "replace_fraction",
            "completed_finite_fraction", "unchanged_max_abs_difference", "mean_imputation_delta",
            "median_abs_imputation_delta", "rmse_from_placeholder", "elapsed_seconds",
        ]
        summary_rows = []
        for cohort, group in metrics_df.groupby("cohort", sort=True):
            for column in numeric:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                summary_rows.append(
                    {
                        "cohort": cohort,
                        "metric": column,
                        "n": len(values),
                        "mean": values.mean(),
                        "sd": values.std(ddof=1),
                        "median": values.median(),
                        "q025": values.quantile(0.025),
                        "q975": values.quantile(0.975),
                        "min": values.min(),
                        "max": values.max(),
                    }
                )
        pd.DataFrame(summary_rows).to_csv(aggregate_dir / "imputation_metric_summary.csv", index=False)

    if taxon_tables:
        taxon = pd.concat(taxon_tables, ignore_index=True)
        rows = []
        for (cohort, feature), group in taxon.groupby(["cohort", "taxon"], sort=False):
            completed = pd.to_numeric(group["mean_completed_log_abundance"], errors="coerce")
            delta = pd.to_numeric(group["mean_imputation_delta"], errors="coerce")
            rows.append(
                {
                    "cohort": cohort,
                    "taxon": feature,
                    "n_successful_bootstraps": int(group["bootstrap_replicate"].nunique()),
                    "completed_mean": completed.mean(),
                    "completed_sd": completed.std(ddof=1),
                    "completed_cv_abs": completed.std(ddof=1) / abs(completed.mean())
                    if np.isfinite(completed.mean()) and completed.mean() != 0 else np.nan,
                    "imputation_delta_mean": delta.mean(),
                    "imputation_delta_sd": delta.std(ddof=1),
                    "imputation_delta_q025": delta.quantile(0.025),
                    "imputation_delta_q975": delta.quantile(0.975),
                }
            )
        pd.DataFrame(rows).to_csv(aggregate_dir / "taxon_imputation_stability.csv.gz", index=False, compression="gzip")

    config = {
        "n_bootstrap": args.n_bootstrap,
        "expected_tasks": len(COHORTS) * args.n_bootstrap,
        "completed_tasks": int(manifest["complete"].sum()),
        "missing_tasks": len(missing_ids),
        "complete": len(missing_ids) == 0,
    }
    (aggregate_dir / "stage1_aggregate_status.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))
    return 0 if not missing_ids else 2


if __name__ == "__main__":
    raise SystemExit(main())
