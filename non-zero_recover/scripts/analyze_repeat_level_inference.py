from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_UNIT_COLS = ["mechanism", "frac", "repeat", "replacement_mode"]


@dataclass(frozen=True)
class MetricSpec:
    experiment: str
    source: str
    filename: str
    metric: str
    lower_is_better: bool
    extra_unit_cols: tuple[str, ...] = ()


SPECS = [
    MetricSpec("experiment1_zero_recovery", "metrics_long", "metrics_long.csv", "mae", True),
    MetricSpec("experiment1_zero_recovery", "metrics_long", "metrics_long.csv", "rmse", True),
    MetricSpec("experiment1_zero_recovery", "metrics_long", "metrics_long.csv", "wasserstein", True),
    MetricSpec("experiment1_zero_recovery", "metrics_long", "metrics_long.csv", "pearson", False),
    MetricSpec("experiment1_zero_recovery", "metrics_long", "metrics_long.csv", "spearman", False),
    MetricSpec("experiment1_zero_recovery", "metrics_long", "metrics_long.csv", "trajectory_rmse_affected", True),
    MetricSpec("experiment1_zero_recovery", "metrics_long", "metrics_long.csv", "lag1_autocorr_abs_error", True),
    MetricSpec("experiment2_distribution", "taxon_distribution", "distribution_metrics_long.csv", "wasserstein_masked", True),
    MetricSpec("experiment2_distribution", "taxon_distribution", "distribution_metrics_long.csv", "js_full", True),
    MetricSpec("experiment2_distribution", "taxon_distribution", "distribution_metrics_long.csv", "taxon_mean_rank_spearman", False),
    MetricSpec("experiment2_distribution", "global_distribution", "raw_nonzero_vs_imputed_at_masked_distribution_metrics.csv", "raw_nonzero_vs_imputed_at_masked_wasserstein", True),
    MetricSpec("experiment2_distribution", "global_distribution", "raw_nonzero_vs_imputed_at_masked_distribution_metrics.csv", "raw_nonzero_vs_imputed_at_masked_js", True),
    MetricSpec("experiment2_distribution", "global_distribution", "raw_nonzero_vs_imputed_at_masked_distribution_metrics.csv", "masked_truth_vs_imputed_at_masked_wasserstein", True),
    MetricSpec("experiment2_distribution", "global_distribution", "raw_nonzero_vs_imputed_at_masked_distribution_metrics.csv", "masked_truth_vs_imputed_at_masked_js", True),
    MetricSpec("experiment3_trajectory", "trajectory_long", "trajectory_metrics_long.csv", "trajectory_rmse", True),
    MetricSpec("experiment3_trajectory", "trajectory_long", "trajectory_metrics_long.csv", "trajectory_pearson", False),
    MetricSpec("experiment3_trajectory", "trajectory_long", "trajectory_metrics_long.csv", "trajectory_spearman", False),
    MetricSpec("experiment3_trajectory", "trajectory_long", "trajectory_metrics_long.csv", "lag1_error_imputed", True),
    MetricSpec("experiment3_trajectory", "trajectory_long", "trajectory_metrics_long.csv", "second_difference_mae_imputed", True),
    MetricSpec("experiment3_trajectory", "trajectory_long", "trajectory_metrics_long.csv", "dtw_imputed", True),
    MetricSpec("experiment3_trajectory", "trajectory_repeat_level", "trajectory_metrics_repeat_level.csv", "trajectory_rmse", True),
    MetricSpec("experiment3_trajectory", "trajectory_repeat_level", "trajectory_metrics_repeat_level.csv", "trajectory_pearson", False),
    MetricSpec("experiment3_trajectory", "trajectory_repeat_level", "trajectory_metrics_repeat_level.csv", "trajectory_spearman", False),
    MetricSpec("experiment3_trajectory", "trajectory_repeat_level", "trajectory_metrics_repeat_level.csv", "lag1_error_imputed", True),
    MetricSpec("experiment3_trajectory", "trajectory_repeat_level", "trajectory_metrics_repeat_level.csv", "second_difference_mae_imputed", True),
    MetricSpec("experiment3_trajectory", "trajectory_repeat_level", "trajectory_metrics_repeat_level.csv", "dtw_imputed", True),
    MetricSpec("experiment3_trajectory", "subject_level", "subject_level_metrics.csv", "subject_trajectory_rmse", True),
    MetricSpec("experiment3_trajectory", "taxon_clustering", "taxon_clustering_metrics.csv", "ari_imputed", False),
    MetricSpec("experiment3_trajectory", "taxon_clustering", "taxon_clustering_metrics.csv", "nmi_imputed", False),
    MetricSpec("experiment4_dynamic_correlation", "static_correlation", "static_correlation_metrics.csv", "static_pearson", False),
    MetricSpec("experiment4_dynamic_correlation", "static_correlation", "static_correlation_metrics.csv", "static_frobenius", True),
    MetricSpec("experiment4_dynamic_correlation", "static_correlation", "static_correlation_metrics.csv", "nonzero_static_pearson", False),
    MetricSpec("experiment4_dynamic_correlation", "static_correlation", "static_correlation_metrics.csv", "nonzero_static_frobenius", True),
    MetricSpec("experiment4_dynamic_correlation", "lagged_correlation", "lagged_correlation_metrics.csv", "lagged_pearson", False),
    MetricSpec("experiment4_dynamic_correlation", "lagged_correlation", "lagged_correlation_metrics.csv", "lagged_frobenius", True),
    MetricSpec("experiment4_dynamic_correlation", "network_edges", "network_edge_overlap.csv", "topk_precision", False, ("k",)),
    MetricSpec("experiment4_dynamic_correlation", "network_edges", "network_edge_overlap.csv", "jaccard", False, ("k",)),
    MetricSpec("experiment4_dynamic_correlation", "hub_taxa", "hub_taxa_overlap.csv", "degree_distribution_pearson", False),
    MetricSpec("experiment4_dynamic_correlation", "hub_taxa", "hub_taxa_overlap.csv", "hub_jaccard", False),
]


def exact_one_sided_sign_p(wins: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) * 0.5**n)


def normalize_replacement_mode(mode: str) -> str:
    mode = str(mode)
    if mode in {"known_mask", "oracle_mask", "provided_mask", "provided_imputation_mask"}:
        return "known_mask"
    return mode


def bootstrap_ci(values: np.ndarray, seed: int = 20260705, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def holm_adjust(p_values: list[float]) -> list[float]:
    finite = [(i, p) for i, p in enumerate(p_values) if np.isfinite(p)]
    adjusted = [float("nan")] * len(p_values)
    ordered = sorted(finite, key=lambda x: x[1])
    m = len(ordered)
    running = 0.0
    for rank, (idx, p) in enumerate(ordered, start=1):
        running = max(running, min(1.0, (m - rank + 1) * p))
        adjusted[idx] = running
    return adjusted


def read_source(root: Path, spec: MetricSpec, cache: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame | None:
    key = (spec.experiment, spec.filename)
    if key not in cache:
        path = root / spec.experiment / spec.filename
        if not path.exists():
            return None
        try:
            cache[key] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return None
    return cache[key]


def scenario_table(df: pd.DataFrame, spec: MetricSpec, replacement_mode: str) -> pd.DataFrame:
    needed = [*BASE_UNIT_COLS, *spec.extra_unit_cols, "method", spec.metric]
    if any(col not in df.columns for col in needed):
        return pd.DataFrame()
    sub = df[needed].copy()
    sub["replacement_mode"] = sub["replacement_mode"].map(normalize_replacement_mode)
    replacement_mode = normalize_replacement_mode(replacement_mode)
    sub = sub.loc[sub["replacement_mode"].eq(replacement_mode)].copy()
    sub[spec.metric] = pd.to_numeric(sub[spec.metric], errors="coerce")
    sub = sub.dropna(subset=[spec.metric, "method"])
    if sub.empty:
        return pd.DataFrame()
    unit_cols = [*BASE_UNIT_COLS, *spec.extra_unit_cols]
    return (
        sub.groupby(unit_cols + ["method"], dropna=False)[spec.metric]
        .mean()
        .reset_index()
    )


def paired_stats(table: pd.DataFrame, spec: MetricSpec, baseline: str, method: str) -> dict[str, object] | None:
    unit_cols = [*BASE_UNIT_COLS, *spec.extra_unit_cols]
    pivot = table.pivot_table(index=unit_cols, columns="method", values=spec.metric, aggfunc="mean")
    if baseline not in pivot.columns or method not in pivot.columns:
        return None
    pairs = pivot[[baseline, method]].dropna().rename(columns={baseline: "baseline", method: "method"})
    if pairs.empty:
        return None
    if spec.lower_is_better:
        delta = pairs["baseline"] - pairs["method"]
    else:
        delta = pairs["method"] - pairs["baseline"]
    rel = 100.0 * delta / (pairs["baseline"].abs() + 1e-12)
    wins = int((delta > 0).sum())
    n_nonzero = int((delta != 0).sum())
    delta_lo, delta_hi = bootstrap_ci(delta.to_numpy())
    rel_lo, rel_hi = bootstrap_ci(rel.to_numpy())
    return {
        "experiment": spec.experiment,
        "source": spec.source,
        "metric": spec.metric,
        "replacement_mode": str(table["replacement_mode"].iloc[0]) if "replacement_mode" in table.columns and not table.empty else "",
        "method": method,
        "baseline": baseline,
        "direction": "lower_is_better" if spec.lower_is_better else "higher_is_better",
        "unit": " x ".join(unit_cols),
        "n_pairs": int(len(pairs)),
        "wins": wins,
        "win_rate": wins / len(pairs),
        "one_sided_exact_sign_p": exact_one_sided_sign_p(wins, n_nonzero),
        "baseline_mean": float(pairs["baseline"].mean()),
        "method_mean": float(pairs["method"].mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "mean_delta_ci95_low": delta_lo,
        "mean_delta_ci95_high": delta_hi,
        "mean_relative_change_pct": float(rel.mean()),
        "median_relative_change_pct": float(rel.median()),
        "mean_relative_change_pct_ci95_low": rel_lo,
        "mean_relative_change_pct_ci95_high": rel_hi,
    }


def ranking_stats(table: pd.DataFrame, spec: MetricSpec) -> list[dict[str, object]]:
    unit_cols = [*BASE_UNIT_COLS, *spec.extra_unit_cols]
    pivot = table.pivot_table(index=unit_cols, columns="method", values=spec.metric, aggfunc="mean")
    rows = []
    for method in pivot.columns:
        ranks = []
        best = 0
        valid = 0
        for _, row in pivot.iterrows():
            vals = row.dropna()
            if method not in vals.index or vals.size < 2:
                continue
            rank = vals.rank(ascending=spec.lower_is_better, method="min")[method]
            ranks.append(float(rank))
            best += int(rank == 1)
            valid += 1
        if valid:
            rows.append(
                {
                    "experiment": spec.experiment,
                    "source": spec.source,
                    "metric": spec.metric,
                    "replacement_mode": str(table["replacement_mode"].iloc[0]) if "replacement_mode" in table.columns and not table.empty else "",
                    "method": method,
                    "direction": "lower_is_better" if spec.lower_is_better else "higher_is_better",
                    "unit": " x ".join(unit_cols),
                    "n_units": valid,
                    "average_rank": float(np.mean(ranks)),
                    "best_count": int(best),
                    "best_fraction": best / valid,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat-level paired inference and method ranking for known-mask experiments.")
    parser.add_argument("--root", type=Path, default=Path("outputs_known_mask"))
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--replacement-mode", default="known_mask")
    parser.add_argument("--baseline", default="no_imputation")
    args = parser.parse_args()

    outdir = args.outdir or (args.root / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    paired_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for spec in SPECS:
        df = read_source(args.root, spec, cache)
        path = args.root / spec.experiment / spec.filename
        if df is None:
            manifest_rows.append({"experiment": spec.experiment, "source": spec.source, "filename": str(path), "metric": spec.metric, "status": "missing_or_empty"})
            continue
        table = scenario_table(df, spec, args.replacement_mode)
        if table.empty:
            manifest_rows.append({"experiment": spec.experiment, "source": spec.source, "filename": str(path), "metric": spec.metric, "status": "missing_columns_or_no_rows"})
            continue
        methods = sorted([m for m in table["method"].dropna().unique() if m != args.baseline])
        for method in methods:
            row = paired_stats(table, spec, args.baseline, method)
            if row is not None:
                paired_rows.append(row)
        ranking_rows.extend(ranking_stats(table, spec))
        manifest_rows.append({"experiment": spec.experiment, "source": spec.source, "filename": str(path), "metric": spec.metric, "status": "ok", "rows": len(table)})

    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["holm_p_within_all_tests"] = holm_adjust(paired["one_sided_exact_sign_p"].astype(float).tolist())
        paired.sort_values(["experiment", "source", "metric", "method"]).to_csv(outdir / "repeat_level_inference_stats.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "repeat_level_inference_stats.csv", index=False)

    rankings = pd.DataFrame(ranking_rows)
    if not rankings.empty:
        rankings.sort_values(["experiment", "source", "metric", "average_rank", "method"]).to_csv(outdir / "method_rankings.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "method_rankings.csv", index=False)

    pd.DataFrame(manifest_rows).to_csv(outdir / "analysis_manifest.csv", index=False)
    print(f"[analysis] wrote {outdir / 'repeat_level_inference_stats.csv'}")
    print(f"[analysis] wrote {outdir / 'method_rankings.csv'}")
    print(f"[analysis] wrote {outdir / 'analysis_manifest.csv'}")


if __name__ == "__main__":
    main()
