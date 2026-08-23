from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


METHOD = "penalized_2d2_pca"
BASELINE = "no_imputation"


def exact_one_sided_sign_p(wins: int, n: int) -> float:
    """P(X >= wins), X ~ Binomial(n, 0.5)."""
    if n <= 0:
        return float("nan")
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) * 0.5**n)


def bootstrap_ci(values: np.ndarray, seed: int = 20260705, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def holm_adjust(p_values: list[float]) -> list[float]:
    finite = [(i, p) for i, p in enumerate(p_values) if np.isfinite(p)]
    adjusted = [float("nan")] * len(p_values)
    if not finite:
        return adjusted
    ordered = sorted(finite, key=lambda x: x[1])
    m = len(ordered)
    running = 0.0
    for rank, (idx, p) in enumerate(ordered, start=1):
        val = min(1.0, (m - rank + 1) * p)
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def paired_summary(
    df: pd.DataFrame,
    metric: str,
    lower_is_better: bool,
    unit_cols: list[str],
    source: str,
    mode: str,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    grouped = (
        df.groupby(unit_cols + ["method"], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot_table(index=unit_cols, columns="method", values=metric, aggfunc="mean")
    if METHOD not in pivot.columns or BASELINE not in pivot.columns:
        empty = pd.DataFrame()
        return {
            "source": source,
            "replacement_mode": mode,
            "metric": metric,
            "unit": " x ".join(unit_cols),
            "n_pairs": 0,
            "error": "missing method or baseline",
        }, empty

    pairs = pivot[[BASELINE, METHOD]].dropna().copy()
    pairs = pairs.rename(columns={BASELINE: "baseline", METHOD: "method"})
    if lower_is_better:
        pairs["delta"] = pairs["baseline"] - pairs["method"]
    else:
        pairs["delta"] = pairs["method"] - pairs["baseline"]
    pairs["relative_change_pct"] = 100.0 * pairs["delta"] / (pairs["baseline"].abs() + 1e-12)
    pairs["improved"] = pairs["delta"] > 0
    n_nonzero = int((pairs["delta"] != 0).sum())
    wins = int(pairs["improved"].sum())
    p = exact_one_sided_sign_p(wins, n_nonzero)
    ci_lo, ci_hi = bootstrap_ci(pairs["delta"].to_numpy())
    rel_ci_lo, rel_ci_hi = bootstrap_ci(pairs["relative_change_pct"].to_numpy())

    out = {
        "source": source,
        "replacement_mode": mode,
        "metric": metric,
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
        "unit": " x ".join(unit_cols),
        "n_pairs": int(len(pairs)),
        "wins": wins,
        "win_rate": wins / len(pairs) if len(pairs) else float("nan"),
        "one_sided_exact_sign_p": p,
        "baseline_mean": float(pairs["baseline"].mean()),
        "method_mean": float(pairs["method"].mean()),
        "mean_delta": float(pairs["delta"].mean()),
        "median_delta": float(pairs["delta"].median()),
        "mean_delta_ci95_low": ci_lo,
        "mean_delta_ci95_high": ci_hi,
        "mean_relative_change_pct": float(pairs["relative_change_pct"].mean()),
        "median_relative_change_pct": float(pairs["relative_change_pct"].median()),
        "mean_relative_change_pct_ci95_low": rel_ci_lo,
        "mean_relative_change_pct_ci95_high": rel_ci_hi,
    }
    return out, pairs.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute Experiment 3 significance using mechanism x fraction x repeat "
            "as paired units instead of collapsing repeats into 6 scenario means."
        )
    )
    parser.add_argument("--expdir", type=Path, default=Path("outputs/experiment3_trajectory"))
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or (args.expdir / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    subject_path = args.expdir / "subject_level_metrics.csv"
    cluster_path = args.expdir / "taxon_clustering_metrics.csv"
    subject = pd.read_csv(subject_path)
    clusters = pd.read_csv(cluster_path)

    rows: list[dict[str, float | str]] = []
    pair_frames: list[pd.DataFrame] = []
    repeat_unit = ["mechanism", "frac", "repeat", "replacement_mode"]
    scenario_unit = ["mechanism", "frac", "replacement_mode"]

    specs = [
        (subject, "subject_level_metrics", "subject_trajectory_rmse", True),
        (clusters, "taxon_clustering_metrics", "ari_imputed", False),
        (clusters, "taxon_clustering_metrics", "nmi_imputed", False),
    ]
    for df, source, metric, lower_is_better in specs:
        for mode in sorted(df["replacement_mode"].dropna().unique()):
            sub = df[df["replacement_mode"].eq(mode)]
            for unit_name, unit_cols in [
                ("repeat_level_primary", repeat_unit),
                ("scenario_level_legacy", scenario_unit),
            ]:
                row, pairs = paired_summary(sub, metric, lower_is_better, unit_cols, source, mode)
                row["analysis"] = unit_name
                rows.append(row)
                if not pairs.empty:
                    pairs.insert(0, "analysis", unit_name)
                    pairs.insert(1, "source", source)
                    pairs.insert(2, "metric", metric)
                    pair_frames.append(pairs)

    stats = pd.DataFrame(rows)
    primary = stats["analysis"].eq("repeat_level_primary")
    stats.loc[primary, "holm_p_within_primary_tests"] = holm_adjust(
        stats.loc[primary, "one_sided_exact_sign_p"].astype(float).tolist()
    )

    stats_path = outdir / "experiment3_repeat_level_inference_stats.csv"
    pairs_path = outdir / "experiment3_repeat_level_paired_units.csv"
    stats.to_csv(stats_path, index=False)
    pd.concat(pair_frames, ignore_index=True).to_csv(pairs_path, index=False)

    print(f"wrote {stats_path}")
    print(f"wrote {pairs_path}")
    print(
        stats[
            [
                "analysis",
                "source",
                "replacement_mode",
                "metric",
                "unit",
                "n_pairs",
                "wins",
                "win_rate",
                "one_sided_exact_sign_p",
                "holm_p_within_primary_tests",
                "mean_relative_change_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
