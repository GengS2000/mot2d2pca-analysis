"""Generate Supplementary Tables S1-S2 for the Sim1/2/3 imputation benchmark.

S1 reports the aggregate results behind Figure 1A-C: every simulation x masking
fraction x method x metric, plus the paired contrasts of MOT-(2D)^2PCA against
each comparator. S2 reports the same quantities resolved by time bin, i.e. the
full grid that Figure 1D shows for a single masking fraction.

Nothing is filtered or summarised beyond what the figure already does: all five
methods, all five masking fractions, all ten time bins, and all three metrics
are reported, whether or not they favour MOT-(2D)^2PCA.

Contrasts are paired within a repeat (the same mask/model seed), tested with a
two-sided Wilcoxon signed-rank test, and Benjamini-Hochberg corrected across all
contrasts within the table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_figure1_sim123_abcd import (  # noqa: E402
    BIN_ORDER,
    FRACTIONS,
    METHOD_LABELS,
    METHODS,
    METRICS,
    OUTPUT_DIR,
    RESULTS_ROOT,
    SIMULATIONS,
    load_results,
)

REFERENCE_METHOD = "penalized_2d2_pca"
# True when a smaller value is the better reconstruction.
METRIC_LOWER_IS_BETTER = {"mse_after": True, "pearson_after": False, "wd_after": True}
METRIC_LABELS = {
    "mse_after": "MSE",
    "pearson_after": "Pearson r",
    "wd_after": "Wasserstein distance",
}
TABLE_DIR = OUTPUT_DIR.parent / "supplementary"


def repeat_column(sim: str) -> str:
    return SIMULATIONS[sim]["repeat_col"]


def descriptive_table(
    data: dict[str, pd.DataFrame], group_cols: list[str]
) -> pd.DataFrame:
    """Per-cell mean/SD/median/IQR plus the direction-aware rank of each method."""
    frames = []
    for sim, df in data.items():
        long_df = df.melt(
            id_vars=group_cols + ["method"],
            value_vars=METRICS,
            var_name="metric",
            value_name="value",
        )
        summary = (
            long_df.groupby(group_cols + ["method", "metric"], as_index=False)["value"]
            .agg(
                n_repeats="count",
                mean="mean",
                sd=lambda x: x.std(ddof=1),
                median="median",
                q1=lambda x: x.quantile(0.25),
                q3=lambda x: x.quantile(0.75),
                minimum="min",
                maximum="max",
            )
        )
        summary.insert(0, "simulation", sim)
        frames.append(summary)
    out = pd.concat(frames, ignore_index=True)

    ranks = []
    rank_keys = ["simulation"] + group_cols + ["metric"]
    for _, group in out.groupby(rank_keys, sort=False):
        metric = group["metric"].iloc[0]
        ascending = METRIC_LOWER_IS_BETTER[metric]
        ranks.append(group["mean"].rank(ascending=ascending, method="min").astype(int))
    out["rank_within_cell"] = pd.concat(ranks).sort_index()

    out.insert(0, "scope", out["simulation"].map(lambda s: SIMULATIONS[s]["scope"]))
    out["method_label"] = out["method"].map(METHOD_LABELS)
    out["metric_label"] = out["metric"].map(METRIC_LABELS)
    out["better_when"] = out["metric"].map(
        lambda m: "lower" if METRIC_LOWER_IS_BETTER[m] else "higher"
    )
    ordered = (
        ["simulation", "scope"]
        + group_cols
        + [
            "metric",
            "metric_label",
            "better_when",
            "method",
            "method_label",
            "n_repeats",
            "mean",
            "sd",
            "median",
            "q1",
            "q3",
            "minimum",
            "maximum",
            "rank_within_cell",
        ]
    )
    return sort_rows(out[ordered], group_cols, ["metric", "method"])


def sort_rows(df: pd.DataFrame, group_cols: list[str], tail_cols: list[str]) -> pd.DataFrame:
    """Sort by simulation, grouping keys, then the tail columns.

    ``bin`` is a string label ("0", "1-4", ...) that does not sort lexically, so
    it is ordered through BIN_ORDER rather than left as text.
    """
    out = df.copy()
    keys = ["simulation"] + list(group_cols) + list(tail_cols)
    if "bin" in out.columns:
        out["_bin_order"] = out["bin"].map({name: i for i, name in enumerate(BIN_ORDER)})
        keys = ["_bin_order" if key == "bin" else key for key in keys]
    out = out.sort_values(keys).reset_index(drop=True)
    return out.drop(columns=["_bin_order"], errors="ignore")


def contrast_table(
    data: dict[str, pd.DataFrame], group_cols: list[str]
) -> pd.DataFrame:
    """Paired MOT-(2D)^2PCA minus comparator, within matched repeats."""
    rows = []
    for sim, df in data.items():
        repeat_col = repeat_column(sim)
        for metric in METRICS:
            lower_better = METRIC_LOWER_IS_BETTER[metric]
            for keys, group in df.groupby(group_cols, sort=False):
                keys = keys if isinstance(keys, tuple) else (keys,)
                wide = group.pivot(index=repeat_col, columns="method", values=metric)
                if REFERENCE_METHOD not in wide.columns:
                    continue
                reference = wide[REFERENCE_METHOD]
                for comparator in METHODS:
                    if comparator == REFERENCE_METHOD:
                        continue
                    paired = pd.concat(
                        [reference, wide[comparator]], axis=1, keys=["ref", "cmp"]
                    ).dropna()
                    diff = (paired["ref"] - paired["cmp"]).to_numpy(dtype=float)
                    n = diff.size
                    sd = float(diff.std(ddof=1))
                    half_width = (
                        float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
                        if n > 1 and sd > 0
                        else 0.0
                    )
                    if np.allclose(diff, 0.0):
                        statistic, p_value = np.nan, 1.0
                    else:
                        statistic, p_value = stats.wilcoxon(
                            diff, alternative="two-sided", zero_method="wilcox"
                        )
                    mean_diff = float(diff.mean())
                    # A negative difference is an improvement when lower is better.
                    mot_better = mean_diff < 0 if lower_better else mean_diff > 0
                    rows.append(
                        dict(
                            simulation=sim,
                            scope=SIMULATIONS[sim]["scope"],
                            **dict(zip(group_cols, keys)),
                            metric=metric,
                            metric_label=METRIC_LABELS[metric],
                            better_when="lower" if lower_better else "higher",
                            comparator=comparator,
                            comparator_label=METHOD_LABELS[comparator],
                            n_pairs=n,
                            mean_mot=float(paired["ref"].mean()),
                            mean_comparator=float(paired["cmp"].mean()),
                            diff_mean=mean_diff,
                            diff_sd=sd,
                            diff_ci95_low=mean_diff - half_width,
                            diff_ci95_high=mean_diff + half_width,
                            diff_median=float(np.median(diff)),
                            wilcoxon_statistic=statistic,
                            p_value=float(p_value),
                            direction_mot="better" if mot_better else "worse",
                        )
                    )
    out = pd.DataFrame(rows)
    out["q_value_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["significant_q05"] = out["q_value_bh"] < 0.05
    return sort_rows(out, group_cols, ["metric", "comparator"])


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(ranked, 1.0)
    return out


def notes_sheet() -> pd.DataFrame:
    notes = [
        ("Source", f"Recomputed from repeat-level result files under {RESULTS_ROOT}."),
        (
            "Table S1",
            "Aggregate results behind Figure 1A-C: simulation x masking fraction x "
            "method x metric. S1a is descriptive; S1b holds the paired contrasts.",
        ),
        (
            "Table S2",
            "The same quantities resolved by time bin (the full grid that Figure 1D "
            "shows for one masking fraction). S2a is descriptive; S2b holds the "
            "paired contrasts.",
        ),
        (
            "Metrics",
            "All metrics are computed in log10-transformed abundance space. Lower MSE "
            "and Wasserstein distance and higher Pearson r indicate better "
            "reconstruction; the better_when column states the direction per row.",
        ),
        (
            "Evaluation scope",
            "Sim1 is evaluated on masked cells (MaskedCells) and Sim2/Sim3 on injected "
            "cells (InjectedCells). These are different evaluation sets, so methods "
            "should be compared within a simulation, not across simulations.",
        ),
        (
            "Repeats",
            "50 repeats per cell. In Sim1 both mask_seed and model_seed vary across "
            "replicates; in Sim2 and Sim3 mask_seed and algo_seed vary across repeats. "
            "Reported variability therefore reflects these seeds only, not variation "
            "over independently designed studies.",
        ),
        (
            "Contrasts",
            "diff = MOT-(2D)^2PCA minus comparator, paired within a repeat. CIs are "
            "t-based (df = n-1). p-values come from a two-sided Wilcoxon signed-rank "
            "test and are Benjamini-Hochberg corrected across all contrasts within "
            "that table (S1b and S2b corrected separately).",
        ),
        (
            "rank_within_cell",
            "Rank of the method's mean among the five methods within one simulation x "
            "fraction (x time bin) x metric cell; 1 is best given the metric direction.",
        ),
        (
            "Generative models",
            "Sim1 truth is generated by the MOT model and Sim2 truth by CP-ALS, so "
            "those two simulations measure recovery under a matched generative model "
            "rather than a neutral comparison. Sim3 is the model-agnostic benchmark.",
        ),
    ]
    return pd.DataFrame(notes, columns=["Item", "Description"])


def main() -> None:
    repeat_data, bin_data = load_results()
    s1a = descriptive_table(repeat_data, ["frac"])
    s1b = contrast_table(repeat_data, ["frac"])
    s2a = descriptive_table(bin_data, ["frac", "bin"])
    s2b = contrast_table(bin_data, ["frac", "bin"])

    expected_s1a = len(SIMULATIONS) * len(FRACTIONS) * len(METHODS) * len(METRICS)
    expected_s2a = expected_s1a * len(BIN_ORDER)
    if len(s1a) != expected_s1a or len(s2a) != expected_s2a:
        raise ValueError(
            f"unexpected table sizes: S1a={len(s1a)} (want {expected_s1a}), "
            f"S2a={len(s2a)} (want {expected_s2a})"
        )
    if not s1a["n_repeats"].eq(50).all() or not s2a["n_repeats"].eq(50).all():
        raise ValueError("every cell must summarise exactly 50 repeats")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    s1a.to_csv(TABLE_DIR / "table_s1a_aggregate_descriptive.csv", index=False)
    s1b.to_csv(TABLE_DIR / "table_s1b_aggregate_contrasts.csv", index=False)
    s2a.to_csv(TABLE_DIR / "table_s2a_time_bin_descriptive.csv", index=False)
    s2b.to_csv(TABLE_DIR / "table_s2b_time_bin_contrasts.csv", index=False)

    workbook = TABLE_DIR / "supplementary_tables_S1_S2.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        notes_sheet().to_excel(writer, sheet_name="Notes", index=False)
        s1a.to_excel(writer, sheet_name="S1a_aggregate", index=False)
        s1b.to_excel(writer, sheet_name="S1b_contrasts", index=False)
        s2a.to_excel(writer, sheet_name="S2a_time_bin", index=False)
        s2b.to_excel(writer, sheet_name="S2b_time_bin_contrasts", index=False)

    print(f"table_dir={TABLE_DIR}")
    print(f"S1a rows={len(s1a)}  S1b rows={len(s1b)}")
    print(f"S2a rows={len(s2a)}  S2b rows={len(s2b)}")
    print(f"workbook={workbook}")


if __name__ == "__main__":
    main()
