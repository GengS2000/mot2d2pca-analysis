from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = ROOT / "outputs" / "experiment2_distribution"
LOCAL_DEPS = DEFAULT_EXPERIMENT_DIR / "_plot_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import seaborn as sns


METHOD_ORDER = ["no_imputation", "penalized_2d2_pca", "mean", "spline"]
METHOD_LABELS = {
    "no_imputation": "No imputation",
    "penalized_2d2_pca": "2D2-PCA",
    "mean": "Mean",
    "spline": "Spline",
}
METHOD_PALETTE = {
    "No imputation": "#6B7280",
    "2D2-PCA": "#0072B2",
    "Mean": "#E69F00",
    "Spline": "#009E73",
}
MECHANISM_ORDER = ["random_entry", "low_abundance_biased", "consecutive_time_window_dropout"]
MECHANISM_LABELS = {
    "random_entry": "Random",
    "low_abundance_biased": "Low-abund.",
    "consecutive_time_window_dropout": "Time-window",
}
REPLACEMENT_LABELS = {
    "detector_longitudinal_neighbor": "Detector-defined zeros",
    "oracle_mask": "Oracle artificial mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-quality figures for experiment 2 distribution restoration."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "distribution_metrics_long.csv",
        help="Path to distribution_metrics_long.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "figures",
        help="Directory for exported figures.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "config.json",
        help="Experiment config used to reconstruct direct non-zero abundance distributions.",
    )
    parser.add_argument(
        "--direct-repeat",
        type=int,
        default=0,
        help="Repeat index used for the direct original-vs-imputed distribution overlay.",
    )
    parser.add_argument(
        "--skip-direct-distribution",
        action="store_true",
        help="Skip reconstruction of the direct non-zero original abundance distribution overlay.",
    )
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "grid.linewidth": 0.35,
        }
    )


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[np.isfinite(out["wasserstein_masked"])].copy()
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    out["method_label"] = out["method"].astype(str).map(METHOD_LABELS)
    out["method_label"] = pd.Categorical(
        out["method_label"], [METHOD_LABELS[m] for m in METHOD_ORDER], ordered=True
    )
    out["mechanism"] = pd.Categorical(out["mechanism"], MECHANISM_ORDER, ordered=True)
    out["mechanism_label"] = out["mechanism"].astype(str).map(MECHANISM_LABELS)
    out["frac_label"] = (out["frac"] * 100).round().astype(int).astype(str) + "%"
    out["condition"] = out["mechanism_label"] + "\n" + out["frac_label"]
    condition_order = [
        f"{MECHANISM_LABELS[m]}\n{int(frac * 100)}%"
        for m in MECHANISM_ORDER
        for frac in sorted(out["frac"].dropna().unique())
    ]
    out["condition"] = pd.Categorical(out["condition"], condition_order, ordered=True)
    out["replacement_label"] = out["replacement_mode"].map(REPLACEMENT_LABELS)
    return out


def paired_improvement(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["mechanism", "frac", "repeat", "replacement_mode", "taxon"]
    base = (
        df[df["method"].astype(str) == "no_imputation"][keys + ["wasserstein_masked"]]
        .rename(columns={"wasserstein_masked": "wasserstein_no_imputation"})
        .copy()
    )
    merged = df.merge(base, on=keys, how="left")
    merged["absolute_improvement"] = (
        merged["wasserstein_no_imputation"] - merged["wasserstein_masked"]
    )
    merged["relative_recovery_pct"] = np.where(
        merged["wasserstein_no_imputation"] > 0,
        100.0 * merged["absolute_improvement"] / merged["wasserstein_no_imputation"],
        np.nan,
    )
    return merged


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return out
    m = float(len(valid))
    ranks = np.arange(1, len(valid) + 1, dtype=float)
    adjusted = (valid.to_numpy() * m / ranks)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[valid.index] = np.clip(adjusted, 0, 1)
    return out


def make_stats_table(df_with_improvement: pd.DataFrame, outpath: Path) -> pd.DataFrame:
    rows = []
    group_cols = ["replacement_mode", "mechanism", "frac", "method"]
    for key, g in df_with_improvement.groupby(group_cols, observed=True, dropna=False):
        replacement_mode, mechanism, frac, method = key
        g = g[np.isfinite(g["wasserstein_masked"])]
        row = {
            "replacement_mode": replacement_mode,
            "mechanism": mechanism,
            "frac": float(frac),
            "method": str(method),
            "n_taxon_runs": int(len(g)),
            "wasserstein_mean": float(g["wasserstein_masked"].mean()),
            "wasserstein_median": float(g["wasserstein_masked"].median()),
            "wasserstein_q25": float(g["wasserstein_masked"].quantile(0.25)),
            "wasserstein_q75": float(g["wasserstein_masked"].quantile(0.75)),
        }
        if str(method) != "no_imputation":
            paired = g[
                np.isfinite(g["wasserstein_no_imputation"])
                & np.isfinite(g["relative_recovery_pct"])
            ]
            row.update(
                {
                    "mean_distance_reduction_vs_no_imputation": float(
                        paired["absolute_improvement"].mean()
                    ),
                    "median_distance_reduction_vs_no_imputation": float(
                        paired["absolute_improvement"].median()
                    ),
                    "mean_relative_recovery_pct": float(
                        paired["relative_recovery_pct"].mean()
                    ),
                    "median_relative_recovery_pct": float(
                        paired["relative_recovery_pct"].median()
                    ),
                    "pct_taxon_runs_improved": float(
                        100.0 * (paired["absolute_improvement"] > 0).mean()
                    ),
                }
            )
            try:
                wilcoxon = stats.wilcoxon(
                    paired["wasserstein_masked"],
                    paired["wasserstein_no_imputation"],
                    alternative="less",
                    zero_method="wilcox",
                )
                row["wilcoxon_p_less_than_no_imputation"] = float(wilcoxon.pvalue)
            except ValueError:
                row["wilcoxon_p_less_than_no_imputation"] = np.nan
        else:
            row.update(
                {
                    "mean_distance_reduction_vs_no_imputation": np.nan,
                    "median_distance_reduction_vs_no_imputation": np.nan,
                    "mean_relative_recovery_pct": np.nan,
                    "median_relative_recovery_pct": np.nan,
                    "pct_taxon_runs_improved": np.nan,
                    "wilcoxon_p_less_than_no_imputation": np.nan,
                }
            )
        rows.append(row)

    stats_df = pd.DataFrame(rows)
    stats_df["wilcoxon_fdr_bh"] = benjamini_hochberg(
        stats_df["wilcoxon_p_less_than_no_imputation"]
    )
    stats_df.to_csv(outpath, index=False)
    return stats_df


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def save_figure(fig: plt.Figure, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(outbase.with_suffix(suffix), bbox_inches="tight")
    fig.savefig(
        outbase.with_suffix(".tiff"),
        bbox_inches="tight",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def draw_main_figure(df: pd.DataFrame, outbase: Path) -> None:
    detector = df[df["replacement_mode"] == "detector_longitudinal_neighbor"].copy()
    detector = detector.sort_values(["mechanism", "frac", "method"])

    fig = plt.figure(figsize=(7.4, 6.8))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.2, 1.0],
        left=0.075,
        right=0.985,
        top=0.89,
        bottom=0.095,
        hspace=0.46,
        wspace=0.33,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    sns.boxplot(
        data=detector,
        x="condition",
        y="wasserstein_masked",
        hue="method_label",
        hue_order=[METHOD_LABELS[m] for m in METHOD_ORDER],
        palette=METHOD_PALETTE,
        showfliers=False,
        linewidth=0.65,
        width=0.72,
        ax=ax_a,
    )
    ax_a.set_title("Distribution recovery at artificially masked non-zero abundances")
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Masked Wasserstein distance\n(lower is closer to original)")
    ax_a.set_ylim(bottom=0)
    ax_a.grid(axis="x", visible=False)
    ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=4,
        frameon=False,
        title=None,
        handlelength=1.3,
    )
    for xpos in [1.5, 3.5]:
        ax_a.axvline(xpos, color="#D1D5DB", lw=0.6, zorder=0)
    panel_label(ax_a, "A")

    imputed = detector[detector["method"].astype(str) != "no_imputation"].copy()
    rep_level = (
        imputed.groupby(["condition", "method_label", "repeat"], observed=True)[
            "relative_recovery_pct"
        ]
        .mean()
        .reset_index()
    )
    sns.pointplot(
        data=rep_level,
        x="condition",
        y="relative_recovery_pct",
        hue="method_label",
        hue_order=[METHOD_LABELS[m] for m in METHOD_ORDER[1:]],
        palette=METHOD_PALETTE,
        dodge=0.36,
        errorbar=("ci", 95),
        markers="o",
        linestyles="none",
        capsize=0.12,
        err_kws={"linewidth": 0.8},
        ax=ax_b,
    )
    ax_b.axhline(0, color="#111827", lw=0.7, ls="--")
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Distance reduction vs no imputation (%)")
    ax_b.set_title("Paired restoration gain")
    ax_b.tick_params(axis="x", rotation=45)
    ax_b.grid(axis="x", visible=False)
    ax_b.legend(loc="lower right", frameon=False, title=None)
    panel_label(ax_b, "B")

    moment = detector[detector["method"].astype(str) == "penalized_2d2_pca"].copy()
    moment = (
        moment.groupby("taxon", observed=True)
        .agg(original_mean=("original_mean", "mean"), imputed_mean=("imputed_mean", "mean"))
        .reset_index()
    )
    x = moment["original_mean"].to_numpy()
    y = moment["imputed_mean"].to_numpy()
    r = float(np.corrcoef(x, y)[0, 1])
    mae = float(np.mean(np.abs(x - y)))
    lim_min = float(min(np.nanmin(x), np.nanmin(y)) - 0.05)
    lim_max = float(max(np.nanmax(x), np.nanmax(y)) + 0.05)
    ax_c.scatter(
        x,
        y,
        s=18,
        color=METHOD_PALETTE["2D2-PCA"],
        alpha=0.72,
        edgecolors="white",
        linewidths=0.25,
    )
    ax_c.plot([lim_min, lim_max], [lim_min, lim_max], color="#111827", lw=0.8, ls="--")
    ax_c.set_xlim(lim_min, lim_max)
    ax_c.set_ylim(lim_min, lim_max)
    ax_c.set_xlabel("Original taxon mean log10 abundance")
    ax_c.set_ylabel("Imputed taxon mean log10 abundance")
    ax_c.set_title("Taxon-level abundance moments")
    ax_c.text(
        0.04,
        0.96,
        f"2D2-PCA, detector mode\nr = {r:.3f}; MAE = {mae:.3f}",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#E5E7EB", "boxstyle": "round,pad=0.25"},
    )
    panel_label(ax_c, "C")

    save_figure(fig, outbase)
    plt.close(fig)


def draw_all_modes_figure(df: pd.DataFrame, outbase: Path) -> None:
    plot_df = df.sort_values(["replacement_mode", "frac", "mechanism", "method"]).copy()
    g = sns.catplot(
        data=plot_df,
        x="mechanism_label",
        y="wasserstein_masked",
        hue="method_label",
        col="frac_label",
        row="replacement_label",
        kind="box",
        hue_order=[METHOD_LABELS[m] for m in METHOD_ORDER],
        palette=METHOD_PALETTE,
        showfliers=False,
        linewidth=0.55,
        width=0.68,
        height=2.35,
        aspect=1.35,
        margin_titles=True,
        sharey=False,
        legend=False,
    )
    g.set_axis_labels("", "Masked Wasserstein distance")
    g.set_titles(row_template="{row_name}", col_template="{col_name} missing")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(bottom=0)
    handles = [
        Line2D([0], [0], color=METHOD_PALETTE[METHOD_LABELS[m]], lw=5, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    g.figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
    )
    g.figure.subplots_adjust(top=0.88, bottom=0.13, left=0.08, right=0.98, hspace=0.28)
    save_figure(g.figure, outbase)
    plt.close(g.figure)


def draw_ecdf_figure(df: pd.DataFrame, outbase: Path) -> None:
    detector = df[df["replacement_mode"] == "detector_longitudinal_neighbor"].copy()
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for method in METHOD_ORDER:
        vals = detector.loc[detector["method"].astype(str) == method, "wasserstein_masked"]
        vals = np.sort(vals.dropna().to_numpy())
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(
            vals,
            y,
            lw=1.4 if method != "no_imputation" else 1.1,
            color=METHOD_PALETTE[METHOD_LABELS[method]],
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Masked Wasserstein distance")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_title("Detector mode: imputed distributions shift toward lower distance")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(fig=fig)
    save_figure(fig, outbase)
    plt.close(fig)


def load_experiment_config(config_path: Path) -> SimpleNamespace:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {
        "tax_path",
        "meta_path",
        "outdir",
        "eps",
        "min_pos_cells",
        "min_participants",
        "rank_d",
        "rank_q",
        "lam",
        "sigma",
        "T0",
        "T",
        "fracs",
        "repeats",
        "seeds",
        "methods",
        "mechanisms",
        "replacement_modes",
        "max_participants",
        "max_taxa",
    }
    return SimpleNamespace(**{k: cfg.get(k) for k in allowed})


def reconstruct_direct_distribution(config_path: Path, repeat: int) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT))
    from src.experiment_runner import load_data_from_args, pca_params_from_args, parse_csv_list, seeds_from_args
    from src.imputation_wrappers import impute_method
    from src.masking import detect_nonbiological_zeros, make_masked_dataset
    from src.penalized_2d2_pca import build_observed_mask

    args = load_experiment_config(config_path)
    exp_dir = Path(args.outdir) / "experiment2_distribution"
    da_log, _da_raw, observed_positive_mask, _meta_info, zero_value, _methods, mechanisms, fracs, _replacement_modes = load_data_from_args(args, exp_dir)
    pca_params = pca_params_from_args(args)
    seeds = seeds_from_args(args)
    if repeat >= len(seeds):
        raise ValueError(f"--direct-repeat {repeat} is outside available repeats 0..{len(seeds) - 1}")
    seed = seeds[int(repeat)]
    methods = [m for m in parse_csv_list(args.methods, str) if m in METHOD_ORDER]
    rows = []

    for mechanism in mechanisms:
        for frac in fracs:
            artificial_mask, masked_da, _mask_meta = make_masked_dataset(
                da_log, observed_positive_mask, mechanism, frac, seed, zero_value, {}
            )
            truth_mask = artificial_mask.values.astype(bool)
            true_vals = da_log.values[truth_mask]
            rows.append(
                pd.DataFrame(
                    {
                        "mechanism": mechanism,
                        "frac": float(frac),
                        "repeat": int(repeat),
                        "curve": "Original non-zero truth",
                        "log10_abundance": true_vals,
                    }
                )
            )

            replace_mask = detect_nonbiological_zeros(
                masked_da,
                zero_value,
                "longitudinal_neighbor",
                params={"min_positive_neighbors": 1},
            )
            observed_mask = build_observed_mask(
                masked_da, zero_value, "positive_only_except_replace", replace_mask
            )
            for method in methods:
                da_imp = impute_method(
                    masked_da,
                    replace_mask,
                    observed_mask,
                    method,
                    {},
                    pca_params,
                    zero_value,
                    seed + 200000,
                )
                rows.append(
                    pd.DataFrame(
                        {
                            "mechanism": mechanism,
                            "frac": float(frac),
                            "repeat": int(repeat),
                            "curve": METHOD_LABELS[method],
                            "log10_abundance": da_imp.values[truth_mask],
                        }
                    )
                )

    direct = pd.concat(rows, ignore_index=True)
    direct["mechanism_label"] = direct["mechanism"].map(MECHANISM_LABELS)
    direct["frac_label"] = (direct["frac"] * 100).round().astype(int).astype(str) + "%"
    direct["condition"] = direct["mechanism_label"] + "\n" + direct["frac_label"]
    condition_order = [
        f"{MECHANISM_LABELS[m]}\n{int(frac * 100)}%"
        for m in MECHANISM_ORDER
        for frac in sorted(direct["frac"].dropna().unique())
    ]
    direct["condition"] = pd.Categorical(direct["condition"], condition_order, ordered=True)
    return direct


def draw_direct_nonzero_distribution_figure(direct: pd.DataFrame, outbase: Path) -> None:
    curve_order = ["Original non-zero truth", "No imputation", "2D2-PCA", "Mean", "Spline"]
    palette = {
        "Original non-zero truth": "#111827",
        "No imputation": METHOD_PALETTE["No imputation"],
        "2D2-PCA": METHOD_PALETTE["2D2-PCA"],
        "Mean": METHOD_PALETTE["Mean"],
        "Spline": METHOD_PALETTE["Spline"],
    }

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, condition in zip(axes, direct["condition"].cat.categories):
        sub = direct[direct["condition"] == condition]
        for curve in curve_order:
            vals = np.sort(sub.loc[sub["curve"] == curve, "log10_abundance"].dropna().to_numpy())
            if len(vals) == 0:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(
                vals,
                y,
                where="post",
                color=palette[curve],
                lw=1.6 if curve in {"Original non-zero truth", "2D2-PCA"} else 1.15,
                alpha=0.95 if curve != "No imputation" else 0.75,
                label=curve,
            )
        ax.set_title(str(condition).replace("\n", ", "))
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=True)
        ax.set_xlim(-3.05, 0.05)
        ax.set_ylim(-0.02, 1.02)

    for ax in axes[3:]:
        ax.set_xlabel("log10(relative abundance + 1e-3)")
    for ax in axes[::3]:
        ax.set_ylabel("Empirical cumulative fraction")

    handles = [
        Line2D([0], [0], color=palette[curve], lw=2, label=curve)
        for curve in curve_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        handlelength=1.8,
    )
    fig.suptitle(
        f"Direct recovery of original non-zero abundance distributions at masked positions (repeat {int(direct['repeat'].iloc[0])})",
        y=0.94,
        fontsize=10,
    )
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.99, hspace=0.38, wspace=0.22)
    save_figure(fig, outbase)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    df = pd.read_csv(args.metrics)
    df = add_labels(df)
    df = paired_improvement(df)

    args.outdir.mkdir(parents=True, exist_ok=True)
    stats_path = args.metrics.parent / "distribution_restoration_publication_stats.csv"
    make_stats_table(df, stats_path)
    draw_main_figure(df, args.outdir / "experiment2_distribution_restoration_main")
    draw_all_modes_figure(df, args.outdir / "experiment2_distribution_restoration_all_modes")
    draw_ecdf_figure(df, args.outdir / "experiment2_distribution_restoration_ecdf")
    if not args.skip_direct_distribution:
        direct = reconstruct_direct_distribution(args.config, args.direct_repeat)
        direct_path = args.metrics.parent / f"direct_nonzero_distribution_values_repeat{args.direct_repeat}.csv"
        direct.to_csv(direct_path, index=False)
        draw_direct_nonzero_distribution_figure(
            direct,
            args.outdir / "experiment2_direct_nonzero_original_distribution",
        )

    print(f"Wrote figures to {args.outdir}")
    print(f"Wrote statistics to {stats_path}")


if __name__ == "__main__":
    main()
