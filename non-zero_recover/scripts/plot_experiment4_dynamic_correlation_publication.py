from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOT_DEPS = ROOT / "outputs" / "experiment2_distribution" / "_plot_deps"
if sys.version_info[:2] == (3, 10) and PLOT_DEPS.exists():
    sys.path.insert(0, str(PLOT_DEPS))

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required. In this workspace, run this script with the "
        "Python 3.10 executable so outputs/experiment2_distribution/_plot_deps "
        "can provide matplotlib."
    ) from exc


METHOD_ORDER = ["no_imputation", "penalized_2d2_pca"]
METHOD_LABELS = {
    "no_imputation": "Masked zeros",
    "penalized_2d2_pca": "Penalized 2D2-PCA",
}
METHOD_COLORS = {
    "no_imputation": "#6B7280",
    "penalized_2d2_pca": "#0072B2",
}
MECHANISM_ORDER = [
    "random_entry",
    "low_abundance_biased",
    "consecutive_time_window_dropout",
]
MECHANISM_LABELS = {
    "random_entry": "Random entry",
    "low_abundance_biased": "Low-abundance",
    "consecutive_time_window_dropout": "Window dropout",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


def condition_frame(fracs: list[float]) -> pd.DataFrame:
    rows = []
    idx = 0
    for mechanism in MECHANISM_ORDER:
        for frac in sorted(fracs):
            rows.append(
                {
                    "mechanism": mechanism,
                    "frac": frac,
                    "condition": f"{int(round(frac * 100))}%",
                    "x": idx,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def load_inputs(base: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    static = pd.read_csv(base / "static_correlation_metrics.csv")
    lagged = pd.read_csv(base / "lagged_correlation_metrics.csv")
    edge = pd.read_csv(base / "network_edge_overlap.csv")
    hub = pd.read_csv(base / "hub_taxa_overlap.csv")
    return static, lagged, edge, hub


def oracle_methods(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df["replacement_mode"] == "oracle_mask") & df["method"].isin(METHOD_ORDER)].copy()
    out["method_label"] = out["method"].map(METHOD_LABELS)
    return out


def summarize(df: pd.DataFrame, value_col: str, condition: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["mechanism", "frac", "method"], dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"].fillna(0) / np.sqrt(grouped["count"].clip(lower=1))
    grouped["ci95"] = 1.96 * grouped["sem"]
    grouped["metric"] = value_col
    grouped = grouped.merge(condition, on=["mechanism", "frac"], how="left")
    grouped["method_label"] = grouped["method"].map(METHOD_LABELS)
    return grouped.sort_values(["x", "method"])


def raw_for_plot(df: pd.DataFrame, value_col: str, condition: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(condition, on=["mechanism", "frac"], how="left")
    out["metric"] = value_col
    out["value"] = out[value_col]
    return out


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def add_mechanism_groups(ax, condition: pd.DataFrame) -> None:
    for mechanism in MECHANISM_ORDER:
        xs = condition.loc[condition["mechanism"] == mechanism, "x"].to_numpy(dtype=float)
        if xs.size == 0:
            continue
        center = float(np.mean(xs))
        ax.plot(
            [min(xs) - 0.33, max(xs) + 0.33],
            [-0.105, -0.105],
            transform=ax.get_xaxis_transform(),
            color="#9CA3AF",
            linewidth=0.8,
            clip_on=False,
        )
        ax.text(
            center,
            -0.185,
            MECHANISM_LABELS[mechanism],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
            color="#374151",
            clip_on=False,
        )


def draw_panel(
    ax,
    raw: pd.DataFrame,
    stats: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    hline: float | None = None,
) -> None:
    offsets = {"no_imputation": -0.15, "penalized_2d2_pca": 0.15}
    rng = np.random.default_rng(314159)
    for method in METHOD_ORDER:
        method_raw = raw[raw["method"] == method]
        method_stats = stats[stats["method"] == method]
        color = METHOD_COLORS[method]
        x_raw = (
            method_raw["x"].to_numpy(dtype=float)
            + offsets[method]
            + rng.normal(0.0, 0.025, size=len(method_raw))
        )
        ax.scatter(
            x_raw,
            method_raw[value_col].to_numpy(dtype=float),
            s=10,
            color=color,
            alpha=0.20,
            linewidths=0,
            zorder=1,
        )
        ax.errorbar(
            method_stats["x"].to_numpy(dtype=float) + offsets[method],
            method_stats["mean"].to_numpy(dtype=float),
            yerr=method_stats["ci95"].to_numpy(dtype=float),
            fmt="o-",
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=4.2,
            linewidth=1.45,
            capsize=2.5,
            label=METHOD_LABELS[method],
            zorder=3,
        )
    if hline is not None:
        ax.axhline(hline, color="#111827", linestyle=(0, (2, 2)), linewidth=0.8, alpha=0.65, zorder=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", pad=5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_plot(base: Path, outdir: Path, prefix: str) -> None:
    configure_style()
    static, lagged, edge, hub = load_inputs(base)
    static = oracle_methods(static)
    lagged = oracle_methods(lagged)
    edge = oracle_methods(edge)
    hub = oracle_methods(hub)
    edge_200 = edge[edge["k"] == 200].copy()

    fracs = sorted(static["frac"].dropna().unique().tolist())
    condition = condition_frame(fracs)

    panels = [
        (
            "A",
            static,
            "static_pearson",
            "Pearson r",
            "Static correlation similarity",
            (0.955, 1.002),
            1.0,
        ),
        (
            "B",
            static,
            "static_frobenius",
            "Frobenius distance",
            "Static correlation error",
            (0.0, 3.4),
            None,
        ),
        (
            "C",
            lagged,
            "lagged_pearson",
            "Pearson r",
            "Lagged correlation similarity",
            (0.955, 1.002),
            1.0,
        ),
        (
            "D",
            lagged,
            "lagged_frobenius",
            "Frobenius distance",
            "Lagged correlation error",
            (0.0, 2.5),
            None,
        ),
        (
            "E",
            edge_200,
            "topk_precision",
            "Top-200 precision",
            "Directed network edge recovery",
            (0.68, 1.01),
            1.0,
        ),
        (
            "F",
            hub,
            "hub_jaccard",
            "Hub Jaccard index",
            "Hub taxa recovery",
            (0.62, 1.01),
            1.0,
        ),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.6), sharex=True)
    summary_rows = []
    for ax, panel in zip(axes.ravel(), panels):
        label, df, value_col, ylabel, title, ylim, hline = panel
        raw = raw_for_plot(df, value_col, condition)
        stats = summarize(df, value_col, condition)
        summary_rows.append(stats)
        draw_panel(ax, raw, stats, value_col, ylabel, title, ylim=ylim, hline=hline)
        panel_label(ax, label)

    xticks = condition["x"].to_numpy(dtype=float)
    xlabels = condition["condition"].tolist()
    for ax in axes.ravel():
        ax.set_xticks(xticks, xlabels)
        ax.tick_params(axis="x", length=0, pad=2)
    for ax in axes[-1, :]:
        add_mechanism_groups(ax, condition)
        ax.set_xlabel("Inserted-zero fraction", labelpad=30)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.4,
    )
    fig.suptitle(
        "Oracle artificial-zero mask: preservation of taxon-taxon correlation structure",
        y=1.03,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=1.2, w_pad=1.1)

    outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(outdir / f"{prefix}_summary_values.csv", index=False)

    for ext, dpi in [("png", 600), ("tiff", 600), ("pdf", None)]:
        path = outdir / f"{prefix}.{ext}"
        save_kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        if ext == "tiff":
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **save_kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create publication-grade oracle-mask Experiment 4 figures."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "outputs" / "experiment4_dynamic_correlation",
        help="Experiment 4 output directory containing the metrics CSV files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "experiment4_dynamic_correlation" / "figures",
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="experiment4_oracle_mask_publication_summary",
        help="Output filename prefix.",
    )
    args = parser.parse_args()
    build_plot(args.base, args.outdir, args.prefix)
    print(f"Wrote publication figure files to {args.outdir}")


if __name__ == "__main__":
    main()
