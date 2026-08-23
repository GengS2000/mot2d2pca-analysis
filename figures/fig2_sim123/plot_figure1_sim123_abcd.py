"""Build a publication-oriented ABCD draft from the merged Sim1/2/3 results.

Panels A-C compare five imputation methods across five masking fractions for
MSE, Pearson correlation, and Wasserstein distance. Panel D fixes a single
masking fraction and shows all three metrics across ten time bins.

Panel D reports all three metrics because per-bin method rankings from MSE and
Pearson r are near-redundant (Spearman rho 0.82-1.00 within bins) whereas
Wasserstein distance ranks methods differently (rho 0.12-0.50 against r).
Showing only one of the three would therefore hide the non-redundant metric.

The panel-D masking fraction is fixed at the mildest level (50%) by a rule that
does not depend on the outcome; the full fraction x metric x time-bin grid is
delivered as supplementary figures and tables.

All bars, lines, and uncertainty intervals are recomputed from repeat-level
files. Error bars/bands are +/- 1 sample SD across 50 masking repeats.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = Path(
    r"D:\penalized_2d2\sim123_d1q1_lam2_sig16_20260805_130548_results_20260805"
)
RESULTS_ROOT = Path(os.environ.get("SIM123_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
OUTPUT_DIR = Path(os.environ.get("SIM123_FIGURE_OUTDIR", PROJECT_ROOT / "output" / "pdf"))
MOT_CONFIG_LABEL = os.environ.get(
    "SIM123_MOT_CONFIG_LABEL",
    "d=1, q=1, lambda=2, sigma=16",
)
# Panel D fixes one masking fraction so that all three metrics fit in the main
# figure. The level is chosen by an outcome-independent rule (mildest masking),
# and every other fraction is reported in the supplement.
PANEL_D_FRACTION = float(os.environ.get("SIM123_PANEL_D_FRACTION", "0.5"))

METHODS = [
    "penalized_2d2_pca",
    "ctf",
    "mean",
    "nearest_positive",
    "temporal_knn",
]
METHOD_LABELS = {
    "penalized_2d2_pca": r"MOT-(2D)$^2$PCA",
    "ctf": "CTF",
    "mean": "Mean",
    "nearest_positive": "Nearest positive",
    "temporal_knn": "Temporal-KNN",
}
METHOD_COLORS = {
    # Publication palette selected from the user's candidate colors.
    # This mapping is shared by lines, markers, and the global legend.
    "penalized_2d2_pca": "#05B9E2",  # MOT-(2D)^2PCA
    "ctf": "#FA7F6F",               # CTF
    "mean": "#54B345",              # Mean
    "nearest_positive": "#BB9727",  # Nearest positive
    "temporal_knn": "#82B0D2",      # Temporal-KNN
}
METHOD_MARKERS = {
    "penalized_2d2_pca": "o",
    "ctf": "s",
    "mean": "^",
    "nearest_positive": "D",
    "temporal_knn": "v",
}
METHOD_LINESTYLES = {
    "penalized_2d2_pca": "-",
    "ctf": "--",
    "mean": "-.",
    "nearest_positive": ":",
    "temporal_knn": (0, (3, 1, 1, 1)),
}
REFERENCE_COLOR = "#8C8C8C"
REFERENCE_LABEL = "Before imputation"
# In panel D the method of interest is drawn last and slightly thinner than the
# comparators. Sim3 at 50% masking has MOT and Mean agreeing to a median 0.006
# in Pearson r, so with a shared z-order the later-drawn Mean line covered MOT
# completely; the thinner top line leaves the comparator visible as a halo.
HIGHLIGHT_METHOD = "penalized_2d2_pca"
HIGHLIGHT_LINEWIDTH = 1.0
COMPARATOR_LINEWIDTH = 1.4

FRACTIONS = [0.5, 0.6, 0.7, 0.8, 0.9]
METRICS = ["mse_after", "pearson_after", "wd_after"]
# Pre-imputation counterpart of each metric; Pearson r is undefined before
# imputation in these runs, so that panel carries no reference line.
METRIC_REFERENCE = {
    "mse_after": "mse_before",
    "pearson_after": None,
    "wd_after": "wd_before",
}
METRIC_TITLES = {
    "mse_after": "MSE",
    "pearson_after": "Pearson r",
    "wd_after": "Wasserstein distance",
}
METRIC_AXIS_LABELS = {
    "mse_after": "MSE",
    "pearson_after": "Pearson r",
    "wd_after": "Wasserstein dist.",
}
# Panel D only: MSE and Wasserstein distance span ~2 decades across time bins,
# where a linear axis would compress every method except the worst one into the
# floor. Panels A-C keep linear axes because their marks are bars.
METRIC_LOG = {"mse_after": True, "pearson_after": False, "wd_after": True}

SIMULATIONS = {
    "sim1": {
        "label": "Sim1 | MOT-generated truth",
        "repeat_file": "all_replicates_metrics_sim1.csv",
        "bin_file": "per_bin_metrics_sim1.csv",
        "scope": "MaskedCells",
        "repeat_col": "replicate",
    },
    "sim2": {
        "label": "Sim2 | CP-ALS-generated truth",
        "repeat_file": "repeat_level_metrics.csv",
        "bin_file": "repeat_bin_metrics_all.csv",
        "scope": "InjectedCells",
        "repeat_col": "repeat",
    },
    "sim3": {
        "label": "Sim3 | effect-perturbed truth",
        "repeat_file": "repeat_level_metrics.csv",
        "bin_file": "repeat_bin_metrics_all.csv",
        "scope": "InjectedCells",
        "repeat_col": "repeat",
    },
}

BIN_ORDER = ["0"] + [f"{start}-{start + 3}" for start in range(1, 37, 4)]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.8,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            # Keep the physical A4 canvas; tight cropping would silently
            # change the delivered print width.
            "savefig.bbox": None,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_results() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if PANEL_D_FRACTION not in FRACTIONS:
        raise ValueError(f"PANEL_D_FRACTION={PANEL_D_FRACTION} is not one of {FRACTIONS}")
    repeat_data: dict[str, pd.DataFrame] = {}
    bin_data: dict[str, pd.DataFrame] = {}
    for sim, cfg in SIMULATIONS.items():
        repeat_path = RESULTS_ROOT / sim / cfg["repeat_file"]
        bin_path = RESULTS_ROOT / sim / cfg["bin_file"]
        if not repeat_path.is_file():
            raise FileNotFoundError(repeat_path)
        if not bin_path.is_file():
            raise FileNotFoundError(bin_path)

        repeat_df = pd.read_csv(repeat_path)
        repeat_df = repeat_df.loc[
            repeat_df["scope"].eq(cfg["scope"])
            & repeat_df["method"].isin(METHODS)
            & repeat_df["frac"].isin(FRACTIONS)
        ].copy()
        bin_df = pd.read_csv(bin_path)
        bin_df["bin"] = bin_df["bin"].astype(str)
        bin_df = bin_df.loc[
            bin_df["scope"].eq(cfg["scope"])
            & bin_df["method"].isin(METHODS)
            & bin_df["frac"].isin(FRACTIONS)
            & bin_df["bin"].isin(BIN_ORDER)
        ].copy()

        expected_repeat_rows = len(FRACTIONS) * 50 * len(METHODS)
        expected_bin_rows = len(FRACTIONS) * 50 * len(BIN_ORDER) * len(METHODS)
        if len(repeat_df) != expected_repeat_rows:
            raise ValueError(
                f"{sim}: expected {expected_repeat_rows} masked/injected repeat rows, "
                f"found {len(repeat_df)}"
            )
        if len(bin_df) != expected_bin_rows:
            raise ValueError(
                f"{sim}: expected {expected_bin_rows} all-method per-bin rows, found {len(bin_df)}"
            )
        if repeat_df[METRICS].isna().any().any() or bin_df[METRICS].isna().any().any():
            raise ValueError(f"{sim}: non-finite primary metric found")

        repeat_data[sim] = repeat_df
        bin_data[sim] = bin_df
    return repeat_data, bin_data


def summarize_repeat_data(repeat_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for sim, df in repeat_data.items():
        long_df = df.melt(
            id_vars=["frac", "method"],
            value_vars=METRICS,
            var_name="metric",
            value_name="value",
        )
        summary = (
            long_df.groupby(["frac", "method", "metric"], as_index=False)["value"]
            .agg(mean="mean", sd=lambda x: x.std(ddof=1), n="count")
        )
        summary.insert(0, "simulation", sim)
        frames.append(summary)
    out = pd.concat(frames, ignore_index=True)
    if not out["n"].eq(50).all():
        raise ValueError("Every method x fraction x metric point must contain 50 repeats")
    return out


def summarize_bin_data(bin_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for sim, df in bin_data.items():
        long_df = df.melt(
            id_vars=["frac", "bin", "method"],
            value_vars=METRICS,
            var_name="metric",
            value_name="value",
        )
        summary = (
            long_df.groupby(["frac", "bin", "method", "metric"], as_index=False)["value"]
            .agg(mean="mean", sd=lambda x: x.std(ddof=1), n="count")
        )
        summary["bin"] = pd.Categorical(summary["bin"], categories=BIN_ORDER, ordered=True)
        summary = summary.sort_values(
            ["frac", "metric", "method", "bin"]
        ).reset_index(drop=True)
        summary.insert(0, "simulation", sim)
        frames.append(summary)
    out = pd.concat(frames, ignore_index=True)
    if not out["n"].eq(50).all():
        raise ValueError("Every method x fraction x time-bin point must contain 50 repeats")
    return out


def summarize_reference(
    data: dict[str, pd.DataFrame], group_cols: list[str]
) -> pd.DataFrame:
    """Pre-imputation error, which is shared by every method within a cell.

    All methods start from the same masked tensor, so the ``*_before`` columns
    must be constant across methods; the check below keeps that assumption
    honest instead of silently averaging heterogeneous values.
    """
    frames = []
    for sim, df in data.items():
        for metric, ref_col in METRIC_REFERENCE.items():
            if ref_col is None:
                continue
            per_method = df.groupby(group_cols + ["method"])[ref_col].mean()
            spread = per_method.groupby(level=list(range(len(group_cols)))).agg(
                lambda x: float(np.ptp(x.to_numpy()))
            )
            if float(spread.max()) > 1e-6:
                raise ValueError(
                    f"{sim}: {ref_col} differs across methods (max spread {spread.max():.3g})"
                )
            summary = df.groupby(group_cols, as_index=False)[ref_col].mean()
            summary = summary.rename(columns={ref_col: "reference"})
            summary.insert(0, "metric", metric)
            summary.insert(0, "simulation", sim)
            frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.65, zorder=0)
    ax.tick_params(axis="both", length=2.5, width=0.6, color="#444444")
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")


def log_axis_limits(values: np.ndarray, pad: float = 1.35) -> tuple[float, float]:
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return 1e-3, 1.0
    return float(finite.min()) / pad, float(finite.max()) * pad


def apply_log_ticks(ax: plt.Axes) -> None:
    """Label 1/2/5 decades in plain decimal form.

    Several panels span well under one decade, where the default log locator
    emits a single 10^0 tick and the axis becomes unreadable.
    """
    ax.yaxis.set_major_locator(
        mpl.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=14)
    )
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def plot_fraction_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    sim: str,
    metric: str,
    show_title: bool,
    show_xlabel: bool,
) -> None:
    """Grouped bars: one metric for one simulation across the five fractions."""
    data = summary.loc[
        summary["simulation"].eq(sim) & summary["metric"].eq(metric)
    ]
    x = np.arange(len(FRACTIONS), dtype=float)
    width = 0.145
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width

    for method, offset in zip(METHODS, offsets):
        method_data = data.loc[data["method"].eq(method)].set_index("frac").loc[FRACTIONS]
        ax.bar(
            x + offset,
            method_data["mean"].to_numpy(),
            width=width * 0.92,
            yerr=method_data["sd"].to_numpy(),
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.35,
            error_kw={"ecolor": "#333333", "elinewidth": 0.55, "capsize": 1.2, "capthick": 0.55},
            zorder=3,
        )

    if show_title:
        ax.set_title(METRIC_TITLES[metric], pad=16.0, weight="semibold")
    ax.set_xticks(x, [f"{int(frac * 100)}%" for frac in FRACTIONS])
    if show_xlabel:
        ax.set_xlabel("Masked fraction")
    ax.set_ylabel(METRIC_AXIS_LABELS[metric])

    values = data["mean"].to_numpy()
    errors = data["sd"].to_numpy()
    if metric == "pearson_after":
        lower = min(-0.06, float(np.nanmin(values - errors)) - 0.035)
        ax.set_ylim(lower, 1.035)
        ax.axhline(0.0, color="#777777", linewidth=0.55, zorder=1)
    else:
        upper = float(np.nanmax(values + errors))
        ax.set_ylim(0.0, upper * 1.12 if upper > 0 else 1.0)
    # No grid here: the bar groups already carry the vertical structure, and the
    # horizontal rules read as clutter behind 25 bars.
    style_axis(ax, grid=False)


def plot_time_axis(
    ax: plt.Axes,
    bin_summary: pd.DataFrame,
    reference: pd.DataFrame,
    sim: str,
    metric: str,
    frac: float,
    ylim: tuple[float, float],
    show_title: bool,
    show_xlabel: bool,
    sim_prefix: bool,
) -> None:
    x = np.arange(len(BIN_ORDER), dtype=float)
    data = bin_summary.loc[
        bin_summary["simulation"].eq(sim)
        & bin_summary["frac"].eq(frac)
        & bin_summary["metric"].eq(metric)
    ]
    log_scale = METRIC_LOG[metric]

    ref = reference.loc[
        reference["simulation"].eq(sim)
        & reference["metric"].eq(metric)
        & reference["frac"].eq(frac)
    ]
    if not ref.empty:
        ref_values = ref.set_index("bin").loc[BIN_ORDER, "reference"].to_numpy(dtype=float)
        ax.plot(
            x,
            ref_values,
            color=REFERENCE_COLOR,
            linewidth=0.85,
            linestyle=(0, (4, 2)),
            zorder=2,
        )

    # Comparators first, the highlighted method last, so it is never painted over.
    draw_order = [m for m in METHODS if m != HIGHLIGHT_METHOD] + [HIGHLIGHT_METHOD]
    for method in draw_order:
        method_index = METHODS.index(method)
        highlighted = method == HIGHLIGHT_METHOD
        method_data = data.loc[data["method"].eq(method)].set_index("bin").loc[BIN_ORDER]
        mean = method_data["mean"].to_numpy(dtype=float)
        sd = method_data["sd"].to_numpy(dtype=float)
        lower = mean - sd
        if log_scale:
            # A log axis cannot render a band that crosses zero; clipping at the
            # axis floor keeps the visible part faithful, and the caption says so
            # rather than the band being silently dropped.
            lower = np.maximum(lower, ylim[0])
        ax.fill_between(
            x, lower, mean + sd, color=METHOD_COLORS[method], alpha=0.14, linewidth=0, zorder=2
        )
        ax.plot(
            x,
            mean,
            color=METHOD_COLORS[method],
            linewidth=HIGHLIGHT_LINEWIDTH if highlighted else COMPARATOR_LINEWIDTH,
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markevery=[method_index, method_index + 5],
            markersize=2.8,
            markeredgewidth=0.0,
            zorder=5 if highlighted else 3,
        )

    if show_title:
        ax.set_title(METRIC_TITLES[metric], pad=16.0, weight="semibold")
    tick_positions = [0, 1, 3, 5, 7, 9]
    ax.set_xticks(tick_positions)
    if show_xlabel:
        ax.set_xticklabels([BIN_ORDER[idx] for idx in tick_positions], rotation=45, ha="right")
    else:
        ax.set_xticklabels([])

    label = METRIC_AXIS_LABELS[metric]
    if sim_prefix:
        sim_short = SIMULATIONS[sim]["label"].split("|")[0].strip()
        label = f"{sim_short}\n{label}"
    ax.set_ylabel(label)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        apply_log_ticks(ax)
    else:
        ax.axhline(0.0, color="#777777", linewidth=0.55, zorder=1)
        ax.set_ylim(*ylim)
    style_axis(ax)


def add_panel_heading(ax: plt.Axes, letter: str, text: str, y: float = 1.16) -> None:
    ax.text(
        -0.34,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=10.5,
        weight="bold",
        va="center",
        ha="left",
        clip_on=False,
    )
    ax.text(
        -0.22,
        y,
        text,
        transform=ax.transAxes,
        fontsize=7.8,
        weight="semibold",
        va="center",
        ha="left",
        clip_on=False,
    )


def panel_d_ylimits(
    bin_summary: pd.DataFrame, reference: pd.DataFrame, frac: float
) -> dict[tuple[str, str], tuple[float, float]]:
    """One y-range per (simulation, metric) cell of panel D.

    Sim1 is evaluated on masked cells and Sim2/Sim3 on injected cells, so the
    rows are not on a comparable footing and must not be forced onto a shared
    axis; the unbounded metrics get a per-cell log range instead. Pearson r is
    bounded and keeps a fixed range so the rows stay directly comparable.
    """
    limits: dict[tuple[str, str], tuple[float, float]] = {}
    for sim in SIMULATIONS:
        for metric in METRICS:
            if not METRIC_LOG[metric]:
                limits[(sim, metric)] = (-0.05, 1.05)
                continue
            subset = bin_summary.loc[
                bin_summary["simulation"].eq(sim)
                & bin_summary["frac"].eq(frac)
                & bin_summary["metric"].eq(metric)
            ]
            ref = reference.loc[
                reference["simulation"].eq(sim)
                & reference["metric"].eq(metric)
                & reference["frac"].eq(frac)
            ]
            candidates = np.concatenate(
                [subset["mean"].to_numpy(dtype=float), ref["reference"].to_numpy(dtype=float)]
            )
            limits[(sim, metric)] = log_axis_limits(candidates, pad=1.5)
    return limits


def build_figure(
    fraction_summary: pd.DataFrame,
    bin_summary: pd.DataFrame,
    bin_reference: pd.DataFrame,
) -> plt.Figure:
    configure_matplotlib()
    # A4 portrait width (210 mm = 8.2677 in); height preserves the current
    # information density while remaining within the printable A4 page height.
    fig = plt.figure(figsize=(8.2677, 10.78), facecolor="white")
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.0, 1.0, 1.0, 3.35],
        hspace=0.48,
        top=0.940,
        bottom=0.085,
        left=0.115,
        right=0.985,
    )

    panel_letters = ["A", "B", "C"]
    for row_index, sim in enumerate(SIMULATIONS):
        inner = outer[row_index].subgridspec(1, 3, wspace=0.45)
        axes = [fig.add_subplot(inner[0, idx]) for idx in range(3)]
        for metric_index, (ax, metric) in enumerate(zip(axes, METRICS)):
            plot_fraction_axis(
                ax,
                fraction_summary,
                sim=sim,
                metric=metric,
                show_title=row_index == 0,
                # One centred x-label per block; repeating it under all three
                # columns crowds the gap ahead of panel D.
                show_xlabel=row_index == len(SIMULATIONS) - 1 and metric_index == 1,
            )
        add_panel_heading(
            axes[0],
            panel_letters[row_index],
            SIMULATIONS[sim]["label"],
            y=1.44 if row_index == 0 else 1.16,
        )

    ylimits = panel_d_ylimits(bin_summary, bin_reference, PANEL_D_FRACTION)
    inner_d = outer[3].subgridspec(3, 3, hspace=0.22, wspace=0.45)
    time_axes: list[list[plt.Axes]] = []
    for sim_index, sim in enumerate(SIMULATIONS):
        row_axes = []
        for metric_index, metric in enumerate(METRICS):
            ax = fig.add_subplot(inner_d[sim_index, metric_index])
            plot_time_axis(
                ax,
                bin_summary,
                bin_reference,
                sim=sim,
                metric=metric,
                frac=PANEL_D_FRACTION,
                ylim=ylimits[(sim, metric)],
                show_title=sim_index == 0,
                show_xlabel=sim_index == len(SIMULATIONS) - 1,
                sim_prefix=metric_index == 0,
            )
            row_axes.append(ax)
        time_axes.append(row_axes)
    time_axes[-1][1].set_xlabel("Time bin (weeks)")
    add_panel_heading(
        time_axes[0][0],
        "D",
        f"Time-bin profile at {int(PANEL_D_FRACTION * 100)}% masked",
        y=1.44,
    )

    method_handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=METHOD_COLORS[m],
            linestyle=METHOD_LINESTYLES[m],
            marker=METHOD_MARKERS[m],
            markersize=3.2,
            markeredgewidth=0.0,
            linewidth=(
                HIGHLIGHT_LINEWIDTH if m == HIGHLIGHT_METHOD else COMPARATOR_LINEWIDTH
            ),
            label=METHOD_LABELS[m],
        )
        for m in METHODS
    ]
    method_handles.append(
        mpl.lines.Line2D(
            [],
            [],
            color=REFERENCE_COLOR,
            linestyle=(0, (4, 2)),
            linewidth=0.9,
            label=REFERENCE_LABEL,
        )
    )
    method_legend = fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.008),
        ncol=6,
        frameon=False,
        handlelength=1.9,
        columnspacing=1.0,
    )
    fig.add_artist(method_legend)
    return fig


def write_caption() -> None:
    frac_pct = int(PANEL_D_FRACTION * 100)
    caption = (
        "Figure 1. Imputation performance across simulation truths, masking fractions, "
        "and time bins. Panels A-C compare MOT-(2D)^2PCA, CTF, Mean, Nearest Positive "
        "Observation, and Temporal-KNN using MSE, Pearson correlation, and Wasserstein "
        "distance as a function of the masked fraction, evaluated on masked cells (Sim1, "
        "scope MaskedCells) or injected cells (Sim2-Sim3, scope InjectedCells); these two "
        "evaluation sets are not identical, so methods are compared within a simulation "
        "and not across simulations. Panel D fixes the masking fraction at the mildest "
        f"level ({frac_pct}%), a choice specified independently of the results, and shows "
        "all three metrics across ten time bins, with rows for simulations and columns for "
        "metrics; the remaining four masking fractions appear in Supplementary Figures "
        "S1-S3 and Supplementary Tables S1-S2. Bars (A-C) and lines (D) show the mean over "
        "50 independent masking repeats; error bars and ribbons show +/- 1 sample SD across "
        "those repeats and therefore describe repeat-to-repeat variability rather than the "
        "uncertainty of paired between-method differences, which is reported in "
        "Supplementary Table S1. Panels A-C use linear axes; in panel D, MSE and Wasserstein "
        "distance use log axes scaled independently for each cell with ribbons clipped at "
        "the axis floor, while Pearson r keeps a common range throughout. Grey dashed lines "
        "in panel D mark the pre-imputation error, which is identical for all five methods. "
        "Lower MSE and "
        "Wasserstein distance and higher Pearson correlation indicate better "
        "reconstruction. Sim1 truth is generated by the MOT model itself and Sim2 truth by "
        "CP-ALS, so panels A and B quantify recovery under a matched generative model "
        "rather than a neutral comparison; Sim3 is the model-agnostic benchmark. "
        f"MOT-(2D)^2PCA was fitted using {MOT_CONFIG_LABEL}. All metrics are evaluated in "
        "log10-transformed abundance space."
    )
    (OUTPUT_DIR / "figure1_sim123_caption.txt").write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    repeat_data, bin_data = load_results()
    fraction_summary = summarize_repeat_data(repeat_data)
    bin_summary = summarize_bin_data(bin_data)
    bin_reference = summarize_reference(bin_data, ["frac", "bin"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fraction_summary.to_csv(OUTPUT_DIR / "figure1_fraction_plot_data.csv", index=False)
    bin_summary.to_csv(OUTPUT_DIR / "figure1_time_plot_data.csv", index=False)
    bin_reference.to_csv(OUTPUT_DIR / "figure1_time_reference_data.csv", index=False)

    fig = build_figure(fraction_summary, bin_summary, bin_reference)
    base = OUTPUT_DIR / "figure1_sim123_d1q1_lam2_sig16_method_fraction_time_abcd_a4_600dpi"
    save_kwargs = {"dpi": 600, "bbox_inches": None, "pad_inches": 0.02}
    fig.savefig(base.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(base.with_suffix(".svg"), **save_kwargs)
    fig.savefig(base.with_suffix(".png"), **save_kwargs)
    fig.savefig(
        base.with_suffix(".tiff"),
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
        **save_kwargs,
    )
    plt.close(fig)

    # Deliver RGB rasters for journal/submission-system compatibility. The
    # plot itself remains white-background; removing alpha avoids RGBA TIFF
    # rejection in common production workflows.
    for suffix in (".png", ".tiff"):
        raster_path = base.with_suffix(suffix)
        with Image.open(raster_path) as image:
            rgb = image.convert("RGB")
            raster_kwargs = {"dpi": (600, 600)}
            if suffix == ".tiff":
                raster_kwargs["compression"] = "tiff_lzw"
            rgb.save(raster_path, **raster_kwargs)
    write_caption()

    print(f"results_root={RESULTS_ROOT}")
    print(f"mot_config={MOT_CONFIG_LABEL}")
    print(f"panel_d_fraction={PANEL_D_FRACTION}")
    print(f"output_pdf={base.with_suffix('.pdf')}")
    print(f"output_svg={base.with_suffix('.svg')}")
    print(f"output_png={base.with_suffix('.png')}")
    print(f"output_tiff={base.with_suffix('.tiff')}")


if __name__ == "__main__":
    main()
