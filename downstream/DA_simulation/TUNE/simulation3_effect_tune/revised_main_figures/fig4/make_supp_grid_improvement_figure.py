"""Supplementary figure: parameter-grid improvement fractions for the four DA
engines, plus the masking-resolved decomposition that the revised main text
now claims.

Panel A reproduces the content that was previously carried by Figure 4E
(fraction of the 99 effect-size x masking cells improved by MOT-(2D)^2 PCA
completion, for six outcomes and four DA engines). Panels B and C resolve the
aggregate medians by masking fraction, which is what makes the LinDA result
regime-dependent rather than uniform.

All inputs come from the corrected additive-effect LinDA run used for the
revised Figure 4; no numbers are recomputed here beyond grouping.

Outputs: PDF (vector, for submission), PNG (screen), TIFF (600 dpi).
No figure number is drawn inside the artwork, so the panel can be numbered
from the caption without regenerating the file.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = Path(__file__).resolve().parent
DELTAS_CSV = SCRIPT_DIR / "panel_CDE_corrected_DA_deltas.csv"
IMPROVEMENT_CSV = SCRIPT_DIR / "panel_E_improvement_proportions.csv"

# Written next to the script by default; override with SUPP_OUTPUT_DIR.
DEFAULT_OUT = SCRIPT_DIR / "supplementary_figures"
OUT = Path(os.environ.get("SUPP_OUTPUT_DIR", str(DEFAULT_OUT))).resolve()
OUT.mkdir(parents=True, exist_ok=True)
STEM = "Figure_S5_grid_cell_improvement_fractions"

# Palette carried over from make_figure4_revised.py so the supplement matches
# the main figure it supports.
HEAT_GOLD = "#FDBF6F"
HEAT_PURPLE = "#2C004B"
GRID = "#D9D7D3"
TEXT = "#222222"
PROP_CMAP = LinearSegmentedColormap.from_list(
    "proportion_gold_purple", [HEAT_GOLD, HEAT_PURPLE]
)

METHOD_ORDER = ["LinDA", "MaAsLin2", "ZIGMM", "glmmPQL"]
METHOD_COLOR = {
    "LinDA": "#168C80",
    "MaAsLin2": "#D49A3A",
    "ZIGMM": "#5A3D8C",
    "glmmPQL": "#526FA5",
}
METHOD_MARKER = {"LinDA": "o", "MaAsLin2": "s", "ZIGMM": "^", "glmmPQL": "D"}

OUTCOME_ORDER = [
    "F1 gain",
    "Recall gain",
    "FDR reduction",
    "TP gain",
    "FP reduction",
    "FN reduction",
]
OUTCOME_LABEL = {
    "F1 gain": "F1\ngain",
    "Recall gain": "Recall\ngain",
    "FDR reduction": "FDR\nreduction",
    "TP gain": "TP\ngain",
    "FP reduction": "FP\nreduction",
    "FN reduction": "FN\nreduction",
}


def set_publication_style() -> None:
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#555555",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "lines.linewidth": 1.35,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False, which="both")


def panel_letter(fig: mpl.figure.Figure, ax: mpl.axes.Axes, letter: str,
                 dx: float = -0.055, dy: float = 0.055) -> None:
    box = ax.get_position()
    fig.text(
        box.x0 + dx,
        box.y1 + dy,
        letter,
        fontsize=11.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=TEXT,
    )


def draw_matrix(ax: mpl.axes.Axes, improvement: pd.DataFrame) -> mpl.image.AxesImage:
    grid = (
        improvement.pivot(index="da_method", columns="outcome",
                          values="proportion_improved")
        .reindex(index=METHOD_ORDER, columns=OUTCOME_ORDER)
    )
    counts = (
        improvement.pivot(index="da_method", columns="outcome", values="n_improved")
        .reindex(index=METHOD_ORDER, columns=OUTCOME_ORDER)
    )
    evaluable = (
        improvement.pivot(index="da_method", columns="outcome", values="n_evaluable")
        .reindex(index=METHOD_ORDER, columns=OUTCOME_ORDER)
    )

    image = ax.imshow(
        grid.to_numpy(float),
        cmap=PROP_CMAP,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    for i, method in enumerate(METHOD_ORDER):
        for j, outcome in enumerate(OUTCOME_ORDER):
            value = grid.loc[method, outcome]
            n_up = int(counts.loc[method, outcome])
            n_all = int(evaluable.loc[method, outcome])
            colour = "white" if value > 0.55 else TEXT
            ax.text(
                j, i - 0.13, f"{value * 100:.0f}%",
                ha="center", va="center", fontsize=9.0,
                fontweight="bold", color=colour,
            )
            ax.text(
                j, i + 0.24, f"{n_up}/{n_all}",
                ha="center", va="center", fontsize=7.0, color=colour, alpha=0.85,
            )

    ax.set_xticks(range(len(OUTCOME_ORDER)))
    ax.set_xticklabels([OUTCOME_LABEL[o] for o in OUTCOME_ORDER], fontsize=8.0)
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER, fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, len(OUTCOME_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def draw_masking_curve(
    ax: mpl.axes.Axes,
    deltas: pd.DataFrame,
    column: str,
    ylabel: str,
) -> None:
    for method in METHOD_ORDER:
        subset = deltas.loc[deltas["da_method"].eq(method)]
        series = subset.groupby("masking_fraction")[column].median()
        ax.plot(
            series.index.to_numpy(float),
            series.to_numpy(float),
            marker=METHOD_MARKER[method],
            markersize=3.4,
            color=METHOD_COLOR[method],
            label=method,
            linewidth=1.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    ax.axhline(0.0, color="#555555", linewidth=0.9, linestyle="-", zorder=1)
    ax.set_xlabel("Masking fraction")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    clean_axis(ax)


def main() -> None:
    set_publication_style()
    improvement = pd.read_csv(IMPROVEMENT_CSV)
    deltas = pd.read_csv(DELTAS_CSV)

    n_settings = int(
        deltas.loc[deltas["da_method"].eq("LinDA")]
        .groupby(["effect_size", "masking_fraction"]).ngroups
    )
    if n_settings != 99:
        raise ValueError(f"Expected 99 parameter-grid settings, found {n_settings}")

    fig = plt.figure(figsize=(7.09, 5.55))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.05, 1.0],
        hspace=0.60, wspace=0.30,
        left=0.115, right=0.925, top=0.925, bottom=0.155,
    )

    ax_a = fig.add_subplot(gs[0, :])
    image = draw_matrix(ax_a, improvement)
    ax_a.set_title(
        "Fraction of the 99 effect-size × masking settings improved by completion",
        fontsize=9.5, fontweight="bold", pad=9,
    )

    cbar = fig.colorbar(image, ax=ax_a, fraction=0.026, pad=0.018)
    cbar.set_label("Proportion of settings improved", fontsize=7.8, labelpad=4)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=7.5, length=2)
    cbar.outline.set_visible(False)

    ax_b = fig.add_subplot(gs[1, 0])
    draw_masking_curve(ax_b, deltas, "delta_F1", r"Median $\Delta$F1")
    ax_b.set_title("Corrected ΔF1 by masking fraction", fontsize=9.0,
                   fontweight="bold", pad=6)

    ax_c = fig.add_subplot(gs[1, 1])
    draw_masking_curve(ax_c, deltas, "delta_recall", r"Median $\Delta$Recall")
    ax_c.set_title("ΔRecall by masking fraction", fontsize=9.0,
                   fontweight="bold", pad=6)

    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.52, 0.005),
        ncol=4, frameon=False, fontsize=8.2, handlelength=1.9,
        columnspacing=1.8,
    )

    for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
        panel_letter(fig, ax, letter)

    for suffix, kwargs in (
        (".pdf", {}),
        (".png", {"dpi": 400}),
        (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_adobe_deflate"}}),
    ):
        fig.savefig(OUT / f"{STEM}{suffix}", **kwargs)
    plt.close(fig)

    # Source data for the supplement, so the panel is reproducible from a table.
    source = improvement.copy()
    source["percent_improved"] = (source["proportion_improved"] * 100).round(1)
    source.to_csv(OUT / f"{STEM}_source_data.csv", index=False)

    masking = (
        deltas.groupby(["da_method", "masking_fraction"])[
            ["delta_F1", "delta_recall", "benefit_FDR"]
        ]
        .median()
        .round(4)
        .reset_index()
    )
    masking.to_csv(OUT / f"{STEM}_masking_resolved_medians.csv", index=False)
    print(f"Wrote {STEM}.[pdf|png|tiff] and source data to {OUT}")


if __name__ == "__main__":
    main()
