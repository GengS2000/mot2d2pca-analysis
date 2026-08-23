"""Create Supplementary Figure S9 from the audited three-engine source table.

The figure summarizes antibiotic-exposure results for LinDA, MaAsLin2, and
glmmPQL in DIABIMMUNE and ECAM.  The prespecified significance threshold is
q < 0.05 in both cohorts; q = 0.10 is shown only as a trend reference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "tables" / "source_csv" / "Figure_S9_antibiotic_three_engine_source.csv"
OUTPUT_STEM = HERE / "Figure_S9_antibiotic_exposure_null"

OBSERVED = "#9AA0A6"
COMPLETED = "#344563"
CONNECTOR = "#BCC2C8"
TREND = "#D55E00"
THRESHOLD = "#5F6368"
METHODS = ["LinDA", "MaAsLin2", "glmmPQL"]
COHORTS = ["DIABIMMUNE", "ECAM"]


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5F6368")
    ax.spines["bottom"].set_color("#5F6368")
    ax.tick_params(colors="#333333", labelsize=8)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6, zorder=0)


def _q_label(q):
    return f"{q:.3f}"


def make_figure():
    data = pd.read_csv(SOURCE)
    if int(data["n_q_lt_0_05"].sum()) != 0:
        raise ValueError("Figure S9 expects zero q<0.05 antibiotic-associated taxa.")

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(8.2, 3.6),
        gridspec_kw={"width_ratios": [1.7, 1.0]},
    )

    # Panel A: minimum q across all taxa for all 12 cohort-engine-representation analyses.
    positions = []
    labels = []
    x = 0
    for cohort in COHORTS:
        for method in METHODS:
            positions.append(x)
            labels.append(method)
            g = data[(data["cohort"] == cohort) & (data["method"] == method)]
            obs = g[g["representation"] == "observed"].iloc[0]
            imp = g[g["representation"] == "completed"].iloc[0]
            y_obs = -np.log10(obs["min_q"])
            y_imp = -np.log10(imp["min_q"])
            ax_a.plot([x - 0.12, x + 0.12], [y_obs, y_imp], color=CONNECTOR, linewidth=1.2, zorder=2)
            ax_a.scatter(x - 0.12, y_obs, s=42, color=OBSERVED, edgecolor="white", linewidth=0.7, zorder=3)
            ax_a.scatter(x + 0.12, y_imp, s=46, marker="s", color=COMPLETED,
                         edgecolor="white", linewidth=0.7, zorder=3)
            ax_a.text(x - 0.17, y_obs + 0.025, _q_label(obs["min_q"]), ha="right", va="bottom",
                      fontsize=6.2, color="#5F6368")
            ax_a.text(x + 0.17, y_imp + 0.025, _q_label(imp["min_q"]), ha="left", va="bottom",
                      fontsize=6.2, color=COMPLETED)
            x += 1
        x += 0.7

    q05_y = -np.log10(0.05)
    q10_y = -np.log10(0.10)
    ax_a.axhline(q05_y, color=THRESHOLD, linestyle=(0, (2, 2)), linewidth=1.0, zorder=1)
    ax_a.axhline(q10_y, color=TREND, linestyle=(0, (5, 3)), linewidth=0.9, zorder=1)
    ax_a.text(-0.46, q05_y, "q=0.05", ha="left", va="bottom",
              fontsize=7, color=THRESHOLD)
    ax_a.text(-0.46, q10_y, "q=0.10 (trend only)", ha="left", va="bottom",
              fontsize=7, color=TREND)
    ax_a.set_xticks(positions)
    ax_a.set_xticklabels(labels, fontsize=7)
    ax_a.text(positions[1], -0.15, "DIABIMMUNE", transform=ax_a.get_xaxis_transform(),
              ha="center", va="top", fontsize=7.5, fontweight="bold", color="#444444")
    ax_a.text(positions[4], -0.15, "ECAM", transform=ax_a.get_xaxis_transform(),
              ha="center", va="top", fontsize=7.5, fontweight="bold", color="#444444")
    ax_a.set_xlim(-0.55, positions[-1] + 0.55)
    ax_a.set_ylim(-0.04, 1.47)
    ax_a.set_ylabel(r"$-\log_{10}$(minimum BH-adjusted q)", fontsize=9)
    ax_a.set_title("A  No antibiotic-associated taxon reached q<0.05", loc="left",
                   fontsize=10, fontweight="bold")
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=OBSERVED,
                   markeredgecolor="white", label="observed"),
            Line2D([0], [0], marker="s", linestyle="", markerfacecolor=COMPLETED,
                   markeredgecolor="white", label="MOT-completed"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=7,
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.9,
    )
    _style_axis(ax_a)

    # Panel B: cross-engine magnitude and representation check for T0487.
    t0487 = data[data["cohort"] == "DIABIMMUNE"]
    bx = np.arange(len(METHODS))
    for i, method in enumerate(METHODS):
        g = t0487[t0487["method"] == method]
        obs = g[g["representation"] == "observed"].iloc[0]
        imp = g[g["representation"] == "completed"].iloc[0]
        ax_b.plot([i - 0.11, i + 0.11], [obs["t0487_effect"], imp["t0487_effect"]],
                  color=CONNECTOR, linewidth=1.4, zorder=2)
        ax_b.scatter(i - 0.11, obs["t0487_effect"], s=44, color=OBSERVED,
                     edgecolor="white", linewidth=0.7, zorder=3)
        ax_b.scatter(i + 0.11, imp["t0487_effect"], s=48, marker="s", color=COMPLETED,
                     edgecolor="white", linewidth=0.7, zorder=3)
        ax_b.text(i + 0.11, imp["t0487_effect"] + 0.055,
                  f"q={imp['t0487_q']:.3f}", ha="center", va="bottom",
                  fontsize=6.5, color=COMPLETED)

    ax_b.axhline(0, color=THRESHOLD, linewidth=0.9)
    ax_b.set_xticks(bx)
    ax_b.set_xticklabels(METHODS, fontsize=8)
    ax_b.set_xlim(-0.5, len(METHODS) - 0.5)
    ax_b.set_ylim(-0.05, 0.91)
    ax_b.set_ylabel("effect estimate for T0487", fontsize=9)
    ax_b.set_title("B  T0487 remains positive across engines", loc="left",
                   fontsize=10, fontweight="bold")
    ax_b.text(
        0.02,
        0.98,
        "DIABIMMUNE; q=0.089 is trend-level",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="#444444",
    )
    _style_axis(ax_b)

    fig.tight_layout(w_pad=2.0, rect=(0, 0.07, 1, 1))
    for ext, dpi in (("png", 600), ("tiff", 600), ("pdf", None), ("svg", None)):
        kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(OUTPUT_STEM.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
