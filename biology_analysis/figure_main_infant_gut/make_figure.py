"""
Build the redesigned main figure for the infant-gut MOT-(2D)^2 PCA results.

Panel order follows the results logic (landing point: joint analysis of the
observed and completed grids acts as a biological structure stress test):
  a  design schematic
  b  real-data 2 x 2 hit-count test
  c  literature gold-standard scorecard
  d  DIABIMMUNE counterexample resolved by data/significance decoupling
  e  temporal-fidelity hinge: completion regularizes trajectories, and the
     observed/completed comparison distinguishes cohort-coherent structure from
     transient dynamics, technical artifacts, and sparse boundary calls.
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

import figstyle as S
from figstyle import C

HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.join(HERE, "panel_data")
OUT = os.path.join(HERE, "output")
SVG = os.path.join(HERE, "svg")
os.makedirs(OUT, exist_ok=True)
os.makedirs(SVG, exist_ok=True)
S.apply_rc()


# ---------------------------------------------------------------- helpers ----
def box(ax, x, y, w, h, fc, ec=None, lw=0.9, rounding=0.035, alpha=1.0, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec or fc, facecolor=fc, alpha=alpha, zorder=z))


def arrow(ax, x0, y0, x1, y1, color=C["ink"], lw=1.4, z=3, mut=9):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>",
        mutation_scale=mut, lw=lw, color=color, zorder=z))


def minigrid(ax, x, y, w, h, n=5, fill=None, color=C["base"],
             edge="#D0D5DB", z=2):
    cw, ch = w / n, h / n
    for i in range(n):
        for j in range(n):
            filled = True if fill is None else (i, j) in fill
            ax.add_patch(Rectangle(
                (x + j * cw, y + i * ch), cw * 0.86, ch * 0.86,
                facecolor=color if filled else "white",
                edgecolor=edge, linewidth=0.45, zorder=z))


def panel_label(ax, letter, dx=-0.04, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="right", color=C["ink"])


def q_fmt(q):
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "<.001"
    return f"{q:.2f}"


def role_label(role):
    return {
        "recover": "known signal recovered",
        "confirm": "observed bio benchmark",
        "noeffect": "negative control",
        "suppress": "artifact suppressed",
    }[role]


# ================================================================ panel a ====
def draw_a(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def tensor_cube(x, y, w, h, dx, dy, mode):
        ncol, nrow = 5, 4
        sparse_cells = {(0, 0), (0, 3), (1, 1), (2, 0), (2, 2), (3, 4)}
        top = [(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)]
        side = [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)]
        ax.add_patch(Polygon(top, closed=True, facecolor="#F3F5F7", edgecolor="#CDD2D7", lw=0.7, zorder=1))
        ax.add_patch(Polygon(side, closed=True, facecolor="#E7EBEF", edgecolor="#CDD2D7", lw=0.7, zorder=1))
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="#CDD2D7", lw=0.7, zorder=2))
        for i in range(nrow):
            for j in range(ncol):
                observed = (i, j) in sparse_cells or mode == "dense"
                if mode == "dense":
                    fc = C["imputed"]
                    alpha = 0.92
                elif observed:
                    fc = C["base"]
                    alpha = 0.88
                else:
                    fc = "white"
                    alpha = 1.0
                ax.add_patch(Rectangle((x + j * w / ncol, y + i * h / nrow),
                                       w / ncol, h / nrow,
                                       facecolor=fc, alpha=alpha,
                                       edgecolor="#D7DCE1", lw=0.45, zorder=3))
        for k in range(1, 4):
            t = k / 4
            ax.plot([x + w + dx * t, x + w + dx * t],
                    [y + dy * t, y + h + dy * t], color="#D7DCE1", lw=0.45, zorder=2)
            ax.plot([x + dx * t, x + w + dx * t],
                    [y + h + dy * t, y + h + dy * t], color="#D7DCE1", lw=0.45, zorder=2)
        ax.plot([x, x + dx, x + w + dx, x + w], [y, y + dy, y + dy, y],
                color="#B8C0C8", lw=0.7, zorder=2)

    def tensor_cube_inset(bounds, mode):
        iax = ax.inset_axes(bounds)
        iax.set_xlim(0, 1)
        iax.set_ylim(0, 1)
        iax.set_aspect("equal")
        iax.axis("off")
        x, y, w, h, dx, dy = 0.13, 0.18, 0.55, 0.55, 0.22, 0.18
        ncol, nrow = 5, 4
        sparse_cells = {(0, 0), (0, 3), (1, 1), (2, 0), (2, 2), (3, 4)}
        top = [(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)]
        side = [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)]
        iax.add_patch(Polygon(top, closed=True, facecolor="#F3F5F7", edgecolor="#C9D0D7", lw=0.8, zorder=1))
        iax.add_patch(Polygon(side, closed=True, facecolor="#E7EBEF", edgecolor="#C9D0D7", lw=0.8, zorder=1))
        iax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="#C9D0D7", lw=0.8, zorder=2))
        for i in range(nrow):
            for j in range(ncol):
                observed = mode == "dense" or (i, j) in sparse_cells
                if mode == "dense":
                    fc, alpha = C["imputed"], 0.92
                elif observed:
                    fc, alpha = C["base"], 0.88
                else:
                    fc, alpha = "white", 1.0
                iax.add_patch(Rectangle((x + j * w / ncol, y + i * h / nrow),
                                        w / ncol, h / nrow,
                                        facecolor=fc, alpha=alpha,
                                        edgecolor="#D7DCE1", lw=0.5, zorder=3))
        for k in range(1, 4):
            t = k / 4
            iax.plot([x + w + dx * t, x + w + dx * t],
                     [y + dy * t, y + h + dy * t], color="#D7DCE1", lw=0.5, zorder=2)
            iax.plot([x + dx * t, x + w + dx * t],
                     [y + h + dy * t, y + h + dy * t], color="#D7DCE1", lw=0.5, zorder=2)
        iax.plot([x, x + dx, x + w + dx, x + w, x],
                 [y, y + dy, y + dy, y, y], color="#AEB7C0", lw=0.8, zorder=4)

    def curve_arrow(x0, y0, x1, y1, color, rad=0.15, lw=1.3):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
            connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, zorder=5))

    yc = 0.40
    tensor_cube_inset([0.050, 0.125, 0.155, 0.610], "sparse")
    ax.text(0.128, 0.720, "subject x time x taxa", ha="center", va="bottom",
            fontsize=5.8, color="0.35")
    ax.text(0.128, 0.125, "observed grid\n(base_raw)", ha="center", va="top",
             fontsize=6.2, color=C["base"], fontweight="bold")
    ax.text(0.184, 0.595, "subject", ha="left", va="center", fontsize=4.8, color="0.45", rotation=27)
    ax.text(0.128, 0.218, "time", ha="center", va="top", fontsize=4.8, color="0.45")

    arrow(ax, 0.217, yc, 0.285, yc, lw=1.5)
    box(ax, 0.292, yc - 0.105, 0.130, 0.210, fc="white", ec=C["filter"],
        lw=1.5, rounding=0.040)
    ax.text(0.357, yc, "MOT-(2D)$^2$ PCA", ha="center", va="center",
            fontsize=7.0, color=C["filter"], fontweight="bold")

    arrow(ax, 0.438, yc, 0.505, yc, lw=1.5)
    tensor_cube_inset([0.505, 0.125, 0.155, 0.610], "dense")
    ax.text(0.583, 0.125, "MOT-(2D)$^2$ PCA\ncompleted grid", ha="center", va="top",
             fontsize=6.2, color=C["imputed"], fontweight="bold")

    gx, gy, gw, gh = 0.875, 0.235, 0.105, 0.335
    ax.add_patch(Rectangle((gx, gy), gw, gh, facecolor="white", edgecolor=C["ink"], lw=0.8, zorder=3))
    ax.plot([gx, gx + gw], [gy + gh / 2] * 2, color=C["ink"], lw=0.6, zorder=4)
    ax.plot([gx + gw / 2] * 2, [gy, gy + gh], color=C["ink"], lw=0.6, zorder=4)
    ax.text(gx + gw * 0.25, gy + gh + 0.035, "observed", ha="center", va="bottom",
             fontsize=5.2, color=C["base"])
    ax.text(gx + gw * 0.75, gy + gh + 0.035, "completed", ha="center", va="bottom",
             fontsize=5.2, color=C["imputed"])
    for px, py, fc in [
        (gx + gw * 0.25, gy + gh * 0.75, C["base"]),
        (gx + gw * 0.75, gy + gh * 0.75, C["imputed"]),
        (gx + gw * 0.25, gy + gh * 0.25, C["base"]),
        (gx + gw * 0.75, gy + gh * 0.25, C["imputed"]),
    ]:
        ax.scatter([px], [py], s=16, color=fc, edgecolor="white", linewidth=0.4, zorder=5)

    box(ax, 0.690, 0.475, 0.110, 0.115, fc=C["linda"], ec=C["linda"], alpha=0.16, rounding=0.025)
    box(ax, 0.690, 0.270, 0.110, 0.115, fc=C["maaslin"], ec=C["maaslin"], alpha=0.16, rounding=0.025)
    ax.text(0.745, 0.535, "LinDA", ha="center", va="center", fontsize=6.8,
            color=C["linda"], fontweight="bold")
    ax.text(0.745, 0.330, "MaAsLin2", ha="center", va="center", fontsize=6.8,
            color=C["maaslin"], fontweight="bold")
    arrow(ax, 0.806, 0.533, gx - 0.006, gy + gh * 0.75, color=C["linda"], lw=1.2, mut=7)
    arrow(ax, 0.806, 0.328, gx - 0.006, gy + gh * 0.25, color=C["maaslin"], lw=1.2, mut=7)


# ================================================================ panel b ====
def draw_b(ax):
    df = pd.read_csv(os.path.join(PD, "panel_b_hitcounts.csv"))
    order = [
        "Solid-food timing (22 foods)",
        "Delivery mode",
        "Feeding (breast/formula)",
        "Antibiotics (DIABIMMUNE)",
        "Antibiotics (ECAM)",
    ]
    label_short = {
        "Solid-food timing (22 foods)": "Solid-food\ntiming",
        "Delivery mode": "Delivery\nmode",
        "Feeding (breast/formula)": "Feeding\nmode",
        "Antibiotics (DIABIMMUNE)": "Antibiotics\nDIABIMMUNE",
        "Antibiotics (ECAM)": "Antibiotics\nECAM",
    }
    y0 = len(order) - 1
    yticks, ylabels = [], []
    for idx, lab in enumerate(order):
        y = y0 - idx
        ax.axhspan(y - 0.47, y + 0.47, color="#FAFBFC" if idx % 2 == 0 else "white", zorder=0)
        for method, off, col in [("LinDA", 0.16, C["linda"]),
                                 ("MaAsLin2", -0.16, C["maaslin"])]:
            r = df[(df["label"] == lab) & (df["method"] == method)].iloc[0]
            b, imp = int(r["base_hits"]), int(r["imputed_hits"])
            yy = y + off
            if b != imp:
                ax.annotate("", xy=(imp, yy), xytext=(b, yy),
                            arrowprops=dict(arrowstyle="-|>", lw=1.6, color=col),
                            zorder=2)
            ax.plot(b, yy, "o", mfc="white", mec=col, mew=1.3, ms=5.2, zorder=3)
            ax.plot(imp, yy, "o", mfc=col, mec=col, mew=1.0, ms=5.2, zorder=3)
            if abs(imp - b) >= 10:
                ax.text(imp + (2.2 if imp >= b else -1.0), yy,
                        f"{b}->{imp}", ha="left" if imp >= b else "right",
                        va="center", fontsize=5.5, color=col, fontweight="bold")
        yticks.append(y)
        ylabels.append(label_short[lab])

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("significant taxa: observed -> MOT-completed", labelpad=2)
    ax.set_xlim(-2, 55)
    ax.set_ylim(-0.7, y0 + 0.7)
    ax.grid(axis="x", color="#E6E8EB", lw=0.55)
    S.clean(ax)
    ax.text(0.98, 1.015, "DIABIMMUNE q<0.10; ECAM q<0.05",
            transform=ax.transAxes, fontsize=5.3, color="0.45", ha="right")
    handles = [
        Line2D([0], [0], color=C["linda"], lw=2, label="LinDA"),
        Line2D([0], [0], color=C["maaslin"], lw=2, label="MaAsLin2"),
        Line2D([0], [0], marker="o", ls="", mfc="white", mec="0.4",
               label="observed grid"),
        Line2D([0], [0], marker="o", ls="", mfc="0.4", mec="0.4",
               label="MOT-completed grid"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=5.6,
              handlelength=1.3, borderpad=0.1)


# ================================================================ panel c ====
def draw_c(ax):
    """Four evidence lanes with a distinct miniature graphic for each claim."""
    # case | observed | completed | conclusion columns
    x = [0.00, 0.295, 0.530, 0.765, 1.00]
    top, header_h, bottom = 0.96, 0.13, 0.05
    row_h = (top - header_h - bottom) / 4
    centers = [top - header_h - (i + 0.5) * row_h for i in range(4)]
    accents = [C["true"], C["null"], C["false"], C["filter"]]
    cases = [
        "ECAM delivery\nBacteroides T0024",
        "Antibiotic exposure\n(prespecified contrasts)",
        "DIABIMMUNE carrot timing\nchloroplast reads",
        "ECAM Bacteroides\ntemporal profile",
    ]
    conclusions = [
        "Stable depletion;\nsupport strengthened",
        "No reproducible\nmean shift",
        "Non-target signal\nattenuated",
        "Temporal attenuation\nsmoothed",
    ]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, label in enumerate(("Evidence case", "Observed", "MOT-completed", "Joint interpretation")):
        ax.text(x[i] + 0.012, top, label, ha="left", va="top", fontsize=5.8,
                color=C["ink"], fontweight="bold")
    ax.plot([0, 1], [top - header_h, top - header_h], color="#AEB5BD", lw=0.75)
    for xv in x[1:-1]:
        ax.plot([xv, xv], [bottom, top - 0.012], color="#E0E4E8", lw=0.55)

    for i, yc in enumerate(centers):
        y0, y1 = yc - row_h / 2, yc + row_h / 2
        ax.add_patch(Rectangle((0, y0), 1, row_h,
                               facecolor="#FAFBFC" if i % 2 == 0 else "white",
                               edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((0, y0 + 0.018), 0.010, row_h - 0.036,
                               facecolor=accents[i], edgecolor="none", zorder=2))
        ax.text(0.019, yc, cases[i], ha="left", va="center", fontsize=5.55,
                color=C["ink"], fontweight="bold", linespacing=1.22)
        ax.text(x[3] + 0.017, yc, conclusions[i], ha="left", va="center",
                fontsize=5.55, color=accents[i], fontweight="bold", linespacing=1.22)
        ax.plot([0, 1], [y0, y0], color="#E0E4E8", lw=0.55)

    # Row 1: paired effect-size mini forest plots. The common zero line makes
    # direction stability visually explicit without implying independent replication.
    def effect_glyph(x0, x1, beta, q, filled, significant):
        lo, hi = -4.5, 0.5
        mapx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
        y = centers[0] + 0.012
        zero = mapx(0)
        ax.plot([x0, x1], [y, y], color="#C7CDD4", lw=0.8, zorder=2)
        ax.plot([zero, zero], [y - 0.035, y + 0.035], color="#727980", lw=0.7, zorder=2)
        px = mapx(beta)
        ax.plot([px, zero], [y, y], color=C["down"], lw=1.25, zorder=3)
        ax.scatter([px], [y], s=24, facecolor=C["imputed"] if filled else "white",
                   edgecolor=C["true"] if significant else C["down"], linewidth=1.05, zorder=4)
        ax.text(x0, y - 0.047, f"beta {beta:.2f}; q={q}", ha="left", va="top",
                fontsize=4.55, color="0.32")
        if significant:
            ax.text(px + 0.009, y + 0.028, "*", ha="left", va="bottom",
                    fontsize=8.0, color=C["true"], fontweight="bold")

    effect_glyph(x[1] + 0.025, x[2] - 0.025, -3.60, ".063", False, False)
    effect_glyph(x[2] + 0.025, x[3] - 0.025, -3.34, ".020", True, True)

    # Row 2: two-engine agreement matrix. Open symbols deliberately show that
    # neither representation provides a cross-engine reproducible antibiotic call.
    def engine_matrix(x0, x1, completed=False):
        yy = [centers[1] + 0.025, centers[1] - 0.025]
        labels = ["L", "M"]
        for y, label in zip(yy, labels):
            ax.text(x0, y, label, ha="left", va="center", fontsize=4.7,
                    color=C["linda"] if label == "L" else C["maaslin"], fontweight="bold")
            ax.scatter([x0 + 0.050], [y], s=21, facecolor="white", edgecolor="0.42",
                       linewidth=0.9, zorder=3)
        ax.text(x0 + 0.072, centers[1], "no shared\ncall", ha="left", va="center",
                fontsize=4.55, color="0.38", linespacing=1.05)
        if completed:
            ax.text(x1 - 0.002, centers[1] - 0.053, "one boundary\nOTU only", ha="right",
                    va="top", fontsize=3.9, color="0.50", style="italic")

    engine_matrix(x[1] + 0.030, x[2] - 0.020)
    engine_matrix(x[2] + 0.030, x[3] - 0.020, completed=True)

    # Row 3: paired -log10(q) bars show the marked loss of organellar support.
    def signal_bars(x0, x1, vals, labels, muted=False):
        baseline = centers[2] - 0.045
        avail = (x1 - x0) * 0.80
        bh = 0.072
        for k, (value, label) in enumerate(zip(vals, labels)):
            xx = x0 + 0.020 + k * (avail / 2)
            h = bh * min(value, 3.0) / 3.0
            ax.add_patch(Rectangle((xx, baseline), avail / 4.2, h,
                                   facecolor="#D8DDE3" if muted else C["false"],
                                   edgecolor="none", zorder=3))
            ax.text(xx + avail / 8.4, baseline - 0.012, label, ha="center", va="top",
                    fontsize=4.15, color="0.34")
        ax.plot([x0 + 0.015, x0 + 0.015 + avail], [baseline, baseline], color="#BAC1C8", lw=0.55)

    # Small leaf: an annotation of plant-origin 16S, not a decorative icon.
    leaf_x, leaf_y = 0.025, centers[2] + 0.030
    ax.add_patch(Ellipse((leaf_x, leaf_y), 0.018, 0.037, angle=35, facecolor="#79A96B", edgecolor="none"))
    ax.add_patch(Ellipse((leaf_x + 0.012, leaf_y + 0.006), 0.016, 0.032, angle=-30, facecolor="#79A96B", edgecolor="none"))
    ax.plot([leaf_x - 0.005, leaf_x + 0.017], [leaf_y - 0.026, leaf_y + 0.020], color="#5D8F53", lw=0.75)
    signal_bars(x[1] + 0.020, x[2] - 0.015, [2.7, 4.0], [".002", "<.001"])
    signal_bars(x[2] + 0.020, x[3] - 0.015, [0.01, 0.01], ["~1", "~1"], muted=True)

    # Row 4: paired C/V sparklines refer to the detailed trajectories in panel e.
    def ratio_sparkline(x0, x1, a, b, label, col):
        y0, y1 = centers[3] - 0.045, centers[3] + 0.050
        rlo, rhi = 0.25, 0.90
        mapy = lambda v: y0 + (v - rlo) / (rhi - rlo) * (y1 - y0)
        xx0, xx1 = x0 + 0.025, x1 - 0.025
        ax.plot([xx0, xx1], [mapy(0.5)] * 2, color="#D3D8DD", lw=0.55)
        ax.plot([xx0, xx1], [mapy(a), mapy(b)], color=col, lw=1.55, zorder=3)
        ax.scatter([xx0, xx1], [mapy(a), mapy(b)], s=15, color=col, edgecolor="white", linewidth=0.45, zorder=4)
        ax.text((xx0 + xx1) / 2, y0 - 0.012, label, ha="center", va="top", fontsize=4.45, color="0.34")

    ratio_sparkline(x[1], x[2], 0.36, 0.80, "C/V  .36 -> .80", C["true"])
    ratio_sparkline(x[2], x[3], 0.49, 0.51, "C/V  .49 -> .51", C["filter"])


# ================================================================ panel d ====
def draw_d(ax_eff, ax_sig):
    long = pd.read_csv(os.path.join(PD, "panel_d_decoupling_long.csv"))
    summ = pd.read_csv(os.path.join(PD, "panel_c_decoupling.csv"))
    methods = ["LinDA", "MaAsLin2"]
    positions = [0.00, 0.42, 1.15, 1.57]
    keys = [("LinDA", "base"), ("LinDA", "imputed"),
            ("MaAsLin2", "base"), ("MaAsLin2", "imputed")]
    colors = [C["base"], C["imputed"], C["base"], C["imputed"]]

    data = [long[(long["method"] == m) & (long["pipeline"] == p)]["abs_effect"].to_numpy()
            for m, p in keys]
    bp = ax_eff.boxplot(data, positions=positions, widths=0.30, patch_artist=True,
                        showfliers=False, whis=(10, 90),
                        medianprops=dict(color="white", lw=1.2),
                        boxprops=dict(linewidth=0.8, edgecolor="white"),
                        whiskerprops=dict(linewidth=0.7, color="0.25"),
                        capprops=dict(linewidth=0.7, color="0.25"))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.92)
        patch.set_edgecolor("white")
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(1.15)
    ax_eff.set_ylabel("|effect size|")
    ax_eff.set_ylim(0, 1.35)
    ax_eff.set_xticks([0.21, 1.36])
    ax_eff.set_xticklabels(methods)
    ax_eff.grid(axis="y", color="#E6E8EB", lw=0.55)
    S.clean(ax_eff)
    for x, txt, col in [(0.00, "base", C["base"]), (0.42, "imp", C["imputed"]),
                        (1.15, "base", C["base"]), (1.57, "imp", C["imputed"])]:
        display = "observed" if txt == "base" else "completed"
        ax_eff.text(x, -0.13, display, transform=ax_eff.get_xaxis_transform(),
                    ha="center", va="top", fontsize=5.4, color=col)
    fold_labels = []
    for method in methods:
        sb = summ[(summ["method"] == method) & (summ["pipeline"] == "base")]["median_abs_effect"].iloc[0]
        si = summ[(summ["method"] == method) & (summ["pipeline"] == "imputed")]["median_abs_effect"].iloc[0]
        fold_labels.append(f"{si / sb:.1f}x")
    for center, label in zip((0.21, 1.36), fold_labels):
        ax_eff.annotate(label, xy=(center, 1.18), xytext=(center, 1.30),
                        ha="center", va="center", fontsize=6.0, color=C["ink"],
                        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6))
    x = np.arange(len(methods))
    width = 0.34
    for k, (pipe, col) in enumerate([("base", C["base"]), ("imputed", C["imputed"])]):
        vals = [int(summ[(summ["method"] == m) & (summ["pipeline"] == pipe)]["n_sig_q10"].iloc[0])
                for m in methods]
        bars = ax_sig.bar(x + (k - 0.5) * width, vals, width=width,
                          color=col, edgecolor="white", linewidth=0.7,
                          label="observed" if pipe == "base" else "MOT-completed")
        for bar, val in zip(bars, vals):
            ax_sig.text(bar.get_x() + bar.get_width() / 2, val + 1.0, str(val),
                        ha="center", va="bottom", fontsize=6.0)
    ax_sig.set_ylabel("significant taxa (q<0.10)")
    ax_sig.set_xticks(x)
    ax_sig.set_xticklabels(methods)
    linda_imp = int(summ[(summ["method"] == "LinDA") &
                         (summ["pipeline"] == "imputed")]["n_sig_q10"].iloc[0])
    maas_imp = int(summ[(summ["method"] == "MaAsLin2") &
                        (summ["pipeline"] == "imputed")]["n_sig_q10"].iloc[0])
    ymax = max(53, linda_imp + 8)
    ax_sig.set_ylim(0, ymax)
    S.clean(ax_sig)
    ax_sig.annotate("increases", xy=(0 + width / 2, linda_imp),
                    xytext=(0.34, ymax - 2),
                    ha="center", fontsize=6, color=C["linda"],
                    arrowprops=dict(arrowstyle="-|>", color=C["linda"], lw=1))
    ax_sig.annotate("decreases", xy=(1 + width / 2, maas_imp),
                    xytext=(1.34, max(maas_imp + 16, 30)),
                    ha="center", fontsize=6, color=C["maaslin"],
                    arrowprops=dict(arrowstyle="-|>", color=C["maaslin"], lw=1))
    ax_sig.legend(frameon=False, loc="upper right", fontsize=5.6, handlelength=1.0)

# ================================================================ panel e ====
def _draw_bacteroides(ax, bac, kind, ratios):
    """kind = 'base' (observed dots) | 'imp' (MOT-completed regularized lines)."""
    col = {"Vaginal": C["true"], "Cesarean": C["false"]}
    for delv in ("Vaginal", "Cesarean"):
        d = bac[bac["delivery"] == delv].sort_values("month")
        if kind == "base":
            dd = d[d["base_n"] >= 2]
            ax.plot(dd["month"], dd["base_frac"] * 100, "o-", color=col[delv],
                    ms=3.0, lw=1.0, label=delv)
        else:
            ax.plot(d["month"], d["imp_frac"] * 100, "-", color=col[delv],
                    lw=1.9, label=delv)
    ax.axvline(12, color="0.55", ls=":", lw=0.8)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 12, 24])
    ax.set_xlabel("age (months)", labelpad=1)
    ax.grid(axis="y", color="#E6E8EB", lw=0.5)
    S.clean(ax)
    r1, r2 = ratios
    if kind == "base":
        ax.set_ylabel("Bacteroides (%)", labelpad=1)
        ax.set_title("observed", loc="left", fontsize=6.6, color=C["ink"])
        ax.text(0.5, 1.16, f"C/V {r1:.2f} → {r2:.2f}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=5.8, color=C["true"], fontweight="bold")
        ax.text(0.5, 1.015, "deficit closes (Bokulich)", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=5.2, color="0.40")
        ax.legend(frameon=False, fontsize=5.2, loc="lower right", handlelength=1.0,
                  borderpad=0.1, labelspacing=0.2)
    else:
        ax.set_title("MOT-completed", loc="left", fontsize=6.6, color=C["ink"])
        ax.text(0.5, 1.16, f"C/V {r1:.2f} → {r2:.2f}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=5.8, color=C["false"], fontweight="bold")
        ax.text(0.5, 1.015, "regularized offset (timing not inferred)", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=5.2, color="0.40")


def _draw_verdict(ax, verdict):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    role_color = {"true": C["true"], "null": C["null"],
                  "tempo": C["maaslin"], "false": C["false"]}
    n = len(verdict)
    top, bot = 0.97, 0.06
    row_h = (top - bot) / n
    ax.text(0.0, top + 0.045,
            "biological gain from observed + MOT-completed triangulation",
            ha="left", va="bottom", fontsize=5.8, color=C["filter"], style="italic")
    for idx, (_, row) in enumerate(verdict.iterrows()):
        y = top - (idx + 0.5) * row_h
        rc = role_color.get(row["role"], C["null"])
        if idx % 2 == 0:
            ax.axhspan(y - row_h * 0.48, y + row_h * 0.48, color="#FAFBFC", zorder=0)
        ax.add_patch(Rectangle((0.0, y - row_h * 0.34), 0.02, row_h * 0.68,
                               color=rc, clip_on=False, zorder=3))
        ax.text(0.045, y + 0.028, row["exposure"], ha="left", va="center",
                fontsize=6.4, color=C["ink"], fontweight="bold")
        ax.text(0.045, y - 0.030, row["call"], ha="left", va="center",
                fontsize=5.5, color=rc, fontweight="bold")
        ax.text(0.99, y + 0.028, "obs: " + row["observed"], ha="right", va="center",
                fontsize=4.7, color="0.45")
        ax.text(0.99, y - 0.030, "imp: " + row["completion"], ha="right", va="center",
                fontsize=4.7, color="0.55", style="italic")


def draw_e(ax):
    """Temporal-fidelity hinge: observed trajectories constrain completion."""
    ax.axis("off")
    bac = pd.read_csv(os.path.join(PD, "panel_e_ecam_bacteroides.csv"))
    pers = pd.read_csv(os.path.join(PD, "panel_e_persistence.csv"))
    bp = pers[pers["genus"] == "Bacteroides"].sort_values("window")
    obs_ratios = (bp["observed_CoverV"].iloc[0], bp["observed_CoverV"].iloc[1])
    imp_ratios = (bp["imputed_CoverV"].iloc[0], bp["imputed_CoverV"].iloc[1])

    ax.text(0.02, 1.02,
            "Observed trajectories retain the early-life attenuation regularized by completion",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
            color=C["ink"], style="italic")

    ax_obs = ax.inset_axes([0.055, 0.12, 0.365, 0.70])
    ax_imp = ax.inset_axes([0.575, 0.12, 0.365, 0.70])

    _draw_bacteroides(ax_obs, bac, "base", obs_ratios)
    _draw_bacteroides(ax_imp, bac, "imp", imp_ratios)
    ax.annotate("", xy=(0.555, 0.49), xytext=(0.440, 0.49),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color=C["filter"], lw=1.3))
    ax.text(0.4975, 0.55, "MOT-(2D)$^2$ PCA", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=5.3, color=C["filter"], fontweight="bold")


# =============================================================== compose =====
def compose():
    fig = plt.figure(figsize=(8.3, 9.1))
    gs = fig.add_gridspec(
        4, 12,
        height_ratios=[0.40, 1.45, 1.45, 1.42],
        hspace=0.30, wspace=1.05,
        left=0.09, right=0.96, top=0.955, bottom=0.060)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0:5])
    ax_c = fig.add_subplot(gs[1, 5:12])
    ax_d1 = fig.add_subplot(gs[2, 0:6])
    ax_d2 = fig.add_subplot(gs[2, 6:12])
    ax_e = fig.add_subplot(gs[3, :])

    draw_a(ax_a)
    draw_b(ax_b)
    draw_c(ax_c)
    draw_d(ax_d1, ax_d2)
    draw_e(ax_e)

    # Use capital panel letters and a common top baseline across all rows.
    panel_label(ax_a, "A", dx=-0.015, dy=1.10)
    panel_label(ax_b, "B", dx=-0.075, dy=1.10)
    panel_label(ax_c, "C", dx=-0.055, dy=1.10)
    panel_label(ax_d1, "D", dx=-0.060, dy=1.10)
    panel_label(ax_e, "E", dx=-0.020, dy=1.10)

    fig.savefig(os.path.join(OUT, "figure_main.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "figure_main.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "figure_main.png"), bbox_inches="tight", dpi=600)
    fig.savefig(os.path.join(OUT, "figure_main.tiff"), bbox_inches="tight", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    print("Wrote output/figure_main.{svg,pdf,png,tiff}")
    plt.close(fig)


def render_individual():
    specs = [
        ("a", (7.2, 1.6)),
        ("b", (3.6, 3.0)),
        ("c", (5.0, 3.0)),
        ("d", (6.6, 3.0)),
        ("e", (7.8, 3.2)),
    ]
    for name, size in specs:
        fig = plt.figure(figsize=size)
        if name == "a":
            ax = fig.add_axes([0.04, 0.16, 0.93, 0.70])
            draw_a(ax)
        elif name == "b":
            ax = fig.add_axes([0.20, 0.19, 0.75, 0.68])
            draw_b(ax)
        elif name == "c":
            ax = fig.add_axes([0.23, 0.16, 0.68, 0.70])
            draw_c(ax)
        elif name == "d":
            ax1 = fig.add_axes([0.10, 0.20, 0.40, 0.68])
            ax2 = fig.add_axes([0.58, 0.20, 0.38, 0.68])
            draw_d(ax1, ax2)
        else:
            ax = fig.add_axes([0.04, 0.14, 0.93, 0.74])
            draw_e(ax)
        fig.savefig(os.path.join(SVG, f"panel_{name}.svg"), bbox_inches="tight")
        plt.close(fig)
    print("Wrote svg/panel_[a-e].svg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--individual", action="store_true",
                    help="also emit per-panel SVGs")
    args = ap.parse_args()
    compose()
    if args.individual:
        render_individual()
