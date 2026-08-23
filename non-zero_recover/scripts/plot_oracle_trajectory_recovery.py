from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


FIG_W = 720
FIG_H = 620
METHOD = "penalized_2d2_pca"
BASELINE = "no_imputation"

COLORS = {
    "ink": "#1F2933",
    "muted": "#697386",
    "grid": "#D8DEE9",
    "baseline": "#6B7280",
    "pca": "#0072B2",
    "pca_light": "#D8ECF8",
    "positive": "#009E73",
    "negative": "#D55E00",
    "pair": "#C7CED9",
}

METHOD_LABELS = {
    BASELINE: "No imputation",
    METHOD: "Penalized 2D2-PCA",
}

MECH_LABELS = {
    "random_entry": "Random",
    "low_depth_sample": "Low-depth",
    "time_point": "Time point",
    "subject_specific_dropout": "Subject",
    "consecutive_time_window_dropout": "Window",
    "low_abundance_biased": "Low-abund.",
}

MECH_ORDER = [
    "random_entry",
    "low_depth_sample",
    "time_point",
    "subject_specific_dropout",
    "consecutive_time_window_dropout",
    "low_abundance_biased",
]


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.strip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def sign_test_pvalue(wins: int, n: int) -> float:
    """One-sided exact sign test: P(X >= wins), X ~ Binomial(n, 0.5)."""
    if n <= 0:
        return float("nan")
    return sum(math.comb(n, k) for k in range(wins, n + 1)) * 0.5**n


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p = NA"
    if p < 1e-40:
        return "p < 1e-40"
    if p < 1e-3:
        return f"p = {p:.1e}"
    return f"p = {p:.3f}"


def mean_ci(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, 0.0
    return mean, float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size))


def nice_ticks(ymin: float, ymax: float, n: int = 5) -> list[float]:
    if ymax <= ymin:
        return [ymin]
    raw = (ymax - ymin) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda x: abs(x * mag - raw)) * mag
    lo = math.floor(ymin / step) * step
    hi = math.ceil(ymax / step) * step
    ticks = []
    cur = lo
    while cur <= hi + step * 0.5:
        ticks.append(round(cur, 10))
        cur += step
    return ticks


def load_pairs(metrics_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(metrics_path)
    oracle = df[df["replacement_mode"].eq("oracle_mask")].copy()
    keys = ["mechanism", "frac", "repeat", "seed"]
    base = oracle[oracle["method"].eq(BASELINE)][
        keys + ["trajectory_rmse_affected", "lag1_autocorr_abs_error"]
    ].rename(
        columns={
            "trajectory_rmse_affected": "base_traj",
            "lag1_autocorr_abs_error": "base_lag",
        }
    )
    pca = oracle[oracle["method"].eq(METHOD)][
        keys + ["trajectory_rmse_affected", "lag1_autocorr_abs_error"]
    ].rename(
        columns={
            "trajectory_rmse_affected": "pca_traj",
            "lag1_autocorr_abs_error": "pca_lag",
        }
    )
    pairs = base.merge(pca, on=keys, how="inner")
    pairs["traj_improvement_pct"] = 100.0 * (pairs["base_traj"] - pairs["pca_traj"]) / pairs["base_traj"]
    pairs["lag_improvement_pct"] = 100.0 * (pairs["base_lag"] - pairs["pca_lag"]) / pairs["base_lag"]

    oracle_method = oracle[oracle["method"].isin([BASELINE, METHOD])].copy()
    return pairs, oracle_method


def summarize_metric(pairs: pd.DataFrame, base_col: str, method_col: str, improvement_col: str) -> dict[str, float]:
    valid = pairs[[base_col, method_col, improvement_col]].dropna()
    wins = int((valid[method_col] < valid[base_col]).sum())
    n = int(len(valid))
    return {
        "n": n,
        "baseline_mean": float(valid[base_col].mean()),
        "method_mean": float(valid[method_col].mean()),
        "mean_improvement_pct": float(valid[improvement_col].mean()),
        "median_improvement_pct": float(valid[improvement_col].median()),
        "wins": wins,
        "win_rate": wins / n if n else float("nan"),
        "sign_test_p_one_sided": sign_test_pvalue(wins, n),
    }


def save_stats(pairs: pd.DataFrame, outdir: Path) -> dict[str, dict[str, float]]:
    overall = {
        "trajectory_rmse_affected": summarize_metric(pairs, "base_traj", "pca_traj", "traj_improvement_pct"),
        "lag1_autocorr_abs_error": summarize_metric(pairs, "base_lag", "pca_lag", "lag_improvement_pct"),
    }

    stats_path = outdir / "oracle_mask_trajectory_recovery_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "n",
                "baseline_mean",
                "method_mean",
                "mean_improvement_pct",
                "median_improvement_pct",
                "wins",
                "win_rate",
                "sign_test_p_one_sided",
            ],
        )
        writer.writeheader()
        for metric, row in overall.items():
            writer.writerow({"metric": metric, **row})

    mech_path = outdir / "oracle_mask_trajectory_recovery_by_mechanism.csv"
    rows = []
    for mech, sub in pairs.groupby("mechanism", dropna=False):
        rows.append(
            {
                "mechanism": mech,
                "label": MECH_LABELS.get(str(mech), str(mech)),
                "n": len(sub),
                "trajectory_improvement_pct_mean": float(sub["traj_improvement_pct"].mean()),
                "trajectory_improvement_pct_ci95": mean_ci(sub["traj_improvement_pct"])[1],
                "lag1_improvement_pct_mean": float(sub["lag_improvement_pct"].mean()),
                "lag1_improvement_pct_ci95": mean_ci(sub["lag_improvement_pct"])[1],
            }
        )
    pd.DataFrame(rows).to_csv(mech_path, index=False)
    return overall


class PDFRenderer:
    def __init__(self, path: Path, width: int = FIG_W, height: int = FIG_H):
        self.path = path
        self.width = width
        self.height = height
        self.canvas = canvas.Canvas(str(path), pagesize=(width, height))

    def _y(self, y: float) -> float:
        return self.height - y

    def line(self, x1, y1, x2, y2, color=COLORS["ink"], width=1.0):
        self.canvas.setStrokeColor(HexColor(color))
        self.canvas.setLineWidth(width)
        self.canvas.line(x1, self._y(y1), x2, self._y(y2))

    def rect(self, x, y, w, h, fill=None, stroke=None, width=1.0):
        self.canvas.setLineWidth(width)
        self.canvas.setFillColor(HexColor(fill or "#FFFFFF"))
        self.canvas.setStrokeColor(HexColor(stroke or fill or "#FFFFFF"))
        self.canvas.rect(x, self._y(y + h), w, h, fill=1 if fill else 0, stroke=1 if stroke else 0)

    def circle(self, x, y, r, fill=COLORS["ink"], stroke=None, width=1.0):
        self.canvas.setLineWidth(width)
        self.canvas.setFillColor(HexColor(fill))
        self.canvas.setStrokeColor(HexColor(stroke or fill))
        self.canvas.circle(x, self._y(y), r, fill=1, stroke=1 if stroke else 0)

    def text(self, x, y, text, size=9, color=COLORS["ink"], bold=False, align="left"):
        font = "Helvetica-Bold" if bold else "Helvetica"
        self.canvas.setFont(font, size)
        self.canvas.setFillColor(HexColor(color))
        yy = self._y(y) - size * 0.72
        if align == "center":
            self.canvas.drawCentredString(x, yy, str(text))
        elif align == "right":
            self.canvas.drawRightString(x, yy, str(text))
        else:
            self.canvas.drawString(x, yy, str(text))

    def save(self):
        self.canvas.showPage()
        self.canvas.save()


class SVGRenderer:
    def __init__(self, path: Path, width: int = FIG_W, height: int = FIG_H):
        self.path = path
        self.width = width
        self.height = height
        self.items: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        ]

    def line(self, x1, y1, x2, y2, color=COLORS["ink"], width=1.0):
        self.items.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width:.2f}" stroke-linecap="round"/>'
        )

    def rect(self, x, y, w, h, fill=None, stroke=None, width=1.0):
        fill_attr = fill if fill else "none"
        stroke_attr = stroke if stroke else "none"
        self.items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill_attr}" stroke="{stroke_attr}" stroke-width="{width:.2f}"/>'
        )

    def circle(self, x, y, r, fill=COLORS["ink"], stroke=None, width=1.0):
        stroke_attr = stroke if stroke else fill
        self.items.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" '
            f'stroke="{stroke_attr}" stroke-width="{width:.2f}"/>'
        )

    def text(self, x, y, text, size=9, color=COLORS["ink"], bold=False, align="left"):
        anchor = {"left": "start", "center": "middle", "right": "end"}[align]
        weight = "700" if bold else "400"
        safe = (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        self.items.append(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{safe}</text>'
        )

    def save(self):
        self.items.append("</svg>")
        self.path.write_text("\n".join(self.items), encoding="utf-8")


class PNGRenderer:
    def __init__(self, path: Path, width: int = FIG_W, height: int = FIG_H, scale: int = 4):
        self.path = path
        self.scale = scale
        self.width = width
        self.height = height
        self.img = Image.new("RGB", (width * scale, height * scale), "white")
        self.draw = ImageDraw.Draw(self.img)
        self.font_regular = self._font("arial.ttf")
        self.font_bold = self._font("arialbd.ttf")

    def _font(self, name: str):
        candidates = [
            Path("C:/Windows/Fonts") / name,
            Path("C:/Windows/Fonts") / name.lower(),
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def _xy(self, x, y) -> tuple[int, int]:
        return int(round(x * self.scale)), int(round(y * self.scale))

    def _font_obj(self, size: int, bold: bool = False):
        font_path = self.font_bold if bold else self.font_regular
        if font_path:
            return ImageFont.truetype(font_path, size * self.scale)
        return ImageFont.load_default()

    def line(self, x1, y1, x2, y2, color=COLORS["ink"], width=1.0):
        self.draw.line([self._xy(x1, y1), self._xy(x2, y2)], fill=hex_to_rgb(color), width=max(1, int(width * self.scale)))

    def rect(self, x, y, w, h, fill=None, stroke=None, width=1.0):
        box = [*self._xy(x, y), *self._xy(x + w, y + h)]
        self.draw.rectangle(
            box,
            fill=hex_to_rgb(fill) if fill else None,
            outline=hex_to_rgb(stroke) if stroke else None,
            width=max(1, int(width * self.scale)),
        )

    def circle(self, x, y, r, fill=COLORS["ink"], stroke=None, width=1.0):
        box = [*self._xy(x - r, y - r), *self._xy(x + r, y + r)]
        self.draw.ellipse(
            box,
            fill=hex_to_rgb(fill),
            outline=hex_to_rgb(stroke) if stroke else None,
            width=max(1, int(width * self.scale)),
        )

    def text(self, x, y, text, size=9, color=COLORS["ink"], bold=False, align="left"):
        font = self._font_obj(size, bold)
        text = str(text)
        sx, sy = self._xy(x, y)
        bbox = self.draw.textbbox((sx, sy), text, font=font)
        tw = bbox[2] - bbox[0]
        if align == "center":
            sx -= tw // 2
        elif align == "right":
            sx -= tw
        self.draw.text((sx, sy), text, fill=hex_to_rgb(color), font=font)

    def save(self):
        self.img.save(self.path, dpi=(600, 600))


def draw_axes(renderer, x, y, w, h, ymin, ymax, ylabel, yfmt="{:.1f}", ticks: Sequence[float] | None = None):
    renderer.line(x, y + h, x + w, y + h, COLORS["ink"], 0.8)
    renderer.line(x, y, x, y + h, COLORS["ink"], 0.8)
    ticks = list(ticks) if ticks is not None else nice_ticks(ymin, ymax, 5)
    for tick in ticks:
        if tick < ymin - 1e-9 or tick > ymax + 1e-9:
            continue
        ty = y + h - (tick - ymin) / (ymax - ymin) * h
        renderer.line(x - 4, ty, x, ty, COLORS["ink"], 0.8)
        renderer.line(x, ty, x + w, ty, COLORS["grid"], 0.5)
        renderer.text(x - 7, ty - 5, yfmt.format(tick), 7.2, COLORS["muted"], align="right")
    renderer.text(x, y - 13, ylabel, 8, COLORS["ink"], bold=True)


def ymap(v: float, y: float, h: float, ymin: float, ymax: float) -> float:
    return y + h - (v - ymin) / (ymax - ymin) * h


def draw_errorbar(renderer, x, y_mean, y_ci_low, y_ci_high, color):
    renderer.line(x, y_ci_low, x, y_ci_high, color, 1.5)
    renderer.line(x - 5, y_ci_low, x + 5, y_ci_low, color, 1.5)
    renderer.line(x - 5, y_ci_high, x + 5, y_ci_high, color, 1.5)
    renderer.circle(x, y_mean, 4.2, fill=color, stroke="#FFFFFF", width=0.8)


def draw_paired_panel(
    renderer,
    panel,
    pairs: pd.DataFrame,
    base_col: str,
    method_col: str,
    improvement_col: str,
    ylabel: str,
    y_max: float,
    letter: str,
    title: str,
    stat: dict[str, float],
):
    x, y, w, h = panel
    renderer.text(x, y - 21, letter, 13, COLORS["ink"], bold=True)
    renderer.text(x + 20, y - 19, title, 10, COLORS["ink"], bold=True)
    plot_y = y + 13
    plot_h = h - 55
    draw_axes(renderer, x + 48, plot_y, w - 62, plot_h, 0, y_max, ylabel, yfmt="{:.1f}")
    ax_x = x + 48
    ax_w = w - 62
    x0 = ax_x + ax_w * 0.31
    x1 = ax_x + ax_w * 0.73
    valid = pairs[[base_col, method_col, improvement_col]].dropna()
    for _, row in valid.iterrows():
        y0 = ymap(float(row[base_col]), plot_y, plot_h, 0, y_max)
        y1 = ymap(float(row[method_col]), plot_y, plot_h, 0, y_max)
        renderer.line(x0, y0, x1, y1, COLORS["pair"], 0.45)

    base_mean, base_ci = mean_ci(valid[base_col])
    meth_mean, meth_ci = mean_ci(valid[method_col])
    draw_errorbar(
        renderer,
        x0,
        ymap(base_mean, plot_y, plot_h, 0, y_max),
        ymap(base_mean - base_ci, plot_y, plot_h, 0, y_max),
        ymap(base_mean + base_ci, plot_y, plot_h, 0, y_max),
        COLORS["baseline"],
    )
    draw_errorbar(
        renderer,
        x1,
        ymap(meth_mean, plot_y, plot_h, 0, y_max),
        ymap(meth_mean - meth_ci, plot_y, plot_h, 0, y_max),
        ymap(meth_mean + meth_ci, plot_y, plot_h, 0, y_max),
        COLORS["pca"],
    )
    renderer.text(x0, plot_y + plot_h + 11, "No imp.", 7.2, COLORS["baseline"], align="center")
    renderer.text(x1, plot_y + plot_h + 11, "2D2-PCA", 7.2, COLORS["pca"], align="center")
    text = (
        f"Mean improvement {stat['mean_improvement_pct']:.1f}% "
        f"({stat['wins']}/{stat['n']} pairs); {fmt_p(stat['sign_test_p_one_sided'])}"
    )
    renderer.text(x + 55, y + h - 18, text, 7.4, COLORS["ink"])
    renderer.text(x + 55, y + h - 6, "Lines are paired runs; dots show mean +/- 95% CI.", 7.0, COLORS["muted"])


def draw_fraction_panel(renderer, panel, pairs: pd.DataFrame, letter: str):
    x, y, w, h = panel
    renderer.text(x, y - 21, letter, 13, COLORS["ink"], bold=True)
    renderer.text(x + 20, y - 19, "Trajectory recovery across masking fractions", 10, COLORS["ink"], bold=True)
    plot_y = y + 13
    plot_h = h - 55
    ymin, ymax = 0, 1.35
    ax_x = x + 48
    ax_w = w - 62
    draw_axes(renderer, ax_x, plot_y, ax_w, plot_h, ymin, ymax, "Trajectory RMSE", yfmt="{:.1f}")
    fracs = sorted(pairs["frac"].dropna().unique())
    xmap = {frac: ax_x + (frac - min(fracs)) / (max(fracs) - min(fracs)) * ax_w for frac in fracs}
    renderer.line(ax_x, plot_y + plot_h + 5, ax_x + ax_w, plot_y + plot_h + 5, COLORS["ink"], 0.8)
    for frac in fracs:
        px = xmap[frac]
        renderer.line(px, plot_y + plot_h, px, plot_y + plot_h + 4, COLORS["ink"], 0.8)
        renderer.text(px, plot_y + plot_h + 11, f"{frac:.2f}", 7.2, COLORS["muted"], align="center")
    renderer.text(ax_x + ax_w / 2, y + h - 5, "Artificial zero fraction", 8, COLORS["ink"], bold=True, align="center")

    series = [("base_traj", COLORS["baseline"], "No imputation"), ("pca_traj", COLORS["pca"], "Penalized 2D2-PCA")]
    for col, color, _label in series:
        pts = []
        for frac in fracs:
            sub = pairs[pairs["frac"].eq(frac)][col].dropna()
            mean, ci = mean_ci(sub)
            px = xmap[frac]
            py = ymap(mean, plot_y, plot_h, ymin, ymax)
            lo = ymap(mean - ci, plot_y, plot_h, ymin, ymax)
            hi = ymap(mean + ci, plot_y, plot_h, ymin, ymax)
            pts.append((px, py))
            renderer.line(px, lo, px, hi, color, 1.2)
            renderer.circle(px, py, 3.5, fill=color, stroke="#FFFFFF", width=0.8)
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            renderer.line(x1, y1, x2, y2, color, 1.8)

    leg_x = x + w - 138
    leg_y = y + 20
    renderer.circle(leg_x, leg_y, 3.3, fill=COLORS["baseline"])
    renderer.text(leg_x + 9, leg_y - 5, "No imputation", 7.4, COLORS["ink"])
    renderer.circle(leg_x, leg_y + 14, 3.3, fill=COLORS["pca"])
    renderer.text(leg_x + 9, leg_y + 9, "Penalized 2D2-PCA", 7.4, COLORS["ink"])
    renderer.text(x + 55, y + h - 18, "Lower values indicate better recovery.", 7.2, COLORS["muted"])


def draw_mechanism_panel(renderer, panel, pairs: pd.DataFrame, letter: str):
    x, y, w, h = panel
    renderer.text(x, y - 21, letter, 13, COLORS["ink"], bold=True)
    renderer.text(x + 20, y - 19, "Mechanism-specific trajectory improvement", 10, COLORS["ink"], bold=True)
    plot_y = y + 13
    plot_h = h - 68
    ax_x = x + 48
    ax_w = w - 62
    ymin, ymax = -70, 90
    draw_axes(renderer, ax_x, plot_y, ax_w, plot_h, ymin, ymax, "Improvement (%)", yfmt="{:.0f}", ticks=[-60, -30, 0, 30, 60, 90])
    zero_y = ymap(0, plot_y, plot_h, ymin, ymax)
    renderer.line(ax_x, zero_y, ax_x + ax_w, zero_y, COLORS["ink"], 1.0)
    bar_gap = 7
    n = len(MECH_ORDER)
    bar_w = (ax_w - bar_gap * (n + 1)) / n
    for idx, mech in enumerate(MECH_ORDER):
        sub = pairs[pairs["mechanism"].eq(mech)]
        mean, ci = mean_ci(sub["traj_improvement_pct"])
        cx = ax_x + bar_gap + idx * (bar_w + bar_gap) + bar_w / 2
        top = ymap(max(mean, 0), plot_y, plot_h, ymin, ymax)
        bottom = ymap(min(mean, 0), plot_y, plot_h, ymin, ymax)
        color = COLORS["positive"] if mean >= 0 else COLORS["negative"]
        renderer.rect(cx - bar_w / 2, top, bar_w, bottom - top, fill=color, stroke=color, width=0.5)
        ci_lo = ymap(mean - ci, plot_y, plot_h, ymin, ymax)
        ci_hi = ymap(mean + ci, plot_y, plot_h, ymin, ymax)
        renderer.line(cx, ci_lo, cx, ci_hi, COLORS["ink"], 0.9)
        renderer.line(cx - 4, ci_lo, cx + 4, ci_lo, COLORS["ink"], 0.9)
        renderer.line(cx - 4, ci_hi, cx + 4, ci_hi, COLORS["ink"], 0.9)
        renderer.text(cx, plot_y + plot_h + 12, MECH_LABELS.get(mech, mech), 6.6, COLORS["muted"], align="center")
    renderer.text(x + 55, y + h - 18, "Positive values mean lower trajectory RMSE than no imputation.", 7.2, COLORS["muted"])


def draw_figure(renderer, pairs: pd.DataFrame, overall: dict[str, dict[str, float]]):
    renderer.text(36, 18, "Oracle-mask imputation restores longitudinal trajectories", 15, COLORS["ink"], bold=True)
    renderer.text(
        36,
        38,
        "Paired semi-synthetic zero-recovery runs; comparison is Penalized 2D2-PCA versus no imputation.",
        8.5,
        COLORS["muted"],
    )
    panels = {
        "A": (38, 75, 305, 235),
        "B": (382, 75, 305, 235),
        "C": (38, 365, 305, 220),
        "D": (382, 365, 305, 220),
    }
    draw_paired_panel(
        renderer,
        panels["A"],
        pairs,
        "base_traj",
        "pca_traj",
        "traj_improvement_pct",
        "Trajectory RMSE",
        1.4,
        "A",
        "Overall longitudinal trajectory error",
        overall["trajectory_rmse_affected"],
    )
    draw_fraction_panel(renderer, panels["B"], pairs, "B")
    draw_mechanism_panel(renderer, panels["C"], pairs, "C")
    draw_paired_panel(
        renderer,
        panels["D"],
        pairs,
        "base_lag",
        "pca_lag",
        "lag_improvement_pct",
        "Lag-1 autocorr. error",
        0.42,
        "D",
        "Lag-1 autocorrelation recovery",
        overall["lag1_autocorr_abs_error"],
    )
    renderer.text(
        36,
        FIG_H - 18,
        "Statistics: one-sided paired sign test; error bars are 95% CI. Data source: outputs/experiment1_zero_recovery/metrics_long.csv.",
        7.2,
        COLORS["muted"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publication-style oracle-mask trajectory recovery figure.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/experiment1_zero_recovery/metrics_long.csv"),
        help="Path to experiment 1 long metrics CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/experiment1_zero_recovery/figures"),
        help="Figure output directory.",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    pairs, _oracle_method = load_pairs(args.metrics)
    overall = save_stats(pairs, args.outdir)

    base = args.outdir / "oracle_mask_trajectory_recovery_publication"
    renderers = [
        PDFRenderer(base.with_suffix(".pdf")),
        SVGRenderer(base.with_suffix(".svg")),
        PNGRenderer(base.with_suffix(".png")),
    ]
    for renderer in renderers:
        draw_figure(renderer, pairs, overall)
        renderer.save()

    print(f"wrote {base.with_suffix('.pdf')}")
    print(f"wrote {base.with_suffix('.svg')}")
    print(f"wrote {base.with_suffix('.png')}")
    print(f"wrote {args.outdir / 'oracle_mask_trajectory_recovery_stats.csv'}")
    print(f"wrote {args.outdir / 'oracle_mask_trajectory_recovery_by_mechanism.csv'}")


if __name__ == "__main__":
    main()
