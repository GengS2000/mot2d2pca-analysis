from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = ROOT / "outputs" / "experiment2_distribution"

METHOD_NO = "no_imputation"
METHOD_PCA = "penalized_2d2_pca"
MODE_ORDER = ["oracle_mask", "detector_longitudinal_neighbor"]
MODE_LABELS = {
    "oracle_mask": "Oracle mask",
    "detector_longitudinal_neighbor": "Detected zeros",
}
MODE_SHORT = {
    "oracle_mask": "Oracle",
    "detector_longitudinal_neighbor": "Detected",
}
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

COLORS = {
    "text": "#111827",
    "muted": "#64748B",
    "axis": "#334155",
    "grid": "#E2E8F0",
    "line": "#CBD5E1",
    "no": "#6B7280",
    "pca": "#0072B2",
    "pca_light": "#56B4E9",
    "raw": "#111827",
    "bg": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SCI-style publication figures for Experiment 2, focused on penalized_2d2_pca."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Directory containing experiment2_distribution CSV outputs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "figures",
        help="Directory for exported SVG/PNG/TIFF figures.",
    )
    parser.add_argument(
        "--basename",
        default="experiment2_penalized_2d2_pca_distribution_sci",
        help="Base filename for exported figure files.",
    )
    return parser.parse_args()


def as_float(value: str) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def mean(values) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else math.nan


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one_sided_sign_p_value(successes: int, n: int) -> float:
    if n <= 0 or successes < 0:
        return math.nan
    tail = 0.0
    for k in range(successes, n + 1):
        tail += math.comb(n, k)
    return tail / (2**n)


def hex_to_rgba(color: str, opacity: float = 1.0) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return (
        int(color[0:2], 16),
        int(color[2:4], 16),
        int(color[4:6], 16),
        max(0, min(255, round(255 * opacity))),
    )


def font_path(bold: bool = False) -> str | None:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


class DualCanvas:
    def __init__(self, width: int, height: int, scale: int = 3):
        self.width = width
        self.height = height
        self.scale = scale
        self.svg: list[str] = []
        self._font_cache = {}
        try:
            from PIL import Image, ImageDraw

            self.Image = Image
            self.ImageDraw = ImageDraw
            self.image = Image.new("RGBA", (width * scale, height * scale), hex_to_rgba(COLORS["bg"]))
            self.draw = ImageDraw.Draw(self.image, "RGBA")
        except Exception:
            self.Image = None
            self.ImageDraw = None
            self.image = None
            self.draw = None

    def _font(self, size: float, bold: bool = False):
        if self.Image is None:
            return None
        from PIL import ImageFont

        key = (round(size * self.scale), bold)
        if key in self._font_cache:
            return self._font_cache[key]
        path = font_path(bold)
        if path:
            font = ImageFont.truetype(path, key[0])
        else:
            font = ImageFont.load_default()
        self._font_cache[key] = font
        return font

    def line(self, x1, y1, x2, y2, color, width=1.0, opacity=1.0, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width:.2f}" stroke-opacity="{opacity:.3f}" '
            f'stroke-linecap="round"{dash_attr}/>'
        )
        if self.draw is not None:
            xy = [
                (round(x1 * self.scale), round(y1 * self.scale)),
                (round(x2 * self.scale), round(y2 * self.scale)),
            ]
            fill = hex_to_rgba(color, opacity)
            if dash:
                self._dashed_line(xy[0], xy[1], fill, max(1, round(width * self.scale)), dash)
            else:
                self.draw.line(xy, fill=fill, width=max(1, round(width * self.scale)))

    def _dashed_line(self, p1, p2, fill, width, dash):
        parts = [float(v) for v in str(dash).replace(",", " ").split() if v]
        if len(parts) < 2:
            parts = [8.0, 6.0]
        dash_len, gap_len = [v * self.scale for v in parts[:2]]
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist == 0:
            return
        ux, uy = (x2 - x1) / dist, (y2 - y1) / dist
        pos = 0.0
        while pos < dist:
            end = min(pos + dash_len, dist)
            self.draw.line(
                [(x1 + ux * pos, y1 + uy * pos), (x1 + ux * end, y1 + uy * end)],
                fill=fill,
                width=width,
            )
            pos = end + gap_len

    def rect(self, x, y, w, h, fill="none", stroke=None, stroke_width=1.0, opacity=1.0, radius=0):
        stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width:.2f}"' if stroke else ' stroke="none"'
        self.svg.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="{radius:.2f}" fill="{fill}" fill-opacity="{opacity:.3f}"{stroke_attr}/>'
        )
        if self.draw is not None:
            box = [x * self.scale, y * self.scale, (x + w) * self.scale, (y + h) * self.scale]
            fill_rgba = None if fill == "none" else hex_to_rgba(fill, opacity)
            outline = None if stroke is None else hex_to_rgba(stroke, 1.0)
            self.draw.rectangle(box, fill=fill_rgba, outline=outline, width=max(1, round(stroke_width * self.scale)))

    def circle(self, x, y, r, fill, stroke=None, stroke_width=1.0, opacity=1.0):
        stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width:.2f}"' if stroke else ' stroke="none"'
        self.svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" '
            f'fill-opacity="{opacity:.3f}"{stroke_attr}/>'
        )
        if self.draw is not None:
            box = [(x - r) * self.scale, (y - r) * self.scale, (x + r) * self.scale, (y + r) * self.scale]
            self.draw.ellipse(
                box,
                fill=hex_to_rgba(fill, opacity),
                outline=None if stroke is None else hex_to_rgba(stroke),
                width=max(1, round(stroke_width * self.scale)),
            )

    def polyline(self, points, color, width=2.0, opacity=1.0, dash=None):
        if not points:
            return
        point_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(
            f'<polyline points="{point_str}" fill="none" stroke="{color}" '
            f'stroke-width="{width:.2f}" stroke-opacity="{opacity:.3f}" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
        )
        if self.draw is not None and len(points) > 1:
            scaled = [(x * self.scale, y * self.scale) for x, y in points]
            fill = hex_to_rgba(color, opacity)
            if dash:
                for a, b in zip(scaled[:-1], scaled[1:]):
                    self._dashed_line(a, b, fill, max(1, round(width * self.scale)), dash)
            else:
                self.draw.line(scaled, fill=fill, width=max(1, round(width * self.scale)), joint="curve")

    def text(
        self,
        x,
        y,
        value,
        size=22,
        color=None,
        bold=False,
        anchor="start",
        opacity=1.0,
        rotate=0,
    ):
        color = color or COLORS["text"]
        value = str(value)
        weight = "700" if bold else "400"
        transform = f' transform="rotate({rotate:.2f} {x:.2f} {y:.2f})"' if rotate else ""
        escaped = html.escape(value)
        self.svg.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size:.2f}" font-weight="{weight}" fill="{color}" fill-opacity="{opacity:.3f}" '
            f'text-anchor="{anchor}" dominant-baseline="middle"{transform}>{escaped}</text>'
        )
        if self.draw is not None:
            font = self._font(size, bold)
            fill = hex_to_rgba(color, opacity)
            scaled_x, scaled_y = x * self.scale, y * self.scale
            pil_anchor = {"start": "lm", "middle": "mm", "end": "rm"}.get(anchor, "lm")
            if rotate:
                self._rotated_text(scaled_x, scaled_y, value, font, fill, pil_anchor, rotate)
            else:
                self.draw.text((scaled_x, scaled_y), value, font=font, fill=fill, anchor=pil_anchor)

    def _rotated_text(self, x, y, value, font, fill, anchor, rotate):
        from PIL import Image, ImageDraw

        bbox = self.draw.textbbox((0, 0), value, font=font, anchor="lt")
        tw, th = bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 20
        layer = Image.new("RGBA", (tw, th), (255, 255, 255, 0))
        d = ImageDraw.Draw(layer)
        d.text((10, 10), value, font=font, fill=fill, anchor="lt")
        rotated = layer.rotate(-rotate, expand=True, resample=Image.Resampling.BICUBIC)
        if anchor == "mm":
            px, py = x - rotated.width / 2, y - rotated.height / 2
        elif anchor == "rm":
            px, py = x - rotated.width, y - rotated.height / 2
        else:
            px, py = x, y - rotated.height / 2
        self.image.alpha_composite(rotated, (round(px), round(py)))

    def save(self, svg_path: Path, png_path: Path | None = None, tiff_path: Path | None = None):
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        document = "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                (
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="152mm" '
                    f'viewBox="0 0 {self.width} {self.height}">'
                ),
                f'<rect width="{self.width}" height="{self.height}" fill="{COLORS["bg"]}"/>',
                *self.svg,
                "</svg>",
                "",
            ]
        )
        svg_path.write_text(document, encoding="utf-8")
        if self.image is not None:
            rgb = self.image.convert("RGB")
            if png_path is not None:
                rgb.save(png_path, dpi=(600, 600))
            if tiff_path is not None:
                rgb.save(tiff_path, dpi=(600, 600), compression="tiff_lzw")


def load_data(experiment_dir: Path):
    metrics_path = experiment_dir / "raw_nonzero_vs_imputed_at_masked_distribution_metrics.csv"
    quantiles_path = experiment_dir / "raw_nonzero_vs_imputed_at_masked_distribution_quantiles.csv"
    raw_quantiles_path = experiment_dir / "raw_nonzero_da_log_distribution_quantiles.csv"
    raw_summary_path = experiment_dir / "raw_nonzero_da_log_distribution_summary.csv"

    rows = read_csv_rows(metrics_path)
    by_key: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        method = row["method"]
        if method not in {METHOD_NO, METHOD_PCA}:
            continue
        key = (row["mechanism"], row["frac"], row["replacement_mode"], row["repeat"])
        by_key[key][method] = row

    pairs = []
    for (mechanism, frac, mode, repeat), methods in by_key.items():
        if METHOD_NO not in methods or METHOD_PCA not in methods:
            continue
        no = methods[METHOD_NO]
        pca = methods[METHOD_PCA]
        pairs.append(
            {
                "mechanism": mechanism,
                "frac": float(frac),
                "replacement_mode": mode,
                "repeat": int(float(repeat)),
                "w_no": as_float(no["raw_nonzero_vs_imputed_at_masked_wasserstein"]),
                "w_pca": as_float(pca["raw_nonzero_vs_imputed_at_masked_wasserstein"]),
                "js_no": as_float(no["raw_nonzero_vs_imputed_at_masked_js"]),
                "js_pca": as_float(pca["raw_nonzero_vs_imputed_at_masked_js"]),
            }
        )

    raw_quantiles = {}
    for row in read_csv_rows(raw_quantiles_path):
        q = round(as_float(row["quantile"]), 4)
        raw_quantiles[q] = as_float(row["log10_profile_plus_eps"])

    grouped_quantiles: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for row in read_csv_rows(quantiles_path):
        if row["distribution"] != "imputed_at_artificial_mask":
            continue
        method = row["method"]
        if method not in {METHOD_NO, METHOD_PCA}:
            continue
        q = round(as_float(row["quantile"]), 4)
        if q not in {round(x, 4) for x in QUANTILES}:
            continue
        grouped_quantiles[(row["replacement_mode"], method, q)].append(as_float(row["log10_profile_plus_eps"]))

    quantile_means = {
        key: mean(values)
        for key, values in grouped_quantiles.items()
    }

    raw_summary = read_csv_rows(raw_summary_path)[0]
    return pairs, raw_quantiles, quantile_means, raw_summary


def summarize_pairs(pairs):
    summary = {}
    for mode in MODE_ORDER:
        mode_pairs = [p for p in pairs if p["replacement_mode"] == mode]
        n = len(mode_pairs)
        w_no = mean(p["w_no"] for p in mode_pairs)
        w_pca = mean(p["w_pca"] for p in mode_pairs)
        js_no = mean(p["js_no"] for p in mode_pairs)
        js_pca = mean(p["js_pca"] for p in mode_pairs)
        summary[mode] = {
            "n": n,
            "w_no": w_no,
            "w_pca": w_pca,
            "w_reduction_pct": 100.0 * (w_no - w_pca) / w_no,
            "w_improved": sum(1 for p in mode_pairs if p["w_pca"] < p["w_no"]),
            "w_sign_p": one_sided_sign_p_value(sum(1 for p in mode_pairs if p["w_pca"] < p["w_no"]), n),
            "js_no": js_no,
            "js_pca": js_pca,
            "js_reduction_pct": 100.0 * (js_no - js_pca) / js_no,
            "js_improved": sum(1 for p in mode_pairs if p["js_pca"] < p["js_no"]),
            "js_sign_p": one_sided_sign_p_value(sum(1 for p in mode_pairs if p["js_pca"] < p["js_no"]), n),
        }
    return summary


def quantile_error_summary(raw_quantiles, quantile_means):
    out = {}
    for mode in MODE_ORDER:
        for method in [METHOD_NO, METHOD_PCA]:
            diffs = []
            for q in QUANTILES:
                q_key = round(q, 4)
                raw = raw_quantiles[q_key]
                val = quantile_means.get((mode, method, q_key), math.nan)
                if math.isfinite(raw) and math.isfinite(val):
                    diffs.append(abs(val - raw))
            out[(mode, method)] = mean(diffs)
    return out


def scale_linear(value: float, vmin: float, vmax: float, start: float, end: float) -> float:
    if vmax == vmin:
        return (start + end) / 2
    return start + (value - vmin) * (end - start) / (vmax - vmin)


def y_map(value: float, ymin: float, ymax: float, top: float, height: float) -> float:
    return top + height - (value - ymin) * height / (ymax - ymin)


def draw_axes(canvas: DualCanvas, x, y, w, h, ymin, ymax, yticks, xlabel=None, ylabel=None, show_ylabels=True):
    for tick in yticks:
        ty = y_map(tick, ymin, ymax, y, h)
        canvas.line(x, ty, x + w, ty, COLORS["grid"], width=1.0, opacity=1.0)
        canvas.line(x - 5, ty, x, ty, COLORS["axis"], width=1.1)
        if show_ylabels:
            label = f"{tick:g}"
            canvas.text(x - 12, ty, label, size=19, color=COLORS["muted"], anchor="end")
    canvas.line(x, y, x, y + h, COLORS["axis"], width=1.3)
    canvas.line(x, y + h, x + w, y + h, COLORS["axis"], width=1.3)
    if xlabel:
        canvas.text(x + w / 2, y + h + 55, xlabel, size=22, color=COLORS["axis"], anchor="middle")
    if ylabel:
        canvas.text(x - 62, y + h / 2, ylabel, size=22, color=COLORS["axis"], anchor="middle", rotate=-90)


def draw_panel_title(canvas, tag, title, subtitle, x, y):
    canvas.text(x, y, tag, size=28, bold=True, color=COLORS["text"], anchor="start")
    canvas.text(x + 42, y, title, size=27, bold=True, color=COLORS["text"], anchor="start")
    if subtitle:
        canvas.text(x + 42, y + 32, subtitle, size=19, color=COLORS["muted"], anchor="start")


def draw_slope_panel(canvas: DualCanvas, x, y, w, h, pairs, summary, metric: str):
    if metric == "w":
        tag, title = "A", "Distance to raw non-zero distribution"
        subtitle = "Wasserstein distance at artificial mask positions"
        ylabel = "Wasserstein distance"
        y_max = 2.35
        yticks = [0, 0.5, 1.0, 1.5, 2.0]
        no_key, pca_key, red_key, imp_key = "w_no", "w_pca", "w_reduction_pct", "w_improved"
    else:
        tag, title = "B", "Shape divergence to raw non-zero distribution"
        subtitle = "Jensen-Shannon divergence at artificial mask positions"
        ylabel = "JS divergence"
        y_max = 0.76
        yticks = [0, 0.2, 0.4, 0.6]
        no_key, pca_key, red_key, imp_key = "js_no", "js_pca", "js_reduction_pct", "js_improved"

    draw_panel_title(canvas, tag, title, subtitle, x, y)
    plot_top = y + 92
    plot_h = h - 158
    left_margin = 88
    gap = 60
    facet_w = (w - left_margin - 30 - gap) / 2
    for idx, mode in enumerate(MODE_ORDER):
        px = x + left_margin + idx * (facet_w + gap)
        draw_axes(
            canvas,
            px,
            plot_top,
            facet_w,
            plot_h,
            0,
            y_max,
            yticks,
            ylabel=ylabel if idx == 0 else None,
            show_ylabels=idx == 0,
        )
        mode_pairs = [p for p in pairs if p["replacement_mode"] == mode]
        x_no = px + facet_w * 0.25
        x_pca = px + facet_w * 0.75
        for j, pair in enumerate(mode_pairs):
            jitter = ((j % 9) - 4) * 0.6
            y_no = y_map(pair[no_key], 0, y_max, plot_top, plot_h)
            y_pca = y_map(pair[pca_key], 0, y_max, plot_top, plot_h)
            canvas.line(x_no + jitter, y_no, x_pca + jitter, y_pca, COLORS["line"], width=1.1, opacity=0.42)
        s = summary[mode]
        y_no_mean = y_map(s[no_key], 0, y_max, plot_top, plot_h)
        y_pca_mean = y_map(s[pca_key], 0, y_max, plot_top, plot_h)
        canvas.line(x_no, y_no_mean, x_pca, y_pca_mean, COLORS["pca"], width=4.2, opacity=0.95)
        canvas.circle(x_no, y_no_mean, 9, COLORS["no"], stroke="#FFFFFF", stroke_width=2)
        canvas.circle(x_pca, y_pca_mean, 10, COLORS["pca"], stroke="#FFFFFF", stroke_width=2)
        canvas.text(x_no, plot_top + plot_h + 30, "No", size=19, color=COLORS["axis"], anchor="middle")
        canvas.text(x_no, plot_top + plot_h + 54, "imputation", size=19, color=COLORS["axis"], anchor="middle")
        canvas.text(x_pca, plot_top + plot_h + 42, "2D2-PCA", size=20, bold=True, color=COLORS["pca"], anchor="middle")
        canvas.text(px + facet_w / 2, plot_top - 28, MODE_LABELS[mode], size=22, bold=True, anchor="middle")
        canvas.text(
            px + facet_w - 8,
            plot_top + 24,
            f"-{s[red_key]:.1f}%",
            size=24,
            bold=True,
            color=COLORS["pca"],
            anchor="end",
        )
        canvas.text(
            px + facet_w - 8,
            plot_top + 52,
            f"{s[imp_key]}/{s['n']} improved",
            size=17,
            color=COLORS["muted"],
            anchor="end",
        )


def draw_quantile_panel(canvas: DualCanvas, x, y, w, h, raw_quantiles, quantile_means):
    draw_panel_title(
        canvas,
        "C",
        "Recovered abundance quantile profile",
        "Mean profile across mechanisms, fractions and repeats",
        x,
        y,
    )
    px = x + 90
    py = y + 95
    plot_w = w - 140
    plot_h = h - 170
    ymin, ymax = -3.1, 1.1
    yticks = [-3, -2, -1, 0, 1]
    draw_axes(
        canvas,
        px,
        py,
        plot_w,
        plot_h,
        ymin,
        ymax,
        yticks,
        xlabel="Distribution quantile",
        ylabel="log10(abundance + 0.001)",
    )
    for q in QUANTILES:
        tx = scale_linear(q, QUANTILES[0], QUANTILES[-1], px, px + plot_w)
        canvas.line(tx, py + plot_h, tx, py + plot_h + 5, COLORS["axis"], width=1.0)
        label = "0.5" if abs(q - 0.5) < 1e-9 else f"{q:.2g}"
        canvas.text(tx, py + plot_h + 25, label, size=18, color=COLORS["muted"], anchor="middle")

    def line_points(values):
        return [
            (
                scale_linear(q, QUANTILES[0], QUANTILES[-1], px, px + plot_w),
                y_map(values[round(q, 4)], ymin, ymax, py, plot_h),
            )
            for q in QUANTILES
            if math.isfinite(values.get(round(q, 4), math.nan))
        ]

    raw_values = {round(q, 4): raw_quantiles[round(q, 4)] for q in QUANTILES}
    no_values = {
        round(q, 4): mean(
            [
                quantile_means.get((mode, METHOD_NO, round(q, 4)), math.nan)
                for mode in MODE_ORDER
            ]
        )
        for q in QUANTILES
    }
    oracle_values = {
        round(q, 4): quantile_means.get(("oracle_mask", METHOD_PCA, round(q, 4)), math.nan)
        for q in QUANTILES
    }
    detected_values = {
        round(q, 4): quantile_means.get(("detector_longitudinal_neighbor", METHOD_PCA, round(q, 4)), math.nan)
        for q in QUANTILES
    }

    series = [
        ("Original non-zero da_log", raw_values, COLORS["raw"], 4.2, None),
        ("2D2-PCA, oracle mask", oracle_values, COLORS["pca"], 4.0, None),
        ("2D2-PCA, detected zeros", detected_values, COLORS["pca_light"], 4.0, None),
        ("No imputation", no_values, COLORS["no"], 3.2, "14 10"),
    ]
    for _label, values, color, width, dash in series:
        points = line_points(values)
        canvas.polyline(points, color, width=width, dash=dash)
        for sx, sy in points:
            canvas.circle(sx, sy, 4.6, color, stroke="#FFFFFF", stroke_width=1.3)

    legend_x = px + 18
    legend_y = py + 26
    for i, (label, _values, color, width, dash) in enumerate(series):
        ly = legend_y + i * 30
        canvas.line(legend_x, ly, legend_x + 44, ly, color, width=width, dash=dash)
        canvas.circle(legend_x + 22, ly, 4.6, color, stroke="#FFFFFF", stroke_width=1.0)
        canvas.text(legend_x + 58, ly, label, size=18, color=COLORS["text"], anchor="start")


def draw_quantile_error_panel(canvas: DualCanvas, x, y, w, h, qerr):
    draw_panel_title(
        canvas,
        "D",
        "Quantile error reduction",
        "Mean absolute error from the raw non-zero quantiles",
        x,
        y,
    )
    px = x + 92
    py = y + 95
    plot_w = w - 135
    plot_h = h - 170
    ymax = 2.4
    draw_axes(
        canvas,
        px,
        py,
        plot_w,
        plot_h,
        0,
        ymax,
        [0, 0.5, 1.0, 1.5, 2.0],
        xlabel=None,
        ylabel="Mean abs. quantile error",
    )
    group_centers = [
        px + plot_w * 0.30,
        px + plot_w * 0.72,
    ]
    bar_w = 44
    for center, mode in zip(group_centers, MODE_ORDER):
        no_val = qerr[(mode, METHOD_NO)]
        pca_val = qerr[(mode, METHOD_PCA)]
        values = [(METHOD_NO, no_val, COLORS["no"], -bar_w * 0.65), (METHOD_PCA, pca_val, COLORS["pca"], bar_w * 0.65)]
        for method, val, color, offset in values:
            bx = center + offset - bar_w / 2
            by = y_map(val, 0, ymax, py, plot_h)
            bh = py + plot_h - by
            canvas.rect(bx, by, bar_w, bh, fill=color, opacity=0.9)
            if method == METHOD_NO:
                canvas.text(center + offset, by + 24, f"{val:.2f}", size=17, color="#FFFFFF", bold=True, anchor="middle")
            else:
                canvas.text(center + offset, by - 16, f"{val:.2f}", size=18, color=color, bold=True, anchor="middle")
        reduction = 100.0 * (no_val - pca_val) / no_val
        top_y = min(y_map(no_val, 0, ymax, py, plot_h), y_map(pca_val, 0, ymax, py, plot_h))
        canvas.text(center, max(py + 22, top_y - 36), f"-{reduction:.1f}%", size=24, bold=True, color=COLORS["pca"], anchor="middle")
        canvas.text(center, py + plot_h + 34, MODE_SHORT[mode], size=20, color=COLORS["axis"], anchor="middle")

    legend_y = py + plot_h + 82
    legend_x = px + 20
    canvas.rect(legend_x, legend_y - 11, 22, 22, fill=COLORS["no"], opacity=0.9)
    canvas.text(legend_x + 32, legend_y, "No imputation", size=18, color=COLORS["axis"], anchor="start")
    canvas.rect(legend_x + 185, legend_y - 11, 22, 22, fill=COLORS["pca"], opacity=0.9)
    canvas.text(legend_x + 217, legend_y, "2D2-PCA", size=18, color=COLORS["axis"], anchor="start")


def write_summary_csv(path: Path, pair_summary, qerr, raw_summary):
    rows = []
    for mode in MODE_ORDER:
        s = pair_summary[mode]
        rows.extend(
            [
                {
                    "replacement_mode": mode,
                    "metric": "raw_nonzero_vs_imputed_at_masked_wasserstein",
                    "no_imputation_mean": s["w_no"],
                    "penalized_2d2_pca_mean": s["w_pca"],
                    "relative_reduction_pct": s["w_reduction_pct"],
                    "n_pairs": s["n"],
                    "n_pairs_improved": s["w_improved"],
                    "one_sided_sign_test_p": s["w_sign_p"],
                },
                {
                    "replacement_mode": mode,
                    "metric": "raw_nonzero_vs_imputed_at_masked_js",
                    "no_imputation_mean": s["js_no"],
                    "penalized_2d2_pca_mean": s["js_pca"],
                    "relative_reduction_pct": s["js_reduction_pct"],
                    "n_pairs": s["n"],
                    "n_pairs_improved": s["js_improved"],
                    "one_sided_sign_test_p": s["js_sign_p"],
                },
                {
                    "replacement_mode": mode,
                    "metric": "mean_absolute_quantile_error",
                    "no_imputation_mean": qerr[(mode, METHOD_NO)],
                    "penalized_2d2_pca_mean": qerr[(mode, METHOD_PCA)],
                    "relative_reduction_pct": 100.0
                    * (qerr[(mode, METHOD_NO)] - qerr[(mode, METHOD_PCA)])
                    / qerr[(mode, METHOD_NO)],
                    "n_pairs": "",
                    "n_pairs_improved": "",
                    "one_sided_sign_test_p": "",
                },
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "replacement_mode",
            "metric",
            "no_imputation_mean",
            "penalized_2d2_pca_mean",
            "relative_reduction_pct",
            "n_pairs",
            "n_pairs_improved",
            "one_sided_sign_test_p",
            "raw_nonzero_n",
            "raw_nonzero_mean",
            "raw_nonzero_sd",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["raw_nonzero_n"] = raw_summary["raw_nonzero_n"]
            row["raw_nonzero_mean"] = raw_summary["raw_nonzero_mean"]
            row["raw_nonzero_sd"] = raw_summary["raw_nonzero_sd"]
            writer.writerow(row)


def draw_figure(pairs, raw_quantiles, quantile_means, pair_summary, qerr):
    canvas = DualCanvas(1800, 1520, scale=3)
    canvas.text(72, 50, "Penalized 2D2-PCA restores non-zero abundance distributions", size=36, bold=True)
    canvas.text(
        72,
        88,
        "Distances compare imputed values at artificial mask positions with the empirical non-zero da_log distribution; lower is closer.",
        size=20,
        color=COLORS["muted"],
    )
    canvas.line(72, 120, 1728, 120, COLORS["grid"], width=1.2)

    draw_slope_panel(canvas, 80, 154, 810, 510, pairs, pair_summary, "w")
    draw_slope_panel(canvas, 965, 154, 760, 510, pairs, pair_summary, "js")
    draw_quantile_panel(canvas, 80, 730, 1010, 650, raw_quantiles, quantile_means)
    draw_quantile_error_panel(canvas, 1180, 730, 545, 650, qerr)

    canvas.text(
        72,
        1484,
        "Artificial mask experiment: 3 missingness mechanisms x 2 fractions x 10 repeats per replacement mode. W/JS panels show paired means for 2D2-PCA vs no imputation.",
        size=17,
        color=COLORS["muted"],
    )
    return canvas


def main() -> None:
    args = parse_args()
    pairs, raw_quantiles, quantile_means, raw_summary = load_data(args.experiment_dir)
    pair_summary = summarize_pairs(pairs)
    qerr = quantile_error_summary(raw_quantiles, quantile_means)
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary_path = args.outdir / f"{args.basename}_summary_values.csv"
    write_summary_csv(summary_path, pair_summary, qerr, raw_summary)

    canvas = draw_figure(pairs, raw_quantiles, quantile_means, pair_summary, qerr)
    svg_path = args.outdir / f"{args.basename}.svg"
    png_path = args.outdir / f"{args.basename}.png"
    tiff_path = args.outdir / f"{args.basename}.tiff"
    canvas.save(svg_path, png_path, tiff_path)

    print(f"Wrote {svg_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {tiff_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
