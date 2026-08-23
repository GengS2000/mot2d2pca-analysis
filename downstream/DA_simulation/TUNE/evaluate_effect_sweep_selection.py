import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FRACS = [round(x / 10.0, 1) for x in range(1, 10)]
MAIN_METRICS = ["TP", "recall", "F1", "FDR"]


def parse_level(value):
    text = str(value)
    text = Path(text).name
    text = text.replace("group_effect_", "").replace("frac_", "")
    text = text.replace("p", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def canonical_method(value):
    text = str(value).lower()
    if "zero" in text:
        return "zeroed"
    if "imput" in text:
        return "imputed"
    return text


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_metrics(df, engine):
    out = df.copy()

    if "effect_level" in out.columns:
        out["effect_value"] = pd.to_numeric(out["effect_level"], errors="coerce")
    elif "effect_value" in out.columns:
        out["effect_value"] = pd.to_numeric(out["effect_value"], errors="coerce")
    elif "effect" in out.columns:
        out["effect_value"] = out["effect"].map(parse_level)
    elif "effect_name" in out.columns:
        out["effect_value"] = out["effect_name"].map(parse_level)
    elif "effect_dir" in out.columns:
        out["effect_value"] = out["effect_dir"].map(parse_level)
    else:
        raise ValueError(f"{engine}: missing effect column")

    if "frac" not in out.columns:
        raise ValueError(f"{engine}: missing frac column")

    out["frac_value"] = out["frac"].map(parse_level)
    out["method"] = out["method"].map(canonical_method)
    out["engine"] = engine

    for metric in ["TP", "FP", "FN", "TN", "precision", "recall", "F1", "FDR"]:
        if metric in out.columns:
            out[metric] = pd.to_numeric(out[metric], errors="coerce")
        else:
            out[metric] = np.nan

    keep = [
        "engine",
        "effect_value",
        "frac_value",
        "method",
        "TP",
        "FP",
        "FN",
        "TN",
        "precision",
        "recall",
        "F1",
        "FDR",
    ]
    return out.loc[out["method"].isin(["zeroed", "imputed"]), keep]


def load_all_metrics(root):
    sources = {
        "LinDA": [
            root / "results_linda_compare_all_effects" / "linda_metrics_summary_all_effects.csv",
        ],
        "ANCOMBC2": [
            root / "results" / "ancombc2" / "metrics_summary_ancombc2.csv",
        ],
        "LDM": [
            root / "effect_sweep_ldm_metrics_summary_all.csv",
        ],
        "MaAsLin2": [
            root / "results_maaslin2_compare_all_effects" / "maaslin2_metrics_summary_all_effects.csv",
            root / "results_maaslin2_compare_all_effects" / "maaslin2_plot_metrics_summary_all_effects.csv",
        ],
        "GLMM-PQL": [
            root / "results_glmmPQL_compare_all_effects" / "glmmPQL_metrics_summary_all_effects.csv",
        ],
        "ZIGMM": sorted(root.glob("group_effect_*/results_ZIGMM_compare/nbzimm_zinb_metrics_summary.csv")),
    }

    frames = []
    missing = []
    used = []
    for engine, candidates in sources.items():
        if engine == "ZIGMM":
            paths = [p for p in candidates if p.exists()]
            if not paths:
                missing.append(engine)
                continue
            df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
            frames.append(normalize_metrics(df, engine))
            used.extend({"engine": engine, "path": str(p)} for p in paths)
            continue

        path = first_existing(candidates)
        if path is None:
            missing.append(engine)
            continue
        df = pd.read_csv(path)
        frames.append(normalize_metrics(df, engine))
        used.append({"engine": engine, "path": str(path)})

    if not frames:
        raise FileNotFoundError(f"No metrics files found under {root}")

    combined = pd.concat(frames, ignore_index=True)
    value_cols = ["TP", "FP", "FN", "TN", "precision", "recall", "F1", "FDR"]
    combined = (
        combined.groupby(["engine", "effect_value", "frac_value", "method"], as_index=False)[value_cols]
        .mean(numeric_only=True)
        .sort_values(["engine", "effect_value", "frac_value", "method"])
    )
    return combined, used, missing


def metric_series(curves, method, metric):
    sub = curves[curves["method"] == method].set_index("frac_value").reindex(FRACS)
    return sub[metric].astype(float)


def mean_metric(series, fill_zero=False):
    values = series.astype(float)
    if fill_zero:
        values = values.fillna(0.0)
    if values.notna().sum() == 0:
        return np.nan
    return float(values.mean(skipna=True))


def near_decreasing(series):
    values = series.astype(float).reindex(FRACS).fillna(0.0).to_numpy()
    diffs = np.diff(values)
    positive = diffs[diffs > 0]
    max_rebound = float(positive.max()) if positive.size else 0.0
    max_drop = float((-diffs[diffs < 0]).max()) if np.any(diffs < 0) else 0.0
    small_rebounds = int(np.sum(diffs > 0.10))
    end_not_higher = values[-1] <= values[0] + 0.05
    return {
        "near_decreasing": bool(end_not_higher and max_rebound <= 0.20 and small_rebounds <= 2 and max_drop <= 0.60),
        "max_rebound": max_rebound,
        "max_drop": max_drop,
        "n_rebounds_gt_0p10": small_rebounds,
    }


def evaluate_one(engine, effect_value, curves):
    zero_tp = metric_series(curves, "zeroed", "TP").fillna(0.0)
    imp_tp = metric_series(curves, "imputed", "TP").fillna(0.0)
    zero_recall = metric_series(curves, "zeroed", "recall").fillna(0.0)
    imp_recall = metric_series(curves, "imputed", "recall").fillna(0.0)
    zero_f1 = metric_series(curves, "zeroed", "F1")
    imp_f1 = metric_series(curves, "imputed", "F1")
    zero_fdr = metric_series(curves, "zeroed", "FDR")
    imp_fdr = metric_series(curves, "imputed", "FDR")

    complete_grid = (
        curves["frac_value"].isin(FRACS).all()
        and set(curves["frac_value"].dropna().round(1)) >= set(FRACS)
        and set(curves["method"]) >= {"zeroed", "imputed"}
    )

    zero_tp_frac_0p6 = float(zero_tp.loc[0.6])
    imp_better_tp_or_recall_n = int(((imp_tp > zero_tp) | (imp_recall > zero_recall)).sum())
    zero_mean_f1 = mean_metric(zero_f1, fill_zero=True)
    imp_mean_f1 = mean_metric(imp_f1, fill_zero=True)
    zero_mean_fdr = mean_metric(zero_fdr, fill_zero=True)
    imp_mean_fdr = mean_metric(imp_fdr, fill_zero=True)

    zero_shape = near_decreasing(zero_recall)
    imp_shape = near_decreasing(imp_recall)

    rule1_zero_tp_at_0p6 = zero_tp_frac_0p6 > 0
    rule2_imp_better_7_of_9 = imp_better_tp_or_recall_n >= 7
    rule3_imp_mean_f1_higher = imp_mean_f1 > zero_mean_f1
    rule4_fdr_guard = imp_mean_fdr <= zero_mean_fdr + 0.10
    rule5_recall_shape = zero_shape["near_decreasing"] and imp_shape["near_decreasing"]

    return {
        "engine": engine,
        "effect_value": effect_value,
        "complete_grid": complete_grid,
        "zero_tp_frac_0p6": zero_tp_frac_0p6,
        "imp_better_tp_or_recall_n": imp_better_tp_or_recall_n,
        "zero_mean_F1": zero_mean_f1,
        "imp_mean_F1": imp_mean_f1,
        "zero_mean_FDR": zero_mean_fdr,
        "imp_mean_FDR": imp_mean_fdr,
        "zero_recall_max_rebound": zero_shape["max_rebound"],
        "imp_recall_max_rebound": imp_shape["max_rebound"],
        "zero_recall_max_drop": zero_shape["max_drop"],
        "imp_recall_max_drop": imp_shape["max_drop"],
        "rule1_zero_tp_at_0p6": rule1_zero_tp_at_0p6,
        "rule2_imp_better_7_of_9": rule2_imp_better_7_of_9,
        "rule3_imp_mean_f1_higher": rule3_imp_mean_f1_higher,
        "rule4_fdr_guard": rule4_fdr_guard,
        "rule5_recall_shape": rule5_recall_shape,
        "passes": bool(
            complete_grid
            and rule1_zero_tp_at_0p6
            and rule2_imp_better_7_of_9
            and rule3_imp_mean_f1_higher
            and rule4_fdr_guard
            and rule5_recall_shape
        ),
    }


def evaluate_selection(curves):
    rows = []
    for (engine, effect_value), sub in curves.groupby(["engine", "effect_value"]):
        rows.append(evaluate_one(engine, effect_value, sub))
    summary = pd.DataFrame(rows).sort_values(["effect_value", "engine"])

    pass_rows = []
    required_engine_count = int(curves["engine"].nunique())
    for effect_value, sub in summary.groupby("effect_value"):
        passed_engines = sub.loc[sub["passes"], "engine"].tolist()
        n_engines = int(sub["engine"].nunique())
        n_pass = int(sub["passes"].sum())
        pass_rows.append(
            {
                "effect_value": effect_value,
                "n_engines": n_engines,
                "n_required_engines": required_engine_count,
                "n_pass": n_pass,
                "engines_passed": ",".join(passed_engines),
                "passes_all_engines": bool(n_engines == required_engine_count and n_pass == required_engine_count),
            }
        )
    pass_table = (
        pd.DataFrame(pass_rows)
        .sort_values("effect_value")
    )
    return summary, pass_table


def plot_selected(curves, selected_effect, outdir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped] matplotlib unavailable: {exc}")
        return None

    sub = curves[np.isclose(curves["effect_value"], selected_effect)].copy()
    if sub.empty:
        return None

    engines = sorted(sub["engine"].unique())
    colors = {"zeroed": "#666666", "imputed": "#2b6cb0"}
    fig, axes = plt.subplots(len(MAIN_METRICS), len(engines), figsize=(12, 9), sharex=True)
    if len(engines) == 1:
        axes = np.asarray(axes).reshape(len(MAIN_METRICS), 1)

    for row_idx, metric in enumerate(MAIN_METRICS):
        for col_idx, engine in enumerate(engines):
            ax = axes[row_idx, col_idx]
            one = sub[sub["engine"] == engine]
            for method in ["zeroed", "imputed"]:
                y = metric_series(one, method, metric).fillna(0.0)
                ax.plot(FRACS, y.to_numpy(), marker="o", linewidth=1.8, color=colors[method], label=method)
            if row_idx == 0:
                ax.set_title(engine)
            if col_idx == 0:
                ax.set_ylabel(metric)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(0.08, 0.92)

    for ax in axes[-1, :]:
        ax.set_xlabel("masking fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Selected effect = {selected_effect:g}")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    path = outdir / f"selected_effect_{str(selected_effect).replace('.', 'p')}_main_metrics.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="simulation3_effect_tune")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir) if args.outdir else root / "selection_summary"
    outdir.mkdir(parents=True, exist_ok=True)

    curves, used, missing = load_all_metrics(root)
    summary, candidates = evaluate_selection(curves)

    curves.to_csv(outdir / "tune_selection_method_curves.csv", index=False)
    summary.to_csv(outdir / "tune_selection_rule_summary.csv", index=False)
    candidates.to_csv(outdir / "tune_selection_candidates.csv", index=False)
    pd.DataFrame(used).to_csv(outdir / "tune_selection_sources.csv", index=False)

    passed = candidates[candidates["passes_all_engines"]]
    selected = None
    plot_path = None
    if not passed.empty:
        selected = float(passed["effect_value"].min())
        plot_path = plot_selected(curves, selected, outdir)

    print("[sources]")
    for row in used:
        print(f"  {row['engine']}: {row['path']}")
    if missing:
        print("[missing engines] " + ", ".join(missing))
    if selected is None:
        print("[selection] no effect passes all three engines")
    else:
        print(f"[selection] selected minimum passing effect: {selected:g}")
        if plot_path is not None:
            print(f"[plot] {plot_path}")
    print(f"[summary] {outdir}")


if __name__ == "__main__":
    main()
