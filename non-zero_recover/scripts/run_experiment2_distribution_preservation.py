from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_runner import add_common_args, load_data_from_args, pca_params_from_args, save_config, seeds_from_args
from src.imputation_wrappers import impute_method
from src.masking import make_masked_dataset
from src.metrics import js_divergence, safe_spearman, safe_wasserstein, summarize_table, taxon_distribution_metrics
from src.penalized_2d2_pca import build_observed_mask


DISTRIBUTION_QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def finite_1d(values) -> np.ndarray:
    vals = np.asarray(values, dtype=float).ravel()
    return vals[np.isfinite(vals)]


def distribution_summary(values, prefix: str) -> dict:
    vals = finite_1d(values)
    summary = {f"{prefix}_n": int(vals.size)}
    stats = {
        "mean": np.nan,
        "sd": np.nan,
        "min": np.nan,
        "q05": np.nan,
        "q10": np.nan,
        "q25": np.nan,
        "median": np.nan,
        "q75": np.nan,
        "q90": np.nan,
        "q95": np.nan,
        "max": np.nan,
    }
    if vals.size:
        quantiles = np.quantile(vals, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        stats.update(
            {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals)),
                "min": float(np.min(vals)),
                "q05": float(quantiles[0]),
                "q10": float(quantiles[1]),
                "q25": float(quantiles[2]),
                "median": float(quantiles[3]),
                "q75": float(quantiles[4]),
                "q90": float(quantiles[5]),
                "q95": float(quantiles[6]),
                "max": float(np.max(vals)),
            }
        )
    summary.update({f"{prefix}_{name}": value for name, value in stats.items()})
    return summary


def distribution_quantile_rows(context: dict, distributions: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for name, values in distributions.items():
        vals = finite_1d(values)
        quantile_values = np.quantile(vals, DISTRIBUTION_QUANTILES) if vals.size else np.full(len(DISTRIBUTION_QUANTILES), np.nan)
        for q, value in zip(DISTRIBUTION_QUANTILES, quantile_values):
            rows.append(
                {
                    **context,
                    "distribution": name,
                    "quantile": float(q),
                    "log10_profile_plus_eps": float(value),
                    "n_values": int(vals.size),
                }
            )
    return rows


def raw_nonzero_comparison_row(context: dict, raw_nonzero_values, masked_truth_values, imputed_at_masked_values) -> dict:
    raw = finite_1d(raw_nonzero_values)
    masked_truth = finite_1d(masked_truth_values)
    imputed = finite_1d(imputed_at_masked_values)
    row = {
        **context,
        "raw_nonzero_vs_imputed_at_masked_wasserstein": safe_wasserstein(raw, imputed),
        "raw_nonzero_vs_imputed_at_masked_js": js_divergence(raw, imputed),
        "raw_nonzero_vs_masked_truth_wasserstein": safe_wasserstein(raw, masked_truth),
        "raw_nonzero_vs_masked_truth_js": js_divergence(raw, masked_truth),
        "masked_truth_vs_imputed_at_masked_wasserstein": safe_wasserstein(masked_truth, imputed),
        "masked_truth_vs_imputed_at_masked_js": js_divergence(masked_truth, imputed),
    }
    row.update(distribution_summary(raw, "raw_nonzero"))
    row.update(distribution_summary(masked_truth, "masked_truth"))
    row.update(distribution_summary(imputed, "imputed_at_masked"))
    return row


def safe_filename(value) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text)[:140]


def plotting_helpers_or_noops():
    try:
        from src.plotting import ensure_dir, save_boxplot, save_hist_overlay, save_scatter

        return ensure_dir, save_boxplot, save_hist_overlay, save_scatter
    except Exception as exc:
        print(f"[plotting warning] plotting helpers unavailable; CSV outputs already written ({exc!r})")

        def ensure_dir(path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            return path

        def save_boxplot(*_args, **_kwargs):
            return None

        def save_hist_overlay(*_args, **_kwargs):
            return None

        def save_scatter(*_args, **_kwargs):
            return None

        return ensure_dir, save_boxplot, save_hist_overlay, save_scatter


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Experiment 2: nonzero abundance distribution preservation"))
    parser.add_argument("--save-figures", action="store_true", help="Optionally save PNG figures. CSV outputs are always written.")
    parser.set_defaults(fracs="0.10,0.20", mechanisms="random_entry,low_abundance_biased,consecutive_time_window_dropout", replacement_modes="known_mask", methods="penalized_2d2_pca,no_imputation")
    args = parser.parse_args()
    outdir = Path(args.outdir) / "experiment2_distribution"
    da_log, _da_raw, observed_positive_mask, meta_info, zero_value, methods, mechanisms, fracs, replacement_modes = load_data_from_args(args, outdir)
    save_config(outdir / "config.json", args, {"experiment": "distribution_preservation", "meta_info": meta_info})
    raw_nonzero_values = finite_1d(da_log.values[observed_positive_mask.values.astype(bool)])
    pd.DataFrame([distribution_summary(raw_nonzero_values, "raw_nonzero")]).to_csv(outdir / "raw_nonzero_da_log_distribution_summary.csv", index=False)
    pd.DataFrame(distribution_quantile_rows({}, {"raw_da_log_nonzero": raw_nonzero_values})).to_csv(outdir / "raw_nonzero_da_log_distribution_quantiles.csv", index=False)
    pca_params = pca_params_from_args(args)
    seeds = seeds_from_args(args)
    rows, rank_rows, raw_comparison_rows, raw_comparison_quantile_rows, failures = [], [], [], [], []
    plot_distributions = {} if args.save_figures else None
    taxon_hist_masked_da = None
    for mechanism in mechanisms:
        for frac in fracs:
            for repeat, seed in enumerate(seeds):
                try:
                    artificial_mask, masked_da, _mask_meta = make_masked_dataset(da_log, observed_positive_mask, mechanism, frac, seed, zero_value, {})
                    if taxon_hist_masked_da is None:
                        taxon_hist_masked_da = masked_da
                    masked_truth_values = finite_1d(da_log.values[artificial_mask.values.astype(bool)])
                except Exception as exc:
                    failures.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "method": "", "replacement_mode": "", "error": repr(exc), "traceback": traceback.format_exc()})
                    continue
                for mode in replacement_modes:
                    if mode != "known_mask":
                        failures.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "method": "", "replacement_mode": mode, "error": f"unsupported replacement_mode={mode}", "traceback": ""})
                        continue
                    replace_mask = artificial_mask
                    observed_mask = build_observed_mask(masked_da, zero_value, "positive_only_except_replace", replace_mask)
                    for method in methods:
                        print(f"[experiment2] mechanism={mechanism} frac={frac} repeat={repeat} mode={mode} method={method}", flush=True)
                        try:
                            da_imp = impute_method(masked_da, replace_mask, observed_mask, method, {}, pca_params, zero_value, seed + 200000)
                            df_tax = taxon_distribution_metrics(da_log, masked_da, da_imp, artificial_mask, method, mechanism, frac, repeat)
                            df_tax["replacement_mode"] = mode
                            rows.append(df_tax)
                            orig_mean = da_log.values.mean(axis=(0, 2))
                            imp_mean = da_imp.values.mean(axis=(0, 2))
                            rank_rows.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "replacement_mode": mode, "method": method, "taxon_mean_rank_spearman": safe_spearman(orig_mean, imp_mean)})
                            imputed_at_masked_values = finite_1d(da_imp.values[artificial_mask.values.astype(bool)])
                            comparison_context = {
                                "mechanism": mechanism,
                                "frac": float(frac),
                                "repeat": int(repeat),
                                "seed": int(seed),
                                "replacement_mode": mode,
                                "method": method,
                            }
                            raw_comparison_rows.append(
                                raw_nonzero_comparison_row(
                                    comparison_context,
                                    raw_nonzero_values,
                                    masked_truth_values,
                                    imputed_at_masked_values,
                                )
                            )
                            raw_comparison_quantile_rows.extend(
                                distribution_quantile_rows(
                                    comparison_context,
                                    {
                                        "raw_da_log_nonzero": raw_nonzero_values,
                                        "masked_truth_at_artificial_mask": masked_truth_values,
                                        "imputed_at_artificial_mask": imputed_at_masked_values,
                                    },
                                )
                            )
                            if args.save_figures and repeat == 0:
                                plot_key = (mechanism, float(frac), mode)
                                if plot_key not in plot_distributions:
                                    plot_distributions[plot_key] = {
                                        "raw da_log non-zero": raw_nonzero_values,
                                        "masked true at artificial mask": masked_truth_values,
                                    }
                                plot_distributions[plot_key][f"{method} imputed at mask"] = imputed_at_masked_values
                        except Exception as exc:
                            failures.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "method": method, "replacement_mode": mode, "error": repr(exc), "traceback": traceback.format_exc()})
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    ranks = pd.DataFrame(rank_rows)
    raw_comparison = pd.DataFrame(raw_comparison_rows)
    raw_comparison_quantiles = pd.DataFrame(raw_comparison_quantile_rows)
    if not ranks.empty:
        df = df.merge(ranks, on=["mechanism", "frac", "repeat", "replacement_mode", "method"], how="left")
    df.to_csv(outdir / "distribution_metrics_long.csv", index=False)
    summarize_table(df, ["mechanism", "frac", "replacement_mode", "method"]).to_csv(outdir / "distribution_metrics_summary.csv", index=False)
    raw_comparison.to_csv(outdir / "raw_nonzero_vs_imputed_at_masked_distribution_metrics.csv", index=False)
    summarize_table(raw_comparison, ["mechanism", "frac", "replacement_mode", "method"]).to_csv(outdir / "raw_nonzero_vs_imputed_at_masked_distribution_summary.csv", index=False)
    raw_comparison_quantiles.to_csv(outdir / "raw_nonzero_vs_imputed_at_masked_distribution_quantiles.csv", index=False)
    pd.DataFrame(failures).to_csv(outdir / "failed_runs.csv", index=False)
    if args.save_figures:
        ensure_dir, save_boxplot, save_hist_overlay, save_scatter = plotting_helpers_or_noops()
        try:
            figdir = ensure_dir(outdir / "figures")
            save_boxplot(df, figdir / "wasserstein_distance_boxplot.png", "wasserstein_masked", "method", "Taxon-wise masked Wasserstein")
            save_boxplot(df, figdir / "js_divergence_boxplot.png", "js_full", "method", "Taxon-wise JS divergence")
            save_scatter(df, figdir / "mean_sd_original.png", "original_mean", "original_sd", "method", "Original mean-SD")
            save_scatter(df, figdir / "mean_sd_imputed.png", "imputed_mean", "imputed_sd", "method", "Imputed mean-SD")
            save_boxplot(ranks, figdir / "abundance_rank_preservation.png", "taxon_mean_rank_spearman", "method", "Taxon mean rank preservation")
            save_boxplot(raw_comparison, figdir / "raw_nonzero_vs_imputed_at_masked_wasserstein_boxplot.png", "raw_nonzero_vs_imputed_at_masked_wasserstein", "method", "Raw non-zero vs imputed-at-masked Wasserstein")
            save_boxplot(raw_comparison, figdir / "raw_nonzero_vs_imputed_at_masked_js_boxplot.png", "raw_nonzero_vs_imputed_at_masked_js", "method", "Raw non-zero vs imputed-at-masked JS divergence")
            save_hist_overlay({"raw da_log non-zero": raw_nonzero_values}, figdir / "raw_da_log_nonzero_distribution_hist.png", "Raw da_log non-zero distribution")
            for (mechanism, frac, mode), values in plot_distributions.items():
                suffix = safe_filename(f"{mechanism}_frac{frac:g}_{mode}_repeat0")
                title = f"raw non-zero vs imputed-at-masked: {mechanism}, frac={frac:g}, {mode}, repeat=0"
                save_hist_overlay(values, figdir / f"raw_nonzero_vs_imputed_at_masked_hist_{suffix}.png", title)
            if rows:
                first = df.sort_values("n_masked", ascending=False).head(4)["taxon"].unique()
                for taxon in first:
                    j = list(da_log.coords["taxon"].values).index(taxon)
                    save_hist_overlay({
                        "original true": da_log.values[:, j, :].ravel(),
                        "masked sparse": taxon_hist_masked_da.values[:, j, :].ravel(),
                    }, figdir / f"hist_{abs(hash(taxon)) % 10**8}.png", str(taxon).split("|")[-1])
        except Exception as exc:
            print(f"[plotting warning] skipped figures after CSV export ({exc!r})")
    print(f"[experiment2] wrote {outdir}")


if __name__ == "__main__":
    main()
