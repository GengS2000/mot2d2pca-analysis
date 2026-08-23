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
from src.metrics import simple_dtw, summarize_table, trajectory_metrics_long
from src.penalized_2d2_pca import _bin_to_time_value, build_observed_mask
from src.plotting import ensure_dir, save_boxplot, save_metric_vs_frac, save_trajectory_plot


def clustering_metrics(da_true, da_masked, da_imp, method, mechanism, frac, repeat):
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except Exception:
        return {"method": method, "mechanism": mechanism, "frac": frac, "repeat": repeat, "ari_imputed": np.nan, "nmi_imputed": np.nan, "ari_masked": np.nan, "nmi_masked": np.nan, "warning": "sklearn unavailable"}
    n_clusters = min(4, da_true.sizes["taxon"])
    if n_clusters < 2:
        return {"method": method, "mechanism": mechanism, "frac": frac, "repeat": repeat, "ari_imputed": np.nan, "nmi_imputed": np.nan, "ari_masked": np.nan, "nmi_masked": np.nan, "warning": "too few taxa"}
    mats = [x.values.mean(axis=0) for x in [da_true, da_masked, da_imp]]
    labels = [KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(m) for m in mats]
    return {"method": method, "mechanism": mechanism, "frac": frac, "repeat": repeat, "ari_imputed": adjusted_rand_score(labels[0], labels[2]), "nmi_imputed": normalized_mutual_info_score(labels[0], labels[2]), "ari_masked": adjusted_rand_score(labels[0], labels[1]), "nmi_masked": normalized_mutual_info_score(labels[0], labels[1]), "warning": ""}


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Experiment 3: trajectory preservation"))
    parser.set_defaults(fracs="0.10,0.20", mechanisms="random_entry,consecutive_time_window_dropout,subject_specific_dropout", replacement_modes="known_mask", methods="penalized_2d2_pca,no_imputation")
    args = parser.parse_args()
    outdir = Path(args.outdir) / "experiment3_trajectory"
    da_log, _da_raw, observed_positive_mask, meta_info, zero_value, methods, mechanisms, fracs, replacement_modes = load_data_from_args(args, outdir)
    save_config(outdir / "config.json", args, {"experiment": "trajectory_preservation", "meta_info": meta_info})
    pca_params = pca_params_from_args(args)
    seeds = seeds_from_args(args)
    rows, subject_rows, cluster_rows, dtw_rows, failures = [], [], [], [], []
    figdir = ensure_dir(outdir / "figures")
    for mechanism in mechanisms:
        for frac in fracs:
            for repeat, seed in enumerate(seeds):
                try:
                    artificial_mask, masked_da, _mask_meta = make_masked_dataset(da_log, observed_positive_mask, mechanism, frac, seed, zero_value, {})
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
                        print(f"[experiment3] mechanism={mechanism} frac={frac} repeat={repeat} mode={mode} method={method}", flush=True)
                        try:
                            da_imp = impute_method(masked_da, replace_mask, observed_mask, method, {}, pca_params, zero_value, seed + 300000)
                            df_traj, df_subj = trajectory_metrics_long(da_log, masked_da, da_imp, artificial_mask, method, mechanism, frac, repeat)
                            df_traj["replacement_mode"] = mode
                            df_subj["replacement_mode"] = mode
                            rows.append(df_traj)
                            subject_rows.append(df_subj)
                            cluster_rows.append({**clustering_metrics(da_log, masked_da, da_imp, method, mechanism, frac, repeat), "replacement_mode": mode})
                            affected = np.argwhere(artificial_mask.values.any(axis=2))[:25]
                            for i, j in affected:
                                dtw_rows.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "replacement_mode": mode, "method": method, "participant": str(da_log.coords["participant"].values[i]), "taxon": str(da_log.coords["taxon"].values[j]), "dtw_imputed": simple_dtw(da_log.values[i, j], da_imp.values[i, j]), "dtw_masked": simple_dtw(da_log.values[i, j], masked_da.values[i, j])})
                            if repeat == 0 and method == "penalized_2d2_pca" and affected.size:
                                times = [_bin_to_time_value(b) for b in da_log.coords["bin"].values]
                                for idx, (i, j) in enumerate(affected[:3]):
                                    save_trajectory_plot(times, da_log.values[i, j], masked_da.values[i, j], da_imp.values[i, j], artificial_mask.values[i, j], figdir / f"trajectory_{mechanism}_{mode}_{idx}.png", f"{mechanism} {mode}")
                        except Exception as exc:
                            failures.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "method": method, "replacement_mode": mode, "error": repr(exc), "traceback": traceback.format_exc()})
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    subj = pd.concat(subject_rows, ignore_index=True) if subject_rows else pd.DataFrame()
    clusters = pd.DataFrame(cluster_rows)
    dtw = pd.DataFrame(dtw_rows)
    df = pd.concat([df, dtw], ignore_index=True, sort=False) if not dtw.empty else df
    df.to_csv(outdir / "trajectory_metrics_long.csv", index=False)
    summarize_table(df, ["mechanism", "frac", "replacement_mode", "method"]).to_csv(outdir / "trajectory_metrics_summary.csv", index=False)
    subj.to_csv(outdir / "subject_level_metrics.csv", index=False)
    clusters.to_csv(outdir / "taxon_clustering_metrics.csv", index=False)
    pd.DataFrame(failures).to_csv(outdir / "failed_runs.csv", index=False)
    save_boxplot(df, figdir / "lag1_autocorrelation_recovery_boxplot.png", "lag1_error_imputed", "method", "Lag-1 autocorrelation recovery")
    save_boxplot(df, figdir / "second_difference_distribution.png", "second_difference_mae_imputed", "method", "Second-difference error")
    save_metric_vs_frac(df, figdir / "trajectory_rmse_by_method_mechanism.png", "trajectory_rmse", "Trajectory RMSE")
    print(f"[experiment3] wrote {outdir}")


if __name__ == "__main__":
    main()
