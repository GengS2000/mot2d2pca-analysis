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
from src.metrics import corr_matrix_similarity, edge_overlap_metrics, lagged_correlation_matrix, safe_pearson, static_correlation_matrix, top_edges
from src.penalized_2d2_pca import build_observed_mask
from src.plotting import ensure_dir, save_boxplot, save_heatmap, save_scatter


def degree_table(edges: pd.DataFrame) -> pd.Series:
    if edges.empty:
        return pd.Series(dtype=float)
    deg = pd.concat([edges["source_taxon"], edges["target_taxon"]]).value_counts()
    return deg


def hub_metrics(ref_edges: pd.DataFrame, cand_edges: pd.DataFrame, top_n: int = 20):
    ref_deg = degree_table(ref_edges)
    cand_deg = degree_table(cand_edges)
    taxa = sorted(set(ref_deg.index).union(set(cand_deg.index)))
    ref_vec = np.asarray([ref_deg.get(t, 0) for t in taxa], dtype=float)
    cand_vec = np.asarray([cand_deg.get(t, 0) for t in taxa], dtype=float)
    ref_hubs = set(ref_deg.head(top_n).index)
    cand_hubs = set(cand_deg.head(top_n).index)
    inter = len(ref_hubs & cand_hubs)
    union = len(ref_hubs | cand_hubs)
    return {
        "degree_distribution_pearson": safe_pearson(ref_vec, cand_vec),
        "hub_overlap": inter,
        "hub_jaccard": inter / union if union else np.nan,
        "hub_top_n": int(top_n),
        "reference_hubs": ";".join(map(str, list(ref_hubs)[:top_n])),
        "candidate_hubs": ";".join(map(str, list(cand_hubs)[:top_n])),
    }


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Experiment 4: dynamic taxon-taxon correlation recovery"))
    parser.set_defaults(fracs="0.10,0.20", mechanisms="random_entry,low_abundance_biased,consecutive_time_window_dropout", replacement_modes="known_mask", methods="penalized_2d2_pca,no_imputation")
    parser.add_argument("--top-variable-taxa", type=int, default=40)
    args = parser.parse_args()
    outdir = Path(args.outdir) / "experiment4_dynamic_correlation"
    da_log, _da_raw, observed_positive_mask, meta_info, zero_value, methods, mechanisms, fracs, replacement_modes = load_data_from_args(args, outdir)
    save_config(outdir / "config.json", args, {"experiment": "dynamic_correlation", "meta_info": meta_info})
    pca_params = pca_params_from_args(args)
    seeds = seeds_from_args(args)
    taxa_var = np.nanvar(da_log.values, axis=(0, 2))
    top_n = min(int(args.top_variable_taxa), da_log.sizes["taxon"])
    top_idx = np.argsort(taxa_var)[::-1][:top_n]
    top_taxa = da_log.coords["taxon"].values[top_idx]
    static_rows, lagged_rows, edge_rows, hub_rows, failures = [], [], [], [], []
    figdir = ensure_dir(outdir / "figures")

    for mechanism in mechanisms:
        for frac in fracs:
            for repeat, seed in enumerate(seeds):
                try:
                    artificial_mask, masked_da, _mask_meta = make_masked_dataset(da_log, observed_positive_mask, mechanism, frac, seed, zero_value, {})
                    ref_static, _ = static_correlation_matrix(da_log, top_idx)
                    ref_static_nonzero, n_pair = static_correlation_matrix(da_log, top_idx, nonzero_value=zero_value)
                    ref_lagged = lagged_correlation_matrix(da_log, top_idx)
                    masked_static, _ = static_correlation_matrix(masked_da, top_idx)
                    masked_lagged = lagged_correlation_matrix(masked_da, top_idx)
                    ref_edges = top_edges(ref_lagged, top_taxa, k=max(200, top_n * 5))
                    if repeat == 0:
                        save_heatmap(ref_static, figdir / f"static_corr_original_{mechanism}_{frac}.png", top_taxa, "Original static correlation")
                        save_heatmap(ref_lagged, figdir / f"lagged_corr_original_{mechanism}_{frac}.png", top_taxa, "Original lagged correlation")
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
                        print(f"[experiment4] mechanism={mechanism} frac={frac} repeat={repeat} mode={mode} method={method}", flush=True)
                        try:
                            da_imp = impute_method(masked_da, replace_mask, observed_mask, method, {}, pca_params, zero_value, seed + 400000)
                            imp_static, _ = static_correlation_matrix(da_imp, top_idx)
                            imp_static_nonzero, _ = static_correlation_matrix(da_imp, top_idx, nonzero_value=zero_value)
                            imp_lagged = lagged_correlation_matrix(da_imp, top_idx)
                            s = corr_matrix_similarity(ref_static, imp_static, upper_only=True)
                            s_mask = corr_matrix_similarity(ref_static, masked_static, upper_only=True)
                            snz = corr_matrix_similarity(ref_static_nonzero, imp_static_nonzero, upper_only=True)
                            static_rows.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "replacement_mode": mode, "method": method, **{f"static_{k}": v for k, v in s.items()}, **{f"masked_static_{k}": v for k, v in s_mask.items()}, **{f"nonzero_static_{k}": v for k, v in snz.items()}, "nonzero_pair_n_min": int(np.nanmin(n_pair)), "nonzero_pair_n_median": float(np.nanmedian(n_pair))})
                            l = corr_matrix_similarity(ref_lagged, imp_lagged, upper_only=False)
                            l_mask = corr_matrix_similarity(ref_lagged, masked_lagged, upper_only=False)
                            lagged_rows.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "replacement_mode": mode, "method": method, **{f"lagged_{k}": v for k, v in l.items()}, **{f"masked_lagged_{k}": v for k, v in l_mask.items()}})
                            imp_edges = top_edges(imp_lagged, top_taxa, k=max(200, top_n * 5))
                            edge_metric = edge_overlap_metrics(ref_edges, imp_edges, k_values=(50, 100, 200))
                            edge_metric["mechanism"] = mechanism
                            edge_metric["frac"] = frac
                            edge_metric["repeat"] = repeat
                            edge_metric["replacement_mode"] = mode
                            edge_metric["method"] = method
                            edge_rows.append(edge_metric)
                            hub_rows.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "replacement_mode": mode, "method": method, **hub_metrics(ref_edges, imp_edges)})
                            if repeat == 0 and method == "penalized_2d2_pca":
                                save_heatmap(imp_static, figdir / f"static_corr_imputed_{mechanism}_{mode}_{frac}.png", top_taxa, "Imputed static correlation")
                                save_heatmap(imp_lagged, figdir / f"lagged_corr_imputed_{mechanism}_{mode}_{frac}.png", top_taxa, "Imputed lagged correlation")
                                scatter_df = pd.DataFrame({"original": ref_lagged.ravel(), "imputed": imp_lagged.ravel()})
                                save_scatter(scatter_df, figdir / f"lagged_corr_scatter_{mechanism}_{mode}_{frac}.png", "original", "imputed", title="Original vs imputed lagged correlations")
                        except Exception as exc:
                            failures.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "method": method, "replacement_mode": mode, "error": repr(exc), "traceback": traceback.format_exc()})
    static_df = pd.DataFrame(static_rows)
    lagged_df = pd.DataFrame(lagged_rows)
    edge_df = pd.concat(edge_rows, ignore_index=True) if edge_rows else pd.DataFrame()
    hub_df = pd.DataFrame(hub_rows)
    static_df.to_csv(outdir / "static_correlation_metrics.csv", index=False)
    lagged_df.to_csv(outdir / "lagged_correlation_metrics.csv", index=False)
    edge_df.to_csv(outdir / "network_edge_overlap.csv", index=False)
    hub_df.to_csv(outdir / "hub_taxa_overlap.csv", index=False)
    pd.DataFrame(failures).to_csv(outdir / "failed_runs.csv", index=False)
    save_boxplot(edge_df, figdir / "topk_network_overlap_barplot.png", "topk_precision", "method", "Top-k network precision")
    save_boxplot(hub_df, figdir / "hub_taxa_comparison.png", "hub_jaccard", "method", "Hub taxa overlap")
    print(f"[experiment4] wrote {outdir}")


if __name__ == "__main__":
    main()
