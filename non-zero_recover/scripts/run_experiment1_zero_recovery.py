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
from src.metrics import evaluate_recovery, summarize_table, trajectory_summary_metrics
from src.penalized_2d2_pca import build_observed_mask
from src.plotting import ensure_dir, save_metric_vs_frac, save_ranking_heatmap


def replacement_mask_for_mode(masked_da, artificial_mask, zero_value, mode):
    """Return the provided imputation mask for semi-synthetic known-mask evaluation."""
    if mode != "known_mask":
        raise ValueError(f"Unsupported replacement mode for this method-only experiment: {mode}")
    return artificial_mask


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Experiment 1: semi-synthetic zero recovery"))
    args = parser.parse_args()
    outdir = Path(args.outdir) / "experiment1_zero_recovery"
    da_log, _da_raw, observed_positive_mask, meta_info, zero_value, methods, mechanisms, fracs, replacement_modes = load_data_from_args(args, outdir)
    save_config(outdir / "config.json", args, {"experiment": "zero_recovery", "meta_info": meta_info})
    pca_params = pca_params_from_args(args)
    seeds = seeds_from_args(args)
    rows, failures = [], []
    total = len(mechanisms) * len(fracs) * int(args.repeats) * len(replacement_modes) * len(methods)
    step = 0
    for mechanism in mechanisms:
        for frac in fracs:
            for repeat, seed in enumerate(seeds):
                try:
                    artificial_mask, masked_da, mask_meta = make_masked_dataset(da_log, observed_positive_mask, mechanism, frac, seed, zero_value, {})
                except Exception as exc:
                    failures.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "method": "", "replacement_mode": "", "error": repr(exc), "traceback": traceback.format_exc()})
                    continue
                for mode in replacement_modes:
                    try:
                        replace_mask = replacement_mask_for_mode(masked_da, artificial_mask, zero_value, mode)
                    except Exception as exc:
                        failures.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "method": "", "replacement_mode": mode, "error": repr(exc), "traceback": traceback.format_exc()})
                        continue
                    observed_mask = build_observed_mask(masked_da, zero_value, mode="positive_only_except_replace", replace_mask=replace_mask)
                    for method in methods:
                        step += 1
                        print(f"[experiment1] {step}/{total} mechanism={mechanism} frac={frac} repeat={repeat} mode={mode} method={method}", flush=True)
                        try:
                            da_imp = impute_method(masked_da, replace_mask, observed_mask, method, {}, pca_params, zero_value, seed + 100000 + repeat)
                            rec = evaluate_recovery(da_log, da_imp, artificial_mask)
                            traj = trajectory_summary_metrics(da_log, masked_da, da_imp, artificial_mask)
                            rows.append({
                                "mechanism": mechanism, "frac": float(frac), "repeat": int(repeat), "seed": int(seed),
                                "replacement_mode": mode, "method": method,
                                **rec, **traj,
                                "imputation_mask_n": int(replace_mask.values.sum()),
                                "evaluation_mask_n": int(artificial_mask.values.sum()),
                                "n_masked": mask_meta["n_masked"], "eligible_n": mask_meta["eligible_n"], "mask_rate": mask_meta["mask_rate"],
                            })
                        except Exception as exc:
                            failures.append({"mechanism": mechanism, "frac": frac, "repeat": repeat, "method": method, "replacement_mode": mode, "error": repr(exc), "traceback": traceback.format_exc()})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "metrics_long.csv", index=False)
    summary = summarize_table(df, ["mechanism", "frac", "replacement_mode", "method"])
    summary.to_csv(outdir / "metrics_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(outdir / "failed_runs.csv", index=False)
    figdir = ensure_dir(outdir / "figures")
    for mechanism, sub in df.groupby("mechanism") if not df.empty else []:
        save_metric_vs_frac(sub, figdir / f"{mechanism}_rmse_vs_frac.png", "rmse", f"{mechanism}: RMSE")
        save_metric_vs_frac(sub, figdir / f"{mechanism}_pearson_vs_frac.png", "pearson", f"{mechanism}: Pearson")
    save_ranking_heatmap(df, figdir / "method_ranking_heatmap.png", metric="rmse")
    print(f"[experiment1] wrote {outdir}")


if __name__ == "__main__":
    main()
