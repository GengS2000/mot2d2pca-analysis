from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_runner import add_common_args, load_data_from_args, pca_params_from_args, save_config, seeds_from_args
from src.imputation_wrappers import impute_method
from src.masking import make_masked_dataset
from src.metrics import simple_dtw, trajectory_metrics_long
from src.penalized_2d2_pca import build_observed_mask


FAILURE_COLUMNS = ["mechanism", "frac", "repeat", "method", "replacement_mode", "error", "traceback"]
CLUSTER_COLUMNS = [
    "method",
    "mechanism",
    "frac",
    "repeat",
    "replacement_mode",
    "ari_imputed",
    "nmi_imputed",
    "ari_masked",
    "nmi_masked",
    "warning",
]


def clustering_metrics(da_true, da_masked, da_imp, method, mechanism, frac, repeat):
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except Exception:
        return {
            "method": method,
            "mechanism": mechanism,
            "frac": frac,
            "repeat": repeat,
            "ari_imputed": np.nan,
            "nmi_imputed": np.nan,
            "ari_masked": np.nan,
            "nmi_masked": np.nan,
            "warning": "sklearn unavailable",
        }
    n_clusters = min(4, da_true.sizes["taxon"])
    if n_clusters < 2:
        return {
            "method": method,
            "mechanism": mechanism,
            "frac": frac,
            "repeat": repeat,
            "ari_imputed": np.nan,
            "nmi_imputed": np.nan,
            "ari_masked": np.nan,
            "nmi_masked": np.nan,
            "warning": "too few taxa",
        }
    mats = [x.values.mean(axis=0) for x in [da_true, da_masked, da_imp]]
    labels = [KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(m) for m in mats]
    return {
        "method": method,
        "mechanism": mechanism,
        "frac": frac,
        "repeat": repeat,
        "ari_imputed": adjusted_rand_score(labels[0], labels[2]),
        "nmi_imputed": normalized_mutual_info_score(labels[0], labels[2]),
        "ari_masked": adjusted_rand_score(labels[0], labels[1]),
        "nmi_masked": normalized_mutual_info_score(labels[0], labels[1]),
        "warning": "",
    }


def scenario_grid(mechanisms: list[str], fracs: list[float], seeds: list[int]) -> list[dict[str, object]]:
    rows = []
    for mechanism in mechanisms:
        for frac in fracs:
            for repeat, seed in enumerate(seeds):
                rows.append({"mechanism": mechanism, "frac": float(frac), "repeat": int(repeat), "seed": int(seed)})
    return rows


def append_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def remove_outputs(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Experiment 3 chunk runner: one mechanism x frac x repeat."))
    parser.set_defaults(methods="penalized_2d2_pca,no_imputation", replacement_modes="known_mask")
    parser.add_argument("--chunk-index", type=int, default=None, help="Array index for the scenario chunk.")
    parser.add_argument("--chunk-base", type=int, default=1, help="Use 1 for SLURM arrays like 1-N, or 0 for 0-N-1.")
    parser.add_argument("--dtw-trajectories", type=int, default=25, help="Affected trajectories sampled per method for DTW.")
    args = parser.parse_args()

    if args.chunk_index is None:
        env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_idx is None:
            raise ValueError("--chunk-index is required outside a SLURM array")
        args.chunk_index = int(env_idx)

    outdir = Path(args.outdir) / "experiment3_trajectory"
    chunks_dir = outdir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    da_log, _da_raw, observed_positive_mask, meta_info, zero_value, methods, mechanisms, fracs, replacement_modes = load_data_from_args(args, outdir)
    if replacement_modes != ["known_mask"]:
        raise ValueError(f"Experiment 3 chunk runner only supports known_mask, got {replacement_modes}")
    pca_params = pca_params_from_args(args)
    seeds = seeds_from_args(args)
    scenarios = scenario_grid(mechanisms, fracs, seeds)
    scenario_pos = int(args.chunk_index) - int(args.chunk_base)
    if scenario_pos < 0 or scenario_pos >= len(scenarios):
        raise ValueError(f"chunk-index {args.chunk_index} with base {args.chunk_base} is outside 0..{len(scenarios)-1}")
    scenario = scenarios[scenario_pos]
    part = f"{int(args.chunk_index):06d}"

    if scenario_pos == 0:
        save_config(outdir / "config.json", args, {"experiment": "trajectory_preservation_chunked", "meta_info": meta_info, "chunk_count": len(scenarios)})

    traj_path = chunks_dir / f"trajectory_metrics_long_part_{part}.csv"
    subj_path = chunks_dir / f"subject_level_metrics_part_{part}.csv"
    cluster_path = chunks_dir / f"taxon_clustering_metrics_part_{part}.csv"
    failure_path = chunks_dir / f"failed_runs_part_{part}.csv"
    manifest_path = chunks_dir / f"chunk_manifest_part_{part}.csv"
    output_paths = [traj_path, subj_path, cluster_path, failure_path, manifest_path]
    if args.overwrite:
        remove_outputs(output_paths)
    elif all(path.exists() for path in [traj_path, subj_path, cluster_path, manifest_path]):
        print(f"[experiment3-chunk] chunk {args.chunk_index} already exists; use --overwrite to rerun", flush=True)
        return

    failures = []
    manifest = {
        "chunk_index": int(args.chunk_index),
        "chunk_base": int(args.chunk_base),
        "chunk_count": len(scenarios),
        "mechanism": scenario["mechanism"],
        "frac": scenario["frac"],
        "repeat": scenario["repeat"],
        "seed": scenario["seed"],
        "methods": ",".join(methods),
        "replacement_mode": "known_mask",
    }
    print(
        "[experiment3-chunk] "
        f"chunk={args.chunk_index}/{len(scenarios)} mechanism={scenario['mechanism']} "
        f"frac={scenario['frac']} repeat={scenario['repeat']} methods={methods}",
        flush=True,
    )

    try:
        artificial_mask, masked_da, _mask_meta = make_masked_dataset(
            da_log,
            observed_positive_mask,
            str(scenario["mechanism"]),
            float(scenario["frac"]),
            int(scenario["seed"]),
            zero_value,
            {},
        )
        replace_mask = artificial_mask
        observed_mask = build_observed_mask(masked_da, zero_value, "positive_only_except_replace", replace_mask)
    except Exception as exc:
        failures.append(
            {
                "mechanism": scenario["mechanism"],
                "frac": scenario["frac"],
                "repeat": scenario["repeat"],
                "method": "",
                "replacement_mode": "known_mask",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(failure_path, index=False)
        pd.DataFrame([manifest | {"status": "mask_failed"}]).to_csv(manifest_path, index=False)
        raise

    affected = np.argwhere(artificial_mask.values.any(axis=2))[: max(int(args.dtw_trajectories), 0)]
    cluster_rows = []
    for method in methods:
        print(
            f"[experiment3-chunk] chunk={args.chunk_index} mechanism={scenario['mechanism']} "
            f"frac={scenario['frac']} repeat={scenario['repeat']} mode=known_mask method={method}",
            flush=True,
        )
        try:
            da_imp = impute_method(
                masked_da,
                replace_mask,
                observed_mask,
                method,
                {},
                pca_params,
                zero_value,
                int(scenario["seed"]) + 300000,
            )
            df_traj, df_subj = trajectory_metrics_long(
                da_log,
                masked_da,
                da_imp,
                artificial_mask,
                method,
                str(scenario["mechanism"]),
                float(scenario["frac"]),
                int(scenario["repeat"]),
            )
            df_traj["replacement_mode"] = "known_mask"
            df_subj["replacement_mode"] = "known_mask"
            append_csv(df_subj, subj_path)
            cluster_rows.append(
                {
                    **clustering_metrics(
                        da_log,
                        masked_da,
                        da_imp,
                        method,
                        str(scenario["mechanism"]),
                        float(scenario["frac"]),
                        int(scenario["repeat"]),
                    ),
                    "replacement_mode": "known_mask",
                }
            )
            if affected.size:
                dtw_rows = []
                for i, j in affected:
                    dtw_rows.append(
                        {
                            "mechanism": scenario["mechanism"],
                            "frac": float(scenario["frac"]),
                            "repeat": int(scenario["repeat"]),
                            "replacement_mode": "known_mask",
                            "method": method,
                            "participant": str(da_log.coords["participant"].values[i]),
                            "taxon": str(da_log.coords["taxon"].values[j]),
                            "dtw_imputed": simple_dtw(da_log.values[i, j], da_imp.values[i, j]),
                            "dtw_masked": simple_dtw(da_log.values[i, j], masked_da.values[i, j]),
                        }
                    )
                df_traj = pd.concat([df_traj, pd.DataFrame(dtw_rows)], ignore_index=True, sort=False)
            append_csv(df_traj, traj_path)
        except Exception as exc:
            failures.append(
                {
                    "mechanism": scenario["mechanism"],
                    "frac": scenario["frac"],
                    "repeat": scenario["repeat"],
                    "method": method,
                    "replacement_mode": "known_mask",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    pd.DataFrame(cluster_rows, columns=CLUSTER_COLUMNS).to_csv(cluster_path, index=False)
    pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(failure_path, index=False)
    status = "ok" if not failures else "partial_failure"
    pd.DataFrame([manifest | {"status": status, "failure_count": len(failures)}]).to_csv(manifest_path, index=False)
    print(f"[experiment3-chunk] wrote chunk {args.chunk_index} status={status}", flush=True)


if __name__ == "__main__":
    main()
