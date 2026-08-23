#!/usr/bin/env python3
"""Participant-block bootstrap and P2D2PCA imputation for ECAM/DIABIMMUNE.

One logical task is one (cohort, bootstrap replicate). Participants are sampled
with replacement and all taxon x time-bin values for a sampled participant move
together. Duplicate draws receive unique IDs so downstream longitudinal random
effects do not collapse them back onto the original participant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr


COHORTS = ("ECAM", "DIABIMMUNE")


def cohort_config(project_dir: Path, cohort: str) -> Dict[str, Path]:
    if cohort == "ECAM":
        cohort_dir = project_dir / "ECAM_data"
        tensor_dir = cohort_dir / "ECAM_otu_tensor_export"
        return {
            "cohort_dir": cohort_dir,
            "tensor_dir": tensor_dir,
            "base_nc": tensor_dir / "tensor_base_raw.nc",
            "mask_nc": tensor_dir / "replace_mask.nc",
            "config": tensor_dir / "config.json",
            "groups": cohort_dir / "artifacts/ECAM_otu_linda_longitudinal/subject_groups_long.csv",
            "participant_order": cohort_dir / "artifacts/ECAM_otu_linda_longitudinal/participant_order.csv",
            "taxon_index": tensor_dir / "tensor_taxon_index.csv",
        }
    if cohort == "DIABIMMUNE":
        cohort_dir = project_dir / "DIABIMMUNE_data"
        export_dir = cohort_dir / "diabimmune_p2d2pca_netcdf_export"
        tensor_dir = export_dir / "tensors"
        return {
            "cohort_dir": cohort_dir,
            "tensor_dir": tensor_dir,
            "base_nc": tensor_dir / "tensor_base_raw.nc",
            "mask_nc": tensor_dir / "replace_mask.nc",
            "config": export_dir / "config.json",
            "groups": cohort_dir / "artifacts/subject_group_existing_methods_r/subject_groups_long.csv",
            "participant_order": cohort_dir / "participant_order.csv",
            "taxon_index": tensor_dir / "tensor_taxon_index.csv",
        }
    raise ValueError(f"Unsupported cohort: {cohort}")


def load_first_dataarray(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    try:
        if not ds.data_vars:
            raise ValueError(f"No data variable in {path}")
        return ds[next(iter(ds.data_vars))].load()
    finally:
        ds.close()


def canonicalize(da: xr.DataArray, label: str) -> xr.DataArray:
    expected = {"participant", "taxon", "bin"}
    if set(da.dims) != expected:
        raise ValueError(f"{label} dimensions are {da.dims}; expected {sorted(expected)}")
    return da.transpose("participant", "taxon", "bin")


def save_dataarray(da: xr.DataArray, var_name: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({var_name: da})
    try:
        ds.to_netcdf(path, encoding={var_name: {"zlib": True, "complevel": 4}})
    except Exception:
        if path.exists():
            path.unlink()
        ds.to_netcdf(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_task(args: argparse.Namespace) -> Tuple[str, int]:
    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(COHORTS) * args.n_bootstrap:
            raise ValueError("task-id is outside the configured array range")
        cohort = COHORTS[args.task_id // args.n_bootstrap]
        replicate = args.task_id % args.n_bootstrap + 1
        return cohort, replicate
    if args.cohort is None or args.replicate is None:
        raise ValueError("Provide --task-id or both --cohort and --replicate")
    cohort = args.cohort.upper()
    if cohort not in COHORTS:
        raise ValueError(f"Unknown cohort: {cohort}")
    if not 1 <= args.replicate <= args.n_bootstrap:
        raise ValueError("replicate must be between 1 and n-bootstrap")
    return cohort, args.replicate


def import_imputer(project_dir: Path):
    ecam_dir = project_dir / "ECAM_data"
    sys.path.insert(0, str(ecam_dir))
    import sim3_compare  # pylint: disable=import-outside-toplevel

    return sim3_compare


def validate_inputs(config: Dict[str, Path]) -> None:
    required = ("base_nc", "mask_nc", "config", "groups", "participant_order")
    missing = [str(config[key]) for key in required if not config[key].is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))


def make_bootstrap_metadata(
    original_ids: np.ndarray,
    sampled_indices: np.ndarray,
    groups_path: Path,
    cohort: str,
    replicate: int,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    source_ids = original_ids[sampled_indices].astype(str)
    bootstrap_ids = np.asarray(
        [f"B{replicate:03d}D{draw + 1:03d}__{source}" for draw, source in enumerate(source_ids)],
        dtype=str,
    )
    order = pd.DataFrame(
        {
            "participant_idx": np.arange(1, len(source_ids) + 1, dtype=int),
            "tensor_participant_index": np.arange(len(source_ids), dtype=int),
            "participant_id": bootstrap_ids,
            "source_participant_id": source_ids,
            "bootstrap_replicate": replicate,
            "bootstrap_draw": np.arange(1, len(source_ids) + 1, dtype=int),
            "cohort": cohort,
        }
    )

    groups = pd.read_csv(groups_path, dtype={"participant_id": str})
    if "participant_id" not in groups:
        raise ValueError(f"participant_id is absent from {groups_path}")
    pieces = []
    for draw, (source, bootstrap_id) in enumerate(zip(source_ids, bootstrap_ids), start=1):
        block = groups.loc[groups["participant_id"].astype(str) == source].copy()
        if block.empty:
            raise ValueError(f"Participant {source!r} is absent from {groups_path}")
        block["source_participant_id"] = source
        block["participant_id"] = bootstrap_id
        block["bootstrap_replicate"] = replicate
        block["bootstrap_draw"] = draw
        pieces.append(block)
    return bootstrap_ids, order, pd.concat(pieces, ignore_index=True)


def run(args: argparse.Namespace) -> Path:
    project_dir = args.project_dir.resolve()
    output_root = args.output_root.resolve()
    cohort, replicate = resolve_task(args)
    cfg = cohort_config(project_dir, cohort)
    validate_inputs(cfg)

    final_dir = output_root / "stage1" / cohort / f"bootstrap_{replicate:03d}"
    success_path = final_dir / "SUCCESS.json"
    if args.resume and success_path.is_file():
        success = json.loads(success_path.read_text(encoding="utf-8"))
        expected = (final_dir / "tensor_base_raw.nc", final_dir / "tensor_p2d2pca_imputed.nc")
        if success.get("status") == "completed" and all(path.is_file() for path in expected):
            print(f"Reusing completed task: {cohort} bootstrap={replicate}")
            return final_dir
        raise RuntimeError(f"Existing SUCCESS marker is inconsistent: {success_path}")

    tmp_dir = final_dir.with_name(final_dir.name + f".tmp.{os.getpid()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    started = time.time()

    try:
        base = canonicalize(load_first_dataarray(cfg["base_nc"]), "base tensor")
        mask = canonicalize(load_first_dataarray(cfg["mask_nc"]), "replace mask").astype(bool)
        if base.shape != mask.shape:
            raise ValueError(f"base/mask shapes differ: {base.shape} vs {mask.shape}")

        original_ids = base.coords["participant"].values.astype(str)
        n_participants = len(original_ids)
        seed = int(args.seed + (1_000_000 if cohort == "ECAM" else 2_000_000) + replicate)
        rng = np.random.default_rng(seed)
        sampled_indices = rng.choice(n_participants, size=n_participants, replace=True)
        bootstrap_ids, order, groups = make_bootstrap_metadata(
            original_ids, sampled_indices, cfg["groups"], cohort, replicate
        )

        indexer = xr.DataArray(sampled_indices, dims="participant")
        base_boot = base.isel(participant=indexer).assign_coords(participant=bootstrap_ids)
        mask_boot = mask.isel(participant=indexer).assign_coords(participant=bootstrap_ids)
        base_boot.attrs.update(base.attrs)
        base_boot.attrs.update(
            bootstrap_replicate=int(replicate), bootstrap_seed=seed,
            bootstrap_sampling="participant blocks with replacement",
        )

        algorithm_config = json.loads(cfg["config"].read_text(encoding="utf-8"))
        pca_params = {
            "d": int(algorithm_config.get("d", 1)),
            "q": int(algorithm_config.get("q", 1)),
            "lam": float(algorithm_config.get("lam", 0.1)),
            "sigma": float(algorithm_config.get("sigma", 4.0)),
            "T0": int(args.t0 if args.t0 is not None else algorithm_config.get("T0", 30)),
            "T": int(args.t if args.t is not None else algorithm_config.get("T", 30)),
        }
        eps = float(algorithm_config.get("eps", 1e-3))
        zero_value = float(np.log10(eps))
        sim3 = import_imputer(project_dir)
        imputed, _ = sim3.impute_by_mask(
            da_input=base_boot,
            replace_mask=mask_boot,
            model_seed=seed,
            method="penalized_2d2_pca",
            zero_value=zero_value,
            method_params=None,
            pca_params=pca_params,
            observed_mode="positive_only_except_replace",
            verbose=bool(args.verbose),
        )
        imputed = canonicalize(imputed, "imputed tensor")
        imputed.attrs.update(base_boot.attrs)
        imputed.attrs.update(imputation_method="penalized_2d2_pca", p2d2pca_parameters=json.dumps(pca_params))

        base_values = np.asarray(base_boot.values, dtype=float)
        imputed_values = np.asarray(imputed.values, dtype=float)
        mask_values = np.asarray(mask_boot.values, dtype=bool)
        untouched = ~mask_values
        unchanged_max = float(np.nanmax(np.abs(imputed_values[untouched] - base_values[untouched])))
        if not np.isfinite(unchanged_max) or unchanged_max > args.unchanged_tolerance:
            raise RuntimeError(f"Observed entries changed (max abs difference={unchanged_max})")
        completed = imputed_values[mask_values]
        delta = completed - base_values[mask_values]
        finite_fraction = float(np.mean(np.isfinite(completed))) if completed.size else 1.0
        if finite_fraction < 1.0:
            raise RuntimeError(f"Non-finite completed entries detected: finite_fraction={finite_fraction}")

        save_dataarray(base_boot, str(base.name or "abundance_binned_log"), tmp_dir / "tensor_base_raw.nc")
        save_dataarray(imputed, "abundance_binned_log_p2d2pca_imputed", tmp_dir / "tensor_p2d2pca_imputed.nc")
        save_dataarray(mask_boot.astype("int8"), "replace_mask", tmp_dir / "replace_mask.nc")
        order.to_csv(tmp_dir / "participant_order.csv", index=False)
        groups.to_csv(tmp_dir / "subject_groups_long.csv", index=False)
        draw_manifest = order[["cohort", "bootstrap_replicate", "bootstrap_draw", "participant_id", "source_participant_id"]]
        draw_manifest.to_csv(tmp_dir / "bootstrap_draws.csv", index=False)
        if cfg["taxon_index"].is_file():
            shutil.copy2(cfg["taxon_index"], tmp_dir / "tensor_taxon_index.csv")

        taxon_axis = base_boot.get_axis_num("taxon")
        reduction_axes = tuple(axis for axis in range(base_boot.ndim) if axis != taxon_axis)
        replaced_count_taxon = mask_values.sum(axis=reduction_axes)
        delta_sum_taxon = np.where(mask_values, imputed_values - base_values, 0.0).sum(axis=reduction_axes)
        taxon_summary = pd.DataFrame(
            {
                "cohort": cohort,
                "bootstrap_replicate": replicate,
                "taxon": base_boot.coords["taxon"].values.astype(str),
                "n_replaced": replaced_count_taxon.astype(int),
                "mean_imputation_delta": np.divide(
                    delta_sum_taxon,
                    replaced_count_taxon,
                    out=np.full(replaced_count_taxon.shape, np.nan, dtype=float),
                    where=replaced_count_taxon > 0,
                ),
                "mean_completed_log_abundance": np.nanmean(imputed_values, axis=reduction_axes),
            }
        )
        taxon_summary.to_csv(tmp_dir / "taxon_imputation_summary.csv.gz", index=False, compression="gzip")

        metrics: Dict[str, Any] = {
            "status": "completed",
            "cohort": cohort,
            "bootstrap_replicate": replicate,
            "bootstrap_seed": seed,
            "n_participants": n_participants,
            "n_unique_source_participants": int(len(np.unique(original_ids[sampled_indices]))),
            "duplicate_draw_fraction": float(1 - len(np.unique(original_ids[sampled_indices])) / n_participants),
            "n_taxa": int(base_boot.sizes["taxon"]),
            "n_bins": int(base_boot.sizes["bin"]),
            "replace_entries": int(mask_values.sum()),
            "replace_fraction": float(mask_values.mean()),
            "completed_finite_fraction": finite_fraction,
            "unchanged_max_abs_difference": unchanged_max,
            "mean_imputation_delta": float(np.mean(delta)) if delta.size else 0.0,
            "median_abs_imputation_delta": float(np.median(np.abs(delta))) if delta.size else 0.0,
            "rmse_from_placeholder": float(np.sqrt(np.mean(delta ** 2))) if delta.size else 0.0,
            "pca_params": pca_params,
            "source_base_sha256": file_sha256(cfg["base_nc"]),
            "elapsed_seconds": float(time.time() - started),
        }
        (tmp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (tmp_dir / "SUCCESS.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        tmp_dir.rename(final_dir)
        print(json.dumps(metrics, indent=2))
        return final_dir
    except Exception:
        failure = tmp_dir / "FAILURE.txt"
        failure.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--cohort", choices=["ECAM", "DIABIMMUNE", "ecam", "diabimmune"])
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--n-bootstrap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unchanged-tolerance", type=float, default=1e-10)
    parser.add_argument("--t0", type=int, help="Override Phase-1 iterations (smoke testing only).")
    parser.add_argument("--t", type=int, help="Override Phase-2 iterations (smoke testing only).")
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(0 if run(build_parser().parse_args()) else 1)
