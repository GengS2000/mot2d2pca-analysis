"""Run one isolated competitor benchmark task.

One invocation owns exactly one (simulation, method, fraction, repeat) tuple.
This is the unit submitted by the SLURM arrays, so a failed method/repeat does
not cancel or corrupt any other result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parent
MODULE_PATHS = {
    "sim1": ROOT / "sim1_compare" / "sim1_methodcompare.py",
    "sim2": ROOT / "sim2-compare" / "ctf_reconstruct_compare.py",
    "sim3": ROOT / "sim3_compare" / "sim3_compare.py",
}
SOURCE_COMMITS = {
    "ctf": "gemelli@c53c9ee958948683bc4216cf5fa447e2c3ac4806",
}


def _load_module(simulation: str):
    path = MODULE_PATHS[simulation]
    spec = importlib.util.spec_from_file_location(f"benchmark_{simulation}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import benchmark module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_truth(path: Path) -> xr.DataArray:
    if not path.exists():
        raise FileNotFoundError(f"Truth tensor does not exist: {path}")
    if path.suffix.lower() == ".zarr" or path.is_dir():
        opened = xr.open_zarr(path)
    else:
        try:
            opened = xr.open_dataarray(path)
        except (ValueError, KeyError):
            opened = xr.open_dataset(path)
    if isinstance(opened, xr.Dataset):
        for name in ("y_true", "log_abundance", "log_abundance_truth"):
            if name in opened.data_vars:
                da = opened[name].load()
                opened.close()
                break
        else:
            if len(opened.data_vars) != 1:
                opened.close()
                raise ValueError(f"Cannot choose truth variable in {path}")
            da = next(iter(opened.data_vars.values())).load()
            opened.close()
    else:
        da = opened.load()
        opened.close()
    required = {"participant", "taxon", "bin"}
    if set(da.dims) != required:
        raise ValueError(f"Truth tensor dims must be {sorted(required)}, got {da.dims}")
    return da.transpose("participant", "taxon", "bin")


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _method_params(args: argparse.Namespace) -> dict:
    return {
        "rank": int(args.ctf_rank),
        "n_initializations": int(args.ctf_initializations),
        "max_iterations_rptm": int(args.ctf_max_iterations),
    }

def _annotate_primary_outputs(
    outdir: Path,
    frac_str: str,
    repeat_str: str,
    method: str,
    params: dict,
) -> None:
    variants = {
        "ctf": "gemelli_tensor_rclr_tenals_cellwise_mask",
    }
    variant = variants[method]
    candidates = [
        outdir / f"repeat_metrics_frac_{frac_str}_repeat_{repeat_str}.csv",
        outdir / f"repeat_summary_frac_{frac_str}_repeat_{repeat_str}.csv",
        outdir / f"abundance_eval_summary_frac_{frac_str}_repeat_{int(repeat_str)}.csv",
        outdir / f"abundance_eval_summary_frac_{frac_str}_repeat_{repeat_str}.csv",
    ]
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        table["adapter_variant"] = variant
        table["external_source_commit"] = SOURCE_COMMITS[method]
        table["competitor_params_json"] = json.dumps(params, ensure_ascii=False, sort_keys=True)
        tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        table.to_csv(tmp, index=False)
        tmp.replace(path)


def _run(args: argparse.Namespace) -> dict:
    module = _load_module(args.simulation)
    truth = _load_truth(args.truth_path)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    frac_str = f"{float(args.frac):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    repeat_str = f"{int(args.repeat):03d}"
    params = _method_params(args)

    if args.simulation == "sim1":
        frac_seed_base = int(args.sim1_seed_base) + int(round(float(args.frac) * 1_000_000))
        mask_seed = frac_seed_base + int(args.repeat)
        algo_seed = frac_seed_base + 10000 + int(args.repeat)
        per_bin, summary = module.run_one_replication(
            da_truth=truth,
            simulation="sim1",
            frac=float(args.frac),
            replicate=int(args.repeat),
            mask_seed=mask_seed,
            model_seed=algo_seed,
            zero_value=float(np.log10(args.eps)),
            pca_params={"d": 10, "q": 10, "lam": 0.7, "sigma": 8.0, "T0": 30, "T": 30},
            methods=[args.method],
            impute_method_params={args.method: params},
            second_observed_mode="positive_only_except_replace",
            mask_eligible_mode=args.sim1_mask_eligible_mode,
            outdir=outdir,
            eps=float(args.eps),
            verbose=bool(args.verbose),
        )
        per_bin.to_csv(outdir / f"repeat_metrics_frac_{frac_str}_repeat_{repeat_str}.csv", index=False)
        summary.to_csv(outdir / f"repeat_summary_frac_{frac_str}_repeat_{repeat_str}.csv", index=False)
    else:
        mask_seed = int(args.mask_seed_base) + int(args.repeat)
        algo_seed = int(args.algo_seed)
        module.run_one_repeat(
            da_gt=truth,
            frac=float(args.frac),
            repeat=int(args.repeat),
            mask_seed=mask_seed,
            algo_seed=algo_seed,
            outdir=outdir,
            d=10,
            q=10,
            lam=0.7,
            sigma=8.0,
            T0=30,
            T=30,
            save_imputed_matrix=bool(args.save_imputed_matrix),
            save_format="netcdf",
            compress_level=4,
            verbose=bool(args.verbose),
            eps=float(args.eps),
            methods=[args.method],
            impute_method_params={args.method: params},
        )

    _annotate_primary_outputs(outdir, frac_str, repeat_str, args.method, params)
    return {
        "status": "complete",
        "simulation": args.simulation,
        "method": args.method,
        "frac": float(args.frac),
        "repeat": int(args.repeat),
        "mask_seed": int(mask_seed),
        "algo_seed": int(algo_seed),
        "truth_path": str(args.truth_path.resolve()),
        "task_id": args.task_id,
        "external_source_commit": SOURCE_COMMITS[args.method],
        "parameters": params,
        "adapter": "gemelli tensor_rclr + tenals",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", choices=sorted(MODULE_PATHS), required=True)
    parser.add_argument("--method", choices=["ctf"], required=True)
    parser.add_argument("--frac", type=float, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--truth-path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--sim1-seed-base", type=int, default=123)
    parser.add_argument(
        "--sim1-mask-eligible-mode",
        choices=["positive_only", "all_finite"],
        default="positive_only",
    )
    parser.add_argument("--mask-seed-base", type=int, default=123)
    parser.add_argument("--algo-seed", type=int, default=0)
    parser.add_argument("--ctf-rank", type=int, default=3)
    parser.add_argument("--ctf-initializations", type=int, default=5)
    parser.add_argument("--ctf-max-iterations", type=int, default=50)
    parser.add_argument("--save-imputed-matrix", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.frac <= 1.0:
        raise ValueError("--frac must be in [0, 1]")
    if args.repeat < 0 or args.eps <= 0:
        raise ValueError("--repeat must be non-negative and --eps must be positive")
    args.task_id = f"{args.simulation}_{args.method}_{args.frac:g}_{args.repeat:03d}"
    args.outdir = args.outdir / args.simulation / args.method
    args.outdir.mkdir(parents=True, exist_ok=True)
    marker = args.outdir / f"competitor_complete_{args.task_id}.json"
    failure = args.outdir / f"competitor_failed_{args.task_id}.json"
    if marker.exists() and not args.force:
        print(f"[skip complete] {marker}")
        return
    marker.unlink(missing_ok=True)
    failure.unlink(missing_ok=True)
    try:
        payload = _run(args)
        _atomic_json(marker, payload)
        print(json.dumps(payload, ensure_ascii=False))
    except Exception as exc:
        payload = {
            "status": "failed",
            "simulation": args.simulation,
            "method": args.method,
            "frac": float(args.frac),
            "repeat": int(args.repeat),
            "task_id": args.task_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(failure, payload)
        raise


if __name__ == "__main__":
    main()
