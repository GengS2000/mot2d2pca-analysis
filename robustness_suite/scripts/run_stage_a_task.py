from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustness.common import atomic_json, load_yaml, nrmse, numeric_time, output_root, pearson
from robustness.model_adapter import fit_transform


def main():
    parser = argparse.ArgumentParser(description="Run one Stage A manifest task.")
    parser.add_argument("--manifest", default=ROOT / "manifests" / "stage_a.csv", type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next((r for r in rows if int(r["task_id"]) == args.task_id), None)
    if row is None:
        raise KeyError(f"task_id={args.task_id} not found")
    task_path = output_root() / "stage_a" / "tasks" / f"task_{args.task_id:08d}.json"
    if task_path.exists():
        previous = json.loads(task_path.read_text(encoding="utf-8"))
        if previous.get("status") == "ok":
            return
        task_path.unlink()
    try:
        cfg = load_yaml(ROOT / "configs" / "stage_a_tuning.yaml")
        da = xr.open_dataarray(ROOT / "cache" / "prepared_data" / f"{row['dataset']}.nc").load()
        masks = np.load(ROOT / "cache" / "masks" / f"stage_a_{row['dataset']}.npz")
        test = masks["test_mask"][int(row["outer_repeat"])]
        validation = masks["validation_mask"][int(row["outer_repeat"]), int(row["fold"])]
        observed = np.isfinite(da.values) & ~test & ~validation
        data = da.transpose("bin", "participant", "taxon").values.astype(float)
        observed_t = observed.transpose(2, 0, 1)
        validation_t = validation.transpose(2, 0, 1)
        fitted = fit_transform(data, observed_t, numeric_time(da.coords["bin"].values), int(row["d"]), int(row["q"]), float(row["lam"]), float(row["sigma"]), int(row["algo_seed"]))
        truth = data[validation_t]
        estimate = fitted["imputed"][validation_t]
        baseline = np.zeros_like(truth)
        payload = {**row, "status": "ok", "n_validation": int(validation_t.sum()), "nrmse": nrmse(truth, estimate), "pearson": pearson(truth, estimate), "zero_fill_nrmse": nrmse(truth, baseline), "runtime_seconds": fitted["runtime_seconds"]}
    except Exception as exc:
        payload = {**row, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
    atomic_json(task_path, payload)
    if payload["status"] != "ok":
        raise RuntimeError(payload["error"])


if __name__ == "__main__":
    main()
