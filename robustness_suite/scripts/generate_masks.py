from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustness.common import load_yaml


def draw_within_slices(eligible, fraction, rng, minimum):
    chosen = np.zeros_like(eligible, dtype=bool)
    for time_index in range(eligible.shape[2]):
        for row_index in range(eligible.shape[0]):
            indices = np.flatnonzero(eligible[row_index, :, time_index])
            n = min(int(round(fraction * indices.size)), max(indices.size - minimum, 0))
            if n:
                chosen[row_index, rng.choice(indices, size=n, replace=False), time_index] = True
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Generate nested, leak-free masks for Stage A.")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    cfg = load_yaml(ROOT / "configs" / "stage_a_tuning.yaml")
    da = xr.open_dataarray(ROOT / "cache" / "prepared_data" / f"{args.dataset}.nc").load()
    eligible = np.isfinite(da.values)
    n_repeat, n_fold = int(cfg["outer_repeats"]), int(cfg["inner_folds"])
    test = np.zeros((n_repeat,) + eligible.shape, bool)
    validation = np.zeros((n_repeat, n_fold) + eligible.shape, bool)
    for repeat in range(n_repeat):
        rng = np.random.default_rng(int(cfg["seed"]) + repeat)
        test[repeat] = draw_within_slices(eligible, cfg["outer_test_fraction"], rng, cfg["minimum_observed_per_slice"])
        remaining = eligible & ~test[repeat]
        for fold in range(n_fold):
            validation[repeat, fold] = draw_within_slices(remaining, cfg["inner_validation_fraction"], rng, cfg["minimum_observed_per_slice"])
    target = ROOT / "cache" / "masks" / f"stage_a_{args.dataset}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, test_mask=test, validation_mask=validation)
    print(target)


if __name__ == "__main__":
    main()
