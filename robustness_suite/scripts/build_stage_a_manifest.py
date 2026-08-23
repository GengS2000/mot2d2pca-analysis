from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustness.common import load_yaml


def main():
    cfg = load_yaml(ROOT / "configs" / "stage_a_tuning.yaml")
    fixed_d, fixed_q = cfg["fixed_rank_for_penalty"]
    fixed_lam, fixed_sigma = cfg["fixed_penalty_for_rank"]
    params = [(d, q, fixed_lam, fixed_sigma, "rank") for d, q in itertools.product(cfg["rank_grid"], repeat=2)]
    params += [(fixed_d, fixed_q, lam, sigma, "penalty") for lam, sigma in itertools.product(cfg["lambda_grid"], cfg["sigma_grid"])]
    output = ROOT / "manifests" / "stage_a.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task_id", "dataset", "outer_repeat", "fold", "d", "q", "lam", "sigma", "grid", "mask_seed", "algo_seed"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        task_id = 0
        for repeat, fold, (d, q, lam, sigma, grid) in itertools.product(range(cfg["outer_repeats"]), range(cfg["inner_folds"]), params):
            writer.writerow(dict(task_id=task_id, dataset=cfg["dataset"], outer_repeat=repeat, fold=fold, d=d, q=q, lam=lam, sigma=sigma, grid=grid, mask_seed=int(cfg["seed"])+repeat, algo_seed=int(cfg["seed"])+100000+task_id))
            task_id += 1
    print(f"{output} ({task_id} tasks)")


if __name__ == "__main__":
    main()
