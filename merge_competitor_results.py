"""Merge isolated competitor task outputs and audit task completeness."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MODULE_PATHS = {
    "sim1": ROOT / "sim1_compare" / "sim1_methodcompare.py",
    "sim2": ROOT / "sim2-compare" / "ctf_reconstruct_compare.py",
    "sim3": ROOT / "sim3_compare" / "sim3_compare.py",
}


def _load_module(simulation: str):
    spec = importlib.util.spec_from_file_location(f"merge_{simulation}", MODULE_PATHS[simulation])
    if spec is None or spec.loader is None:
        raise ImportError(MODULE_PATHS[simulation])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_many(paths) -> pd.DataFrame:
    tables = [pd.read_csv(path) for path in sorted(paths)]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _merge_one_method(simulation: str, method_dir: Path) -> None:
    module = _load_module(simulation)
    if simulation == "sim1":
        repeat_df = _read_many(method_dir.glob("repeat_summary_frac_*_repeat_*.csv"))
        bin_df = _read_many(method_dir.glob("repeat_metrics_frac_*_repeat_*.csv"))
        if not repeat_df.empty:
            repeat_df.drop_duplicates().to_csv(method_dir / "all_replicates_metrics_sim1.csv", index=False)
            module.summarize_replicates(repeat_df.drop_duplicates()).to_csv(
                method_dir / "summary_with_errorbar_sim1.csv", index=False
            )
        if not bin_df.empty:
            bin_df.drop_duplicates().to_csv(method_dir / "per_bin_metrics_sim1.csv", index=False)
        module.merge_abundance_eval_outputs(method_dir)
    else:
        module.merge_repeat_outputs(method_dir)
        module.merge_abundance_eval_outputs(method_dir)


def _collect_aggregate(
    root: Path,
    simulation: str,
    methods: list[str],
    names: list[str],
    destination: Path,
) -> None:
    for name in names:
        paths = [root / simulation / method / name for method in methods]
        df = _read_many(path for path in paths if path.is_file())
        if not df.empty:
            df.drop_duplicates().to_csv(destination / name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/competitor_results"))
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--fracs", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["ctf"],
        default=["ctf"],
    )
    args = parser.parse_args()

    audit_rows = []
    for simulation in MODULE_PATHS:
        for method in args.methods:
            method_dir = args.root / simulation / method
            if method_dir.is_dir():
                _merge_one_method(simulation, method_dir)
            complete = list(method_dir.glob("competitor_complete_*.json")) if method_dir.is_dir() else []
            failed = list(method_dir.glob("competitor_failed_*.json")) if method_dir.is_dir() else []
            audit_rows.append(
                {
                    "simulation": simulation,
                    "method": method,
                    "expected": len(args.fracs) * int(args.n_repeats),
                    "complete": len(complete),
                    "failed": len(failed),
                    "missing": max(0, len(args.fracs) * int(args.n_repeats) - len(complete)),
                }
            )

        destination = args.root / "merged" / simulation
        destination.mkdir(parents=True, exist_ok=True)
        names = (
            ["all_replicates_metrics_sim1.csv", "summary_with_errorbar_sim1.csv", "per_bin_metrics_sim1.csv"]
            if simulation == "sim1"
            else ["repeat_level_metrics.csv", "errorbar_summary.csv", "repeat_bin_metrics_all.csv"]
        )
        names += [
            "taxon_marginal_eval_all.csv",
            "taxon_trajectory_eval_all.csv",
            "bin_distribution_eval_all.csv",
            "abundance_repeat_level_metrics.csv",
            "abundance_errorbar_summary.csv",
        ]
        _collect_aggregate(args.root, simulation, args.methods, names, destination)

    audit = pd.DataFrame(audit_rows)
    audit_path = args.root / "competitor_task_audit.csv"
    args.root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    payload = {"audit": str(audit_path), "all_complete": bool((audit["missing"] == 0).all())}
    print(json.dumps(payload, ensure_ascii=False))
    print(audit.to_string(index=False))
    if not payload["all_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
