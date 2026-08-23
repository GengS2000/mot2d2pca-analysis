"""Merge and rank truth-sweep diagnostics without using imputation performance."""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted(Path(args.root).glob("**/truth_diagnostics.csv"))
    if not paths:
        raise RuntimeError(f"No truth_diagnostics.csv files found under {args.root}")
    table = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    table["target_time_variance_fraction"] = 0.30
    table["target_time_variance_abs_error"] = (
        table["log_time_variance_fraction"] - table["target_time_variance_fraction"]
    ).abs()
    table = table.sort_values(
        ["simulation", "target_time_variance_abs_error", "temporal_rank1_share"],
        ascending=[True, True, True],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
