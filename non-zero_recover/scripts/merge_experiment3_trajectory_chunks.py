from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_COLS = ["mechanism", "frac", "repeat", "replacement_mode", "method"]
TRAJECTORY_METRICS = [
    "trajectory_pearson",
    "trajectory_spearman",
    "trajectory_rmse",
    "lag1_error_imputed",
    "lag1_error_masked",
    "second_difference_mae_imputed",
    "second_difference_mae_masked",
    "dtw_imputed",
    "dtw_masked",
]


def list_parts(chunks_dir: Path, prefix: str) -> list[Path]:
    return sorted(chunks_dir.glob(f"{prefix}_part_*.csv"))


def concat_csv_stream(parts: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    wrote_header = False
    with output.open("w", encoding="utf-8", newline="") as out:
        for part in parts:
            if not part.exists() or part.stat().st_size == 0:
                continue
            with part.open("r", encoding="utf-8", newline="") as inp:
                header = inp.readline()
                if not header:
                    continue
                if not wrote_header:
                    out.write(header)
                    wrote_header = True
                for line in inp:
                    out.write(line)
                    row_count += 1
    if not wrote_header:
        output.write_text("", encoding="utf-8")
    return row_count


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def repeat_level_from_part(path: Path) -> pd.DataFrame:
    df = read_csv_or_empty(path)
    if df.empty:
        return pd.DataFrame()
    available = [m for m in TRAJECTORY_METRICS if m in df.columns]
    if not available or any(c not in df.columns for c in GROUP_COLS):
        return pd.DataFrame()
    for metric in available:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    grouped = df.groupby(GROUP_COLS, dropna=False)
    mean_df = grouped[available].mean().reset_index()
    count_df = grouped[available].count().reset_index().rename(columns={m: f"{m}_n" for m in available})
    return mean_df.merge(count_df, on=GROUP_COLS, how="left")


def write_repeat_level(parts: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    wrote = False
    for part in parts:
        rep = repeat_level_from_part(part)
        if rep.empty:
            continue
        rep.to_csv(output, mode="a", header=not wrote, index=False)
        wrote = True
        row_count += len(rep)
    if not wrote:
        output.write_text("", encoding="utf-8")
    return row_count


def summarize_repeat_level(repeat_level_path: Path, output: Path) -> None:
    rep = read_csv_or_empty(repeat_level_path)
    if rep.empty:
        output.write_text("", encoding="utf-8")
        return
    metric_cols = [m for m in TRAJECTORY_METRICS if m in rep.columns]
    rows = []
    for keys, sub in rep.groupby(["mechanism", "frac", "replacement_mode", "method"], dropna=False):
        row = {
            "mechanism": keys[0],
            "frac": keys[1],
            "replacement_mode": keys[2],
            "method": keys[3],
        }
        for metric in metric_cols:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{metric}_count"] = int(len(vals))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output, index=False)


def concat_failures(parts: list[Path], output: Path) -> int:
    frames = []
    for part in parts:
        df = read_csv_or_empty(part)
        if not df.empty:
            frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["mechanism", "frac", "repeat", "method", "replacement_mode", "error", "traceback"])
    out.to_csv(output, index=False)
    return len(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Experiment 3 trajectory chunk CSVs without loading all long rows into memory.")
    parser.add_argument("--expdir", type=Path, default=Path("outputs_known_mask/experiment3_trajectory"))
    parser.add_argument("--write-combined-long", action="store_true", help="Stream-concatenate trajectory_metrics_long.csv. This can be very large.")
    parser.add_argument("--clean-existing", action="store_true", help="Remove previous merged outputs before writing.")
    args = parser.parse_args()

    expdir = args.expdir
    chunks_dir = expdir / "chunks"
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunks_dir}")

    outputs = [
        expdir / "subject_level_metrics.csv",
        expdir / "taxon_clustering_metrics.csv",
        expdir / "failed_runs.csv",
        expdir / "chunk_manifest.csv",
        expdir / "trajectory_metrics_repeat_level.csv",
        expdir / "trajectory_metrics_summary.csv",
    ]
    if args.write_combined_long:
        outputs.append(expdir / "trajectory_metrics_long.csv")
    if args.clean_existing:
        for path in outputs:
            if path.exists():
                path.unlink()

    traj_parts = list_parts(chunks_dir, "trajectory_metrics_long")
    subj_parts = list_parts(chunks_dir, "subject_level_metrics")
    cluster_parts = list_parts(chunks_dir, "taxon_clustering_metrics")
    failure_parts = list_parts(chunks_dir, "failed_runs")
    manifest_parts = list_parts(chunks_dir, "chunk_manifest")

    subject_rows = concat_csv_stream(subj_parts, expdir / "subject_level_metrics.csv")
    cluster_rows = concat_csv_stream(cluster_parts, expdir / "taxon_clustering_metrics.csv")
    manifest_rows = concat_csv_stream(manifest_parts, expdir / "chunk_manifest.csv")
    failure_rows = concat_failures(failure_parts, expdir / "failed_runs.csv")
    repeat_rows = write_repeat_level(traj_parts, expdir / "trajectory_metrics_repeat_level.csv")
    summarize_repeat_level(expdir / "trajectory_metrics_repeat_level.csv", expdir / "trajectory_metrics_summary.csv")

    long_rows = ""
    if args.write_combined_long:
        long_rows = concat_csv_stream(traj_parts, expdir / "trajectory_metrics_long.csv")

    summary_path = expdir / "merge_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["item", "count"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"item": "trajectory_part_files", "count": len(traj_parts)},
                {"item": "subject_rows", "count": subject_rows},
                {"item": "cluster_rows", "count": cluster_rows},
                {"item": "manifest_rows", "count": manifest_rows},
                {"item": "failure_rows", "count": failure_rows},
                {"item": "trajectory_repeat_level_rows", "count": repeat_rows},
                {"item": "trajectory_long_rows", "count": long_rows},
            ]
        )
    print(f"[experiment3-merge] parts={len(traj_parts)} subject_rows={subject_rows} repeat_rows={repeat_rows}")
    print(f"[experiment3-merge] wrote {summary_path}")


if __name__ == "__main__":
    main()
