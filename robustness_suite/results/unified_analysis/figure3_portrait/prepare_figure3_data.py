#!/usr/bin/env python3
"""Prepare the exact source-data package used by the portrait Figure 3.

The published sensitivity/ablation estimates are copied from the audited
unified-analysis tables.  Time-point masking is represented once (one visit),
so the three nominal fraction strata are pooled at the paired-repeat level.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


SOURCE_TABLES = (
    "stage_b_paired_effects.csv",
    "stage_c_paired_effects.csv",
    "stage_c_equivalence_audit.csv",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def task_path(tasks_dir: Path, task_id: int) -> Path:
    return tasks_dir / f"task_{task_id:08d}.json"


def select_candidate(manifest: pd.DataFrame, *, lam: float, sigma: float) -> pd.DataFrame:
    numeric = manifest.assign(
        d_num=pd.to_numeric(manifest["d"]),
        q_num=pd.to_numeric(manifest["q"]),
        lam_num=pd.to_numeric(manifest["lam"]),
        sigma_num=pd.to_numeric(manifest["sigma"]),
    )
    return numeric[
        (numeric["mechanism"] == "time_point")
        & (numeric["d_num"] == 1)
        & (numeric["q_num"] == 1)
        & (numeric["lam_num"] == lam)
        & (numeric["sigma_num"] == sigma)
    ].copy()


def read_nrmse_rows(rows: pd.DataFrame, tasks_dir: Path, label: str) -> pd.DataFrame:
    records = []
    for row in rows.itertuples(index=False):
        path = task_path(tasks_dir, int(row.task_id))
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "dataset": row.dataset,
                "mechanism": row.mechanism,
                "nominal_fraction": float(row.fraction),
                "repeat": int(row.repeat),
                "configuration": label,
                "nrmse": float(payload["nrmse"]),
                "n_masked": int(payload["n_masked"]),
                "task_id": int(row.task_id),
            }
        )
    return pd.DataFrame(records)


def prepare(suite: Path, output: Path) -> None:
    suite = suite.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = suite / "results" / "unified_analysis" / "tables"
    manifest_path = suite / "manifests" / "stage_b.csv"
    tasks_dir = suite / "results" / "stage_b" / "tasks"

    copied = []
    for name in SOURCE_TABLES:
        src = tables / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = output / name
        shutil.copy2(src, dst)
        copied.append((name, src, dst))

    manifest = pd.read_csv(manifest_path, dtype=str)
    configs = {
        "reference": (2.0, 16.0),
        "lambda_0": (0.0, 16.0),
        "sigma_2": (2.0, 2.0),
    }
    frames = []
    for label, (lam, sigma) in configs.items():
        selected = select_candidate(manifest, lam=lam, sigma=sigma)
        if len(selected) != 90:
            raise ValueError(f"Expected 90 time-point tasks for {label}; found {len(selected)}")
        frames.append(read_nrmse_rows(selected, tasks_dir, label))

    long = pd.concat(frames, ignore_index=True)
    pair = ["dataset", "mechanism", "nominal_fraction", "repeat"]
    wide = long.pivot(index=pair, columns="configuration", values="nrmse").reset_index()
    counts = long[long.configuration == "reference"][pair + ["n_masked", "task_id"]].rename(
        columns={"task_id": "reference_task_id"}
    )
    wide = wide.merge(counts, on=pair, how="left", validate="one_to_one")

    rows = []
    for perturbation, candidate_col in (("lambda = 0", "lambda_0"), ("sigma = 2", "sigma_2")):
        out = wide[pair + ["n_masked", "reference_task_id", "reference", candidate_col]].copy()
        out = out.rename(columns={candidate_col: "candidate_nrmse", "reference": "reference_nrmse"})
        out["perturbation"] = perturbation
        out["paired_delta_nrmse"] = out["candidate_nrmse"] - out["reference_nrmse"]
        rows.append(out)
    paired = pd.concat(rows, ignore_index=True)
    paired.to_csv(output / "stage_b_timepoint_one_visit_paired_nrmse.csv", index=False)

    dictionary = pd.DataFrame(
        [
            ("stage_b_paired_effects.csv", "Audited paired Stage B estimates used in panels E-G and the non-time-point rows of panel I."),
            ("stage_c_paired_effects.csv", "Audited paired Stage C ablation estimates used in panel H."),
            ("stage_c_equivalence_audit.csv", "Implementation-equivalence audit used to collapse duplicate ablation arms in panel H."),
            ("stage_b_timepoint_one_visit_paired_nrmse.csv", "Pair-level NRMSE differences for one-visit time-point masking, pooled over the three nominal labels because all mask the same number of entries."),
        ],
        columns=["file", "description"],
    )
    dictionary.to_csv(output / "Figure3_data_dictionary.csv", index=False)

    manifest_rows = []
    for name, src, dst in copied:
        manifest_rows.append(
            {"delivered_file": name, "source_path": str(src), "sha256": sha256(dst)}
        )
    paired_path = output / "stage_b_timepoint_one_visit_paired_nrmse.csv"
    manifest_rows.append(
        {
            "delivered_file": paired_path.name,
            "source_path": f"Derived from {manifest_path} and {tasks_dir}",
            "sha256": sha256(paired_path),
        }
    )
    pd.DataFrame(manifest_rows).to_csv(output / "Figure3_source_manifest.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", default=None, help="Path to robustness_suite")
    parser.add_argument("--output-dir", required=True, help="Directory receiving the Figure 3 CSV package")
    args = parser.parse_args()
    default_suite = Path(__file__).resolve().parents[1]
    prepare(Path(args.suite_root).resolve() if args.suite_root else default_suite, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
