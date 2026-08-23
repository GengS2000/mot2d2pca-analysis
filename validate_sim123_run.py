#!/usr/bin/env python3
"""Validate a complete Sim1/Sim2/Sim3 shard set before merging."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


FRACS = (0.5, 0.6, 0.7, 0.8, 0.9)
REPEATS = range(50)


def sim23_frac(frac: float) -> str:
    return f"{frac:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def require_csv(path: Path, errors: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        errors.append(f"missing/empty CSV: {path}")
        return
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            row = next(reader, None)
        if not header or row is None:
            errors.append(f"CSV has no header or data row: {path}")
    except Exception as exc:  # report all broken shards in one pass
        errors.append(f"unreadable CSV: {path}: {exc}")


def require_sim3_marker(path: Path, frac: float, repeat: int, errors: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        errors.append(f"missing/empty completion marker: {path}")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not math.isclose(float(payload.get("frac")), frac):
            errors.append(f"wrong frac in marker: {path}")
        if int(payload.get("repeat")) != repeat:
            errors.append(f"wrong repeat in marker: {path}")
    except Exception as exc:
        errors.append(f"invalid completion marker: {path}: {exc}")


def validate_manifest(root: Path, errors: list[str]) -> None:
    path = root / "manifest.json"
    if not path.is_file():
        errors.append(f"missing manifest: {path}")
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        params = manifest["impute_params"]
    except Exception as exc:
        errors.append(f"invalid manifest: {path}: {exc}")
        return

    expected = {
        "d": os.environ.get("IMPUTE_D", "1"),
        "q": os.environ.get("IMPUTE_Q", "1"),
        "lam": os.environ.get("IMPUTE_LAM", "2"),
        "sigma": os.environ.get("IMPUTE_SIGMA", "16"),
    }
    for key, value in expected.items():
        try:
            matches = math.isclose(float(params[key]), float(value))
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            errors.append(
                f"manifest impute_params.{key}={params.get(key)!r}; expected {value!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root
    errors: list[str] = []

    validate_manifest(root, errors)
    for frac in FRACS:
        sim1_label = f"{frac:g}"
        sim23_label = sim23_frac(frac)
        for repeat in REPEATS:
            require_csv(
                root / "sim1" / f"replicate_metrics_frac_{sim1_label}_repeat_{repeat}.csv",
                errors,
            )
            require_csv(
                root / "sim2" / f"repeat_summary_frac_{sim23_label}_repeat_{repeat:03d}.csv",
                errors,
            )
            require_sim3_marker(
                root / "sim3" / f"repeat_complete_frac_{sim23_label}_repeat_{repeat:03d}.json",
                frac,
                repeat,
                errors,
            )

    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:30])
        suffix = "" if len(errors) <= 30 else f"\n  ... and {len(errors) - 30} more"
        raise SystemExit(f"pre-merge validation failed ({len(errors)} issue(s)):\n{preview}{suffix}")
    print(f"[validate] manifest parameters and all 750 shard sentinels are valid: {root}")


if __name__ == "__main__":
    main()
