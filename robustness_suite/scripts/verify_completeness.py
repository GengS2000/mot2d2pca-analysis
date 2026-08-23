from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def output_root():
    path = Path(os.environ.get("ROBUSTNESS_OUTPUT_ROOT", "outputs")).expanduser()
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage_a"], required=True)
    args = parser.parse_args()
    manifest = ROOT / "manifests" / f"{args.stage}.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        ids = [int(row["task_id"]) for row in csv.DictReader(handle)]
    missing = []
    for task_id in ids:
        output = output_root() / args.stage / "tasks" / f"task_{task_id:08d}.json"
        if not output.exists():
            missing.append(task_id)
            continue
        if json.loads(output.read_text(encoding="utf-8")).get("status") != "ok":
            missing.append(task_id)
    target = output_root() / args.stage / "failed_or_missing_task_ids.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(map(str, missing)) + ("\n" if missing else ""), encoding="utf-8")
    print(f"{len(missing)} missing/failed task(s): {target}")


if __name__ == "__main__":
    main()
