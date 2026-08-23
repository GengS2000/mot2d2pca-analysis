from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_METHODS = "penalized_2d2_pca,no_imputation"
DEFAULT_MECHANISMS = "random_entry,low_depth_sample,time_point,subject_specific_dropout,consecutive_time_window_dropout,low_abundance_biased"
DEFAULT_FRACS = "0.05,0.10,0.20,0.30"

EXPERIMENT_SCRIPTS = {
    "1": "scripts/run_experiment1_zero_recovery.py",
    "2": "scripts/run_experiment2_distribution_preservation.py",
    "3": "scripts/run_experiment3_trajectory_preservation.py",
    "4": "scripts/run_experiment4_dynamic_correlation.py",
}


def parse_experiments(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return ["1", "2", "3", "4"]
    experiments = [x.strip().replace("experiment", "") for x in value.split(",") if x.strip()]
    invalid = [x for x in experiments if x not in EXPERIMENT_SCRIPTS]
    if invalid:
        raise ValueError(f"Unsupported experiment ids: {invalid}. Use any of 1,2,3,4 or all.")
    return experiments


def build_common_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--tax-path",
        args.tax_path,
        "--meta-path",
        args.meta_path,
        "--outdir",
        args.outdir,
        "--eps",
        str(args.eps),
        "--min-pos-cells",
        str(args.min_pos_cells),
        "--min-participants",
        str(args.min_participants),
        "--rank-d",
        str(args.rank_d),
        "--rank-q",
        str(args.rank_q),
        "--lam",
        str(args.lam),
        "--sigma",
        str(args.sigma),
        "--T0",
        str(args.T0),
        "--T",
        str(args.T),
        "--fracs",
        args.fracs,
        "--repeats",
        str(args.repeats),
        "--methods",
        args.methods,
        "--mechanisms",
        args.mechanisms,
        "--replacement-modes",
        "known_mask",
    ]
    if args.seeds:
        common.extend(["--seeds", args.seeds])
    if args.max_participants is not None:
        common.extend(["--max-participants", str(args.max_participants)])
    if args.max_taxa is not None:
        common.extend(["--max-taxa", str(args.max_taxa)])
    if args.overwrite:
        common.append("--overwrite")
    if not args.write_preprocessing_diagnostics:
        common.append("--skip-preprocessing-diagnostics")
    return common


def command_for_experiment(exp_id: str, args: argparse.Namespace) -> list[str]:
    cmd = [args.python, EXPERIMENT_SCRIPTS[exp_id], *build_common_args(args)]
    if exp_id == "2" and args.save_figures:
        cmd.append("--save-figures")
    if exp_id == "4":
        cmd.extend(["--top-variable-taxa", str(args.top_variable_taxa)])
    return cmd


def run_one(exp_id: str, cmd: list[str], logdir: Path) -> dict[str, str | int | float]:
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / f"experiment{exp_id}.log"
    start = time.time()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        log.write("[command] " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    return {
        "experiment": exp_id,
        "returncode": int(proc.returncode),
        "elapsed_seconds": round(elapsed, 3),
        "log_path": str(log_path),
        "command": " ".join(cmd),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run known-mask experiments 1-4 in parallel while preserving all experiment CSV outputs."
    )
    parser.add_argument("--experiments", default="all", help="Comma-separated experiment ids, e.g. 1,2,3,4 or all.")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of experiments to run concurrently.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use on worker commands.")
    parser.add_argument("--tax-path", default="hmp2_mgx_taxonomy.tsv")
    parser.add_argument("--meta-path", default="metagenomics.xlsx")
    parser.add_argument("--outdir", default="outputs_known_mask")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--min-pos-cells", type=int, default=10)
    parser.add_argument("--min-participants", type=int, default=5)
    parser.add_argument("--rank-d", type=int, default=10)
    parser.add_argument("--rank-q", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--T0", type=int, default=30)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--fracs", default=DEFAULT_FRACS)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--mechanisms", default=DEFAULT_MECHANISMS)
    parser.add_argument("--top-variable-taxa", type=int, default=40)
    parser.add_argument("--max-participants", type=int, default=None)
    parser.add_argument("--max-taxa", type=int, default=None)
    parser.add_argument("--save-figures", action="store_true", help="Ask experiment 2 to save optional diagnostic figures.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--write-preprocessing-diagnostics",
        action="store_true",
        help="Allow worker experiments to write preprocessing_summary.* into the shared outdir.",
    )
    args = parser.parse_args()

    experiments = parse_experiments(args.experiments)
    logdir = Path(args.outdir) / "logs"
    commands = {exp_id: command_for_experiment(exp_id, args) for exp_id in experiments}
    max_workers = max(1, min(int(args.max_workers), len(experiments)))

    print(f"[known-mask runner] experiments={experiments} max_workers={max_workers} outdir={args.outdir}")
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_one, exp_id, cmd, logdir): exp_id
            for exp_id, cmd in commands.items()
        }
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            print(
                f"[known-mask runner] experiment {result['experiment']} "
                f"returncode={result['returncode']} log={result['log_path']}"
            )

    exp_label = "all" if experiments == ["1", "2", "3", "4"] else "exp" + "_".join(experiments)
    manifest_path = Path(args.outdir) / f"known_mask_run_manifest_{exp_label}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "returncode", "elapsed_seconds", "log_path", "command"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: str(x["experiment"])))
    print(f"[known-mask runner] wrote {manifest_path}")

    failed = [row for row in rows if int(row["returncode"]) != 0]
    if failed:
        raise SystemExit(f"{len(failed)} experiment(s) failed; see logs under {logdir}")


if __name__ == "__main__":
    main()
