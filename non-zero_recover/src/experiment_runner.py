from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .preprocessing import load_profile_tensor


def parse_csv_list(value: str | Iterable[str], cast=str) -> List:
    """Parse comma-separated CLI values."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).split(",")
    return [cast(x.strip()) for x in raw if str(x).strip()]


def normalize_replacement_mode(mode: str) -> str:
    """Normalize legacy mask names to the current provided-mask terminology."""
    mode = str(mode).strip()
    if mode in {"known_mask", "oracle_mask", "provided_mask", "provided_imputation_mask"}:
        return "known_mask"
    return mode


def normalize_replacement_modes(modes: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for mode in modes:
        mode = normalize_replacement_mode(mode)
        if mode not in normalized:
            normalized.append(mode)
    return normalized


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach shared experiment CLI arguments."""
    parser.add_argument("--tax-path", default="hmp2_mgx_taxonomy.tsv")
    parser.add_argument("--meta-path", default="metagenomics.xlsx")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--min-pos-cells", type=int, default=10)
    parser.add_argument("--min-participants", type=int, default=5)
    parser.add_argument("--rank-d", type=int, default=10)
    parser.add_argument("--rank-q", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--T0", type=int, default=30)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--fracs", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--methods", default="penalized_2d2_pca,no_imputation")
    parser.add_argument("--mechanisms", default="random_entry,low_depth_sample,time_point,subject_specific_dropout,consecutive_time_window_dropout,low_abundance_biased")
    parser.add_argument("--replacement-modes", default="known_mask")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-participants", type=int, default=None)
    parser.add_argument("--max-taxa", type=int, default=None)
    parser.add_argument(
        "--skip-preprocessing-diagnostics",
        action="store_true",
        help="Do not rewrite preprocessing_summary.*; useful when several experiments share one outdir in parallel.",
    )
    return parser


def load_data_from_args(args, experiment_outdir: Path):
    """Preprocess data and print required run diagnostics."""
    da_log, da_raw, observed_positive_mask, meta_info = load_profile_tensor(
        tax_path=args.tax_path,
        meta_path=args.meta_path,
        eps=args.eps,
        min_pos_cells=args.min_pos_cells,
        min_participants=args.min_participants,
        outdir=Path(args.outdir),
        max_participants=getattr(args, "max_participants", None),
        max_taxa=getattr(args, "max_taxa", None),
        save_diagnostics=not bool(getattr(args, "skip_preprocessing_diagnostics", False)),
        verbose=True,
    )
    zero_value = float(np.log10(args.eps))
    methods = parse_csv_list(args.methods, str)
    mechanisms = parse_csv_list(args.mechanisms, str)
    fracs = parse_csv_list(args.fracs, float)
    replacement_modes = normalize_replacement_modes(parse_csv_list(args.replacement_modes, str))
    args.replacement_modes = ",".join(replacement_modes)
    print(f"[run] data shape: {da_log.shape}")
    print(f"[run] observed positive ratio: {float(observed_positive_mask.values.mean()):.6f}")
    print(f"[run] zero_value: {zero_value:.6f}")
    print(f"[run] methods: {methods}")
    print(f"[run] mechanisms: {mechanisms}")
    print(f"[run] fracs: {fracs}")
    print(f"[run] repeats: {args.repeats}")
    experiment_outdir.mkdir(parents=True, exist_ok=True)
    return da_log, da_raw, observed_positive_mask, meta_info, zero_value, methods, mechanisms, fracs, replacement_modes


def pca_params_from_args(args) -> Dict[str, Any]:
    return {"d": int(args.rank_d), "q": int(args.rank_q), "lam": float(args.lam), "sigma": float(args.sigma), "T0": int(args.T0), "T": int(args.T)}


def seeds_from_args(args) -> List[int]:
    seeds = parse_csv_list(args.seeds, int)
    if not seeds:
        seeds = [202601 + i for i in range(int(args.repeats))]
    while len(seeds) < int(args.repeats):
        seeds.append(seeds[-1] + 1)
    return seeds[: int(args.repeats)]


def save_config(path: str | Path, args, extra: Dict[str, Any]) -> None:
    cfg = {k: v for k, v in vars(args).items()}
    cfg.update(extra)
    Path(path).write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
