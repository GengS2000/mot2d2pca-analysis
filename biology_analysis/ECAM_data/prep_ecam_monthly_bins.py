#!/usr/bin/env python
"""Rebuild ECAM's time axis as whole-month bins before the tensor export.

WHY
---
ECAM's `month_of_life` is recorded to 0.1-month precision (0.0, 0.1, ... 24.3),
and the tensor export uses that column verbatim as the `bin` axis. Across the
cohort that yields 122 distinct time points for 42 subjects, so the
participant x bin grid is only 13.4% occupied and the imputation is asked to
fill 86.6% of it. That is what inverts the variance structure downstream: the
per-participant intercept ends up carrying almost all the variance
(ICC 0.069 -> 0.975 in MaAsLin2's fits), and a between-subject contrast such as
delivery mode then loses power because its standard error scales with
sd_participant / sqrt(n_participants).

Rounding `month_of_life` to whole months gives 19 bins with 79.6% occupancy --
the same density regime DIABIMMUNE already sits in (36 integer-month bins,
70.7% full), which is the arm where the four DA methods agree best.

WHAT THIS DOES
--------------
Writes a copy of the metadata with `month_of_life` replaced by its
whole-month rounding. The export pipeline already averages duplicate
(subject, time) rows -- `x_agg = ...groupby([subject_field, time_field]).mean()`
in run_realdata_tempted_p2d2pca_comparison.py -- so two samples landing in the
same month are averaged, which is the intended "take the mean of the bin".

Nothing else is touched: sample ids, counts, taxonomy and the group fields are
left exactly as they are, so the rebuilt tensor differs from the current one
only in its time axis.

USAGE
-----
    python prep_ecam_monthly_bins.py                     # report + write
    python prep_ecam_monthly_bins.py --report-only       # inspect, write nothing
    python prep_ecam_monthly_bins.py --bin-width 2       # 2-month bins instead

Then re-run the tensor export against the new metadata, e.g.

    python run_realdata_tempted_p2d2pca_comparison.py \
        --meta_path otu_metadata_cleaned_q2_monthly.csv \
        --outdir artifacts/ECAM_otu_tensor_export_monthly \
        --export_tensors_only

and point the DA scripts at the new artifacts directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_META = HERE / "otu_metadata_cleaned_q2.csv"

SUBJECT = "host_subject_id"
TIME = "month_of_life"
SAMPLE = "SampleID"


def summarise(meta: pd.DataFrame, time_col: str, label: str) -> dict:
    n_sub = meta[SUBJECT].nunique()
    n_bin = meta[time_col].nunique()
    occupied = len(meta.drop_duplicates([SUBJECT, time_col]))
    per_sub = meta.groupby(SUBJECT)[time_col].nunique()
    info = {
        "label": label,
        "n_samples": len(meta),
        "n_subjects": n_sub,
        "n_bins": n_bin,
        "grid": n_sub * n_bin,
        "occupied": occupied,
        "fill_pct": 100.0 * occupied / (n_sub * n_bin),
        "bins_per_subject_min": int(per_sub.min()),
        "bins_per_subject_median": float(per_sub.median()),
        "bins_per_subject_max": int(per_sub.max()),
    }
    print(
        f"  {label:22s} subjects={n_sub:3d}  bins={n_bin:4d}  "
        f"grid={info['grid']:6d}  occupied={occupied:5d}  fill={info['fill_pct']:5.1f}%  "
        f"bins/subject {info['bins_per_subject_min']}-{info['bins_per_subject_max']} "
        f"(median {info['bins_per_subject_median']:.0f})"
    )
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta-path", type=Path, default=DEFAULT_META)
    ap.add_argument("--out-path", type=Path, default=None,
                    help="default: <meta stem>_monthly.csv next to the input")
    ap.add_argument("--bin-width", type=float, default=1.0,
                    help="bin width in months (default 1.0 = whole months)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if not args.meta_path.exists():
        raise SystemExit(f"metadata not found: {args.meta_path}")

    meta = pd.read_csv(args.meta_path)
    for col in (SUBJECT, TIME, SAMPLE):
        if col not in meta.columns:
            raise SystemExit(f"metadata is missing required column '{col}'")

    meta[TIME] = pd.to_numeric(meta[TIME], errors="coerce")
    n_bad = int(meta[TIME].isna().sum())
    if n_bad:
        print(f"note: dropping {n_bad} row(s) with non-numeric {TIME}")
        meta = meta[meta[TIME].notna()].copy()

    w = float(args.bin_width)
    if w <= 0:
        raise SystemExit("--bin-width must be positive")

    # Round to the bin centre so the value stays on the original month scale and
    # remains directly interpretable as "month of life".
    binned = (meta[TIME] / w).round() * w
    meta["_binned"] = np.where(np.isclose(w, round(w)) and float(w).is_integer(),
                               binned.round().astype(float), binned)

    print(f"ECAM time axis, bin width = {w:g} month(s)")
    before = summarise(meta, TIME, "current (raw 0.1 mo)")
    after = summarise(meta, "_binned", f"binned ({w:g} mo)")

    collapsed = before["occupied"] - after["occupied"]
    print(
        f"\n  {collapsed} (subject, time) pair(s) collapse into a shared bin and "
        f"will be averaged by the export's groupby(subject, time).mean()"
    )
    print(f"  grid fill {before['fill_pct']:.1f}% -> {after['fill_pct']:.1f}%  "
          f"(imputed share {100 - before['fill_pct']:.1f}% -> {100 - after['fill_pct']:.1f}%)")

    if args.report_only:
        print("\n--report-only: nothing written")
        return

    out_path = args.out_path or args.meta_path.with_name(
        f"{args.meta_path.stem}_monthly{args.meta_path.suffix}")
    out = meta.drop(columns=["_binned"]).copy()
    out[TIME] = meta["_binned"].values
    out.to_csv(out_path, index=False)
    print(f"\nwrote: {out_path}")
    print("next:")
    print(f"  python run_realdata_tempted_p2d2pca_comparison.py \\")
    print(f"      --meta_path {out_path.name} \\")
    print(f"      --outdir artifacts/ECAM_otu_tensor_export_monthly \\")
    print(f"      --export_tensors_only")


if __name__ == "__main__":
    main()
