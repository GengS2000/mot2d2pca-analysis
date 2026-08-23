# Tuned version notes:
# - Differential taxa are preferentially selected from higher-prevalence and higher-abundance taxa.
# - Default effect/noise settings are moderately easier than the original script, so zeroed data should not collapse to TP=0 immediately after frac=0.6.
# - Keep an effect sweep and select the smallest effect level satisfying: zeroed TP>0 at frac_0p6 and imputed Precision/Recall/F1 > zeroed.

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sim3_compare import (
    _default_truth_path,
    _format_frac,
    _save_dataarray,
    build_base_tensor,
    build_truth_once,
    impute_by_mask,
    make_random_mask,
)


FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_EFFECT_MEANS = [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.75, 0.80]


def _effect_truth_path_local(outdir: Path, save_format: str) -> Path:
    return outdir / ("truth_effect.nc" if str(save_format).lower() == "netcdf" else "truth_effect.zarr")


def _format_effect_tag(x: float) -> str:
    s = f"{float(x):.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def choose_detectable_taxa(
    ctf_imp,
    eps,
    rng,
    n_diff,
    prev_thr=0.60,
    abd_q=0.70,
    pos_cutoff=0.01,
):
    pct = np.maximum(np.power(10.0, ctf_imp.values.astype(float)) - eps, 0.0)
    prev = (pct > float(pos_cutoff)).mean(axis=(0, 2))
    mean_abd = pct.mean(axis=(0, 2))

    eligible = np.where(
        (prev >= prev_thr) &
        (mean_abd >= np.quantile(mean_abd, abd_q))
    )[0]

    if len(eligible) < n_diff:
        eligible = np.argsort(mean_abd)[-max(n_diff, len(mean_abd)//3):]

    return rng.choice(eligible, size=n_diff, replace=False)


def _bin_to_time_value(bin_label):
    s = str(bin_label)
    if s == "0":
        return 0.0
    if "-" in s:
        a, b = s.split("-", 1)
        return 0.5 * (float(a) + float(b))
    return float(s)


def _standardize_time_from_bins(bins, mode: str = "zscore") -> np.ndarray:
    t_raw = np.array([_bin_to_time_value(b) for b in bins], dtype=float)
    if mode == "zscore":
        return (t_raw - t_raw.mean()) / (t_raw.std() + 1e-8)
    if mode == "minmax":
        return (t_raw - t_raw.min()) / (t_raw.max() - t_raw.min() + 1e-8)
    raise ValueError(f"Unsupported time scale mode: {mode}")


def _close_log10_composition(
    da_log: xr.DataArray,
    eps: float,
    total: float = 100.0,
    name: str = "y_true",
) -> xr.DataArray:
    vals = da_log.values.astype(float)
    pct = np.maximum(np.power(10.0, vals) - eps, 0.0)

    denom = pct.sum(axis=1, keepdims=True)
    bad = denom <= 0
    if np.any(bad):
        pct = np.where(bad, 1.0, pct)
        denom = pct.sum(axis=1, keepdims=True)

    pct_closed = total * pct / denom
    y = np.log10(pct_closed + eps)
    y = np.where(np.isfinite(y), y, np.log10(eps))

    return xr.DataArray(
        y,
        dims=da_log.dims,
        coords=da_log.coords,
        name=name,
    )


def _print_composition_sanity(da_log: xr.DataArray, eps: float, prefix: str = "[sanity]"):
    pct = np.maximum(np.power(10.0, da_log.values.astype(float)) - eps, 0.0)
    sums = pct.sum(axis=1).ravel()
    print(
        f"{prefix} sum over taxon (percent space): "
        f"min={np.nanmin(sums):.6f}, "
        f"mean={np.nanmean(sums):.6f}, "
        f"max={np.nanmax(sums):.6f}, "
        f"std={np.nanstd(sums):.6e}"
    )


def build_effect_template(
    ctf_imp: xr.DataArray,
    eps: float = 1e-3,
    seed: int = 2026,
    n_cd: int = 65,
    n_noncd: int = 65,
    diff_frac: float = 0.08,
    random_slope_sd: float = 0.02,
    noise_sd: float = 0.18,
    time_scale_mode: str = "zscore",
    effect_prev_thr: float = 0.60,
    effect_abd_q: float = 0.70,
    effect_pos_cutoff: float = 0.01,
):
    rng = np.random.default_rng(seed)

    participants = np.asarray(ctf_imp.coords["participant"].values)
    taxa = np.asarray(ctf_imp.coords["taxon"].values)
    bins = np.asarray(ctf_imp.coords["bin"].values)
    i, j, k = ctf_imp.shape

    if i != (int(n_cd) + int(n_noncd)):
        raise ValueError(
            f"Participant count mismatch: I={i}, but n_cd+n_noncd={int(n_cd)+int(n_noncd)}"
        )

    shuffled_idx = rng.permutation(i)
    idx_cd = shuffled_idx[: int(n_cd)]
    idx_noncd = shuffled_idx[int(n_cd): int(n_cd) + int(n_noncd)]

    x = np.zeros(i, dtype=float)
    x[idx_cd] = 1.0

    pseudo_group = np.full(i, "nonCD", dtype=object)
    pseudo_group[idx_cd] = "CD"

    n_diff = int(np.round(float(diff_frac) * j))
    n_diff = int(np.clip(n_diff, 0, j))

    diff_idx = choose_detectable_taxa(
        ctf_imp=ctf_imp,
        eps=eps,
        rng=rng,
        n_diff=n_diff,
        prev_thr=float(effect_prev_thr),
        abd_q=float(effect_abd_q),
        pos_cutoff=float(effect_pos_cutoff),
    )

    is_diff = np.zeros(j, dtype=bool)
    is_diff[diff_idx] = True

    t_numeric = _standardize_time_from_bins(bins, mode=time_scale_mode)

    if n_diff > 0:
        direction_diff = rng.choice(np.array([-1, 1], dtype=int), size=n_diff, replace=True)
        z_a = rng.normal(0.0, 1.0, size=n_diff)
        z_h = rng.normal(0.0, 1.0, size=n_diff)
    else:
        direction_diff = np.array([], dtype=int)
        z_a = np.array([], dtype=float)
        z_h = np.array([], dtype=float)

    c_ki = rng.normal(0.0, random_slope_sd, size=(i, j))
    eta = rng.normal(0.0, noise_sd, size=(i, j, k))

    template = {
        "seed": int(seed),
        "eps": float(eps),
        "participants": participants.astype(str),
        "taxa": taxa.astype(str),
        "bins": bins,
        "idx_cd": idx_cd.astype(int),
        "idx_noncd": idx_noncd.astype(int),
        "x": x.astype(float),
        "pseudo_group": pseudo_group.astype(object),
        "diff_idx": diff_idx.astype(int),
        "is_diff": is_diff.astype(bool),
        "direction_diff": direction_diff.astype(int),
        "z_a": z_a.astype(float),
        "z_h": z_h.astype(float),
        "c_ki": c_ki.astype(float),
        "eta": eta.astype(float),
        "t_numeric": t_numeric.astype(float),
        "diff_frac": float(diff_frac),
        "time_scale_mode": str(time_scale_mode),
        "n_cd": int(n_cd),
        "n_noncd": int(n_noncd),
        "random_slope_sd": float(random_slope_sd),
        "noise_sd": float(noise_sd),
        "effect_prev_thr": float(effect_prev_thr),
        "effect_abd_q": float(effect_abd_q),
        "effect_pos_cutoff": float(effect_pos_cutoff),
    }
    return template


def save_effect_template_summary(
    outdir: Path,
    template: dict,
):
    outdir.mkdir(parents=True, exist_ok=True)

    participants = np.asarray(template["participants"])
    taxa = np.asarray(template["taxa"])
    idx_cd = np.asarray(template["idx_cd"], dtype=int)
    idx_noncd = np.asarray(template["idx_noncd"], dtype=int)
    diff_idx = np.asarray(template["diff_idx"], dtype=int)
    direction_diff = np.asarray(template["direction_diff"], dtype=int)
    x = np.asarray(template["x"], dtype=int)

    df_pseudo_group = pd.DataFrame(
        {
            "participant": participants.astype(str),
            "pseudo_group": np.where(x == 1, "CD", "nonCD"),
            "X": x,
        }
    )
    df_pseudo_group.to_csv(outdir / "template_pseudo_group.csv", index=False)

    direction = np.zeros(len(taxa), dtype=int)
    if len(diff_idx) > 0:
        direction[diff_idx] = direction_diff

    df_taxon_template = pd.DataFrame(
        {
            "taxon": taxa.astype(str),
            "taxon_index": np.arange(len(taxa), dtype=int),
            "is_differential": np.asarray(template["is_diff"], dtype=bool),
            "direction": direction,
        }
    )
    df_taxon_template.to_csv(outdir / "template_taxon_design.csv", index=False)

    summary = {
        "template_seed": int(template["seed"]),
        "eps": float(template["eps"]),
        "diff_frac": float(template["diff_frac"]),
        "effect_prev_thr": float(template.get("effect_prev_thr", 0.3)),
        "effect_abd_q": float(template.get("effect_abd_q", 0.5)),
        "effect_pos_cutoff": float(template.get("effect_pos_cutoff", 0.01)),
        "time_scale_mode": str(template["time_scale_mode"]),
        "n_cd": int(template["n_cd"]),
        "n_noncd": int(template["n_noncd"]),
        "random_slope_sd": float(template["random_slope_sd"]),
        "noise_sd": float(template["noise_sd"]),
        "participants_cd": participants[idx_cd].astype(str).tolist(),
        "participants_noncd": participants[idx_noncd].astype(str).tolist(),
        "differential_taxa": taxa[diff_idx].astype(str).tolist(),
        "t_numeric": {
            str(bi): float(ti) for bi, ti in zip(template["bins"], np.asarray(template["t_numeric"], dtype=float))
        },
        "frozen_design": True,
    }
    (outdir / "template_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def inject_pseudo_group_effect_from_template(
    ctf_imp: xr.DataArray,
    template: dict,
    eps: float = 1e-3,
    group_effect_mean: float = 0.55,
    group_effect_sd: float = 0.06,
    interaction_mean: float = 0.03,
    interaction_sd: float = 0.01,
    verbose: bool = True,
):
    participants = np.asarray(ctf_imp.coords["participant"].values)
    taxa = np.asarray(ctf_imp.coords["taxon"].values)
    bins = np.asarray(ctf_imp.coords["bin"].values)
    i, j, k = ctf_imp.shape

    y0 = ctf_imp.values.astype(float)
    b = np.maximum(np.power(10.0, y0) - eps, 0.0)

    x = np.asarray(template["x"], dtype=float)
    diff_idx = np.asarray(template["diff_idx"], dtype=int)
    is_diff = np.asarray(template["is_diff"], dtype=bool)
    direction_diff = np.asarray(template["direction_diff"], dtype=int)
    z_a = np.asarray(template["z_a"], dtype=float)
    z_h = np.asarray(template["z_h"], dtype=float)
    c_ki = np.asarray(template["c_ki"], dtype=float)
    eta = np.asarray(template["eta"], dtype=float)
    t_numeric = np.asarray(template["t_numeric"], dtype=float)
    pseudo_group = np.asarray(template["pseudo_group"], dtype=object)
    idx_cd = np.asarray(template["idx_cd"], dtype=int)
    idx_noncd = np.asarray(template["idx_noncd"], dtype=int)

    if c_ki.shape != (i, j):
        raise ValueError(f"Template c_ki shape mismatch: expected {(i, j)}, got {c_ki.shape}")
    if eta.shape != (i, j, k):
        raise ValueError(f"Template eta shape mismatch: expected {(i, j, k)}, got {eta.shape}")
    if len(t_numeric) != k:
        raise ValueError(f"Template t_numeric length mismatch: expected {k}, got {len(t_numeric)}")

    direction = np.zeros(j, dtype=int)
    a_k = np.zeros(j, dtype=float)
    h_k = np.zeros(j, dtype=float)

    if len(diff_idx) > 0:
        a_abs = np.clip(group_effect_mean + group_effect_sd * z_a, 0.0, None)
        h_abs = np.clip(interaction_mean + interaction_sd * z_h, 0.0, None)

        direction[diff_idx] = direction_diff
        a_k[diff_idx] = direction_diff * a_abs
        h_k[diff_idx] = direction_diff * h_abs

    x_3d = x[:, None, None]
    t_3d = t_numeric[None, None, :]

    log_mult = (
        a_k[None, :, None] * x_3d
        + c_ki[:, :, None] * t_3d
        + h_k[None, :, None] * x_3d * t_3d
        + eta
    )

    b_tilde = b * np.power(10.0, log_mult)
    b_tilde = np.where(np.isfinite(b_tilde), b_tilde, 0.0)
    b_tilde = np.maximum(b_tilde, 0.0)

    denom = b_tilde.sum(axis=1, keepdims=True)
    bad = denom <= 0
    if np.any(bad):
        b_tilde = np.where(bad, 1.0, b_tilde)
        denom = b_tilde.sum(axis=1, keepdims=True)

    b_closed = 100.0 * b_tilde / denom

    y_star = np.log10(b_closed + eps)
    y_star = np.where(np.isfinite(y_star), y_star, np.log10(eps))

    da_gt = xr.DataArray(
        y_star,
        dims=ctf_imp.dims,
        coords=ctf_imp.coords,
        name="y_true",
    )

    df_pseudo_group = pd.DataFrame(
        {
            "participant": participants.astype(str),
            "pseudo_group": pseudo_group.astype(str),
            "X": x.astype(int),
        }
    )

    df_taxon_effect = pd.DataFrame(
        {
            "taxon": taxa.astype(str),
            "taxon_index": np.arange(j, dtype=int),
            "is_differential": is_diff.astype(bool),
            "direction": direction.astype(int),
            "a_k": a_k.astype(float),
            "h_k": h_k.astype(float),
            "mean_abs_c_ki": np.mean(np.abs(c_ki), axis=0),
        }
    )

    effect_meta = {
        "template_seed": int(template["seed"]),
        "eps": float(eps),
        "diff_frac": float(template["diff_frac"]),
        "effect_prev_thr": float(template.get("effect_prev_thr", 0.3)),
        "effect_abd_q": float(template.get("effect_abd_q", 0.5)),
        "effect_pos_cutoff": float(template.get("effect_pos_cutoff", 0.01)),
        "time_scale_mode": str(template["time_scale_mode"]),
        "participants_cd": participants[idx_cd].astype(str).tolist(),
        "participants_noncd": participants[idx_noncd].astype(str).tolist(),
        "differential_taxa": taxa[diff_idx].astype(str).tolist(),
        "t_numeric": {str(bi): float(ti) for bi, ti in zip(bins, t_numeric)},
        "parameter_settings": {
            "n_cd": int(template["n_cd"]),
            "n_noncd": int(template["n_noncd"]),
            "group_effect_mean": float(group_effect_mean),
            "group_effect_sd": float(group_effect_sd),
            "interaction_mean": float(interaction_mean),
            "interaction_sd": float(interaction_sd),
            "random_slope_sd": float(template["random_slope_sd"]),
            "noise_sd": float(template["noise_sd"]),
        },
        "frozen_design": True,
        "compositional_closure_applied": True,
    }

    if verbose:
        print(f"[effect-from-template] participants={i}, taxa={j}, bins={k}")
        print(f"[effect-from-template] group_effect_mean={group_effect_mean}")
        print("[effect-from-template] group counts:", df_pseudo_group["pseudo_group"].value_counts().to_dict())
        print(f"[effect-from-template] differential taxa: {int(is_diff.sum())}")
        _print_composition_sanity(da_gt, eps=eps, prefix="[effect-from-template sanity]")

    return da_gt, df_pseudo_group, df_taxon_effect, effect_meta


def inject_pseudo_group_effect_local(
    ctf_imp: xr.DataArray,
    eps: float = 1e-3,
    seed: int = 2026,
    n_cd: int = 65,
    n_noncd: int = 65,
    diff_frac: float = 0.08,
    group_effect_mean: float = 0.55,
    group_effect_sd: float = 0.06,
    interaction_mean: float = 0.03,
    interaction_sd: float = 0.01,
    random_slope_sd: float = 0.02,
    noise_sd: float = 0.18,
    time_scale_mode: str = "zscore",
    effect_prev_thr: float = 0.60,
    effect_abd_q: float = 0.70,
    effect_pos_cutoff: float = 0.01,
    verbose: bool = True,
):
    template = build_effect_template(
        ctf_imp=ctf_imp,
        eps=eps,
        seed=seed,
        n_cd=n_cd,
        n_noncd=n_noncd,
        diff_frac=diff_frac,
        random_slope_sd=random_slope_sd,
        noise_sd=noise_sd,
        time_scale_mode=time_scale_mode,
        effect_prev_thr=effect_prev_thr,
        effect_abd_q=effect_abd_q,
        effect_pos_cutoff=effect_pos_cutoff,
    )
    return inject_pseudo_group_effect_from_template(
        ctf_imp=ctf_imp,
        template=template,
        eps=eps,
        group_effect_mean=group_effect_mean,
        group_effect_sd=group_effect_sd,
        interaction_mean=interaction_mean,
        interaction_sd=interaction_sd,
        verbose=verbose,
    )


def save_local_effect_artifacts(
    truth_dir: Path,
    da_gt: xr.DataArray,
    df_pseudo_group: pd.DataFrame,
    df_taxon_effect: pd.DataFrame,
    effect_meta: dict,
    save_format: str,
    compress_level: int,
):
    truth_dir.mkdir(parents=True, exist_ok=True)
    df_pseudo_group.to_csv(truth_dir / "effect_pseudo_group.csv", index=False)
    df_taxon_effect.to_csv(truth_dir / "effect_taxon_effect.csv", index=False)
    (truth_dir / "effect_meta.json").write_text(
        json.dumps(effect_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _save_dataarray(
        da_gt,
        "y_true",
        _effect_truth_path_local(truth_dir, save_format),
        save_format=save_format,
        compress_level=compress_level,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build compositional CTF truth once, then save mask/zeroed/imputed tensors for fixed fractions."
    )

    parser.add_argument("--tax_path", type=str, default="hmp2_mgx_taxonomy.tsv")
    parser.add_argument("--meta_path", type=str, default="metagenomics.xlsx")
    parser.add_argument("--outdir", type=str, default="simulation3_effect_tune")
    parser.add_argument("--truth_path", type=str, default=None)
    parser.add_argument("--save_format", type=str, default="netcdf", choices=["netcdf", "zarr"])
    parser.add_argument("--compress_level", type=int, default=4)
    parser.add_argument(
        "--force_rebuild_truth",
        "--force-rebuild-truth",
        action="store_true",
        dest="force_rebuild_truth",
    )
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--min_pos_cells", "--min-pos-cells", type=int, default=10, dest="min_pos_cells")
    parser.add_argument("--min_participants", "--min-participants", type=int, default=5, dest="min_participants")

    parser.add_argument("--ctf_rank", "--ctf-rank", type=int, default=3, dest="ctf_rank")
    parser.add_argument(
        "--ctf_method",
        "--ctf-method",
        type=str,
        default="fallback",
        choices=["auto", "gemelli", "fallback"],
        dest="ctf_method",
    )
    parser.add_argument("--ctf_seed", "--ctf-seed", type=int, default=0, dest="ctf_seed")

    parser.add_argument("--enable_effect", "--enable-effect", action="store_true", dest="enable_effect")
    parser.add_argument("--effect_seed", "--effect-seed", type=int, default=2026, dest="effect_seed")
    parser.add_argument("--n_cd", "--n-cd", type=int, default=65, dest="n_cd")
    parser.add_argument("--n_noncd", "--n-noncd", type=int, default=65, dest="n_noncd")

    parser.add_argument("--diff_frac", "--diff-frac", type=float, default=0.08, dest="diff_frac")
    parser.add_argument("--group_effect_mean", "--group-effect-mean", type=float, default=0.50, dest="group_effect_mean")
    parser.add_argument(
        "--group_effect_means",
        "--group-effect-means",
        type=float,
        nargs="+",
        default=DEFAULT_EFFECT_MEANS,
        dest="group_effect_means",
        help="Multiple group effect means, e.g. 0.50 0.525 0.55 ...",
    )
    parser.add_argument("--group_effect_sd", "--group-effect-sd", type=float, default=0.03, dest="group_effect_sd")
    parser.add_argument("--interaction_mean", "--interaction-mean", type=float, default=0.01, dest="interaction_mean")
    parser.add_argument("--interaction_sd", "--interaction-sd", type=float, default=0.003, dest="interaction_sd")
    parser.add_argument("--random_slope_sd", "--random-slope-sd", type=float, default=0.008, dest="random_slope_sd")
    parser.add_argument("--noise_sd", "--noise-sd", type=float, default=0.10, dest="noise_sd")
    parser.add_argument("--time_scale_mode", type=str, default="zscore", choices=["zscore", "minmax"])

    parser.add_argument("--effect_prev_thr", "--effect-prev-thr", type=float, default=0.75, dest="effect_prev_thr")
    parser.add_argument("--effect_abd_q", "--effect-abd-q", type=float, default=0.75, dest="effect_abd_q")
    parser.add_argument("--effect_pos_cutoff", "--effect-pos-cutoff", type=float, default=0.01, dest="effect_pos_cutoff")

    parser.add_argument("--mask_seed", "--mask-seed", type=int, default=123, dest="mask_seed")
    parser.add_argument("--algo_seed", "--algo-seed", type=int, default=0, dest="algo_seed")
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--T0", type=int, default=30)
    parser.add_argument("--T", type=int, default=30)

    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    effect_means = args.group_effect_means if args.group_effect_means is not None else [args.group_effect_mean]

    if (len(effect_means) > 1) and (args.truth_path is not None):
        raise ValueError(
            "When using multiple --group_effect_means, please leave --truth_path unset. "
            "Each scenario will save its own truth file under a separate subdirectory."
        )

    base_da, base_observed_mask, meta_info = build_base_tensor(
        tax_path=args.tax_path,
        meta_path=args.meta_path,
        eps=args.eps,
        min_pos_cells=args.min_pos_cells,
        min_participants=args.min_participants,
        verbose=verbose,
    )

    truth_dir = outdir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)

    if args.truth_path is None:
        truth_path = _default_truth_path(
            outdir=truth_dir,
            save_format=args.save_format,
            enable_effect=args.enable_effect,
            effect_seed=args.effect_seed,
            n_cd=args.n_cd,
            n_noncd=args.n_noncd,
            diff_frac=args.diff_frac,
            group_effect_mean=args.group_effect_mean,
            group_effect_sd=args.group_effect_sd,
            interaction_mean=args.interaction_mean,
            interaction_sd=args.interaction_sd,
            random_slope_sd=args.random_slope_sd,
            noise_sd=args.noise_sd,
        )
    else:
        truth_path = Path(args.truth_path)

    base_truth_path = truth_dir / (
        "truth_ctf_base.zarr" if str(args.save_format).lower() == "zarr" else "truth_ctf_base.nc"
    )

    da_gt_base = build_truth_once(
        base_da=base_da,
        base_observed_mask=base_observed_mask,
        truth_path=base_truth_path,
        force_rebuild=args.force_rebuild_truth,
        save_format=args.save_format,
        compress_level=args.compress_level,
        rank=args.ctf_rank,
        method=args.ctf_method,
        ctf_seed=args.ctf_seed,
        ctf_verbose=verbose,
        eps=args.eps,
        enable_effect=False,
        effect_artifact_dir=truth_dir,
    )
    da_gt_base.name = "y_true"

    if verbose:
        _print_composition_sanity(da_gt_base, eps=args.eps, prefix="[base truth sanity]")

    pca_params = {
        "d": args.d,
        "q": args.q,
        "lam": args.lam,
        "sigma": args.sigma,
        "T0": args.T0,
        "T": args.T,
    }

    bins_use = np.array(da_gt_base.coords["bin"].values)
    zero_value = np.log10(args.eps)

    effect_template = None
    if args.enable_effect:
        effect_template = build_effect_template(
            ctf_imp=da_gt_base,
            eps=args.eps,
            seed=args.effect_seed,
            n_cd=args.n_cd,
            n_noncd=args.n_noncd,
            diff_frac=args.diff_frac,
            random_slope_sd=args.random_slope_sd,
            noise_sd=args.noise_sd,
            time_scale_mode=args.time_scale_mode,
            effect_prev_thr=args.effect_prev_thr,
            effect_abd_q=args.effect_abd_q,
            effect_pos_cutoff=args.effect_pos_cutoff,
        )
        common_design_dir = outdir / "common_design"
        save_effect_template_summary(common_design_dir, effect_template)

    common_mask_dir = outdir / "common_masks"
    common_mask_dir.mkdir(parents=True, exist_ok=True)
    fixed_masks = {}

    for frac in FRACS:
        frac_tag = _format_frac(frac)
        da_rand_mask = make_random_mask(
            da_ref=da_gt_base,
            bins=bins_use,
            frac=float(frac),
            seed=int(args.mask_seed),
            zero_value=zero_value,
            eligible_mode="all_finite",
        )
        fixed_masks[frac] = da_rand_mask
        _save_dataarray(
            da_rand_mask,
            "mask",
            common_mask_dir / f"mask_frac_{frac_tag}",
            save_format=args.save_format,
            compress_level=args.compress_level,
        )

    if not args.enable_effect:
        effect_means = [None]

    for this_effect_mean in effect_means:
        if args.enable_effect:
            effect_tag = _format_effect_tag(this_effect_mean)
            scenario_outdir = outdir / f"group_effect_{effect_tag}"
            scenario_truth_dir = scenario_outdir / "truth"
            scenario_truth_dir.mkdir(parents=True, exist_ok=True)

            da_gt, df_pseudo_group, df_taxon_effect, effect_meta = inject_pseudo_group_effect_from_template(
                ctf_imp=da_gt_base,
                template=effect_template,
                eps=args.eps,
                group_effect_mean=float(this_effect_mean),
                group_effect_sd=args.group_effect_sd,
                interaction_mean=args.interaction_mean,
                interaction_sd=args.interaction_sd,
                verbose=verbose,
            )

            save_local_effect_artifacts(
                truth_dir=scenario_truth_dir,
                da_gt=da_gt,
                df_pseudo_group=df_pseudo_group,
                df_taxon_effect=df_taxon_effect,
                effect_meta=effect_meta,
                save_format=args.save_format,
                compress_level=args.compress_level,
            )

            truth_path_this = _default_truth_path(
                outdir=scenario_truth_dir,
                save_format=args.save_format,
                enable_effect=True,
                effect_seed=args.effect_seed,
                n_cd=args.n_cd,
                n_noncd=args.n_noncd,
                diff_frac=args.diff_frac,
                group_effect_mean=float(this_effect_mean),
                group_effect_sd=args.group_effect_sd,
                interaction_mean=args.interaction_mean,
                interaction_sd=args.interaction_sd,
                random_slope_sd=args.random_slope_sd,
                noise_sd=args.noise_sd,
            )
        else:
            scenario_outdir = outdir
            da_gt = da_gt_base.copy(deep=True)
            truth_path_this = truth_path

        da_gt.name = "y_true"
        _save_dataarray(
            da_gt,
            "y_true",
            truth_path_this,
            save_format=args.save_format,
            compress_level=args.compress_level,
        )

        for frac in FRACS:
            frac_tag = _format_frac(frac)
            frac_dir = scenario_outdir / f"frac_{frac_tag}"
            frac_dir.mkdir(parents=True, exist_ok=True)

            da_rand_mask = fixed_masks[frac].copy(deep=True)

            da_zeroed = da_gt.copy(deep=True)
            vals = da_zeroed.sel(bin=bins_use).values.copy()
            vals[da_rand_mask.values.astype(bool)] = zero_value
            da_zeroed.loc[dict(bin=bins_use)] = vals
            da_zeroed.name = "y_zeroed"

            da_imputed, bins_trained = impute_by_mask(
                da_input=da_zeroed,
                replace_mask=da_rand_mask,
                model_seed=int(args.algo_seed),
                method="penalized_2d2_pca",
                method_params={},
                pca_params=pca_params,
                zero_value=zero_value,
                observed_mode="positive_only_except_replace",
                verbose=verbose,
            )
            da_imputed.name = "y_imputed"

            _save_dataarray(
                da_rand_mask,
                "mask",
                frac_dir / f"mask_frac_{frac_tag}",
                save_format=args.save_format,
                compress_level=args.compress_level,
            )
            _save_dataarray(
                da_zeroed,
                "y_zeroed",
                frac_dir / f"zeroed_frac_{frac_tag}",
                save_format=args.save_format,
                compress_level=args.compress_level,
            )
            _save_dataarray(
                da_imputed,
                "y_imputed",
                frac_dir / f"imputed_penalized_2d2_pca_frac_{frac_tag}",
                save_format=args.save_format,
                compress_level=args.compress_level,
            )

            if verbose:
                print(
                    f"[saved] effect_mean={this_effect_mean} frac={frac:.1f} "
                    f"bins_trained={len(bins_trained)} dir={frac_dir}"
                )


if __name__ == "__main__":
    main()
