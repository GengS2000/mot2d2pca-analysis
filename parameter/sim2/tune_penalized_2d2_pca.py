from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

import ctf_reconstruct as ctf


@dataclass(frozen=True)
class ParamConfig:
    """Parameter bundle for one penalized_2d2_pca run."""

    d: int
    q: int
    lam: float
    sigma: float
    T0: int
    T: int
    stage: str
    config_id: str
    parent_config_id: str | None = None
    source_rank: int | None = None


def parse_int_grid(text: str) -> list[int]:
    """Parse a comma-separated integer grid."""

    items = [x.strip() for x in str(text).split(",") if x.strip()]
    if not items:
        raise ValueError("Empty integer grid.")
    return [int(x) for x in items]


def parse_float_grid(text: str) -> list[float]:
    """Parse a comma-separated float grid."""

    items = [x.strip() for x in str(text).split(",") if x.strip()]
    if not items:
        raise ValueError("Empty float grid.")
    return [float(x) for x in items]


def _id_token(value: Any) -> str:
    """Convert a value into a filesystem-stable token."""

    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = np.format_float_positional(value, trim="-", precision=8, unique=False)
        text = text.rstrip("0").rstrip(".") if "." in text else text
        text = text if text else "0"
        return text.replace("-", "m").replace(".", "p")
    text = str(value)
    return text.replace("-", "m").replace(".", "p").replace(" ", "_")


def build_config_id(d: int, q: int, lam: float, sigma: float, T0: int, T: int) -> str:
    """Build a stable config identifier."""

    return (
        f"d{_id_token(d)}_q{_id_token(q)}_"
        f"lam{_id_token(lam)}_sig{_id_token(sigma)}_"
        f"T0{_id_token(T0)}_T{_id_token(T)}"
    )


def create_param_config(
    d: int,
    q: int,
    lam: float,
    sigma: float,
    T0: int,
    T: int,
    stage: str,
    parent_config_id: str | None = None,
    source_rank: int | None = None,
) -> ParamConfig:
    """Create a parameter config with a deterministic id."""

    return ParamConfig(
        d=int(d),
        q=int(q),
        lam=float(lam),
        sigma=float(sigma),
        T0=int(T0),
        T=int(T),
        stage=str(stage),
        config_id=build_config_id(int(d), int(q), float(lam), float(sigma), int(T0), int(T)),
        parent_config_id=parent_config_id,
        source_rank=source_rank,
    )


def make_random_mask_allfinite(
    da_gt: xr.DataArray,
    bins: Iterable[Any],
    frac: float,
    seed: int,
) -> xr.DataArray:
    """Randomly mark a fraction of all finite cells as artificial missing cells."""

    da_sub = da_gt.sel(bin=list(bins))
    vals = da_sub.values
    eligible = np.isfinite(vals)
    idx = np.flatnonzero(eligible)
    n_eligible = int(idx.size)

    if n_eligible == 0 or frac <= 0:
        mask = np.zeros(vals.shape, dtype=bool)
        return xr.DataArray(mask, coords=da_sub.coords, dims=da_sub.dims, name="replace_mask")

    n_pick = int(np.round(float(frac) * n_eligible))
    if frac > 0 and n_pick == 0:
        n_pick = 1
    n_pick = min(n_pick, n_eligible)

    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=n_pick, replace=False)
    mask = np.zeros(vals.size, dtype=bool)
    mask[pick] = True
    mask = mask.reshape(vals.shape)
    return xr.DataArray(mask, coords=da_sub.coords, dims=da_sub.dims, name="replace_mask")


def inject_missing_by_mask(
    da_gt: xr.DataArray,
    replace_mask: xr.DataArray,
    missing_value: float,
) -> xr.DataArray:
    """Inject missing_value into cells explicitly indicated by replace_mask."""

    if replace_mask.dims != da_gt.dims or replace_mask.shape != da_gt.shape:
        raise ValueError("replace_mask must match da_gt exactly.")

    da_corrupted = da_gt.copy(deep=True)
    vals = da_corrupted.values.copy()
    vals[replace_mask.values.astype(bool)] = float(missing_value)
    da_corrupted.values[:] = vals
    da_corrupted.name = "log_abundance_corrupted"
    return da_corrupted


def apply_reconstruction_on_mask(
    da_corrupted: xr.DataArray,
    replace_mask: xr.DataArray,
    X: np.ndarray,
    Z: np.ndarray,
    C: np.ndarray,
    meta: dict[str, Any],
) -> xr.DataArray:
    """Apply reconstruction only on explicit artificial missing positions."""

    bins_trained = np.array(meta["bin"])
    abar = np.asarray(meta["Abar"], dtype=float)
    da_imputed = da_corrupted.copy(deep=True)

    for ti, b in enumerate(bins_trained):
        ai = da_imputed.sel(bin=b).values
        ahat = abar + (Z @ C[ti] @ X.T)
        mk = replace_mask.sel(bin=b).values.astype(bool)
        if np.any(mk):
            ai_new = ai.copy()
            ai_new[mk] = ahat[mk]
            da_imputed.loc[dict(bin=b)] = ai_new

    da_imputed.name = "log_abundance_imputed"
    return da_imputed


def _repeat_output_paths(base_dir: Path, frac: float, repeat: int) -> tuple[Path, Path, Path]:
    """Build output paths for one repeat."""

    frac_str = ctf._format_frac(frac)
    repeat_str = f"{int(repeat):03d}"
    bin_path = base_dir / f"repeat_metrics_frac_{frac_str}_repeat_{repeat_str}.csv"
    repeat_path = base_dir / f"repeat_summary_frac_{frac_str}_repeat_{repeat_str}.csv"
    fail_path = base_dir / f"repeat_failed_frac_{frac_str}_repeat_{repeat_str}.log"
    return bin_path, repeat_path, fail_path


def run_one_repeat_allfinite(
    da_gt: xr.DataArray,
    frac: float,
    repeat: int,
    mask_seed: int,
    algo_seed: int,
    outdir: Path,
    config: ParamConfig,
    missing_value: float,
    save_imputed_matrix: bool = False,
    save_format: str = "netcdf",
    compress_level: int = 4,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one repeat with explicit replace_mask over all finite truth cells."""

    bins_use = np.array(da_gt.coords["bin"].values)
    replace_mask = make_random_mask_allfinite(da_gt=da_gt, bins=bins_use, frac=frac, seed=mask_seed)
    injected_cells = int(replace_mask.values.sum())
    da_corrupted = inject_missing_by_mask(da_gt=da_gt, replace_mask=replace_mask, missing_value=missing_value)

    Xf, Zf, Cf, metaf = ctf.penalized_2d2_pca(
        da=da_corrupted,
        time=bins_use,
        d=config.d,
        q=config.q,
        lam=config.lam,
        sigma=config.sigma,
        T0=config.T0,
        T=config.T,
        seed=algo_seed,
        verbose=verbose,
    )
    da_imputed = apply_reconstruction_on_mask(
        da_corrupted=da_corrupted,
        replace_mask=replace_mask,
        X=Xf,
        Z=Zf,
        C=Cf,
        meta=metaf,
    )

    df_injected = ctf.evaluate_per_bin(
        da_true=da_gt,
        da_before=da_corrupted,
        da_after=da_imputed,
        bins=bins_use,
        eval_mask=replace_mask,
        bin_dim="bin",
    )
    df_injected["scope"] = "InjectedCells"

    df_all = ctf.evaluate_per_bin(
        da_true=da_gt,
        da_before=da_corrupted,
        da_after=da_imputed,
        bins=bins_use,
        eval_mask=None,
        bin_dim="bin",
    )
    df_all["scope"] = "AllCells"

    df_bin = pd.concat([df_injected, df_all], ignore_index=True)
    df_bin["frac"] = float(frac)
    df_bin["repeat"] = int(repeat)
    df_bin["mask_seed"] = int(mask_seed)
    df_bin["algo_seed"] = int(algo_seed)
    df_bin["injected_cells"] = int(injected_cells)
    df_bin["config_id"] = config.config_id
    df_bin["stage"] = config.stage
    df_bin["d"] = int(config.d)
    df_bin["q"] = int(config.q)
    df_bin["lam"] = float(config.lam)
    df_bin["sigma"] = float(config.sigma)
    df_bin["T0"] = int(config.T0)
    df_bin["T"] = int(config.T)
    df_bin["parent_config_id"] = config.parent_config_id
    df_bin["source_rank"] = config.source_rank
    df_bin["bin_order"] = df_bin["bin"].map(ctf._bin_key)
    df_bin = df_bin.sort_values(["scope", "bin_order"]).drop(columns=["bin_order"])

    df_repeat = ctf._build_weighted_summary(df_bin)
    df_repeat["frac"] = float(frac)
    df_repeat["repeat"] = int(repeat)
    df_repeat["mask_seed"] = int(mask_seed)
    df_repeat["algo_seed"] = int(algo_seed)
    df_repeat["injected_cells"] = int(injected_cells)
    df_repeat["config_id"] = config.config_id
    df_repeat["stage"] = config.stage
    df_repeat["d"] = int(config.d)
    df_repeat["q"] = int(config.q)
    df_repeat["lam"] = float(config.lam)
    df_repeat["sigma"] = float(config.sigma)
    df_repeat["T0"] = int(config.T0)
    df_repeat["T"] = int(config.T)
    df_repeat["parent_config_id"] = config.parent_config_id
    df_repeat["source_rank"] = config.source_rank

    outdir.mkdir(parents=True, exist_ok=True)
    bin_path, repeat_path, _ = _repeat_output_paths(outdir, frac=frac, repeat=repeat)
    df_bin.to_csv(bin_path, index=False)
    df_repeat.to_csv(repeat_path, index=False)

    if save_imputed_matrix:
        frac_str = ctf._format_frac(frac)
        repeat_str = f"{int(repeat):03d}"
        ctf._save_dataarray(
            da_imputed,
            "y_imp",
            outdir / f"imputed_frac_{frac_str}_repeat_{repeat_str}",
            save_format=save_format,
            compress_level=compress_level,
        )

    return df_bin, df_repeat


def run_repeats_for_frac_allfinite(
    da_gt: xr.DataArray,
    frac: float,
    n_repeats: int,
    repeat_start: int,
    mask_seed_base: int,
    algo_seed_base: int,
    outdir: Path,
    config: ParamConfig,
    missing_value: float,
    save_imputed_matrix: bool = False,
    save_format: str = "netcdf",
    compress_level: int = 4,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run repeated experiments for a single frac under one parameter config."""

    outdir.mkdir(parents=True, exist_ok=True)
    bin_parts: list[pd.DataFrame] = []
    repeat_parts: list[pd.DataFrame] = []
    fail_rows: list[dict[str, Any]] = []

    for r in range(repeat_start, repeat_start + n_repeats):
        mask_seed = int(mask_seed_base + r)
        algo_seed = int(algo_seed_base + r)
        try:
            df_bin, df_repeat = run_one_repeat_allfinite(
                da_gt=da_gt,
                frac=frac,
                repeat=r,
                mask_seed=mask_seed,
                algo_seed=algo_seed,
                outdir=outdir,
                config=config,
                missing_value=missing_value,
                save_imputed_matrix=save_imputed_matrix,
                save_format=save_format,
                compress_level=compress_level,
                verbose=verbose,
            )
            bin_parts.append(df_bin)
            repeat_parts.append(df_repeat)
            if verbose:
                print(
                    f"[repeat done] stage={config.stage} config={config.config_id} "
                    f"frac={frac:.3f} repeat={r} mask_seed={mask_seed} algo_seed={algo_seed}"
                )
        except Exception as exc:
            _, _, fail_path = _repeat_output_paths(outdir, frac=frac, repeat=r)
            fail_path.write_text(traceback.format_exc(), encoding="utf-8")
            fail_rows.append(
                {
                    "stage": config.stage,
                    "config_id": config.config_id,
                    "d": int(config.d),
                    "q": int(config.q),
                    "lam": float(config.lam),
                    "sigma": float(config.sigma),
                    "T0": int(config.T0),
                    "T": int(config.T),
                    "frac": float(frac),
                    "repeat": int(r),
                    "mask_seed": mask_seed,
                    "algo_seed": algo_seed,
                    "error": str(exc),
                    "log_path": str(fail_path),
                }
            )
            if verbose:
                print(
                    f"[repeat failed] stage={config.stage} config={config.config_id} "
                    f"frac={frac:.3f} repeat={r} log={fail_path}"
                )

    df_bin_all = pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame()
    df_repeat_all = pd.concat(repeat_parts, ignore_index=True) if repeat_parts else pd.DataFrame()
    df_fail = pd.DataFrame(fail_rows)

    frac_str = ctf._format_frac(frac)
    if not df_bin_all.empty:
        df_bin_all.to_csv(outdir / f"frac_{frac_str}_bin_metrics_all_repeats.csv", index=False)
    if not df_repeat_all.empty:
        df_repeat_all.to_csv(outdir / f"frac_{frac_str}_repeat_metrics_all_repeats.csv", index=False)
        df_repeat_summary = ctf.summarize_across_repeats(df_repeat_all)
        if not df_repeat_summary.empty:
            df_repeat_summary.to_csv(outdir / f"frac_{frac_str}_repeat_errorbar_summary.csv", index=False)
    if not df_fail.empty:
        df_fail.to_csv(outdir / f"frac_{frac_str}_repeat_failures.csv", index=False)

    return df_bin_all, df_repeat_all, df_fail


def aggregate_config_performance(
    df_repeat_all: pd.DataFrame,
    df_fail_all: pd.DataFrame,
    expected_repeats: int,
) -> pd.DataFrame:
    """Aggregate repeat-level metrics into config-frac-scope summaries."""

    if df_repeat_all.empty:
        return pd.DataFrame()

    meta_cols = ["stage", "config_id", "d", "q", "lam", "sigma", "T0", "T"]
    if "parent_config_id" in df_repeat_all.columns:
        meta_cols.append("parent_config_id")
    if "source_rank" in df_repeat_all.columns:
        meta_cols.append("source_rank")

    rows: list[dict[str, Any]] = []
    group_cols = meta_cols + ["frac", "scope"]
    for group_vals, g in df_repeat_all.groupby(group_cols, dropna=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        meta = dict(zip(group_cols, group_vals))
        config_fail = df_fail_all
        for key in ["stage", "config_id", "frac"]:
            if key in config_fail.columns:
                config_fail = config_fail[config_fail[key] == meta[key]]
        n_done = int(g["repeat"].nunique())
        n_fail = int(config_fail["repeat"].nunique()) if not config_fail.empty else 0
        rows.append(
            {
                **meta,
                "n_repeats": n_done,
                "expected_repeats": int(expected_repeats),
                "n_failures": n_fail,
                "failure_rate": float(n_fail / expected_repeats) if expected_repeats > 0 else np.nan,
                "complexity": int(meta["d"] + meta["q"]),
                "mean_mse_after": float(g["mse_after"].mean()),
                "sd_mse_after": float(g["mse_after"].std(ddof=1)) if n_done > 1 else np.nan,
                "sem_mse_after": float(g["mse_after"].std(ddof=1) / np.sqrt(n_done)) if n_done > 1 else np.nan,
                "mean_mse_improve": float(g["mse_improve"].mean()),
                "mean_pearson_after": float(g["pearson_after"].mean()),
                "mean_pearson_before": float(g["pearson_before"].mean()),
                "sd_pearson_after": float(g["pearson_after"].std(ddof=1)) if n_done > 1 else np.nan,
                "mean_wd_after": float(g["wd_after"].mean()),
                "sd_wd_after": float(g["wd_after"].std(ddof=1)) if n_done > 1 else np.nan,
                "mse_improve_nonpositive_rate": float((g["mse_improve"] <= 0).mean()),
            }
        )

    out = pd.DataFrame(rows)
    sort_cols = ["stage", "scope", "frac", "mean_mse_after", "mean_wd_after", "complexity"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def rank_parameter_configs(
    df_repeat_all: pd.DataFrame,
    df_fail_all: pd.DataFrame,
    stage: str,
    expected_repeats: int,
    catastrophic_failure_rate: float = 0.30,
    catastrophic_injected_mse_improve: float = -1e-8,
) -> pd.DataFrame:
    """Rank parameter configs from repeat-level summaries."""

    if df_repeat_all.empty:
        return pd.DataFrame(
            columns=[
                "stage",
                "config_id",
                "d",
                "q",
                "lam",
                "sigma",
                "frac",
                "scope",
                "n_repeats",
                "mean_mse_after",
                "sd_mse_after",
                "mean_pearson_after",
                "mean_wd_after",
                "complexity",
                "rank",
            ]
        )

    df_stage_repeat = df_repeat_all[df_repeat_all["stage"] == stage].copy()
    df_stage_fail = df_fail_all[df_fail_all["stage"] == stage].copy() if not df_fail_all.empty else pd.DataFrame()
    agg = aggregate_config_performance(
        df_repeat_all=df_stage_repeat,
        df_fail_all=df_stage_fail,
        expected_repeats=expected_repeats,
    )
    if agg.empty:
        return agg

    if stage == "coarse":
        all_scope = agg[agg["scope"] == "AllCells"].copy()
        inj_scope = agg[agg["scope"] == "InjectedCells"].copy()

        all_group = (
            all_scope.groupby(["config_id", "d", "q", "lam", "sigma", "T0", "T", "complexity"], dropna=False)
            .agg(
                n_repeats=("n_repeats", "sum"),
                mean_mse_after=("mean_mse_after", "mean"),
                sd_mse_after=("mean_mse_after", "std"),
                mean_pearson_after=("mean_pearson_after", "mean"),
                mean_wd_after=("mean_wd_after", "mean"),
                max_failure_rate=("failure_rate", "max"),
                n_fracs=("frac", "nunique"),
            )
            .reset_index()
        )
        inj_group = (
            inj_scope.groupby("config_id", dropna=False)
            .agg(
                worst_injected_mse_improve=("mean_mse_improve", "min"),
                worst_injected_pearson_after=("mean_pearson_after", "min"),
                worst_injected_nonpositive_rate=("mse_improve_nonpositive_rate", "max"),
            )
            .reset_index()
        )
        out = all_group.merge(inj_group, on="config_id", how="left")
        out["stage"] = stage
        out["frac"] = "multi"
        out["scope"] = "AllCells"
        out["catastrophic"] = (
            (out["max_failure_rate"] > catastrophic_failure_rate)
            | (out["worst_injected_mse_improve"] < catastrophic_injected_mse_improve)
        )
        out = out.sort_values(
            by=[
                "catastrophic",
                "mean_mse_after",
                "mean_pearson_after",
                "mean_wd_after",
                "sd_mse_after",
                "complexity",
            ],
            ascending=[True, True, False, True, True, True],
        ).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    fine_scope = agg[agg["scope"] == "AllCells"].copy()
    inj_scope = agg[agg["scope"] == "InjectedCells"].copy()
    inj_scope = inj_scope[
        [
            "config_id",
            "frac",
            "mean_mse_improve",
            "mean_pearson_after",
            "mse_improve_nonpositive_rate",
            "failure_rate",
        ]
    ].rename(
        columns={
            "mean_mse_improve": "injected_mean_mse_improve",
            "mean_pearson_after": "injected_mean_pearson_after",
            "mse_improve_nonpositive_rate": "injected_nonpositive_rate",
            "failure_rate": "injected_failure_rate",
        }
    )
    out = fine_scope.merge(inj_scope, on=["config_id", "frac"], how="left")
    out = out.sort_values(
        by=[
            "mean_mse_after",
            "mean_pearson_after",
            "mean_wd_after",
            "sd_mse_after",
            "complexity",
        ],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _neighbor_values(center: float, grid: list[float]) -> list[float]:
    """Return center and adjacent values from a sorted grid."""

    uniq = sorted({float(x) for x in grid})
    if center not in uniq:
        uniq.append(float(center))
        uniq = sorted(set(uniq))
    idx = uniq.index(float(center))
    keep = {uniq[idx]}
    if idx - 1 >= 0:
        keep.add(uniq[idx - 1])
    if idx + 1 < len(uniq):
        keep.add(uniq[idx + 1])
    return sorted(keep)


def _bounded_rank_candidates(center: int, max_value: int) -> list[int]:
    """Build integer neighbors around a center rank."""

    out = {int(center)}
    for delta in (-1, 1):
        cand = int(center + delta)
        if 1 <= cand <= max_value:
            out.add(cand)
    return sorted(out)


def run_coarse_search(
    da_gt: xr.DataArray,
    configs: list[ParamConfig],
    fracs: list[float],
    n_repeats: int,
    outdir: Path,
    mask_seed_base: int,
    algo_seed_base: int,
    missing_value: float,
    save_imputed_matrix: bool,
    save_format: str,
    compress_level: int,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run coarse search over shared frac and mask schedules."""

    all_bin: list[pd.DataFrame] = []
    all_repeat: list[pd.DataFrame] = []
    all_fail: list[pd.DataFrame] = []

    for idx, config in enumerate(configs, start=1):
        config_dir = outdir / config.config_id
        if verbose:
            print(f"[coarse] {idx}/{len(configs)} config={config.config_id}")
        for frac in fracs:
            frac_dir = config_dir / f"frac_{ctf._format_frac(frac)}"
            df_bin, df_repeat, df_fail = run_repeats_for_frac_allfinite(
                da_gt=da_gt,
                frac=frac,
                n_repeats=n_repeats,
                repeat_start=0,
                mask_seed_base=mask_seed_base,
                algo_seed_base=algo_seed_base,
                outdir=frac_dir,
                config=config,
                missing_value=missing_value,
                save_imputed_matrix=save_imputed_matrix,
                save_format=save_format,
                compress_level=compress_level,
                verbose=verbose,
            )
            if not df_bin.empty:
                all_bin.append(df_bin)
            if not df_repeat.empty:
                all_repeat.append(df_repeat)
            if not df_fail.empty:
                all_fail.append(df_fail)

    df_bin_all = pd.concat(all_bin, ignore_index=True) if all_bin else pd.DataFrame()
    df_repeat_all = pd.concat(all_repeat, ignore_index=True) if all_repeat else pd.DataFrame()
    df_fail_all = pd.concat(all_fail, ignore_index=True) if all_fail else pd.DataFrame()
    leaderboard = rank_parameter_configs(
        df_repeat_all=df_repeat_all,
        df_fail_all=df_fail_all,
        stage="coarse",
        expected_repeats=n_repeats,
    )

    if not df_bin_all.empty:
        df_bin_all.to_csv(outdir / "coarse_bin_metrics_all.csv", index=False)
    if not df_repeat_all.empty:
        df_repeat_all.to_csv(outdir / "coarse_repeat_metrics_all.csv", index=False)
    if not df_fail_all.empty:
        df_fail_all.to_csv(outdir / "coarse_failures_all.csv", index=False)
    if not leaderboard.empty:
        leaderboard.to_csv(outdir / "coarse_leaderboard.csv", index=False)

    detail = aggregate_config_performance(df_repeat_all, df_fail_all, expected_repeats=n_repeats)
    if not detail.empty:
        detail.to_csv(outdir / "coarse_performance_detail.csv", index=False)
    return df_bin_all, df_repeat_all, df_fail_all, leaderboard


def expand_top_configs_for_fine_search(
    coarse_leaderboard: pd.DataFrame,
    top_k: int,
    lam_grid: list[float],
    sigma_grid: list[float],
    da_gt: xr.DataArray,
    T0: int,
    T: int,
) -> list[ParamConfig]:
    """Expand top coarse configs into a local fine-search neighborhood."""

    if coarse_leaderboard.empty:
        return []

    n_participants = int(da_gt.sizes["participant"])
    n_taxa = int(da_gt.sizes["taxon"])
    winners = coarse_leaderboard.sort_values("rank").head(top_k)

    configs: dict[str, ParamConfig] = {}
    for _, row in winners.iterrows():
        r = int(row["d"])
        d_candidates = _bounded_rank_candidates(r, max_value=n_taxa)
        q_candidates = _bounded_rank_candidates(r, max_value=n_participants)
        lam_candidates = _neighbor_values(float(row["lam"]), lam_grid)
        sigma_candidates = _neighbor_values(float(row["sigma"]), sigma_grid)

        for d in d_candidates:
            for q in q_candidates:
                for lam in lam_candidates:
                    for sigma in sigma_candidates:
                        cfg = create_param_config(
                            d=d,
                            q=q,
                            lam=lam,
                            sigma=sigma,
                            T0=T0,
                            T=T,
                            stage="fine",
                            parent_config_id=str(row["config_id"]),
                            source_rank=int(row["rank"]),
                        )
                        configs[cfg.config_id] = cfg
    return sorted(configs.values(), key=lambda x: x.config_id)


def run_fine_search(
    da_gt: xr.DataArray,
    configs: list[ParamConfig],
    frac: float,
    n_repeats: int,
    outdir: Path,
    mask_seed_base: int,
    algo_seed_base: int,
    missing_value: float,
    save_imputed_matrix: bool,
    save_format: str,
    compress_level: int,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run fine search at frac=0.5."""

    all_bin: list[pd.DataFrame] = []
    all_repeat: list[pd.DataFrame] = []
    all_fail: list[pd.DataFrame] = []

    for idx, config in enumerate(configs, start=1):
        config_dir = outdir / config.config_id / f"frac_{ctf._format_frac(frac)}"
        if verbose:
            print(f"[fine] {idx}/{len(configs)} config={config.config_id}")
        df_bin, df_repeat, df_fail = run_repeats_for_frac_allfinite(
            da_gt=da_gt,
            frac=frac,
            n_repeats=n_repeats,
            repeat_start=0,
            mask_seed_base=mask_seed_base,
            algo_seed_base=algo_seed_base,
            outdir=config_dir,
            config=config,
            missing_value=missing_value,
            save_imputed_matrix=save_imputed_matrix,
            save_format=save_format,
            compress_level=compress_level,
            verbose=verbose,
        )
        if not df_bin.empty:
            all_bin.append(df_bin)
        if not df_repeat.empty:
            all_repeat.append(df_repeat)
        if not df_fail.empty:
            all_fail.append(df_fail)

    df_bin_all = pd.concat(all_bin, ignore_index=True) if all_bin else pd.DataFrame()
    df_repeat_all = pd.concat(all_repeat, ignore_index=True) if all_repeat else pd.DataFrame()
    df_fail_all = pd.concat(all_fail, ignore_index=True) if all_fail else pd.DataFrame()
    leaderboard = rank_parameter_configs(
        df_repeat_all=df_repeat_all,
        df_fail_all=df_fail_all,
        stage="fine",
        expected_repeats=n_repeats,
    )

    if not df_bin_all.empty:
        df_bin_all.to_csv(outdir / "fine_bin_metrics_all.csv", index=False)
    if not df_repeat_all.empty:
        df_repeat_all.to_csv(outdir / "fine_repeat_metrics_all.csv", index=False)
    if not df_fail_all.empty:
        df_fail_all.to_csv(outdir / "fine_failures_all.csv", index=False)
    if not leaderboard.empty:
        leaderboard.to_csv(outdir / "fine_leaderboard.csv", index=False)

    detail = aggregate_config_performance(df_repeat_all, df_fail_all, expected_repeats=n_repeats)
    if not detail.empty:
        detail.to_csv(outdir / "fine_performance_detail.csv", index=False)
    return df_bin_all, df_repeat_all, df_fail_all, leaderboard


def run_safety_check(
    df_repeat_all: pd.DataFrame,
    df_fail_all: pd.DataFrame,
    stage: str,
    mse_improve_threshold: float = 0.0,
    pearson_drop_tolerance: float = 0.02,
    nonpositive_rate_threshold: float = 0.50,
    failure_rate_threshold: float = 0.30,
) -> pd.DataFrame:
    """Run veto-style safety checks on ALLCells performance."""

    if df_repeat_all.empty:
        return pd.DataFrame()

    df_stage = df_repeat_all[df_repeat_all["stage"] == stage].copy()
    df_stage_fail = df_fail_all[df_fail_all["stage"] == stage].copy() if not df_fail_all.empty else pd.DataFrame()
    inj = df_stage[df_stage["scope"] == "AllCells"].copy()
    if inj.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for config_id, g in inj.groupby("config_id", dropna=False):
        fail_cfg = df_stage_fail[df_stage_fail["config_id"] == config_id] if not df_stage_fail.empty else pd.DataFrame()
        n_done = int(g["repeat"].nunique())
        n_fail = int(fail_cfg["repeat"].nunique()) if not fail_cfg.empty else 0
        denom = n_done + n_fail
        mean_mse_improve = float(g["mse_improve"].mean())
        mean_pear_after = float(g["pearson_after"].mean())
        mean_pear_before = float(g["pearson_before"].mean())
        nonpositive_rate = float((g["mse_improve"] <= 0).mean())
        failure_rate = float(n_fail / denom) if denom > 0 else 0.0
        recommend = (
            (mean_mse_improve > mse_improve_threshold)
            and (mean_pear_after >= (mean_pear_before - pearson_drop_tolerance))
            and (nonpositive_rate <= nonpositive_rate_threshold)
            and (failure_rate <= failure_rate_threshold)
        )
        first = g.iloc[0]
        rows.append(
            {
                "stage": stage,
                "config_id": config_id,
                "d": int(first["d"]),
                "q": int(first["q"]),
                "lam": float(first["lam"]),
                "sigma": float(first["sigma"]),
                "frac": float(first["frac"]),
                "n_repeats": n_done,
                "mean_mse_improve": mean_mse_improve,
                "mean_pearson_before": mean_pear_before,
                "mean_pearson_after": mean_pear_after,
                "nonpositive_mse_improve_rate": nonpositive_rate,
                "failure_rate": failure_rate,
                "recommend": bool(recommend),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["recommend", "mean_mse_improve", "mean_pearson_after"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def save_final_reports(
    reports_dir: Path,
    coarse_leaderboard: pd.DataFrame,
    fine_leaderboard: pd.DataFrame,
    safety_df: pd.DataFrame,
    best_config: ParamConfig | None,
    selection_summary: dict[str, Any],
) -> None:
    """Persist final ranking reports and best parameter selection."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    if not coarse_leaderboard.empty:
        coarse_leaderboard.to_csv(reports_dir / "coarse_leaderboard.csv", index=False)
    if not fine_leaderboard.empty:
        fine_leaderboard.to_csv(reports_dir / "fine_leaderboard.csv", index=False)
    if not safety_df.empty:
        safety_df.to_csv(reports_dir / "safety_check.csv", index=False)
    if not fine_leaderboard.empty:
        final_compare = fine_leaderboard.copy()
        if not safety_df.empty:
            final_compare = final_compare.merge(
                safety_df[
                    [
                        "config_id",
                        "recommend",
                        "mean_mse_improve",
                        "mean_pearson_before",
                        "mean_pearson_after",
                        "nonpositive_mse_improve_rate",
                        "failure_rate",
                    ]
                ],
                on="config_id",
                how="left",
                suffixes=("", "_safety"),
            )
        final_compare.to_csv(reports_dir / "final_candidate_comparison.csv", index=False)

    payload: dict[str, Any] = {
        "selected": best_config is not None,
        "selection_summary": selection_summary,
    }
    if best_config is not None:
        payload.update(
            {
                "d": best_config.d,
                "q": best_config.q,
                "lam": best_config.lam,
                "sigma": best_config.sigma,
                "T0": best_config.T0,
                "T": best_config.T,
                "config_id": best_config.config_id,
                "stage": best_config.stage,
            }
        )
    (reports_dir / "best_params.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def choose_final_config(
    fine_leaderboard: pd.DataFrame,
    fine_configs: list[ParamConfig],
    safety_df: pd.DataFrame,
) -> tuple[ParamConfig | None, dict[str, Any]]:
    """Choose the best fine-stage config, subject to safety veto."""

    if fine_leaderboard.empty:
        return None, {"reason": "fine leaderboard is empty"}

    safe_ids = set()
    if not safety_df.empty:
        safe_ids = set(safety_df.loc[safety_df["recommend"], "config_id"].astype(str))

    config_map = {cfg.config_id: cfg for cfg in fine_configs}
    for _, row in fine_leaderboard.sort_values("rank").iterrows():
        cfg_id = str(row["config_id"])
        if safe_ids and cfg_id not in safe_ids:
            continue
        cfg = config_map.get(cfg_id)
        if cfg is None:
            continue
        return cfg, {
            "reason": "top fine config passing InjectedCells safety veto",
            "leaderboard_rank": int(row["rank"]),
            "mean_mse_after": float(row["mean_mse_after"]),
            "mean_pearson_after": float(row["mean_pearson_after"]),
            "mean_wd_after": float(row["mean_wd_after"]),
            "complexity": int(row["complexity"]),
        }

    top_row = fine_leaderboard.sort_values("rank").iloc[0]
    top_cfg = config_map.get(str(top_row["config_id"]))
    return top_cfg, {
        "reason": "no fine config passed safety veto; fallback to top fine config with warning",
        "leaderboard_rank": int(top_row["rank"]),
        "mean_mse_after": float(top_row["mean_mse_after"]),
        "mean_pearson_after": float(top_row["mean_pearson_after"]),
        "mean_wd_after": float(top_row["mean_wd_after"]),
        "complexity": int(top_row["complexity"]),
        "warning": "InjectedCells safety criteria were not satisfied by any fine-stage candidate.",
    }


def build_truth(args: argparse.Namespace, truth_dir: Path) -> xr.DataArray:
    """Load or build the fixed truth tensor once for the whole tuning workflow."""

    truth_path = Path(args.truth_path) if args.truth_path else truth_dir / "sim2_truth.nc"
    return ctf.build_truth_once(
        truth_path=truth_path,
        force_rebuild=bool(args.force_rebuild_truth),
        save_format=args.save_format,
        compress_level=args.compress_level,
        rank=args.ctf_rank,
        method=args.ctf_method,
        ctf_seed=args.ctf_seed,
        ctf_verbose=bool(args.ctf_verbose),
    )


def build_coarse_configs(args: argparse.Namespace) -> list[ParamConfig]:
    """Build coarse-stage configs with d=q=r."""

    configs: list[ParamConfig] = []
    for r in args.r_grid:
        for lam in args.lam_grid:
            for sigma in args.sigma_grid:
                configs.append(
                    create_param_config(
                        d=r,
                        q=r,
                        lam=lam,
                        sigma=sigma,
                        T0=args.T0,
                        T=args.T,
                        stage="coarse",
                    )
                )
    return configs


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments."""

    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts") / "tuning")
    parser.add_argument("--truth-path", type=Path, default=None)
    parser.add_argument("--force-rebuild-truth", action="store_true")
    parser.add_argument("--ctf-rank", type=int, default=3)
    parser.add_argument("--ctf-method", type=str, default="fallback", choices=["auto", "gemelli", "fallback"])
    parser.add_argument("--ctf-seed", type=int, default=0)
    parser.add_argument("--ctf-verbose", action="store_true")

    parser.add_argument("--coarse-fracs", type=parse_float_grid, default=[0.2, 0.5, 0.8])
    parser.add_argument("--n-repeats-coarse", type=int, default=20)
    parser.add_argument("--n-repeats-fine", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=8)

    parser.add_argument("--r-grid", type=parse_int_grid, default=[2, 4, 6, 8, 10])
    parser.add_argument("--lam-grid", type=parse_float_grid, default=[1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 2.0, 5.0])
    parser.add_argument("--sigma-grid", type=parse_float_grid, default=[1.0, 2.0, 4.0, 8.0, 12.0, 16.0])

    parser.add_argument("--T0", type=int, default=30)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--mask-seed-base", type=int, default=123)
    parser.add_argument("--algo-seed-base", type=int, default=0)
    parser.add_argument("--save-imputed-matrix", action="store_true")
    parser.add_argument("--save-format", type=str, default="netcdf", choices=["netcdf", "zarr"])
    parser.add_argument("--compress-level", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--fine-frac", type=float, default=0.5)
    parser.add_argument("--safety-mse-improve-threshold", type=float, default=0.0)
    parser.add_argument("--safety-pearson-drop-tolerance", type=float, default=0.02)
    parser.add_argument("--safety-nonpositive-rate-threshold", type=float, default=0.50)
    parser.add_argument("--safety-failure-rate-threshold", type=float, default=0.30)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Coarse-to-fine tuning workflow for penalized_2d2_pca with explicit artificial-missing masks."
    )
    add_common_args(parser)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    truth_dir = artifacts_dir / "truth"
    coarse_dir = artifacts_dir / "coarse"
    fine_dir = artifacts_dir / "fine"
    reports_dir = artifacts_dir / "reports"
    for path in [truth_dir, coarse_dir, fine_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    da_gt = build_truth(args, truth_dir=truth_dir)
    da_gt.name = "y_true"
    missing_value = float(np.log10(ctf.eps))

    coarse_configs = build_coarse_configs(args)
    _, _, _, coarse_leaderboard = run_coarse_search(
        da_gt=da_gt,
        configs=coarse_configs,
        fracs=[float(x) for x in args.coarse_fracs],
        n_repeats=int(args.n_repeats_coarse),
        outdir=coarse_dir,
        mask_seed_base=int(args.mask_seed_base),
        algo_seed_base=int(args.algo_seed_base),
        missing_value=missing_value,
        save_imputed_matrix=bool(args.save_imputed_matrix),
        save_format=args.save_format,
        compress_level=int(args.compress_level),
        verbose=bool(args.verbose),
    )

    fine_configs = expand_top_configs_for_fine_search(
        coarse_leaderboard=coarse_leaderboard,
        top_k=int(args.top_k),
        lam_grid=[float(x) for x in args.lam_grid],
        sigma_grid=[float(x) for x in args.sigma_grid],
        da_gt=da_gt,
        T0=int(args.T0),
        T=int(args.T),
    )
    _, fine_repeat, fine_fail, fine_leaderboard = run_fine_search(
        da_gt=da_gt,
        configs=fine_configs,
        frac=float(args.fine_frac),
        n_repeats=int(args.n_repeats_fine),
        outdir=fine_dir,
        mask_seed_base=int(args.mask_seed_base),
        algo_seed_base=int(args.algo_seed_base),
        missing_value=missing_value,
        save_imputed_matrix=bool(args.save_imputed_matrix),
        save_format=args.save_format,
        compress_level=int(args.compress_level),
        verbose=bool(args.verbose),
    )

    safety_df = run_safety_check(
        df_repeat_all=fine_repeat,
        df_fail_all=fine_fail,
        stage="fine",
        mse_improve_threshold=float(args.safety_mse_improve_threshold),
        pearson_drop_tolerance=float(args.safety_pearson_drop_tolerance),
        nonpositive_rate_threshold=float(args.safety_nonpositive_rate_threshold),
        failure_rate_threshold=float(args.safety_failure_rate_threshold),
    )
    best_config, selection_summary = choose_final_config(
        fine_leaderboard=fine_leaderboard,
        fine_configs=fine_configs,
        safety_df=safety_df,
    )

    save_final_reports(
        reports_dir=reports_dir,
        coarse_leaderboard=coarse_leaderboard,
        fine_leaderboard=fine_leaderboard,
        safety_df=safety_df,
        best_config=best_config,
        selection_summary=selection_summary,
    )

    if args.verbose:
        print("[done] tuning workflow completed")
        if best_config is not None:
            print(json.dumps({**asdict(best_config), **selection_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
