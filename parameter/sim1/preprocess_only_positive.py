import argparse
import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import wasserstein_distance


METRIC_COLS = [
    "mse_before",
    "mse_after",
    "mse_improve",
    "pearson_before",
    "pearson_after",
    "pearson_improve",
    "wd_before",
    "wd_after",
    "wd_improve",
]

PARAM_COLS = ["d", "q", "lam", "sigma"]
TRUTH_PARAM_COLS = ["truth_d", "truth_q", "truth_lam", "truth_sigma"]


def week_to_bin(week: int) -> Optional[str]:
    if week == 0:
        return "0"
    if 1 <= week <= 36:
        start = ((week - 1) // 4) * 4 + 1
        end = start + 3
        return f"{start}-{end}"
    return None


def _bin_to_time_value(x) -> float:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "0":
        return 0.0
    if "-" in s:
        a, b = s.split("-", 1)
        return 0.5 * (float(a) + float(b))
    return float(s)


def _bin_key(x):
    return _bin_to_time_value(x)


def _orthonormalize(mat: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(mat, mode="reduced")
    return q


def _top_eigvecs(mat: np.ndarray, k: int) -> np.ndarray:
    w, v = np.linalg.eigh(mat)
    idx = np.argsort(w)[::-1][:k]
    return v[:, idx]


def _build_time_weights(t: np.ndarray, sigma: float) -> np.ndarray:
    dt2 = (t[:, None] - t[None, :]) ** 2
    w = np.exp(-dt2 / (2.0 * sigma**2))
    np.fill_diagonal(w, 0.0)
    return w


def build_observed_mask(
    da_input: xr.DataArray,
    zero_value: float,
    mode: str,
    replace_mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    vals = da_input.values
    finite = np.isfinite(vals)

    if mode == "positive_only":
        obs = finite & (vals > zero_value)
    elif mode == "all_finite_except_replace":
        if replace_mask is None:
            raise ValueError("replace_mask is required when mode='all_finite_except_replace'")
        obs = finite & (~replace_mask.values.astype(bool))
    elif mode == "all_finite":
        obs = finite
    else:
        raise ValueError(f"Unsupported observed mode: {mode}")

    return xr.DataArray(obs, coords=da_input.coords, dims=da_input.dims, name="observed_mask")


def penalized_2d2_pca(
    da: xr.DataArray,
    time=None,
    d: int = 1,
    q: int = 1,
    lam: float = 0.1,
    sigma: float = 4.0,
    t0: int = 30,
    t: int = 30,
    seed: int = 0,
    observed_mask: Optional[xr.DataArray] = None,
    verbose: bool = False,
):
    rng = np.random.default_rng(seed)

    if time is None:
        time = da.coords["bin"].values
    time = np.array(time)
    m_time = len(time)

    a_list, m_list = [], []
    for w in time:
        ai = da.sel(bin=w).values.astype(float)
        if observed_mask is None:
            mi = np.isfinite(ai)
        else:
            mi = observed_mask.sel(bin=w).values.astype(bool)
        a_list.append(ai)
        m_list.append(mi)

    a = np.stack(a_list, axis=0)
    obs = np.stack(m_list, axis=0)
    _, m, n = a.shape

    tvals = np.array([_bin_to_time_value(x) for x in time], dtype=float)
    w_time = _build_time_weights(tvals, sigma=sigma)

    abar = np.where(obs, a, 0.0).sum(axis=0) / np.maximum(obs.sum(axis=0), 1.0)
    ac = np.where(obs, a - abar[None, :, :], 0.0)

    z = _orthonormalize(rng.standard_normal((m, q)))

    if verbose:
        print(f"[Phase 1] lambda=0, iter={t0}, M={m_time}, m={m}, n={n}, q={q}, d={d}")
    for _ in range(t0):
        zzt = z @ z.T
        astar0 = np.zeros((n, n), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astar0 += bi.T @ zzt @ bi
        astar0 /= m_time
        x = _top_eigvecs(astar0, d)

        xxt = x @ x.T
        astarstar0 = np.zeros((m, m), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astarstar0 += bi @ xxt @ bi.T
        astarstar0 /= m_time
        z = _top_eigvecs(astarstar0, q)

    if verbose:
        print(f"[Phase 2] lambda={lam}, sigma={sigma}, iter={t}")
    for _ in range(t):
        zzt = z @ z.T
        astar = np.zeros((n, n), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astar += bi.T @ zzt @ bi
        astar /= m_time

        pen = np.zeros((n, n), dtype=float)
        for i in range(m_time):
            for j in range(m_time):
                wij = w_time[i, j]
                if wij == 0.0:
                    continue
                oij = obs[i] & obs[j]
                if not np.any(oij):
                    continue
                dij = np.where(oij, a[i] - a[j], 0.0)
                pen += wij * (dij.T @ zzt @ dij)
        astar = astar - lam * pen
        x = _top_eigvecs(astar, d)

        xxt = x @ x.T
        astarstar = np.zeros((m, m), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astarstar += bi @ xxt @ bi.T
        astarstar /= m_time

        pen2 = np.zeros((m, m), dtype=float)
        for i in range(m_time):
            for j in range(m_time):
                wij = w_time[i, j]
                if wij == 0.0:
                    continue
                oij = obs[i] & obs[j]
                if not np.any(oij):
                    continue
                dij = np.where(oij, a[i] - a[j], 0.0)
                pen2 += wij * (dij @ xxt @ dij.T)
        astarstar = astarstar - lam * pen2
        z = _top_eigvecs(astarstar, q)

    c = np.zeros((m_time, q, d), dtype=float)
    for i in range(m_time):
        ai_eff = np.where(obs[i], a[i], abar)
        c[i] = z.T @ ai_eff @ x

    meta = {"bin": time, "t": tvals, "W": w_time, "Abar": abar}
    return x, z, c, meta


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _safe_wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return np.nan
    return float(wasserstein_distance(x, y))


def evaluate_per_bin(
    da_true: xr.DataArray,
    da_before: xr.DataArray,
    da_after: xr.DataArray,
    bins: Iterable,
    eval_mask: Optional[xr.DataArray] = None,
    bin_dim: str = "bin",
) -> pd.DataFrame:
    rows = []
    for b in bins:
        yt = da_true.sel({bin_dim: b}).values.ravel()
        yb = da_before.sel({bin_dim: b}).values.ravel()
        ya = da_after.sel({bin_dim: b}).values.ravel()
        finite = np.isfinite(yt) & np.isfinite(yb) & np.isfinite(ya)
        if eval_mask is not None:
            mk = eval_mask.sel({bin_dim: b}).values.ravel().astype(bool)
            use = finite & mk
        else:
            use = finite

        yt_u, yb_u, ya_u = yt[use], yb[use], ya[use]
        n_eval = int(use.sum())
        if n_eval == 0:
            mse_before = mse_after = np.nan
            pear_before = pear_after = np.nan
            wd_before = wd_after = np.nan
        else:
            mse_before = float(np.mean((yb_u - yt_u) ** 2))
            mse_after = float(np.mean((ya_u - yt_u) ** 2))
            pear_before = _safe_pearson(yb_u, yt_u)
            pear_after = _safe_pearson(ya_u, yt_u)
            wd_before = _safe_wasserstein(yb_u, yt_u)
            wd_after = _safe_wasserstein(ya_u, yt_u)

        rows.append(
            {
                "bin": b,
                "n_eval": n_eval,
                "mse_before": mse_before,
                "mse_after": mse_after,
                "mse_improve": mse_before - mse_after,
                "pearson_before": pear_before,
                "pearson_after": pear_after,
                "pearson_improve": pear_after - pear_before,
                "wd_before": wd_before,
                "wd_after": wd_after,
                "wd_improve": wd_before - wd_after,
            }
        )
    return pd.DataFrame(rows)


def _weighted_avg(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.average(v[m], weights=w[m]))


def build_weighted_summary(df_per_bin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, g in df_per_bin.groupby("scope"):
        w = g["n_eval"].to_numpy(dtype=float)
        row = {"scope": scope, "n_eval": int(np.nansum(w))}
        for c in METRIC_COLS:
            row[c] = _weighted_avg(g[c], w)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scope").reset_index(drop=True)


def make_random_mask(
    da_ref: xr.DataArray,
    bins,
    frac: float,
    seed: int,
    zero_value: float,
    eligible_mode: str = "all_finite",
) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    da_sub = da_ref.sel(bin=bins)
    vals = da_sub.values
    finite = np.isfinite(vals)

    if eligible_mode == "positive_only":
        eligible = finite & (vals > zero_value)
    elif eligible_mode == "all_finite":
        eligible = finite
    else:
        raise ValueError(f"Unsupported eligible_mode: {eligible_mode}")

    idx = np.flatnonzero(eligible)
    if idx.size == 0:
        mask = np.zeros(vals.shape, dtype=bool)
    else:
        n_pick = int(np.round(frac * idx.size))
        if frac > 0 and n_pick == 0:
            n_pick = 1
        n_pick = min(n_pick, idx.size)
        pick = rng.choice(idx, size=n_pick, replace=False) if n_pick > 0 else np.array([], dtype=int)
        mask = np.zeros(vals.size, dtype=bool)
        mask[pick] = True
        mask = mask.reshape(vals.shape)

    return xr.DataArray(mask, coords=da_sub.coords, dims=da_sub.dims, name="rand_mask_to_impute")


def impute_by_mask(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    model_seed: int,
    pca_params: Dict,
    zero_value: float,
    observed_mode: str = "all_finite_except_replace",
    verbose: bool = False,
) -> Tuple[xr.DataArray, np.ndarray]:
    bins_use = np.array(da_input.coords["bin"].values)

    observed_mask = build_observed_mask(
        da_input=da_input,
        zero_value=zero_value,
        mode=observed_mode,
        replace_mask=replace_mask,
    )

    x, z, c, meta = penalized_2d2_pca(
        da=da_input,
        time=bins_use,
        d=pca_params["d"],
        q=pca_params["q"],
        lam=pca_params["lam"],
        sigma=pca_params["sigma"],
        t0=pca_params["T0"],
        t=pca_params["T"],
        seed=model_seed,
        observed_mask=observed_mask,
        verbose=verbose,
    )

    bins_trained = np.array(meta["bin"])
    abar = meta["Abar"]
    da_imp = da_input.copy(deep=True)

    for ti, b in enumerate(bins_trained):
        ai = da_imp.sel(bin=b).values
        ahat = abar + (z @ c[ti] @ x.T)
        mk = replace_mask.sel(bin=b).values.astype(bool)
        if np.any(mk):
            ai_new = ai.copy()
            ai_new[mk] = ahat[mk]
            da_imp.loc[dict(bin=b)] = ai_new

    da_imp.name = "log_abundance_imputed"
    return da_imp, bins_trained


def run_one_replication(
    da_truth: xr.DataArray,
    simulation: str,
    frac: float,
    replicate: int,
    mask_seed: int,
    model_seed: int,
    zero_value: float,
    pca_params: Dict,
    second_observed_mode: str = "all_finite_except_replace",
    mask_eligible_mode: str = "all_finite",
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bins_use = np.array(da_truth.coords["bin"].values)

    da_rand_mask = make_random_mask(
        da_truth,
        bins_use,
        frac=frac,
        seed=mask_seed,
        zero_value=zero_value,
        eligible_mode=mask_eligible_mode,
    )

    da_before = da_truth.copy(deep=True)
    vals = da_before.sel(bin=bins_use).values.copy()
    vals[da_rand_mask.values.astype(bool)] = zero_value
    da_before.loc[dict(bin=bins_use)] = vals
    da_before.name = "log_abundance_zero_inflated"

    da_after, bins_trained = impute_by_mask(
        da_input=da_before,
        replace_mask=da_rand_mask,
        model_seed=model_seed,
        pca_params=pca_params,
        zero_value=zero_value,
        observed_mode=second_observed_mode,
        verbose=verbose,
    )

    df_masked = evaluate_per_bin(
        da_true=da_truth,
        da_before=da_before,
        da_after=da_after,
        bins=bins_trained,
        eval_mask=da_rand_mask,
        bin_dim="bin",
    )
    df_masked["scope"] = "MaskedCells"

    df_all = evaluate_per_bin(
        da_true=da_truth,
        da_before=da_before,
        da_after=da_after,
        bins=bins_trained,
        eval_mask=None,
        bin_dim="bin",
    )
    df_all["scope"] = "AllCells"

    df_per_bin = pd.concat([df_masked, df_all], ignore_index=True)
    df_per_bin["bin_order"] = df_per_bin["bin"].map(_bin_key)
    df_per_bin = df_per_bin.sort_values(["scope", "bin_order"]).drop(columns=["bin_order"])
    df_per_bin["simulation"] = simulation
    df_per_bin["frac"] = float(frac)
    df_per_bin["replicate"] = int(replicate)
    df_per_bin["mask_seed"] = int(mask_seed)
    df_per_bin["model_seed"] = int(model_seed)

    df_summary = build_weighted_summary(df_per_bin)
    df_summary["simulation"] = simulation
    df_summary["frac"] = float(frac)
    df_summary["replicate"] = int(replicate)
    df_summary["mask_seed"] = int(mask_seed)
    df_summary["model_seed"] = int(model_seed)

    ordered = ["simulation", "frac", "replicate", "scope", "mask_seed", "model_seed", "n_eval"] + METRIC_COLS
    df_summary = df_summary[ordered]
    return df_per_bin, df_summary


def run_repeated_experiments(
    da_truth: xr.DataArray,
    simulation: str,
    frac: float,
    n_repeats: int,
    seed_base: int,
    zero_value: float,
    pca_params: Dict,
    second_observed_mode: str = "all_finite_except_replace",
    mask_eligible_mode: str = "all_finite",
    collect_per_bin: bool = True,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    rep_rows = []
    bin_rows = []
    for replicate in range(n_repeats):
        mask_seed = seed_base + replicate
        model_seed = seed_base + 10000 + replicate
        df_per_bin, df_summary = run_one_replication(
            da_truth=da_truth,
            simulation=simulation,
            frac=frac,
            replicate=replicate,
            mask_seed=mask_seed,
            model_seed=model_seed,
            zero_value=zero_value,
            pca_params=pca_params,
            second_observed_mode=second_observed_mode,
            mask_eligible_mode=mask_eligible_mode,
            verbose=verbose,
        )
        rep_rows.append(df_summary)
        if collect_per_bin:
            bin_rows.append(df_per_bin)

    df_reps = pd.concat(rep_rows, ignore_index=True)
    df_bins = pd.concat(bin_rows, ignore_index=True) if collect_per_bin else None
    return df_reps, df_bins


def summarize_replicates(df_replicates: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["simulation", "frac", "scope", "replicate"]
    long_df = df_replicates[id_cols + METRIC_COLS].melt(
        id_vars=id_cols,
        value_vars=METRIC_COLS,
        var_name="metric",
        value_name="value",
    )
    g = long_df.groupby(["simulation", "frac", "scope", "metric"], as_index=False)
    summary = g.agg(
        mean=("value", "mean"),
        sd=("value", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan),
        n_repeats=("value", lambda x: int(np.sum(np.isfinite(x)))),
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n_repeats"].replace(0, np.nan))
    summary["ci_low"] = summary["mean"] - 1.96 * summary["se"]
    summary["ci_high"] = summary["mean"] + 1.96 * summary["se"]
    return summary.sort_values(["simulation", "frac", "scope", "metric"]).reset_index(drop=True)


def load_base_tensor_with_missing_mask(tax_path: str, meta_path: str, eps: float) -> Tuple[xr.DataArray, xr.DataArray]:
    tax = pd.read_csv(tax_path, sep="\t")
    tax = tax.rename(columns={tax.columns[0]: "taxon"}).set_index("taxon")
    meta = pd.read_excel(meta_path)

    tax.columns = tax.columns.str.replace(r"_profile$", "", regex=True)
    meta_mgx = (
        meta.loc[meta["data_type"].eq("metagenomics"), ["External ID", "Participant ID", "week_num"]]
        .dropna(subset=["External ID", "Participant ID", "week_num"])
        .rename(columns={"External ID": "external_id", "Participant ID": "participant", "week_num": "week"})
    )
    meta_mgx["week"] = meta_mgx["week"].astype(int)
    meta_mgx["participant"] = meta_mgx["participant"].astype(str)

    idx = tax.index.astype(str)
    has_species = idx.str.contains("s__", na=False)
    not_euk = ~idx.str.contains("k__Eukaryota", na=False)
    has_unclassified = idx.str.contains("unclassified", na=False)
    has_cag = idx.str.contains(r"CAG[_-]?\d+", regex=True, na=False)
    bad_unclassified = has_unclassified & (~has_cag)
    tax = tax.loc[has_species & not_euk & (~bad_unclassified)].copy()

    common = sorted(set(tax.columns).intersection(set(meta_mgx["external_id"])))
    tax = tax[common]
    meta_mgx = meta_mgx.drop_duplicates("external_id").set_index("external_id").loc[common]

    x = tax.T
    x.index.name = "external_id"
    x = x.join(meta_mgx, how="left")
    taxa = tax.index.astype(str).tolist()

    x_taxa = x[taxa].copy()
    x_taxa["participant"] = x["participant"].astype(str)
    x_taxa["week"] = x["week"].astype(int)
    x_taxa["bin"] = x_taxa["week"].map(week_to_bin)
    x_taxa = x_taxa.dropna(subset=["bin"])
    x_taxa = x_taxa.set_index(["participant", "bin"])[taxa]
    x_agg = x_taxa.groupby(level=["participant", "bin"]).mean()

    participants = sorted(x_agg.index.get_level_values("participant").unique().tolist())
    bins = ["0"] + [f"{s}-{s+3}" for s in range(1, 37, 4)]
    full_index = pd.MultiIndex.from_product([participants, bins], names=["participant", "bin"])
    x_full = x_agg.reindex(full_index)

    init_missing_mask_df = x_full.isna()
    x_full_0 = x_full.fillna(0.0)
    x_full_log = np.log10(x_full_0 + eps)

    arr = x_full_log.to_numpy().reshape(len(participants), len(bins), len(taxa))
    tensor = np.transpose(arr, (0, 2, 1))
    da = xr.DataArray(
        tensor,
        dims=("participant", "taxon", "bin"),
        coords={"participant": participants, "taxon": taxa, "bin": bins},
        name="abundance_binned_log",
    )

    mask_arr = init_missing_mask_df.to_numpy().reshape(len(participants), len(bins), len(taxa))
    mask_tensor = np.transpose(mask_arr, (0, 2, 1))
    init_missing_mask = xr.DataArray(
        mask_tensor.astype(bool),
        dims=("participant", "taxon", "bin"),
        coords={"participant": participants, "taxon": taxa, "bin": bins},
        name="initial_missing_mask",
    )
    return da, init_missing_mask


def prepare_truth_data_for_sim1(
    da_base: xr.DataArray,
    init_missing_mask: xr.DataArray,
    zero_value: float,
    pca_params: Dict,
    seed_base: int,
    truth_observed_mode: str = "all_finite_except_replace",
    verbose: bool = False,
) -> xr.DataArray:
    da_truth, _ = impute_by_mask(
        da_input=da_base.copy(deep=True),
        replace_mask=init_missing_mask,
        model_seed=seed_base + 500000,
        pca_params=pca_params,
        zero_value=zero_value,
        observed_mode=truth_observed_mode,
        verbose=verbose,
    )
    da_truth.name = "da_truth_sim1"
    return da_truth


def _build_pca_params(d: int, q: int, lam: float, sigma: float, t0: int, t: int) -> Dict:
    return {
        "d": int(d),
        "q": int(q),
        "lam": float(lam),
        "sigma": float(sigma),
        "T0": int(t0),
        "T": int(t),
    }


def _parse_grid_arg(values, cast):
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [cast(v) for v in values]


def _build_truth_pca_params(args) -> Dict:
    return _build_pca_params(
        d=args.truth_d if args.truth_d is not None else args.d,
        q=args.truth_q if args.truth_q is not None else args.q,
        lam=args.truth_lam if args.truth_lam is not None else args.lam,
        sigma=args.truth_sigma if args.truth_sigma is not None else args.sigma,
        t0=args.t0,
        t=args.t,
    )


def _build_fit_param_grid(args) -> List[Dict]:
    grid_d = _parse_grid_arg(args.grid_d, int) or [args.d]
    grid_q = _parse_grid_arg(args.grid_q, int) or [args.q]
    grid_lam = _parse_grid_arg(args.grid_lam, float) or [args.lam]
    grid_sigma = _parse_grid_arg(args.grid_sigma, float) or [args.sigma]

    combos = []
    for d, q, lam, sigma in itertools.product(grid_d, grid_q, grid_lam, grid_sigma):
        combos.append(_build_pca_params(d=d, q=q, lam=lam, sigma=sigma, t0=args.t0, t=args.t))
    return combos


def _append_param_metadata(
    df: pd.DataFrame,
    fit_pca_params: Dict,
    truth_pca_params: Dict,
) -> pd.DataFrame:
    out = df.copy()
    metadata = {
        "d": fit_pca_params["d"],
        "q": fit_pca_params["q"],
        "lam": fit_pca_params["lam"],
        "sigma": fit_pca_params["sigma"],
        "truth_d": truth_pca_params["d"],
        "truth_q": truth_pca_params["q"],
        "truth_lam": truth_pca_params["lam"],
        "truth_sigma": truth_pca_params["sigma"],
    }
    for col, value in metadata.items():
        out[col] = value
    return out


def _format_param_value(value) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return np.format_float_positional(float(value), trim="-", fractional=False)
    return str(value)


def _build_param_tag(fit_pca_params: Dict, truth_pca_params: Dict) -> str:
    parts = [
        f"d{_format_param_value(fit_pca_params['d'])}",
        f"q{_format_param_value(fit_pca_params['q'])}",
        f"lam{_format_param_value(fit_pca_params['lam'])}",
        f"sigma{_format_param_value(fit_pca_params['sigma'])}",
    ]
    if any(fit_pca_params[k] != truth_pca_params[k] for k in ["d", "q", "lam", "sigma"]):
        parts.extend(
            [
                f"truthd{_format_param_value(truth_pca_params['d'])}",
                f"truthq{_format_param_value(truth_pca_params['q'])}",
                f"truthlam{_format_param_value(truth_pca_params['lam'])}",
                f"truthsigma{_format_param_value(truth_pca_params['sigma'])}",
            ]
        )
    return "_".join(parts)


def _build_output_name(prefix: str, param_tag: str, suffix: str) -> str:
    return f"{prefix}_sim1_{param_tag}{suffix}"


def build_tuning_summary(summary: pd.DataFrame) -> pd.DataFrame:
    keep_metrics = [
        "mse_after",
        "pearson_after",
        "wd_after",
        "mse_improve",
        "pearson_improve",
        "wd_improve",
    ]
    stat_cols = ["mean", "sd", "se", "ci_low", "ci_high"]
    id_cols = ["simulation", "frac", "scope"] + PARAM_COLS + TRUTH_PARAM_COLS

    base = summary[summary["metric"].isin(keep_metrics)].copy()
    wide = base.pivot_table(index=id_cols, columns="metric", values=stat_cols, aggfunc="first")
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    wide = wide.reset_index()

    sort_scope = np.where(wide["scope"].eq("MaskedCells"), 0, 1)
    wide = (
        wide.assign(_scope_rank=sort_scope)
        .sort_values(["_scope_rank", "frac", "mse_after_mean", "pearson_after_mean"], ascending=[True, True, True, False])
        .drop(columns="_scope_rank")
        .reset_index(drop=True)
    )
    return wide


def _safe_output_path(outdir: Path, name: str) -> Path:
    target = outdir / name
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return outdir / f"{stem}_{ts}{suffix}"


def run_sim1_pipeline(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    eps = args.eps
    zero_value = np.log10(eps)
    truth_pca_params = _build_truth_pca_params(args)
    fit_param_grid = _build_fit_param_grid(args)

    da_base, init_missing_mask = load_base_tensor_with_missing_mask(args.tax_path, args.meta_path, eps=eps)

    da_truth = prepare_truth_data_for_sim1(
        da_base=da_base,
        init_missing_mask=init_missing_mask,
        zero_value=zero_value,
        pca_params=truth_pca_params,
        seed_base=args.seed_base,
        truth_observed_mode=args.truth_observed_mode,
        verbose=args.verbose,
    )
    truth_tag = _build_param_tag(truth_pca_params, truth_pca_params)
    truth_path = _safe_output_path(outdir, _build_output_name("truth_tensor", truth_tag, ".nc"))
    da_truth.to_netcdf(truth_path)
    print(f"Saved truth tensor: {truth_path}")

    for fit_pca_params in fit_param_grid:
        rep_tables = []
        bin_tables = []
        for frac in args.fracs:
            frac_seed_base = args.seed_base + int(round(frac * 1_000_000))
            df_reps, df_bins = run_repeated_experiments(
                da_truth=da_truth,
                simulation="sim1",
                frac=float(frac),
                n_repeats=args.n_repeats,
                seed_base=frac_seed_base,
                zero_value=zero_value,
                pca_params=fit_pca_params,
                second_observed_mode=args.second_observed_mode,
                mask_eligible_mode=args.mask_eligible_mode,
                collect_per_bin=True,
                verbose=args.verbose,
            )
            rep_tables.append(_append_param_metadata(df_reps, fit_pca_params, truth_pca_params))
            if df_bins is not None:
                bin_tables.append(_append_param_metadata(df_bins, fit_pca_params, truth_pca_params))

        all_reps = pd.concat(rep_tables, ignore_index=True)
        all_reps = all_reps.sort_values(
            ["simulation", "frac", "d", "q", "lam", "sigma", "replicate", "scope"]
        ).reset_index(drop=True)
        summary = summarize_replicates(all_reps)
        summary = _append_param_metadata(summary, fit_pca_params, truth_pca_params)
        summary = summary.sort_values(
            ["simulation", "frac", "d", "q", "lam", "sigma", "scope", "metric"]
        ).reset_index(drop=True)
        tuning_summary = build_tuning_summary(summary)

        param_tag = _build_param_tag(fit_pca_params, truth_pca_params)
        reps_path = _safe_output_path(outdir, _build_output_name("all_replicates_metrics", param_tag, ".csv"))
        summary_path = _safe_output_path(outdir, _build_output_name("summary_with_errorbar", param_tag, ".csv"))
        bins_path = _safe_output_path(outdir, _build_output_name("per_bin_metrics", param_tag, ".csv"))
        tuning_path = _safe_output_path(outdir, _build_output_name("tuning_summary", param_tag, ".csv"))

        all_reps.to_csv(reps_path, index=False)
        summary.to_csv(summary_path, index=False)
        tuning_summary.to_csv(tuning_path, index=False)
        print(f"Saved replicate-level table: {reps_path}")
        print(f"Saved summary table: {summary_path}")
        print(f"Saved tuning summary table: {tuning_path}")

        if len(bin_tables) > 0:
            all_bins = pd.concat(bin_tables, ignore_index=True)
            all_bins = all_bins.sort_values(
                ["simulation", "frac", "d", "q", "lam", "sigma", "replicate", "scope", "bin"]
            ).reset_index(drop=True)
            all_bins.to_csv(bins_path, index=False)
            print(f"Saved per-bin table: {bins_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sim1-only repeated imputation experiment with explicit initial-missing mask."
    )
    parser.add_argument("--tax-path", type=str, default="hmp2_mgx_taxonomy.tsv")
    parser.add_argument("--meta-path", type=str, default="metagenomics.xlsx")
    parser.add_argument("--outdir", type=str, default="results_sim1")
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--seed-base", type=int, default=20260322)

    parser.add_argument(
        "--truth-observed-mode",
        type=str,
        default="all_finite_except_replace",
        choices=["positive_only", "all_finite_except_replace", "all_finite"],
        help="Observed set used when building da_truth from the original tensor.",
    )
    parser.add_argument(
        "--second-observed-mode",
        type=str,
        default="all_finite_except_replace",
        choices=["positive_only", "all_finite_except_replace", "all_finite"],
        help="Observed set used in the second-stage imputation after random masking.",
    )
    parser.add_argument(
        "--mask-eligible-mode",
        type=str,
        default="all_finite",
        choices=["positive_only", "all_finite"],
        help="Which cells are eligible for random masking in da_truth.",
    )

    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--grid-d", type=int, nargs="+", default=None)
    parser.add_argument("--grid-q", type=int, nargs="+", default=None)
    parser.add_argument("--grid-lam", type=float, nargs="+", default=None)
    parser.add_argument("--grid-sigma", type=float, nargs="+", default=None)
    parser.add_argument("--truth-d", type=int, default=None)
    parser.add_argument("--truth-q", type=int, default=None)
    parser.add_argument("--truth-lam", type=float, default=None)
    parser.add_argument("--truth-sigma", type=float, default=None)
    parser.add_argument("--t0", type=int, default=30)
    parser.add_argument("--t", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_sim1_pipeline(args)


if __name__ == "__main__":
    main()
