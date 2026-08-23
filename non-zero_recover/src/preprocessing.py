from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

EXPECTED_DIMS = ("participant", "taxon", "bin")
DEFAULT_BINS = ["0"] + [f"{s}-{s+3}" for s in range(1, 37, 4)]


def week_to_bin(week: int | float | str) -> Optional[str]:
    """Map HMP2 week numbers to the longitudinal bins used by the experiments."""
    if pd.isna(week):
        return None
    week_i = int(float(week))
    if week_i == 0:
        return "0"
    if 1 <= week_i <= 36:
        start = ((week_i - 1) // 4) * 4 + 1
        return f"{start}-{start+3}"
    return None


def validate_tensor(da: xr.DataArray, name: str = "data") -> None:
    """Raise a clear error unless a tensor uses participant, taxon, bin dims."""
    if tuple(da.dims) != EXPECTED_DIMS:
        raise ValueError(f"{name} must have dims {EXPECTED_DIMS}, got {da.dims}")
    for dim in EXPECTED_DIMS:
        if dim not in da.coords:
            raise ValueError(f"{name} is missing coord {dim!r}")


def _taxon_filter(index: pd.Index) -> np.ndarray:
    idx = index.astype(str)
    has_species = idx.str.contains("s__", na=False)
    not_euk = ~idx.str.contains("k__Eukaryota", na=False)
    has_unclassified = idx.str.contains("unclassified", case=False, na=False)
    has_cag = idx.str.contains(r"CAG[_-]?\d+", regex=True, na=False)
    return np.asarray(has_species & not_euk & ~(has_unclassified & ~has_cag), dtype=bool)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_profile_tensor(
    tax_path: str | Path = "hmp2_mgx_taxonomy.tsv",
    meta_path: str | Path = "metagenomics.xlsx",
    eps: float = 1e-3,
    min_pos_cells: int = 10,
    min_participants: int = 5,
    outdir: str | Path = "outputs",
    save_diagnostics: bool = True,
    max_participants: Optional[int] = None,
    max_taxa: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, Dict[str, Any]]:
    """Read MetaPhlAn-style profile data and build log/raw longitudinal tensors.

    The taxonomy table is treated as a relative-abundance/profile table, not raw
    counts. Missing participant-bin grid cells are filled with profile value 0
    before applying log10(profile + eps).
    """
    tax_path = Path(tax_path)
    meta_path = Path(meta_path)
    if eps <= 0:
        raise ValueError("eps must be positive")

    tax = pd.read_csv(tax_path, sep="\t")
    first_col = tax.columns[0]
    tax = tax.rename(columns={first_col: "taxon"}).set_index("taxon")
    tax.columns = tax.columns.astype(str).str.replace(r"_profile$", "", regex=True)
    tax = tax.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    meta = pd.read_excel(meta_path)
    needed = ["External ID", "Participant ID", "data_type", "week_num"]
    missing = [c for c in needed if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata is missing columns: {missing}")
    meta_mgx = (
        meta.loc[meta["data_type"].eq("metagenomics"), needed]
        .dropna(subset=["External ID", "Participant ID", "week_num"])
        .rename(columns={"External ID": "external_id", "Participant ID": "participant", "week_num": "week"})
    )
    meta_mgx["external_id"] = meta_mgx["external_id"].astype(str).str.replace(r"_profile$", "", regex=True)
    meta_mgx["participant"] = meta_mgx["participant"].astype(str)
    meta_mgx["bin"] = meta_mgx["week"].map(week_to_bin)
    meta_mgx = meta_mgx.dropna(subset=["bin"]).drop_duplicates("external_id")

    tax_before = int(tax.shape[0])
    keep_tax = _taxon_filter(tax.index)
    tax = tax.loc[keep_tax].copy()
    tax_after_taxon_filter = int(tax.shape[0])

    common = sorted(set(tax.columns).intersection(set(meta_mgx["external_id"])))
    if not common:
        raise ValueError("No common External ID values between profile table and metadata")
    tax = tax.loc[:, common]
    meta_mgx = meta_mgx.set_index("external_id").loc[common].reset_index()

    x = tax.T
    x.index.name = "external_id"
    x = x.join(meta_mgx.set_index("external_id")[["participant", "bin"]], how="left")
    taxa_all = tax.index.astype(str).tolist()
    x_agg = x.set_index(["participant", "bin"])[taxa_all].groupby(level=["participant", "bin"]).mean()

    participants = sorted(x_agg.index.get_level_values("participant").unique().astype(str).tolist())
    if max_participants is not None:
        participants = participants[: int(max_participants)]
        x_agg = x_agg.loc[x_agg.index.get_level_values("participant").isin(participants)]
    bins = DEFAULT_BINS
    full_index = pd.MultiIndex.from_product([participants, bins], names=["participant", "bin"])
    x_full = x_agg.reindex(full_index)

    pos_mask_prevalence = x_full.gt(0).fillna(False)
    pos_cells = pos_mask_prevalence.sum(axis=0)
    pos_participants = pos_mask_prevalence.groupby(level="participant").any().sum(axis=0)
    keep_prev = (pos_cells >= int(min_pos_cells)) & (pos_participants >= int(min_participants))
    taxa_before_prev = int(x_full.shape[1])
    x_full = x_full.loc[:, keep_prev].copy()
    taxa_after_prev = int(x_full.shape[1])
    if max_taxa is not None and taxa_after_prev > int(max_taxa):
        prevalence_order = pos_cells[keep_prev].sort_values(ascending=False).index.astype(str).tolist()
        x_full = x_full.loc[:, prevalence_order[: int(max_taxa)]].copy()
    taxa = x_full.columns.astype(str).tolist()

    x_raw = x_full.fillna(0.0).clip(lower=0.0)
    observed_positive_df = x_full.gt(0.0).fillna(False)
    x_log = np.log10(x_raw + float(eps))

    raw_arr = x_raw.to_numpy().reshape(len(participants), len(bins), len(taxa)).transpose(0, 2, 1)
    log_arr = x_log.to_numpy().reshape(len(participants), len(bins), len(taxa)).transpose(0, 2, 1)
    obs_arr = observed_positive_df.to_numpy().reshape(len(participants), len(bins), len(taxa)).transpose(0, 2, 1)
    coords = {"participant": participants, "taxon": taxa, "bin": bins}
    da_log = xr.DataArray(log_arr, dims=EXPECTED_DIMS, coords=coords, name="log10_relative_abundance_profile_eps")
    da_raw = xr.DataArray(raw_arr, dims=EXPECTED_DIMS, coords=coords, name="relative_abundance_profile")
    observed_positive_mask = xr.DataArray(obs_arr.astype(bool), dims=EXPECTED_DIMS, coords=coords, name="observed_positive_mask")

    prevalence_kept = pos_cells.reindex(taxa).fillna(0).astype(int)
    participant_sample_counts = (
        x_agg.reset_index().groupby("participant")["bin"].nunique().reindex(participants).fillna(0).astype(int)
    )
    meta_info: Dict[str, Any] = {
        "input_type": "relative_abundance_profile",
        "tax_path": str(tax_path),
        "meta_path": str(meta_path),
        "participants": participants,
        "taxa": taxa,
        "bins": bins,
        "eps": float(eps),
        "min_pos_cells": int(min_pos_cells),
        "min_participants": int(min_participants),
        "taxa_before_taxon_filter": tax_before,
        "taxa_after_taxon_filter": tax_after_taxon_filter,
        "taxa_before_prevalence_filter": taxa_before_prev,
        "taxa_after_prevalence_filter": taxa_after_prev,
        "taxa_after_optional_cap": len(taxa),
        "observed_positive_ratio": float(observed_positive_mask.values.mean()) if observed_positive_mask.size else np.nan,
        "participant_count": len(participants),
        "bin_count": len(bins),
        "taxon_count": len(taxa),
    }

    if save_diagnostics:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        prev_summary = prevalence_kept.describe().to_dict()
        sample_summary = participant_sample_counts.describe().to_dict()
        summary_json = {**meta_info, "taxon_prevalence_summary": prev_summary, "participant_observed_sample_count_summary": sample_summary}
        (outdir / "preprocessing_summary.json").write_text(json.dumps(summary_json, indent=2, default=_json_safe), encoding="utf-8")
        diag = pd.DataFrame({"taxon": taxa, "positive_cells": prevalence_kept.to_numpy(), "positive_participants": pos_participants.reindex(taxa).fillna(0).astype(int).to_numpy()})
        diag.to_csv(outdir / "preprocessing_summary.csv", index=False)

    if verbose:
        print(f"[preprocessing] data shape={da_log.shape}, observed positive ratio={meta_info['observed_positive_ratio']:.6f}")
        print(f"[preprocessing] taxa filter {tax_before} -> {tax_after_taxon_filter}; prevalence {taxa_before_prev} -> {taxa_after_prev}; final {len(taxa)}")
    return da_log, da_raw, observed_positive_mask, meta_info


def build_base_tensor(*args, **kwargs):
    """Backward-compatible alias returning log tensor, positive mask, and metadata."""
    da_log, _da_raw, observed_positive_mask, meta_info = load_profile_tensor(*args, **kwargs)
    return da_log, observed_positive_mask, meta_info
