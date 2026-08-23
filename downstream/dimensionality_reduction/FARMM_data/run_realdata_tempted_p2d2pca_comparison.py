from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DOC_SOURCES = {
    "gemelli_repo": "https://github.com/biocore/gemelli",
    "tempted_paper": "https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03453-x",
    "tempted_r_manual": "https://cran.r-universe.dev/tempted/doc/manual.html",
}


def _bin_to_time_value(bin_label: Any) -> float:
    s = str(bin_label)
    if s.upper() == "PS":
        return 16.0
    if s == "0":
        return 0.0
    if "-" in s:
        a, b = s.split("-", 1)
        return 0.5 * (float(a) + float(b))
    return float(s)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stringify_id(value: Any) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        try:
            return str(int(float(s)))
        except Exception:
            return s
    return s


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _import_modules(*names: str) -> Dict[str, Any]:
    import importlib

    modules = {}
    for name in names:
        modules[name] = importlib.import_module(name)
    return modules


def _import_sim3() -> Any:
    import importlib

    return importlib.import_module("sim3_compare")


def _is_farmm_dataset(tax_path: str, meta_path: str) -> bool:
    return Path(tax_path).name.lower() == "kraken_results.tsv" and Path(meta_path).name.lower() == "20200619_farmm_metadata.tsv"


def _read_metadata_table(meta_path: str) -> Any:
    mods = _import_modules("pandas")
    pd = mods["pandas"]
    path = Path(meta_path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t")


def _detect_dependency_status() -> Dict[str, str]:
    import importlib

    status = {}
    for name in [
        "numpy",
        "pandas",
        "xarray",
        "openpyxl",
        "scipy",
        "sklearn",
        "matplotlib",
        "skbio",
        "biom",
        "qiime2",
        "gemelli",
    ]:
        try:
            importlib.import_module(name)
            status[name] = "OK"
        except Exception as exc:
            status[name] = f"FAIL: {type(exc).__name__}: {exc}"
    return status


@dataclass
class LabelPreparationResult:
    subject_metadata: Any
    conflict_df: Any
    excluded_df: Any
    diagnosis_counts: Dict[str, int]
    label_counts: Dict[str, int]


def load_and_prepare_labels(meta_path: str) -> LabelPreparationResult:
    mods = _import_modules("pandas")
    pd = mods["pandas"]

    meta = _read_metadata_table(meta_path)
    if _is_farmm_dataset("", meta_path) or {"SampleID", "SubjectID", "study_day", "study_group"}.issubset(meta.columns):
        required = ["SampleID", "SubjectID", "study_day", "study_group"]
        mgx = meta.loc[:, required].copy()
        mgx["participant"] = mgx["SubjectID"].map(_stringify_id)
        mgx["diagnosis"] = mgx["study_group"].astype("string")
        mgx["group_label"] = mgx["diagnosis"]
        mgx = mgx.dropna(subset=["SubjectID", "study_day", "study_group"])

        excluded = (
            meta.loc[
                meta[["SubjectID", "study_day", "study_group"]].isna().any(axis=1),
                ["SampleID", "SubjectID", "study_day", "study_group"],
            ]
            .copy()
            .rename(columns={"SampleID": "sample_id", "study_group": "excluded_diagnosis"})
        )

        participant_groups = (
            mgx.groupby("participant", as_index=False)
            .agg(
                diagnoses=("diagnosis", lambda s: sorted(set(map(str, s.dropna())))),
                groups=("group_label", lambda s: sorted(set(map(str, s.dropna())))),
                n_rows=("group_label", "size"),
            )
        )
        participant_groups["label_conflict"] = participant_groups["groups"].apply(lambda x: len(x) > 1)
        conflict_df = participant_groups.loc[participant_groups["label_conflict"]].copy()
        subject_metadata = participant_groups.loc[~participant_groups["label_conflict"]].copy()
        subject_metadata["diagnosis"] = subject_metadata["diagnoses"].apply(lambda x: "|".join(x))
        subject_metadata["group_label"] = subject_metadata["groups"].apply(lambda x: x[0] if x else "")
        subject_metadata = subject_metadata[["participant", "diagnosis", "group_label", "n_rows"]]
        subject_metadata = subject_metadata.sort_values("participant").reset_index(drop=True)

        diagnosis_counts = {
            str(k): int(v)
            for k, v in mgx["diagnosis"].value_counts(dropna=False).sort_index().items()
        }
        label_counts = {
            str(k): int(v)
            for k, v in subject_metadata["group_label"].value_counts(dropna=False).sort_index().items()
        }
        return LabelPreparationResult(
            subject_metadata=subject_metadata,
            conflict_df=conflict_df,
            excluded_df=excluded,
            diagnosis_counts=diagnosis_counts,
            label_counts=label_counts,
        )

    required = ["External ID", "Participant ID", "week_num", "diagnosis", "data_type"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    mgx = meta.loc[meta["data_type"].eq("metagenomics"), required].copy()
    mgx = mgx.dropna(subset=["Participant ID"])
    mgx["Participant ID"] = mgx["Participant ID"].astype(str)
    mgx["diagnosis"] = mgx["diagnosis"].astype(str)
    mgx["group_label"] = mgx["diagnosis"].map({"CD": "IBD", "UC": "IBD", "nonIBD": "nonIBD"})

    excluded = (
        mgx.loc[mgx["group_label"].isna(), ["Participant ID", "diagnosis"]]
        .drop_duplicates()
        .rename(columns={"Participant ID": "participant", "diagnosis": "excluded_diagnosis"})
    )

    participant_groups = (
        mgx.dropna(subset=["group_label"])
        .groupby("Participant ID", as_index=False)
        .agg(
            diagnoses=("diagnosis", lambda s: sorted(set(map(str, s)))),
            groups=("group_label", lambda s: sorted(set(map(str, s)))),
            n_rows=("group_label", "size"),
        )
        .rename(columns={"Participant ID": "participant"})
    )
    participant_groups["label_conflict"] = participant_groups["groups"].apply(lambda x: len(x) > 1)

    conflict_df = participant_groups.loc[participant_groups["label_conflict"]].copy()
    subject_metadata = participant_groups.loc[~participant_groups["label_conflict"]].copy()
    subject_metadata["diagnosis"] = subject_metadata["diagnoses"].apply(lambda x: "|".join(x))
    subject_metadata["group_label"] = subject_metadata["groups"].apply(lambda x: x[0])
    subject_metadata = subject_metadata[["participant", "diagnosis", "group_label", "n_rows"]]
    subject_metadata = subject_metadata.sort_values("participant").reset_index(drop=True)

    diagnosis_counts = {
        str(k): int(v)
        for k, v in mgx["diagnosis"].value_counts(dropna=False).sort_index().items()
    }
    label_counts = {
        str(k): int(v)
        for k, v in subject_metadata["group_label"].value_counts(dropna=False).sort_index().items()
    }

    return LabelPreparationResult(
        subject_metadata=subject_metadata,
        conflict_df=conflict_df,
        excluded_df=excluded,
        diagnosis_counts=diagnosis_counts,
        label_counts=label_counts,
    )


def build_real_tensor(
    tax_path: str,
    meta_path: str,
    eps: float,
    min_pos_cells: int,
    min_participants: int,
    verbose: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    if _is_farmm_dataset(tax_path, meta_path):
        return build_farmm_tensor(
            count_path=tax_path,
            meta_path=meta_path,
            eps=eps,
            min_pos_cells=min_pos_cells,
            min_participants=min_participants,
            verbose=verbose,
        )
    sim3 = _import_sim3()
    return sim3.build_base_tensor(
        tax_path=tax_path,
        meta_path=meta_path,
        eps=eps,
        min_pos_cells=min_pos_cells,
        min_participants=min_participants,
        verbose=verbose,
    )


def _farmm_time_sort_key(value: Any) -> Tuple[int, float, str]:
    s = str(value).strip()
    if s.upper() == "PS":
        return (1, 16.0, s)
    try:
        return (0, float(s), s)
    except Exception:
        return (2, float("inf"), s)


def build_farmm_tensor(
    count_path: str,
    meta_path: str,
    eps: float,
    min_pos_cells: int,
    min_participants: int,
    verbose: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    mods = _import_modules("numpy", "pandas", "xarray")
    np = mods["numpy"]
    pd = mods["pandas"]
    xr = mods["xarray"]

    count = pd.read_csv(count_path, sep="\t")
    metadata = pd.read_csv(meta_path, sep="\t")
    if count.empty:
        raise ValueError(f"Empty count table: {count_path}")
    required_meta = ["SampleID", "SubjectID", "study_day", "study_group"]
    missing_meta = [c for c in required_meta if c not in metadata.columns]
    if missing_meta:
        raise ValueError(f"Missing required FARMM metadata columns: {missing_meta}")

    feature_col = count.columns[0]
    raw_feature_count = int(count.shape[0])
    raw_sample_count = int(count.shape[1] - 1)
    raw_metadata_rows = int(metadata.shape[0])

    count = count.rename(columns={feature_col: "taxon"}).set_index("taxon")
    count.columns = count.columns.astype(str)
    metadata = metadata.copy()
    metadata["SampleID"] = metadata["SampleID"].astype(str)
    metadata["participant"] = metadata["SubjectID"].map(_stringify_id)
    metadata["bin"] = metadata["study_day"].astype(str).str.strip()

    count_sample_ids = count.columns.astype(str).tolist()
    metadata_sample_ids = metadata["SampleID"].astype(str).tolist()
    common_samples = [sid for sid in count_sample_ids if sid in set(metadata_sample_ids)]
    unmatched_count = sorted(set(count_sample_ids) - set(metadata_sample_ids))
    unmatched_metadata = sorted(set(metadata_sample_ids) - set(count_sample_ids))
    duplicate_count_sample_ids = int(len(count_sample_ids) - len(set(count_sample_ids)))
    duplicate_metadata_sample_ids = int(metadata["SampleID"].duplicated().sum())

    metadata_aligned = metadata.drop_duplicates("SampleID").set_index("SampleID").loc[common_samples].copy()
    valid_mask = metadata_aligned[["SubjectID", "study_day"]].notna().all(axis=1)
    valid_mask &= metadata_aligned["participant"].astype(str).str.len().gt(0)
    valid_mask &= metadata_aligned["bin"].astype(str).str.len().gt(0)
    filtered_missing_subject_time = int((~valid_mask).sum())
    metadata_valid = metadata_aligned.loc[valid_mask].copy()
    valid_samples = metadata_valid.index.astype(str).tolist()

    tax = count.loc[:, valid_samples].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    idx = tax.index.astype(str)
    has_species = idx.str.contains("s__", na=False)
    not_euk = ~idx.str.contains("k__Eukaryota", na=False)
    has_unclassified = idx.str.contains("unclassified", na=False)
    has_cag = idx.str.contains(r"CAG[_-]?\d+", regex=True, na=False)
    bad_unclassified = has_unclassified & (~has_cag)
    keep_tax_mask = has_species & not_euk & (~bad_unclassified)
    tax_after_taxonomy_filter = int(keep_tax_mask.sum())
    tax = tax.loc[keep_tax_mask].copy()

    x = tax.T
    x.index.name = "sample_id"
    taxa = tax.index.astype(str).tolist()
    x["participant"] = metadata_valid["participant"].astype(str).values
    x["bin"] = metadata_valid["bin"].astype(str).values

    duplicate_subject_time_count = int(x.duplicated(["participant", "bin"]).sum())
    x_taxa = x.set_index(["participant", "bin"])[taxa]
    x_agg = x_taxa.groupby(level=["participant", "bin"]).mean()

    participants = sorted(x_agg.index.get_level_values("participant").unique().astype(str).tolist())
    bins = sorted(x_agg.index.get_level_values("bin").unique().astype(str).tolist(), key=_farmm_time_sort_key)
    full_index = pd.MultiIndex.from_product([participants, bins], names=["participant", "bin"])
    x_full = x_agg.reindex(full_index)

    pos_mask_full = x_full.gt(0).fillna(False)
    pos_cells_per_taxon = pos_mask_full.sum(axis=0)
    pos_by_participant = pos_mask_full.groupby(level="participant").any()
    pos_participants_per_taxon = pos_by_participant.sum(axis=0)
    keep_prev_mask = (
        (pos_cells_per_taxon >= min_pos_cells)
        & (pos_participants_per_taxon >= min_participants)
    )

    tax_before_prev = int(x_full.shape[1])
    x_full = x_full.loc[:, keep_prev_mask].copy()
    tax_after_prev = int(x_full.shape[1])
    taxa = x_full.columns.astype(str).tolist()

    if verbose:
        print(f"[FARMM input] count matrix: {raw_feature_count} taxa x {raw_sample_count} samples")
        print(f"[FARMM input] metadata rows: {raw_metadata_rows}")
        print(f"[FARMM input] matched samples: {len(common_samples)}")
        print(f"[FARMM input] valid subject-time samples: {len(valid_samples)}")
        print(f"[FARMM input] taxonomy filter taxa: {raw_feature_count} -> {tax_after_taxonomy_filter}")
        print(
            f"[prevalence filter] taxa: {tax_before_prev} -> {tax_after_prev} "
            f"(min_pos_cells={min_pos_cells}, min_participants={min_participants})"
        )

    x_full_0 = x_full.fillna(0.0)
    x_full_log = np.log10(x_full_0 + eps)
    observed_positive_df = x_full.gt(0.0).fillna(False)

    arr = x_full_log.to_numpy().reshape(len(participants), len(bins), len(taxa))
    tensor = np.transpose(arr, (0, 2, 1))
    obs_arr = observed_positive_df.to_numpy().reshape(len(participants), len(bins), len(taxa))
    obs_tensor = np.transpose(obs_arr, (0, 2, 1))

    da = xr.DataArray(
        tensor,
        dims=("participant", "taxon", "bin"),
        coords={"participant": participants, "taxon": taxa, "bin": bins},
        name="abundance_binned_log",
    )
    da_observed_mask = xr.DataArray(
        obs_tensor,
        dims=("participant", "taxon", "bin"),
        coords={"participant": participants, "taxon": taxa, "bin": bins},
        name="observed_positive_mask",
    )

    sample_counts_by_group = (
        metadata_valid["study_group"].astype(str).value_counts(dropna=False).sort_index().to_dict()
        if "study_group" in metadata_valid.columns
        else {}
    )
    meta_info = {
        "dataset": "FARMM",
        "participants": participants,
        "bins": bins,
        "taxa": taxa,
        "eps": eps,
        "min_pos_cells": min_pos_cells,
        "min_participants": min_participants,
        "input_checks": {
            "count_raw_shape_taxa_by_samples": [raw_feature_count, raw_sample_count],
            "metadata_shape_rows_by_columns": [raw_metadata_rows, int(metadata.shape[1])],
            "count_direction": "feature_by_sample",
            "feature_id_field": "Term",
            "count_sample_id_location": "columns_after_first_feature_column",
            "metadata_sample_id_field": "SampleID",
            "matched_samples": int(len(common_samples)),
            "unmatched_count_samples": int(len(unmatched_count)),
            "unmatched_metadata_samples": int(len(unmatched_metadata)),
            "duplicate_count_sample_ids": duplicate_count_sample_ids,
            "duplicate_metadata_sample_ids": duplicate_metadata_sample_ids,
            "filtered_missing_subject_time_samples": filtered_missing_subject_time,
            "valid_subject_time_samples": int(len(valid_samples)),
            "duplicate_subject_time_count": duplicate_subject_time_count,
            "duplicate_subject_time_handling": "mean aggregation over feature values, matching the original grouped tensor builder",
            "subject_field": "SubjectID",
            "time_field": "study_day",
            "group_field": "study_group",
            "diagnosis_or_label_field": "study_group",
            "label_mapping": {"diagnosis": "study_group", "group_label": "study_group"},
            "time_mapping": {"PS": 16.0},
            "sample_counts_by_group": {str(k): int(v) for k, v in sample_counts_by_group.items()},
        },
    }
    return da, da_observed_mask, meta_info


def define_real_replace_mask(da: Any, eps: float, mode: str, tolerance: float = 1e-12) -> Tuple[Any, Dict[str, Any]]:
    mods = _import_modules("numpy", "xarray")
    np = mods["numpy"]
    xr = mods["xarray"]

    zero_value = math.log10(float(eps))
    vals = da.values.astype(float)
    nan_mask = np.isnan(vals)
    eps_mask = np.isfinite(vals) & (vals <= zero_value + tolerance)

    if mode == "eps_only":
        mask = eps_mask
    elif mode == "nan_only":
        mask = nan_mask
    elif mode == "nan_and_eps":
        mask = nan_mask | eps_mask
    else:
        raise ValueError(f"Unsupported impute_mode: {mode}")

    replace_mask = xr.DataArray(mask, coords=da.coords, dims=da.dims, name="replace_mask")

    participant_rates = (
        replace_mask.mean(dim=("taxon", "bin")).to_series().rename("replace_fraction").reset_index()
    )
    bin_rates = replace_mask.mean(dim=("participant", "taxon")).to_series().rename("replace_fraction").reset_index()
    taxon_rates = replace_mask.mean(dim=("participant", "bin")).to_series().rename("replace_fraction").reset_index()

    abundance_zero = eps_mask
    all_zero_participant_bin = abundance_zero.all(axis=1)
    all_zero_taxon = abundance_zero.all(axis=(0, 2))
    all_zero_participant = abundance_zero.all(axis=(1, 2))

    report = {
        "total_entries": int(vals.size),
        "eps_or_zero_entries": int(eps_mask.sum()),
        "nan_entries": int(nan_mask.sum()),
        "replace_entries": int(mask.sum()),
        "all_zero_participant_bin_count": int(all_zero_participant_bin.sum()),
        "all_zero_taxon_count": int(all_zero_taxon.sum()),
        "all_zero_participant_count": int(all_zero_participant.sum()),
        "participant_rates": participant_rates,
        "bin_rates": bin_rates,
        "taxon_rates": taxon_rates,
        "all_zero_participant_bin_flags": all_zero_participant_bin,
        "all_zero_taxon_flags": all_zero_taxon,
        "all_zero_participant_flags": all_zero_participant,
        "zero_value": zero_value,
        "tolerance": tolerance,
        "mode": mode,
    }
    return replace_mask, report


def run_p2d2pca_imputation(
    da: Any,
    replace_mask: Any,
    pca_params: Dict[str, Any],
    zero_value: float,
    algo_seed: int,
    verbose: bool = False,
) -> Any:
    sim3 = _import_sim3()
    da_imputed, _ = sim3.impute_by_mask(
        da_input=da,
        replace_mask=replace_mask,
        model_seed=int(algo_seed),
        method="penalized_2d2_pca",
        zero_value=float(zero_value),
        method_params=None,
        pca_params=pca_params,
        observed_mode="positive_only_except_replace",
        verbose=verbose,
    )
    return da_imputed


def da_to_tempted_inputs_grid_filled(
    da: Any,
    subject_metadata: Any,
    eps: float,
    read_depth: int,
) -> Tuple[Any, Any, Dict[str, Any]]:
    mods = _import_modules("numpy", "pandas")
    np = mods["numpy"]
    pd = mods["pandas"]

    if da.dims != ("participant", "taxon", "bin"):
        raise ValueError(f"Expected dims ('participant', 'taxon', 'bin'), got {da.dims}")

    label_map = subject_metadata.set_index("participant")[["diagnosis", "group_label"]]
    taxa = da.coords["taxon"].values.tolist()
    rows = []
    meta_rows = []
    zero_filled_samples = []

    for participant in da.coords["participant"].values.tolist():
        if participant not in label_map.index:
            continue
        for bin_label in da.coords["bin"].values.tolist():
            sample_id = f"{participant}__{bin_label}"
            vec_log = da.sel(participant=participant, bin=bin_label).values.astype(float)

            # Preserve the longitudinal grid.  Missing/empty participant-bin slices are
            # represented as all-zero count vectors instead of being removed.  This is
            # important for TEMPTED, which requires each subject to retain repeated states.
            with np.errstate(over="ignore", invalid="ignore"):
                abundance = np.power(10.0, vec_log) - float(eps)
            abundance = np.where(np.isfinite(abundance), abundance, 0.0)
            abundance = np.maximum(abundance, 0.0)
            total = float(abundance.sum())

            if np.isfinite(total) and total > 0.0:
                rel = abundance / total
                counts = np.rint(rel * int(read_depth)).astype(int)
                if counts.sum() <= 0:
                    # Extremely unlikely with the default read_depth, but keep the sample
                    # rather than dropping it.  Put one count on the largest abundance taxon.
                    counts = np.zeros(len(taxa), dtype=int)
                    counts[int(np.argmax(abundance))] = 1
                    zero_filled_samples.append(
                        {
                            "sample_id": sample_id,
                            "participant": participant,
                            "bin": str(bin_label),
                            "reason": "rounded_to_zero_count_rescued_with_single_count",
                        }
                    )
            else:
                counts = np.zeros(len(taxa), dtype=int)
                if np.isnan(vec_log).all():
                    reason = "all_nan_filled_with_zero_counts"
                else:
                    reason = "all_zero_or_nonfinite_after_back_transform_filled_with_zero_counts"
                zero_filled_samples.append(
                    {
                        "sample_id": sample_id,
                        "participant": participant,
                        "bin": str(bin_label),
                        "reason": reason,
                    }
                )

            rows.append(pd.Series(counts, index=taxa, name=sample_id))
            meta_rows.append(
                {
                    "sample_id": sample_id,
                    "participant_id": participant,
                    "bin": str(bin_label),
                    "time_numeric": _bin_to_time_value(bin_label),
                    "diagnosis": label_map.loc[participant, "diagnosis"],
                    "group_label": label_map.loc[participant, "group_label"],
                }
            )

    if not rows:
        raise ValueError("No participant-bin samples available for TEMPTED input after label matching.")

    feature_table_df = pd.DataFrame(rows).fillna(0).astype(int)
    sample_metadata_df = pd.DataFrame(meta_rows).sort_values(["participant_id", "time_numeric", "sample_id"])
    feature_table_df = feature_table_df.loc[sample_metadata_df["sample_id"].tolist()]
    feature_table_df = feature_table_df.T
    feature_table_df.index.name = "feature_id"
    sample_metadata_df = sample_metadata_df.reset_index(drop=True)

    timepoint_counts = sample_metadata_df.groupby("participant_id")["time_numeric"].nunique()
    subjects_lt2 = timepoint_counts.loc[timepoint_counts < 2]
    if not subjects_lt2.empty:
        preview = subjects_lt2.head(10).to_dict()
        raise ValueError(
            "Some subjects still have fewer than two time points after zero-filling. "
            "This means the base tensor itself lacks repeated bins for these subjects, "
            f"not that samples were dropped. First examples: {preview}"
        )

    zero_filled_df = pd.DataFrame(zero_filled_samples)
    sample_sums = feature_table_df.sum(axis=0)
    diagnostics = {
        "n_features": int(feature_table_df.shape[0]),
        "n_samples": int(feature_table_df.shape[1]),
        "dropped_samples": pd.DataFrame([]),
        "zero_filled_samples": zero_filled_df,
        "zero_filled_sample_count": int(len(zero_filled_samples)),
        "subjects_with_zero_filled_samples": int(zero_filled_df["participant"].nunique()) if not zero_filled_df.empty else 0,
        "subjects_retained": int(sample_metadata_df["participant_id"].nunique()),
        "min_timepoints_per_subject": int(timepoint_counts.min()),
        "max_timepoints_per_subject": int(timepoint_counts.max()),
        "min_sample_sum": int(sample_sums.min()),
        "max_sample_sum": int(sample_sums.max()),
    }
    return feature_table_df, sample_metadata_df, diagnostics


def da_to_tempted_inputs_observed_only(
    da: Any,
    subject_metadata: Any,
    eps: float,
    read_depth: int,
    min_timepoints_per_subject: int = 2,
    drop_empty_samples: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Create TEMPTED inputs from genuinely observed, non-empty states only."""
    mods = _import_modules("numpy", "pandas")
    np = mods["numpy"]
    pd = mods["pandas"]

    if da.dims != ("participant", "taxon", "bin"):
        raise ValueError(f"Expected dims ('participant', 'taxon', 'bin'), got {da.dims}")
    if not drop_empty_samples:
        raise ValueError("observed-only TEMPTED input requires drop_empty_samples=True.")

    label_map = subject_metadata.set_index("participant")[["diagnosis", "group_label"]]
    taxa = da.coords["taxon"].values.tolist()
    rows = []
    meta_rows = []
    dropped_samples = []

    for participant in da.coords["participant"].values.tolist():
        if participant not in label_map.index:
            continue
        for bin_label in da.coords["bin"].values.tolist():
            sample_id = f"{participant}__{bin_label}"
            vec_log = da.sel(participant=participant, bin=bin_label).values.astype(float)
            with np.errstate(over="ignore", invalid="ignore"):
                abundance = np.power(10.0, vec_log) - float(eps)
            abundance = np.where(np.isfinite(abundance), abundance, 0.0)
            abundance = np.maximum(abundance, 0.0)
            total = float(abundance.sum())
            if not np.isfinite(total) or total <= 0.0:
                reason = (
                    "all_nan_or_nonfinite"
                    if not np.isfinite(vec_log).any()
                    else "all_zero_or_empty_after_back_transform"
                )
                dropped_samples.append(
                    {"sample_id": sample_id, "participant": participant, "bin": str(bin_label), "reason": reason}
                )
                continue

            counts = np.rint((abundance / total) * int(read_depth)).astype(int)
            if int(counts.sum()) <= 0:
                dropped_samples.append(
                    {"sample_id": sample_id, "participant": participant, "bin": str(bin_label), "reason": "rounded_to_zero"}
                )
                continue

            rows.append(pd.Series(counts, index=taxa, name=sample_id))
            meta_rows.append(
                {
                    "sample_id": sample_id,
                    "participant_id": participant,
                    "bin": str(bin_label),
                    "time_numeric": _bin_to_time_value(bin_label),
                    "diagnosis": label_map.loc[participant, "diagnosis"],
                    "group_label": label_map.loc[participant, "group_label"],
                }
            )

    if not rows:
        raise ValueError("No non-empty observed participant-bin samples available for direct TEMPTED input.")

    row_df = pd.DataFrame(rows).fillna(0).astype(int)
    sample_metadata_df = pd.DataFrame(meta_rows)
    timepoint_counts = sample_metadata_df.groupby("participant_id")["time_numeric"].nunique()
    subjects_dropped = timepoint_counts.loc[timepoint_counts < int(min_timepoints_per_subject)].index.astype(str).tolist()
    if subjects_dropped:
        removed = sample_metadata_df.loc[sample_metadata_df["participant_id"].astype(str).isin(subjects_dropped)]
        for item in removed[["sample_id", "participant_id", "bin"]].to_dict("records"):
            dropped_samples.append(
                {
                    "sample_id": item["sample_id"],
                    "participant": item["participant_id"],
                    "bin": str(item["bin"]),
                    "reason": "subject_lt_min_timepoints",
                }
            )
        keep = ~sample_metadata_df["participant_id"].astype(str).isin(subjects_dropped)
        sample_metadata_df = sample_metadata_df.loc[keep].copy()

    if sample_metadata_df.empty:
        raise ValueError("No subjects retain the minimum observed time points required for direct TEMPTED input.")

    sample_metadata_df = sample_metadata_df.sort_values(["participant_id", "time_numeric", "sample_id"]).reset_index(drop=True)
    retained_ids = sample_metadata_df["sample_id"].tolist()
    feature_table_df = row_df.loc[retained_ids].T
    feature_table_df.index.name = "feature_id"
    retained_timepoints = sample_metadata_df.groupby("participant_id")["time_numeric"].nunique()
    sample_sums = feature_table_df.sum(axis=0)
    dropped_df = pd.DataFrame(dropped_samples, columns=["sample_id", "participant", "bin", "reason"])
    diagnostics = {
        "n_features": int(feature_table_df.shape[0]),
        "n_samples": int(feature_table_df.shape[1]),
        "subjects_retained": int(sample_metadata_df["participant_id"].nunique()),
        "subjects_dropped_lt_min_timepoints": int(len(subjects_dropped)),
        "dropped_sample_count": int(len(dropped_df)),
        "dropped_empty_sample_count": int((dropped_df["reason"] != "subject_lt_min_timepoints").sum()),
        "dropped_samples": dropped_df,
        "min_timepoints_per_subject": int(retained_timepoints.min()),
        "max_timepoints_per_subject": int(retained_timepoints.max()),
        "min_sample_sum": int(sample_sums.min()),
        "max_sample_sum": int(sample_sums.max()),
    }
    return feature_table_df, sample_metadata_df, diagnostics


def da_to_tempted_inputs(
    da: Any,
    subject_metadata: Any,
    eps: float,
    read_depth: int,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Backward-compatible alias for the historical complete-grid conversion."""
    return da_to_tempted_inputs_grid_filled(da, subject_metadata, eps, read_depth)


def _ordination_samples_to_dataframe(artifact: Any) -> Optional[Any]:
    """Extract subject/sample coordinates from a QIIME2 PCoAResults artifact.

    gemelli TEMPTED's individual_biplot is a QIIME2 PCoAResults biplot artifact.
    It is not a regular TSV table.  Reading its exported ordination.txt with
    pd.read_csv can incorrectly produce a one-column dataframe such as
    ['Eigvals\t3'].  The stable representation is skbio.OrdinationResults.samples.
    """
    mods = _import_modules("pandas")
    pd = mods["pandas"]

    try:
        from skbio import OrdinationResults
    except Exception:
        OrdinationResults = None

    def _samples_from_ordination(ordination: Any) -> Optional[Any]:
        # For TEMPTED's individual_biplot the subject coordinates are stored in
        # the "Site"/samples section.  Some skbio/biplot artifacts leave .samples
        # empty and place the subject loadings in .features; fall back to that.
        for attr in ("samples", "features"):
            coords = getattr(ordination, attr, None)
            if isinstance(coords, pd.DataFrame) and not coords.empty:
                numeric = coords.apply(pd.to_numeric, errors="coerce")
                if numeric.notna().to_numpy().any():
                    out = coords.copy()
                    out.index.name = out.index.name or "id"
                    return out.reset_index()
        return None

    # Path 1: view the artifact directly as skbio OrdinationResults.
    if OrdinationResults is not None:
        try:
            ordination = artifact.view(OrdinationResults)
            out = _samples_from_ordination(ordination)
            if out is not None:
                return out
        except Exception:
            pass

        # Path 2: export to disk and let skbio's own reader parse ordination.txt.
        # This is the most robust route because skbio never mis-reads the
        # "Eigvals\t<n>" header as a coordinate column the way pd.read_csv does.
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact.export_data(tmpdir)
                base = Path(tmpdir)
                ord_files = (
                    list(base.rglob("ordination.txt"))
                    + list(base.rglob("*ordination*.txt"))
                )
                for path in ord_files:
                    try:
                        ordination = OrdinationResults.read(str(path))
                    except Exception:
                        continue
                    out = _samples_from_ordination(ordination)
                    if out is not None:
                        return out
        except Exception:
            pass

    return None


def _parse_exported_ordination_txt(path: Path) -> Optional[Any]:
    """Parse QIIME2/skbio ordination.txt and return the Site/Samples coordinates."""
    mods = _import_modules("pandas")
    pd = mods["pandas"]

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        parts = line.split("\t")
        section = parts[0].strip().lower()
        n_rows = None
        n_axes = None
        if len(parts) >= 2:
            try:
                n_rows = int(float(parts[1]))
            except Exception:
                n_rows = None
        if len(parts) >= 3:
            try:
                n_axes = int(float(parts[2]))
            except Exception:
                n_axes = None

        # In skbio/QIIME2 ordination exports, sample coordinates are stored in
        # the "Site" section.  Some versions may use a sample-like label.
        if section in {"site", "sites", "sample", "samples"} and n_rows and n_axes:
            rows = []
            for raw in lines[i + 1 : i + 1 + n_rows]:
                fields = raw.rstrip("\n").split("\t")
                if len(fields) >= 1 + n_axes:
                    rows.append([fields[0]] + fields[1 : 1 + n_axes])
            if rows:
                cols = ["id"] + [f"Axis {j + 1}" for j in range(n_axes)]
                return pd.DataFrame(rows, columns=cols)

        # Skip known sections with a declared row count.  This prevents generic
        # CSV parsing from treating "Eigvals\t3" as a coordinate header.
        if n_rows is not None:
            i += 1 + max(n_rows, 0)
        else:
            i += 1

    return None


def _artifact_to_dataframe(artifact: Any) -> Optional[Any]:
    mods = _import_modules("pandas")
    pd = mods["pandas"]

    # First handle QIIME2 PCoAResults / biplot artifacts correctly.
    df = _ordination_samples_to_dataframe(artifact)
    if df is not None and not df.empty:
        return df

    # Keep the generic DataFrame view for true tabular artifacts.
    for viewer in (pd.DataFrame,):
        try:
            viewed = artifact.view(viewer)
            if isinstance(viewed, pd.DataFrame) and not viewed.empty:
                if not _looks_like_ordination_frame(viewed):
                    return viewed
        except Exception:
            pass

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact.export_data(tmpdir)
        base = Path(tmpdir)

        # Ordination artifacts export as ordination.txt; parse the Site section.
        for path in list(base.rglob("ordination.txt")) + list(base.rglob("*ordination*.txt")):
            df = _parse_exported_ordination_txt(path)
            if df is not None and not df.empty:
                return df

        candidates = list(base.rglob("*.tsv")) + list(base.rglob("*.csv")) + list(base.rglob("*.txt"))
        for path in candidates:
            try:
                first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
            except Exception:
                first = ""
            # Ordination/biplot text files (first token "Eigvals", "Proportion",
            # "Species", "Site", ...) are not rectangular tables; route them to
            # the dedicated parser above rather than pd.read_csv.
            if first.split("\t", 1)[0].strip().lower() in {
                "eigvals", "proportion explained", "species", "site",
                "biplot", "site constraints",
            }:
                continue
            for sep in ("\t", ","):
                try:
                    df = pd.read_csv(path, sep=sep)
                    if not df.empty and not _looks_like_ordination_frame(df):
                        return df
                except Exception:
                    continue
    return None


def _looks_like_ordination_frame(df: Any) -> bool:
    """True if a DataFrame is actually mis-parsed ordination text.

    pd.read_csv / artifact.view(pd.DataFrame) on a skbio ordination file can
    collapse the whole header into one object column such as 'Eigvals\\t3' or
    'Proportion explained\\t3'.  Such frames must never reach coordinate
    standardization.
    """
    cols = [str(c) for c in df.columns]
    if len(cols) == 1:
        only = cols[0]
        if "\t" in only:
            return True
        if only.split()[0].lower() in {"eigvals", "proportion", "species", "site", "biplot"}:
            return True
    return False

def _namedtuple_to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "_asdict"):
        return dict(obj._asdict())
    if isinstance(obj, tuple):
        return {f"item_{i}": v for i, v in enumerate(obj)}
    if isinstance(obj, dict):
        return dict(obj)
    return {"result": obj}


def run_tempted(feature_table_df: Any, sample_metadata_df: Any, args: argparse.Namespace) -> Dict[str, Any]:
    try:
        import biom
        import qiime2
        from qiime2.plugins.gemelli.actions import clr_transformation, tempted_factorize
    except Exception as exc:
        raise RuntimeError(
            "QIIME2/gemelli/biom is required to run TEMPTED. "
            f"Current import failure: {type(exc).__name__}: {exc}"
        ) from exc

    table = biom.Table(
        feature_table_df.to_numpy(dtype=int),
        observation_ids=feature_table_df.index.astype(str).tolist(),
        sample_ids=feature_table_df.columns.astype(str).tolist(),
    )
    table_artifact = qiime2.Artifact.import_data("FeatureTable[Frequency]", table)

    # QIIME2 Metadata requires the dataframe index name to be a valid ID header.
    # "sample_id" is rejected; "sample-id" is accepted and does not change the actual sample IDs.
    metadata_df = sample_metadata_df.set_index("sample_id").copy()
    metadata_df.index.name = "sample-id"
    metadata = qiime2.Metadata(metadata_df)

    clr_result = clr_transformation(table_artifact)
    clr_artifact = clr_result[0] if isinstance(clr_result, tuple) else clr_result

    try:
        tempted_result = tempted_factorize(
            table=clr_artifact,
            sample_metadata=metadata,
            individual_id_column="participant_id",
            state_column="time_numeric",
            n_components=int(args.tempted_rank),
            smooth=float(args.tempted_smooth),
            resolution=int(args.resolution),
            max_iterations=int(args.max_iterations),
            epsilon=float(args.tempted_epsilon),
            replicate_handling=str(args.replicate_handling),
            svd_centralized=bool(args.svd_centralized),
        )
    except TypeError as exc:
        raise RuntimeError(
            "tempted_factorize call failed. The installed gemelli/QIIME2 API likely differs "
            "from the expected signature in this script."
        ) from exc

    return {
        "clr_artifact": clr_artifact,
        "result_dict": _namedtuple_to_dict(tempted_result),
        "sample_metadata_df": sample_metadata_df,
    }


def _clean_column_name(col: Any) -> str:
    return str(col).strip().replace("\ufeff", "")


def _standardize_coordinate_df(df: Any, id_candidates: Sequence[str]) -> Any:
    """Return a subject-level coordinate table with columns: participant, PC1, PC2, ...

    QIIME2/gemelli ordination exports are not stable across versions.  Coordinates
    may be named PC1/PC2, Axis 1/Axis 2, Dim1/Dim2, numeric 0/1/2, or may be read
    back as object strings.  This function normalizes those cases before ML.
    """
    mods = _import_modules("pandas")
    pd = mods["pandas"]

    if df is None or df.empty:
        raise ValueError("Empty coordinate dataframe.")

    if _looks_like_ordination_frame(df):
        raise ValueError(
            "Received mis-parsed ordination text instead of a coordinate table "
            f"(columns: {list(df.columns)}). The skbio OrdinationResults view "
            "should have been used for this artifact."
        )

    out = df.copy()
    out.columns = [_clean_column_name(c) for c in out.columns]

    # Some QIIME2 exports keep IDs in the index after artifact.view(pd.DataFrame).
    if out.index.name is not None or not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()
        out.columns = [_clean_column_name(c) for c in out.columns]

    lower_to_col = {_clean_column_name(c).lower().replace("-", "_"): c for c in out.columns}
    id_col = None
    for cand in id_candidates:
        key = cand.lower().replace("-", "_")
        if key in lower_to_col:
            id_col = lower_to_col[key]
            break

    if id_col is None:
        # Fall back to the first non-numeric-looking column, then to the first column.
        for c in out.columns:
            parsed = pd.to_numeric(out[c], errors="coerce")
            if parsed.notna().mean() < 0.5:
                id_col = c
                break
    if id_col is None:
        id_col = out.columns[0]

    out = out.rename(columns={id_col: "participant"})
    out["participant"] = out["participant"].astype(str)

    skip = {
        "participant", "participant_id", "subject_id", "individual_id",
        "sample_id", "sampleid", "sample-id",
        "group_label", "diagnosis", "bin", "state", "time", "time_numeric",
    }

    coord_cols = []
    for c in out.columns:
        c_norm = _clean_column_name(c).lower().replace("-", "_")
        if c_norm in skip:
            continue
        numeric = pd.to_numeric(out[c], errors="coerce")
        if numeric.notna().any():
            out[c] = numeric
            coord_cols.append(c)

    if not coord_cols:
        raise ValueError(
            "No usable numeric coordinate columns after parsing TEMPTED output. "
            f"Columns were: {list(df.columns)}"
        )

    # Rename all selected numeric coordinates to PC1, PC2, ... so downstream code
    # and plotting do not depend on gemelli's exact column naming convention.
    rename_map = {c: f"PC{i + 1}" for i, c in enumerate(coord_cols)}
    out = out[["participant"] + coord_cols].rename(columns=rename_map)
    out = out.dropna(subset=list(rename_map.values()), how="all")
    if out.empty:
        raise ValueError("All coordinate rows are NaN after numeric conversion.")
    return out.reset_index(drop=True)


def extract_subject_coordinates(tempted_result: Dict[str, Any]) -> Any:
    result_dict = tempted_result["result_dict"]
    sample_metadata_df = tempted_result["sample_metadata_df"]

    subject_candidates = [
        "subject_biplot",
        "individual_biplot",
        "subject_ordination",
        "subject_loadings",
    ]
    last_errors = []
    for key in subject_candidates:
        artifact = result_dict.get(key)
        if artifact is None:
            continue
        df = _artifact_to_dataframe(artifact)
        if df is None or df.empty:
            continue
        try:
            return _standardize_coordinate_df(
                df,
                id_candidates=("participant", "participant_id", "subject_id", "individual_id", "id"),
            )
        except Exception as exc:
            last_errors.append(f"{key}: {type(exc).__name__}: {exc}")

    state_candidates = [
        "state_subject_ordination",
        "state_ordination",
        "sample_ordination",
    ]
    for key in state_candidates:
        artifact = result_dict.get(key)
        if artifact is None:
            continue
        df = _artifact_to_dataframe(artifact)
        if df is None or df.empty:
            continue

        try:
            state_df = _standardize_coordinate_df(
                df,
                id_candidates=("sample_id", "sampleid", "sample-id", "id"),
            )
            state_df = state_df.rename(columns={"participant": "sample_id"})
            merged = state_df.merge(
                sample_metadata_df[["sample_id", "participant_id"]],
                on="sample_id",
                how="left",
            )
            pc_cols = [c for c in merged.columns if str(c).upper().startswith("PC")]
            merged = merged.dropna(subset=["participant_id"])
            if merged.empty:
                raise ValueError("No state/sample coordinates matched sample_metadata_df.")
            out = merged.groupby("participant_id", as_index=False)[pc_cols].mean()
            out = out.rename(columns={"participant_id": "participant"})
            out["participant"] = out["participant"].astype(str)
            return out
        except Exception as exc:
            last_errors.append(f"{key}: {type(exc).__name__}: {exc}")

    detail = "; ".join(last_errors) if last_errors else "No candidate TEMPTED coordinate artifact was found."
    raise RuntimeError(f"Unable to extract subject-level coordinates from TEMPTED outputs. Details: {detail}")


def extract_sample_coordinates(tempted_result: Dict[str, Any]) -> Any:
    """Extract TEMPTED state/sample coordinates and attach longitudinal metadata."""
    mods = _import_modules("pandas")
    pd = mods["pandas"]
    result_dict = tempted_result["result_dict"]
    sample_metadata_df = tempted_result["sample_metadata_df"]
    candidates = [
        "state_subject_ordination",
        "state_ordination",
        "sample_ordination",
        "sample_biplot",
        "individual_trajectory",
    ]
    errors = []
    for key in candidates:
        artifact = result_dict.get(key)
        if artifact is None:
            continue
        df = _artifact_to_dataframe(artifact)
        if df is None or df.empty:
            continue
        try:
            standardized = _standardize_coordinate_df(
                df,
                id_candidates=("sample_id", "sampleid", "sample-id", "id"),
            ).rename(columns={"participant": "sample_id"})
            merged = standardized.merge(
                sample_metadata_df[["sample_id", "participant_id", "time_numeric", "diagnosis", "group_label"]],
                on="sample_id",
                how="inner",
            )
            pc_cols = [c for c in merged.columns if str(c).upper().startswith("PC")]
            if merged.empty or not pc_cols:
                raise ValueError("No sample coordinate rows matched sample metadata.")
            return merged[["sample_id", "participant_id", "time_numeric", "diagnosis", "group_label"] + pc_cols].reset_index(drop=True)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    detail = "; ".join(errors) if errors else "No candidate state/sample TEMPTED artifact was found."
    raise RuntimeError(f"Unable to extract sample-level coordinates from TEMPTED outputs. Details: {detail}")


def export_tempted_loadings(tempted_result: Dict[str, Any], pipeline_name: str, outdir: Any) -> None:
    """Dump feature (taxon) loadings b and temporal loading xi(t) from a TEMPTED result.

    The subject loadings a are already saved elsewhere via extract_subject_coordinates.
    Here we persist the two loadings the interpretation panels need but that the main
    pipeline never exported:
      * feature loadings b   -> individual_biplot(.features): taxon x component
      * temporal loading xi(t) -> the state/temporal loading artifact (tabular)
    Version-agnostic: we iterate every artifact in result_dict, save the raw .qza as a
    fallback, export any OrdinationResults' .features/.samples, and dump any tabular
    artifact as CSV.  Exact gemelli field names differ across versions, so we do not
    hard-code them.
    """
    mods = _import_modules("pandas")
    pd = mods["pandas"]
    try:
        from skbio import OrdinationResults
    except Exception:
        OrdinationResults = None

    ld = _ensure_dir(Path(outdir) / "tempted_loadings")
    result_dict = tempted_result.get("result_dict", {}) or {}

    for key, artifact in result_dict.items():
        if artifact is None or not hasattr(artifact, "save"):
            continue
        # 1) always keep the raw artifact so nothing is lost even on field-name drift
        try:
            artifact.save(str(ld / f"{pipeline_name}__{key}.qza"))
        except Exception:
            pass
        # 2) ordination artifacts -> feature loadings (b) and subject loadings (a)
        exported = False
        if OrdinationResults is not None:
            try:
                ordn = artifact.view(OrdinationResults)
                for attr in ("features", "samples"):
                    coords = getattr(ordn, attr, None)
                    if isinstance(coords, pd.DataFrame) and not coords.empty:
                        coords.to_csv(ld / f"{pipeline_name}__{key}__{attr}.csv")
                        exported = True
                prop = getattr(ordn, "proportion_explained", None)
                if prop is not None:
                    try:
                        prop.to_csv(ld / f"{pipeline_name}__{key}__prop_explained.csv")
                    except Exception:
                        pass
            except Exception:
                exported = False
        if exported:
            continue
        # 3) tabular artifacts (e.g. the state / temporal loading grid) -> csv
        try:
            df = artifact.view(pd.DataFrame)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(ld / f"{pipeline_name}__{key}__table.csv")
        except Exception:
            pass


def run_sample_level_permanova(sample_coords: Any, pipeline_name: str, args: argparse.Namespace) -> Any:
    mods = _import_modules("numpy", "pandas")
    np = mods["numpy"]
    pd = mods["pandas"]
    row = {
        "pipeline": pipeline_name,
        "grouping": "group_label",
        "n_samples": int(len(sample_coords)) if sample_coords is not None else 0,
        "n_subjects": (
            int(sample_coords["participant_id"].nunique())
            if sample_coords is not None and "participant_id" in sample_coords.columns
            else 0
        ),
        "n_pcs": 0,
        "permutations": int(args.permanova_permutations),
        "pseudo_F": np.nan,
        "p_value": np.nan,
        "error": "",
    }
    try:
        if sample_coords is None or sample_coords.empty:
            raise ValueError("No sample-level TEMPTED coordinates were extracted.")
        pc_cols = [c for c in sample_coords.columns if str(c).upper().startswith("PC")]
        pc_cols = pc_cols[: min(2, len(pc_cols))]
        if not pc_cols:
            raise ValueError("No PC columns available for sample-level PERMANOVA.")
        data = sample_coords.dropna(subset=pc_cols + ["group_label", "sample_id"]).copy()
        x = data[pc_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x).all(axis=1)
        data = data.loc[finite].reset_index(drop=True)
        x = x[finite]
        row.update(
            {
                "n_samples": int(len(data)),
                "n_subjects": int(data["participant_id"].nunique()),
                "n_pcs": int(len(pc_cols)),
            }
        )
        if len(data) < 2 or data["group_label"].nunique() < 2:
            raise ValueError("PERMANOVA requires at least two samples and two group labels.")
        if data["sample_id"].astype(str).duplicated().any():
            raise ValueError("Duplicate sample IDs prevent construction of a PERMANOVA distance matrix.")
        from skbio.stats.distance import DistanceMatrix, permanova

        delta = x[:, None, :] - x[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        ids = data["sample_id"].astype(str).tolist()
        dm = DistanceMatrix(distances, ids=ids)
        grouping = pd.Series(data["group_label"].astype(str).values, index=ids, name="group_label")
        result = permanova(dm, grouping, permutations=int(args.permanova_permutations))
        row.update(
            {
                "pseudo_F": float(result["test statistic"]),
                "p_value": float(result["p-value"]),
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        logs_dir = _ensure_dir(Path(args.outdir) / "logs")
        _write_text(logs_dir / f"permanova_failure_{pipeline_name}.txt", row["error"] + "\n")
    return pd.DataFrame([row])


def run_subject_level_permanova(
    coords: Any,
    subject_metadata: Any,
    pipeline_name: str,
    args: argparse.Namespace,
    grouping_mode: str,
) -> Any:
    mods = _import_modules("numpy", "pandas")
    np = mods["numpy"]
    pd = mods["pandas"]

    columns = [
        "pipeline", "grouping", "n_subjects", "n_groups", "group_counts_json",
        "n_pcs", "pc_cols_json", "permutations", "pseudo_F", "p_value", "error",
    ]
    row = {
        "pipeline": pipeline_name,
        "grouping": grouping_mode,
        "n_subjects": 0,
        "n_groups": 0,
        "group_counts_json": json.dumps({}),
        "n_pcs": 0,
        "pc_cols_json": json.dumps([]),
        "permutations": int(args.permanova_permutations),
        "pseudo_F": np.nan,
        "p_value": np.nan,
        "error": "",
    }

    try:
        if coords is None or coords.empty:
            raise ValueError("No subject-level TEMPTED coordinates were provided.")
        if subject_metadata is None or subject_metadata.empty:
            raise ValueError("No subject metadata were provided.")
        if "participant" not in coords.columns:
            raise ValueError("Subject coordinates must contain a participant column.")

        pc_cols = [c for c in coords.columns if str(c).upper().startswith("PC")]
        if not pc_cols:
            raise ValueError("No PC columns available for subject-level PERMANOVA.")
        if int(args.permanova_n_pcs) > 0:
            pc_cols = pc_cols[: min(int(args.permanova_n_pcs), len(pc_cols))]
        row["n_pcs"] = int(len(pc_cols))
        row["pc_cols_json"] = json.dumps([str(c) for c in pc_cols])

        coord_df = coords[["participant"] + pc_cols].copy()
        coord_df["participant"] = coord_df["participant"].astype(str)
        for pc in pc_cols:
            coord_df[pc] = pd.to_numeric(coord_df[pc], errors="coerce")
        coord_df = coord_df.groupby("participant", as_index=False)[pc_cols].mean()

        metadata_cols = ["participant", "diagnosis", "group_label"]
        missing_meta = [c for c in metadata_cols if c not in subject_metadata.columns]
        if missing_meta:
            raise ValueError(f"Subject metadata missing required columns: {missing_meta}")
        metadata = subject_metadata[metadata_cols].copy()
        metadata["participant"] = metadata["participant"].astype(str)
        metadata = metadata.groupby("participant", as_index=False).agg(
            diagnosis=("diagnosis", "first"),
            group_label=("group_label", "first"),
        )

        merged = coord_df.merge(metadata, on="participant", how="inner")
        if grouping_mode == "group_label":
            group_col = "group_label"
        else:
            raise ValueError(f"Unsupported subject-level PERMANOVA grouping_mode: {grouping_mode}")

        merged = merged.dropna(subset=["participant", group_col] + pc_cols).copy()
        if merged.empty:
            raise ValueError("No subjects remained after merging coordinates with metadata.")

        x = merged[pc_cols].to_numpy(dtype=float)
        finite = np.isfinite(x).all(axis=1)
        merged = merged.loc[finite].reset_index(drop=True)
        x = x[finite]
        group_counts = {
            str(k): int(v)
            for k, v in merged[group_col].astype(str).value_counts().sort_index().items()
        }
        row.update(
            {
                "n_subjects": int(len(merged)),
                "n_groups": int(len(group_counts)),
                "group_counts_json": json.dumps(group_counts, sort_keys=True),
            }
        )

        if merged.empty:
            raise ValueError("No subjects remained after removing NaN/Inf PC coordinates.")
        if len(group_counts) < 2:
            raise ValueError("PERMANOVA requires at least two groups.")
        small_groups = {k: v for k, v in group_counts.items() if v < 2}
        if small_groups:
            raise ValueError(f"Each PERMANOVA group requires at least two subjects; small_groups={small_groups}.")
        if merged["participant"].astype(str).duplicated().any():
            raise ValueError("Duplicate participant IDs remained after aggregation.")

        from skbio.stats.distance import DistanceMatrix, permanova

        delta = x[:, None, :] - x[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        ids = merged["participant"].astype(str).tolist()
        dm = DistanceMatrix(distances, ids=ids)
        grouping = pd.Series(merged[group_col].astype(str).values, index=ids, name=grouping_mode)
        result = permanova(dm, grouping, permutations=int(args.permanova_permutations))
        row.update(
            {
                "pseudo_F": float(result["test statistic"]),
                "p_value": float(result["p-value"]),
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        logs_dir = _ensure_dir(Path(args.outdir) / "logs")
        _write_text(
            logs_dir / f"subject_permanova_failure_{pipeline_name}_{grouping_mode}.txt",
            row["error"] + "\n",
        )
    return pd.DataFrame([row], columns=columns)


def _benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    mods = _import_modules("numpy")
    np = mods["numpy"]
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(values))
    if not len(valid_idx):
        return adjusted.tolist()
    order = valid_idx[np.argsort(values[valid_idx])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted.tolist()


def analyze_group_loading_tests(
    coords: Any,
    subject_metadata: Any,
    pipeline_name: str,
    outdir: Path,
    label_col: str = "group_label",
    group_order: Optional[Sequence[str]] = None,
    n_pcs: Optional[int] = None,
) -> Any:
    """Per-PC loading tests for the FARMM diet groups.

    This is the subject-loading analogue of asking whether each TEMPTED PC carries
    EEN/Omnivore/Vegan separation. It records both parametric one-way ANOVA and
    rank-based Kruskal-Wallis tests, with FDR correction over PCs.
    """
    mods = _import_modules("numpy", "pandas")
    np = mods["numpy"]
    pd = mods["pandas"]
    columns = [
        "pipeline", "label_col", "component", "n_subjects", "n_groups",
        "group_counts_json", "group_means_json", "group_sds_json",
        "anova_F", "anova_p", "anova_fdr_bh", "eta_squared",
        "kruskal_H", "kruskal_p", "kruskal_fdr_bh", "error",
    ]

    def _empty(err: str) -> Any:
        return pd.DataFrame([{
            "pipeline": pipeline_name, "label_col": label_col, "component": None,
            "n_subjects": 0, "n_groups": 0, "group_counts_json": json.dumps({}),
            "group_means_json": json.dumps({}), "group_sds_json": json.dumps({}),
            "anova_F": np.nan, "anova_p": np.nan, "anova_fdr_bh": np.nan,
            "eta_squared": np.nan, "kruskal_H": np.nan, "kruskal_p": np.nan,
            "kruskal_fdr_bh": np.nan, "error": err,
        }], columns=columns)

    try:
        if coords is None or coords.empty:
            return _empty("No coordinates provided.")
        if subject_metadata is None or subject_metadata.empty:
            return _empty("No subject metadata provided.")
        if label_col not in subject_metadata.columns:
            return _empty(f"label_col '{label_col}' missing from subject metadata.")
        if "participant" not in coords.columns:
            return _empty("Coordinates must contain a participant column.")

        meta = subject_metadata[["participant", label_col]].copy()
        meta["participant"] = meta["participant"].astype(str)
        meta[label_col] = meta[label_col].astype(str)
        if group_order is None:
            group_order = sorted(meta[label_col].dropna().unique().tolist())
        group_order = [str(g) for g in group_order]
        meta = meta.loc[meta[label_col].isin(group_order)].drop_duplicates("participant")

        merged = coords.copy()
        merged["participant"] = merged["participant"].astype(str)
        merged = merged.merge(meta, on="participant", how="inner")
        pc_cols = [c for c in merged.columns if str(c).upper().startswith("PC")]
        if n_pcs is not None and int(n_pcs) > 0:
            pc_cols = pc_cols[: min(int(n_pcs), len(pc_cols))]
        if not pc_cols:
            return _empty("No PC columns available.")

        scipy_stats = None
        try:
            scipy_stats = _import_modules("scipy.stats")["scipy.stats"]
        except Exception:
            scipy_stats = None

        rows = []
        for pc in pc_cols:
            pc_df = merged[[label_col, pc]].copy()
            pc_df[pc] = pd.to_numeric(pc_df[pc], errors="coerce")
            pc_df = pc_df.dropna(subset=[label_col, pc])
            arrays = []
            counts = {}
            means = {}
            sds = {}
            for group in group_order:
                arr = pc_df.loc[pc_df[label_col].astype(str).eq(group), pc].to_numpy(dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    arrays.append(arr)
                counts[group] = int(arr.size)
                means[group] = float(np.mean(arr)) if arr.size else np.nan
                sds[group] = float(np.std(arr, ddof=1)) if arr.size >= 2 else np.nan

            row = {
                "pipeline": pipeline_name,
                "label_col": label_col,
                "component": pc,
                "n_subjects": int(sum(counts.values())),
                "n_groups": int(sum(1 for v in counts.values() if v > 0)),
                "group_counts_json": json.dumps(counts, sort_keys=False),
                "group_means_json": json.dumps(means, sort_keys=False, default=_json_default),
                "group_sds_json": json.dumps(sds, sort_keys=False, default=_json_default),
                "anova_F": np.nan,
                "anova_p": np.nan,
                "anova_fdr_bh": np.nan,
                "eta_squared": np.nan,
                "kruskal_H": np.nan,
                "kruskal_p": np.nan,
                "kruskal_fdr_bh": np.nan,
                "error": "",
            }
            if len(arrays) < 2 or any(a.size < 2 for a in arrays):
                row["error"] = "Each observed group requires at least two subjects."
            else:
                try:
                    all_values = np.concatenate(arrays)
                    grand_mean = float(np.mean(all_values))
                    ss_between = float(sum(a.size * (float(np.mean(a)) - grand_mean) ** 2 for a in arrays))
                    ss_total = float(np.sum((all_values - grand_mean) ** 2))
                    ss_within = ss_total - ss_between
                    df_between = len(arrays) - 1
                    df_within = all_values.size - len(arrays)
                    if df_between > 0 and df_within > 0 and ss_within > 0:
                        row["anova_F"] = float((ss_between / df_between) / (ss_within / df_within))
                    row["eta_squared"] = ss_between / ss_total if ss_total > 0 else np.nan
                    if scipy_stats is not None and np.isfinite(row["anova_F"]):
                        row["anova_p"] = float(scipy_stats.f.sf(row["anova_F"], df_between, df_within))
                except Exception as exc:
                    row["error"] = f"anova: {type(exc).__name__}: {exc}"
                try:
                    if scipy_stats is None:
                        raise ImportError("scipy.stats unavailable for Kruskal-Wallis p-value.")
                    kw = scipy_stats.kruskal(*arrays)
                    row["kruskal_H"] = float(kw.statistic)
                    row["kruskal_p"] = float(kw.pvalue)
                except Exception as exc:
                    row["error"] = (row["error"] + " | " if row["error"] else "") + \
                        f"kruskal: {type(exc).__name__}: {exc}"
            rows.append(row)

        out = pd.DataFrame(rows, columns=columns)
        out["anova_fdr_bh"] = _benjamini_hochberg(out["anova_p"].tolist())
        out["kruskal_fdr_bh"] = _benjamini_hochberg(out["kruskal_p"].tolist())

        try:
            plt = _import_modules("matplotlib.pyplot")["matplotlib.pyplot"]
            plots_dir = _ensure_dir(outdir / "plots")
            for pc in pc_cols[: min(3, len(pc_cols))]:
                pc_df = merged[[label_col, pc]].copy()
                pc_df[pc] = pd.to_numeric(pc_df[pc], errors="coerce")
                pc_df = pc_df.dropna(subset=[label_col, pc])
                groups = [
                    pc_df.loc[pc_df[label_col].astype(str).eq(group), pc].to_numpy(dtype=float)
                    for group in group_order
                ]
                labels = [group for group, arr in zip(group_order, groups) if len(arr)]
                groups = [arr for arr in groups if len(arr)]
                if len(groups) < 2:
                    continue
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.boxplot(groups, labels=labels)
                for idx, arr in enumerate(groups, start=1):
                    ax.scatter(np.full(len(arr), idx), arr, alpha=0.75, s=22)
                ax.set_ylabel(pc)
                ax.set_title(f"{pipeline_name}: {label_col} loading on {pc}")
                fig.tight_layout()
                fig.savefig(plots_dir / f"group_loading_{pipeline_name}_{pc}.png", dpi=180)
                plt.close(fig)
        except Exception as exc:
            _write_text(
                _ensure_dir(outdir / "logs") / f"group_loading_plot_failure_{pipeline_name}.txt",
                f"{type(exc).__name__}: {exc}\n",
            )
        return out
    except Exception as exc:
        _write_text(
            _ensure_dir(outdir / "logs") / f"group_loading_failure_{pipeline_name}.txt",
            f"{type(exc).__name__}: {exc}\n",
        )
        return _empty(f"{type(exc).__name__}: {exc}")


def _build_classifier(name: str) -> Any:
    mods = _import_modules(
        "sklearn.pipeline",
        "sklearn.preprocessing",
        "sklearn.linear_model",
        "sklearn.ensemble",
    )
    Pipeline = mods["sklearn.pipeline"].Pipeline
    StandardScaler = mods["sklearn.preprocessing"].StandardScaler
    LogisticRegression = mods["sklearn.linear_model"].LogisticRegression
    RandomForestClassifier = mods["sklearn.ensemble"].RandomForestClassifier

    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            random_state=2026,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
    raise ValueError(f"Unsupported classifier: {name}")


def run_ml_evaluation(coords: Any, labels: Any, args: argparse.Namespace, pipeline_name: str) -> Any:
    mods = _import_modules(
        "numpy",
        "pandas",
        "sklearn.metrics",
        "sklearn.model_selection",
        "sklearn.base",
    )
    np = mods["numpy"]
    pd = mods["pandas"]
    metrics = mods["sklearn.metrics"]
    StratifiedShuffleSplit = mods["sklearn.model_selection"].StratifiedShuffleSplit
    clone = mods["sklearn.base"].clone

    coords = coords.copy()
    labels = labels[["participant", "group_label"]].copy()
    coords["participant"] = coords["participant"].astype(str)
    labels["participant"] = labels["participant"].astype(str)

    merged = coords.merge(labels, on="participant", how="inner")
    if merged.empty:
        raise ValueError(
            "No participants overlap between TEMPTED coordinates and labels. "
            f"coord_participant_examples={coords['participant'].head(5).tolist()}, "
            f"label_participant_examples={labels['participant'].head(5).tolist()}"
        )

    pc_cols = [c for c in merged.columns if str(c).upper().startswith("PC")]
    if not pc_cols:
        excluded = {"participant", "group_label", "diagnosis", "sample_id", "sampleid", "bin", "time_numeric"}
        parsed_cols = []
        for c in merged.columns:
            if str(c).lower().replace("-", "_") in excluded:
                continue
            numeric = pd.to_numeric(merged[c], errors="coerce")
            if numeric.notna().any():
                merged[c] = numeric
                parsed_cols.append(c)
        pc_cols = parsed_cols

    if not pc_cols:
        raise ValueError(
            "No numeric coordinate columns found for machine learning evaluation. "
            f"coords_columns={list(coords.columns)}, merged_columns={list(merged.columns)}, "
            f"coords_dtypes={coords.dtypes.astype(str).to_dict()}"
        )

    x_df = merged[pc_cols].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(x_df.to_numpy(dtype=float)).all(axis=1)
    if not finite_mask.any():
        raise ValueError("All rows have NaN/Inf coordinate values after numeric conversion.")
    if int((~finite_mask).sum()) > 0:
        merged = merged.loc[finite_mask].reset_index(drop=True)
        x_df = x_df.loc[finite_mask].reset_index(drop=True)

    x = x_df.to_numpy(dtype=float)
    observed_labels = merged["group_label"].astype(str)
    unique_labels = sorted(observed_labels.dropna().unique().tolist())
    class_labels = unique_labels
    class_to_id = {label: idx for idx, label in enumerate(class_labels)}
    y = observed_labels.map(class_to_id).to_numpy(dtype=int)
    n_classes = len(class_labels)
    if np.unique(y).size < 2:
        raise ValueError(
            "Only one class remains after merging coordinates with labels. "
            f"group_label_counts={merged['group_label'].value_counts(dropna=False).to_dict()}"
        )
    positive_label = class_labels[1] if n_classes == 2 else ""
    splitter = StratifiedShuffleSplit(
        n_splits=int(args.n_repeats),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )

    rows = []
    classifiers = ["logistic_regression", "random_forest"]
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for clf_name in classifiers:
            clf = clone(_build_classifier(clf_name))
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            if hasattr(clf, "predict_proba"):
                raw_score = clf.predict_proba(x_test)
                y_score_full = np.zeros((len(y_test), n_classes), dtype=float)
                for col_idx, class_id in enumerate(clf.classes_):
                    y_score_full[:, int(class_id)] = raw_score[:, col_idx]
            elif hasattr(clf, "decision_function"):
                raw_score = clf.decision_function(x_test)
                if n_classes == 2:
                    y_score_full = np.column_stack([-raw_score, raw_score])
                else:
                    y_score_full = np.asarray(raw_score, dtype=float)
            else:
                y_score_full = np.eye(n_classes, dtype=float)[y_pred.astype(int)]

            try:
                if n_classes == 2:
                    auroc = float(metrics.roc_auc_score(y_test, y_score_full[:, 1]))
                else:
                    auroc = float(
                        metrics.roc_auc_score(
                            y_test,
                            y_score_full,
                            labels=list(range(n_classes)),
                            multi_class="ovr",
                            average="macro",
                        )
                    )
            except Exception:
                auroc = float("nan")
            try:
                if n_classes == 2:
                    auprc = float(metrics.average_precision_score(y_test, y_score_full[:, 1]))
                else:
                    y_test_one_hot = np.eye(n_classes, dtype=int)[y_test]
                    auprc = float(
                        metrics.average_precision_score(
                            y_test_one_hot,
                            y_score_full,
                            average="macro",
                        )
                    )
            except Exception:
                auprc = float("nan")
            if n_classes == 2:
                f1 = float(metrics.f1_score(y_test, y_pred, zero_division=0))
            else:
                f1 = float(metrics.f1_score(y_test, y_pred, average="macro", zero_division=0))
            train_counts = {
                class_labels[idx]: int((y_train == idx).sum())
                for idx in range(n_classes)
            }
            test_counts = {
                class_labels[idx]: int((y_test == idx).sum())
                for idx in range(n_classes)
            }

            rows.append(
                {
                    "pipeline": pipeline_name,
                    "classifier": clf_name,
                    "repeat": int(math.ceil(split_id)),
                    "split_id": int(split_id),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_class_counts_json": json.dumps(train_counts, sort_keys=False),
                    "test_class_counts_json": json.dumps(test_counts, sort_keys=False),
                    "n_classes": int(n_classes),
                    "class_labels_json": json.dumps(class_labels, sort_keys=False),
                    "positive_label": positive_label,
                    "n_positive_train": int((y_train == 1).sum()) if n_classes == 2 else 0,
                    "n_negative_train": int((y_train == 0).sum()) if n_classes == 2 else 0,
                    "n_positive_test": int((y_test == 1).sum()) if n_classes == 2 else 0,
                    "n_negative_test": int((y_test == 0).sum()) if n_classes == 2 else 0,
                    "AUROC": auroc,
                    "AUPRC": auprc,
                    "balanced_accuracy": float(metrics.balanced_accuracy_score(y_test, y_pred)),
                    "F1": f1,
                    "accuracy": float(metrics.accuracy_score(y_test, y_pred)),
                    "metric_averaging": "binary_positive_class" if n_classes == 2 else "macro_multiclass",
                    "evaluation_mode": "transductive_unsupervised_dr",
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(metrics_all: Any) -> Any:
    mods = _import_modules("numpy", "pandas")
    np = mods["numpy"]
    pd = mods["pandas"]

    metric_cols = ["AUROC", "AUPRC", "balanced_accuracy", "F1", "accuracy"]
    rows = []
    for (pipeline, classifier), group in metrics_all.groupby(["pipeline", "classifier"]):
        for metric in metric_cols:
            vals = group[metric].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                rows.append(
                    {
                        "pipeline": pipeline,
                        "classifier": classifier,
                        "metric": metric,
                        "mean": np.nan,
                        "sd": np.nan,
                        "se": np.nan,
                        "ci95_lower": np.nan,
                        "ci95_upper": np.nan,
                    }
                )
                continue
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0
            se = float(sd / math.sqrt(vals.size)) if vals.size >= 2 else 0.0
            ci = 1.96 * se
            rows.append(
                {
                    "pipeline": pipeline,
                    "classifier": classifier,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "se": se,
                    "ci95_lower": mean - ci,
                    "ci95_upper": mean + ci,
                }
            )
    return pd.DataFrame(rows)


def run_unsupervised_clustering_evaluation(
    coords: Any,
    labels: Any,
    pipeline_name: str,
    outdir: Path,
    n_pcs: Optional[int] = None,
) -> Any:
    """Evaluate whether unsupervised clusters recover EEN/Omnivore/Vegan labels."""
    base_mods = _import_modules("numpy", "pandas")
    np = base_mods["numpy"]
    pd = base_mods["pandas"]

    columns = [
        "pipeline", "method", "n_subjects", "n_classes", "class_labels_json",
        "n_clusters", "n_pcs", "pc_cols_json", "ARI", "NMI", "homogeneity",
        "completeness", "v_measure", "silhouette", "calinski_harabasz",
        "davies_bouldin", "cluster_counts_json", "error",
    ]

    def _empty(method: str, err: str) -> Dict[str, Any]:
        return {
            "pipeline": pipeline_name, "method": method, "n_subjects": 0,
            "n_classes": 0, "class_labels_json": json.dumps([]),
            "n_clusters": 0, "n_pcs": 0, "pc_cols_json": json.dumps([]),
            "ARI": np.nan, "NMI": np.nan, "homogeneity": np.nan,
            "completeness": np.nan, "v_measure": np.nan, "silhouette": np.nan,
            "calinski_harabasz": np.nan, "davies_bouldin": np.nan,
            "cluster_counts_json": json.dumps({}), "error": err,
        }

    rows = []
    pred_rows = []
    methods = ["kmeans", "gaussian_mixture", "agglomerative_ward"]
    try:
        mods = _import_modules(
            "sklearn.cluster",
            "sklearn.mixture",
            "sklearn.metrics",
            "sklearn.preprocessing",
        )
        KMeans = mods["sklearn.cluster"].KMeans
        AgglomerativeClustering = mods["sklearn.cluster"].AgglomerativeClustering
        GaussianMixture = mods["sklearn.mixture"].GaussianMixture
        metrics = mods["sklearn.metrics"]
        StandardScaler = mods["sklearn.preprocessing"].StandardScaler

        if coords is None or coords.empty:
            raise ValueError("No coordinates provided.")
        lab = labels[["participant", "group_label"]].copy()
        lab["participant"] = lab["participant"].astype(str)
        lab["group_label"] = lab["group_label"].astype(str)
        df = coords.copy()
        df["participant"] = df["participant"].astype(str)
        merged = df.merge(lab, on="participant", how="inner")
        pc_cols = [c for c in merged.columns if str(c).upper().startswith("PC")]
        if n_pcs is not None and int(n_pcs) > 0:
            pc_cols = pc_cols[: min(int(n_pcs), len(pc_cols))]
        if not pc_cols:
            raise ValueError("No PC columns available for clustering.")
        for pc in pc_cols:
            merged[pc] = pd.to_numeric(merged[pc], errors="coerce")
        merged = merged.dropna(subset=["participant", "group_label"] + pc_cols).copy()
        x = merged[pc_cols].to_numpy(dtype=float)
        finite = np.isfinite(x).all(axis=1)
        merged = merged.loc[finite].reset_index(drop=True)
        x = x[finite]
        class_labels = sorted(merged["group_label"].dropna().astype(str).unique().tolist())
        n_clusters = len(class_labels)
        if len(merged) < 2 or n_clusters < 2:
            raise ValueError("Clustering evaluation requires at least two subjects and two classes.")
        if len(merged) <= n_clusters:
            raise ValueError("Clustering evaluation requires more subjects than classes.")
        y_true = merged["group_label"].astype(str).to_numpy()
        x_scaled = StandardScaler().fit_transform(x)

        for method in methods:
            row = _empty(method, "")
            row.update({
                "n_subjects": int(len(merged)),
                "n_classes": int(n_clusters),
                "class_labels_json": json.dumps(class_labels, sort_keys=False),
                "n_clusters": int(n_clusters),
                "n_pcs": int(len(pc_cols)),
                "pc_cols_json": json.dumps([str(c) for c in pc_cols]),
            })
            try:
                if method == "kmeans":
                    y_pred = KMeans(n_clusters=n_clusters, n_init=50, random_state=2026).fit_predict(x_scaled)
                elif method == "gaussian_mixture":
                    y_pred = GaussianMixture(
                        n_components=n_clusters,
                        covariance_type="full",
                        random_state=2026,
                        n_init=20,
                    ).fit_predict(x_scaled)
                elif method == "agglomerative_ward":
                    y_pred = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(x_scaled)
                else:
                    raise ValueError(f"Unsupported clustering method: {method}")

                cluster_counts = {
                    str(k): int(v)
                    for k, v in pd.Series(y_pred).value_counts().sort_index().items()
                }
                row.update({
                    "ARI": float(metrics.adjusted_rand_score(y_true, y_pred)),
                    "NMI": float(metrics.normalized_mutual_info_score(y_true, y_pred)),
                    "homogeneity": float(metrics.homogeneity_score(y_true, y_pred)),
                    "completeness": float(metrics.completeness_score(y_true, y_pred)),
                    "v_measure": float(metrics.v_measure_score(y_true, y_pred)),
                    "cluster_counts_json": json.dumps(cluster_counts, sort_keys=True),
                })
                if len(set(y_pred)) > 1 and len(set(y_pred)) < len(y_pred):
                    row["silhouette"] = float(metrics.silhouette_score(x_scaled, y_pred))
                    row["calinski_harabasz"] = float(metrics.calinski_harabasz_score(x_scaled, y_pred))
                    row["davies_bouldin"] = float(metrics.davies_bouldin_score(x_scaled, y_pred))
                for participant, true_label, cluster in zip(
                    merged["participant"].astype(str), y_true, y_pred
                ):
                    pred_rows.append({
                        "pipeline": pipeline_name,
                        "method": method,
                        "participant": participant,
                        "group_label": true_label,
                        "cluster": int(cluster),
                    })
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    except Exception as exc:
        rows = [_empty(method, f"{type(exc).__name__}: {exc}") for method in methods]

    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(outdir / f"cluster_assignments_{pipeline_name}.csv", index=False)
    return pd.DataFrame(rows, columns=columns)


def plot_results(metrics_summary: Any, coords_map: Dict[str, Any], outdir: Path) -> None:
    try:
        mods = _import_modules("matplotlib.pyplot", "pandas")
    except Exception:
        return

    plt = mods["matplotlib.pyplot"]
    pd = mods["pandas"]

    plots_dir = _ensure_dir(outdir / "plots")

    for pipeline_name, coords in coords_map.items():
        pc_cols = [c for c in coords.columns if str(c).upper().startswith("PC")]
        if len(pc_cols) < 2 or "group_label" not in coords.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        for label, g in coords.groupby("group_label"):
            ax.scatter(g[pc_cols[0]], g[pc_cols[1]], label=label, alpha=0.8)
        ax.set_xlabel(pc_cols[0])
        ax.set_ylabel(pc_cols[1])
        ax.set_title(pipeline_name)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plots_dir / f"embedding_{pipeline_name.replace(' ', '_')}.png", dpi=180)
        plt.close(fig)

    for metric in ["AUROC", "AUPRC", "balanced_accuracy"]:
        df = metrics_summary.loc[metrics_summary["metric"] == metric].copy()
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot = df.pivot(index="classifier", columns="pipeline", values="mean")
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} comparison")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric.lower()}_comparison.png", dpi=180)
        plt.close(fig)


def _save_mask_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["section", "key", "value"])
        for key in [
            "mode",
            "zero_value",
            "tolerance",
            "total_entries",
            "eps_or_zero_entries",
            "nan_entries",
            "replace_entries",
            "all_zero_participant_bin_count",
            "all_zero_taxon_count",
            "all_zero_participant_count",
        ]:
            writer.writerow(["summary", key, report[key]])
        for section_name, df in [
            ("participant_rates", report["participant_rates"]),
            ("bin_rates", report["bin_rates"]),
            ("taxon_rates", report["taxon_rates"]),
        ]:
            writer.writerow([section_name, "", ""])
            df.to_csv(fh, index=False)
            writer.writerow([])


def _build_dry_run_report(lines: List[str], out_path: Path) -> None:
    _write_text(out_path, "\n".join(lines) + "\n")


def _compact_tensor_meta_info(meta_info: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(meta_info)
    for key in ("participants", "taxa"):
        if key in compact:
            values = compact.pop(key)
            try:
                compact[f"n_{key}"] = len(values)
            except Exception:
                pass
    if "bins" in compact:
        bins = compact["bins"]
        try:
            compact["n_bins"] = len(bins)
        except Exception:
            pass
    return compact


def _save_config(args: argparse.Namespace, outdir: Path) -> None:
    config = vars(args).copy()
    config["doc_sources"] = DOC_SOURCES
    (outdir / "config.json").write_text(json.dumps(config, indent=2, default=_json_default), encoding="utf-8")


def _apply_auto_data_paths(args: argparse.Namespace) -> None:
    default_tax = "hmp2_mgx_taxonomy.tsv"
    default_meta = "metagenomics.xlsx"
    farmm_count = Path("kraken_results.tsv")
    farmm_meta = Path("20200619_farmm_metadata.tsv")
    if (
        args.tax_path == default_tax
        and args.meta_path == default_meta
        and not Path(default_tax).exists()
        and not Path(default_meta).exists()
        and farmm_count.exists()
        and farmm_meta.exists()
    ):
        args.tax_path = str(farmm_count)
        args.meta_path = str(farmm_meta)


def _prepare_output_dirs(outdir: Path) -> Dict[str, Path]:
    return {
        "root": _ensure_dir(outdir),
        "logs": _ensure_dir(outdir / "logs"),
        "plots": _ensure_dir(outdir / "plots"),
    }


def run_dry_run(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    dirs = _prepare_output_dirs(outdir)
    _save_config(args, dirs["root"])
    lines: List[str] = []
    dep_status = _detect_dependency_status()

    lines.append("Dry run report")
    lines.append(f"tax_path_exists: {Path(args.tax_path).exists()}")
    lines.append(f"meta_path_exists: {Path(args.meta_path).exists()}")
    lines.append("dependency_status:")
    for key, value in dep_status.items():
        lines.append(f"  {key}: {value}")

    try:
        label_result = load_and_prepare_labels(args.meta_path)
        lines.append(f"participant_labels_constructed: {len(label_result.subject_metadata)}")
        lines.append(f"participant_label_conflicts: {len(label_result.conflict_df)}")
        lines.append(f"diagnosis_counts: {label_result.diagnosis_counts}")
        lines.append(f"group_label_counts: {label_result.label_counts}")
        label_result.subject_metadata.to_csv(outdir / "subject_metadata.csv", index=False)
        label_result.conflict_df.to_csv(dirs["logs"] / "label_conflict_report.csv", index=False)
    except Exception as exc:
        lines.append(f"label_preparation_failed: {type(exc).__name__}: {exc}")
        _build_dry_run_report(lines, dirs["logs"] / "dry_run_report.txt")
        return 1

    try:
        base_da, _, meta_info = build_real_tensor(
            tax_path=args.tax_path,
            meta_path=args.meta_path,
            eps=args.eps,
            min_pos_cells=args.min_pos_cells,
            min_participants=args.min_participants,
            verbose=False,
        )
        lines.append(f"tensor_dims: {base_da.dims}")
        lines.append(f"tensor_shape: {base_da.shape}")
        lines.append(f"tensor_meta_info_summary: {_compact_tensor_meta_info(meta_info)}")
        input_checks = meta_info.get("input_checks", {}) if isinstance(meta_info, dict) else {}
        if input_checks:
            lines.append("data_input_checks:")
            for key, value in input_checks.items():
                lines.append(f"  {key}: {value}")
    except Exception as exc:
        lines.append(f"build_base_tensor_failed: {type(exc).__name__}: {exc}")
        _build_dry_run_report(lines, dirs["logs"] / "dry_run_report.txt")
        return 1

    try:
        mods = _import_modules("numpy")
        np = mods["numpy"]
        vals = base_da.values.astype(float)
        zero_value = math.log10(float(args.eps))
        lines.append(f"nan_entries: {int(np.isnan(vals).sum())}")
        lines.append(f"inf_entries: {int(np.isinf(vals).sum())}")
        lines.append(f"log10_eps_entries: {int((np.isfinite(vals) & (vals <= zero_value + args.mask_tolerance)).sum())}")

        replace_mask, report = define_real_replace_mask(base_da, eps=args.eps, mode=args.impute_mode, tolerance=args.mask_tolerance)
        lines.append(f"replace_mask_entries: {int(report['replace_entries'])}")
        _save_mask_report(report, dirs["logs"] / "imputation_mask_report.csv")

        table_df, sample_meta_df, table_diag = da_to_tempted_inputs_observed_only(
            base_da,
            label_result.subject_metadata,
            eps=args.eps,
            read_depth=args.read_depth,
        )
        lines.append(f"direct_input_mode: observed_only_no_zero_filled_missing_timepoints")
        lines.append(f"observed_only_feature_table_shape: {table_df.shape}")
        lines.append(f"observed_only_sample_metadata_shape: {sample_meta_df.shape}")
        lines.append(
            "dimensionality_api_input: feature_table_df is feature x sample; "
            "sample_metadata_df preserves participant_id and time_numeric from the constructed tensor"
        )
        lines.append(f"final_tensor_dimension_order: participant x taxon x bin")
        lines.append(f"final_tempted_feature_table_dim_order: taxon x sample")
        diag_for_log = {k: v for k, v in table_diag.items() if k not in {"dropped_samples", "zero_filled_samples"}}
        lines.append(f"observed_only_input_diag: {diag_for_log}")
        sample_meta_df.to_csv(outdir / "sample_metadata_tempted_direct_observed_only.tsv", sep="\t", index=False)
        dropped = table_diag.get("dropped_samples")
        if dropped is not None:
            dropped.to_csv(dirs["logs"] / "dropped_samples_direct_observed_only.csv", index=False)
    except Exception as exc:
        lines.append(f"tempted_input_prep_failed: {type(exc).__name__}: {exc}")
        _build_dry_run_report(lines, dirs["logs"] / "dry_run_report.txt")
        return 1

    qiime_ok = dep_status.get("qiime2", "").startswith("OK")
    gemelli_ok = dep_status.get("gemelli", "").startswith("OK")
    biom_ok = dep_status.get("biom", "").startswith("OK")
    lines.append(f"qiime2_importable: {qiime_ok}")
    lines.append(f"gemelli_importable: {gemelli_ok}")
    lines.append(f"biom_importable: {biom_ok}")

    _build_dry_run_report(lines, dirs["logs"] / "dry_run_report.txt")
    return 0


def run_full(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    dirs = _prepare_output_dirs(outdir)
    _save_config(args, dirs["root"])

    label_result = load_and_prepare_labels(args.meta_path)
    label_result.subject_metadata.to_csv(outdir / "subject_metadata.csv", index=False)
    label_result.conflict_df.to_csv(dirs["logs"] / "label_conflict_report.csv", index=False)
    label_result.excluded_df.to_csv(dirs["logs"] / "excluded_diagnosis_report.csv", index=False)

    base_da, _, _ = build_real_tensor(
        tax_path=args.tax_path,
        meta_path=args.meta_path,
        eps=args.eps,
        min_pos_cells=args.min_pos_cells,
        min_participants=args.min_participants,
        verbose=not args.quiet,
    )

    sim3 = _import_sim3()
    sim3._save_dataarray(base_da, "tensor_base_raw", outdir / "tensor_base_raw")

    replace_mask, mask_report = define_real_replace_mask(
        base_da,
        eps=args.eps,
        mode=args.impute_mode,
        tolerance=args.mask_tolerance,
    )
    _save_mask_report(mask_report, dirs["logs"] / "imputation_mask_report.csv")
    zero_value = mask_report["zero_value"]

    pca_params = {
        "d": int(args.d),
        "q": int(args.q),
        "lam": float(args.lam),
        "sigma": float(args.sigma),
        "T0": int(args.T0),
        "T": int(args.T),
    }
    da_imputed = run_p2d2pca_imputation(
        da=base_da,
        replace_mask=replace_mask,
        pca_params=pca_params,
        zero_value=zero_value,
        algo_seed=args.algo_seed,
        verbose=not args.quiet,
    )
    sim3._save_dataarray(da_imputed, "tensor_p2d2pca_imputed", outdir / "tensor_p2d2pca_imputed")

    base_table_df, base_sample_md, base_table_diag = da_to_tempted_inputs_observed_only(
        base_da, label_result.subject_metadata, eps=args.eps, read_depth=args.read_depth
    )
    if args.imputed_input_mode == "observed_only":
        imp_table_df, imp_sample_md, imp_table_diag = da_to_tempted_inputs_observed_only(
            da_imputed, label_result.subject_metadata, eps=args.eps, read_depth=args.read_depth
        )
        imputed_input_mode = "observed_only_after_p2d2pca_imputation"
    else:
        imp_table_df, imp_sample_md, imp_table_diag = da_to_tempted_inputs_grid_filled(
            da_imputed, label_result.subject_metadata, eps=args.eps, read_depth=args.read_depth
        )
        imputed_input_mode = "grid_filled_after_p2d2pca_imputation"
    base_sample_md.to_csv(outdir / "sample_metadata_tempted_direct_observed_only.tsv", sep="\t", index=False)
    imp_sample_md.to_csv(outdir / "sample_metadata_tempted_imputed.tsv", sep="\t", index=False)
    base_table_diag["dropped_samples"].to_csv(dirs["logs"] / "dropped_samples_direct_observed_only.csv", index=False)
    if args.imputed_input_mode == "observed_only":
        imp_table_diag["dropped_samples"].to_csv(dirs["logs"] / "dropped_samples_imputed.csv", index=False)
    else:
        zf = imp_table_diag.get("zero_filled_samples")
        if zf is not None:
            zf.to_csv(dirs["logs"] / "zero_filled_samples_imputed.csv", index=False)

    direct_tempted = run_tempted(base_table_df, base_sample_md, args)
    imputed_tempted = run_tempted(imp_table_df, imp_sample_md, args)

    # Persist feature (taxon) loadings b and temporal loading xi(t) for the
    # component-interpretation panels; failures here must not abort the pipeline.
    for _tres, _pname in (
        (direct_tempted, "direct_tempted"),
        (imputed_tempted, "p2d2pca_imputed_then_tempted"),
    ):
        try:
            export_tempted_loadings(_tres, _pname, outdir)
        except Exception as _exc:  # pragma: no cover - best-effort export
            _write_text(
                _ensure_dir(outdir / "logs") / f"tempted_loadings_export_failure_{_pname}.txt",
                f"{type(_exc).__name__}: {_exc}",
            )

    direct_coords = extract_subject_coordinates(direct_tempted)
    imputed_coords = extract_subject_coordinates(imputed_tempted)
    direct_coords.to_csv(outdir / "tempted_coordinates_direct.csv", index=False)
    imputed_coords.to_csv(outdir / "tempted_coordinates_p2d2pca_imputed.csv", index=False)

    mods = _import_modules("pandas")
    pd = mods["pandas"]
    subject_permanova_rows = []
    for pipeline, coords in [
        ("direct", direct_coords),
        ("p2d2pca_imputed", imputed_coords),
    ]:
        subject_permanova_rows.append(
            run_subject_level_permanova(
                coords,
                label_result.subject_metadata,
                pipeline,
                args,
                "group_label",
            )
        )
    pd.concat(subject_permanova_rows, ignore_index=True).to_csv(outdir / "permanova_subject_level.csv", index=False)

    # ---- Goal 2: matched-subject PERMANOVA (direct/imputed share identical subjects) ----
    try:
        matched_permanova = run_matched_contrast_evaluations(
            direct_coords, imputed_coords, label_result.subject_metadata, args, outdir)
        matched_permanova.to_csv(outdir / "permanova_subject_level_matched.csv", index=False)
    except Exception as exc:
        _write_text(dirs["logs"] / "matched_permanova_block_failure.txt",
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    # ---- FARMM diet-group per-PC loading tests ----
    try:
        group_order = sorted(label_result.subject_metadata["group_label"].dropna().astype(str).unique().tolist())
        group_loading_rows = [
            analyze_group_loading_tests(
                direct_coords,
                label_result.subject_metadata,
                "direct_tempted",
                outdir,
                label_col="group_label",
                group_order=group_order,
            ),
            analyze_group_loading_tests(
                imputed_coords,
                label_result.subject_metadata,
                "p2d2pca_imputed_then_tempted",
                outdir,
                label_col="group_label",
                group_order=group_order,
            ),
        ]
        group_loading = pd.concat(group_loading_rows, ignore_index=True)
        group_loading.to_csv(outdir / "subject_loading_tests_group_label.csv", index=False)
        group_loading.to_csv(outdir / "subject_loading_tests_all_contrasts.csv", index=False)
    except Exception as exc:
        _write_text(dirs["logs"] / "loading_contrast_block_failure.txt",
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    # ---- Matched contrast-specific projection plots ----
    try:
        plot_matched_contrast_embeddings(
            direct_coords, imputed_coords, label_result.subject_metadata, outdir)
    except Exception as exc:
        _write_text(dirs["logs"] / "matched_plot_block_failure.txt",
                    f"{type(exc).__name__}: {exc}\n")

    labels = label_result.subject_metadata[["participant", "group_label"]].copy()
    ml_common_subjects: Any = "NA_not_aligned"
    direct_coords_ml = direct_coords
    imputed_coords_ml = imputed_coords
    labels_ml = labels
    if args.align_subjects_for_ml:
        common = (
            set(direct_coords["participant"].astype(str))
            & set(imputed_coords["participant"].astype(str))
            & set(labels["participant"].astype(str))
        )
        ml_common_subjects = int(len(common))
        direct_coords_ml = direct_coords.loc[direct_coords["participant"].astype(str).isin(common)].copy()
        imputed_coords_ml = imputed_coords.loc[imputed_coords["participant"].astype(str).isin(common)].copy()
        labels_ml = labels.loc[labels["participant"].astype(str).isin(common)].copy()
    metrics_direct = run_ml_evaluation(direct_coords_ml, labels_ml, args, "direct_tempted")
    metrics_imputed = run_ml_evaluation(imputed_coords_ml, labels_ml, args, "p2d2pca_imputed_then_tempted")
    metrics_all = pd.concat([metrics_direct, metrics_imputed], ignore_index=True)
    metrics_summary = summarize_metrics(metrics_all)

    try:
        cluster_rows = [
            run_unsupervised_clustering_evaluation(
                direct_coords_ml,
                labels_ml,
                "direct_tempted",
                outdir,
                n_pcs=args.permanova_n_pcs,
            ),
            run_unsupervised_clustering_evaluation(
                imputed_coords_ml,
                labels_ml,
                "p2d2pca_imputed_then_tempted",
                outdir,
                n_pcs=args.permanova_n_pcs,
            ),
        ]
        pd.concat(cluster_rows, ignore_index=True).to_csv(outdir / "cluster_evaluation.csv", index=False)
    except Exception as exc:
        _write_text(dirs["logs"] / "cluster_evaluation_failure.txt",
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    merged_direct = direct_coords.merge(labels, on="participant", how="left")
    merged_imputed = imputed_coords.merge(labels, on="participant", how="left")
    plot_results(
        metrics_summary,
        {
            "direct": merged_direct,
            "p2d2pca_imputed": merged_imputed,
        },
        outdir,
    )

    metrics_all.to_csv(outdir / "metrics_all.csv", index=False)
    metrics_summary.to_csv(outdir / "metrics_summary.csv", index=False)
    metrics_all.to_csv(outdir / "ml_predictions.csv", index=False)

    tempted_report = [
        "TEMPTED run report",
        "evaluation_mode=transductive_unsupervised_dr",
        "direct_input_mode=observed_only_no_zero_filled_missing_timepoints",
        f"imputed_input_mode={imputed_input_mode}",
        "subject_level_permanova_enabled=True",
        "sample_level_permanova_enabled=False_no_sample_level_tempted_artifact",
        f"permanova_n_pcs={args.permanova_n_pcs}",
        "subject_level_permanova_file=permanova_subject_level.csv",
        "group_label_subject_loading_analysis_enabled=True",
        "group_label_subject_loading_file=subject_loading_tests_group_label.csv",
        "unsupervised_clustering_evaluation_file=cluster_evaluation.csv",
        "The dimensionality reduction stage used all unlabeled participants before train/test label splits.",
        f"tempted_rank={args.tempted_rank}",
        f"tempted_smooth={args.tempted_smooth}",
        f"replicate_handling={args.replicate_handling}",
        f"svd_centralized={args.svd_centralized}",
        f"resolution={args.resolution}",
        f"max_iterations={args.max_iterations}",
        f"tempted_epsilon={args.tempted_epsilon}",
    ]
    _write_text(dirs["logs"] / "tempted_run_report.txt", "\n".join(tempted_report) + "\n")

    preprocessing_report = [
        "Preprocessing report",
        f"participants={base_da.sizes['participant']}",
        f"taxa={base_da.sizes['taxon']}",
        f"bins={base_da.sizes['bin']}",
        f"label_counts={label_result.label_counts}",
        f"replace_entries={mask_report['replace_entries']}",
        "direct_input_mode=observed_only_no_zero_filled_missing_timepoints",
        f"imputed_input_mode={imputed_input_mode}",
        f"direct_observed_samples={base_table_diag.get('n_samples', 'NA')}",
        f"direct_dropped_empty_samples={base_table_diag.get('dropped_empty_sample_count', 'NA')}",
        f"direct_subjects_retained={base_table_diag.get('subjects_retained', 'NA')}",
        f"direct_subjects_dropped_lt2_timepoints={base_table_diag.get('subjects_dropped_lt_min_timepoints', 'NA')}",
        f"imputed_samples={imp_table_diag.get('n_samples', 'NA')}",
        f"imputed_subjects_retained={imp_table_diag.get('subjects_retained', 'NA')}",
        f"imputed_zero_filled_sample_count={imp_table_diag.get('zero_filled_sample_count', 'NA')}",
        "subject_level_permanova_file=permanova_subject_level.csv",
        f"ml_common_subjects_if_aligned={ml_common_subjects}",
        f"all_zero_participant_bin_count={mask_report['all_zero_participant_bin_count']}",
        f"all_zero_taxon_count={mask_report['all_zero_taxon_count']}",
        f"all_zero_participant_count={mask_report['all_zero_participant_count']}",
    ]
    _write_text(dirs["logs"] / "preprocessing_report.txt", "\n".join(preprocessing_report) + "\n")
    return 0


# ============================================================
# Incremental additions (Goals 1-3 + matched plots).
# All functions below are additive and resolved at call time.
# ============================================================


def _hedges_g(group1: Any, group0: Any) -> float:
    """Hedges g for (group1 - group0); small-sample corrected Cohen's d."""
    mods = _import_modules("numpy")
    np = mods["numpy"]
    a = np.asarray(group1, dtype=float)
    b = np.asarray(group0, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n0 = a.size, b.size
    if n1 < 2 or n0 < 2:
        return float("nan")
    v1 = np.var(a, ddof=1)
    v0 = np.var(b, ddof=1)
    pooled = ((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2)
    if not np.isfinite(pooled) or pooled <= 0:
        return float("nan")
    d = (np.mean(a) - np.mean(b)) / math.sqrt(pooled)
    j = 1.0 - (3.0 / (4.0 * (n1 + n0) - 9.0))
    return float(d * j)


def analyze_subject_loading_by_contrast(
    coords: Any,
    subject_metadata: Any,
    pipeline_name: str,
    contrast_name: str,
    label_col: str,
    groups: Sequence[str],
    outdir: Path,
    n_pcs: Optional[int] = None,
) -> Any:
    """Generalised per-PC two-group loading test (group1 - group0).

    groups = [group0, group1]; all signed quantities are group1 minus group0.
    Reuses _benjamini_hochberg for FDR over PCs within this pipeline/contrast.
    """
    mods = _import_modules("numpy", "pandas")
    np = mods["numpy"]
    pd = mods["pandas"]

    columns = [
        "pipeline", "contrast", "label_col", "component",
        "group0", "group1", "n_group0", "n_group1",
        "mean_group0", "mean_group1", "mean_diff_group1_minus_group0",
        "effect_size_group1_minus_group0",
        "mannwhitney_p", "mannwhitney_fdr_bh",
        "welch_t_p", "welch_t_fdr_bh", "error",
    ]
    g0, g1 = str(groups[0]), str(groups[1])

    def _empty(err: str) -> Any:
        return pd.DataFrame([{
            "pipeline": pipeline_name, "contrast": contrast_name, "label_col": label_col,
            "component": None, "group0": g0, "group1": g1,
            "n_group0": 0, "n_group1": 0, "mean_group0": np.nan, "mean_group1": np.nan,
            "mean_diff_group1_minus_group0": np.nan, "effect_size_group1_minus_group0": np.nan,
            "mannwhitney_p": np.nan, "mannwhitney_fdr_bh": np.nan,
            "welch_t_p": np.nan, "welch_t_fdr_bh": np.nan, "error": err,
        }], columns=columns)

    try:
        if coords is None or coords.empty:
            return _empty("No coordinates provided.")
        if label_col not in subject_metadata.columns:
            return _empty(f"label_col '{label_col}' missing from subject metadata.")
        meta = subject_metadata[["participant", label_col]].copy()
        meta["participant"] = meta["participant"].astype(str)
        meta = meta.loc[meta[label_col].astype(str).isin([g0, g1])]
        merged = coords.copy()
        merged["participant"] = merged["participant"].astype(str)
        merged = merged.merge(meta, on="participant", how="inner")
        pc_cols = [c for c in merged.columns if str(c).upper().startswith("PC")]
        if n_pcs is not None and int(n_pcs) > 0:
            pc_cols = pc_cols[: min(int(n_pcs), len(pc_cols))]
        if not pc_cols:
            return _empty("No PC columns available.")

        from scipy.stats import mannwhitneyu, ttest_ind
        rows = []
        for pc in pc_cols:
            vals = pd.to_numeric(merged[pc], errors="coerce")
            a1 = vals.loc[merged[label_col].astype(str).eq(g1) & np.isfinite(vals)].to_numpy(dtype=float)
            a0 = vals.loc[merged[label_col].astype(str).eq(g0) & np.isfinite(vals)].to_numpy(dtype=float)
            row = {
                "pipeline": pipeline_name, "contrast": contrast_name, "label_col": label_col,
                "component": pc, "group0": g0, "group1": g1,
                "n_group0": int(a0.size), "n_group1": int(a1.size),
                "mean_group0": float(np.mean(a0)) if a0.size else np.nan,
                "mean_group1": float(np.mean(a1)) if a1.size else np.nan,
                "mean_diff_group1_minus_group0": (
                    float(np.mean(a1) - np.mean(a0)) if a1.size and a0.size else np.nan
                ),
                "effect_size_group1_minus_group0": _hedges_g(a1, a0),
                "mannwhitney_p": np.nan, "mannwhitney_fdr_bh": np.nan,
                "welch_t_p": np.nan, "welch_t_fdr_bh": np.nan, "error": "",
            }
            if a1.size < 2 or a0.size < 2:
                row["error"] = "Each group requires at least two subjects."
            else:
                try:
                    row["mannwhitney_p"] = float(mannwhitneyu(a1, a0, alternative="two-sided").pvalue)
                except Exception as exc:
                    row["error"] = f"mannwhitney: {type(exc).__name__}: {exc}"
                try:
                    row["welch_t_p"] = float(ttest_ind(a1, a0, equal_var=False).pvalue)
                except Exception as exc:
                    row["error"] = (row["error"] + " | " if row["error"] else "") + \
                        f"welch_t: {type(exc).__name__}: {exc}"
            rows.append(row)

        out = pd.DataFrame(rows, columns=columns)
        out["mannwhitney_fdr_bh"] = _benjamini_hochberg(out["mannwhitney_p"].tolist())
        out["welch_t_fdr_bh"] = _benjamini_hochberg(out["welch_t_p"].tolist())
        return out
    except Exception as exc:
        _write_text(
            _ensure_dir(outdir / "logs") / f"loading_contrast_failure_{pipeline_name}_{contrast_name}.txt",
            f"{type(exc).__name__}: {exc}\n",
        )
        return _empty(f"{type(exc).__name__}: {exc}")


def get_matched_subjects_for_contrast(
    direct_coords: Any,
    imputed_coords: Any,
    subject_metadata: Any,
    label_col: str,
    groups: Sequence[str],
) -> Tuple[List[str], Any]:
    """Participants present in BOTH coordinate sets and in the contrast's groups.

    Returns (sorted_matched_participants, matched_metadata). Raises ValueError on
    label inconsistency so the caller can decide how to surface it.
    """
    mods = _import_modules("pandas")
    pd = mods["pandas"]
    for name, obj in [("direct_coords", direct_coords), ("imputed_coords", imputed_coords),
                      ("subject_metadata", subject_metadata)]:
        if obj is None or "participant" not in obj.columns:
            raise ValueError(f"{name} must contain a 'participant' column.")
    if label_col not in subject_metadata.columns:
        raise ValueError(f"label_col '{label_col}' missing from subject metadata.")

    g = [str(x) for x in groups]
    meta = subject_metadata[["participant", label_col]].copy()
    meta["participant"] = meta["participant"].astype(str)
    meta = meta.loc[meta[label_col].astype(str).isin(g)].copy()

    dup = meta.groupby("participant")[label_col].nunique()
    conflict = dup.loc[dup > 1].index.tolist()
    if conflict:
        raise ValueError(f"Inconsistent {label_col} labels for participants: {conflict[:10]}")
    meta = meta.drop_duplicates(subset=["participant"])

    d_ids = set(direct_coords["participant"].astype(str))
    i_ids = set(imputed_coords["participant"].astype(str))
    matched = sorted(set(meta["participant"]) & d_ids & i_ids)
    matched_meta = meta.loc[meta["participant"].isin(matched)].sort_values("participant").reset_index(drop=True)
    return matched, matched_meta


def subset_coords_by_matched_subjects(coords: Any, matched_subjects: Sequence[str]) -> Any:
    """Return coords restricted to matched_subjects in exactly that order (mean-aggregated)."""
    mods = _import_modules("pandas")
    pd = mods["pandas"]
    df = coords.copy()
    df["participant"] = df["participant"].astype(str)
    pc_cols = [c for c in df.columns if str(c).upper().startswith("PC")]
    for pc in pc_cols:
        df[pc] = pd.to_numeric(df[pc], errors="coerce")
    df = df[["participant"] + pc_cols].groupby("participant", as_index=False)[pc_cols].mean()
    df = df.set_index("participant").reindex([str(s) for s in matched_subjects]).reset_index()
    return df


def run_subject_level_permanova_on_matched_coords(
    coords: Any,
    matched_metadata: Any,
    pipeline_name: str,
    args: argparse.Namespace,
    contrast_name: str,
    label_col: str,
) -> Any:
    """PERMANOVA on a fixed matched subject set, grouped by label_col directly.

    Coordinates are TEMPTED PCs already; no re-PCA. Errors are recorded, not raised.
    """
    mods = _import_modules("numpy", "pandas")
    np = mods["numpy"]
    pd = mods["pandas"]
    columns = [
        "pipeline", "contrast", "label_col", "n_subjects", "n_groups",
        "group_counts_json", "n_pcs", "pc_cols_json", "permutations",
        "pseudo_F", "p_value", "error",
    ]
    row = {
        "pipeline": pipeline_name, "contrast": contrast_name, "label_col": label_col,
        "n_subjects": 0, "n_groups": 0, "group_counts_json": json.dumps({}),
        "n_pcs": 0, "pc_cols_json": json.dumps([]),
        "permutations": int(args.permanova_permutations),
        "pseudo_F": np.nan, "p_value": np.nan, "error": "",
    }
    try:
        if coords is None or coords.empty:
            raise ValueError("No matched coordinates provided.")
        pc_cols = [c for c in coords.columns if str(c).upper().startswith("PC")]
        if not pc_cols:
            raise ValueError("No PC columns available.")
        if int(args.permanova_n_pcs) > 0:
            pc_cols = pc_cols[: min(int(args.permanova_n_pcs), len(pc_cols))]
        row["n_pcs"] = int(len(pc_cols))
        row["pc_cols_json"] = json.dumps([str(c) for c in pc_cols])

        meta = matched_metadata[["participant", label_col]].copy()
        meta["participant"] = meta["participant"].astype(str)
        df = coords.copy()
        df["participant"] = df["participant"].astype(str)
        merged = df.merge(meta, on="participant", how="inner").dropna(subset=["participant", label_col] + pc_cols)
        x = merged[pc_cols].to_numpy(dtype=float)
        finite = np.isfinite(x).all(axis=1)
        merged = merged.loc[finite].reset_index(drop=True)
        x = x[finite]
        group_counts = {
            str(k): int(v)
            for k, v in merged[label_col].astype(str).value_counts().sort_index().items()
        }
        row.update({
            "n_subjects": int(len(merged)),
            "n_groups": int(len(group_counts)),
            "group_counts_json": json.dumps(group_counts, sort_keys=True),
        })
        if len(group_counts) < 2:
            raise ValueError("PERMANOVA requires at least two groups.")
        small = {k: v for k, v in group_counts.items() if v < 2}
        if small:
            raise ValueError(f"Each group needs >=2 matched subjects; small_groups={small}.")
        if merged["participant"].duplicated().any():
            raise ValueError("Duplicate participants after matching.")

        from skbio.stats.distance import DistanceMatrix, permanova
        delta = x[:, None, :] - x[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        ids = merged["participant"].tolist()
        dm = DistanceMatrix(distances, ids=ids)
        grouping = pd.Series(merged[label_col].astype(str).values, index=ids, name=label_col)
        result = permanova(dm, grouping, permutations=int(args.permanova_permutations))
        row.update({"pseudo_F": float(result["test statistic"]), "p_value": float(result["p-value"])})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        _write_text(
            _ensure_dir(Path(args.outdir) / "logs")
            / f"matched_permanova_failure_{pipeline_name}_{contrast_name}.txt",
            row["error"] + "\n",
        )
    return pd.DataFrame([row], columns=columns)


def _build_sample_audit_row(
    contrast_name: str,
    pipeline_name: str,
    label_col: str,
    coords: Any,
    matched_metadata: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    import hashlib
    mods = _import_modules("pandas")
    pd = mods["pandas"]
    pc_cols = [c for c in coords.columns if str(c).upper().startswith("PC")]
    if int(args.permanova_n_pcs) > 0:
        pc_cols = pc_cols[: min(int(args.permanova_n_pcs), len(pc_cols))]
    parts = sorted(coords["participant"].astype(str).tolist())
    joined = ",".join(parts)
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
    preview = joined if len(joined) <= 200 else joined[:200] + f"...(+{len(parts)} total)"
    meta = matched_metadata[["participant", label_col]].copy()
    meta["participant"] = meta["participant"].astype(str)
    gc = {str(k): int(v) for k, v in meta[label_col].astype(str).value_counts().sort_index().items()}
    return {
        "contrast": contrast_name, "pipeline": pipeline_name, "label_col": label_col,
        "n_subjects": int(len(parts)),
        "group_counts_json": json.dumps(gc, sort_keys=True),
        "participant_hash_or_joined_preview": f"md5={digest} | {preview}",
        "n_pc_columns_used": int(len(pc_cols)),
        "pc_cols_json": json.dumps([str(c) for c in pc_cols]),
    }


# Contrast registry shared by matched PERMANOVA, plots and tuning.
CONTRAST_SPECS = [
    {"name": "een_omnivore_vegan", "label_col": "group_label", "groups": ["EEN", "Omnivore", "Vegan"]},
]


def run_matched_contrast_evaluations(
    direct_coords: Any,
    imputed_coords: Any,
    subject_metadata: Any,
    args: argparse.Namespace,
    outdir: Path,
) -> Any:
    """Goal 2 wiring: per-contrast matched-subject PERMANOVA + audits.

    For each contrast, direct and imputed are forced onto an identical, identically
    ordered subject set (AssertionError otherwise). Too-few-subject contrasts are
    logged and skipped rather than crashing.
    """
    mods = _import_modules("pandas")
    pd = mods["pandas"]
    all_rows = []
    for spec in CONTRAST_SPECS:
        cname, label_col, groups = spec["name"], spec["label_col"], spec["groups"]
        try:
            matched, matched_meta = get_matched_subjects_for_contrast(
                direct_coords, imputed_coords, subject_metadata, label_col, groups
            )
        except Exception as exc:
            _write_text(_ensure_dir(outdir / "logs") / f"matched_subjects_failure_{cname}.txt",
                        f"{type(exc).__name__}: {exc}\n")
            continue

        d_sub = subset_coords_by_matched_subjects(direct_coords, matched)
        i_sub = subset_coords_by_matched_subjects(imputed_coords, matched)
        # Hard invariant: identical participants in identical order.
        d_ids = d_sub["participant"].astype(str).tolist()
        i_ids = i_sub["participant"].astype(str).tolist()
        assert d_ids == i_ids, (
            f"Matched participants differ between direct and imputed for contrast {cname}."
        )

        matched_meta.to_csv(outdir / f"matched_subjects_{cname}.csv", index=False)
        audit_rows = [
            _build_sample_audit_row(cname, "direct_tempted", label_col, d_sub, matched_meta, args),
            _build_sample_audit_row(cname, "p2d2pca_imputed_then_tempted", label_col, i_sub, matched_meta, args),
        ]
        pd.DataFrame(audit_rows).to_csv(outdir / f"sample_audit_{cname}.csv", index=False)

        if len(matched) < 4:
            _write_text(_ensure_dir(outdir / "logs") / f"matched_too_few_{cname}.txt",
                        f"Only {len(matched)} matched subjects for contrast {cname}; PERMANOVA skipped.\n")
            continue

        all_rows.append(run_subject_level_permanova_on_matched_coords(
            d_sub, matched_meta, "direct_tempted", args, cname, label_col))
        all_rows.append(run_subject_level_permanova_on_matched_coords(
            i_sub, matched_meta, "p2d2pca_imputed_then_tempted", args, cname, label_col))

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame(columns=[
        "pipeline", "contrast", "label_col", "n_subjects", "n_groups",
        "group_counts_json", "n_pcs", "pc_cols_json", "permutations",
        "pseudo_F", "p_value", "error",
    ])


def run_all_loading_contrasts(direct_coords: Any, imputed_coords: Any, subject_metadata: Any, outdir: Path) -> Dict[str, Any]:
    """Goal 1 wiring: generalised per-PC loading tests for both contrasts/pipelines."""
    mods = _import_modules("pandas")
    pd = mods["pandas"]
    pipelines = [("direct_tempted", direct_coords),
                 ("p2d2pca_imputed_then_tempted", imputed_coords)]
    per_contrast: Dict[str, Any] = {}
    for spec in CONTRAST_SPECS:
        rows = []
        for pname, coords in pipelines:
            rows.append(analyze_subject_loading_by_contrast(
                coords, subject_metadata, pname, spec["name"], spec["label_col"], spec["groups"], outdir))
        per_contrast[spec["name"]] = pd.concat(rows, ignore_index=True)
    return per_contrast


def plot_matched_contrast_embeddings(direct_coords: Any, imputed_coords: Any, subject_metadata: Any, outdir: Path) -> None:
    """Contrast-specific projection plots on matched subjects (PC1 vs PC2)."""
    try:
        mods = _import_modules("matplotlib.pyplot", "pandas")
    except Exception:
        return
    plt = mods["matplotlib.pyplot"]
    pd = mods["pandas"]
    plots_dir = _ensure_dir(outdir / "plots")
    pipelines = [("direct_tempted", direct_coords),
                 ("p2d2pca_imputed_then_tempted", imputed_coords)]
    for spec in CONTRAST_SPECS:
        cname, label_col, groups = spec["name"], spec["label_col"], spec["groups"]
        try:
            matched, matched_meta = get_matched_subjects_for_contrast(
                direct_coords, imputed_coords, subject_metadata, label_col, groups)
        except Exception as exc:
            _write_text(_ensure_dir(outdir / "logs") / f"matched_plot_failure_{cname}.txt",
                        f"{type(exc).__name__}: {exc}\n")
            continue
        if not matched:
            continue
        meta = matched_meta[["participant", label_col]].copy()
        meta["participant"] = meta["participant"].astype(str)
        for pname, coords in pipelines:
            try:
                sub = subset_coords_by_matched_subjects(coords, matched).merge(meta, on="participant", how="left")
                pc_cols = [c for c in sub.columns if str(c).upper().startswith("PC")]
                if len(pc_cols) < 2:
                    continue
                fig, ax = plt.subplots(figsize=(6, 5))
                for label, g in sub.groupby(label_col):
                    ax.scatter(g[pc_cols[0]], g[pc_cols[1]], label=str(label), alpha=0.8)
                ax.set_xlabel(pc_cols[0]); ax.set_ylabel(pc_cols[1])
                ax.set_title(f"{pname} | {cname} (matched n={len(matched)})")
                ax.legend(frameon=False)
                fig.tight_layout()
                fig.savefig(plots_dir / f"matched_embedding_{pname}_{cname}.png", dpi=180)
                plt.close(fig)
            except Exception as exc:
                _write_text(_ensure_dir(outdir / "logs") / f"matched_plot_failure_{pname}_{cname}.txt",
                            f"{type(exc).__name__}: {exc}\n")


# ------------------------- Goal 3: label-aware exploratory tuning -------------------------

def _parse_grid(text: str, cast) -> List[Any]:
    out = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(cast(tok))
    return out


def _mean_metric(ml_df: Any, classifier: str, metric: str) -> float:
    mods = _import_modules("numpy")
    np = mods["numpy"]
    if ml_df is None or ml_df.empty:
        return float("nan")
    sub = ml_df.loc[ml_df["classifier"] == classifier, metric].dropna().to_numpy(dtype=float)
    return float(np.mean(sub)) if sub.size else float("nan")


def _evaluate_param_combo_group_label(
    direct_coords: Any,
    imputed_coords: Any,
    subject_metadata: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Evaluate one param combo on the FARMM matched group-label subject set.

    Returns matched PERMANOVA F/p and CV classifier means for direct and imputed,
    plus deltas. direct_coords is the fixed baseline; only imputed_coords vary.
    """
    mods = _import_modules("numpy", "pandas")
    np = mods["numpy"]
    pd = mods["pandas"]
    spec = CONTRAST_SPECS[0]
    label_col, groups = spec["label_col"], spec["groups"]
    res: Dict[str, Any] = {
        "n_matched_subjects": 0, "group_counts_json": json.dumps({}),
        "imputed_permanova_F": np.nan, "imputed_permanova_p": np.nan,
        "direct_permanova_F": np.nan, "direct_permanova_p": np.nan,
        "delta_permanova_F": np.nan,
        "imputed_logistic_AUROC": np.nan, "direct_logistic_AUROC": np.nan, "delta_logistic_AUROC": np.nan,
        "imputed_logistic_balanced_accuracy": np.nan, "direct_logistic_balanced_accuracy": np.nan,
        "delta_logistic_balanced_accuracy": np.nan,
        "imputed_random_forest_AUROC": np.nan, "direct_random_forest_AUROC": np.nan, "delta_rf_AUROC": np.nan,
        "status": "ok", "error": "",
    }
    try:
        matched, matched_meta = get_matched_subjects_for_contrast(
            direct_coords, imputed_coords, subject_metadata, label_col, groups)
        res["n_matched_subjects"] = int(len(matched))
        gc = {str(k): int(v) for k, v in
              matched_meta[label_col].astype(str).value_counts().sort_index().items()}
        res["group_counts_json"] = json.dumps(gc, sort_keys=True)
        if len(matched) < 4 or len(gc) < 2 or min(gc.values()) < 2:
            res["status"] = "too_few_subjects"
            return res

        d_sub = subset_coords_by_matched_subjects(direct_coords, matched)
        i_sub = subset_coords_by_matched_subjects(imputed_coords, matched)
        assert d_sub["participant"].astype(str).tolist() == i_sub["participant"].astype(str).tolist()
        contrast_name = spec["name"]

        d_perm = run_subject_level_permanova_on_matched_coords(
            d_sub, matched_meta, "direct_tempted", args, contrast_name, label_col)
        i_perm = run_subject_level_permanova_on_matched_coords(
            i_sub, matched_meta, "p2d2pca_imputed_then_tempted", args, contrast_name, label_col)
        res["direct_permanova_F"] = float(d_perm["pseudo_F"].iloc[0])
        res["direct_permanova_p"] = float(d_perm["p_value"].iloc[0])
        res["imputed_permanova_F"] = float(i_perm["pseudo_F"].iloc[0])
        res["imputed_permanova_p"] = float(i_perm["p_value"].iloc[0])
        if np.isfinite(res["imputed_permanova_F"]) and np.isfinite(res["direct_permanova_F"]):
            res["delta_permanova_F"] = res["imputed_permanova_F"] - res["direct_permanova_F"]

        labels = matched_meta[["participant", label_col]].copy()
        labels = labels.rename(columns={label_col: "group_label"}) if label_col != "group_label" else labels
        try:
            ml_d = run_ml_evaluation(d_sub, labels, args, "direct_tempted")
        except Exception as exc:
            ml_d = pd.DataFrame(); res["error"] += f"direct_ml: {type(exc).__name__}: {exc}; "
        try:
            ml_i = run_ml_evaluation(i_sub, labels, args, "p2d2pca_imputed_then_tempted")
        except Exception as exc:
            ml_i = pd.DataFrame(); res["error"] += f"imputed_ml: {type(exc).__name__}: {exc}; "

        res["direct_logistic_AUROC"] = _mean_metric(ml_d, "logistic_regression", "AUROC")
        res["imputed_logistic_AUROC"] = _mean_metric(ml_i, "logistic_regression", "AUROC")
        res["direct_logistic_balanced_accuracy"] = _mean_metric(ml_d, "logistic_regression", "balanced_accuracy")
        res["imputed_logistic_balanced_accuracy"] = _mean_metric(ml_i, "logistic_regression", "balanced_accuracy")
        res["direct_random_forest_AUROC"] = _mean_metric(ml_d, "random_forest", "AUROC")
        res["imputed_random_forest_AUROC"] = _mean_metric(ml_i, "random_forest", "AUROC")
        res["delta_logistic_AUROC"] = res["imputed_logistic_AUROC"] - res["direct_logistic_AUROC"]
        res["delta_logistic_balanced_accuracy"] = (
            res["imputed_logistic_balanced_accuracy"] - res["direct_logistic_balanced_accuracy"])
        res["delta_rf_AUROC"] = res["imputed_random_forest_AUROC"] - res["direct_random_forest_AUROC"]
    except Exception as exc:
        res["status"] = "error"
        res["error"] += f"{type(exc).__name__}: {exc}"
    return res


def _tuning_score(res: Dict[str, Any], mode: str) -> float:
    mods = _import_modules("numpy")
    np = mods["numpy"]
    if res.get("status") != "ok":
        return float("-1e9")
    parts = [
        2.0 * res.get("delta_logistic_AUROC", np.nan),
        1.5 * res.get("delta_logistic_balanced_accuracy", np.nan),
        1.0 * res.get("delta_rf_AUROC", np.nan),
        0.5 * res.get("delta_permanova_F", np.nan),
    ]
    if any(not np.isfinite(p) for p in parts):
        return float("-1e9")
    score = float(sum(parts))
    if int(res.get("n_matched_subjects", 0)) < 6:
        score -= 0.5
    return score


def run_tuning(args: argparse.Namespace) -> int:
    """Label-aware exploratory tuning of penalized_2d2_pca for FARMM group separation.

    Direct observed-only TEMPTED is computed exactly once as a fixed baseline. Each
    param combo runs imputation -> TEMPTED -> matched EEN/Omnivore/Vegan evaluation. The
    best config is then re-run through run_full() to regenerate all canonical outputs.
    """
    mods = _import_modules("numpy", "pandas")
    np = mods["numpy"]
    pd = mods["pandas"]

    outdir = Path(args.outdir)
    dirs = _prepare_output_dirs(outdir)
    _save_config(args, dirs["root"])

    notes = [
        "# Tuning design notes",
        "",
        "This tuning is label-aware and exploratory.",
        "It optimizes EEN/Omnivore/Vegan group-label separation.",
        "It should not be reported as unbiased predictive validation unless validated "
        "in nested split or external cohort.",
        "",
        "## Score definition (tune_score_mode=farmm_group_label_aware)",
        "score = 2.0*delta_logistic_AUROC + 1.5*delta_logistic_balanced_accuracy "
        "+ 1.0*delta_rf_AUROC + 0.5*delta_permanova_F",
        "All deltas are imputed minus direct on the SAME EEN/Omnivore/Vegan matched subject set.",
        "Failed/too-few-subject combos receive score = -1e9 (excluded from the top).",
        "A small -0.5 penalty applies when fewer than 6 matched subjects remain.",
        "PERMANOVA p-values are reported but NOT used for ranking (permutation/sample-size sensitive).",
        "",
    ]
    _write_text(outdir / "tuning_design_notes.md", "\n".join(notes) + "\n")

    label_result = load_and_prepare_labels(args.meta_path)
    label_result.subject_metadata.to_csv(outdir / "subject_metadata.csv", index=False)

    base_da, _, _ = build_real_tensor(
        tax_path=args.tax_path, meta_path=args.meta_path, eps=args.eps,
        min_pos_cells=args.min_pos_cells, min_participants=args.min_participants,
        verbose=not args.quiet,
    )
    replace_mask, mask_report = define_real_replace_mask(
        base_da, eps=args.eps, mode=args.impute_mode, tolerance=args.mask_tolerance)
    zero_value = mask_report["zero_value"]

    # Fixed direct observed-only baseline (run once).
    base_table_df, base_sample_md, _ = da_to_tempted_inputs_observed_only(
        base_da, label_result.subject_metadata, eps=args.eps, read_depth=args.read_depth)
    direct_tempted = run_tempted(base_table_df, base_sample_md, args)
    direct_coords = extract_subject_coordinates(direct_tempted)
    direct_coords.to_csv(dirs["logs"] / "tuning_direct_baseline_coordinates.csv", index=False)

    d_grid = _parse_grid(args.d_grid, int)
    q_grid = _parse_grid(args.q_grid, int)
    lam_grid = _parse_grid(args.lam_grid, float)
    sigma_grid = _parse_grid(args.sigma_grid, float)
    if args.tune_quick:
        d_grid, q_grid = d_grid[:2], q_grid[:1]
        lam_grid, sigma_grid = lam_grid[:2], sigma_grid[:1]

    combos = [(d, q, lam, sg) for d in d_grid for q in q_grid for lam in lam_grid for sg in sigma_grid]
    if len(combos) > int(args.tune_max_combos):
        combos = combos[: int(args.tune_max_combos)]

    partial_path = outdir / "tuning_results_partial.csv"
    results = []
    for idx, (d, q, lam, sg) in enumerate(combos, start=1):
        rec = {"combo_index": idx, "d": d, "q": q, "lam": lam, "sigma": sg,
               "T0": int(args.T0), "T": int(args.T)}
        try:
            pca_params = {"d": int(d), "q": int(q), "lam": float(lam), "sigma": float(sg),
                          "T0": int(args.T0), "T": int(args.T)}
            da_imp = run_p2d2pca_imputation(
                da=base_da, replace_mask=replace_mask, pca_params=pca_params,
                zero_value=zero_value, algo_seed=args.algo_seed, verbose=False)
            if args.imputed_input_mode == "observed_only":
                imp_table_df, imp_sample_md, _ = da_to_tempted_inputs_observed_only(
                    da_imp, label_result.subject_metadata, eps=args.eps, read_depth=args.read_depth)
            else:
                imp_table_df, imp_sample_md, _ = da_to_tempted_inputs_grid_filled(
                    da_imp, label_result.subject_metadata, eps=args.eps, read_depth=args.read_depth)
            imputed_tempted = run_tempted(imp_table_df, imp_sample_md, args)
            imputed_coords = extract_subject_coordinates(imputed_tempted)
            if args.save_all_tuning_coords:
                cd = _ensure_dir(dirs["logs"] / "tuning_coords")
                imputed_coords.to_csv(cd / f"imputed_coords_combo{idx}_d{d}_q{q}_lam{lam}_sigma{sg}.csv", index=False)
            ev = _evaluate_param_combo_group_label(direct_coords, imputed_coords, label_result.subject_metadata, args)
            rec.update(ev)
        except Exception as exc:
            rec.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            _write_text(dirs["logs"] / f"tuning_combo_failure_{idx}.txt",
                        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        rec["score"] = _tuning_score(rec, args.tune_score_mode)
        results.append(rec)
        pd.DataFrame(results).to_csv(partial_path, index=False)  # incremental safety
        if not args.quiet:
            print(f"[tune {idx}/{len(combos)}] d={d} q={q} lam={lam} sigma={sg} "
                  f"status={rec.get('status')} score={rec.get('score'):.4f}")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)
    results_df.to_csv(outdir / "tuning_results.csv", index=False)

    ok = results_df.loc[results_df["status"] == "ok"] if not results_df.empty else results_df
    if ok is None or ok.empty:
        _write_text(dirs["logs"] / "tuning_no_valid_combo.txt",
                    "No parameter combination produced a valid EEN/Omnivore/Vegan evaluation.\n")
        print("No valid tuning combination found; see logs.", file=sys.stderr)
        return 1

    best = ok.iloc[0].to_dict()
    best_config = {"d": int(best["d"]), "q": int(best["q"]), "lam": float(best["lam"]),
                   "sigma": float(best["sigma"]), "T0": int(best["T0"]), "T": int(best["T"]),
                   "score": float(best["score"]), "tune_score_mode": args.tune_score_mode,
                   "delta_logistic_AUROC": best.get("delta_logistic_AUROC"),
                   "delta_rf_AUROC": best.get("delta_rf_AUROC"),
                   "delta_permanova_F": best.get("delta_permanova_F"),
                   "n_matched_subjects": best.get("n_matched_subjects")}
    _write_text(outdir / "best_p2d2pca_config.json",
                json.dumps(best_config, indent=2, default=_json_default))

    # Final canonical run with the best config (regenerates all standard + matched outputs).
    final_outdir = outdir / "best_final_run"
    final_args = argparse.Namespace(**vars(args))
    final_args.tune_p2d2pca = False
    final_args.outdir = str(final_outdir)
    final_args.d = int(best["d"]); final_args.q = int(best["q"])
    final_args.lam = float(best["lam"]); final_args.sigma = float(best["sigma"])
    rc = run_full(final_args)

    # Copy best_* convenience artifacts up to the tuning root.
    try:
        for src, dst in [
            ("tempted_coordinates_p2d2pca_imputed.csv", "best_imputed_tempted_coordinates.csv"),
            ("permanova_subject_level_matched.csv", "best_direct_vs_imputed_matched_permanova.csv"),
            ("metrics_summary.csv", "best_direct_vs_imputed_classifier_summary.csv"),
            ("subject_loading_tests_group_label.csv", "best_subject_loading_tests_group_label.csv"),
        ]:
            sp = final_outdir / src
            if sp.exists():
                (outdir / dst).write_text(sp.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as exc:
        _write_text(dirs["logs"] / "best_copy_failure.txt", f"{type(exc).__name__}: {exc}\n")
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct TEMPTED vs penalized_2d2_pca-imputed TEMPTED on FARMM "
            "metagenomics longitudinal data for EEN/Omnivore/Vegan separation."
        )
    )
    parser.add_argument("--tax_path", type=str, default="kraken_results.tsv")
    parser.add_argument("--meta_path", type=str, default="20200619_farmm_metadata.tsv")
    parser.add_argument("--outdir", type=str, default="artifacts/FARMM_full")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--min_pos_cells", type=int, default=10)
    parser.add_argument("--min_participants", type=int, default=5)
    parser.add_argument("--impute_mode", type=str, default="eps_only", choices=["eps_only", "nan_only", "nan_and_eps"])
    parser.add_argument("--imputed_input_mode", type=str, default="grid_filled", choices=["grid_filled", "observed_only"])
    parser.add_argument("--mask_tolerance", type=float, default=1e-12)
    parser.add_argument("--read_depth", type=int, default=100000)
    parser.add_argument("--tempted_rank", type=int, default=3)
    parser.add_argument("--tempted_smooth", type=float, default=1e-4)
    parser.add_argument("--tempted_epsilon", type=float, default=1e-4)
    parser.add_argument("--replicate_handling", type=str, default="random")
    parser.add_argument("--svd_centralized", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--resolution", type=int, default=101)
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--n_repeats", type=int, default=20)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=2026)
    parser.add_argument("--permanova_permutations", type=int, default=999)
    parser.add_argument("--permanova_n_pcs", type=int, default=2)
    parser.add_argument("--align_subjects_for_ml", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--T0", type=int, default=30)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--algo_seed", type=int, default=0)
    # ---- Goal 3: label-aware exploratory tuning of penalized_2d2_pca ----
    parser.add_argument("--tune_p2d2pca", action="store_true")
    parser.add_argument("--d_grid", type=str, default="2,4,8,12")
    parser.add_argument("--q_grid", type=str, default="2,4,8")
    parser.add_argument("--lam_grid", type=str, default="0,0.01,0.03,0.1,0.3,0.7")
    parser.add_argument("--sigma_grid", type=str, default="2,4,8")
    parser.add_argument("--tune_max_combos", type=int, default=60)
    parser.add_argument("--tune_quick", action="store_true")
    parser.add_argument("--tune_score_mode", type=str, default="farmm_group_label_aware")
    parser.add_argument("--save_all_tuning_coords", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_auto_data_paths(args)

    try:
        if args.dry_run:
            return run_dry_run(args)
        if getattr(args, "tune_p2d2pca", False):
            return run_tuning(args)
        return run_full(args)
    except Exception as exc:
        outdir = Path(args.outdir)
        logs_dir = _ensure_dir(outdir / "logs")
        err_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        _write_text(logs_dir / "failure_traceback.txt", err_text)
        print(err_text, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
