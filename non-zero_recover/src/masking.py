from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr

from .preprocessing import EXPECTED_DIMS, validate_tensor


def _empty_mask_like(da: xr.DataArray, name: str = "mask") -> xr.DataArray:
    return xr.DataArray(np.zeros(da.shape, dtype=bool), dims=EXPECTED_DIMS, coords=da.coords, name=name)


def _finalize_mask(da_true: xr.DataArray, mask: np.ndarray, zero_value: float, meta: Dict[str, Any]) -> Tuple[xr.DataArray, xr.DataArray, Dict[str, Any]]:
    mask_da = xr.DataArray(mask.astype(bool), dims=EXPECTED_DIMS, coords=da_true.coords, name="artificial_mask")
    vals = da_true.values.astype(float).copy()
    vals[mask] = float(zero_value)
    masked_da = xr.DataArray(vals, dims=EXPECTED_DIMS, coords=da_true.coords, name="masked_log_profile")
    eligible_n = int(meta.get("eligible_n", 0))
    n_masked = int(mask.sum())
    meta = {**meta, "n_masked": n_masked, "mask_rate": float(n_masked / eligible_n) if eligible_n else 0.0}
    return mask_da, masked_da, meta


def _choose_from_indices(indices: np.ndarray, n: int, rng: np.random.Generator, weights: Optional[np.ndarray] = None) -> np.ndarray:
    n = int(min(max(n, 0), len(indices)))
    if n == 0:
        return np.empty((0, indices.shape[1]), dtype=int)
    p = None
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        if weights.sum() > 0:
            p = weights / weights.sum()
    choice = rng.choice(len(indices), size=n, replace=False, p=p)
    return indices[choice]


def random_entry_masking(da_true: xr.DataArray, observed_positive_mask: xr.DataArray, frac: float, seed: int, zero_value: float, **params):
    """Randomly mask a fraction of truly positive profile entries."""
    validate_tensor(da_true, "da_true")
    rng = np.random.default_rng(int(seed))
    eligible = observed_positive_mask.values.astype(bool)
    indices = np.argwhere(eligible)
    target = int(round(float(frac) * len(indices)))
    selected = _choose_from_indices(indices, target, rng)
    mask = np.zeros(da_true.shape, dtype=bool)
    if selected.size:
        mask[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return _finalize_mask(da_true, mask, zero_value, {"mechanism": "random_entry", "frac": float(frac), "seed": int(seed), "eligible_n": int(len(indices))})


def low_depth_sample_masking(da_true: xr.DataArray, observed_positive_mask: xr.DataArray, frac: float, seed: int, zero_value: float, **params):
    """Bias masking toward participant-bin slices with fewer observed taxa or lower total profile."""
    rng = np.random.default_rng(int(seed))
    eligible = observed_positive_mask.values.astype(bool)
    raw = np.maximum(10.0 ** da_true.values - 10.0 ** float(zero_value), 0.0)
    counts = eligible.sum(axis=1)
    totals = np.where(eligible, raw, 0.0).sum(axis=1)
    score = counts + totals / (np.nanmedian(totals[totals > 0]) + 1e-12)
    slice_weight = 1.0 / (score + 1.0)
    weights = np.repeat(slice_weight[:, None, :], da_true.shape[1], axis=1)[eligible]
    indices = np.argwhere(eligible)
    target = int(round(float(frac) * len(indices)))
    selected = _choose_from_indices(indices, target, rng, weights=weights)
    mask = np.zeros(da_true.shape, dtype=bool)
    if selected.size:
        mask[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return _finalize_mask(da_true, mask, zero_value, {"mechanism": "low_depth_sample", "frac": float(frac), "seed": int(seed), "eligible_n": int(len(indices))})


def time_point_masking(da_true: xr.DataArray, observed_positive_mask: xr.DataArray, frac: float, seed: int, zero_value: float, **params):
    """Mask positives within one or more time bins."""
    rng = np.random.default_rng(int(seed))
    eligible = observed_positive_mask.values.astype(bool)
    n_bins = int(params.get("n_bins", max(1, round(float(frac) * da_true.sizes["bin"]))))
    bins = rng.choice(da_true.sizes["bin"], size=min(n_bins, da_true.sizes["bin"]), replace=False)
    candidate = eligible.copy()
    keep_bin = np.zeros(da_true.sizes["bin"], dtype=bool)
    keep_bin[bins] = True
    candidate[:, :, ~keep_bin] = False
    indices = np.argwhere(candidate)
    target = int(round(float(frac) * int(eligible.sum())))
    selected = _choose_from_indices(indices, target, rng)
    mask = np.zeros(da_true.shape, dtype=bool)
    if selected.size:
        mask[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return _finalize_mask(da_true, mask, zero_value, {"mechanism": "time_point", "frac": float(frac), "seed": int(seed), "eligible_n": int(eligible.sum()), "selected_bins": [str(da_true.coords["bin"].values[i]) for i in bins]})


def subject_specific_dropout(da_true: xr.DataArray, observed_positive_mask: xr.DataArray, frac: float, seed: int, zero_value: float, **params):
    """Mask positives from a subset of participants."""
    rng = np.random.default_rng(int(seed))
    eligible = observed_positive_mask.values.astype(bool)
    n_subjects = int(params.get("n_subjects", max(1, round(float(frac) * da_true.sizes["participant"]))))
    participants = rng.choice(da_true.sizes["participant"], size=min(n_subjects, da_true.sizes["participant"]), replace=False)
    candidate = eligible.copy()
    keep = np.zeros(da_true.sizes["participant"], dtype=bool)
    keep[participants] = True
    candidate[~keep, :, :] = False
    indices = np.argwhere(candidate)
    target = int(round(float(frac) * int(eligible.sum())))
    selected = _choose_from_indices(indices, target, rng)
    mask = np.zeros(da_true.shape, dtype=bool)
    if selected.size:
        mask[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return _finalize_mask(da_true, mask, zero_value, {"mechanism": "subject_specific_dropout", "frac": float(frac), "seed": int(seed), "eligible_n": int(eligible.sum()), "selected_participants": [str(da_true.coords["participant"].values[i]) for i in participants]})


def consecutive_time_window_dropout(da_true: xr.DataArray, observed_positive_mask: xr.DataArray, frac: float, seed: int, zero_value: float, **params):
    """Mask positives inside participant-specific consecutive 2-3 bin windows."""
    rng = np.random.default_rng(int(seed))
    eligible = observed_positive_mask.values.astype(bool)
    n_subjects = int(params.get("n_subjects", max(1, round(float(frac) * da_true.sizes["participant"]))))
    participants = rng.choice(da_true.sizes["participant"], size=min(n_subjects, da_true.sizes["participant"]), replace=False)
    candidate = np.zeros(da_true.shape, dtype=bool)
    windows = {}
    for p in participants:
        width = int(rng.integers(2, min(4, da_true.sizes["bin"] + 1)))
        start = int(rng.integers(0, max(1, da_true.sizes["bin"] - width + 1)))
        candidate[p, :, start : start + width] = eligible[p, :, start : start + width]
        windows[str(da_true.coords["participant"].values[p])] = [str(x) for x in da_true.coords["bin"].values[start : start + width]]
    indices = np.argwhere(candidate)
    target = int(round(float(frac) * int(eligible.sum())))
    selected = _choose_from_indices(indices, target, rng)
    mask = np.zeros(da_true.shape, dtype=bool)
    if selected.size:
        mask[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return _finalize_mask(da_true, mask, zero_value, {"mechanism": "consecutive_time_window_dropout", "frac": float(frac), "seed": int(seed), "eligible_n": int(eligible.sum()), "windows": windows})


def low_abundance_biased_masking(da_true: xr.DataArray, observed_positive_mask: xr.DataArray, frac: float, seed: int, zero_value: float, **params):
    """Mask positives with probability inversely related to relative abundance."""
    rng = np.random.default_rng(int(seed))
    eligible = observed_positive_mask.values.astype(bool)
    indices = np.argwhere(eligible)
    abundance = np.maximum(10.0 ** da_true.values[eligible] - 10.0 ** float(zero_value), 0.0)
    weights = 1.0 / (abundance + float(params.get("small_const", 1e-6)))
    weights = np.minimum(weights, np.nanquantile(weights, 0.99)) if weights.size else weights
    target = int(round(float(frac) * len(indices)))
    selected = _choose_from_indices(indices, target, rng, weights=weights)
    mask = np.zeros(da_true.shape, dtype=bool)
    if selected.size:
        mask[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return _finalize_mask(da_true, mask, zero_value, {"mechanism": "low_abundance_biased", "frac": float(frac), "seed": int(seed), "eligible_n": int(len(indices))})


MASKING_FUNCTIONS = {
    "random_entry": random_entry_masking,
    "low_depth_sample": low_depth_sample_masking,
    "time_point": time_point_masking,
    "subject_specific_dropout": subject_specific_dropout,
    "consecutive_time_window_dropout": consecutive_time_window_dropout,
    "low_abundance_biased": low_abundance_biased_masking,
}


def make_masked_dataset(
    da_true: xr.DataArray,
    observed_positive_mask: xr.DataArray,
    mechanism: str,
    frac: float,
    seed: int,
    zero_value: float,
    mechanism_params: Optional[Dict[str, Any]] = None,
):
    """Dispatch a semi-synthetic masking mechanism."""
    validate_tensor(da_true, "da_true")
    validate_tensor(observed_positive_mask, "observed_positive_mask")
    mechanism_params = {} if mechanism_params is None else dict(mechanism_params)
    if mechanism not in MASKING_FUNCTIONS:
        raise ValueError(f"Unsupported masking mechanism: {mechanism}")
    return MASKING_FUNCTIONS[mechanism](da_true, observed_positive_mask, frac, seed, zero_value, **mechanism_params)


def detect_nonbiological_zeros(
    masked_da: xr.DataArray,
    zero_value: float,
    strategy: str,
    observed_positive_threshold: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
) -> xr.DataArray:
    """Predict which zero-valued entries should be replaced by imputation."""
    validate_tensor(masked_da, "masked_da")
    params = {} if params is None else dict(params)
    vals = masked_da.values.astype(float)
    zero = np.isfinite(vals) & (vals <= float(zero_value) + 1e-12)
    if strategy in {"all_zero", "detector_all_zero"}:
        pred = zero
    elif strategy in {"longitudinal_neighbor", "detector_longitudinal_neighbor"}:
        min_neighbors = int(params.get("min_positive_neighbors", 1))
        positive = vals > float(zero_value) + 1e-12
        pred = np.zeros_like(zero, dtype=bool)
        for t in range(vals.shape[2]):
            neigh = np.zeros(vals.shape[:2], dtype=int)
            if t > 0:
                neigh += positive[:, :, t - 1]
            if t + 1 < vals.shape[2]:
                neigh += positive[:, :, t + 1]
            pred[:, :, t] = zero[:, :, t] & (neigh >= min_neighbors)
    elif strategy == "taxon_prevalence":
        threshold = float(observed_positive_threshold if observed_positive_threshold is not None else params.get("min_prevalence", 0.1))
        prevalence = (vals > float(zero_value) + 1e-12).mean(axis=(0, 2))
        pred = zero & (prevalence[None, :, None] >= threshold)
    else:
        raise ValueError(f"Unsupported zero detector strategy: {strategy}")
    return xr.DataArray(pred, dims=EXPECTED_DIMS, coords=masked_da.coords, name=f"predicted_{strategy}_mask")


def zero_detection_metrics(true_mask: xr.DataArray, predicted_mask: xr.DataArray) -> Dict[str, Any]:
    """Compute binary classification metrics for artificial zero detection."""
    y = true_mask.values.astype(bool).ravel()
    p = predicted_mask.values.astype(bool).ravel()
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    tn = int(np.sum(~y & ~p))
    fn = int(np.sum(y & ~p))
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if np.isfinite(precision) and np.isfinite(sensitivity) and (precision + sensitivity) else np.nan
    return {"sensitivity": sensitivity, "precision": precision, "specificity": specificity, "f1": f1, "predicted_n": int(p.sum()), "true_n": int(y.sum()), "tp": tp, "fp": fp, "tn": tn, "fn": fn}

