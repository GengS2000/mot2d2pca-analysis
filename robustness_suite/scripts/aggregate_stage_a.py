from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustness.common import atomic_json, load_yaml, output_root


PARAM_KEYS = ["d", "q", "lam", "sigma"]
NUMERIC_COLUMNS = [
    "task_id",
    "outer_repeat",
    "fold",
    "d",
    "q",
    "lam",
    "sigma",
    "nrmse",
    "pearson",
    "zero_fill_nrmse",
    "runtime_seconds",
]


def _coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required = ["task_id", "outer_repeat", "fold", *PARAM_KEYS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Stage A task records lack required fields: {missing}")
    if frame[required].isna().any().any():
        bad = frame.index[frame[required].isna().any(axis=1)].tolist()[:10]
        raise RuntimeError(f"Non-numeric identifiers/parameters in Stage A rows: {bad}")
    frame[["task_id", "outer_repeat", "fold", "d", "q"]] = frame[
        ["task_id", "outer_repeat", "fold", "d", "q"]
    ].astype(int)
    return frame


def _rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["nrmse_rank"] = ranked["validation_nrmse"].rank(method="min")
    ranked["pearson_rank"] = (-ranked["validation_pearson"]).rank(method="min")
    ranked["composite_rank"] = ranked["nrmse_rank"] + ranked["pearson_rank"]
    ranked["complexity"] = ranked["d"] + ranked["q"]
    return ranked


def _one_se_select(ranked: pd.DataFrame, se_column: str) -> tuple[pd.DataFrame, pd.Series, float]:
    best = ranked.sort_values(
        ["composite_rank", "validation_nrmse", "complexity", "runtime_seconds"],
        kind="stable",
    ).iloc[0]
    best_se = float(best[se_column]) if pd.notna(best[se_column]) else 0.0
    threshold = float(best["validation_nrmse"] + best_se)
    one_se = ranked.loc[ranked["validation_nrmse"] <= threshold].copy()
    one_se = one_se.sort_values(
        ["complexity", "runtime_seconds", "validation_nrmse", "composite_rank"],
        kind="stable",
    )
    return one_se, one_se.iloc[0], threshold


def main() -> None:
    cfg = load_yaml(ROOT / "configs" / "stage_a_tuning.yaml")
    runtime_root = output_root()
    task_dir = runtime_root / "stage_a" / "tasks"
    paths = sorted(task_dir.glob("task_*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not rows:
        raise RuntimeError("No Stage A task outputs found.")

    all_results = _coerce_numeric(pd.DataFrame(rows))
    if "status" not in all_results.columns:
        raise RuntimeError("Stage A task records lack status.")
    if all_results["task_id"].duplicated().any():
        duplicates = all_results.loc[all_results["task_id"].duplicated(False), "task_id"].tolist()[:10]
        raise RuntimeError(f"Duplicate Stage A task IDs: {duplicates}")

    aggregate = runtime_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(aggregate / "fold_metrics.csv", index=False)
    failures = all_results.loc[all_results["status"] != "ok"].copy()
    failures.to_csv(aggregate / "failed_tasks.csv", index=False)

    ok = all_results.loc[all_results["status"] == "ok"].copy()
    required_metrics = ["nrmse", "pearson", "zero_fill_nrmse", "runtime_seconds"]
    if ok.empty:
        raise RuntimeError("No successful Stage A task outputs found.")
    if ok[required_metrics].isna().any().any():
        counts = ok[required_metrics].isna().sum().loc[lambda x: x > 0].to_dict()
        raise RuntimeError(f"Successful Stage A rows contain missing metrics: {counts}")

    # The default parameter tuple occurs in both manifest grids. Treat it as one
    # hyperparameter setting; `grid` is provenance, not part of the parameter.
    provenance = (
        all_results.groupby(PARAM_KEYS, as_index=False)["grid"]
        .agg(lambda values: ",".join(sorted(set(map(str, values)))))
        .rename(columns={"grid": "source_grids"})
    )

    # First average folds (and the duplicated default-grid entry) inside each
    # outer repeat. These 30 repeat-level rows are the independent units used
    # for the global standard error and one-SE rule.
    repeat_keys = ["outer_repeat", *PARAM_KEYS]
    repeat_metrics = ok.groupby(repeat_keys, as_index=False).agg(
        validation_nrmse=("nrmse", "mean"),
        fold_nrmse_se=("nrmse", "sem"),
        validation_pearson=("pearson", "mean"),
        zero_fill_nrmse=("zero_fill_nrmse", "mean"),
        runtime_seconds=("runtime_seconds", "mean"),
        successful_fold_tasks=("task_id", "count"),
    )
    repeat_metrics = repeat_metrics.merge(provenance, on=PARAM_KEYS, how="left")
    repeat_metrics.to_csv(aggregate / "outer_repeat_parameter_metrics.csv", index=False)

    expected_tasks = all_results.groupby(PARAM_KEYS).size().rename("expected_fold_tasks")
    successful_tasks = ok.groupby(PARAM_KEYS).size().rename("completed_fold_tasks")
    summary = (
        repeat_metrics.groupby(PARAM_KEYS, as_index=False)
        .agg(
            validation_nrmse=("validation_nrmse", "mean"),
            nrmse_sd=("validation_nrmse", "std"),
            nrmse_se=("validation_nrmse", "sem"),
            validation_pearson=("validation_pearson", "mean"),
            pearson_sd=("validation_pearson", "std"),
            zero_fill_nrmse=("zero_fill_nrmse", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
            completed_outer_repeats=("outer_repeat", "nunique"),
        )
        .merge(provenance, on=PARAM_KEYS, how="left")
        .merge(expected_tasks, on=PARAM_KEYS, how="left")
        .merge(successful_tasks, on=PARAM_KEYS, how="left")
    )
    summary["completed_fold_tasks"] = summary["completed_fold_tasks"].fillna(0).astype(int)
    summary["failure_rate"] = 1 - summary["completed_fold_tasks"] / summary["expected_fold_tasks"]
    summary["eligible"] = (
        (summary["failure_rate"] <= float(cfg["failure_rate_threshold"]))
        & (summary["validation_nrmse"] <= summary["zero_fill_nrmse"])
        & (summary["completed_outer_repeats"] == int(cfg["outer_repeats"]))
    )

    eligible = summary.loc[summary["eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("No parameter setting passed completeness, failure-rate, and zero-fill filters.")
    eligible = _rank_candidates(eligible)
    one_se, selected, threshold = _one_se_select(eligible, "nrmse_se")

    ranking_columns = PARAM_KEYS + ["nrmse_rank", "pearson_rank", "composite_rank", "complexity"]
    summary = summary.merge(eligible[ranking_columns], on=PARAM_KEYS, how="left")
    summary = summary.sort_values(
        ["eligible", "composite_rank", "validation_nrmse"],
        ascending=[False, True, True],
        kind="stable",
    )
    summary.to_csv(aggregate / "parameter_ranking.csv", index=False)
    one_se.assign(one_se_threshold=threshold).to_csv(
        aggregate / "one_se_parameter_set.csv", index=False
    )

    # Reproduce parameter selection independently inside every outer repeat.
    # Each repeat uses its inner-fold mean and the best candidate's fold-level
    # SE for its own one-SE threshold.
    eligible_keys = eligible[PARAM_KEYS]
    repeat_candidates = repeat_metrics.merge(eligible_keys, on=PARAM_KEYS, how="inner")
    selections: list[dict] = []
    for outer_repeat, candidates in repeat_candidates.groupby("outer_repeat", sort=True):
        ranked = _rank_candidates(candidates)
        repeat_one_se, choice, repeat_threshold = _one_se_select(ranked, "fold_nrmse_se")
        selections.append(
            {
                "outer_repeat": int(outer_repeat),
                "d": int(choice["d"]),
                "q": int(choice["q"]),
                "lam": float(choice["lam"]),
                "sigma": float(choice["sigma"]),
                "validation_nrmse": float(choice["validation_nrmse"]),
                "validation_pearson": float(choice["validation_pearson"]),
                "one_se_threshold": repeat_threshold,
                "one_se_set_size": int(len(repeat_one_se)),
            }
        )
    outer_selections = pd.DataFrame(selections)
    outer_selections.to_csv(aggregate / "outer_repeat_selections.csv", index=False)
    frequencies = (
        outer_selections.groupby(PARAM_KEYS, as_index=False)
        .size()
        .rename(columns={"size": "selected_outer_repeats"})
    )
    frequencies["selection_frequency"] = frequencies["selected_outer_repeats"] / int(
        cfg["outer_repeats"]
    )
    frequencies.sort_values(
        ["selected_outer_repeats", "d", "q", "lam", "sigma"],
        ascending=[False, True, True, True, True],
    ).to_csv(aggregate / "parameter_selection_frequency.csv", index=False)

    selection = {
        "d": int(selected["d"]),
        "q": int(selected["q"]),
        "lam": float(selected["lam"]),
        "sigma": float(selected["sigma"]),
        "rule": (
            "failure<=5%, complete outer repeats, no worse than zero-fill, "
            "repeat-level composite NRMSE/Pearson rank, one-SE, minimum d+q, runtime"
        ),
        "independent_unit": "outer_repeat",
        "n_outer_repeats": int(cfg["outer_repeats"]),
        "n_inner_folds": int(cfg["inner_folds"]),
        "n_tasks_seen": int(len(all_results)),
        "n_tasks_ok": int(len(ok)),
        "one_se_threshold": threshold,
        "selection_basis": "inner_validation_only",
        "outer_test_evaluated": False,
        "ready_for_downstream": False,
    }
    atomic_json(aggregate / "selected_params.json", selection)
    audit = {
        "stage": "stage_a",
        "status": "validation_aggregation_complete",
        "n_task_files": len(paths),
        "n_tasks_seen": len(all_results),
        "n_tasks_ok": len(ok),
        "n_tasks_failed": len(failures),
        "n_unique_parameters": int(summary.shape[0]),
        "n_outer_repeats_observed": int(ok["outer_repeat"].nunique()),
        "duplicate_task_ids": 0,
        "independent_unit": "outer_repeat",
        "outer_test_available_in_task_outputs": False,
        "required_next_gate": "evaluate CV-selected, fixed-default, and oracle-best on outer test",
    }
    atomic_json(aggregate / "stage_a_aggregation_audit.json", audit)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
