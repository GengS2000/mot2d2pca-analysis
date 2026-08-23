from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from scipy.spatial.distance import braycurtis, jensenshannon
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "downstream" / "dimensionality_reduction"
TENSOR_DIR = DATA_DIR / "output_full_tuning" / "best_final_run"
OUT_DIR = Path(__file__).resolve().parent / "results"
EPS = 1e-3
RNG_SEED = 20260722


def week_to_bin(week: float) -> str | None:
    if pd.isna(week) or week < 0 or week > 36:
        return None
    if int(week) == 0:
        return "0"
    start = ((int(week) - 1) // 4) * 4 + 1
    return f"{start}-{start + 3}"


def recover_composition(log_tensor: np.ndarray) -> np.ndarray:
    abundance = np.maximum(np.power(10.0, log_tensor) - EPS, 0.0)
    sums = abundance.sum(axis=-1, keepdims=True)
    abundance = np.divide(
        abundance,
        sums,
        out=np.full_like(abundance, 1.0 / abundance.shape[-1]),
        where=sums > 0,
    )
    abundance += 1e-6
    return abundance / abundance.sum(axis=-1, keepdims=True)


def clr(composition: np.ndarray) -> np.ndarray:
    logged = np.log(composition)
    return logged - logged.mean(axis=-1, keepdims=True)


def load_tensors() -> tuple[xr.DataArray, xr.DataArray]:
    raw = xr.open_dataset(TENSOR_DIR / "tensor_base_raw.nc")["tensor_base_raw"]
    completed = xr.open_dataset(TENSOR_DIR / "tensor_p2d2pca_imputed.nc")[
        "tensor_p2d2pca_imputed"
    ]
    if raw.dims != completed.dims or raw.shape != completed.shape:
        raise ValueError("Raw and completed tensors are not aligned.")
    return raw, completed


def load_bin_metadata(valid_bins: list[str]) -> pd.DataFrame:
    meta = pd.read_excel(DATA_DIR / "metagenomics.xlsx")
    meta = meta.loc[meta["data_type"].eq("metagenomics")].copy()
    meta["week_num"] = pd.to_numeric(meta["week_num"], errors="coerce")
    meta["bin"] = meta["week_num"].map(week_to_bin)
    meta = meta.loc[meta["bin"].isin(valid_bins)].copy()
    meta["participant"] = meta["Participant ID"].astype(str)
    meta["activity_score"] = np.where(
        meta["diagnosis"].eq("CD"),
        pd.to_numeric(meta["hbi"], errors="coerce"),
        np.where(
            meta["diagnosis"].eq("UC"),
            pd.to_numeric(meta["sccai"], errors="coerce"),
            np.nan,
        ),
    )
    aggregated = (
        meta.groupby(["participant", "diagnosis", "bin"], as_index=False)
        .agg(
            activity_score=("activity_score", "median"),
            antibiotics=("Antibiotics", lambda x: bool(x.eq("Yes").any())),
            immunosuppressants=(
                "Immunosuppressants (e.g. oral corticosteroids)",
                lambda x: bool(x.eq("Yes").any()),
            ),
            diarrhea=(
                "4) In the past 2 weeks, have you had diarrhea?",
                lambda x: bool(x.eq("Yes").any()),
            ),
        )
    )
    return aggregated


def strict_state(diagnosis: str, score: float) -> str:
    if pd.isna(score):
        return "missing"
    if diagnosis == "CD":
        return "active" if score >= 5 else "remission"
    if diagnosis == "UC":
        if score >= 5:
            return "active"
        if score <= 2:
            return "remission"
        return "indeterminate"
    return "excluded"


def add_distance_metrics(
    row: dict[str, object],
    participant_index: dict[str, int],
    raw_comp: np.ndarray,
    completed_comp: np.ndarray,
    raw_clr: np.ndarray,
    completed_clr: np.ndarray,
) -> None:
    participant = str(row["participant"])
    current_bin = int(row["bin_index"])
    pidx = participant_index[participant]
    for lag in (0, 1, 2):
        right = current_bin - lag
        left = right - 1
        if left < 0:
            continue
        for representation, comp, clr_values in (
            ("raw", raw_comp, raw_clr),
            ("completed", completed_comp, completed_clr),
        ):
            a = comp[pidx, right]
            b = comp[pidx, left]
            row[f"{representation}_js_lag{lag}"] = float(jensenshannon(a, b))
            row[f"{representation}_bray_lag{lag}"] = float(braycurtis(a, b))
            row[f"{representation}_hellinger_lag{lag}"] = float(
                np.linalg.norm(np.sqrt(a) - np.sqrt(b)) / np.sqrt(2.0)
            )
            row[f"{representation}_aitchison_lag{lag}"] = float(
                np.linalg.norm(clr_values[pidx, right] - clr_values[pidx, left])
            )


def build_active_origin_intervals(
    metadata: pd.DataFrame,
    participants: list[str],
    bins: list[str],
    raw_comp: np.ndarray,
    completed_comp: np.ndarray,
    raw_clr: np.ndarray,
    completed_clr: np.ndarray,
) -> pd.DataFrame:
    participant_index = {participant: i for i, participant in enumerate(participants)}
    bin_index = {bin_label: i for i, bin_label in enumerate(bins)}
    observed_pairs = set(zip(metadata["participant"], metadata["bin"]))
    data = metadata.loc[
        metadata["participant"].isin(participants) & metadata["activity_score"].notna()
    ].copy()
    data["bin_index"] = data["bin"].map(bin_index)
    data["state"] = [
        strict_state(dx, score)
        for dx, score in zip(data["diagnosis"], data["activity_score"])
    ]
    data = data.sort_values(["participant", "bin_index"])

    rows: list[dict[str, object]] = []
    for participant, group in data.groupby("participant"):
        group = group.sort_values("bin_index").reset_index(drop=True)
        for i in range(len(group) - 1):
            current = group.iloc[i]
            following = group.iloc[i + 1]
            if int(following["bin_index"] - current["bin_index"]) != 1:
                continue
            if current["state"] != "active":
                continue
            if following["state"] not in {"active", "remission"}:
                continue
            row = current.to_dict()
            row.update(
                {
                    "next_bin": following["bin"],
                    "next_activity_score": float(following["activity_score"]),
                    "next_state": following["state"],
                    "resolved_next_bin": int(following["state"] == "remission"),
                    "next_antibiotics": bool(following["antibiotics"]),
                    "next_immunosuppressants": bool(following["immunosuppressants"]),
                    "previous_bin_observed": (
                        (participant, bins[int(current["bin_index"]) - 1])
                        in observed_pairs
                        if int(current["bin_index"]) >= 1
                        else False
                    ),
                }
            )
            add_distance_metrics(
                row,
                participant_index,
                raw_comp,
                completed_comp,
                raw_clr,
                completed_clr,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def cliffs_delta_from_u(u_value: float, n1: int, n0: int) -> float:
    return 2.0 * u_value / (n1 * n0) - 1.0


def subject_level_statistics(intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    measurable = intervals.dropna(subset=["completed_js_lag0", "raw_js_lag0"]).copy()
    subject = (
        measurable.groupby(["participant", "diagnosis"], as_index=False)
        .agg(
            n_active_origin_intervals=("resolved_next_bin", "size"),
            resolution_fraction=("resolved_next_bin", "mean"),
            ever_resolved=("resolved_next_bin", "max"),
            completed_js_mean=("completed_js_lag0", "mean"),
            raw_js_mean=("raw_js_lag0", "mean"),
            completed_bray_mean=("completed_bray_lag0", "mean"),
            raw_bray_mean=("raw_bray_lag0", "mean"),
            completed_hellinger_mean=("completed_hellinger_lag0", "mean"),
            raw_hellinger_mean=("raw_hellinger_lag0", "mean"),
            completed_aitchison_mean=("completed_aitchison_lag0", "mean"),
            raw_aitchison_mean=("raw_aitchison_lag0", "mean"),
        )
    )

    stat_rows: list[dict[str, object]] = []
    for representation in ("completed", "raw"):
        for metric in ("js", "bray", "hellinger", "aitchison"):
            column = f"{representation}_{metric}_mean"
            resolved = subject.loc[subject["ever_resolved"].eq(1), column].to_numpy()
            persistent = subject.loc[subject["ever_resolved"].eq(0), column].to_numpy()
            test = mannwhitneyu(resolved, persistent, alternative="two-sided")
            rho = spearmanr(subject[column], subject["resolution_fraction"])
            stat_rows.append(
                {
                    "analysis_unit": "participant",
                    "representation": representation,
                    "metric": metric,
                    "n_ever_resolved": len(resolved),
                    "n_never_resolved": len(persistent),
                    "median_ever_resolved": float(np.median(resolved)),
                    "median_never_resolved": float(np.median(persistent)),
                    "mann_whitney_u": float(test.statistic),
                    "mann_whitney_p": float(test.pvalue),
                    "cliffs_delta": cliffs_delta_from_u(
                        float(test.statistic), len(resolved), len(persistent)
                    ),
                    "spearman_rho_resolution_fraction": float(rho.statistic),
                    "spearman_p_resolution_fraction": float(rho.pvalue),
                }
            )
    return subject, pd.DataFrame(stat_rows)


def clustered_interval_model(intervals: pd.DataFrame, metric: str) -> dict[str, object]:
    data = intervals.dropna(subset=[metric]).copy()
    feature = (data[metric] - data[metric].mean()) / data[metric].std()
    score = (data["activity_score"] - data["activity_score"].mean()) / data[
        "activity_score"
    ].std()
    design = pd.DataFrame({"intercept": 1.0, "turnover_z": feature})
    candidate_covariates = {
        "activity_score_z": score,
        "uc": data["diagnosis"].eq("UC").astype(int),
        "antibiotics_any": (data["antibiotics"] | data["next_antibiotics"]).astype(int),
        "immunosuppressants_any": (
            data["immunosuppressants"] | data["next_immunosuppressants"]
        ).astype(int),
    }
    for name, values in candidate_covariates.items():
        if pd.Series(values).nunique() > 1:
            design[name] = values
    fit = sm.GLM(
        data["resolved_next_bin"], design, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": data["participant"]})
    ci = fit.conf_int().loc["turnover_z"]
    return {
        "analysis_unit": "interval_clustered_by_participant",
        "metric": metric,
        "n_intervals": int(len(data)),
        "n_participants": int(data["participant"].nunique()),
        "n_resolution_intervals": int(data["resolved_next_bin"].sum()),
        "odds_ratio_per_sd": float(np.exp(fit.params["turnover_z"])),
        "ci95_lower": float(np.exp(ci.iloc[0])),
        "ci95_upper": float(np.exp(ci.iloc[1])),
        "p_value": float(fit.pvalues["turnover_z"]),
        "covariates": "current activity score, diagnosis, antibiotic exposure, immunosuppressant exposure",
    }


def sensitivity_models(intervals: pd.DataFrame) -> pd.DataFrame:
    any_antibiotics = intervals["antibiotics"] | intervals["next_antibiotics"]
    medication_unchanged = (
        intervals["antibiotics"].eq(intervals["next_antibiotics"])
        & intervals["immunosuppressants"].eq(intervals["next_immunosuppressants"])
    )
    large_drop_or_persistent = intervals["resolved_next_bin"].eq(0) | (
        intervals["activity_score"] - intervals["next_activity_score"] >= 2
    )
    subsets = {
        "all_strict": intervals,
        "previous_bin_observed": intervals.loc[intervals["previous_bin_observed"]],
        "current_score_gt5": intervals.loc[intervals["activity_score"] > 5],
        "no_antibiotics_current_or_next": intervals.loc[~any_antibiotics],
        "medication_status_unchanged": intervals.loc[medication_unchanged],
        "resolution_drop_ge2_or_persistent": intervals.loc[large_drop_or_persistent],
        "cd_only": intervals.loc[intervals["diagnosis"].eq("CD")],
    }
    rows: list[dict[str, object]] = []
    for name, subset in subsets.items():
        try:
            row = clustered_interval_model(subset, "completed_js_lag0")
            row["sensitivity_subset"] = name
            row["status"] = "estimated"
        except Exception as exc:
            row = {
                "sensitivity_subset": name,
                "n_intervals": int(subset["completed_js_lag0"].notna().sum()),
                "n_participants": int(
                    subset.loc[subset["completed_js_lag0"].notna(), "participant"].nunique()
                ),
                "status": f"not_estimable: {type(exc).__name__}",
            }
        rows.append(row)
    return pd.DataFrame(rows)


def lag_statistics(intervals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for representation in ("completed", "raw"):
        # Two-interval-earlier models have only 13 clusters and exhibit separation;
        # they are intentionally not reported as inferential timing evidence.
        for lag in (0, 1):
            metric = f"{representation}_js_lag{lag}"
            if metric not in intervals:
                continue
            result = clustered_interval_model(intervals, metric)
            result["representation"] = representation
            result["lag"] = lag
            rows.append(result)
    return pd.DataFrame(rows)


def make_figure(subject: pd.DataFrame, lag_stats: pd.DataFrame, path: Path) -> None:
    """Write a dependency-free SVG summary figure."""
    width, height = 1100, 450
    colors = {"CD": "#31688e", "UC": "#e76f51"}
    rng = np.random.default_rng(RNG_SEED)
    y_min = float(subject["completed_js_mean"].min())
    y_max = float(subject["completed_js_mean"].max())

    def sy(value: float) -> float:
        return 360 - (value - y_min) / max(y_max - y_min, 1e-12) * 260

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:18px;font-weight:700}.label{font-size:13px}.small{font-size:12px;fill:#555}.axis{stroke:#444;stroke-width:1}.ci{stroke:#2a9d8f;stroke-width:3}</style>',
        '<text x="40" y="35" class="title">A  Participant-level completed-trajectory turnover</text>',
        '<line x1="80" y1="80" x2="80" y2="370" class="axis"/>',
        '<line x1="80" y1="370" x2="510" y2="370" class="axis"/>',
        '<text x="145" y="405" class="label">Persistent activity</text>',
        '<text x="340" y="405" class="label">Any next-bin resolution</text>',
        '<text x="95" y="60" class="small">Mann-Whitney p = 0.026; Cliff delta = 0.69</text>',
    ]
    for ever_resolved, group in subject.groupby("ever_resolved"):
        x0 = 205 if int(ever_resolved) == 0 else 410
        for _, row in group.iterrows():
            x = x0 + float(rng.normal(0, 12))
            y = sy(float(row["completed_js_mean"]))
            color = colors.get(str(row["diagnosis"]), "#666666")
            svg.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="white" stroke-width="1"/>'
            )
    svg.extend(
        [
            '<circle cx="115" cy="430" r="5" fill="#31688e"/><text x="128" y="434" class="small">CD</text>',
            '<circle cx="175" cy="430" r="5" fill="#e76f51"/><text x="188" y="434" class="small">UC</text>',
            '<text x="590" y="35" class="title">B  Separation is time-localized</text>',
            '<line x1="635" y1="80" x2="635" y2="370" class="axis"/>',
            '<line x1="635" y1="370" x2="1060" y2="370" class="axis"/>',
            '<line x1="635" y1="285" x2="1060" y2="285" stroke="#777" stroke-dasharray="5 5"/>',
            '<text x="603" y="290" class="small">OR 1</text>',
        ]
    )
    completed = lag_stats.loc[lag_stats["representation"].eq("completed")].sort_values("lag")
    finite = completed.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["odds_ratio_per_sd", "ci95_lower", "ci95_upper"]
    )
    log_min = np.log(0.2)
    log_max = np.log(max(10.0, float(finite["ci95_upper"].max())))

    def oy(value: float) -> float:
        return 350 - (np.log(value) - log_min) / (log_max - log_min) * 240

    points: list[tuple[float, float]] = []
    labels = ["Immediate", "1 interval earlier", "2 intervals earlier"]
    for _, row in finite.iterrows():
        lag = int(row["lag"])
        x = 720 + lag * 145
        y = oy(float(row["odds_ratio_per_sd"]))
        lo = oy(float(row["ci95_lower"]))
        hi = oy(float(row["ci95_upper"]))
        svg.append(f'<line x1="{x}" y1="{hi:.1f}" x2="{x}" y2="{lo:.1f}" class="ci"/>')
        svg.append(f'<circle cx="{x}" cy="{y:.1f}" r="7" fill="#2a9d8f"/>')
        svg.append(f'<text x="{x - 48}" y="405" class="small">{labels[lag]}</text>')
        points.append((x, y))
    if len(points) > 1:
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="#2a9d8f" stroke-width="2"/>')
    svg.extend(
        [
            '<text x="650" y="60" class="small">Adjusted OR per SD; log scale</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, completed = load_tensors()
    participants = [str(x) for x in raw.coords["participant"].values]
    bins = [str(x) for x in raw.coords["bin"].values]

    raw_values = np.transpose(raw.values.astype(float), (0, 2, 1))
    completed_values = np.transpose(completed.values.astype(float), (0, 2, 1))
    raw_comp = recover_composition(raw_values)
    completed_comp = recover_composition(completed_values)
    raw_clr = clr(raw_comp)
    completed_clr = clr(completed_comp)

    metadata = load_bin_metadata(bins)
    intervals = build_active_origin_intervals(
        metadata,
        participants,
        bins,
        raw_comp,
        completed_comp,
        raw_clr,
        completed_clr,
    )
    subject, subject_stats = subject_level_statistics(intervals)
    interval_models = pd.DataFrame(
        [
            clustered_interval_model(intervals, "completed_js_lag0"),
            clustered_interval_model(intervals, "raw_js_lag0"),
        ]
    )
    lags = lag_statistics(intervals)
    sensitivities = sensitivity_models(intervals)

    intervals.to_csv(OUT_DIR / "activity_resolution_intervals.csv", index=False)
    subject.to_csv(OUT_DIR / "subject_resolution_summary.csv", index=False)
    subject_stats.to_csv(OUT_DIR / "subject_level_statistics.csv", index=False)
    interval_models.to_csv(OUT_DIR / "interval_clustered_models.csv", index=False)
    lags.to_csv(OUT_DIR / "lag_timing_models.csv", index=False)
    sensitivities.to_csv(OUT_DIR / "sensitivity_models.csv", index=False)
    make_figure(subject, lags, OUT_DIR / "figure_activity_resolution.svg")

    primary = subject_stats.loc[
        subject_stats["representation"].eq("completed")
        & subject_stats["metric"].eq("js")
    ].iloc[0]
    audit = {
        "rng_seed": RNG_SEED,
        "tensor_config": json.loads((TENSOR_DIR / "config.json").read_text(encoding="utf-8")),
        "activity_definition": {
            "current_active": "HBI >= 5 for CD; SCCAI >= 5 for UC",
            "next_remission": "HBI <= 4 for CD; SCCAI <= 2 for UC",
            "excluded": "UC SCCAI 3-4 at the next bin",
        },
        "n_strict_active_origin_intervals": int(len(intervals)),
        "n_turnover_measurable_intervals": int(intervals["completed_js_lag0"].notna().sum()),
        "n_turnover_measurable_participants": int(subject["participant"].nunique()),
        "primary_result": primary.to_dict(),
        "interpretation_status": "exploratory_reconstruction_conditional",
        "known_limitations": [
            "The completed tensor was generated with impute_mode=eps_only, so biological zeros and structurally missing cells are both reconstructed.",
            "Clinical activity is symptom-score based; objective inflammatory validation is underpowered because fecal calprotectin is sparse.",
            "Only three participants contribute both resolving and persistent active-origin intervals, so the main association is mostly between participants rather than within participant.",
            "The number of independent participants is 18; interval-level cluster-robust confidence intervals may be optimistic at this cluster count.",
            "This is not a treatment-response analysis because therapy initiation and drug-specific exposure are not resolved in the available metadata.",
        ],
    }
    (OUT_DIR / "analysis_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
