#!/usr/bin/env python3
"""Reproducible audit and unified analysis for robustness-suite stages A/B/C.

This script is deliberately read-only with respect to the experimental outputs.
It writes every derived artifact under --output-dir and aborts formal analysis
when a mandatory integrity check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PARAMS = ["d", "q", "lam", "sigma"]
METRICS = ["nrmse", "pearson", "trajectory_correlation"]
REF = (1, 1, 2.0, 16.0)
PAIR = ["dataset", "mechanism", "fraction", "repeat"]


def pkey(frame):
    return frame[PARAMS].astype(str).agg("|".join, axis=1)


def load_json_tasks(folder):
    records, errors = [], []
    for path in sorted(folder.glob("task_*.json")):
        try:
            with path.open(encoding="utf-8") as f:
                rec = json.load(f)
            rec["_path"] = str(path)
            records.append(rec)
        except Exception as exc:  # report parse failures rather than swallowing them
            errors.append({"path": str(path), "error": repr(exc)})
    return pd.DataFrame(records), errors


def digest(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.name).encode())
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def finite_count(s):
    return int(np.isfinite(pd.to_numeric(s, errors="coerce")).sum())


def bootstrap_mean(x, reps, rng):
    """Percentile CI for a repeat-level mean; returns (mean, lower, upper, n)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if not n:
        return (np.nan, np.nan, np.nan, 0)
    # Batching prevents large temporary arrays for the 10,000-bootstrap default.
    samples = []
    for start in range(0, reps, 1000):
        idx = rng.integers(0, n, size=(min(1000, reps - start), n))
        samples.append(x[idx].mean(axis=1))
    b = np.concatenate(samples)
    return (float(x.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975)), n)


def metric_effect(candidate, reference, metric):
    cand = pd.to_numeric(candidate, errors="coerce").to_numpy(float)
    ref = pd.to_numeric(reference, errors="coerce").to_numpy(float)
    valid = np.isfinite(cand) & np.isfinite(ref)
    absolute = np.full(len(ref), np.nan)
    relative = np.full(len(ref), np.nan)
    if metric == "nrmse":
        absolute[valid] = cand[valid] - ref[valid]
        stable = valid & (np.abs(ref) >= 1e-12)
        relative[stable] = absolute[stable] / ref[stable]
        better = valid & (absolute < 0)
    else:
        absolute[valid] = cand[valid] - ref[valid]
        stable = valid & (np.abs(ref) >= 1e-6)
        relative[stable] = (ref[stable] - cand[stable]) / np.abs(ref[stable])
        better = valid & (absolute > 0)
    return absolute, relative, better, valid, stable


def audit(tasks, manifests, root, out):
    required = {"a": 3600, "b": 7200, "c": 1440}
    checks, problems = [], []
    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed: problems.append(name)
    for stage, expected in required.items():
        df = tasks[stage]
        check(f"{stage}: JSON parseable", not df.empty, f"parsed={len(df)}")
        check(f"{stage}: expected task count", len(df) == expected, f"expected={expected}, observed={len(df)}")
        check(f"{stage}: unique task IDs", df.task_id.astype(str).nunique() == len(df), f"unique={df.task_id.astype(str).nunique()}")
        check(f"{stage}: all status=ok", (df.status == "ok").all(), f"ok={(df.status == 'ok').sum()}/{len(df)}")
        for col in ["nrmse", "runtime_seconds"]:
            check(f"{stage}: finite {col}", finite_count(df[col]) == len(df), f"finite={finite_count(df[col])}/{len(df)}")
    for stage in ["b", "c"]:
        df, man = tasks[stage].copy(), manifests[stage].copy()
        df.task_id, man.task_id = df.task_id.astype(str), man.task_id.astype(str)
        check(f"{stage}: manifest one-to-one task IDs", set(df.task_id) == set(man.task_id) and len(man) == len(df),
              f"json={len(df)}, manifest={len(man)}, intersection={len(set(df.task_id) & set(man.task_id))}")
        common = [c for c in man.columns if c in df.columns and c != "task_id"]
        merged = man.merge(df[["task_id"] + common], on="task_id", suffixes=("_manifest", "_json"), validate="one_to_one")
        bad = 0
        for c in common:
            bad += int((merged[f"{c}_manifest"].astype(str) != merged[f"{c}_json"].astype(str)).sum())
        check(f"{stage}: manifest fields match JSON", bad == 0, f"field mismatches={bad}")
        group = ["dataset", "mechanism", "fraction", "repeat"]
        seed_ok = df.groupby(group, dropna=False).agg(mask_seed=("mask_seed", "nunique"), n_masked=("n_masked", "nunique")).max()
        check(f"{stage}: paired masks consistent", bool((seed_ok <= 1).all()), f"max unique mask_seed/n_masked={seed_ok.to_dict()}")
    missing_rows = []
    for stage, df in tasks.items():
        for m in ["pearson"] + (["trajectory_correlation"] if stage != "a" else []):
            vals = pd.to_numeric(df[m], errors="coerce")
            missing_rows.append({"stage": stage.upper(), "metric": m, "total_tasks": len(df), "finite": int(np.isfinite(vals).sum()), "undefined_or_nonfinite": int((~np.isfinite(vals)).sum()), "rate": float((~np.isfinite(vals)).mean())})
    pd.DataFrame(checks).to_json(out / "integrity_audit.json", orient="records", indent=2)
    pd.DataFrame(missing_rows).to_csv(out / "metric_missingness.csv", index=False)
    pd.DataFrame([{"stage": k.upper(), "expected": required[k], "observed": len(v), "ok": int((v.status == "ok").sum()), "failed": int((v.status != "ok").sum())} for k, v in tasks.items()]).to_csv(out / "task_completeness.csv", index=False)
    return problems, checks


def summarize_a(a, aggregate, tables, figs):
    rank = aggregate["parameter_ranking"].copy()
    rank["parameter_key"] = pkey(rank)
    # Same actual parameter tuple can occur in both source grids: retain best row.
    rank = rank.sort_values(["parameter_key", "validation_nrmse", "nrmse_se"], kind="mergesort").drop_duplicates("parameter_key")
    rank.to_csv(tables / "stage_a_parameter_ranking.csv", index=False)
    one = aggregate["one_se"].copy(); one["parameter_key"] = pkey(one)
    one = one.sort_values(["parameter_key", "validation_nrmse"], kind="mergesort").drop_duplicates("parameter_key")
    one.to_csv(tables / "stage_a_one_se_set.csv", index=False)
    freq = aggregate["frequency"].copy(); freq["parameter_key"] = pkey(freq)
    freq = freq.groupby("parameter_key", as_index=False).agg({**{p:"first" for p in PARAMS}, "successful_fold_count":"sum"})
    # Selection frequencies must be calculated at outer-repeat level, not task/fold level.
    choices = a.groupby(["outer_repeat"] + PARAMS, as_index=False).agg(nrmse=("nrmse", "mean"))
    winners = choices.sort_values(["outer_repeat", "nrmse"] + PARAMS, kind="mergesort").groupby("outer_repeat", as_index=False).first()
    selected = winners.groupby(PARAMS, as_index=False).size().rename(columns={"size":"outer_repeat_selection_count"})
    selected["selection_frequency"] = selected.outer_repeat_selection_count / winners.outer_repeat.nunique()
    selected.to_csv(tables / "stage_a_selection_frequency.csv", index=False)
    summary = {"selected_parameters": dict(zip(PARAMS, REF)), "selection_label": "validation-selected", "outer_test_results_available": False,
               "unique_parameter_tuples": int(rank.parameter_key.nunique()), "outer_repeats": int(winners.outer_repeat.nunique()),
               "one_se_threshold": float(rank.validation_nrmse.min() + rank.loc[rank.validation_nrmse.idxmin(), "nrmse_se"]),
               "mean_runtime_seconds": float(pd.to_numeric(a.runtime_seconds).mean()), "failure_rate": float((a.status != "ok").mean())}
    (tables / "stage_a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    top = rank.nsmallest(12, "validation_nrmse").copy(); top["label"] = top[PARAMS].astype(str).agg(",".join, axis=1)
    ax[0].errorbar(top.validation_nrmse, np.arange(len(top)), xerr=top.nrmse_se, fmt="o", color="#4477AA")
    ax[0].axvline(summary["one_se_threshold"], color="gray", ls="--", label="one-SE threshold")
    ax[0].set(yticks=np.arange(len(top)), yticklabels=top.label, xlabel="Validation NRMSE (lower is better)", title="A: parameter candidates")
    ax[0].legend(fontsize=8)
    selected["label"] = selected[PARAMS].astype(str).agg(",".join, axis=1)
    ax[1].plot(selected.selection_frequency, np.arange(len(selected)), "o", color="#4477AA")
    ax[1].set(yticks=np.arange(len(selected)), yticklabels=selected.label, xlabel="Outer-repeat selection frequency", title="A: selection stability")
    fig.tight_layout(); fig.savefig(figs / "fig_a_parameter_selection.png", dpi=300); fig.savefig(figs / "fig_a_parameter_selection.pdf"); plt.close(fig)
    return summary


def paired_stage(df, stage, tables, figs, reps, seed):
    is_b = stage == "b"; reference_label = "reference" if is_b else "full"
    variants = df.copy()
    variants["_key"] = pkey(variants) if is_b else variants.variant.astype(str)
    refmask = variants._key.eq("1|1|2.0|16.0") if is_b else variants._key.eq("full")
    ref = variants.loc[refmask, PAIR + ["_key"] + METRICS + ["runtime_seconds"]].copy()
    ref = ref.rename(columns={m:f"{m}_ref" for m in METRICS + ["runtime_seconds"]}).drop(columns="_key")
    rows = []; category_rows = []; scen_rows = []
    candidates = sorted(variants.loc[~refmask, "_key"].unique())
    for candidate in candidates:
        c = variants.loc[variants._key.eq(candidate)].copy()
        if is_b:
            changed = [p for p, value in zip(PARAMS, candidate.split("|")) if str(value) != str(dict(zip(PARAMS, REF))[p])]
            varied = changed[0] if len(changed) == 1 else ("default" if candidate == "1|1|0.1|8.0" else "multi_change")
        else: varied = candidate
        for group_name, group in [("overall", c)] + [(f"{mech}|{frac}", sub) for (mech, frac), sub in c.groupby(["mechanism", "fraction"], sort=True)]:
            merged = group.merge(ref, on=PAIR, how="inner", validate="one_to_one")
            scen_rows.append({"candidate":candidate, "varied_parameter_or_variant":varied, "stratum":group_name, "candidate_repeats":len(group), "paired_repeats":len(merged)})
            for metric in METRICS:
                absolute, relative, better, valid, stable = metric_effect(merged[metric], merged[f"{metric}_ref"], metric)
                rng = np.random.default_rng(seed + abs(hash((stage, candidate, group_name, metric))) % 1000000)
                mean, lo, hi, n = bootstrap_mean(absolute, reps, rng)
                _, rlo, rhi, rn = bootstrap_mean(relative, reps, rng)
                rows.append({"candidate":candidate, "reference":reference_label, "varied_parameter_or_variant":varied, "stratum":group_name, "metric":metric,
                             "paired_repeat_n":n, "absolute_difference_mean":mean, "absolute_difference_ci_low":lo, "absolute_difference_ci_high":hi,
                             "relative_loss_mean":float(np.nanmean(relative)) if stable.any() else np.nan, "relative_loss_ci_low":rlo, "relative_loss_ci_high":rhi,
                             "relative_loss_defined_n":int(stable.sum()), "undefined_correlation_rate":float(1 - stable.mean()) if metric != "nrmse" and len(stable) else 0.0,
                             "win_rate":float(better[valid].mean()) if valid.any() else np.nan})
                if metric != "runtime_seconds" and group_name == "overall":
                    relvalid = relative[np.isfinite(relative)]
                    category_rows.append({"candidate":candidate, "varied_parameter_or_variant":varied, "metric":metric,
                                          "loss_le_minus_5pct":int((relvalid <= -.05).sum()), "loss_minus_5_to_0pct":int(((relvalid > -.05)&(relvalid <= 0)).sum()), "loss_gt_10pct":int((relvalid > .10).sum()), "defined_n":len(relvalid)})
            runtime = pd.to_numeric(merged.runtime_seconds) - pd.to_numeric(merged.runtime_seconds_ref)
            rng = np.random.default_rng(seed + abs(hash((stage, candidate, group_name, "runtime"))) % 1000000)
            mean, lo, hi, n = bootstrap_mean(runtime, reps, rng)
            rel = runtime.to_numpy(float) / pd.to_numeric(merged.runtime_seconds_ref).to_numpy(float)
            rows.append({"candidate":candidate, "reference":reference_label, "varied_parameter_or_variant":varied, "stratum":group_name, "metric":"runtime_seconds",
                         "paired_repeat_n":n, "absolute_difference_mean":mean, "absolute_difference_ci_low":lo, "absolute_difference_ci_high":hi,
                         "relative_loss_mean":float(np.nanmean(rel)), "relative_loss_ci_low":np.nan, "relative_loss_ci_high":np.nan,
                         "relative_loss_defined_n":int(np.isfinite(rel).sum()), "undefined_correlation_rate":0., "win_rate":float((runtime < 0).mean())})
    effects = pd.DataFrame(rows); effects.to_csv(tables / f"stage_{stage}_paired_effects.csv", index=False)
    pd.DataFrame(scen_rows).to_csv(tables / f"stage_{stage}_scenario_summary.csv", index=False)
    if is_b:
        pd.DataFrame(category_rows).to_csv(tables / "stage_b_loss_categories.csv", index=False)
        effects.query("metric == 'runtime_seconds'").to_csv(tables / "stage_b_runtime.csv", index=False)
    else:
        effects.query("metric == 'runtime_seconds'").to_csv(tables / "stage_c_runtime.csv", index=False)
        effects.query("metric != 'runtime_seconds'")[['candidate','metric','stratum','win_rate','paired_repeat_n','undefined_correlation_rate']].to_csv(tables / "stage_c_win_rates.csv", index=False)
    # Point + bootstrap interval plots; no heatmaps because B is one-at-a-time.
    plotdata = effects[(effects.metric == "nrmse") & (effects.stratum == "overall")].sort_values("absolute_difference_mean", kind="mergesort")
    fig, ax = plt.subplots(figsize=(9, max(4, .32 * len(plotdata))))
    y = np.arange(len(plotdata)); x = plotdata.absolute_difference_mean.to_numpy();
    ax.errorbar(x, y, xerr=np.vstack([x-plotdata.absolute_difference_ci_low.to_numpy(), plotdata.absolute_difference_ci_high.to_numpy()-x]), fmt="o", color="#4477AA")
    ax.axvline(0, color="black", lw=1); ax.set(yticks=y, yticklabels=plotdata.candidate, xlabel="Paired NRMSE difference (candidate − reference; >0 worse)", title=f"{stage.upper()}: paired NRMSE effects (repeat bootstrap 95% CI)")
    fig.tight_layout(); fig.savefig(figs / ("fig_b_parameter_sensitivity.png" if is_b else "fig_c_ablation_forest.png"), dpi=300); fig.savefig(figs / ("fig_b_parameter_sensitivity.pdf" if is_b else "fig_c_ablation_forest.pdf")); plt.close(fig)
    return effects


def equivalence_audit(c, tables):
    pairs = [("no_temporal", "low_rank_only"), ("no_mask_awareness", "plain_2d2pca")]
    rows=[]
    for left, right in pairs:
        l=c[c.variant.eq(left)]; r=c[c.variant.eq(right)]
        m=l.merge(r, on=PAIR, suffixes=("_left", "_right"), validate="one_to_one")
        for metric in METRICS + ["runtime_seconds"]:
            diff=np.abs(pd.to_numeric(m[f"{metric}_left"],errors="coerce")-pd.to_numeric(m[f"{metric}_right"],errors="coerce"))
            maximum = float(np.nanmax(diff)) if np.isfinite(diff).any() else np.nan
            # Floating-point reductions can differ at machine precision even when
            # implementations are numerically the same. Runtime is not used to
            # judge implementation equivalence because independent executions
            # naturally have scheduling noise.
            equivalent = bool(maximum <= 1e-12) if metric != "runtime_seconds" and np.isfinite(maximum) else False
            rows.append({"variant_a":left,"variant_b":right,"metric":metric,"paired_tasks":len(m),"finite_pairs":int(np.isfinite(diff).sum()),"max_absolute_difference":maximum,"implementation_equivalent":equivalent})
    pd.DataFrame(rows).to_csv(tables / "stage_c_equivalence_audit.csv", index=False)
    return rows


def report(out, a_summary, effects_b, effects_c, equiv, checks, metadata):
    def best_worst(e):
        q=e[(e.metric=="nrmse")&(e.stratum=="overall")].sort_values("absolute_difference_mean", kind="mergesort")
        return q.iloc[0], q.iloc[-1]
    bb,bw=best_worst(effects_b); cb,cw=best_worst(effects_c)
    failures=[x for x in checks if not x["passed"]]
    text=f"""# Unified robustness analysis report

## Material Passport

- Analysis status: ANALYZED (reproducible, existing outputs only)
- Input scope: Stage A/B/C task JSONs and required manifests/aggregate tables
- Statistical unit: outer repeat (A); paired repeat (B/C)
- Bootstrap: {metadata['bootstrap_reps']} repeat-level resamples, seed {metadata['seed']}

## Integrity audit

{len(checks)-len(failures)}/{len(checks)} mandatory checks passed. All task counts and manifest/output correspondences were audited before formal analysis. The earlier `results/aggregate/stage_b/aggregation_error.json` is intentionally not used: it is a stale artifact identified by the analysis framework.

## Design and statistical interpretation

Evidence: Stage A uses validation-selected parameters **(d=1, q=1, lam=2.0, sigma=16.0)**. It is not test-validated: no outer-test result is available. Stage B is a one-at-a-time sensitivity design, so parameter interactions and two-dimensional heatmaps are not estimable. Stage C comparisons use a shared-mask paired design.

## Stage A

Evidence: {a_summary['unique_parameter_tuples']} distinct actual parameter tuples were assessed after de-duplicating source-grid repeats. The selected tuple was chosen from validation; {a_summary['outer_repeats']} outer repeats support the selection-frequency summary. Mean task runtime was {a_summary['mean_runtime_seconds']:.2f} s and observed task failure rate was {a_summary['failure_rate']:.1%}.

## Stage B: one-at-a-time sensitivity

Evidence: The least adverse overall NRMSE comparison is `{bb['candidate']}` (mean paired difference {bb['absolute_difference_mean']:.5f}, 95% CI [{bb['absolute_difference_ci_low']:.5f}, {bb['absolute_difference_ci_high']:.5f}]); the most adverse is `{bw['candidate']}` ({bw['absolute_difference_mean']:.5f}, [{bw['absolute_difference_ci_low']:.5f}, {bw['absolute_difference_ci_high']:.5f}]). Positive NRMSE differences mean worse performance than the validation-selected reference.

Inference: conclusions about a performance plateau or deterioration should be read from the per-stratum paired-effect table, rather than extrapolated to parameter interactions.

## Stage C: ablation

Evidence: The most favorable and adverse overall NRMSE ablation comparisons are `{cb['candidate']}` ({cb['absolute_difference_mean']:.5f}) and `{cw['candidate']}` ({cw['absolute_difference_mean']:.5f}), respectively. The equivalence audit is reported separately and does not treat implementation-equivalent variants as independent component evidence.

## Not estimable from current outputs

DA stability, embedding stability, network stability, complete d×q interaction, complete lambda×sigma interaction, and outer-test generalization are not estimable from the current outputs.

## Statistical-fallacy scan — 11/11 checked

1. Simpson's paradox: overall B effects are accompanied by mechanism/fraction strata.
2. Ecological fallacy: no scenario mean is interpreted as an individual observation.
3. Berkson's paradox: not applicable; no selection on a common outcome was introduced.
4. Collider bias: not applicable; analyses condition only on pre-defined design strata.
5. Base-rate neglect: not applicable; no classification outcome is analysed.
6. Regression to the mean: extreme parameter results are not claimed as component effects.
7. Survivorship bias: failed tasks remain in integrity denominators.
8. Look-elsewhere effect: all candidates are exported; inference is descriptive (no selective p-value claims).
9. Garden of forking paths: reference tuple, metrics, pairing, seed, and bootstrap rule were fixed by the supplied framework.
10. Correlation is not causation: algorithm comparisons do not establish biological causality.
11. Reverse causality: not applicable to fixed algorithm settings and simulated masks.

## Cautious paper-ready conclusion

Evidence supports validation-selected reference parameters and paired, scenario-stratified robustness/ablation descriptions only. Inference beyond the observed one-at-a-time settings or to biological mechanisms would exceed the available design.

## Output index

All tables, figures, audit artifacts, and metadata were generated by `scripts/analyze_unified_results.py` in this directory.
"""
    (out / "unified_analysis_report.md").write_text(text, encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--output-dir", default="results/unified_analysis")
    ap.add_argument("--bootstrap-reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026071501)
    ap.add_argument("--overwrite", action="store_true")
    args=ap.parse_args()
    suite=Path.cwd() if (Path.cwd()/"configs").exists() else Path(__file__).resolve().parents[1]
    results=(suite/args.results_root).resolve(); out=(suite/args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        if not args.overwrite: raise SystemExit(f"Output directory exists: {out}; pass --overwrite to replace it.")
        shutil.rmtree(out)
    tables=out/"tables"; figs=out/"figures"; tables.mkdir(parents=True); figs.mkdir()
    tasks={}; parse_errors={}
    for s in "abc": tasks[s],parse_errors[s]=load_json_tasks(results/f"stage_{s}"/"tasks")
    manifests={s:pd.read_csv(suite/"manifests"/f"stage_{s}.csv", dtype=str) for s in "bc"}
    problems, checks=audit(tasks,manifests,results,out)
    if any(parse_errors.values()): problems.extend([f"{s}: JSON parse failure" for s,v in parse_errors.items() if v])
    if problems:
        (out/"integrity_failure.json").write_text(json.dumps({"problems":problems,"parse_errors":parse_errors},indent=2),encoding="utf-8")
        print(f"INTEGRITY FAILURE: {len(problems)} issue(s). Formal analysis stopped. See {out/'integrity_failure.json'}")
        return 2
    missingness = pd.read_csv(out / "metric_missingness.csv")
    missingness[missingness.stage.eq("B")].to_csv(tables / "stage_b_metric_missingness.csv", index=False)
    missingness[missingness.stage.eq("C")].to_csv(tables / "stage_c_metric_missingness.csv", index=False)
    aggregate={"parameter_ranking":pd.read_csv(results/"aggregate"/"parameter_ranking.csv"),"one_se":pd.read_csv(results/"aggregate"/"one_se_parameter_set.csv"),"frequency":pd.read_csv(results/"aggregate"/"parameter_selection_frequency.csv")}
    a_summary=summarize_a(tasks['a'],aggregate,tables,figs)
    effects_b=paired_stage(tasks['b'],"b",tables,figs,args.bootstrap_reps,args.seed)
    effects_c=paired_stage(tasks['c'],"c",tables,figs,args.bootstrap_reps,args.seed)
    equiv=equivalence_audit(tasks['c'],tables)
    study=pd.DataFrame([
        {"stage":"A","dataset":"ctf_pseudotruth","parameters_or_variants":"39 unique parameter tuples","mechanisms":"validation masks","fractions":"validation split","independent_repeats":30,"task_count":3600,"primary_metrics":"NRMSE, Pearson","statistical_unit":"outer_repeat","pairing_key":"N/A","uncertainty_method":"repeat-level summary","limitations":"no outer test"},
        {"stage":"B","dataset":"ctf_pseudotruth","parameters_or_variants":"16 one-at-a-time configurations","mechanisms":"random and configured mechanisms","fractions":"configured fractions","independent_repeats":30,"task_count":7200,"primary_metrics":"NRMSE, Pearson, trajectory","statistical_unit":"repeat","pairing_key":"dataset×mechanism×fraction×repeat","uncertainty_method":"paired repeat bootstrap 95% CI","limitations":"interactions not identifiable"},
        {"stage":"C","dataset":"ctf_pseudotruth","parameters_or_variants":"8 variants","mechanisms":"random and configured mechanisms","fractions":"configured fractions","independent_repeats":30,"task_count":1440,"primary_metrics":"NRMSE, Pearson, trajectory","statistical_unit":"repeat","pairing_key":"dataset×mechanism×fraction×repeat","uncertainty_method":"paired repeat bootstrap 95% CI","limitations":"downstream stability metrics unavailable"}])
    study.to_csv(tables/"study_design_summary.csv",index=False)
    findings=pd.concat([effects_b.assign(stage="B"),effects_c.assign(stage="C")]).query("stratum == 'overall'")
    findings.to_csv(tables/"main_findings.csv",index=False)
    metadata={"analysis_timestamp_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,"platform":platform.platform(),"pandas":pd.__version__,"numpy":np.__version__,"matplotlib":matplotlib.__version__,"bootstrap_reps":args.bootstrap_reps,"seed":args.seed,"selected_parameter_tuple":dict(zip(PARAMS,REF)),"input_task_files":{s:len(tasks[s]) for s in tasks},"manifest_sha256":{s:digest([suite/"manifests"/f"stage_{s}.csv"]) for s in "bc"},"task_json_sha256":{s:digest(sorted((results/f"stage_{s}"/"tasks").glob("task_*.json"))) for s in "abc"}}
    report(out,a_summary,effects_b,effects_c,equiv,checks,metadata)
    metadata["generated_files"]=sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    (out/"analysis_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"Analysis complete. A/B/C tasks: {len(tasks['a'])}/{len(tasks['b'])}/{len(tasks['c'])}; generated {len(metadata['generated_files'])} files in {out}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
