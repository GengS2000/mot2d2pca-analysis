from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from robustness.common import output_root
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['stage_b','stage_c'],required=True);a=p.parse_args();root=output_root(); rows=[json.loads(x.read_text(encoding='utf-8')) for x in sorted((root/a.stage/'tasks').glob('task_*.json'))]
 if not rows: raise RuntimeError('No task outputs')
 all_=pd.DataFrame(rows);out=root/'aggregate'/a.stage;out.mkdir(parents=True,exist_ok=True);all_.to_csv(out/'task_metrics.csv',index=False);all_[all_.status!='ok'].to_csv(out/'failed_tasks.csv',index=False)
 ok=all_[all_.status=='ok'].copy();groups=['mechanism','fraction']+(['name','d','q','lam','sigma'] if a.stage=='stage_b' else ['variant'])
 required=['nrmse','pearson','trajectory_correlation','runtime_seconds']
 if ok.empty:
  errors=all_.loc[all_.status!='ok','error'].value_counts().head(10).to_dict() if 'error' in all_ else {}
  (out/'aggregation_error.json').write_text(json.dumps({'stage':a.stage,'n_tasks_seen':len(all_),'n_tasks_ok':0,'common_errors':errors},indent=2),encoding='utf-8')
  raise RuntimeError(f'No successful {a.stage} tasks. Inspect {out / "failed_tasks.csv"} and aggregation_error.json.')
 missing=[column for column in required if column not in ok.columns]
 if missing: raise RuntimeError(f'Successful task records lack required metrics: {missing}')
 for column in required: ok[column]=pd.to_numeric(ok[column],errors='coerce')
 summary=ok.groupby(groups,as_index=False).agg(n=('task_id','count'),nrmse_mean=('nrmse','mean'),nrmse_sd=('nrmse','std'),pearson_mean=('pearson','mean'),trajectory_correlation_mean=('trajectory_correlation','mean'),runtime_seconds_mean=('runtime_seconds','mean'))
 summary.to_csv(out/'summary.csv',index=False)
 if a.stage=='stage_b':
  best=summary.loc[summary.groupby(['mechanism','fraction'])['nrmse_mean'].idxmin()].copy();best.to_csv(out/'best_by_scenario.csv',index=False)
 else:
  baseline=summary[summary.variant=='full'][['mechanism','fraction','nrmse_mean']].rename(columns={'nrmse_mean':'full_nrmse'})
  summary.merge(baseline,on=['mechanism','fraction'],how='left').assign(relative_loss=lambda x:(x.nrmse_mean-x.full_nrmse)/x.full_nrmse).to_csv(out/'relative_loss_vs_full.csv',index=False)
 print(out)
if __name__=='__main__':main()
