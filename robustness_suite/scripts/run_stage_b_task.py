from __future__ import annotations
import argparse,csv,json,sys,traceback
from pathlib import Path
import numpy as np,xarray as xr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from robustness.common import atomic_json,numeric_time,output_root
from robustness.experiments import evaluate,make_mask,run_variant
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'manifests/stage_b.csv');p.add_argument('--task-id',type=int,required=True);a=p.parse_args()
 with a.manifest.open(newline='',encoding='utf-8') as f: row=next((x for x in csv.DictReader(f) if int(x['task_id'])==a.task_id),None)
 if row is None: raise KeyError(a.task_id)
 dest=output_root()/'stage_b'/'tasks'/f'task_{a.task_id:08d}.json'
 if dest.exists():
  if json.loads(dest.read_text(encoding='utf-8')).get('status')=='ok': return
  dest.unlink()
 try:
  da=xr.open_dataarray(ROOT/'cache'/'prepared_data'/f"{row['dataset']}.nc").load();data=da.transpose('bin','participant','taxon').values.astype(float)
  mask=make_mask(data,float(row['fraction']),row['mechanism'],int(row['mask_seed']))
  fit=run_variant(data,np.isfinite(data)&~mask,numeric_time(da.coords['bin'].values),row,int(row['algo_seed']))
  payload={**row,'status':'ok',**evaluate(data,fit['imputed'],mask),'runtime_seconds':fit['runtime_seconds']}
 except Exception as e: payload={**row,'status':'failed','error':str(e),'traceback':traceback.format_exc()}
 atomic_json(dest,payload)
 if payload['status']!='ok': raise RuntimeError(payload['error'])
if __name__=='__main__': main()
