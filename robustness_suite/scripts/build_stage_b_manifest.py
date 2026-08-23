from __future__ import annotations
import csv, itertools, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from robustness.common import load_yaml, output_root

def main():
    cfg=load_yaml(ROOT/'configs/stage_b_sensitivity.yaml')
    selected=json.loads((output_root()/'aggregate'/'selected_params.json').read_text(encoding='utf-8'))
    rank=pd.read_csv(output_root()/'aggregate'/'parameter_ranking.csv')
    candidates=[dict(name='selected',**{k:selected[k] for k in ('d','q','lam','sigma')}),dict(name='default',d=1,q=1,lam=.1,sigma=8.)]
    for i,row in rank.dropna(subset=['composite_rank']).sort_values('composite_rank').head(int(cfg['top_k'])).iterrows(): candidates.append(dict(name=f'top_{i}',d=int(row.d),q=int(row.q),lam=float(row.lam),sigma=float(row.sigma)))
    grids={'d':[1,2,3,5],'q':[1,2,3,5],'lam':[0,.01,.1,.5,1.,2.],'sigma':[2.,4.,8.,16.]}
    for key, values in grids.items():
        for value in values:
            item=dict(name=f'{key}_{value}',**{k:selected[k] for k in ('d','q','lam','sigma')}); item[key]=value; candidates.append(item)
    unique={ (c['d'],c['q'],c['lam'],c['sigma']):c for c in candidates }; candidates=list(unique.values())
    path=ROOT/'manifests/stage_b.csv'; path.parent.mkdir(parents=True,exist_ok=True)
    fields=['task_id','dataset','mechanism','fraction','repeat','name','d','q','lam','sigma','mask_seed','algo_seed']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); tid=0
        for mech,frac,rep,c in itertools.product(cfg['mechanisms'],cfg['fractions'],range(int(cfg['repeats'])),candidates):
            w.writerow(dict(task_id=tid,dataset=cfg['dataset'],mechanism=mech,fraction=frac,repeat=rep,mask_seed=int(cfg['seed'])+rep+1000*cfg['mechanisms'].index(mech)+10000*cfg['fractions'].index(frac),algo_seed=int(cfg['seed'])+100000+tid,**c)); tid+=1
    print(f'{path} ({tid} tasks; {len(candidates)} parameter configurations)')
if __name__=='__main__': main()
