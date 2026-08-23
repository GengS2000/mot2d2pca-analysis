from __future__ import annotations
import csv,itertools,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from robustness.common import load_yaml,output_root
def main():
 cfg=load_yaml(ROOT/'configs/stage_c_ablation.yaml');params=json.loads((output_root()/'aggregate'/'selected_params.json').read_text(encoding='utf-8'))
 path=ROOT/'manifests/stage_c.csv';path.parent.mkdir(parents=True,exist_ok=True);fields=['task_id','dataset','mechanism','fraction','repeat','variant','d','q','lam','sigma','mask_seed','algo_seed']
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();tid=0
  for variant,mech,frac,rep in itertools.product(cfg['variants'],cfg['mechanisms'],cfg['fractions'],range(int(cfg['repeats']))):
   w.writerow(dict(task_id=tid,dataset=cfg['dataset'],mechanism=mech,fraction=frac,repeat=rep,variant=variant,d=params['d'],q=params['q'],lam=params['lam'],sigma=params['sigma'],mask_seed=int(cfg['seed'])+rep+1000*cfg['mechanisms'].index(mech)+10000*cfg['fractions'].index(frac),algo_seed=int(cfg['seed'])+100000+tid));tid+=1
 print(f'{path} ({tid} tasks; variants={",".join(cfg["variants"])})')
if __name__=='__main__':main()
