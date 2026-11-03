from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
METRICS=['calibration','energy','correlation_error','mean_error']

def select(validation):
 # Validation alone enters this interface; fixed arm order resolves ties.
 return validation.pivot(index=['seed','family','mean_case','dataset'],columns='arm',values='calibration')[['poisson','pooled','conditional']].idxmin(axis=1).rename('selected').reset_index()

def stress():
 files=sorted((H/'results').glob('*.csv'));assert len(files)==120
 x=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);assert len(x)==60480
 assert x.groupby(['seed','family','mean_case','split','arm']).size().eq(72).all()
 assert np.isfinite(x.loc[x.split=='test',METRICS].to_numpy()).all()
 ds=x.groupby(['seed','family','mean_case','split','arm','dataset'])[METRICS].median().reset_index()
 v=ds[ds.split=='validation'];test=ds[ds.split=='test'];choices=select(v)
 choices.to_csv(H/'validation_choices.csv',index=False)
 selected=test.merge(choices,on=['seed','family','mean_case','dataset']);selected=selected[selected.arm==selected.selected].copy();selected['arm']='selected'
 scores=pd.concat([test,selected],ignore_index=True).groupby(['seed','family','mean_case','arm'])[METRICS].median().reset_index();scores.to_csv(H/'seed_summary.csv',index=False)
 records=[]
 for key,g in scores.groupby(['family','mean_case','arm']):
  row=dict(zip(['family','mean_case','arm'],key))
  for metric in METRICS:
   row[metric]=g[metric].median();row[metric+'_lo']=g[metric].quantile(.025);row[metric+'_hi']=g[metric].quantile(.975)
  records.append(row)
 pd.DataFrame(records).to_csv(H/'summary.csv',index=False)
 pairs=[]
 for (fam,mean),g in scores.groupby(['family','mean_case']):
  for metric in METRICS:
   p=g.pivot(index='seed',columns='arm',values=metric);d=p.selected-p.poisson
   pairs.append(dict(family=fam,mean_case=mean,metric=metric,median=d.median(),lo=d.quantile(.025),hi=d.quantile(.975),wins=int((d<0).sum())))
 pd.DataFrame(pairs).to_csv(H/'paired_effects.csv',index=False)
 # Optimistic test-best arm is only a diagnostic comparator, never the deployed choice.
 best=test[test.arm!='reference'].groupby(['seed','family','mean_case','dataset']).calibration.min().rename('test_best')
 reg=selected.set_index(['seed','family','mean_case','dataset']).calibration-best
 assert reg.min()>-1e-12
 reg.rename('regret').reset_index().to_csv(H/'selection_regret.csv',index=False)
 print(pd.DataFrame(records)[['family','mean_case','arm']+METRICS].round(4).to_string(index=False))


def biological():
 fixed=[json.loads((ROOT/f'experiments/phase2q_biological_2026-07-10/results/biological_seed{s}.json').read_text()) for s in [0,1,2]]
 e2e=[json.loads((ROOT/f'experiments/phase2v_e2e_2026-07-19/results/e2e_generation_seed{s}.json').read_text()) for s in [1,2,3]]
 target=set(fixed[0]['marker_recovery']['poisson']['per_type'])
 maps={}
 for d in fixed:
  for a,v in d['marker_recovery'].items():maps[('fixed',d['seed'],a)]=v['per_type']
 for d in e2e:
  for a,v in d['arms'].items():maps[('end_to_end',d['seed'],a)]=v['marker_recall_per_type']
 common=set.intersection(*(set(v) for v in maps.values()));assert all(set(v)<=target for v in maps.values())
 rows=[];per=[]
 for (stage,seed,arm),v in maps.items():
  arm=arm.replace('e2e_nb_gene-cell','nb_conditional').replace('e2e_nb_gene','nb_global').replace('e2e_poisson','poisson')
  rows.append(dict(stage=stage,seed=seed,arm=arm,n_scoreable=len(v),n_common=len(common),n_target=len(target),original_support_mean=np.mean(list(v.values())),common_support_mean=np.mean([v[t] for t in sorted(common)]),coverage_adjusted=np.mean([v.get(t,0.) for t in sorted(target)])))
  for t in sorted(target):per.append(dict(stage=stage,seed=seed,arm=arm,cell_type=t,scoreable=t in v,in_common=t in common,recall=v.get(t,None)))
 b=pd.DataFrame(rows);b.to_csv(H/'biological_support.csv',index=False);pd.DataFrame(per).to_csv(H/'biological_per_type.csv',index=False)
 report={'target_types':len(target),'common_types':len(common),'excluded_from_common':sorted(target-common),'interpretation':'Descriptive support sensitivity from saved results; no new neural fit; stages have independent fit seeds and are not paired with each other. Unscoreable assigned zero only in explicitly named coverage-adjusted composite. Global-vs-conditional end-to-end includes separate fits; conditional-vs-Poisson shares means within a seed.'}
 effects=[]
 for stage,g in b.groupby('stage'):
  for metric in ['original_support_mean','common_support_mean','coverage_adjusted']:
   p=g.pivot(index='seed',columns='arm',values=metric)
   for control in ['poisson','nb_global']:
    d=p.nb_conditional-p[control];effects.append(dict(stage=stage,metric=metric,control=control,mean=float(d.mean()),minimum=float(d.min()),maximum=float(d.max()),seed_differences=list(d)))
 report['effects']=effects
 (H/'biological_support.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 paths=list((ROOT/'experiments/phase2q_biological_2026-07-10/results').glob('biological_seed*.json'))+list((ROOT/'experiments/phase2v_e2e_2026-07-19/results').glob('e2e_generation_seed*.json'))
 (H/'biological_source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')

if __name__=='__main__':
 import sys
 if '--bio-only' not in sys.argv:stress()
 biological()
