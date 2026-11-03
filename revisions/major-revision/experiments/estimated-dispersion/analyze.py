from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
files=sorted((HERE/'results').glob('*.csv'))
assert len(files)==120
raw=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
assert len(raw)==120*648
assert not raw.duplicated(['seed','regime','model','training_cells','dataset','cell_type']).any()
assert np.isfinite(raw[['calibration_error','signed_ratio']]).all().all()
assert (raw.groupby(['seed','regime','model','training_cells']).size()==72).all()
assert raw.seed.nunique()==30 and raw.regime.nunique()==4
keys=['seed','regime','model','training_cells']
datasets=raw.groupby(keys+['dataset'])[['calibration_error','signed_ratio','energy','theta_lower_clip_fraction','theta_upper_clip_fraction']].median().reset_index()
seed=datasets.groupby(keys).median(numeric_only=True).drop(columns=['dataset']).reset_index()
seed.to_csv(HERE/'seed_summary.csv',index=False)
rows=[]
for key,g in seed.groupby(['regime','model','training_cells']):
 row=dict(zip(['regime','model','training_cells'],key))
 for metric in ['calibration_error','energy','theta_lower_clip_fraction','theta_upper_clip_fraction']:
  v=g[metric].dropna()
  for label,q in [('median',.5),('q025',.025),('q975',.975)]:row[metric+'_'+label]=float(v.quantile(q)) if len(v) else None
 rows.append(row)
summary=pd.DataFrame(rows);summary.to_csv(HERE/'summary.csv',index=False)
pairs=[]
for regime in sorted(seed.regime.unique()):
 g=seed[seed.regime==regime]
 p=g[g.model=='poisson'].set_index('seed')
 for (model,n),gg in g.groupby(['model','training_cells']):
  for metric in ['calibration_error','energy']:
   d=gg.set_index('seed')[metric]-p[metric];d=d.dropna()
   if len(d):pairs.append(dict(regime=regime,model=model,training_cells=n,metric=metric,n=len(d),wins=int((d<0).sum()),median=float(d.median()),q025=float(d.quantile(.025)),q975=float(d.quantile(.975))))
pd.DataFrame(pairs).to_csv(HERE/'paired_vs_poisson.csv',index=False)
# Distinguish calibration winners from energy winners and finite-sample error.
print(summary[['regime','model','training_cells','calibration_error_median','energy_median']].to_string(index=False))
print('\nORACLE PAIRED EFFECTS\n',pd.DataFrame(pairs).query('model == "oracle"').to_string(index=False))
(HERE/'validation.json').write_text(json.dumps({'files':len(files),'group_rows':len(raw),'seed_summaries':len(seed),'unique_seeds':30,'complete_factorial':True,'finite_primary_metrics':True,'energy_missing_only_for_small_training_sizes':bool(raw.energy.isna().equals(raw.training_cells.isin([25,100]))),'aggregation':'median within dataset, then median across 12 datasets within each seed; paired differences before across-seed summaries','interval':'central 95% of seed outcomes; not a confidence interval'},indent=2)+'\n')
