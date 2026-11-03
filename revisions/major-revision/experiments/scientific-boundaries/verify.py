"""Arithmetic, source-integrity, independence-interface and completeness checks."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from run import fixture,estimate,run,H,ROOT
from analyze import select
from src.architecture.stochastic_decoder import sample_nb
x=pd.concat([pd.read_csv(p) for p in sorted((H/'results').glob('*.csv'))],ignore_index=True)
assert len(x)==60480
assert not x.duplicated(['seed','family','mean_case','split','arm','dataset','cell_type']).any()
assert set(x.seed)==set(range(52000,52020))
assert np.isfinite(x[x.split=='test'][['calibration','energy','correlation_error','mean_error']]).all().all()
# Independent primary-endpoint arithmetic, bypassing the project's metric helper.
b,theta,splits=fixture(52000,'correlated');y=splits['test'][2];mu=splits['test'][0]
syn=np.random.default_rng(452000).poisson(mu)
r=np.log1p(y[0]);s=np.log1p(syn[0]);mask=y[0].mean(0)>=.5
manual=np.median(abs(np.log2(s.var(0,ddof=1)[mask]/r.var(0,ddof=1)[mask])))
row=x[(x.seed==52000)&(x.family=='correlated')&(x.mean_case=='correct')&(x.split=='test')&(x.arm=='poisson')&(x.dataset==0)&(x.cell_type==0)].iloc[0]
assert abs(row.calibration-manual)<1e-12
# Validation-only selection remains identical when test outcomes are permuted.
v=x[x.split=='validation'].groupby(['seed','family','mean_case','dataset','arm']).calibration.median().reset_index()
a=select(v);altered=x.copy();altered.loc[altered.split=='test','calibration']=999
vv=altered[altered.split=='validation'].groupby(['seed','family','mean_case','dataset','arm']).calibration.median().reset_index()
pd.testing.assert_frame_equal(a,select(vv))
# Bounded estimator edge cases; same underlying fixture across the three perturbations.
assert np.array_equal(estimate(np.ones(3),np.array([-1,0,1.])),[1000,1000,1])
assert all(not np.array_equal(splits['train'][0][:,:200],splits[k][0]) for k in ['validation','test'])
# Distributional construction: both residuals have E[Y|mu]=mu, Var=mu+mu^2/theta.
# Independent genes vs shared lognormal factor have different conditional covariance.
rng=np.random.default_rng(910002);n=150000;t=2.;means=np.array([2.,5.]);z=rng.lognormal(-np.log1p(1/t)/2,np.sqrt(np.log1p(1/t)),(n,1));ys=rng.poisson(z*means)
assert np.allclose(ys.mean(0),means,rtol=.02)
assert np.allclose(ys.var(0),means+means**2/t,rtol=.04)
assert abs(np.cov(ys,rowvar=False)[0,1]-5)<.2
# Attenuation preserves the population mean under the stated lognormal size law.
center=np.exp(.4**2/2);ell=np.random.default_rng(8012).lognormal(0,.4,1000000)
assert abs((center+.25*(ell-center)).mean()-center)<.001
# Original inputs remain untouched.
for path,sha in json.loads((H/'biological_source_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
bio=pd.read_csv(H/'biological_per_type.csv');bs=pd.read_csv(H/'biological_support.csv')
assert len(bio)==18*73 and bio.in_common.groupby([bio.stage,bio.seed,bio.arm]).sum().eq(69).all()
for _,row in bs.iterrows():
 g=bio[(bio.stage==row.stage)&(bio.seed==row.seed)&(bio.arm==row.arm)]
 assert np.isclose(g[g.in_common].recall.mean(),row.common_support_mean)
 assert np.isclose(g.recall.fillna(0).mean(),row.coverage_adjusted)
# One complete job must rerun byte-identically; preserve prior bytes if execution fails.
p=H/'results/52000-correlated-biased.csv';before=p.read_bytes();p.unlink()
try:run((52000,'correlated','biased'));assert p.read_bytes()==before
finally:
 if not p.exists():p.write_bytes(before)
# No seed can be hidden outside the explicit scientific-figure axis limits.
ss=pd.read_csv(H/'seed_summary.csv')
for metric,bounds in [('calibration',(-.58,.12)),('energy',(-.42,.12))]:
 q=ss.pivot(index=['family','mean_case','seed'],columns='arm',values=metric)
 vals=q.selected-q.poisson
 assert vals.between(*bounds).all()
for metric,upper in [('correlation_error',.66),('mean_error',.72)]:
 assert ss[ss.arm.isin(['poisson','selected','reference'])][metric].between(0,upper).all()
# Every generated new manuscript macro agrees with its source summary.
import re
ledger=(H.parents[1]/'manuscript/figs/science_numbers.tex').read_text()
macros=dict(re.findall(r'\\newcommand\{\\([^}]+)\}\{([^}]+)\}',ledger))
summary=pd.read_csv(H/'summary.csv');checked=0
for _,rr in summary.iterrows():
 if rr.arm not in ['selected','poisson','reference']:continue
 key='Stress'+{'nb':'NB','correlated':'Corr'}[rr.family]+rr.mean_case.title()+rr.arm.title()
 for metric,suffix in [('calibration','Cal'),('energy','Energy'),('correlation_error','Correlation'),('mean_error','Mean')]:
  assert macros[key+suffix]==f'{rr[metric]:.3f}';checked+=1
for stage in ['fixed','end_to_end']:
 q=bs[bs.stage==stage].pivot(index='seed',columns='arm',values='common_support_mean')
 prefix='Support'+{'fixed':'Fixed','end_to_end':'End'}[stage]
 for arm,suffix in [('poisson','Pois'),('nb_global','Global'),('nb_conditional','Cond')]:
  assert macros[prefix+suffix]==f'{q[arm].mean():.3f}';checked+=1
 for control,suffix in [('poisson','VsPois'),('nb_global','VsGlobal')]:
  assert macros[prefix+suffix]==f'{(q.nb_conditional-q[control]).mean():.3f}';checked+=1
assert checked==82
report={'status':'PASS','new_numeric_macros_verified':checked,'all_plotted_seeds_within_axes':True,'synthetic_runs':120,'group_rows':len(x),'independent_primary_arithmetic':True,'test_outcomes_cannot_change_selection':True,'independent_split_covariates':True,'lognormal_poisson_moment_check':True,'original_biological_inputs_unchanged':True,'common_support_types':69,'biological_seed_arm_rows':18,'deterministic_complete_job':True,'scope':'These checks verify execution and arithmetic, not external validity or biological replication.'}
(H/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
