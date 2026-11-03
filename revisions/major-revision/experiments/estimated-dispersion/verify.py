from pathlib import Path
import tempfile,json,hashlib
import numpy as np
import pandas as pd
import run as experiment
H=Path(__file__).resolve().parent
checks={}
# Exact moment expectations recover an interior theta; negative/zero excess uses the finite Poisson limit.
num=np.array([8.,8.,8.,8.]);den=np.array([4.,0.,-1.,10000.])
np.testing.assert_array_equal(experiment.estimate(num,den),[2.,1000.,1000.,.05]);checks['moment_estimator_boundaries']=True
# Re-run one complete seed/regime to an isolated temporary folder.
with tempfile.TemporaryDirectory() as tmp:
 experiment.HERE=Path(tmp);(experiment.HERE/'results').mkdir()
 fn=experiment.run((41000,'deficit_hetero'))
 assert (experiment.HERE/'results'/fn).read_bytes()==(H/'results'/fn).read_bytes()
checks['deterministic_full_run_reproduction']=True
# Independent direct variance-ratio arithmetic for one raw group.
real,mu,theta,*_=experiment.L.build_truth('deficit_hetero',41000)
draw=experiment.sample_nb(mu,theta,seed=41101)
r=real[:200].astype(float);g=np.log1p(draw[:200]).astype(np.float32).astype(float)
vr=np.var(r,axis=0,ddof=1);vg=np.var(g,axis=0,ddof=1);mask=(np.expm1(r).mean(0)>=.5)&(vr>1e-12)
v=float(np.median(np.abs(np.log2(np.maximum(vg[mask]/vr[mask],1e-12)))))
raw=pd.read_csv(H/'results/41000-deficit_hetero.csv');stored=raw.query('model == "oracle" and dataset == 0 and cell_type == 0').calibration_error.iloc[0]
assert abs(v-stored)<1e-12;checks['independent_primary_metric_arithmetic']=True
seed=pd.read_csv(H/'seed_summary.csv')
# In homogeneous and near-Poisson controls the true global and oracle draws must coincide.
for regime in ['no_deficit','deficit_homog']:
 a=seed.query('regime == @regime and model == "oracle"').set_index('seed')
 b=seed.query('regime == @regime and model == "original_global"').set_index('seed')
 np.testing.assert_array_equal(a[['calibration_error','energy']],b[['calibration_error','energy']])
checks['identical_controls_match_exactly']=True
# Verify within-seed precision/heterogeneity ordering, without treating seed counts as p-values.
ordering={}
for regime in seed.regime.unique():
 for n in [25,100,400]:
  a=seed.query('regime == @regime and model == "estimated_conditional" and training_cells == @n').set_index('seed')
  b=seed.query('regime == @regime and model == "estimated_global" and training_cells == @n').set_index('seed')
  ordering[f'{regime}-{n}']=int((a.calibration_error<b.calibration_error).sum())
checks['conditional_better_than_pooled_seed_counts']=ordering
checks['training_count_leakage_review']='Estimator uses independent train counts and known mean templates only; real_log is accessed exclusively inside evaluate. Group and gene truth enter training-count generation, not the moment estimate.'
checks['input_source_sha256']={str(p.relative_to(experiment.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [experiment.ROOT/'experiments/phase2m_lawgrid_2026-07-09/run_lawgrid.py',experiment.ROOT/'src/architecture/stochastic_decoder.py',experiment.ROOT/'src/evaluation/distributional.py']}
(H/'verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
