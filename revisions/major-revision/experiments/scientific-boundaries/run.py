"""Held-out scientific stress tests; see pre-execution design.json."""
from pathlib import Path
import sys,json,csv,time,concurrent.futures
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT))
from src.architecture.stochastic_decoder import sample_nb
from src.evaluation.distributional import r_var_per_gene_expressed,energy_distance
NG=72;G=150

def estimate(num,den):
 out=np.full(num.shape,1000.);np.divide(num,den,out=out,where=den>0)
 return np.clip(out,.05,1000.)

def fixture(seed,family):
 rng=np.random.default_rng(seed)
 beta=rng.gamma(.6,1.5,(NG,G));theta=np.exp(rng.uniform(np.log(.3),np.log(8),(NG,1,1)))
 splits={}
 for j,(split,n) in enumerate([('train',400),('validation',200),('test',200),('reference',200)]):
  r=np.random.default_rng(seed+10000*(j+1));ell=r.lognormal(0,.4,(NG,n,1))
  mu=beta[:,None,:]*ell
  if family=='nb':
   y=sample_nb(mu.reshape(-1,G),np.broadcast_to(theta,mu.shape).reshape(-1,G),seed=seed+100000*(j+1)).reshape(mu.shape)
  else:
   rr=np.random.default_rng(seed+100000*(j+1));s2=np.log1p(1/theta)
   factor=np.exp(rr.normal(size=(NG,n,1))*np.sqrt(s2)-s2/2)
   y=rr.poisson(mu*factor)
  splits[split]=(mu,ell,y)
 return beta,theta,splits

def run(job):
 seed,family,mean_case=job;out=H/'results'/f'{seed}-{family}-{mean_case}.csv'
 if out.exists():return out.name
 beta,theta,splits=fixture(seed,family)
 predicted={}
 for name,(mu,ell,y) in splits.items():
  if mean_case=='correct':m=mu
  elif mean_case=='attenuated':center=np.exp(.4**2/2);m=beta[:,None,:]*(center+.25*(ell-center))
  else:m=mu*np.where(np.arange(G)%2,1.4,.6)[None,None,:]
  predicted[name]=m
 train=splits['train'][2];m=predicted['train'];num=(m*m).sum(1);den=((train-m)**2-train).sum(1)
 cond=estimate(num,den);pool=estimate(num.sum(0),den.sum(0))
 genes=np.argsort(train.mean(1),axis=1)[:,-30:];upper=np.triu_indices(30,1)
 rows=[]
 for split in ['validation','test']:
  real=splits[split][2];rl=np.log1p(real);m=predicted[split]
  for a,arm in enumerate(['poisson','pooled','conditional','reference']):
   if arm=='reference':
    if split=='validation':continue
    syn=splits['reference'][2]
   elif arm=='poisson':syn=np.random.default_rng(seed+(300000 if split=='validation' else 400000)).poisson(m)
   else:
    th=np.broadcast_to(pool[None,None,:] if arm=='pooled' else cond[:,None,:],m.shape)
    syn=sample_nb(m.reshape(-1,G),th.reshape(-1,G),seed=seed+(300000 if split=='validation' else 400000)).reshape(m.shape)
   sl=np.log1p(syn)
   for g in range(NG):
    metric=r_var_per_gene_expressed(rl[g],sl[g])['expressed']['median_abs_log2_ratio']
    row=dict(seed=seed,family=family,mean_case=mean_case,split=split,arm=arm,dataset=g//6,cell_type=g%6,calibration=metric)
    if split=='test':
     mask=real[g].mean(0)>=.5
     cr=np.corrcoef(rl[g][:,genes[g]],rowvar=False)[upper];cs=np.corrcoef(sl[g][:,genes[g]],rowvar=False)[upper]
     assert np.isfinite(cr).all() and np.isfinite(cs).all()
     row.update(energy=energy_distance(rl[g],sl[g]),correlation_error=float(np.median(abs(cr-cs))),mean_error=float(np.median(abs(np.log2(np.maximum(syn[g].mean(0)[mask],1e-12)/real[g].mean(0)[mask])))))
    else:row.update(energy='',correlation_error='',mean_error='')
    rows.append(row)
 with out.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return out.name

if __name__=='__main__':
 start=time.time();jobs=[(s,f,m) for s in range(52000,52020) for f in ['nb','correlated'] for m in ['correct','attenuated','biased']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
  for i,res in enumerate(pool.map(run,jobs),1):print(i,len(jobs),res,round(time.time()-start),flush=True)
