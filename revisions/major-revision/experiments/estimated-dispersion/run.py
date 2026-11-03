from pathlib import Path
import sys,os,json,csv,time,importlib.util,concurrent.futures
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/phase2f_f4_resolution_2026-07-09'))
spec=importlib.util.spec_from_file_location('original_lawgrid',ROOT/'experiments/phase2m_lawgrid_2026-07-09/run_lawgrid.py');L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
from src.evaluation.distributional import r_var_per_gene_expressed,energy_distance
from src.architecture.stochastic_decoder import sample_nb,sample_nb_fixed_theta

def estimate(num,den):
 th=np.full(num.shape,1000.);np.divide(num,den,out=th,where=den>0);return np.clip(th,.05,1000.)

def run(job):
 seed,regime=job;out=HERE/'results'/f'{seed}-{regime}.csv'
 if out.exists():return out.name
 real_log,mu,truth,labels,ds=L.build_truth(regime,seed)
 group_n=L.CELLS_PER_GROUP;ng=len(mu)//group_n;genes=L.N_GENES
 mu3=mu.reshape(ng,group_n,genes);th3=truth.reshape(ng,group_n,genes)
 original_global=np.full_like(mu,float(np.median(truth)))
 rows=[]
 def evaluate(model,ntrain,theta,energy=True):
  draw=sample_nb_fixed_theta(mu,float('inf'),seed=seed+101) if theta is None else sample_nb(mu,theta,seed=seed+101)
  log=np.log1p(draw).astype(np.float32)
  for g in range(ng):
   sl=slice(g*group_n,(g+1)*group_n);rr,ss=real_log[sl],log[sl];m=r_var_per_gene_expressed(rr,ss)['expressed']
   rows.append(dict(seed=seed,regime=regime,model=model,training_cells=ntrain,dataset=g//L.N_TYPES,cell_type=g%L.N_TYPES,calibration_error=m['median_abs_log2_ratio'],signed_ratio=m['median_signed_ratio'],energy=energy_distance(rr,ss) if energy else '',theta_lower_clip_fraction='' if theta is None else float(np.mean(theta[sl]<=.05)),theta_upper_clip_fraction='' if theta is None else float(np.mean(theta[sl]>=1000))))
 evaluate('poisson',0,None);evaluate('oracle',0,truth);evaluate('original_global',0,original_global)
 # Independent training counts; nested sizes use prefixes of one 400-cell draw.
 rng=np.random.default_rng(seed+300003)
 idx=rng.integers(0,group_n,size=(ng,400));mt=np.take_along_axis(mu3,idx[:,:,None],axis=1)
 tt=np.repeat(th3[:,:1,:],400,axis=1)
 train=sample_nb(mt.reshape(-1,genes),tt.reshape(-1,genes),seed=seed+700003).reshape(ng,400,genes)
 for n in [25,100,400]:
  num=(mt[:,:n]**2).sum(axis=1);den=((train[:,:n]-mt[:,:n])**2-train[:,:n]).sum(axis=1)
  cond=estimate(num,den);pooled=estimate(num.sum(axis=0),den.sum(axis=0))
  for name,theta in [('estimated_global',np.broadcast_to(pooled,mu.shape)),('estimated_conditional',np.repeat(cond[:,None,:],group_n,axis=1).reshape(mu.shape))]:
   evaluate(name,n,theta,energy=n==400)
 with out.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return out.name

if __name__=='__main__':
 (HERE/'results').mkdir(exist_ok=True);start=time.time();jobs=[(s,r) for s in range(41000,41030) for r in L.REGIMES]
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
  for i,result in enumerate(pool.map(run,jobs),1):print(i,'/',len(jobs),result,'elapsed',round(time.time()-start),flush=True)
 print('COMPLETE',flush=True)
