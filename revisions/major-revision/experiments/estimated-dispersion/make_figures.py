from pathlib import Path
import pandas as pd
import numpy as np
H=Path(__file__).resolve().parent;P=H.parents[1]/'manuscript'/'figs'
s=pd.read_csv(H/'seed_summary.csv');a=pd.read_csv(H/'summary.csv');pairs=pd.read_csv(H/'paired_vs_poisson.csv')
B=chr(92)
regimes=['no_deficit','deficit_homog','deficit_hetero','deficit_mixed'];names=['Near-Poisson','Homogeneous deficit','Heterogeneous deficit','Bimodal deficit']
colors={'estimated_global':'horange','estimated_conditional':'hgreen','oracle':'hgreen','original_global':'horange','poisson':'hblue'}
head=r'\begingroup\sffamily\mdseries\sansmath'+'\n'+r'\begin{tikzpicture}'+'\n'
# Paired effects per seed, not differences of across-seed medians.
f=[head,r'\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.8cm},width=6cm,height=6.2cm,scale only axis,tick align=outside,tick pos=left,tick label style={font=\footnotesize},label style={font=\footnotesize},title style={font=\footnotesize},ymin=0.45,ymax=4.6,ytick={1,2,3,4},yticklabels={Bimodal,Heterogeneous,Homogeneous,Near-Poisson},xmin=-0.56,xmax=0.10,xtick={-0.5,-0.25,0},legend style={font=\footnotesize,draw=none,at={(1.15,1.18)},anchor=south,legend columns=2}]']
for k,metric in enumerate(['calibration_error','energy']):
 f.append(B+'nextgroupplot[title={'+['Variance calibration','Energy distance'][k]+'},xlabel={Difference from Poisson (lower is better)}'+(',yticklabels={}' if k==1 else '')+']')
 f.append(r'\draw[hgray,dashed] (axis cs:0,.5)--(axis cs:0,4.5);')
 for mi,m in enumerate(['original_global','oracle']):
  for ri,r in enumerate(regimes):
   q=s[(s.regime==r)&(s.model==m)].set_index('seed')[metric]-s[(s.regime==r)&(s.model=='poisson')].set_index('seed')[metric]
   y=4-ri+[-.13,.13][mi];ys=y+np.linspace(-.065,.065,30)
   coords=' '.join(f'({v:.7f},{yy:.5f})' for v,yy in zip(q,ys))
   f.append(B+f'addplot[only marks,mark=*,mark size=0.65pt,{colors[m]},opacity=.4,forget plot] coordinates {{{coords}}};')
   lo,md,hi=q.quantile([.025,.5,.975]);f.append(B+f'draw[{colors[m]},line width=1pt] (axis cs:{lo:.7f},{y})--(axis cs:{hi:.7f},{y});')
   f.append(B+f'addplot[only marks,mark=diamond*,mark size=2.1pt,{colors[m]},forget plot] coordinates {{({md:.7f},{y})}};')
  if k==0:f.append(B+f'addlegendimage{{{colors[m]},mark=diamond*,thick}}'+B+'addlegendentry{'+['Global (true median)','Oracle conditional'][mi]+'}')
 f.append(r'\node[font=\scriptsize,anchor=south west] at (rel axis cs:0.02,0.01) {30 seeds; median and central 95\%};')
 f.append(B+'node[anchor=base west,font='+B+'footnotesize'+B+'bfseries,xshift=-19pt,yshift=12pt] at (rel axis cs:0,1) {('+chr(97+k)+')};')
f.extend([r'\end{groupplot}\end{tikzpicture}\endgroup']);(P/'fig_seedreplication.tex').write_text('\n'.join(f)+'\n')
# Estimate rather than supply dispersion; logarithmic size axis with manual shifts.
f=[head,r'\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.8cm,vertical sep=1.9cm},width=6cm,height=4.3cm,scale only axis,tick align=outside,tick pos=left,tick label style={font=\footnotesize},label style={font=\footnotesize},title style={font=\footnotesize},xmode=log,log basis x=4,xmin=18,xmax=560,xtick={25,100,400},xticklabels={25,100,400},xlabel={Training cells per group},ylabel={Calibration error},y tick label style={/pgf/number format/.cd,fixed,precision=2},ymin=0,ymax=.72,legend style={font=\footnotesize,draw=none,at={(1.15,1.24)},anchor=south,legend columns=2,legend cell align=left}]']
for ri,r in enumerate(regimes):
 f.append(B+'nextgroupplot[title={'+names[ri]+'},ymax='+str([.16,.68,.50,.46][ri])+']')
 for model,style in [('poisson','dashed'),('oracle','dotted')]:
  y=float(a[(a.regime==r)&(a.model==model)].calibration_error_median.iloc[0]);f.append(B+f'addplot[{colors[model]},{style},thick,domain=20:500,samples=2] {{{y:.7f}}};')
  if ri==0:f.append(B+'addlegendentry{'+('Poisson' if model=='poisson' else 'Oracle conditional')+'}')
 for mi,m in enumerate(['estimated_global','estimated_conditional']):
  g=a[(a.regime==r)&(a.model==m)].sort_values('training_cells');coords=[]
  for _,row in g.iterrows():
   n=int(row.training_cells);x=n*([.94,1.06][mi]);v=row.calibration_error_median;lo=row.calibration_error_q025;hi=row.calibration_error_q975
   coords.append(f'({x:.4f},{v:.7f})');f.append(B+f'draw[{colors[m]},opacity=.65,line width=.6pt] (axis cs:{x:.4f},{lo:.7f})--(axis cs:{x:.4f},{hi:.7f});')
   vv=s[(s.regime==r)&(s.model==m)&(s.training_cells==n)].calibration_error.to_numpy();xx=x*np.exp(np.linspace(-.045,.045,30))
   points=' '.join(f'({xc:.4f},{yc:.7f})' for xc,yc in zip(xx,vv));f.append(B+f'addplot[only marks,mark=*,mark size=.55pt,opacity=.25,{colors[m]},forget plot] coordinates {{{points}}};')
  f.append(B+f'addplot[{colors[m]},thick,mark='+['square*','*'][mi]+',mark size=1.6pt] coordinates {'+' '.join(coords)+'};')
  if ri==0:f.append(B+'addlegendentry{'+['Estimated pooled','Estimated conditional'][mi]+'}')
 f.append(B+'node[anchor=base west,font='+B+'footnotesize'+B+'bfseries,xshift=-19pt,yshift=12pt] at (rel axis cs:0,1) {('+chr(97+ri)+')};')
f.extend([r'\end{groupplot}\end{tikzpicture}\endgroup']);(P/'fig_estimateddispersion.tex').write_text('\n'.join(f)+'\n')
# Separate new-result macros; the original numerical ledger is immutable.
macros={'RevSeeds':'30','RevRuns':'120','RevGroupRows':str(120*648)}
labels={'no_deficit':'Near','deficit_homog':'Hom','deficit_hetero':'Het','deficit_mixed':'Mix'}
models={'poisson':'Pois','oracle':'Oracle','estimated_global':'Pool','estimated_conditional':'Cond','original_global':'TrueGlobal'}
for _,row in a.iterrows():
 key='Rev'+labels[row.regime]+models[row.model]+{0:'',25:'Small',100:'Mid',400:'Large'}[row.training_cells]
 macros[key]=f'{row.calibration_error_median:.3f}'
 if pd.notna(row.energy_median):macros[key+'Energy']=f'{row.energy_median:.3f}'
for _,row in pairs.iterrows():
 if row.model=='original_global' and row.regime=='deficit_mixed':
  key='RevMixGlobal'+('Cal' if row.metric=='calibration_error' else 'Energy')
  for suffix,val in [('Wins',row.wins),('Effect',row['median']),('Low',row.q025),('High',row.q975)]:macros[key+suffix]=str(int(val)) if suffix=='Wins' else f'{val:.3f}'
(P/'revision_numbers.tex').write_text('% Generated only from isolated revision experiments.\n'+'\n'.join(B+'newcommand{'+B+k+'}{'+v+'}' for k,v in macros.items())+'\n')
print('Wrote Figures 16–17 and',len(macros),'new macros')
