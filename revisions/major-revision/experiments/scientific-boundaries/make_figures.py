from pathlib import Path
import pandas as pd,numpy as np
H=Path(__file__).resolve().parent;P=H.parents[1]/'manuscript/figs';B=chr(92)
s=pd.read_csv(H/'seed_summary.csv');a=pd.read_csv(H/'summary.csv');pairs=pd.read_csv(H/'paired_effects.csv');bio=pd.read_csv(H/'biological_support.csv')
head=r'\begingroup\sffamily\mdseries\sansmath'+'\n'+r'\begin{tikzpicture}'+'\n'
conditions=[(f,m) for f in ['nb','correlated'] for m in ['correct','attenuated','biased']]
labels=['NB: correct','NB: attenuated','NB: biased','Correlated: correct','Correlated: attenuated','Correlated: biased']
colors={'selected':'hgreen','poisson':'hblue','reference':'hgray','nb_global':'horange','nb_conditional':'hgreen'}
def tag(k):return B+'node[anchor=base west,font='+B+'footnotesize'+B+'bfseries,xshift=-19pt,yshift=12pt] at (rel axis cs:0,1) {('+chr(97+k)+')};'
def scatter_interval(f,vals,y,col,offset=0):
 vals=np.asarray(vals);y+=offset;yy=y+np.linspace(-.055,.055,len(vals));lo,md,hi=np.quantile(vals,[.025,.5,.975])
 f.append(B+f'addplot[only marks,mark=*,mark size=.7pt,{col},opacity=.4,forget plot] coordinates '+'{'+ ' '.join(f'({v:.7f},{w:.5f})' for v,w in zip(vals,yy))+'};')
 f.append(B+f'draw[{col},line width=.8pt] (axis cs:{lo:.7f},{y})--(axis cs:{hi:.7f},{y});')
 f.append(B+f'addplot[only marks,mark=diamond*,mark size=1.8pt,{col},forget plot] coordinates '+'{'+f'({md:.7f},{y})'+'};')
f=[head,r'\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.7cm,vertical sep=2.7cm},width=5.7cm,height=5.1cm,scale only axis,tick align=outside,tick pos=left,tick label style={font=\scriptsize},label style={font=\footnotesize},ymin=.5,ymax=6.5,ytick={6,5,4,3,2,1},yticklabels={'+','.join(labels)+r'},x tick label style={/pgf/number format/.cd,fixed,precision=2},legend style={font=\footnotesize,draw=none,at={(1.15,1.18)},anchor=south,legend columns=3}]']
for k,(metric,title,xlabel,lim) in enumerate([('calibration','Test calibration','Difference from Poisson','xmin=-.58,xmax=.12'),('energy','Test energy distance','Difference from Poisson','xmin=-.42,xmax=.12'),('correlation_error','Gene dependence','Absolute correlation error','xmin=0,xmax=.66'),('mean_error','Gene means','Absolute log mean-ratio error','xmin=0,xmax=.72')]):
 f.append(B+'nextgroupplot[title={'+title+'},xlabel={'+xlabel+'},'+lim+(',yticklabels={}' if k%2 else '')+']')
 if k<2:f.append(r'\draw[hgray,dashed] (axis cs:0,.5)--(axis cs:0,6.5);')
 for i,(family,mean) in enumerate(conditions):
  q=s[(s.family==family)&(s.mean_case==mean)].pivot(index='seed',columns='arm',values=metric)
  if k<2:scatter_interval(f,q.selected-q.poisson,6-i,'hgreen')
  else:
   for arm,off in [('poisson',.17),('selected',0),('reference',-.17)]:scatter_interval(f,q[arm],6-i,colors[arm],off)
 if k==2:
  for arm,name in [('poisson','Poisson'),('selected','Validation-selected'),('reference','Independent real draw')]:f.append(B+'addlegendimage{'+colors[arm]+',mark=diamond*}'+B+'addlegendentry{'+name+'}')
 f.append(tag(k))
f.append(r'\end{groupplot}\end{tikzpicture}\endgroup');(P/'fig_scientificboundaries.tex').write_text('\n'.join(f)+'\n')
# Biological support check: every point uses already archived per-type values.
f=[head,r'\begin{groupplot}[group style={group size=2 by 1,horizontal sep=3.0cm},width=5.4cm,height=5.2cm,scale only axis,tick align=outside,tick pos=left,tick label style={font=\footnotesize},label style={font=\footnotesize},legend style={font=\footnotesize,draw=none,at={(1.22,1.20)},anchor=south,legend columns=3}]',r'\nextgroupplot[title={Same 69 cell types},ylabel={Marker recall},xmin=.55,xmax=2.45,ymin=0,ymax=.87,xtick={1,2},xticklabels={Fixed mean,End-to-end}]']
for j,(arm,name) in enumerate([('poisson','Poisson'),('nb_global','Global'),('nb_conditional','Conditional')]):
 col=colors[arm];off=(j-1)*.18
 for i,stage in enumerate(['fixed','end_to_end'],1):
  v=bio[(bio.stage==stage)&(bio.arm==arm)].common_support_mean.to_numpy();x=i+off
  f.append(B+f'draw[{col},line width=.8pt] (axis cs:{x},{v.min():.7f})--(axis cs:{x},{v.max():.7f});')
  f.append(B+f'addplot[only marks,mark=*,mark size=1.2pt,{col},forget plot] coordinates '+'{'+' '.join(f'({x+(q-1)*.035:.4f},{z:.7f})' for q,z in enumerate(v))+'};')
  f.append(B+f'addplot[only marks,mark=diamond*,mark size=2.3pt,{col},forget plot] coordinates '+'{'+f'({x},{v.mean():.7f})'+'};')
 f.append(B+'addlegendimage{'+col+',mark=diamond*}'+B+'addlegendentry{'+name+'}')
f.append(tag(0))
f.append(r'\nextgroupplot[title={Within-stage contrasts},xlabel={Conditional minus comparator},xmin=-.06,xmax=.27,ymin=.5,ymax=4.5,ytick={4,3,2,1},yticklabels={Fixed: Poisson,Fixed: global,End-to-end: Poisson,End-to-end: global},yticklabel style={font=\scriptsize},xtick={0,.1,.2}]')
f.append(r'\draw[hgray,dashed] (axis cs:0,.5)--(axis cs:0,4.5);')
for i,(stage,control) in enumerate([('fixed','poisson'),('fixed','nb_global'),('end_to_end','poisson'),('end_to_end','nb_global')]):
 q=bio[bio.stage==stage].pivot(index='seed',columns='arm',values='common_support_mean');v=q.nb_conditional-q[control];y=4-i
 f.append(B+f'draw[{colors[control]},line width=.8pt] (axis cs:{v.min():.7f},{y})--(axis cs:{v.max():.7f},{y});')
 f.append(B+f'addplot[only marks,mark=*,mark size=1.2pt,{colors[control]},forget plot] coordinates '+'{'+' '.join(f'({z:.7f},{y+(j-1)*.08})' for j,z in enumerate(v))+'};')
 f.append(B+f'addplot[only marks,mark=diamond*,mark size=2.2pt,{colors[control]},forget plot] coordinates '+'{'+f'({v.mean():.7f},{y})'+'};')
f.extend([tag(1),r'\end{groupplot}\end{tikzpicture}\endgroup']);(P/'fig_biologicalsupport.tex').write_text('\n'.join(f)+'\n')
# New ledger is independent of both prior ledgers.
mac={}
for _,r in a.iterrows():
 if r.arm not in ['selected','poisson','reference']:continue
 key='Stress'+{'nb':'NB','correlated':'Corr'}[r.family]+r.mean_case.title()+r.arm.title()
 for metric,suffix in [('calibration','Cal'),('energy','Energy'),('correlation_error','Correlation'),('mean_error','Mean')]:mac[key+suffix]=f'{r[metric]:.3f}'
for stage in ['fixed','end_to_end']:
 q=bio[bio.stage==stage].pivot(index='seed',columns='arm',values='common_support_mean')
 for arm in q.columns:mac['Support'+{'fixed':'Fixed','end_to_end':'End'}[stage]+{'poisson':'Pois','nb_global':'Global','nb_conditional':'Cond'}[arm]]=f'{q[arm].mean():.3f}'
 for ctrl in ['poisson','nb_global']:mac['Support'+{'fixed':'Fixed','end_to_end':'End'}[stage]+{'poisson':'VsPois','nb_global':'VsGlobal'}[ctrl]]=f'{(q.nb_conditional-q[ctrl]).mean():.3f}'
(P/'science_numbers.tex').write_text('% Generated from isolated scientific-boundary results.\n'+'\n'.join(B+'newcommand{'+B+k+'}{'+v+'}' for k,v in mac.items())+'\n')
print('Created Figures 18–19 and',len(mac),'ledger values')
