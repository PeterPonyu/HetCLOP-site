from pathlib import Path
import json,csv,hashlib
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1];src=R/'experiments/phase2g_armb_f4_2026-07-09/results/phase2g.json';d=json.loads(src.read_text());rows=[];mac={};names={'pca_512':'PCA','randproj_512':'RP','scgpt_512':'GPT','geneformer':'GF'}
for r in d['E7_clustered_inference']['per_comparison']:
 if r['metric']!='r_var_abs_log2_expr':continue
 c=r['clustered_50dataset'];q=r['naive_314group'];row=dict(encoder=r['encoder'],seed=r['seed'],gate=r['gate_passed'],n_datasets=c['n_datasets'],paired_group_median=q['median_paired_diff'],nb_group_win_fraction=q['nb_win_fraction'],dataset_median_effect=c['median_of_dataset_medians'],dataset_wilcoxon_p=c['wilcoxon_p'],dataset_sign_p=c['sign_test_p']);rows.append(row)
 k='ArmBPrimary'+names[r['encoder']]+{42:'First',43:'Second',44:'Third'}[r['seed']];mac[k+'Effect']=f"{row['dataset_median_effect']:+.4f}";mac[k+'P']=f"{row['dataset_wilcoxon_p']:.3f}"
with (H/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(P/'manuscript/figs/armb_primary_numbers.tex').write_text('% Extracted verbatim from saved Arm B tests, with display rounding only.\n'+'\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in mac.items())+'\n')
(H/'source.json').write_text(json.dumps({'path':str(src.relative_to(R)),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'new_compute':False,'rows':len(rows)},indent=2)+'\n')
print('Copied all 12 saved primary-endpoint contrasts; no new inferential calculation.')
