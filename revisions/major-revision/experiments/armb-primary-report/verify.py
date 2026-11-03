from pathlib import Path
import json,csv,hashlib,re
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1];meta=json.loads((H/'source.json').read_text());src=R/meta['path'];assert hashlib.sha256(src.read_bytes()).hexdigest()==meta['sha256'];raw=json.loads(src.read_text());saved={(r['encoder'],r['seed']):r for r in raw['E7_clustered_inference']['per_comparison'] if r['metric']=='r_var_abs_log2_expr'};rows=list(csv.DictReader((H/'summary.csv').open()));assert len(rows)==12
mac=dict(re.findall(r'\\newcommand\{\\([^}]+)\}\{([^}]+)\}',(P/'manuscript/figs/armb_primary_numbers.tex').read_text()));names={'pca_512':'PCA','randproj_512':'RP','scgpt_512':'GPT','geneformer':'GF'}
for r in rows:
 o=saved[(r['encoder'],int(r['seed']))];c=o['clustered_50dataset'];assert float(r['dataset_wilcoxon_p'])==c['wilcoxon_p'];assert float(r['dataset_median_effect'])==c['median_of_dataset_medians'];assert float(r['nb_group_win_fraction'])==o['naive_314group']['nb_win_fraction'];assert c['wilcoxon_p']>.05
 k='ArmBPrimary'+names[r['encoder']]+{42:'First',43:'Second',44:'Third'}[int(r['seed'])];assert mac[k+'Effect']==f"{c['median_of_dataset_medians']:+.4f}";assert mac[k+'P']==f"{c['wilcoxon_p']:.3f}"
(H/'verification.json').write_text(json.dumps({'status':'PASS','saved_contrasts':12,'verified_macros':24,'source_unchanged':True,'new_inference':False},indent=2)+'\n');print('PASS: 12 saved contrasts and 24 macros; all dataset-Wilcoxon p > 0.05.')
