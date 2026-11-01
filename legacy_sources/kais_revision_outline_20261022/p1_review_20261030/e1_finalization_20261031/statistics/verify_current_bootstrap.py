"""Independently reproduce an affected E1 bootstrap cell and unchanged historical cells."""
from pathlib import Path
import csv,json,math,hashlib,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:/time_uncertainty_rec')
METRICS=('ndcg10','hr10','hr20','recall50','mrr','mrr10','mean_rank')
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def vector(rank):return np.asarray([1/math.log2(rank+1) if rank<=10 else 0,int(rank<=10),int(rank<=20),int(rank<=50),1/rank,1/rank if rank<=10 else 0,rank],dtype=np.float64)
def main():
 mapping={(r['dataset'],r['method'],r['training_seed']):r for r in json.loads((HERE/'run_overlay.json').read_text(encoding='utf-8'))['runs']}
 current=rows(HERE/'paired_per_seed_user_bootstrap.csv');old=rows(ROOT/'kais_revision_outline_20261022/g2_closure_20261027/C_statistics/paired_per_seed_user_bootstrap.csv')
 key=lambda r:tuple(r[k] for k in ('dataset','method','training_seed','population','metric'))
 prior={key(r):r for r in old};unchanged=0
 for r in current:
  tag,method,seed=r['dataset'],r['method'],int(r['training_seed'])
  if mapping[(tag,method,seed)]['lineage']=='RETAINED_PRE_E1_EVALUATION' and mapping[(tag,'Control',seed)]['lineage']=='RETAINED_PRE_E1_EVALUATION':
   before=prior[key(r)]
   for field in ('mean_delta','ci95_lower','ci95_upper'):
    assert math.isclose(float(r[field]),float(before[field]),abs_tol=3e-12,rel_tol=1e-10)
   unchanged+=1
 tag,method,seed,pop='toys','Ensemble',2026,'affected'
 chosen={r['metric']:r for r in current if (r['dataset'],r['method'],int(r['training_seed']),r['population'])==(tag,method,seed,pop)}
 labels=rows(ROOT/'g2_lbsp_amazon_stability_20261025/snapshot/outputs/toys_seed42/panel_labels.csv');ids=[r['sample_id'] for r in labels if int(r['affected'])]
 control={r['sample_id']:r for r in rows(ROOT/mapping[(tag,'Control',seed)]['output_directory']/'predictions_original.csv')}
 groups=[{r['sample_id']:r for r in rows(ROOT/mapping[(tag,method,seed)]['output_directory']/('predictions_group'+str(g)+'.csv'))} for g in range(1,6)]
 rank=lambda r:int(r.get('target_rank',r.get('rank')))
 delta=np.asarray([np.asarray([math.fsum(vector(rank(g[sid]))[i] for g in groups)/5 for i in range(7)])-vector(rank(control[sid])) for sid in ids])
 rng_seed=int.from_bytes(hashlib.sha256(f'20261027|{tag}|{method}|{seed}|{pop}'.encode('ascii')).digest()[:8],'big');assert rng_seed==int(chosen['ndcg10']['rng_seed'])
 rng=np.random.Generator(np.random.PCG64(rng_seed));distribution=np.empty((2000,7))
 for i in range(2000):
  counts=np.bincount(rng.integers(0,len(ids),size=len(ids)),minlength=len(ids));distribution[i]=counts@delta/len(ids)
 lo,hi=np.quantile(distribution,(0.025,0.975),axis=0,method='linear');comparisons=0
 for i,m in enumerate(METRICS):
  for a,b in ((delta[:,i].mean(),float(chosen[m]['mean_delta'])),(lo[i],float(chosen[m]['ci95_lower'])),(hi[i],float(chosen[m]['ci95_upper']))):
   assert math.isclose(a,b,abs_tol=3e-12,rel_tol=1e-10),(m,a,b);comparisons+=1
 report={'status':'PASS_INDEPENDENT_E1_CHANGED_CELL_AND_RETAINED_CELL_BOOTSTRAP','changed_cell':f'{tag}/{method}/{seed}/{pop}','users':len(ids),'replicates':2000,'numeric_comparisons':comparisons,'retained_cell_metric_rows_matching_old_bootstrap':unchanged,'count_weighted_resampling_independent_of_builder_index_means':True,'original_formula_rng_replicates_unchanged':True,'raw_interaction_checkpoint_Torch_research_code_accessed':False}
 (HERE/'independent_current_bootstrap_verification.json').write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
