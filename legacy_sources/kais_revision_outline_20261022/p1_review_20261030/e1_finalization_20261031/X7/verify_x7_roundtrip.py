"""Independent round-trip checks of X7 CSVs, Cstat crosschecks and log endpoints."""
import csv,hashlib,json,math,statistics,collections
from pathlib import Path
H=Path(__file__).resolve().parent;F=H.parent;R=F.parents[1].parent;W=R.parent
qa=H/'qa';qa.mkdir(exist_ok=True);checks=0
def ck(ok,msg):
 global checks
 checks+=1
 if not ok:raise ValueError(msg)
def close(a,b,msg):ck(math.isclose(float(a),float(b),rel_tol=2e-11,abs_tol=2e-12),msg)
def rows(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def j(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def main():
 t=H/'training';summaries=rows(t/'training_run_summary.csv');curves=rows(t/'epoch_validation_curves.csv');by=collections.defaultdict(list)
 for r in curves:by[(r['dataset'],r['method'],r['training_seed'])].append(r)
 ck(len(summaries)==24 and len(by)==24 and len(curves)==1846,'Training complete24models/1846epochs')
 for r in summaries:
  rr=by[(r['dataset'],r['method'],r['training_seed'])];best=max(float(v['val_ndcg10']) for v in rr);first=next(v for v in rr if float(v['val_ndcg10'])==best);last=rr[-1]
  ck(len(rr)==int(r['epochs_run']) and int(r['best_epoch'])==int(first['epoch']),'Best/stop endpoints')
  close(best,r['best_val_ndcg10'],'BestNDCG');close(last['val_ndcg10'],r['last_val_ndcg10'],'FinalNDCG');close(first['train_loss'],r['train_loss_at_best'],'BestLoss');close(last['train_loss'],r['last_train_loss'],'FinalLoss')
  ck(50<=len(rr)<=200 and int(last['bad_epochs'])>=15,'Final minfloor/patience')
  ck(all(not(int(x['epoch'])>=50 and int(x['bad_epochs'])>=15) for x in rr[:-1]),'First eligible stop')
  for v in rr:ck(int(v['epoch'])==rr.index(v)+1,'Continuous serialized epochs')
 training_checks=checks;write(qa/'training_independent_verification.json',{'status':'PASS_INDEPENDENT_CURRENT24_TRAINING_CSV_ENDPOINTS_AND_STOP_RULE','checks':training_checks,'runs':24,'epoch_rows':len(curves),'all_time_validation_best_preserved':True,'minimum_stopping_epoch':50,'reset_at50':False,'source':str(t/'training_source_contract.json'),'no_original_source_or_output_edits':True})
 a=H/'warm_targets';per=rows(a/'per_seed_metrics.csv');methods=rows(a/'method_summary.csv');dif=rows(a/'paired_differences_per_seed.csv');pairs=rows(a/'paired_difference_summary.csv');group=collections.defaultdict(list);index={}
 for r in per:
  key=(r['dataset'],r['method'],r['population'],r['metric']);group[key].append(r);index[key+(r['training_seed'],)]=r
 ck(len(per)==1296 and len(methods)==432 and len(dif)==648 and len(pairs)==216,'All serialized statistics closure')
 def stats(values):
  m=statistics.mean(values);sd=statistics.stdev(values);half=4.302652729749464*sd/math.sqrt(3);return {'mean':m,'sample_sd':sd,'ci95_lower':m-half,'ci95_upper':m+half}
 for r in methods:
  rr=group[(r['dataset'],r['method'],r['population'],r['metric'])];ck([x['training_seed'] for x in rr]==['42','2026','3407'],'Three fixed seeds permean')
  if rr[0]['value']=='':ck(all(x['value']=='' for x in rr) and all(r[k]=='' for k in ['mean','sample_sd','ci95_lower','ci95_upper']),'Empty populations preserved')
  else:
   expected=stats([float(x['value']) for x in rr])
   for k,v in expected.items():close(v,r[k],'Independent seed meanSDtCI '+k)
 for r in dif:
  key=(r['dataset'],r['method'],r['population'],r['metric'],r['training_seed']);mr=index[key];cr=index[(r['dataset'],'Control',r['population'],r['metric'],r['training_seed'])]
  ck(r['n']==mr['n']==cr['n'],'Matchedpopulation perseed')
  if r['difference']!='':close(float(mr['value'])-float(cr['value']),r['difference'],'Paired seed difference')
 for r in pairs:
  rr=[x for x in dif if all(x[k]==r[k] for k in ['dataset','method','population','metric'])]
  if r['mean_delta']!='':
   expected=stats([float(x['difference']) for x in rr])
   for k,v in expected.items():close(v,r[k],'Paired seed CI '+k)
   close(expected['mean'],r['mean_delta'],'Paired mean delta')
 # Independent unifiedC output comes from a separate builder and includes samefull/affected rows.
 c=F/'statistics/per_seed_metrics.csv';global_rows=rows(c);global_index={(r['dataset'],r['method'],r['population'],r['metric'],r['training_seed']):r for r in global_rows};matches=0
 for r in per:
  if r['population'] in ['full','affected']:
   cr=global_index[(r['dataset'],r['method'],r['population'],r['metric'],r['training_seed'])];ck(r['n']==cr['n'],'Cindependent population');close(r['value'],cr['value'],'Independent unifiedC metric');matches+=1
 ck(matches==432,'432 cross-builder full/affected matches')
 report={'status':'PASS_INDEPENDENT_X7_SEED_AGGREGATION_AND_432_GLOBAL_STATS_CROSSCHECKS','checks':checks-training_checks,'per_seed_rows':len(per),'method_rows':len(methods),'paired_per_seed_rows':len(dif),'paired_summary_rows':len(pairs),'independent_global_C_matches':matches,'global_statistics_sha256':sha(c),'population_counts_sha256':sha(a/'population_counts.csv'),'stale_X5_rank_columns_used':False,'confidence_intervals_untruncated':True,'source_report_sha256':sha(a/'verification.json')};write(qa/'warm_target_independent_verification.json',report)
 acceptance=j(H/'X7_acceptance.json');acceptance['independent_roundtrip_QA']={'training':sha(qa/'training_independent_verification.json'),'warm_targets':sha(qa/'warm_target_independent_verification.json'),'global_C_matches':432};write(H/'X7_acceptance.json',acceptance)
 print(json.dumps({'training_checks':training_checks,'warm_checks':checks-training_checks,'global_C_matches':matches,'X7_acceptance_sha256':sha(H/'X7_acceptance.json')},indent=2))
if __name__=='__main__':main()
