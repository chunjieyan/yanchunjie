"""RebuildX7 from10 accepted new +14 retained Control/LBSP metadata lineages."""
import collections,csv,hashlib,importlib.util,json,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;P1=HERE.parents[1];REV=P1.parent;ROOT=REV.parent
OLD=REV/'g2_x7_diagnostics_20261029';TRAIN=P1/'e1_return_20261031T092136/report_snapshot';INF=P1/'e1_inference_return_20261031T110400/report_snapshot'
SEEDS=(42,2026,3407);TAGS=('ml1m','beauty','sports','toys');METHODS=('Control','LBSP');PERMS=(101,202,303,404,505);POPS=('full','warm','cold','affected','warm_affected','cold_affected');ACCURACY=('ndcg10','hr10','hr20','recall50','mrr','mrr10','mean_rank');METRICS=ACCURACY+('overlap10','hitflip');TC=4.302652729749464
checks=numeric=prediction_rows=0;reads={}
def ck(ok,msg):
 global checks
 checks+=1
 if not ok:raise ValueError(msg)
def near(a,b,msg):
 global numeric
 numeric+=1;ck(math.isclose(float(a),float(b),rel_tol=2e-11,abs_tol=2e-12),msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,expected=None):
 p=Path(p);ck(p.suffix in('.json','.csv'),'Metadata only');payload=p.read_bytes();h=hashlib.sha256(payload).hexdigest();ck(expected is None or h==expected,'Sealed metadata input '+str(p));reads[p.relative_to(ROOT).as_posix()]={'sha256':h,'bytes':len(payload)}
 return json.loads(payload.decode('utf-8-sig')) if p.suffix=='.json' else list(csv.DictReader(payload.decode('utf-8-sig').splitlines()))
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def csvwrite(path,data):
 with path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def load_helper(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def build_training():
 out=HERE/'training';out.mkdir(exist_ok=True)
 oldscript=OLD/'training/analyze_training_logs.py';text=oldscript.read_text(encoding='utf-8')
 a="require(len(rows) <= config['max_epochs'], identity + ': exceeds epoch cap')";b="require(50 <= len(rows) <= config['max_epochs'], identity + ': outside uniform50/200 epoch bounds')";ck(text.count(a)==1,'Reviewed log floor patch');text=text.replace(a,b)
 a="require(bad_epochs < config['patience'] or i == len(rows),";b="require(not (epoch >= 50 and bad_epochs >= config['patience']) or i == len(rows),";ck(text.count(a)==1,'Reviewed log stop gate patch');text=text.replace(a,b)
 # Save a pure metadata helper; only replay_log is called, never its old analyze/main.
 helper=out/'training_replay_helpers.py';helper.write_text(text,encoding='utf-8',newline='\n');mod=load_helper(helper,'owned_e1_x7_replay')
 contract=read(OLD/'training/training_source_contract.json');policy=read(P1/'e1/server_toolkit/e1_policy.json');driver=read(TRAIN/'driver.json');selected={r['id']:r for r in policy['jobs']};summaries=[];allcurves=[];sources=[];new_count=0
 for r in contract['runs']:
  dataset,method,seed=r['dataset'],r['method'],r['training_seed'];ident=dataset+'_'+method.lower()+'_seed'+str(seed);paths={};src={}
  if ident in selected:
   new_count+=1;record=driver['jobs'][ident];native=TRAIN/'attempts'/Path(record['native_stage']).name;proof=read(native/'verification/E1_COMPLETED.json');outpath=native/'outputs'/Path(proof['output_dir']).name;origin='E1_MIN50_ACCEPTED_NEW_TRAINING';servermodel=proof['output_dir']
   for name in ['config.json','metrics.json','training_log.csv']:
    path=outpath/name;paths[name]=path;read(path,proof['metadata_sha256'][name]);src[name]={'local_path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'server_path':servermodel+'/'+name}
  else:
   origin=r['source_collection'];servermodel=Path(r['source_files']['config.json']['server_path']).parent.as_posix()
   for name,e in r['source_files'].items():path=ROOT/e['local_path'];paths[name]=path;read(path,e['sha256']);src[name]=e
  cfg=read(paths['config.json']);metrics=read(paths['metrics.json']);trace=read(paths['training_log.csv']);cfg=dict(cfg)
  cfg.setdefault('max_epochs',cfg.get('epochs'));cfg.setdefault('training_batch_size',cfg.get('batch_size'));cfg.setdefault('loss','full_item_cross_entropy' if cfg.get('training_objective')=='target_level_full_ranking_cross_entropy' else cfg.get('loss'))
  ck(cfg['seed']==metrics['seed']==seed,'Training seed identity');ck(cfg['patience']==15 and cfg['max_epochs']==200 and cfg['dropout']==0.2 and cfg['training_batch_size']==128 and cfg['loss']=='full_item_cross_entropy','Uniform unchanged training profile')
  curves,summary=mod.replay_log(trace,cfg,metrics,ident);identity={'dataset':dataset,'method':method,'training_seed':seed};allcurves.extend({**identity,**v} for v in curves);summaries.append({**identity,**summary,'dropout':cfg['dropout'],'source_collection':origin,'server_model_dir':servermodel,'training_log_sha256':src['training_log.csv']['sha256']});sources.append({**identity,'source_collection':origin,'source_files':src,'minimum_stopping_epochs':50,'new_E1_model':ident in selected})
 ck(len(summaries)==24 and new_count==10,'24 log models =10new+14retained');sports=[r for r in summaries if r['dataset']=='sports']
 csvwrite(out/'training_run_summary.csv',summaries);csvwrite(out/'sports_training_summary.csv',sports);csvwrite(out/'epoch_validation_curves.csv',allcurves)
 write(out/'training_source_contract.json',{'date':'2026-10-31','scope':'Four datasets xControl/LBSP x3 seeds;10newE1 +14retained, noShuffle/Ensemble/TiSAS/F mixed.','minimum_stopping_epochs':50,'checkpoint_selection':'All-time validation best; no counter reset at50; no test-based selection.','runs':sources})
 findings={'date':'2026-10-31','status':'PASS24_CURRENT_CONTROL_LBSP_MIN50_LOGS','runs':24,'new_E1_runs':10,'retained_runs':14,'epoch_rows':len(allcurves),'minimum_epoch_rule':50,'all_final_epochs_at_least50':all(r['epochs_run']>=50 for r in summaries),'all_runs_patience_exhausted':all(r['patience_exhausted'] for r in summaries),'sports_best_epochs':{r['method']+'_'+str(r['training_seed']):r['best_epoch'] for r in sports},'sports_final_epochs':{r['method']+'_'+str(r['training_seed']):r['epochs_run'] for r in sports},'postbest_count_range':[min(r['epochs_since_best'] for r in summaries),max(r['epochs_since_best'] for r in summaries)],'sports_validation_monotonic_declines':sum(r['post_best_validation_strictly_decreasing'] for r in sports),'sports_loss_monotonic_declines':sum(r['post_best_training_loss_strictly_decreasing'] for r in sports),'replay_helper_sha256':sha(helper),'original_helper_sha256':sha(oldscript),'replay_checks':mod.CHECKS,'interpretation':['Each retained or new run satisfies epoch>=50 and bad>=15, with cap200 and all-time validation-best selection.','Minimum50 does not imply a sufficiently converged model or an optimal floor. Some all-time best checkpoints may be earlier than50.','Post-best values can fluctuate; being lower than a selected maximum is not independent evidence of overfitting.','Fresh training prefixes can differ from earlier runs, so observed changes cannot be attributed solely to the stopping gate.','Logged epoch durations are observational and cannot replace controlled isolated X6 costs.'],'interactions_or_checkpoints_accessed':False,'research_code_executed':False,'torch_imported':False,'new_models_trained_locally':0,'server_actions_required':[]}
 write(out/'training_provenance.json',findings)
 (out/'training_findings.md').write_text('# E1后X7日志诊断\n\n24个Control/LBSP由10个新记录和14个保留记录组成；均满足最少50轮、连续15轮无新最佳后停止、最多200轮。最佳模型始终在所有轮次中按验证NDCG@10选择，不在50轮重置计数。\n\n不能把最少轮数视为收敛证明；新训练前缀存在差异，不把结果改变仅归因于停止门槛。各Sports记录的最佳/停止轮次见sports_training_summary.csv；曲线由保存日志生成，不需要新服务器运行。\n',encoding='utf-8',newline='\n')
 return findings

def build_warm():
 global prediction_rows
 out=HERE/'warm_targets';out.mkdir(exist_ok=True);helper=load_helper(OLD/'warm_targets/analyze_warm_targets.py','owned_old_x7_pure_metric_helpers')
 plan=(OLD/'warm_targets/analysis_plan.json').read_bytes();(out/'analysis_plan.json').write_bytes(plan)
 seals=read(REV/'g2_closure_20261027/C_statistics/source_manifest.json')['files'];x5=ROOT/'g2_x5_results_20261025';x5seal={r['path']:r['sha256'] for r in read(x5/'REPORT_SHA256.csv')};inputs=read(INF/'scripts/e1_inference_inputs.json');newjobs={r['id']:r for r in inputs['jobs']};perseed=[];poprows=[];baselines=[];model_sources=[];membership_disagreements=0;newcount=0
 for tag in TAGS:
  core=ROOT/('g2_lbsp_ml1m_20261023' if tag=='ml1m' else 'g2_lbsp_amazon_stability_20261025')/'snapshot';labelpath=core/'outputs'/(tag+'_seed42')/'panel_labels.csv';labelrows=read(labelpath,seals[labelpath.relative_to(ROOT).as_posix()]['sha256']);labels={r['sample_id']:r for r in labelrows};metapath=x5/'derived'/(tag+'_test.csv');metarows=read(metapath,x5seal['derived/'+tag+'_test.csv']);meta={r['sample_id']:r for r in metarows};ck(set(labels)==set(meta),'Frozen label/X5 identities')
  for sid,r in labels.items():ck(r['user_id']==meta[sid]['user_id'] and r['target_item']==meta[sid]['target_item'] and int(r['affected'])==int(len({r['perm_seed'+str(p)+'_sequence_sha256'] for p in PERMS})>=2),'Target/group identities')
  ids={p:helper.select_population(labels,meta,p) for p in POPS}
  for pop,selected in ids.items():
   poprows.append({'dataset':tag,'population':pop,'n':len(selected),'full_n':len(labels),'affected_n':len(ids['affected']),'warm_n':len(ids['warm']),'cold_n':len(ids['cold'])})
   for baseline in ['random','popularity']:
    for metric in ACCURACY:
     vals=[float(meta[sid][baseline+'_'+metric]) for sid in selected];baselines.append({'dataset':tag,'method':baseline.title(),'population':pop,'n':len(selected),'metric':metric,'mean':math.fsum(vals)/len(vals) if vals else None,'estimate_kind':'analytic_population_expectation_no_training_seed_variance'})
  for method in METHODS:
   for seed in SEEDS:
    ident=tag+'_'+method.lower()+'_seed'+str(seed);new=ident in newjobs;newcount+=int(new)
    if new:
     folder=INF/'attempts'/(ident+'_a1')/'outputs'/ident;proof=read(folder.parents[1]/'verification/COMPLETED.json');hashes=proof['metadata_sha256'];origin='NEW_E1_ACCEPTED_INFERENCE'
    else:
     folder=core/('control_reference/outputs' if method=='Control' else 'outputs')/(tag+'_seed'+str(seed));hashes={p.name:v['sha256'] for name,v in seals.items() if Path(name).parent==(folder.relative_to(ROOT)) for p in [Path(name)]};origin='RETAINED_ACCEPTED_INFERENCE'
    runlabels=read(folder/'panel_labels.csv',hashes['panel_labels.csv']);ck(runlabels==labelrows,'All24 model labels identical frozen')
    predictions=[];sourcefiles={}
    for name in ['predictions_original.csv']+['predictions_perm_seed'+str(p)+'.csv' for p in PERMS]:
     path=folder/name;data=read(path,hashes[name]);sourcefiles[name]={'path':path.relative_to(ROOT).as_posix(),'sha256':hashes[name]};values={}
     for row in data:
      prediction_rows+=1;sid=row['sample_id'];ck(sid in labels and sid not in values and row['user_id']==labels[sid]['user_id'] and row['target_item']==labels[sid]['target_item'],'Prediction fixed population');rank=int(row.get('target_rank',row.get('rank')));top=tuple(str(v) for v in json.loads(row['top10_raw_item_id']));ck(1<=rank<=int(meta[sid]['candidate_count']) and len(top)==len(set(top))==10,'Rank/Top10 valid');membership_disagreements+=int((rank<=10)!=(row['target_item']in top));values[sid]={'rank':rank,'top':frozenset(top)}
     ck(set(values)==set(labels),'Complete prediction panel');predictions.append(values)
    canonical=predictions[0];panels=predictions[1:];local={}
    for pop,selected in ids.items():
     mm=helper.rank_metrics([canonical[sid]['rank'] for sid in selected])|helper.stability(panels,selected)
     for metric,value in mm.items():
      ck(value is None or math.isfinite(value),'Finite subgroup metric');perseed.append({'dataset':tag,'method':method,'training_seed':seed,'population':pop,'n':len(selected),'affected_n':len(ids['affected']),'accuracy_view':'canonical','metric':metric,'value':value});local[(pop,metric)]=value
    for parent,a,b in [('full','warm','cold'),('affected','warm_affected','cold_affected')]:
     for metric in METRICS:
      recomposed=math.fsum(len(ids[p])*local[(p,metric)] for p in [a,b] if ids[p])/len(ids[parent]);near(recomposed,local[(parent,metric)],'Weighted warm/cold reconstruction')
    model_sources.append({'id':ident,'dataset':tag,'method':method,'training_seed':seed,'origin':origin,'source_files':sourcefiles,'label_sha256':hashes['panel_labels.csv']})
 ck(newcount==10 and len(model_sources)==24,'Warm24=10new+14old');ck(len(perseed)==1296,'All6 populations*24models*9metrics')
 grouped=collections.defaultdict(list);lookup={}
 for r in perseed:grouped[(r['dataset'],r['method'],r['population'],r['metric'])].append(r);lookup[(r['dataset'],r['method'],r['population'],r['training_seed'],r['metric'])]=r
 methodrows=[];differences=[];paired=[]
 for (tag,method,pop,metric),rr in grouped.items():
  ck([r['training_seed'] for r in rr]==list(SEEDS),'Same3 fixed seeds');common={k:rr[0][k] for k in ['dataset','method','population','n','affected_n','accuracy_view','metric']};common.update(training_seeds='42;2026;3407',n_training_seeds=3);methodrows.append(common|helper.seed_stats([r['value'] for r in rr]))
  if method=='LBSP':
   values=[]
   for r in rr:
    ctrl=lookup[(tag,'Control',pop,r['training_seed'],metric)];delta=r['value']-ctrl['value'] if r['value'] is not None else None;values.append(delta);differences.append({k:r[k] for k in ['dataset','method','population','n','affected_n','training_seed','metric','accuracy_view']}|{'method_value':r['value'],'control_value':ctrl['value'],'difference':delta})
   stats=helper.seed_stats(values);paired.append(common|stats|{'mean_delta':stats['mean'],'sample_sd_delta':stats['sample_sd'],'comparison':'LBSP_minus_same_seed_Control'})
 for filename,data in [('per_seed_metrics.csv',perseed),('method_summary.csv',methodrows),('paired_differences_per_seed.csv',differences),('paired_difference_summary.csv',paired),('population_counts.csv',poprows),('analytic_baselines.csv',baselines)]:csvwrite(out/filename,data)
 oldpop=read(OLD/'warm_targets/population_counts.csv');ck([{k:str(v) for k,v in r.items()} for r in poprows]==oldpop,'Frozen population counts unchanged')
 oldmethods=read(OLD/'warm_targets/method_summary.csv');oldbeauty={(r['method'],r['population'],r['metric']):r for r in oldmethods if r['dataset']=='beauty'}
 for r in methodrows:
  if r['dataset']=='beauty':
   old=oldbeauty[(r['method'],r['population'],r['metric'])]
   for k in ['mean','sample_sd','ci95_lower','ci95_upper']:
    if r[k]is not None:near(r[k],old[k],'Unchanged Beauty subgroup result')
 ml={(r['dataset'],r['method'],r['population'],r['metric']):r for r in methodrows};pl={(r['dataset'],r['population'],r['metric']):r for r in paired};findings={'status':'COMPLETE_WARM_TARGET_DIAGNOSTIC_DESCRIPTIVE_RESULTS','population_counts':poprows,'warm_accuracy_and_stability':[],'interpretation':['Fixed target exposure and affected masks retained; model-specific staleX5 rank columns are not reused.','Positive training exposure is not randomized; these subgroup directions do not identify causal effects.','Warm target exposure does not guarantee positive LBSP-minus-Control accuracy.','A zero training-positive count does not imply no embedding updates under full-item CE.']}
 for tag in TAGS:
  for pop in ['warm','warm_affected']:
   row={'dataset':tag,'population':pop,'n':ml[(tag,'Control',pop,'ndcg10')]['n']}
   for method in METHODS:
    for metric in ['ndcg10','mean_rank','overlap10','hitflip']:r=ml[(tag,method,pop,metric)];row[method+'_'+metric]={'mean':r['mean'],'sample_sd':r['sample_sd']}
   row['paired_ndcg10_difference']={k:pl[(tag,pop,'ndcg10')][k] for k in ['mean_delta','sample_sd_delta','ci95_lower','ci95_upper']};findings['warm_accuracy_and_stability'].append(row)
 write(out/'findings.json',findings);write(out/'source_manifest.json',{'scope':'Accepted10new +14old prediction lineages andfixedX5label/candidate metadata; no raw/checkpoints','files':reads,'models':model_sources})
 verification={'date':'2026-10-31','status':'PASS_X7_24_CURRENT_WARM_COLD_TARGET_DIAGNOSTICS','model_configurations':24,'new_E1_models':10,'retained_models':14,'prediction_rows':prediction_rows,'rank10_top10_membership_mismatches_preserved':membership_disagreements,'stale_X5_model_rank_columns_used':False,'frozen_populations_unchanged':True,'Beauty_results_unchanged':True,'rows':{'per_seed_metrics':len(perseed),'method_summary':len(methodrows),'paired_differences_per_seed':len(differences),'paired_difference_summary':len(paired),'population_counts':len(poprows),'analytic_baselines':len(baselines)},'raw_interaction_or_checkpoint_read_or_stat':False,'new_model_training_or_inference':False,'submitted_research_code_executed':False,'source_files':len(reads),'checks':checks,'numeric_comparisons':numeric}
 write(out/'verification.json',verification);return verification

def main():
 HERE.mkdir(exist_ok=True);t=build_training();w=build_warm();write(HERE/'X7_acceptance.json',{'status':'PASS_X7_CURRENT24_LOGS_AND_TARGET_DIAGNOSTICS','training':t,'warm_targets':w,'all_outputs_sha256':{p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in ['.csv','.json'] and p.name!='X7_acceptance.json'},'no_PDF_or_LaTeX_generated':True});print(json.dumps({'training':t,'warm_targets':w},indent=2))
if __name__=='__main__':main()
