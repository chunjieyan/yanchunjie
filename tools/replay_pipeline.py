#!/usr/bin/env python3
"""Inspect all 42 training / 54 evaluation configurations without importing research models.

This tool preserves frozen scientific sources. It can prepare an isolated trainer
copy with module-level path changes and prove that all function/class ASTs stay
identical. It deliberately does not present the 19-run E1 follow-up as a full run.
Original drivers remain the executable experiment implementations; their guarded
stage/data/checkpoint contracts must be satisfied before invoking a prepared copy.
"""
from pathlib import Path
import argparse,ast,csv,difflib,hashlib,json,os,re,shutil,sys
REPO=Path(__file__).resolve().parents[1]
RECORDED_ROOT='/root/autodl-tmp/time_uncertainty_rec'
PERMUTATIONS=(101,202,303,404,505)

def disk(p):
 p=Path(p).absolute()
 return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p

def sha(p):
 h=hashlib.sha256()
 with disk(p).open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()

def read(p):return json.loads(disk(p).read_text(encoding='utf-8-sig'))
def dump(p,j):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);disk(p).write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def norm(s):return str(s).replace('\\','/')
def key(r):return (r['dataset'],r['method'],int(r['training_seed']))
def jid(r):return f"{r['dataset']}_{r['method'].lower()}_seed{r['training_seed']}"
def wrapped_rows(value):return value if isinstance(value,list) else value.get('models',value.get('runs',value.get('model_matrix',[])))

def matrices():
 models=wrapped_rows(read(REPO/'config/model_matrix.json'));ens=[dict(row,method='Ensemble') for row in wrapped_rows(read(REPO/'config/ensemble_evaluations.json'))]
 expected={(d,m,s) for d in ('ml1m','beauty','sports','toys') for m in ('Control','Shuffle','LBSP') for s in (42,2026,3407)}|{(d,'TiSASRec',s) for d in ('ml1m','beauty') for s in (42,2026,3407)}
 assert len(models)==42 and {key(x) for x in models}==expected,'Training matrix is not the complete independent 42-model set'
 assert len(ens)==12 and {key(x) for x in ens}=={(d,'Ensemble',s) for d in ('ml1m','beauty','sports','toys') for s in (42,2026,3407)},'Ensemble must reuse 12 Control checkpoints'
 return models,ens

def sources():return read(REPO/'provenance/source_index.json')['files']
def repository_path(x):return x.get('repository_path',x['package_relative_path'])
def resolve_ref(reference,files):
 if reference.startswith('archive:'):
  arc,member=reference[len('archive:'):].rsplit('!',1)
  hits=[x for x in files if norm(x.get('archive_path',''))==norm(arc) and x.get('archive_member')==member]
 else:hits=[x for x in files if norm(x.get('local_path',''))==norm(reference)]
 if not hits:return None
 hashes={x['sha256'] for x in hits};assert len(hashes)==1,'Ambiguous source reference with differing SHA256'
 return sorted(hits,key=lambda x:(len(repository_path(x)),repository_path(x)))[0]

def definitions(tree):
 found={}
 def visit(body,prefix=''):
  for n in body:
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
    name=prefix+n.name;found[name]=hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest();visit(n.body,name+'.')
 visit(tree.body);return found

def inspect_source(x):
 p=REPO/repository_path(x);raw=disk(p).read_bytes();assert hashlib.sha256(raw).hexdigest()==x['sha256'],repository_path(x)
 text=raw.decode('utf-8-sig');tree=ast.parse(text,filename=repository_path(x));assignments={};guards=[];cli=[];absolute=[]
 for node in tree.body:
  if isinstance(node,(ast.Assign,ast.AnnAssign)):
   targets=node.targets if isinstance(node,ast.Assign) else [node.target]
   for t in targets:
    if isinstance(t,ast.Name) and re.search(r'ROOT|BASE|STAGE|OUTPUT|SEED|PATIENCE|EPOCH',t.id):assignments[t.id]=ast.unparse(node.value)
 for n in ast.walk(tree):
  if isinstance(n,ast.If) and re.search(r'bad_epochs|patience',ast.unparse(n.test),re.I):guards.append(ast.unparse(n.test))
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant):cli.append({'flag':n.args[0].value,'declaration':ast.unparse(n)})
  if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith(('/root/','/home/')):absolute.append(n.value)
 return {'source':repository_path(x),'sha256':x['sha256'],'module_settings':assignments,'early_stopping_tests':guards,'minimum_50_literal_gate_detected':any(re.search(r'epoch\s*>=\s*50',g) for g in guards),'cli_arguments':cli,'absolute_path_literals':sorted(set(absolute)),'class_function_ast_sha256':definitions(tree)}

def evaluation_records(models,ens):
 records=[]
 for x in models+ens:
  r=dict(x);r['output_directory']=f"runs/{x['dataset']}/{x['method'].lower()}/seed{x['training_seed']}";r['independently_trained']=x['method']!='Ensemble';r['training_job_id']=f"{x['dataset']}_{'control' if x['method']=='Ensemble' else x['method'].lower()}_seed{x['training_seed']}";r['required_predictions']=[f'predictions_group{i}.csv' for i in range(1,6)] if x['method']=='Ensemble' else ['predictions_original.csv']+[f'predictions_perm_seed{s}.csv' for s in PERMUTATIONS];records.append(r)
 assert len(records)==54
 return records

def plan(phase):
 models,ens=matrices();files=sources();inspections={};jobs=[];missing=[]
 for row in models:
  resolved=[]
  for ref in row.get('training_source_candidates',[]):
   x=resolve_ref(ref,files)
   if x is None:missing.append({'job':jid(row),'source_reference':ref});continue
   path=repository_path(x)
   if path not in inspections:inspections[path]=inspect_source(x)
   resolved.append({'source':path,'sha256':x['sha256'],'minimum_50_literal_gate_detected':inspections[path]['minimum_50_literal_gate_detected']})
  jobs.append(dict(row,job_id=jid(row),resolved_training_sources=resolved,source_resolution_complete=bool(resolved)))
 result={'schema':'KAIS_COMPLETE_PIPELINE_PLAN_V1','status':'PASS_FULL_MATRIX_AND_SOURCE_REFERENCE_PLAN' if not missing else 'SOURCE_REFERENCE_PLAN_HAS_GAPS','phase':phase,'independent_training_models':42,'evaluation_configurations':54,'Ensemble_reused_Control_models':12,'training_jobs':jobs if phase in ('all','training') else [],'evaluation_jobs':evaluation_records(models,ens) if phase in ('all','evaluation') else [],'source_contract_inspections':inspections,'missing_source_references':missing,'recorded_Linux_root':RECORDED_ROOT,'training_or_evaluation_executed':False,'execution_note':'This inventories complete scientific sources and contracts. E1 is a 19-model follow-up, not a fresh 42-model driver. Candidate historical trainer identity and actual invocation evidence retain their original evidence levels. No new GPU regression has been run; absence of that regression is not absence of source code.','historical_replay_note':'Replaying initial experiments plus the recorded E1 follow-up can train more than 42 physical runs; only 42 final independent checkpoints and 54 evaluations enter the final registry.'}
 return result

def preflight(phase):
 result=plan(phase);bad=[];checked=0
 for x in sources():
  p=REPO/repository_path(x)
  if not disk(p).is_file():bad.append({'source':repository_path(x),'reason':'missing'});continue
  if sha(p)!=x['sha256']:bad.append({'source':repository_path(x),'reason':'sha256 mismatch'});continue
  ast.parse(disk(p).read_text(encoding='utf-8-sig'),filename=repository_path(x));checked+=1
 return {'schema':'KAIS_OFFLINE_REPLAY_PREFLIGHT_V1','status':'PASS_OFFLINE_FULL_SOURCE_AND_MATRIX_PREFLIGHT' if not bad and not result['missing_source_references'] else 'FAIL_OFFLINE_REPLAY_PREFLIGHT','sources_checked':checked,'source_failures':bad,'independent_training_models':42,'evaluation_configurations':54,'source_reference_failures':result['missing_source_references'],'GPU_runtime_imported':False,'scientific_training_or_evaluation_run':False,'new_GPU_regression_completed':False,'readiness_note':'Checks integrity, Python syntax and complete matrix/source references. Original root/stage/manifest/data/checkpoint contracts still apply to execution; this preflight does not assert they have all been instantiated.'}

def prepare_job(job_id,data_root,runtime_root):
 models,_=matrices();row=next((x for x in models if jid(x)==job_id),None);assert row,'Unknown training job'
 files=sources();refs=row.get('training_source_candidates',[]);assert refs,'No frozen source candidate';x=resolve_ref(refs[0],files);assert x,'Unresolved source reference'
 source=REPO/repository_path(x);raw=disk(source).read_bytes();assert hashlib.sha256(raw).hexdigest()==x['sha256'];text=raw.decode('utf-8-sig');tree=ast.parse(text)
 stage=runtime_root.resolve()/job_id;assert not stage.exists(),'Runtime job already exists; refusing overwrite';assert not stage.is_relative_to(REPO.resolve()),'Runtime must be outside frozen source repository'
 assert Path(data_root).is_absolute(),'Data root must be absolute';data_root=Path(data_root)
 lines=text.encode('utf-8').splitlines(keepends=True);offsets=[0]
 for line in lines:offsets.append(offsets[-1]+len(line))
 def span(node):return offsets[node.lineno-1]+node.col_offset,offsets[node.end_lineno-1]+node.end_col_offset
 patches=[];stage_literals=[]
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.endswith('STAGE') for t in n.targets):
   for c in ast.walk(n.value):
    if isinstance(c,ast.Constant) and isinstance(c.value,str) and c.value.startswith(RECORDED_ROOT+'/'):stage_literals.append(c.value)
 old_stage=max(stage_literals,key=len) if stage_literals else None
 def collect(node):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):return
  if isinstance(node,ast.Assign):
   names=[t.id for t in node.targets if isinstance(t,ast.Name)]
   if names==['ROOT']:
    a,b=span(node.value);patches.append((a,b,f'Path({str(data_root)!r})'.encode(),'module ROOT'));return
   if len(names)==1 and names[0] in ('OUTPUT_DIR','OUT_DIR','OUTDIR'):
    a,b=span(node.value);patches.append((a,b,f'Path({str(stage/"outputs")!r})'.encode(),'isolated output path'));return
  if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith(RECORDED_ROOT):
   v=node.value
   if old_stage and v.startswith(old_stage):new=str(stage)+v[len(old_stage):]
   else:new=str(data_root)+v[len(RECORDED_ROOT):]
   a,b=span(node);patches.append((a,b,repr(new).encode(),'module path literal'));return
  for child in ast.iter_child_nodes(node):collect(child)
 for node in tree.body:collect(node)
 candidate=text.encode('utf-8')
 for a,b,repl,_ in sorted(patches,reverse=True):candidate=candidate[:a]+repl+candidate[b:]
 derived=candidate.decode('utf-8');assert definitions(tree)==definitions(ast.parse(derived)),'Scientific function/class AST changed'
 # Preserve all siblings and stage configuration when present, but never interactions/weights.
 primary_rel=Path(repository_path(x));script_prefix=primary_rel.parent.as_posix()+'/';stage_prefix=primary_rel.parent.parent.as_posix()+'/'
 copies=[]
 for item in files:
  n=repository_path(item)
  if n.startswith(script_prefix):
   source_item=REPO/n;destination=stage/'scripts'/n[len(script_prefix):];destination.parent.mkdir(parents=True,exist_ok=True);assert sha(source_item)==item['sha256'];disk(destination).write_bytes(disk(source_item).read_bytes());copies.append({'source':n,'runtime_path':str(destination),'sha256':item['sha256']})
 configs=read(REPO/'provenance/source_index.json').get('configuration_dependencies',[])
 for item in configs:
  n=item.get('repository_path',item.get('package_relative_path',''))
  if n.startswith(stage_prefix) and not n.startswith(script_prefix):
   source_item=REPO/n
   if disk(source_item).is_file():
    destination=stage/n[len(stage_prefix):];destination.parent.mkdir(parents=True,exist_ok=True);disk(destination).write_bytes(disk(source_item).read_bytes());copies.append({'source':n,'runtime_path':str(destination),'sha256':sha(source_item)})
 dest=stage/'scripts'/source.name;dest.parent.mkdir(parents=True,exist_ok=True);disk(dest).write_text(derived,encoding='utf-8')
 result={'status':'PASS_ISOLATED_PATH_ONLY_RUNTIME_DERIVATION','job':job_id,'original_source':repository_path(x),'original_sha256':x['sha256'],'runtime_source':str(dest),'runtime_sha256':sha(dest),'module_path_patches':len(patches),'all_class_function_ASTs_identical':True,'class_function_ast_sha256':definitions(tree),'copied_source_and_stage_records':copies,'unmodified_legacy_sources':True,'training_started':False,'loss_masks_permutations_aggregation_and_early_stopping_unchanged':True,'original_data_root':str(data_root),'requires_legacy_root_stage_and_manifest_contract_review':True,'direct_python_argv_after_contract_validation':[sys.executable,str(dest)],'execution_validation_note':'Path-only construction is verified statically. It has not imported PyTorch, validated CUDA predictions or satisfied every referenced frozen archive/manifest. Do not label this a completed new experiment.'}
 result['unified_diff']=''.join(difflib.unified_diff(text.splitlines(True),derived.splitlines(True),fromfile='frozen/'+source.name,tofile='runtime/'+source.name))
 dump(stage/'runtime_derivation.json',result);return result

def registry_template():
 models,ens=matrices();return {'schema':'KAIS_RUN_REGISTRY_V1','status':'TEMPLATE_EXPECTED_FRESH_OUTPUTS_NOT_EXECUTED','runs':evaluation_records(models,ens),'training_model_count':42,'evaluation_configuration_count':54,'checkpoint_note':'The 12 Ensemble entries must refer to the matching Control checkpoint; they do not train independently.'}

def validate_results(root,registry):
 registry_spec=read(registry);rows=registry_spec['runs'];models,ens=matrices();assert len(rows)==54 and {key(x) for x in rows}=={key(x) for x in models+ens},'Registry must cover every one of the 54 configurations exactly once'
 failures=[];out=[];panels={}
 for row in rows:
  folder=(root/row['output_directory']).resolve();assert folder.is_relative_to(root.resolve()),'Output directory escapes result root'
  names=['panel_labels.csv','predictions_original.csv']+([f'predictions_group{i}.csv' for i in range(1,6)] if row['method']=='Ensemble' else [f'predictions_perm_seed{s}.csv' for s in PERMUTATIONS])
  hashes={}
  for name in names:
   p=folder/name
   if not p.is_file():failures.append({'job':jid(row),'missing':name});continue
   hashes[name]=sha(p)
  p=folder/'panel_labels.csv'
  if p.is_file():
   with p.open(encoding='utf-8-sig',newline='') as f:labels=list(csv.DictReader(f))
   require={'sample_id','user_id','target_item','last_bucket_size','affected'}
   if not labels or not require.issubset(labels[0]):failures.append({'job':jid(row),'reason':'Panel label schema or rows missing'})
   else:
    identity=sorted(tuple(r[k] for k in sorted(require)) for r in labels)
    previous=panels.setdefault(row['dataset'],identity)
    if previous!=identity:failures.append({'job':jid(row),'reason':'Dataset user/target/bucket/affected identities differ across methods or seeds'})
  out.append({'dataset':row['dataset'],'method':row['method'],'training_seed':row['training_seed'],'output_directory':row['output_directory'],'files_sha256_scope':'output_directory','files_sha256':hashes})
 return {'status':'PASS_COMPLETE_54_CONFIGURATION_RESULT_CONTRACT' if not failures else 'FAIL_FRESH_RESULT_CONTRACT','configurations_checked':54,'result_root':str(root),'runs':out,'failures':failures,'models_or_predictions_regenerated':False,**{name:registry_spec[name] for name in ('label_files','target_metadata','baseline_metadata') if name in registry_spec}}

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('all','training','evaluation'),default='all');m=p.add_mutually_exclusive_group(required=True);m.add_argument('--plan',action='store_true');m.add_argument('--preflight',action='store_true');m.add_argument('--prepare-job',metavar='JOB_ID');m.add_argument('--write-registry',type=Path);m.add_argument('--validate-results',type=Path);p.add_argument('--runtime-root',type=Path);p.add_argument('--data-root',type=Path);p.add_argument('--run-registry',type=Path);p.add_argument('--output',type=Path)
 a=p.parse_args(argv)
 if a.plan:result=plan(a.phase)
 elif a.preflight:result=preflight(a.phase)
 elif a.prepare_job:
  if not a.runtime_root or not a.data_root:p.error('--prepare-job requires --runtime-root and --data-root')
  result=prepare_job(a.prepare_job,a.data_root,a.runtime_root)
 elif a.write_registry:result=registry_template();dump(a.write_registry,result)
 else:
  if not a.run_registry:p.error('--validate-results requires --run-registry')
  result=validate_results(a.validate_results.resolve(),a.run_registry)
 if a.output:dump(a.output,result)
 print(json.dumps({k:result[k] for k in ('status','phase','independent_training_models','evaluation_configurations','sources_checked','training_model_count','evaluation_configuration_count','job','runtime_source','all_class_function_ASTs_identical') if k in result},ensure_ascii=False,indent=2))
 return 1 if result['status'].startswith('FAIL_') or result.get('missing_source_references') else 0
if __name__=='__main__':sys.exit(main())
