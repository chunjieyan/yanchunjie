#!/usr/bin/env python3
"""Offline source integrity, requirements inspection and plan inspection; never starts training."""
from pathlib import Path
import argparse,ast,csv,hashlib,importlib.metadata as metadata,json,os,re,runpy,sys
ROOT=Path(__file__).resolve().parents[1]
def native(p):
 p=Path(p).resolve()
 return Path('\\\\?\\'+str(p))if os.name=='nt'and not str(p).startswith('\\\\?\\')else p
def digest(p):
 h=hashlib.sha256()
 with native(p).open('rb')as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(x):print(json.dumps(x,ensure_ascii=False,indent=2))
def closure():
 m=read_json(ROOT/'MANIFEST.sha256.json');bad=[]
 for name,v in m['files'].items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT.resolve()):bad.append({'path':name,'error':'manifest path leaves repository'})
  elif not native(p).is_file():bad.append({'path':name,'error':'missing'})
  elif digest(p)!=v['sha256']:bad.append({'path':name,'error':'sha256 mismatch'})
 return m,bad
def integrity():
 m,bad=closure();dump({'status':'PASS_SOURCE_INTEGRITY'if not bad else'FAIL_SOURCE_INTEGRITY','checked_files':len(m['files']),'failures':bad,'network_access':False,'research_models_imported_or_run':False});return bool(bad)
def packages():
 result={}
 for name in ['torch','numpy','pandas','scipy','tqdm','matplotlib','pypdf']:
  try:result[name]=metadata.version(name)
  except metadata.PackageNotFoundError:result[name]=None
 return result
def data_check(data_root,verify):
 records=list(csv.DictReader((ROOT/'data_manifest/expected_sha256.csv').open(encoding='utf-8-sig',newline='')));out=[]
 for row in records:
  p=(data_root/row['path']).resolve()
  status='PATH_OUTSIDE_ROOT'if not p.is_relative_to(data_root.resolve())else'MISSING'if not p.is_file()else'PRESENT_NOT_HASHED'
  item={'path':row['path'],'expected_sha256':row['sha256'],'status':status}
  if verify and status=='PRESENT_NOT_HASHED':
   item['actual_sha256']=digest(p);item['status']='MATCH'if item['actual_sha256']==row['sha256']else'MISMATCH'
  out.append(item)
 return out
def preflight(a):
 m,bad=closure();p=packages();data=data_check(a.data_root.resolve(),a.verify_data)if a.data_root else[]
 missing=[k for k,v in p.items()if v is None];data_bad=[x for x in data if x['status']not in('MATCH','PRESENT_NOT_HASHED')]
 syntax=[]
 for rel in m['files']:
  if rel.endswith('.py'):
   try:ast.parse(native(ROOT/rel).read_text(encoding='utf-8-sig'),filename=rel,feature_version=(3,10))
   except (SyntaxError,UnicodeError)as e:syntax.append({'path':rel,'error':str(e)})
 result={'status':'PASS_OFFLINE_SOURCE_PREFLIGHT'if not bad and not syntax else'FAIL_OFFLINE_SOURCE_PREFLIGHT','source_failures':bad,'python_syntax_failures':syntax,'interpreter':sys.version,'installed_distributions':p,'missing_optional_execution_distributions':missing,'data_root_provided':bool(a.data_root),'data_verification':data,'path_bound_sources':'See relocation-plan; archived scientific drivers are not automatically portable.','full_training_readiness':'NOT_CLAIMED: original stage/manifests, exact data, device/runtime and source-path guards require reconstruction and validation.','GPU_or_training_executed':False,'network_access':False}
 dump(result);return bool(bad or syntax or(a.strict and(missing or not a.data_root or data_bad)))
def plan():
 p=read_json(ROOT/'config/model_matrix.json');dump({'status':'RECORDED_MODEL_PLAN_ONLY','models':p,'training_started':False,'note':'42 independent fitted models. Ensemble reuses 12 corresponding Control configurations; input views are not independent training runs.'});return 0
def relocation(a):
 p=read_json(ROOT/'provenance/source_index.json');files=p['files'];absolute=[]
 for f in files:
  for value in f.get('root_literals',[]):
   if isinstance(value,str)and(value.startswith('/')or re.match(r'^[A-Za-z]:[\\/]',value)):absolute.append({'source':f['repository_path'],'literal':value})
 dump({'status':'RELOCATION_REVIEW_PLAN_ONLY','proposed_workspace_root':str(a.target_root)if a.target_root else None,'recorded_server_workspace_root':'/root/autodl-tmp/time_uncertainty_rec','path_literals':absolute,'actions':['Keep archived scientific sources byte-identical.','Acquire data separately and compare data_manifest hashes.','Restore recorded stage structures and non-data manifests from supplied source records.','Inspect original drivers and their root/stage/absolute-__file__ guards before running.','Where supported, regenerate stages from source templates. Do not silently rewrite sealed manifests or expected hashes.','For an adapted portable runner, work on a separate copy and record every change; validate canonical predictions and checkpoint/state identities before using it.'],'known_constraint':'run_e1_min50.py enforces POSIX and the recorded exact workspace root; several child scripts bind a concrete attempt stage. A --root option alone does not establish portability.','files_rewritten':False,'training_started':False});return 0
def redraw(a):
 _,bad=closure()
 if bad:dump({'status':'FAIL_SOURCE_INTEGRITY','failures':bad});return 1
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);fig=ROOT/'figures';oldargv=sys.argv[:];oldpath=sys.path[:]
 try:
  if a.font_dir:
   from matplotlib import font_manager
   for f in a.font_dir.rglob('*'):
    if f.suffix.lower()in('.ttf','.otf'):font_manager.fontManager.addfont(str(f))
  sys.path.insert(0,str(fig))
  for name in ['build_structural_figures.py','build_quantitative_figures.py']:
   sys.argv=[str(fig/name),'--output-dir',str(out)]
   if name=='build_structural_figures.py'and a.font_dir:sys.argv+=['--font-dir',str(a.font_dir.resolve())]
   runpy.run_path(str(fig/name),run_name='__main__')
 finally:sys.argv=oldargv;sys.path[:]=oldpath
 dump({'status':'FIGURES_REDRAWN_FROM_AGGREGATES','output_directory':str(out),'training_started':False,'raw_interactions_or_checkpoints_used':False});return 0
def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
 s.add_parser('source-integrity');f=s.add_parser('preflight');f.add_argument('--data-root',type=Path);f.add_argument('--verify-data',action='store_true');f.add_argument('--strict',action='store_true',help='Also fail for missing execution packages/data; never trains.')
 s.add_parser('model-plan');r=s.add_parser('relocation-plan');r.add_argument('--target-root',type=Path)
 r=s.add_parser('redraw');r.add_argument('--output-dir',type=Path,default=ROOT/'outputs/figures');r.add_argument('--font-dir',type=Path)
 a=p.parse_args();code=integrity()if a.command=='source-integrity'else preflight(a)if a.command=='preflight'else plan()if a.command=='model-plan'else relocation(a)if a.command=='relocation-plan'else redraw(a);sys.exit(int(code))
if __name__=='__main__':main()
