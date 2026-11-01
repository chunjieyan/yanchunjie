#!/usr/bin/env python3
"""Measure a complete16-row inference-cost matrix on one newly registered GPU."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid
import zipfile

HERE=Path(__file__).resolve().parent
VERSION='e1_cost_refresh_v1'
PREFIX='kais_e1_cost_refresh_v1_'
INPUT_SHA='f64a2a74b296d6d549233b31952d90aaeaac4ad21e42e69c496c995ee29550c2'
RUNTIME=('run_e1_cost_refresh.py','e1_cost_refresh_inputs.json','e1_accuracy_validator.py','frozen_x6_inference.py','x3_core_helpers.py','x3_inference_adapter.py','x3_inference_protocol.py','x6_model_definitions.py')
HELPERS=('x3_core_helpers.py','x3_inference_adapter.py','x3_inference_protocol.py','x6_model_definitions.py')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def safe(root,relative):
    root=Path(root).resolve();p=Path(relative)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in str(relative) and ':' not in str(relative),'Unsafe relative path')
    result=root/p;require(result.resolve().is_relative_to(root),'Path leaves root')
    for part in (result,)+tuple(result.parents):
        if part==root.parent:break
        require(not part.is_symlink(),'Symlink rejected: '+str(part))
    return result


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex[:10]+'.tmp')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def load_inputs():
    require(sha(HERE/'e1_cost_refresh_inputs.json')==INPUT_SHA,'Reviewed cost-refresh inputs changed')
    inputs=read(HERE/'e1_cost_refresh_inputs.json');delivery=read(HERE/'delivery_manifest.json')
    require(set(delivery['runtime_sha256'])==set(RUNTIME),'Delivery closure differs')
    for name,expected in delivery['runtime_sha256'].items():require(sha(safe(HERE,name))==expected,'Delivered runtime changed: '+name)
    require(inputs['version']==VERSION and len(inputs['jobs'])==16,'Expected complete16-row cost matrix')
    require({(j['tag'],j['method'],j['seed']) for j in inputs['jobs']}=={(tag,method,42) for tag in ('ml1m','beauty','sports','toys') for method in ('control','shuffle','ensemble','lbsp')},'Cost profile inventory differs')
    require(len({j['id'] for j in inputs['jobs']})==16 and len(inputs['source_accuracy_jobs'])==18,'Duplicate cost task or missing18 accuracy models')
    require(len(inputs['expected_environment_without_uuid'])==15 and 'device_uuid' not in inputs['expected_environment_without_uuid'],'Only15 shared software/device/backend fields may be pinned before execution')
    for name,expected in inputs['frozen_helper_sha256'].items():require(sha(HERE/name)==expected,'Frozen helper changed: '+name)
    return inputs,delivery


def verify_root(root,inputs,job=None):
    files=inputs['root_files_sha256'] if job is None else {p:inputs['root_files_sha256'][p] for p in job['required_root_files']}
    for relative,expected in files.items():require(sha(safe(root,relative))==expected,'Saved model/input/evidence changed: '+relative)
    return len(files)


def validate_accuracy(root,inputs,validator_path=None):
    path=Path(validator_path) if validator_path else HERE/'e1_accuracy_validator.py'
    require(sha(path)==inputs['source_accuracy_validator_sha256'],'Unchanged accuracy validator seal differs')
    validator=module(path,'e1_saved_accuracy_validator')
    source=Path(inputs['source_accuracy_stage'])
    manifest,original=validator.verify_stage(root,source)
    require(sha(source/'scripts/e1_inference_inputs.json')==inputs['source_accuracy_input_sha256'] and original['jobs']==inputs['source_accuracy_jobs'],'Source18 accuracy input identity differs')
    state=read(source/'verification/driver.json')
    for job in inputs['source_accuracy_jobs']:
        record=state['jobs'][job['id']];require(record.get('status')=='COMPLETED','Source accuracy job is incomplete: '+job['id'])
        native=validator.checked_attempt(source,job['id'],record['native_stage'])
        validator.verify_completion(root,source,job,native)
    return {'status':'PASS18_SAVED_ACCURACY_JOBS_UNCHANGED','stage':str(source),'jobs':18,'validator_sha256':inputs['source_accuracy_validator_sha256']}


def prepare(root,inputs,delivery):
    root=Path(root).resolve();require(str(root)==inputs['root'] and root.is_dir(),'Wrong Linux project root')
    verify_root(root,inputs);accuracy=validate_accuracy(root,inputs)
    require(not list(root.glob(PREFIX+'*')),'Existing cost-refresh stage found; resume it with --stage')
    stage=root/(PREFIX+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:10]);stage.mkdir()
    for folder in ('scripts','attempts','verification','reports'):(stage/folder).mkdir()
    for name in RUNTIME+('delivery_manifest.json',):shutil.copyfile(HERE/name,stage/'scripts'/name)
    manifest={'version':VERSION,'root':str(root),'stage':str(stage),'input_sha256':INPUT_SHA,'jobs':[j['id'] for j in inputs['jobs']],
              'tools_sha256':{'scripts/'+n:sha(stage/'scripts'/n) for n in RUNTIME+('delivery_manifest.json',)},
              'measurement_protocol':inputs['measurement_protocol'],'source_accuracy_validation':accuracy,
              'expected_environment_without_uuid':inputs['expected_environment_without_uuid'],'training_started':False,'accuracy_inference_rerun':False}
    write(stage/'manifest.json',manifest)
    write(stage/'verification/driver.json',{'version':VERSION,'root':str(root),'stage':str(stage),'manifest_sha256':sha(stage/'manifest.json'),
          'status':'PREPARED','jobs':{j['id']:{'attempts':[]} for j in inputs['jobs']},'training_started':False,'accuracy_inference_rerun':False})
    return stage


def verify_stage(root,stage):
    root=Path(root).resolve();stage=Path(stage).resolve()
    require(stage.parent==root and stage.name.startswith(PREFIX) and not stage.is_symlink(),'Expected direct isolated cost-refresh stage')
    manifest=read(stage/'manifest.json')
    require(manifest['version']==VERSION and manifest['root']==str(root) and manifest['stage']==str(stage) and manifest['input_sha256']==INPUT_SHA,'Cost stage identity/input seal differs')
    require(set(manifest['tools_sha256'])=={'scripts/'+n for n in RUNTIME+('delivery_manifest.json',)},'Cost runtime closure differs')
    for relative,expected in manifest['tools_sha256'].items():require(sha(safe(stage,relative))==expected,'Cost runtime changed: '+relative)
    inputs=read(stage/'scripts/e1_cost_refresh_inputs.json')
    require(sha(stage/'scripts/e1_cost_refresh_inputs.json')==INPUT_SHA and manifest['jobs']==[j['id'] for j in inputs['jobs']] and manifest['measurement_protocol']==inputs['measurement_protocol'] and manifest['expected_environment_without_uuid']==inputs['expected_environment_without_uuid'],'Cost stage inputs/protocol differ')
    return manifest,inputs


def checked_attempt(stage,jobid,value):
    path=Path(value);require(path.is_absolute() and path.parent==stage/'attempts' and path.name.startswith(jobid+'_a'),'Cost attempt leaves isolated stage')
    return safe(stage,path.relative_to(stage).as_posix())


def benchmark_module(root,stage,job,native,create=False):
    outer,inputs=verify_stage(root,stage);native=checked_attempt(stage,job['id'],native)
    plan=dict(inputs['datasets'][job['tag']]);run={k:job[k] for k in ('model_dir','checkpoint','checkpoint_sha256','best_epoch')}
    run.update(reference_map='references/item_id_map.csv',reference_predictions='references/predictions_saved.csv',required_root_files=job['required_root_files'])
    plan['runs']={job['method']:run}
    context={'version':VERSION,'datasets':{job['tag']:plan},'protocol':inputs['measurement_protocol']}
    refs={'references/item_id_map.csv':inputs['root_files_sha256'][job['reference_map']], 'references/predictions_saved.csv':inputs['root_files_sha256'][job['reference_predictions']]}
    if create:
        (native/'scripts').mkdir();(native/'references').mkdir()
        for name in HELPERS:shutil.copyfile(stage/'scripts'/name,native/'scripts'/name)
        shutil.copyfile(safe(root,job['reference_map']),native/'references/item_id_map.csv')
        shutil.copyfile(safe(root,job['reference_predictions']),native/'references/predictions_saved.csv')
        value={'version':VERSION,'root':str(root),'stage':str(native),'parent_stage':str(stage),'parent_manifest_sha256':sha(stage/'manifest.json'),
               'input_sha256':INPUT_SHA,'job':job,'tools_sha256':{'scripts/'+n:inputs['frozen_helper_sha256'][n] for n in HELPERS},
               'references_sha256':refs,'frozen_worker_sha256':inputs['frozen_helper_sha256']['frozen_x6_inference.py'],'protocol':inputs['measurement_protocol']}
        write(native/'manifest.json',value)
    def checked(r,n):
        require(Path(r)==root and Path(n)==native,'Frozen cost worker targets another stage')
        value=read(native/'manifest.json')
        require(value['version']==VERSION and value['root']==str(root) and value['stage']==str(native) and value['parent_stage']==str(stage) and value['parent_manifest_sha256']==sha(stage/'manifest.json') and value['input_sha256']==INPUT_SHA,'Cost attempt manifest identity differs')
        require(value['job']==job and value['protocol']==inputs['measurement_protocol'] and value['frozen_worker_sha256']==inputs['frozen_helper_sha256']['frozen_x6_inference.py'],'Cost job/source/protocol differs')
        require(value['tools_sha256']=={'scripts/'+n:inputs['frozen_helper_sha256'][n] for n in HELPERS} and value['references_sha256']==refs,'Cost helper/reference bindings differ')
        for relative,expected in dict(value['tools_sha256'],**value['references_sha256']).items():require(sha(safe(native,relative))==expected,'Cost helper/reference changed: '+relative)
        verify_stage(root,stage)
        return value,context
    checked(root,native)
    frozen=module(stage/'scripts/frozen_x6_inference.py','e1_frozen_cost_refresh_worker')
    frozen.VERSION=VERSION;frozen.INPUT_SHA256=INPUT_SHA;frozen.verify_stage=checked
    frozen.verify_root=lambda r,i,required=None:verify_root(root,inputs,job)
    return frozen


def environment_fingerprint(root,stage,job,native):
    _,inputs=verify_stage(root,stage)
    environment=read(native/'outputs'/(job['tag']+'_'+job['method'])/'benchmark.json')['environment']
    pinned=inputs['expected_environment_without_uuid']
    require(set(pinned)<=set(environment) and {k:environment[k] for k in pinned}==pinned,'Software/device/backend differs from the frozen cost protocol')
    identity=environment.get('device_uuid')
    require(isinstance(identity,str) and identity.strip() not in ('','None','unknown'),'Actual physical GPU UUID is required')
    current={k:environment[k] for k in pinned};current['device_uuid']=identity
    state=read(stage/'verification/driver.json')
    require('environment_fingerprint' in state or not any(r.get('status')=='COMPLETED' for r in state['jobs'].values()),'Completed cost rows have no registered environment fingerprint')
    if 'environment_fingerprint' in state:require(state['environment_fingerprint']==current,'Physical GPU/runtime changed during the16-row matrix; preserve this stage and do not mix environments')
    return current


def verify_completion(root,stage,job,native):
    frozen=benchmark_module(root,stage,job,native)
    proof_path=frozen.verify_completion(native,job['tag'],job['method']);proof=read(proof_path)
    require(proof['checkpoint_sha256']==job['checkpoint_sha256'],'Measured checkpoint differs from paired accuracy checkpoint')
    current=environment_fingerprint(root,stage,job,native)
    return dict(proof,id=job['id'],benchmark_proof_sha256=sha(proof_path),environment_fingerprint=current,accuracy_origin=job['origin'])


def child(root,stage,jobid,native):
    _,inputs=verify_stage(root,stage);job=next(j for j in inputs['jobs'] if j['id']==jobid);native=checked_attempt(stage,jobid,native)
    verify_root(root,inputs,job);frozen=benchmark_module(root,stage,job,native,create=True)
    # Source and scientific/timing AST of the original X6 child are unchanged.
    frozen.child(root,native,job['tag'],job['method']);verify_completion(root,stage,job,native)
    print('E1_COST_REFRESH_JOB_COMPLETED='+jobid,flush=True)


def run_child(command,logfile):
    process=None
    try:
        with logfile.open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',start_new_session=True,
                env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'))
            for line in process.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
            require(process.wait()==0,'Cost child failed; see '+str(logfile))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def collect(root,stage):
    manifest,inputs=verify_stage(root,stage);verify_root(root,inputs);validate_accuracy(root,inputs,stage/'scripts/e1_accuracy_validator.py')
    state=read(stage/'verification/driver.json');require(state['manifest_sha256']==sha(stage/'manifest.json'),'Cost driver manifest binding differs')
    require(set(state['jobs'])=={j['id'] for j in inputs['jobs']},'Cost driver inventory differs')
    require(not any(r.get('status')=='COMPLETED' for r in state['jobs'].values()) or 'environment_fingerprint' in state,'Completed cost rows require a registered run fingerprint')
    selected={'manifest.json':stage/'manifest.json','driver.json':stage/'verification/driver.json'}
    for relative in manifest['tools_sha256']:selected[relative]=safe(stage,relative)
    completed=[];pending=[]
    for job in inputs['jobs']:
        record=state['jobs'][job['id']]
        if record.get('status')=='COMPLETED':completed.append(verify_completion(root,stage,job,checked_attempt(stage,job['id'],record['native_stage'])))
        else:pending.append(job['id'])
        for attempt in record['attempts']:
            native=checked_attempt(stage,job['id'],attempt['native_stage'])
            for relative in ('manifest.json','verification/inference.log','verification/completed_'+job['tag']+'_'+job['method']+'.json','outputs/'+job['tag']+'_'+job['method']+'/benchmark.json'):
                path=safe(native,relative)
                if path.is_file():selected[path.relative_to(stage).as_posix()]=path
    summary={'version':VERSION,'status':'ALL16_COST_PROFILES_COMPLETE' if not pending else 'INCOMPLETE','completed_count':len(completed),'pending_jobs':pending,
             'matrix_rows':16,'all_cost_rows_measured_afresh':True,'old_cost_rows_merged':0,'partial_old_stage_measurements_adopted':False,
             'source_accuracy_stage':inputs['source_accuracy_stage'],'accuracy_jobs_preserved':18,'training_started':False,'accuracy_inference_rerun':False,'optimizer_steps':0,
             'new_results_promoted_to_manuscript':False,'environment_fingerprint':state.get('environment_fingerprint'),'historical_provenance':inputs['historical_provenance'],'jobs':completed}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:8];summary_path=stage/'reports'/('SUMMARY_'+stamp+'.json');write(summary_path,summary);selected['SUMMARY.json']=summary_path
    archive=stage/'reports'/('e1_cost_refresh_results_'+stamp+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for name,path in sorted(selected.items()):require(path.suffix not in ('.pt','.pth'),'Checkpoint export rejected');zipped.write(path,name)
    checksum=archive.with_suffix('.zip.sha256.txt');checksum.write_bytes((sha(archive)+'  '+archive.name+'\n').encode('ascii'))
    print('E1_COST_REFRESH_REPORT_STATUS='+summary['status']);print('E1_RESULTS_ZIP='+str(archive));print('E1_RESULTS_SHA256='+str(checksum))
    return 0


def run(root,stage=None):
    inputs,delivery=load_inputs();root=Path(root).resolve();require(os.name=='posix' and str(root)==inputs['root'],'Execution restricted to the original Linux server root')
    import fcntl
    with (root/'.kais_e1_cost_refresh.lock').open('a') as global_guard:
        fcntl.flock(global_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stage is None:stage=prepare(root,inputs,delivery)
        else:stage=Path(stage).resolve();verify_stage(root,stage);verify_root(root,inputs);validate_accuracy(root,inputs,stage/'scripts/e1_accuracy_validator.py')
        print('E1_COST_REFRESH_STAGE='+str(stage),flush=True)
        with (stage/'verification/driver.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=read(stage/'verification/driver.json');require(state['manifest_sha256']==sha(stage/'manifest.json') and set(state['jobs'])=={j['id'] for j in inputs['jobs']},'Driver identity/inventory differs')
            def stop(signum,frame):raise KeyboardInterrupt('Received signal '+str(signum))
            signal.signal(signal.SIGTERM,stop);result=1
            try:
                state['status']='RUNNING';state.pop('error',None);write(stage/'verification/driver.json',state)
                for job in inputs['jobs']:
                    record=state['jobs'][job['id']]
                    if record.get('status')=='COMPLETED':
                        verify_completion(root,stage,job,checked_attempt(stage,job['id'],record['native_stage']));print('E1_COST_REFRESH_KEEP_COMPLETED='+job['id'],flush=True);continue
                    number=len(record['attempts'])+1;native=stage/'attempts'/(job['id']+'_a'+str(number));native.mkdir()
                    for folder in ('outputs','verification'):(native/folder).mkdir()
                    attempt={'number':number,'native_stage':str(native),'status':'RUNNING'};record['attempts'].append(attempt);record.update(status='RUNNING',native_stage=str(native));write(stage/'verification/driver.json',state)
                    command=[sys.executable,'-u',str(stage/'scripts/run_e1_cost_refresh.py'),'--root',str(root),'--stage',str(stage),'--worker',job['id'],'--attempt',str(native)]
                    print('E1_COST_REFRESH_START='+job['id'],flush=True);run_child(command,native/'verification/inference.log')
                    proof=verify_completion(root,stage,job,native)
                    if 'environment_fingerprint' not in state:
                        state['environment_fingerprint']=proof['environment_fingerprint'];print('E1_COST_REFRESH_GPU_REGISTERED='+state['environment_fingerprint']['device_uuid'],flush=True)
                    require(state['environment_fingerprint']==proof['environment_fingerprint'],'Cost profile GPU/runtime differs from this run')
                    record['status']='COMPLETED';attempt['status']='COMPLETED';write(stage/'verification/driver.json',state)
                verify_root(root,inputs);validate_accuracy(root,inputs,stage/'scripts/e1_accuracy_validator.py');state['status']='ALL16_COST_PROFILES_COMPLETE';result=0
            except BaseException as exc:
                state['status']='STOPPED_INCOMPLETE';state['error']=type(exc).__name__+': '+str(exc);print('E1_COST_REFRESH_ERROR='+state['error'],flush=True)
            finally:
                write(stage/'verification/driver.json',state);print('E1_COST_REFRESH_DRIVER_STATUS='+state['status'],flush=True)
                try:collect(root,stage)
                except Exception as exc:print('E1_COLLECTION_ERROR='+type(exc).__name__+': '+str(exc),flush=True);result=1
            return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/time_uncertainty_rec'));parser.add_argument('--stage',type=Path)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--run',action='store_true');mode.add_argument('--preflight',action='store_true');mode.add_argument('--collect',action='store_true')
    parser.add_argument('--worker',help=argparse.SUPPRESS);parser.add_argument('--attempt',type=Path,help=argparse.SUPPRESS);args=parser.parse_args();inputs,delivery=load_inputs()
    if args.worker:
        require(args.stage is not None and args.attempt is not None and os.name=='posix' and str(args.root.resolve())==inputs['root'],'Internal cost worker requires recorded Linux root/stage/attempt')
        child(args.root.resolve(),args.stage.resolve(),args.worker,args.attempt);return 0
    if args.preflight:
        require(os.name=='posix' and str(args.root.resolve())==inputs['root'],'Preflight requires original Linux root')
        count=verify_root(args.root,inputs);validate_accuracy(args.root,inputs);print('E1_COST_PREFLIGHT_PASS files='+str(count)+' saved_accuracy=18; no Torch/GPU/training/inference');return 0
    if args.collect:
        require(args.stage is not None,'Collection requires --stage');return collect(args.root.resolve(),args.stage.resolve())
    if args.run:return run(args.root,args.stage)
    print('PLAN_ONLY: no original server inputs/checkpoints/Torch/GPU accessed.')
    print('16 fresh cost profiles:4datasets x4methods seed42,7new E1 accuracy references+9retained references; preserve18completed accuracy jobs; no training or accuracy reruns.')
    print('Unchanged X6 worker:canonical reproduction+512warmup+5 synchronized full-panel repeats. All16 replace historical costs; first actual GPU UUID registered, then exact16-key fingerprint required across run/resume/collection.')
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:print('ERROR: '+type(exc).__name__+': '+str(exc),file=sys.stderr);sys.exit(1)
