#!/usr/bin/env python3
"""E1 inference-only follow-up:14 affected models and4 paired Control ensembles."""
import argparse
import collections
import contextlib
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
VERSION = 'e1_inference_v1'
PREFIX = 'kais_e1_inference_v1_'
INPUT_SHA = '1a10ac2d3ddbe898269b8144404888b24f6b5e49902f394489dc01f25a38bb8f'
PERMS = (101,202,303,404,505)
GROUPS = (PERMS,(1101,1202,1303,1404,1505),(2101,2202,2303,2404,2505),(3101,3202,3303,3404,3505),(4101,4202,4303,4404,4505))
RUNTIME = ('run_e1_inference.py','e1_inference_inputs.json','frozen_x3_inference.py','x3_inference_adapter.py','x3_inference_protocol.py','x6_model_definitions.py','frozen_x6_inference.py','x3_core_helpers.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def safe(root, relative):
    root=Path(root).resolve();p=Path(relative)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in str(relative) and ':' not in str(relative),'Unsafe relative path')
    result=root/p
    require(result.resolve().is_relative_to(root),'Path leaves root')
    for part in (result,)+tuple(result.parents):
        if part==root.parent:break
        require(not part.is_symlink(),'Symlink rejected: '+str(part))
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex[:10]+'.tmp')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def load_inputs():
    require(sha(HERE/'e1_inference_inputs.json')==INPUT_SHA,'Reviewed inference inputs changed')
    inputs=read(HERE/'e1_inference_inputs.json');delivery=read(HERE/'delivery_manifest.json')
    require(set(delivery['runtime_sha256'])==set(RUNTIME),'Delivery runtime closure changed')
    for name,expected in delivery['runtime_sha256'].items():
        require(sha(safe(HERE,name))==expected,'Delivered runtime changed: '+name)
    require(inputs['version']==VERSION and len(inputs['jobs'])==18 and len({j['id'] for j in inputs['jobs']})==18,'Inference inventory differs')
    require(inputs['permutation_seeds']==list(PERMS) and inputs['ensemble_groups']==[list(g) for g in GROUPS],'Permutation protocol differs')
    require(sum(not j['is_ensemble'] for j in inputs['jobs'])==14 and sum(j['is_ensemble'] for j in inputs['jobs'])==4,'Expected14single+4paired ensembles')
    for name,expected in inputs['frozen_helper_sha256'].items():
        require(sha(HERE/name)==expected,'Frozen scientific helper changed: '+name)
    require(len(inputs['benchmarks'])==7 and len({j['id'] for j in all_jobs(inputs)})==25,'Expected25 unique tasks:18 evaluations+7 paired cost profiles')
    return inputs,delivery


def all_jobs(inputs):
    return inputs['jobs']+inputs['benchmarks']


def verify_root(root,inputs,job=None):
    files=inputs['root_files_sha256']
    if job is not None:
        plan=inputs['datasets'][job['tag']]
        paths={plan['original_path'],*plan['permutation_paths'].values()}
        files={p:d for p,d in files.items() if p.startswith(job['native_stage']+'/') or p in paths}
    for relative,expected in files.items():
        require(sha(safe(root,relative))==expected,'Frozen model/input/evidence changed: '+relative)
    return len(files)


def prepare(root,inputs,delivery):
    root=Path(root).resolve()
    require(str(root)==inputs['root'] and root.is_dir(),'Wrong original Linux project root')
    verify_root(root,inputs)
    require(not list(root.glob(PREFIX+'*')),'Existing E1 inference stage found; resume with --stage')
    stage=root/(PREFIX+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:10])
    stage.mkdir()
    for name in ('scripts','attempts','verification','reports'):(stage/name).mkdir()
    for name in RUNTIME+('delivery_manifest.json',):shutil.copyfile(HERE/name,stage/'scripts'/name)
    tools={'scripts/'+name:sha(stage/'scripts'/name) for name in RUNTIME+('delivery_manifest.json',)}
    value={'version':VERSION,'root':str(root),'stage':str(stage),'input_sha256':INPUT_SHA,'tools_sha256':tools,
           'protocol':inputs['protocol'],'jobs':[j['id'] for j in all_jobs(inputs)],'training_started':False}
    write(stage/'manifest.json',value)
    write(stage/'verification/driver.json',{'version':VERSION,'root':str(root),'stage':str(stage),'manifest_sha256':sha(stage/'manifest.json'),
        'status':'PREPARED','jobs':{j['id']:{'attempts':[]} for j in all_jobs(inputs)},'training_started':False})
    return stage


def verify_stage(root,stage):
    root=Path(root).resolve();stage=Path(stage).resolve()
    require(stage.parent==root and stage.name.startswith(PREFIX) and not stage.is_symlink(),'Expected direct isolated E1 inference stage')
    value=read(stage/'manifest.json')
    require(value['version']==VERSION and value['root']==str(root) and value['stage']==str(stage) and value['input_sha256']==INPUT_SHA,'Stage identity/input seal changed')
    require(set(value['tools_sha256'])=={'scripts/'+n for n in RUNTIME+('delivery_manifest.json',)},'Stage runtime closure differs')
    for relative,expected in value['tools_sha256'].items():require(sha(safe(stage,relative))==expected,'Stage runtime changed: '+relative)
    inputs=read(stage/'scripts/e1_inference_inputs.json')
    require(sha(stage/'scripts/e1_inference_inputs.json')==INPUT_SHA and value['jobs']==[j['id'] for j in all_jobs(inputs)] and value['protocol']==inputs['protocol'],'Stage inputs/jobs/protocol differ')
    return value,inputs


def checked_attempt(stage,jobid,value):
    path=Path(value)
    require(path.is_absolute() and path.parent==stage/'attempts' and path.name.startswith(jobid+'_a'),'Attempt leaves isolated inference stage')
    return safe(stage,path.relative_to(stage).as_posix())


def output_files(job):
    names={'config.json','item_id_map.csv','panel_labels.csv','predictions_original.csv','metrics.json','stability.json','input_view_hashes.csv','reproduction.json'}
    if job['is_ensemble']:
        names.update('predictions_group'+str(g)+'.csv' for g in range(1,6))
        names.update('predictions_view_seed'+str(s)+'.csv' for s in PERMS)
        names.update(('score_mean_probes.json','control_dependency.json'))
    else:names.update('predictions_perm_seed'+str(s)+'.csv' for s in PERMS)
    return names


def counts_expected(job,plan):
    multiplier=26 if job['is_ensemble'] else 6
    return {'samples':plan['expected_samples'],'affected':plan['expected_affected'],'items':plan['expected_items'],
            'logical_view_rows':plan['expected_samples']*multiplier,'actual_forward_rows':plan['expected_samples']*multiplier,
            'forward_batch_calls':math.ceil(plan['expected_samples']/256)*multiplier,'cache_hits':0}


def control_dependency(root,stage,job,inputs):
    source=next(j for j in inputs['jobs'] if j['id']==job['control_source_job_id'])
    require(source['method']=='control' and source['checkpoint_sha256']==job['checkpoint_sha256'] and source['seed']==job['seed'] and source['model_dir']==job['model_dir'],'Ensemble must reuse paired new Control')
    state=read(stage/'verification/driver.json');record=state['jobs'][source['id']]
    require(record.get('status')=='COMPLETED','Paired new Control inference must complete before Ensemble')
    native=checked_attempt(stage,source['id'],record['native_stage'])
    proof=verify_completion(root,stage,source,native)
    output=native/'outputs'/source['id']
    binding={'id':source['id'],'native_stage':str(native),'checkpoint_sha256':proof['checkpoint_sha256'],
             'completion_proof_sha256':sha(native/'verification/COMPLETED.json'),
             'five_permutation_sha256':{str(s):sha(output/('predictions_perm_seed'+str(s)+'.csv')) for s in PERMS}}
    return binding,output


def verify_completion(root,stage,job,native):
    manifest,inputs=verify_stage(root,stage)
    proof=read(native/'verification/COMPLETED.json');output=native/'outputs'/job['id']
    expected={'version':VERSION,'status':'COMPLETED','id':job['id'],'method':job['method'],'seed':job['seed'],'tag':job['tag'],
              'input_sha256':INPUT_SHA,'manifest_sha256':sha(stage/'manifest.json'),'checkpoint_sha256':job['checkpoint_sha256'],
              'training_started':False,'optimizer_steps':0,'canonical_reproduction':True,'originals_unchanged':True,
              'source_completion_proof':job['source_completion_proof'],'source_completion_sha256':job['source_completion_sha256']}
    require(all(proof.get(k)==v for k,v in expected.items()),'Inference completion identity/provenance differs')
    require(proof['counts']==counts_expected(job,inputs['datasets'][job['tag']]),'Inference counts differ')
    require(set(proof['metadata_sha256'])==output_files(job) and {p.name for p in output.iterdir()}==output_files(job),'Inference output closure differs')
    for name,expected_sha in proof['metadata_sha256'].items():require(sha(safe(output,name))==expected_sha,'Saved inference output changed: '+name)
    config=read(output/'config.json')
    require(config.get('job')==job and config.get('protocol')==inputs['protocol'] and config.get('input_sha256')==INPUT_SHA and config.get('manifest_sha256')==sha(stage/'manifest.json') and config.get('source_completion_sha256')==job['source_completion_sha256'] and config.get('batch_size')==256 and config.get('training_started') is False,'Inference config/job/source binding differs')
    reproduction=read(output/'reproduction.json')
    require(reproduction['status']=='PASS' and reproduction['checkpoint_before_sha256']==reproduction['checkpoint_after_sha256']==job['checkpoint_sha256'],'Checkpoint reproduction proof differs')
    canonical=reproduction['canonical']
    require(canonical['status']=='PASS' and canonical['reference_sha256']==inputs['root_files_sha256'][job['reference_predictions']] and canonical['samples']==inputs['datasets'][job['tag']]['expected_samples'],'Canonical reference proof differs')
    require(all(canonical[k]==0 for k in ('rank_mismatches','ordered_top10_mismatches','raw_top10_mismatches')),'Canonical predictions differ')
    require(set(reproduction['fixed_five_views'])==({str(s) for s in PERMS} if job['is_ensemble'] else set()),'Fixed-five proof scope differs')
    if job['is_ensemble']:
        binding,control_output=control_dependency(root,stage,job,inputs)
        require(read(output/'control_dependency.json')==binding,'Ensemble new Control dependency changed')
        for seed in PERMS:
            p=reproduction['fixed_five_views'][str(seed)]
            require(p['status']=='PASS' and p['reference_sha256']==binding['five_permutation_sha256'][str(seed)] and p['samples']==canonical['samples'] and all(p[k]==0 for k in ('rank_mismatches','ordered_top10_mismatches','raw_top10_mismatches')),'Ensemble member does not reproduce new Control')
    return proof


def child(root,stage,jobid,native):
    manifest,inputs=verify_stage(root,stage);job=next(j for j in inputs['jobs'] if j['id']==jobid)
    native=checked_attempt(stage,jobid,native);verify_root(root,inputs,job)
    core=module(stage/'scripts/frozen_x3_inference.py','e1_frozen_core')
    adapter=module(stage/'scripts/x3_inference_adapter.py','e1_frozen_adapter')
    protocol=module(stage/'scripts/x3_inference_protocol.py','e1_frozen_protocol')
    plan=inputs['datasets'][job['tag']]
    references=native/'references';references.mkdir()
    run={}
    for key,name in (('reference_map','item_id_map.csv'),('reference_predictions','canonical_saved.csv')):
        source=safe(root,job[key]);target=references/name
        shutil.copyfile(source,target);require(sha(target)==inputs['root_files_sha256'][job[key]],'Reference changed while copied')
        run[key]=target.relative_to(stage).as_posix()
    mapping,inverse,original,views,labels=core.load_panels(root,stage,plan,run)
    dependency,control_output=control_dependency(root,stage,job,inputs) if job['is_ensemble'] else (None,None)
    import numpy as np
    import torch
    require(torch.cuda.is_available(),'CUDA required; refusing accidental CPU inference')
    require(np.random.default_rng(0).bit_generator.__class__.__name__=='PCG64','Frozen protocol requires PCG64')
    torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=safe(root,job['model_dir']+'/'+job['checkpoint'])
    before=sha(checkpoint);require(before==job['checkpoint_sha256'],'Checkpoint differs from completed E1 training')
    value=torch.load(checkpoint,map_location='cpu',weights_only=True)
    expected={'model_name':'RecentBucketSetR1SASRec' if job['method']=='lbsp' else 'SASRecControl','dataset':plan['dataset'],
              'seed':job['seed'],'epoch':job['best_epoch'],'num_items':len(mapping),'max_len':50,'hidden_size':64,'num_heads':2,'num_blocks':2,'dropout':0.2}
    require(all(value.get(k)==v for k,v in expected.items()),'Checkpoint model/seed/best epoch/architecture differs')
    definitions=module(stage/'scripts/x6_model_definitions.py','e1_frozen_models')
    klass=definitions.RecentBucketSetR1SASRec if job['method']=='lbsp' else definitions.SASRecControl
    model=klass(num_items=len(mapping),max_len=50,hidden_size=64,num_heads=2,num_blocks=2,dropout=0.2).cuda()
    model.load_state_dict(value['model_state_dict'],strict=True);del value;model.eval()
    output=native/'outputs'/jobid;output.mkdir()
    counts={'samples':len(original),'affected':sum(row['affected'] for row in labels),'items':len(mapping),
            'logical_view_rows':0,'actual_forward_rows':0,'forward_batch_calls':0,'cache_hits':0}
    def score(rows):
        # Same frozen X6 adapter and feature preparation; BS256 partition retained.
        base='lbsp' if job['method']=='lbsp' else 'control'
        features=[adapter.prepare_history(base,row['_items'],row['_k']) for row in rows]
        seq=torch.tensor([f['seq'] for f in features],dtype=torch.long,device='cuda')
        tokens=torch.tensor([f['num_tokens'] for f in features],dtype=torch.long,device='cuda')
        groups=torch.tensor([f['groups'] for f in features],dtype=torch.long,device='cuda') if base=='lbsp' else None
        raw=adapter.score_batch(model,base,seq,tokens,groups=groups,num_tokens=tokens if base=='lbsp' else None)
        require(bool(torch.isfinite(raw).all()),'Nonfinite raw logits')
        counts['actual_forward_rows']+=len(rows);counts['forward_batch_calls']+=1
        return raw
    def evaluate(rows):
        result=[]
        for start in range(0,len(rows),256):
            batch=rows[start:start+256];result.extend(core.predict_rows(score(batch),batch,inverse,torch))
        return result
    core.write_csv(output/'item_id_map.csv',[{'raw_item_id':raw,'item_idx':idx} for raw,idx in mapping.items()])
    core.write_csv(output/'panel_labels.csv',labels)
    write(output/'config.json',{'version':VERSION,'job':job,'protocol':inputs['protocol'],'manifest_sha256':sha(stage/'manifest.json'),
                               'input_sha256':INPUT_SHA,'source_completion_sha256':job['source_completion_sha256'],'batch_size':256,'training_started':False})
    ledgers=core.view_ledger(original,0,'original');results={};fixed={};probes=[]
    with torch.no_grad():
        canonical=evaluate(original)
        reproduced=core.reproduce(canonical,safe(root,job['reference_predictions']))
        core.write_csv(output/'predictions_original.csv',canonical);results['original']=canonical
        if not job['is_ensemble']:
            for seed in PERMS:
                predictions=evaluate(views[seed]);results['perm_seed'+str(seed)]=predictions
                core.write_csv(output/('predictions_perm_seed'+str(seed)+'.csv'),predictions)
                ledgers.extend(core.view_ledger(views[seed],1,seed))
            compared=[results['perm_seed'+str(s)] for s in PERMS]
        else:
            # Reproduce sealed group1 joint item/event views; later groups use the same legacy generator.
            for group_index,group in enumerate(GROUPS,1):
                for seed in group:
                    generated=[]
                    for base in original:
                        key=protocol.legacy_seed(plan['dataset'],base['sample_id'],seed)
                        indices=[int(v) for v in np.random.default_rng(key).permutation(base['_k'])]
                        items,events,_=protocol.apply_last_bucket(base['_raw'],base['_events'],None,base['_k'],indices)
                        generated.append(dict(base,_raw=items,_events=events,_items=[mapping[item] for item in items]))
                    if group_index==1:
                        require(all(a['_raw']==b['_raw'] and a['_events']==b['_events'] for a,b in zip(generated,views[seed])),'Legacy group1 inputs differ from frozen views')
                    else:views[seed]=generated
                predictions=[];members={seed:[] for seed in group} if group_index==1 else {}
                for start in range(0,len(original),256):
                    size=len(original[start:start+256]);accumulator=None;member_probe=[];take=max(0,min(16-start,size))
                    for seed in group:
                        batch=views[seed][start:start+256];raw=score(batch)
                        if take:member_probe.append(raw[:take].clone())
                        if group_index==1:members[seed].extend(core.predict_rows(raw,batch,inverse,torch))
                        cast=raw.to(torch.float64);accumulator=cast if accumulator is None else accumulator+cast
                    mean=accumulator/5.0;batch_predictions=core.predict_rows(mean,original[start:start+256],inverse,torch)
                    predictions.extend(batch_predictions)
                    for i in range(take):
                        pred=batch_predictions[i];indices=[int(pred['target_idx'])]+[idx for idx in json.loads(pred['top10_item_idx']) if idx!=int(pred['target_idx'])]
                        probes.append({'sample_id':pred['sample_id'],'group':group_index,'member_seeds':list(group),'target_idx':pred['target_idx'],
                            'top10_item_idx':json.loads(pred['top10_item_idx']),'candidate_indices':indices,
                            'member_scores':[member[i,indices].cpu().tolist() for member in member_probe],'mean_scores':mean[i,indices].cpu().tolist()})
                results['group'+str(group_index)]=predictions;core.write_csv(output/('predictions_group'+str(group_index)+'.csv'),predictions)
                for seed in group:
                    ledgers.extend(core.view_ledger(views[seed],group_index,seed))
                    if group_index==1:
                        core.write_csv(output/('predictions_view_seed'+str(seed)+'.csv'),members[seed])
                        fixed[str(seed)]=core.reproduce(members[seed],control_output/('predictions_perm_seed'+str(seed)+'.csv'))
            compared=[results['group'+str(g)] for g in range(1,6)]
            write(output/'control_dependency.json',dependency)
            write(output/'score_mean_probes.json',{'scope':'first at most16 samples per group, not all candidate scores','accumulation_dtype':'torch.float64',
                'member_score_dtype':'torch.float32','reduction_order':[list(g) for g in GROUPS],'raw_scores_averaged_before_mask':True,
                'scores_averaged_not_ranks':True,'legacy_first_group_input_reproduced':True,'numpy_bit_generator':'PCG64','probes':probes})
    panel_metrics={name:core.metrics([int(row['rank']) for row in rows]) for name,rows in results.items()}
    report={'panels':panel_metrics}
    if job['is_ensemble']:report['ensemble_mean_accuracy']={key:math.fsum(panel_metrics['group'+str(g)][key] for g in range(1,6))/5 for key in panel_metrics['original']}
    write(output/'metrics.json',report);write(output/'stability.json',core.stability(compared,labels));core.write_csv(output/'input_view_hashes.csv',ledgers)
    after=sha(checkpoint);require(after==before,'Checkpoint changed during inference')
    write(output/'reproduction.json',{'status':'PASS','canonical':reproduced,'fixed_five_views':fixed,'checkpoint_before_sha256':before,'checkpoint_after_sha256':after})
    counts['logical_view_rows']=len(original)*(26 if job['is_ensemble'] else 6)
    require(counts==counts_expected(job,plan),'Inference accounting differs')
    verify_root(root,inputs,job);verify_stage(root,stage)
    metadata={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    require(set(metadata)==output_files(job),'Generated output closure differs')
    proof={'version':VERSION,'status':'COMPLETED','id':jobid,'method':job['method'],'seed':job['seed'],'tag':job['tag'],
           'input_sha256':INPUT_SHA,'manifest_sha256':sha(stage/'manifest.json'),'checkpoint_sha256':after,'metadata_sha256':metadata,
           'training_started':False,'optimizer_steps':0,'canonical_reproduction':True,'originals_unchanged':True,'counts':counts,
           'source_completion_proof':job['source_completion_proof'],'source_completion_sha256':job['source_completion_sha256'],
           'environment':{'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)}}
    write(native/'verification/COMPLETED.json',proof)
    verify_completion(root,stage,job,native)
    print('E1_INFERENCE_JOB_COMPLETED='+jobid,flush=True)


def accuracy_binding(root,stage,job,inputs):
    source=next(j for j in inputs['jobs'] if j['id']==job['accuracy_source_job_id'])
    require(source['checkpoint_sha256']==job['checkpoint_sha256'] and source['seed']==42 and source['method']==job['method'],'Cost profile must reuse paired new accuracy checkpoint')
    state=read(stage/'verification/driver.json');record=state['jobs'][source['id']]
    require(record.get('status')=='COMPLETED','New accuracy inference must complete before cost profiling')
    native=checked_attempt(stage,source['id'],record['native_stage']);verify_completion(root,stage,source,native)
    output=native/'outputs'/source['id'];name='predictions_group1.csv' if source['is_ensemble'] else 'predictions_original.csv'
    binding={'id':source['id'],'native_stage':str(native),'completion_proof_sha256':sha(native/'verification/COMPLETED.json'),
             'checkpoint_sha256':source['checkpoint_sha256'],'reference_filename':name,'reference_sha256':sha(output/name),'map_sha256':sha(output/'item_id_map.csv')}
    return binding,output


def benchmark_module(root,stage,job,native,create=False):
    outer,inputs=verify_stage(root,stage);native=checked_attempt(stage,job['id'],native)
    binding,source_output=accuracy_binding(root,stage,job,inputs)
    helpers=('x3_inference_adapter.py','x3_inference_protocol.py','x3_core_helpers.py','x6_model_definitions.py')
    plan=dict(inputs['datasets'][job['tag']]);plan['normalized_mapping_sha256']=inputs['cost_normalized_mapping_sha256'][job['tag']]
    run={k:job[k] for k in ('model_dir','checkpoint','checkpoint_sha256','best_epoch')}
    run.update(reference_map='references/item_id_map.csv',reference_predictions='references/predictions_new.csv',required_root_files=[])
    plan['runs']={job['method']:run}
    context={'version':'e1_inference_cost_v1','datasets':{job['tag']:plan},'protocol':inputs['cost_protocol']}
    if create:
        (native/'scripts').mkdir();(native/'references').mkdir()
        for name in helpers:shutil.copyfile(stage/'scripts'/name,native/'scripts'/name)
        shutil.copyfile(source_output/'item_id_map.csv',native/'references/item_id_map.csv')
        shutil.copyfile(source_output/binding['reference_filename'],native/'references/predictions_new.csv')
        value={'version':'e1_inference_cost_v1','root':str(root),'stage':str(native),'parent_stage':str(stage),
               'parent_manifest_sha256':sha(stage/'manifest.json'),'input_sha256':INPUT_SHA,'accuracy_binding':binding,
               'tools_sha256':{'scripts/'+n:sha(native/'scripts'/n) for n in helpers},
               'references_sha256':{'references/item_id_map.csv':binding['map_sha256'],'references/predictions_new.csv':binding['reference_sha256']},
               'frozen_worker_sha256':inputs['frozen_helper_sha256']['frozen_x6_inference.py'],'protocol':inputs['cost_protocol']}
        write(native/'manifest.json',value)
    def checked(r,n):
        require(Path(r)==root and Path(n)==native,'Cost worker root/stage differs')
        value=read(native/'manifest.json')
        require(value['version']=='e1_inference_cost_v1' and value['root']==str(root) and value['stage']==str(native) and value['parent_stage']==str(stage) and value['parent_manifest_sha256']==sha(stage/'manifest.json') and value['input_sha256']==INPUT_SHA,'Cost manifest identity changed')
        require(value['accuracy_binding']==binding and value['protocol']==inputs['cost_protocol'] and value['frozen_worker_sha256']==inputs['frozen_helper_sha256']['frozen_x6_inference.py'],'Cost source accuracy/protocol/worker differs')
        require(value['tools_sha256']=={'scripts/'+n:inputs['frozen_helper_sha256'][n] for n in helpers},'Cost helper seals differ')
        require(value['references_sha256']=={'references/item_id_map.csv':binding['map_sha256'],'references/predictions_new.csv':binding['reference_sha256']},'Cost new reference binding differs')
        for relative,expected in dict(value['tools_sha256'],**value['references_sha256']).items():require(sha(safe(native,relative))==expected,'Cost helper/reference changed: '+relative)
        verify_stage(root,stage)
        return value,context
    checked(root,native)
    frozen=module(stage/'scripts/frozen_x6_inference.py','e1_frozen_cost_worker')
    frozen.VERSION='e1_inference_cost_v1';frozen.INPUT_SHA256=INPUT_SHA
    frozen.verify_stage=checked
    frozen.verify_root=lambda r,i,required=None:verify_root(root,inputs,job)
    return frozen


def benchmark_child(root,stage,job,native):
    verify_root(root,verify_stage(root,stage)[1],job)
    frozen=benchmark_module(root,stage,job,native,create=True)
    # Original X6 child is executed without any source/AST/timing changes.
    frozen.child(root,native,job['tag'],job['method'])
    verify_benchmark(root,stage,job,native)
    print('E1_COST_JOB_COMPLETED='+job['id'],flush=True)


def verify_benchmark(root,stage,job,native):
    frozen=benchmark_module(root,stage,job,native)
    path=frozen.verify_completion(native,job['tag'],job['method'])
    proof=read(path)
    require(proof['checkpoint_sha256']==job['checkpoint_sha256'],'Cost checkpoint differs from paired new model')
    _,inputs=verify_stage(root,stage)
    environment=read(native/'outputs'/(job['tag']+'_'+job['method'])/'benchmark.json')['environment']
    prior=inputs['prior_cost_baseline']['environment_fingerprint']
    require({key:environment.get(key) for key in prior}==prior,'Cost environment differs from the retained nine historical rows; do not combine measurements')
    return dict(proof,id=job['id'],benchmark_proof_sha256=sha(path),accuracy_source_job_id=job['accuracy_source_job_id'])


def verify_task(root,stage,job,native):
    return verify_benchmark(root,stage,job,native) if job.get('is_benchmark') else verify_completion(root,stage,job,native)


def verify_cost_environment(state,job,native):
    if not job.get('is_benchmark'):
        return
    environment=read(native/'outputs'/(job['tag']+'_'+job['method'])/'benchmark.json')['environment']
    keys=('python','torch','numpy','cuda','device','device_count','device_index','device_capability','device_uuid','visible_devices','cuda_device_order','cublas_workspace_config','deterministic_algorithms','cudnn_deterministic','allow_tf32_matmul','allow_tf32_cudnn')
    current={key:environment.get(key) for key in keys}
    if 'cost_environment_fingerprint' in state:
        require(state['cost_environment_fingerprint']==current,'GPU/runtime environment differs across cost profiles')
    else:
        state['cost_environment_fingerprint']=current


def run_child(command,logfile):
    process=None
    try:
        with logfile.open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',start_new_session=True,
                                     env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'))
            for line in process.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
            require(process.wait()==0,'Inference child failed; see '+str(logfile))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def collect(root,stage):
    manifest,inputs=verify_stage(root,stage);state=read(stage/'verification/driver.json')
    require(state['manifest_sha256']==sha(stage/'manifest.json'),'Driver manifest binding differs')
    completed=[];pending=[];selected={'manifest.json':stage/'manifest.json','driver.json':stage/'verification/driver.json'}
    for relative in manifest['tools_sha256']:selected[relative]=safe(stage,relative)
    for job in all_jobs(inputs):
        record=state['jobs'][job['id']]
        if record.get('status')=='COMPLETED':
            native=checked_attempt(stage,job['id'],record['native_stage']);proof=verify_benchmark(root,stage,job,native) if job.get('is_benchmark') else verify_completion(root,stage,job,native)
            completed.append(proof)
        else:pending.append(job['id'])
        for attempt in record['attempts']:
            native=checked_attempt(stage,job['id'],attempt['native_stage'])
            proof_name='verification/completed_'+job['tag']+'_'+job['method']+'.json'
            relatives=('manifest.json',proof_name,'verification/inference.log') if job.get('is_benchmark') else ('verification/inference.log','verification/COMPLETED.json')
            for relative in relatives:
                p=safe(native,relative)
                if p.is_file():selected[p.relative_to(stage).as_posix()]=p
            names={'benchmark.json'} if job.get('is_benchmark') else output_files(job)
            directory=native/'outputs'/(job['tag']+'_'+job['method'] if job.get('is_benchmark') else job['id'])
            for name in names:
                p=directory/name
                if p.is_file():selected[p.relative_to(stage).as_posix()]=p
    summary={'version':VERSION,'status':'ALL25_INFERENCE_AND_COST_JOBS_COMPLETE' if not pending else 'INCOMPLETE','completed_count':len(completed),'pending_jobs':pending,
             'single_model_total':14,'paired_ensemble_total':4,'cost_profile_total':7,
             'evaluation_completed_count':sum(state['jobs'][j['id']].get('status')=='COMPLETED' for j in inputs['jobs']),
             'cost_completed_count':sum(state['jobs'][j['id']].get('status')=='COMPLETED' for j in inputs['benchmarks']),
             'prior_cost_baseline':inputs['prior_cost_baseline'],'training_started':False,'optimizer_steps':0,'new_results_promoted_to_manuscript':False,
             'retained_tisas_provenance':inputs['retained_tisas_provenance'],'jobs':completed}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:8]
    summary_path=stage/'reports'/('SUMMARY_'+stamp+'.json');write(summary_path,summary);selected['SUMMARY.json']=summary_path
    archive=stage/'reports'/('e1_inference_results_'+stamp+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for name,p in sorted(selected.items()):
            require(p.suffix not in ('.pt','.pth'),'Checkpoint export rejected');zipped.write(p,name)
    checksum=archive.with_suffix('.zip.sha256.txt');checksum.write_bytes((sha(archive)+'  '+archive.name+'\n').encode('ascii'))
    print('E1_INFERENCE_REPORT_STATUS='+summary['status']);print('E1_RESULTS_ZIP='+str(archive));print('E1_RESULTS_SHA256='+str(checksum))
    return 0


def run(root,stage=None):
    inputs,delivery=load_inputs();root=Path(root).resolve()
    require(os.name=='posix' and str(root)==inputs['root'],'Execution restricted to the recorded Linux server root')
    import fcntl
    with (root/'.kais_e1_inference.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stage is None:stage=prepare(root,inputs,delivery)
        else:stage=Path(stage).resolve();verify_stage(root,stage);verify_root(root,inputs)
        print('E1_INFERENCE_STAGE='+str(stage),flush=True)
        with (stage/'verification/driver.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=read(stage/'verification/driver.json');require(state['manifest_sha256']==sha(stage/'manifest.json'),'Driver manifest differs')
            require(set(state['jobs'])=={j['id'] for j in all_jobs(inputs)},'Driver job inventory differs')
            def stop(signum,frame):raise KeyboardInterrupt('Received signal '+str(signum))
            signal.signal(signal.SIGTERM,stop);result=1
            try:
                state['status']='RUNNING';state.pop('error',None);write(stage/'verification/driver.json',state)
                for job in all_jobs(inputs):
                    record=state['jobs'][job['id']]
                    if record.get('status')=='COMPLETED':
                        native=checked_attempt(stage,job['id'],record['native_stage']);verify_task(root,stage,job,native);verify_cost_environment(state,job,native)
                        print('E1_INFERENCE_KEEP_COMPLETED='+job['id'],flush=True);continue
                    if job.get('is_benchmark'):accuracy_binding(root,stage,job,inputs)
                    elif job['is_ensemble']:control_dependency(root,stage,job,inputs)
                    number=len(record['attempts'])+1;native=stage/'attempts'/(job['id']+'_a'+str(number));native.mkdir()
                    for folder in ('outputs','verification'):(native/folder).mkdir()
                    attempt={'number':number,'native_stage':str(native),'status':'RUNNING'};record['attempts'].append(attempt);record.update(status='RUNNING',native_stage=str(native))
                    write(stage/'verification/driver.json',state)
                    command=[sys.executable,'-u',str(stage/'scripts/run_e1_inference.py'),'--root',str(root),'--stage',str(stage),'--worker',job['id'],'--attempt',str(native)]
                    print('E1_INFERENCE_START='+job['id'],flush=True);run_child(command,native/'verification/inference.log')
                    verify_task(root,stage,job,native);verify_cost_environment(state,job,native);record['status']='COMPLETED';attempt['status']='COMPLETED';write(stage/'verification/driver.json',state)
                verify_root(root,inputs);state['status']='ALL25_INFERENCE_AND_COST_JOBS_COMPLETE';result=0
            except BaseException as exc:
                state['status']='STOPPED_INCOMPLETE';state['error']=type(exc).__name__+': '+str(exc);print('E1_INFERENCE_ERROR='+state['error'],flush=True)
            finally:
                write(stage/'verification/driver.json',state);print('E1_INFERENCE_DRIVER_STATUS='+state['status'],flush=True)
                try:collect(root,stage)
                except Exception as exc:print('E1_COLLECTION_ERROR='+type(exc).__name__+': '+str(exc),flush=True);result=1
            return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/time_uncertainty_rec'));parser.add_argument('--stage',type=Path)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--run',action='store_true');mode.add_argument('--preflight',action='store_true');mode.add_argument('--collect',action='store_true')
    parser.add_argument('--worker',help=argparse.SUPPRESS);parser.add_argument('--attempt',type=Path,help=argparse.SUPPRESS);args=parser.parse_args()
    inputs,delivery=load_inputs()
    if args.worker:
        require(args.stage is not None and args.attempt is not None and os.name=='posix' and str(args.root.resolve())==inputs['root'],'Internal worker requires recorded Linux root/stage/attempt')
        job=next(j for j in all_jobs(inputs) if j['id']==args.worker)
        if job.get('is_benchmark'):benchmark_child(args.root.resolve(),args.stage.resolve(),job,args.attempt)
        else:child(args.root.resolve(),args.stage.resolve(),args.worker,args.attempt)
        return 0
    if args.preflight:
        require(os.name=='posix' and str(args.root.resolve())==inputs['root'],'Preflight requires recorded Linux root')
        print('E1_PREFLIGHT_FROZEN_FILES='+str(verify_root(args.root,inputs))+' no Torch/GPU/training');return 0
    if args.collect:
        require(args.stage is not None,'Collection requires --stage');return collect(args.root.resolve(),args.stage.resolve())
    if args.run:return run(args.root,args.stage)
    print('PLAN_ONLY: no original server inputs/checkpoints/Torch/GPU accessed.')
    print('14 saved models:canonical+5permutations;4 pairedControlEnsembles:canonicalControl+5groups x5raw-score members;5 TiSAS panels retained. No training.')
    print('Fixed BS256, frozen observed-history panels/candidate mask/PCG64, exact NEW canonical reproduction and exact paired NEW Control group1 member reproduction required.')
    print('Then7 paired seed42 inference-cost profiles reuse the unchanged frozen X6 timing worker:512warmup,5 synchronized full-panel repeats;25 tasks total.')
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:print('ERROR: '+type(exc).__name__+': '+str(exc),file=sys.stderr);sys.exit(1)
