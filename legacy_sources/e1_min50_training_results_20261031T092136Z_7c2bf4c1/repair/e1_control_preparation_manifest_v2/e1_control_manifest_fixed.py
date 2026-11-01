#!/usr/bin/env python3
"""Resume the pinned E1 batch after creating its missing Control preparation records."""
import argparse
import ast
import contextlib
import copy
import datetime
import io
import json
import os
from pathlib import Path
import shutil
import sys
import types
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
PATCH_ID = 'e1_control_preparation_manifest_v2'
V1_SCRIPT_SHA = 'c49c85d2af206c9ba697cc4dbfb43232b779cb747e2cd711f79fef0b3f76b60e'
V1_MANIFEST_SHA = '0e8a883a49a9209c97daef5bcb73d34081a60ed02957ef49fbc28a7e5e377152'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def manifest():
    value = read(HERE / 'control_fix_manifest.json')
    require(value['patch_id'] == PATCH_ID, 'Wrong Control repair manifest')
    require(value['entrypoint_sha256'] == sha(__file__), 'Control repair entrypoint changed')
    require(value['included_v1_sha256'] == {'e1_resume_fixed.py': V1_SCRIPT_SHA, 'repair_manifest.json': V1_MANIFEST_SHA}, 'Wrong composed V1 repair')
    for name, expected in value['included_v1_sha256'].items():
        require(sha(HERE / name) == expected, 'Composed V1 repair changed: ' + name)
    return value


def load_v1(value):
    module = types.ModuleType('_e1_sealed_v1')
    module.__file__ = str(HERE / 'e1_resume_fixed.py')
    exec(compile((HERE / 'e1_resume_fixed.py').read_text(encoding='utf-8'), module.__file__, 'exec'), module.__dict__)
    prior = module.overlay_manifest()
    require(prior['root'] == value['root'] and prior['stage'] == value['stage'], 'V1/V2 batch mismatch')
    return module, prior


def literal(source, name):
    found = [n.value for n in ast.parse(source).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    require(len(found) == 1, 'Expected one frozen Control assignment: ' + name)
    return ast.literal_eval(found[0])


def control_record(common, root, native, job, policy, value):
    """Describe actual prepared files and verify frozen inputs without importing a trainer."""
    require(job['family'] == 'control', 'Control manifest requested for another family')
    scripts = policy['families']['control']['scripts']
    generated = { 'scripts/' + name: common.sha(common.safe(native, 'scripts/' + name)) for name in sorted(scripts) }
    for name in policy['families']['control']['patch_scripts']:
        source = common.safe(native, 'scripts/' + name).read_text(encoding='utf-8')
        require(literal(source, '_INPUT_SHA256') == value['control_input_sha256'], 'Control source frozen inputs differ')
        require(literal(source, '_FROZEN_MODEL_SHA256') == value['control_model_sha256'], 'Control source frozen model differs')
    for relative, expected in value['control_input_sha256'].items():
        require(common.sha(common.safe(root, relative)) == expected, 'Control frozen input changed: ' + relative)
    require(generated['scripts/model_definition.py'] == value['control_model_sha256'], 'Control model source changed')
    return {'version': 'kais_e1_control_preparation_v2', 'status': 'PREPARED_NOT_TRAINING_COMPLETION',
            'patch_id': PATCH_ID, 'root': str(root), 'stage': str(native), 'job_id': job['id'],
            'dataset': job['dataset'], 'seed': job['seed'], 'generated_sha256': generated,
            'input_sha256': value['control_input_sha256'], 'frozen_model_sha256': value['control_model_sha256'],
            'original_policy_sha256': value['original_policy_sha256'],
            'source_vendor_sha256': {name: policy['bundle_sha256'][vendor] for name, vendor in scripts.items()},
            'training_rule': {'minimum_epochs': 50, 'patience': 15, 'max_epochs': 200,
                              'selection': 'validation-best across all epochs; no reset at50'},
            'sequence_sort_keys': ['user_id', 'time_bucket', 'raw_line'], 'sort_kind': 'stable',
            'scientific_trainers_or_original_bundle_changed_by_this_repair': False}


def verify_control(common, root, native, job, policy, value):
    expected = control_record(common, root, native, job, policy, value)
    path = common.safe(native, 'manifests/orderfix_manifest.json')
    require(read(path) == expected, 'Control preparation manifest differs')
    config_path = common.output_path(native, job) / 'config.json'
    if config_path.exists():
        config = read(config_path)
        require(config['orderfix']['preparation_manifest_sha256'] == common.sha(path), 'Control config/preparation manifest hash differs')
    return path


def install_common(common, value):
    original_make, original_validate = common.make_attempt, common.validate_output
    def make(root, outer, job, number, policy):
        native = original_make(root, outer, job, number, policy)
        if job['family'] == 'control':
            record = control_record(common, root, native, job, policy, value)
            common.write_json(native / 'manifests/orderfix_manifest.json', record)
            verify_control(common, root, native, job, policy, value)
        return native
    def validate(native, job):
        if job['family'] == 'control':
            verify_control(common, Path(value['root']), native, job, common.lock(), value)
        return original_validate(native, job)
    common.make_attempt, common.validate_output = make, validate


def prepare(common, root, stage, state, policy, value, v1, prior_manifest):
    """Original driver calls this only while holding both its locks."""
    require(str(root) == value['root'] and str(stage) == value['stage'], 'Control repair targets a different batch')
    require(state['root'] == str(root) and state['stage'] == str(stage) and state['policy_sha256'] == value['original_policy_sha256'], 'Batch identity/policy differs')
    require(read(stage / 'policy.json') == policy, 'Stage policy contents changed')
    prior = [r for r in state.get('repair_history', []) if r.get('patch_id') == PATCH_ID]
    require(len(prior) <= 1, 'Duplicate Control repair registration')
    if not prior:
        require(common.sha(stage / 'driver.json') == value['driver_preimage_sha256'], 'Driver differs from the returned10-complete failure')
        require(state['status'] == 'STOPPED_INCOMPLETE' and state.get('error') == value['failure_error'], 'Unexpected Control failure reason')
        require({k for k,r in state['jobs'].items() if r.get('status') == 'COMPLETED'} == set(value['completed_job_ids']), 'Returned completed10 inventory differs')
    # Verify every returned completed result and its checkpoint before any state/provenance writes.
    for job in policy['jobs']:
        if job['id'] not in value['completed_job_ids']:
            continue
        saved = state['jobs'][job['id']]
        require(saved.get('status') == 'COMPLETED', 'A retained completed model lost its completed status')
        native = common.checked_native(root, stage, saved['native_stage'])
        evidence = value['retained_evidence'][job['id']]
        require(str(native) == evidence['native_stage'], 'Retained native stage changed')
        for relative, expected in evidence['files_sha256'].items():
            require(common.sha(common.safe(native, relative)) == expected, 'Retained completed evidence changed: ' + job['id'] + '/' + relative)
        proof_path = native / 'verification/E1_COMPLETED.json'
        require(saved['completion_proof'] == str(proof_path), 'Retained completion proof path differs')
        require(common.validate_output(native, job) == read(proof_path), 'Retained completed model failed validation')
    # Dry-run the existing V1 history checks; its existing-history branch has only one write.
    require(len([r for r in state.get('repair_history', []) if r.get('patch_id') == v1.PATCH_ID]) == 1, 'Existing V1 repair registration missing')
    proxy = types.SimpleNamespace(**common.__dict__)
    proxy.write_json = lambda path, data: None
    v1.prepare(proxy, root, stage, copy.deepcopy(state), policy, prior_manifest)
    location = common.safe(stage, 'repair/' + PATCH_ID)
    names = ('e1_control_manifest_fixed.py', 'control_fix_manifest.json', 'e1_resume_fixed.py', 'repair_manifest.json')
    entry = {'patch_id': PATCH_ID, 'path': str(location), 'driver_preimage_sha256': value['driver_preimage_sha256'],
             'runtime_sha256': {name: sha(HERE / name) for name in names},
             'original_policy_sha256': value['original_policy_sha256'], 'retained_completed_job_ids': value['completed_job_ids'],
             'scientific_trainers_or_original_bundle_changed': False}
    if prior:
        stored = read(location / 'repair_registration.json')
        require(stored == prior[0] and all(stored[k] == entry[k] for k in entry), 'Control repair registration differs')
    elif location.exists():
        # Recover a completed atomic provenance rename interrupted before the driver commit.
        stored = read(location / 'repair_registration.json')
        require(all(stored[k] == entry[k] for k in entry), 'Unregistered Control repair provenance differs')
    else:
        stored = dict(entry, prior_status=state['status'], prior_error=state.get('error'),
                      registered_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        temporary = location.with_name(location.name + '.prepare_' + uuid.uuid4().hex[:10])
        temporary.mkdir()
        try:
            for name in names:
                (temporary / name).write_bytes((HERE / name).read_bytes())
            (temporary / 'driver_before_repair.json').write_bytes((stage / 'driver.json').read_bytes())
            common.write_json(temporary / 'repair_registration.json', stored)
            temporary.rename(location)
        except BaseException:
            # Temporary preparation evidence is retained for inspection; accepted results are untouched.
            raise
    require(common.sha(location / 'driver_before_repair.json') == value['driver_preimage_sha256'], 'Control driver preimage changed')
    for name in names:
        require(common.sha(location / name) == sha(HERE / name), 'Retained Control repair runtime changed')
    if not prior:
        state.setdefault('repair_history', []).append(stored)
    v1.prepare(common, root, stage, state, policy, prior_manifest)
    print('E1_CONTROL_MANIFEST_REPAIR_READY retained10; original policy and trainers unchanged', flush=True)


def driver_source(toolkit, v1):
    return v1.driver_source(toolkit)


def collect(common, root, stage, policy, value, v1, prior_manifest):
    state = read(stage / 'driver.json')
    require(str(root) == value['root'] and str(stage) == value['stage'], 'Collector targets another batch')
    location = common.safe(stage, 'repair/' + PATCH_ID)
    registration = read(location / 'repair_registration.json')
    require(registration in state.get('repair_history', []), 'Control repair not registered')
    selected = {}
    for name, expected in registration['runtime_sha256'].items():
        require(name in ('e1_control_manifest_fixed.py', 'control_fix_manifest.json', 'e1_resume_fixed.py', 'repair_manifest.json'), 'Unexpected Control repair export name')
        require(common.sha(location / name) == expected == sha(HERE / name), 'Control collector provenance changed')
        selected['repair/' + PATCH_ID + '/' + name] = location / name
    require(set(registration['runtime_sha256']) == {'e1_control_manifest_fixed.py', 'control_fix_manifest.json', 'e1_resume_fixed.py', 'repair_manifest.json'}, 'Control repair export closure differs')
    require(common.sha(location / 'driver_before_repair.json') == value['driver_preimage_sha256'], 'Control collector preimage changed')
    for name in ('driver_before_repair.json', 'repair_registration.json'):
        selected['repair/' + PATCH_ID + '/' + name] = location / name
    for job in policy['jobs']:
        if job['family'] != 'control':
            continue
        for attempt in state['jobs'][job['id']]['attempts']:
            native = common.checked_native(root, stage, attempt['native_stage'])
            path = native / 'manifests/orderfix_manifest.json'
            if path.exists():
                verify_control(common, root, native, job, policy, value)
                selected['attempts/' + job['id'] + '_a' + str(attempt['number']) + '/manifests/orderfix_manifest.json'] = path
            else:
                require(job['id'] == value['failed_job_id'] and attempt['number'] == 1, 'New Control attempt is missing its preparation manifest')
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = v1.collect(common, root, stage, policy, prior_manifest)
    require(code == 0, 'Composed V1 collector failed')
    lines = output.getvalue().splitlines()
    archives = [Path(line.split('=',1)[1]) for line in lines if line.startswith('E1_RESULTS_ZIP=')]
    require(len(archives) == 1, 'Expected one composed result ZIP')
    archive = archives[0]
    require(archive.parent == stage / 'reports' and archive.name.startswith('e1_min50_training_results_'), 'Unexpected result archive path')
    with zipfile.ZipFile(archive, 'a', compression=zipfile.ZIP_DEFLATED) as zipped:
        require(not set(selected).intersection(zipped.namelist()), 'Duplicate Control provenance ZIP members')
        for name, path in sorted(selected.items()):
            zipped.write(path, name)
    checksum = archive.with_suffix('.zip.sha256.txt')
    checksum.write_bytes((common.sha(archive) + '  ' + archive.name + '\n').encode('ascii'))
    for line in lines:
        if not line.startswith(('E1_RESULTS_ZIP=', 'E1_RESULTS_SHA256=')):
            print(line)
    print('E1_CONTROL_MANIFEST_PROVENANCE_BUNDLED=' + PATCH_ID)
    print('E1_RESULTS_ZIP=' + str(archive))
    print('E1_RESULTS_SHA256=' + str(checksum))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/time_uncertainty_rec'))
    parser.add_argument('--stage', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    value = manifest()
    if not args.run and not args.collect:
        print('PLAN_ONLY: no original project inputs, checkpoints, Torch or GPU accessed.')
        print('Returned batch:10 completed,9 pending. Keep all completed results; retry Sports Control in a new a2; generate preparation manifests for all4 Control jobs.')
        print('Includes unchanged V1 epoch-schema repair. No changes to original80 dependencies, scientific trainers, min50/patience15/cap200 or validation-best selection.')
        return 0
    root = args.root.resolve()
    stage = args.stage.resolve() if args.stage else Path(value['stage'])
    require(os.name == 'posix' and str(root) == value['root'] and str(stage) == value['stage'] and not stage.is_symlink(), 'Execution restricted to the recorded Linux batch')
    v1, prior_manifest = load_v1(value)
    toolkit = root / value['original_toolkit_directory']
    common, policy = v1.load_common(toolkit, prior_manifest)
    install_common(common, value)
    if args.collect:
        return collect(common, root, stage, policy, value, v1, prior_manifest)
    module = types.ModuleType('_e1_control_manifest_driver')
    module.__file__ = str(toolkit / 'run_e1_min50.py')
    module.__dict__.update(
        _e1_repair_prepare=lambda r,s,state,p: prepare(common,r,s,state,p,value,v1,prior_manifest),
        _e1_repair_adopt=lambda r,s,state,p,j: v1.adopt(common,r,s,state,p,j,prior_manifest),
        _e1_repair_collect_command=lambda r,s: [sys.executable,str(Path(__file__).resolve()),'--root',str(r),'--stage',str(s),'--collect'])
    exec(compile(driver_source(toolkit,v1), module.__file__, 'exec'), module.__dict__)
    previous = sys.argv[:]
    try:
        sys.argv = [module.__file__,'--root',str(root),'--stage',str(stage),'--run']
        return module.main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('ERROR: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(1)
