#!/usr/bin/env python3
"""Resume one pinned E1 batch after correcting saved-config epoch field aliases."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import zipfile

HERE = Path(__file__).resolve().parent
PATCH_ID = 'e1_epoch_schema_resume_v1'
POLICY_SHA = 'a17cc41f3e38d83abe74e6152106d547d3582147d6673a2376f2c6b72b1abcb4'
BUDGET_OLD = "    require(hyper.get('patience') == 15 and hyper.get('max_epochs') == 200, 'Output stopping budget differs')"
BUDGET_NEW = "    _e1_fixed_budget(config)"
DRIVER_PREPARE_OLD = "        result = 1\n        try:"
DRIVER_PREPARE_NEW = "        _e1_repair_prepare(root, stage, state, policy)\n        result = 1\n        try:"
DRIVER_ADOPT_OLD = "                record = state['jobs'][job['id']]\n                if record.get('status') == 'COMPLETED':"
DRIVER_ADOPT_NEW = "                record = state['jobs'][job['id']]\n                _e1_repair_adopt(root, stage, state, policy, job)\n                if record.get('status') == 'COMPLETED':"
DRIVER_COLLECT_OLD = "            collected = subprocess.run([sys.executable, str(HERE / 'collect_e1_min50.py'), '--stage', str(stage)])"
DRIVER_COLLECT_NEW = "            collected = subprocess.run(_e1_repair_collect_command(root, stage))"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def replace_once(text, old, new):
    require(text.count(old) == 1, 'Expected one reviewed patch site: ' + old[:90])
    return text.replace(old, new, 1)


def budget(config):
    """Accept the two archived cap names, requiring all supplied values to agree."""
    require(isinstance(config, dict), 'Config must be an object')
    hyper = config.get('hyperparameters', config)
    require(isinstance(hyper, dict), 'Hyperparameters must be an object')
    caps = [hyper[k] for k in ('max_epochs', 'epochs') if k in hyper]
    require(bool(caps), 'Missing maximum training epochs')
    require(all(type(v) is int and v == 200 for v in caps), 'Conflicting or wrong maximum epoch budget')
    require(type(hyper.get('patience')) is int and hyper['patience'] == 15, 'Wrong patience budget')
    # When a nested schema repeats these fields outside, no contradiction is allowed.
    if hyper is not config:
        require(all(type(config[k]) is int and config[k] == 200 for k in ('max_epochs', 'epochs') if k in config),
                'Outer maximum epoch budget conflicts')
        if 'patience' in config:
            require(type(config['patience']) is int and config['patience'] == 15, 'Outer patience conflicts')


def overlay_manifest():
    manifest = read(HERE / 'repair_manifest.json')
    require(manifest['patch_id'] == PATCH_ID and manifest['original_policy_sha256'] == POLICY_SHA, 'Unexpected repair manifest')
    require(manifest['entrypoint_sha256'] == sha(Path(__file__)), 'Repair entrypoint changed')
    return manifest


def load_common(toolkit, manifest):
    """Verify the original sealed bundle, then change one validator assertion in memory."""
    toolkit = Path(toolkit).resolve()
    require(not toolkit.is_symlink(), 'Toolkit symlink rejected')
    for name, expected in manifest['original_runtime_sha256'].items():
        require(sha(toolkit / name) == expected, 'Original runtime changed: ' + name)
    require(sha(toolkit / 'e1_policy.json') == POLICY_SHA, 'Original policy changed')
    source = (toolkit / 'e1_common.py').read_text(encoding='utf-8')
    patched = replace_once(source, BUDGET_OLD, BUDGET_NEW)
    common = types.ModuleType('e1_common')
    common.__file__ = str(toolkit / 'e1_common.py')
    common.__dict__['_e1_fixed_budget'] = budget
    sys.modules['e1_common'] = common
    exec(compile(patched, common.__file__, 'exec'), common.__dict__)
    policy = common.lock()  # Verifies every one of the 80 original dependencies.
    require(policy['minimum_epochs'] == 50 and policy['patience'] == 15 and policy['max_epochs'] == 200, 'Scientific policy changed')
    return common, policy


def expected_native(common, root, stage, state, manifest):
    record = state['jobs'][manifest['adopt_job_id']]
    require(len(record['attempts']) == 1, 'Known completed training must have exactly one original attempt')
    require(record['attempts'][0]['number'] == 1, 'Unexpected adoption attempt number')
    native = common.checked_native(root, stage, record['native_stage'])
    require(native == stage / 'attempts' / manifest['adopt_attempt_name'], 'Wrong adoption native directory')
    require(Path(record['attempts'][0]['native_stage']) == native, 'Attempt/native record disagreement')
    return native, record


def verify_candidate(common, root, stage, state, policy, manifest):
    job = next(j for j in policy['jobs'] if j['id'] == manifest['adopt_job_id'])
    native, record = expected_native(common, root, stage, state, manifest)
    # The exact saved run returned by the user, including original trainer and completion metadata.
    for relative, expected in manifest['adopt_native_sha256'].items():
        require(sha(common.safe(native, relative)) == expected, 'Adoption evidence changed: ' + relative)
    completion = read(native / 'verification/completed_ml1m_seed42.json')
    output = common.output_path(native, job)
    require(completion['status'] == 'COMPLETED' and completion['seed'] == 42
            and completion['tag'] == 'ml1m' and completion['mode'] == 'train', 'X3 completion identity differs')
    require(completion['output_dir'] == str(output) and completion['checkpoint'] == 'matched_control_sasrec_best.pt', 'X3 completion output/checkpoint differs')
    require(completion['trainer_sha256'] == sha(native / 'scripts/train_ml1m_shuffle.py'), 'Trained scientific source changed')
    require(completion.get('originals_unchanged') is True and completion.get('all_original_hashes_verified') is True
            and completion.get('canonical_evaluation_mask_verified') is True, 'X3 integrity/mask proof missing')
    required = {'config.json', 'metrics.json', 'training_log.csv', 'test_per_sample_predictions.csv', 'item_id_map.csv'}
    require(set(completion['output_metadata_sha256']) == required, 'Unexpected X3 output-metadata closure')
    for name, expected in completion['output_metadata_sha256'].items():
        require(sha(common.safe(output, name)) == expected, 'X3 saved metadata changed: ' + name)
    proof = common.validate_output(native, job)  # Includes streaming checkpoint SHA on the server.
    require(proof['best_epoch'] == 31 and proof['stop_epoch'] == 50, 'Known saved run epoch selection differs')
    require(proof['old_prefix_max_abs_differences'] == {'val_ndcg10': 0.0, 'train_loss': 0.0}, 'Shuffle original prefix differs')
    return native, record, proof


def repair_record(common, stage, manifest):
    return {
        'patch_id': PATCH_ID,
        'overlay_manifest_sha256': sha(HERE / 'repair_manifest.json'),
        'entrypoint_sha256': sha(Path(__file__)),
        'original_policy_sha256': POLICY_SHA,
        'original_runtime_sha256': manifest['original_runtime_sha256'],
        'original_driver_preimage_sha256': manifest['original_driver_preimage_sha256'],
        'reason': 'Saved Shuffle config uses epochs=200; original validator read only max_epochs.',
        'validator_delta': 'Recognize epochs/max_epochs, require each supplied cap to equal200 and patience15; no config/source edits.',
        'adopt_job_id': manifest['adopt_job_id'],
        'scientific_trainers_or_original_bundle_changed': False,
        'path': str(stage / 'repair' / PATCH_ID),
    }


def prepare(common, root, stage, state, policy, manifest):
    """Called only after the original global and per-stage locks have been acquired."""
    require(str(root) == manifest['root'] and str(stage) == manifest['stage'], 'Repair targets a different batch')
    require(state['root'] == str(root) and state['stage'] == str(stage) and state['policy_sha256'] == POLICY_SHA, 'Batch identity/policy differs')
    require(read(stage / 'policy.json') == policy, 'Stage policy logical contents differ')
    prior = [v for v in state.get('repair_history', []) if v.get('patch_id') == PATCH_ID]
    location = common.safe(stage, 'repair/' + PATCH_ID)
    entry = repair_record(common, stage, manifest)
    if not prior:
        require(sha(stage / 'driver.json') == manifest['original_driver_preimage_sha256'], 'Initial driver differs from returned stopped batch')
        require(state['status'] == 'STOPPED_INCOMPLETE' and state.get('error') == 'ValueError: Output stopping budget differs', 'Unexpected original stop reason')
        require({k for k, v in state['jobs'].items() if v.get('status') == 'COMPLETED'} == set(manifest['completed_job_ids']), 'Unexpected original completed jobs')
        # Verify all three validated models before changing any state.
        for job in policy['jobs']:
            if job['id'] in manifest['completed_job_ids']:
                record = state['jobs'][job['id']]
                native = common.checked_native(root, stage, record['native_stage'])
                expected = native / 'verification/E1_COMPLETED.json'
                require(Path(record['completion_proof']) == expected, 'Completed proof path differs')
                require(common.validate_output(native, job) == read(expected), 'Previously completed result changed')
        verify_candidate(common, root, stage, state, policy, manifest)
        require(not location.exists(), 'Repair provenance folder already exists without registered history')
        location.mkdir(parents=True)
        for name in ('e1_resume_fixed.py', 'repair_manifest.json'):
            with (location / name).open('xb') as stream:
                stream.write((HERE / name).read_bytes())
        (location / 'original_runtime').mkdir()
        for name, expected in manifest['original_runtime_sha256'].items():
            source = common.HERE / name
            require(sha(source) == expected, 'Original runtime changed during provenance copy')
            with (location / 'original_runtime' / name).open('xb') as stream:
                stream.write(source.read_bytes())
        with (location / 'driver_before_repair.json').open('xb') as stream:
            stream.write((stage / 'driver.json').read_bytes())
        entry['prior_status'] = state['status']
        entry['prior_error'] = state.get('error')
        entry['registered_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        common.write_json(location / 'repair_registration.json', entry)
        state.setdefault('repair_history', []).append(entry)
    else:
        require(len(prior) == 1, 'Duplicate repair history')
        stored = read(location / 'repair_registration.json')
        require(stored == prior[0], 'Repair registration/state differs')
        require(all(stored[k] == entry[k] for k in entry), 'Repair manifest/runtime changed')
        require(sha(location / 'driver_before_repair.json') == manifest['original_driver_preimage_sha256'], 'Original driver preimage changed')
        for name in ('e1_resume_fixed.py', 'repair_manifest.json'):
            require(sha(location / name) == sha(HERE / name), 'Retained repair code changed')
    for name, expected in manifest['original_runtime_sha256'].items():
        require(sha(common.safe(location, 'original_runtime/' + name)) == expected, 'Retained original runtime changed')
    if prior and state.get('error'):
        state.setdefault('repair_resume_errors', []).append({'error': state['error'], 'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    state.pop('error', None)
    common.write_json(stage / 'driver.json', state)


def adopt(common, root, stage, state, policy, job, manifest):
    if job['id'] != manifest['adopt_job_id'] or state['jobs'][job['id']].get('status') == 'COMPLETED':
        return
    native, record, proof = verify_candidate(common, root, stage, state, policy, manifest)
    proof_path = native / 'verification/E1_COMPLETED.json'
    if proof_path.exists():
        require(read(proof_path) == proof, 'Existing adoption proof differs')
    else:
        common.write_json(proof_path, proof)
    record.update(status='COMPLETED', completion_proof=str(proof_path))
    record['attempts'][0]['status'] = 'COMPLETED'
    common.write_json(stage / 'repair' / PATCH_ID / 'adoption.json',
        {'patch_id': PATCH_ID, 'status': 'ADOPTED_SAVED_COMPLETED_RUN', 'id': job['id'],
         'best_epoch': proof['best_epoch'], 'stop_epoch': proof['stop_epoch'],
         'checkpoint_sha256': proof['checkpoint_sha256'], 'fourth_checkpoint_digest_first_recorded_at_repair': True,
         'checkpoint_note': 'Original X3 training proof records the filename but no checkpoint digest; this repair binds its first digest. Subsequent inference must reproduce the saved canonical outputs.',
         'output_metadata_sha256': proof['metadata_sha256']})
    common.write_json(stage / 'driver.json', state)
    print('E1_ADOPT_COMPLETED=' + job['id'] + ' best31 stop50; no retraining', flush=True)


def driver_source(toolkit):
    source = (toolkit / 'run_e1_min50.py').read_text(encoding='utf-8')
    source = replace_once(source, DRIVER_PREPARE_OLD, DRIVER_PREPARE_NEW)
    source = replace_once(source, DRIVER_ADOPT_OLD, DRIVER_ADOPT_NEW)
    return replace_once(source, DRIVER_COLLECT_OLD, DRIVER_COLLECT_NEW)


def collect(common, root, stage, policy, manifest):
    """Run the original collector with corrected validation, then append pinned repair provenance."""
    require(str(root) == manifest['root'] and str(stage) == manifest['stage'], 'Collector targets a different batch')
    state = read(stage / 'driver.json')
    require(state['policy_sha256'] == POLICY_SHA, 'Collector batch policy changed')
    location = common.safe(stage, 'repair/' + PATCH_ID)
    registration = read(location / 'repair_registration.json')
    require(registration in state.get('repair_history', []), 'Repair not registered in driver')
    for name in ('e1_resume_fixed.py', 'repair_manifest.json'):
        require(sha(location / name) == sha(HERE / name), 'Collector repair provenance changed')
    for name, expected in manifest['original_runtime_sha256'].items():
        require(sha(common.safe(location, 'original_runtime/' + name)) == expected, 'Collector original-runtime provenance changed')
    require(sha(location / 'driver_before_repair.json') == manifest['original_driver_preimage_sha256'], 'Collector original driver preimage changed')
    original = (common.HERE / 'collect_e1_min50.py').read_text(encoding='utf-8')
    # Execute only the locally generated, sealed standard-library collector, never a ZIP member.
    module = types.ModuleType('_e1_original_collector')
    module.__file__ = str(common.HERE / 'collect_e1_min50.py')
    exec(compile(original, module.__file__, 'exec'), module.__dict__)
    import contextlib
    import io
    captured = io.StringIO()
    previous_argv = sys.argv[:]
    try:
        sys.argv = [module.__file__, '--stage', str(stage)]
        with contextlib.redirect_stdout(captured):
            code = module.main()
    finally:
        sys.argv = previous_argv
    require(code == 0, 'Original collector failed')
    messages = captured.getvalue().splitlines()
    paths = [line.split('=', 1)[1] for line in messages if line.startswith('E1_RESULTS_ZIP=')]
    require(len(paths) == 1, 'Collector did not identify one result ZIP')
    archive = Path(paths[0])
    require(archive.parent == stage / 'reports' and archive.name.startswith('e1_min50_training_results_'), 'Unexpected result archive path')
    names = ('e1_resume_fixed.py', 'repair_manifest.json', 'driver_before_repair.json', 'repair_registration.json')
    names += tuple('original_runtime/' + n for n in sorted(manifest['original_runtime_sha256']))
    if (location / 'adoption.json').is_file():
        adoption = read(location / 'adoption.json')
        require(adoption['status'] == 'ADOPTED_SAVED_COMPLETED_RUN' and adoption['id'] == manifest['adopt_job_id']
                and adoption['fourth_checkpoint_digest_first_recorded_at_repair'] is True, 'Adoption report identity differs')
        saved = state['jobs'][manifest['adopt_job_id']]
        native = common.checked_native(root, stage, saved['native_stage'])
        proof = read(native / 'verification/E1_COMPLETED.json')
        require(adoption['checkpoint_sha256'] == proof['checkpoint_sha256']
                and adoption['output_metadata_sha256'] == proof['metadata_sha256'], 'Adoption report differs from completed proof')
        names += ('adoption.json',)
    with zipfile.ZipFile(archive, 'a', compression=zipfile.ZIP_DEFLATED) as zipped:
        require(not any(n.startswith('repair/') for n in zipped.namelist()), 'Duplicate repair archive members')
        for name in names:
            zipped.write(location / name, 'repair/' + PATCH_ID + '/' + name)
    checksum = archive.with_suffix('.zip.sha256.txt')
    checksum.write_bytes((sha(archive) + '  ' + archive.name + '\n').encode('ascii'))
    # Print final archive/checksum paths only after the provenance and final checksum are complete.
    for line in messages:
        if not line.startswith(('E1_RESULTS_ZIP=', 'E1_RESULTS_SHA256=')):
            print(line)
    print('E1_REPAIR_PROVENANCE_BUNDLED=' + PATCH_ID)
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
    manifest = overlay_manifest()
    if not args.run and not args.collect:
        print('PLAN_ONLY: no original project files, checkpoints, Torch or GPU accessed.')
        print('Verify and keep three completed LBSP models; adopt saved ML1M Shuffle42 (best31/stop50); continue15 pending jobs.')
        print('Original scientific code, original policy and all80 original runtime dependencies remain unchanged.')
        return 0
    root = args.root.resolve()
    stage = args.stage.resolve() if args.stage else Path(manifest['stage'])
    require(os.name == 'posix' and str(root) == manifest['root'] and str(stage) == manifest['stage'], 'Execution is restricted to the recorded Linux batch')
    require(not stage.is_symlink(), 'Stage symlink rejected')
    toolkit = root / manifest['original_toolkit_directory']
    common, policy = load_common(toolkit, manifest)
    if args.collect:
        return collect(common, root, stage, policy, manifest)
    module = types.ModuleType('_e1_original_driver_fixed')
    module.__file__ = str(toolkit / 'run_e1_min50.py')
    module.__dict__.update(
        _e1_repair_prepare=lambda r, s, state, p: prepare(common, r, s, state, p, manifest),
        _e1_repair_adopt=lambda r, s, state, p, j: adopt(common, r, s, state, p, j, manifest),
        _e1_repair_collect_command=lambda r, s: [sys.executable, str(Path(__file__).resolve()), '--root', str(r), '--stage', str(s), '--collect'])
    exec(compile(driver_source(toolkit), module.__file__, 'exec'), module.__dict__)
    previous_argv = sys.argv[:]
    try:
        sys.argv = [module.__file__, '--root', str(root), '--stage', str(stage), '--run']
        return module.main()
    finally:
        sys.argv = previous_argv


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('ERROR: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(1)

