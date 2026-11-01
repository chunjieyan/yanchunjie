"""Recheck and package X6 temporary training-cost metadata, never data/weights."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import csv
import datetime
import hashlib
import io
import json
import math
import statistics
import sys
import uuid
import zipfile

VERSION = 'x6_train_cost_v1'
INPUT_NAME = 'x6_train_cost_inputs_20261027.json'
INPUT_SHA256 = '1c03866a3b8e0a8c6f6e7a8653db613a2188bef5b929b72671520a054d614f75'
TAGS = ('ml1m', 'beauty', 'sports', 'toys')
METHODS = ('control', 'shuffle', 'lbsp')
TOOLS = ('run_x6_train_cost.py', 'collect_x6_train_cost.py', 'x6_train_cost_prepare.py', 'x3_inference_adapter.py')
PREFIX = 'kais_x6_train_cost_v1_'
SIZE_LIMIT = 32 * 1024 * 1024
TELEMETRY_SCOPE = 'device-wide samples include this process; they do not establish exclusive GPU access'


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read(stage, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or '..' in part.parts or '\\' in relative:
        raise ValueError('Unsafe report path: ' + relative)
    path = stage.joinpath(*part.parts)
    path.resolve().relative_to(stage.resolve())
    if path.is_symlink() or not path.is_file() or path.stat().st_size > SIZE_LIMIT:
        raise ValueError('Missing/unsafe/oversized metadata: ' + relative)
    return path.read_bytes()


def obj(blob):
    value = json.loads(blob.decode('utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value


def ast_digest(node):
    return digest(ast.dump(node, include_attributes=False).encode('utf-8'))


def checked_def(text, name):
    nodes = [node for node in ast.parse(text).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
    if len(nodes) != 1:
        raise ValueError('Expected one scientific definition: ' + name)
    return ast_digest(nodes[0])


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric measurement')
    return float(value)


def validate_common(value, inputs, manifest, manifest_sha, tag, method, status):
    expected = {'version': VERSION, 'tag': tag, 'method': method, 'seed': 42, 'status': status,
                'input_sha256': INPUT_SHA256, 'manifest_sha256': manifest_sha, 'files_sha256': manifest['files_sha256'],
                'protocol': inputs['protocol'], 'formal_checkpoint_loaded_or_saved': False,
                'training_model': 'fresh temporary seed42 model and Adam optimizer'}
    if any(value.get(key) != v for key, v in expected.items()):
        raise ValueError('Proof identity/protocol/source changed')
    plan = inputs['datasets'][tag]
    if value.get('contract') != {'rows': plan['expected_training_rows'], 'positions': plan['expected_training_positions'], 'items': plan['expected_items']}:
        raise ValueError('Frozen data coverage differs')
    env = value['environment']
    if (any(not isinstance(env.get(key), str) or not env[key] for key in ('python', 'torch', 'numpy', 'cuda', 'device', 'device_uuid'))
            or env.get('cublas_workspace_config') != ':4096:8' or env.get('deterministic_algorithms') is not True
            or env.get('cuda_matmul_tf32') is not False or env.get('cudnn_tf32') is not False
            or int(env['total_memory_bytes']) <= 0 or int(env['device_count']) <= 0):
        raise ValueError('Missing GPU/software or numerical backend evidence')
    return env


def validate_smoke(value, inputs, manifest, manifest_sha, tag, method):
    env = validate_common(value, inputs, manifest, manifest_sha, tag, method, 'SMOKE_PASS')
    proof = value['proof']
    expected_worker = 'two_epoch_parent_worker_match' if method == 'shuffle' and tag in ('ml1m', 'beauty') else 'not_applicable'
    changed = proof.get('shuffle_changed_probe_windows')
    if (not math.isfinite(finite(proof['finite_loss'])) or int(proof['finite_gradient_tensors']) < 1
            or proof['optimizer_steps'] != 0 or proof['real_training_epoch_started'] is not False
            or proof['validation_or_test_inference'] is not False or proof['checkpoint_io'] is not False
            or proof['shuffle_persistent_workers'] != expected_worker or type(changed) is not int
            or not (1 <= changed <= 32 if expected_worker != 'not_applicable' else changed == 0)):
        raise ValueError('Gradient/worker/smoke scope proof invalid')
    return env


def validate_telemetry(value):
    if not isinstance(value, dict) or type(value.get('available')) is not bool or value.get('scope') != TELEMETRY_SCOPE:
        raise ValueError('Device telemetry scope/availability invalid')
    timestamp = datetime.datetime.fromisoformat(value['recorded_utc'])
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError('Telemetry timestamp is not timezone-aware')
    if value['available']:
        if not isinstance(value.get('csv'), str) or not value['csv'].strip() or value.get('exit_code') != 0:
            raise ValueError('Available device telemetry requires successful nonempty output')
        rows = list(csv.reader(io.StringIO(value['csv'])))
        if not rows or any(len(row) != 7 or any(not field.strip() for field in row) for row in rows):
            raise ValueError('Device telemetry must contain seven nonempty fields per GPU')
    elif not isinstance(value.get('reason'), str) or not value['reason']:
        raise ValueError('Unavailable device telemetry must provide a reason')
    return timestamp


def validate_cost(value, inputs, manifest, manifest_sha, tag, method):
    env = validate_common(value, inputs, manifest, manifest_sha, tag, method, 'COMPLETED')
    if (value.get('training_checkpoint_written') is not False or value.get('validation_or_test_inference') is not False
            or value.get('originals_unchanged') is not True):
        raise ValueError('Cost scope or preservation proof invalid')
    epochs = value['epochs']
    if len(epochs) != 6 or [entry['epoch'] for entry in epochs] != list(range(1, 7)):
        raise ValueError('Expected warmup1 and measured2..6')
    plan = inputs['datasets'][tag]
    for row in epochs:
        before = validate_telemetry(row['device_telemetry_before'])
        after = validate_telemetry(row['device_telemetry_after'])
        if before > after:
            raise ValueError('Telemetry samples are not in epoch order')
        total, setup = finite(row['seconds']), finite(row['augmentation_setup_seconds'])
        if (row['warmup'] != (row['epoch'] == 1) or total <= 0 or not 0 <= setup <= total
                or (method != 'shuffle' and setup != 0)
                or row['positions'] != plan['expected_training_positions']
                or row['batches'] != math.ceil(plan['expected_training_positions'] / 128)
                or row['validation_included'] is not False or row['checkpoint_io_included'] is not False
                or row['training_state'] != 'temporary model and optimizer only'):
            raise ValueError('Epoch time, coverage, or scope invalid')
        finite(row['loss_sum_as_logged'])
        for kind in ('allocated', 'reserved'):
            b, p, inc = (row[key + '_' + kind + '_bytes'] for key in ('baseline', 'peak', 'incremental_peak'))
            if any(isinstance(n, bool) or not isinstance(n, int) for n in (b, p, inc)) or b < 0 or p < b or inc != p - b:
                raise ValueError('Memory baseline/peak inconsistency')
        if row['peak_allocated_bytes'] > row['peak_reserved_bytes']:
            raise ValueError('Allocated peak exceeds reserved peak')
    measured = [float(row['seconds']) for row in epochs[1:]]
    setups = [float(row['augmentation_setup_seconds']) for row in epochs[1:]]
    for key, correct in [('mean_seconds', statistics.mean(measured)), ('sample_sd_seconds', statistics.stdev(measured)),
                         ('mean_augmentation_setup_seconds', statistics.mean(setups))]:
        if not math.isclose(finite(value[key]), correct, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Five-epoch aggregate mismatch: ' + key)
    return env


def source_proofs(stage, inputs, manifest, files):
    expected_generated = {'scripts/x6_cost_models.py'} | {'scripts/kernel_' + tag + '_' + method + '.py' for tag in TAGS for method in METHODS}
    expected_files = {'scripts/' + name for name in TOOLS} | {INPUT_NAME} | expected_generated | {'sources/' + path for path in inputs['sources_sha256']}
    if set(manifest['files_sha256']) != expected_files or manifest['files_sha256'][INPUT_NAME] != INPUT_SHA256:
        raise ValueError('Stage source/tool inventory differs')
    for relative, expected in manifest['files_sha256'].items():
        blob = read(stage, relative)
        if digest(blob) != expected:
            raise ValueError('Tool/source hash differs: ' + relative)
        files['snapshot/' + relative] = blob
    for path, expected in inputs['sources_sha256'].items():
        if digest(files['snapshot/sources/' + path]) != expected:
            raise ValueError('Pinned original source hash differs: ' + path)
    for name, expected in inputs['helper_sha256'].items():
        if manifest['files_sha256']['scripts/' + name] != expected:
            raise ValueError('Pinned generator/adapter changed')
    if digest(Path(__file__).read_bytes()) != manifest['files_sha256']['scripts/collect_x6_train_cost.py']:
        raise ValueError('Executing collector does not match staged collector')
    sources = {path: files['snapshot/sources/' + path].decode('utf-8-sig') for path in inputs['sources_sha256']}
    model = files['snapshot/scripts/x6_cost_models.py'].decode('utf-8-sig')
    checks = [('SASRecControl', inputs['model_control_source']), ('RecentBucketSetR1SASRec', inputs['model_lbsp_source']),
              ('build_r1_group_positions_from_size', inputs['model_lbsp_source']),
              ('canonicalize_last_bucket_items', inputs['model_lbsp_source']),
              ('require', inputs['model_lbsp_forward_source']), ('predict_batch_masked', inputs['model_lbsp_forward_source'])]
    for name, path in checks:
        if checked_def(model, name) != checked_def(sources[path], name):
            raise ValueError('Frozen model definition AST changed: ' + name)
    for tag in TAGS:
        plan = inputs['datasets'][tag]
        origins = {'control_training': plan['control_source'], 'shuffle_training': plan['shuffle_source'],
                   'lbsp_training': plan['lbsp_training_source'], 'lbsp_base': inputs['model_lbsp_source']}
        for method in METHODS:
            spec = manifest['kernel_specs'][tag + '_' + method]
            kernel = files['snapshot/scripts/kernel_' + tag + '_' + method + '.py'].decode('utf-8-sig')
            if spec['tag'] != tag or spec['method'] != method or spec['seed'] != 42:
                raise ValueError('Kernel spec identity differs')
            for name, expected in spec['scientific_definition_ast_sha256'].items():
                source_path = origins[spec['scientific_definition_source'][name]]
                if checked_def(kernel, name) != expected or checked_def(sources[source_path], name) != expected:
                    raise ValueError('Training scientific definition AST changed: ' + name)
    return {'status': 'PASS', 'pinned_source_files': len(inputs['sources_sha256']),
            'training_data_and_map_hash_records_not_opened': len(inputs['data_sha256']), 'generated_kernels': 12,
            'scientific_AST_matches_pinned_sources': True, 'model_AST_matches_pinned_sources': True}


def collect(stage):
    stage = Path(stage).resolve()
    if not stage.name.startswith(PREFIX):
        raise ValueError('Expected X6 training cost stage')
    files, errors, integrity = {}, [], []
    out = stage / 'verification' / ('x6_train_cost_review_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    source_check = {'status': 'FAIL'}
    inputs, manifest, manifest_sha = {}, {}, ''
    try:
        manifest_blob = read(stage, 'manifest.json')
        manifest_sha = digest(manifest_blob)
        manifest = obj(manifest_blob)
        input_blob = read(stage, INPUT_NAME)
        if digest(input_blob) != INPUT_SHA256:
            raise ValueError('Reviewed input SHA differs')
        inputs = obj(input_blob)
        if (manifest.get('version') != VERSION or inputs.get('version') != VERSION
                or manifest.get('stage') != str(stage) or manifest.get('root') != str(stage.parent)
                or manifest.get('input_sha256') != INPUT_SHA256 or manifest.get('sources_sha256') != inputs['sources_sha256']
                or manifest.get('data_sha256') != inputs['data_sha256'] or manifest.get('protocol') != inputs['protocol']
                or len(inputs['sources_sha256']) != 13 or len(inputs['data_sha256']) != 8
                or set(inputs['datasets']) != set(TAGS) or manifest.get('tasks') != [{'tag': t, 'method': m, 'seed': 42} for t in TAGS for m in METHODS]):
            raise ValueError('Manifest, task or frozen-input scope differs')
        files['snapshot/manifest.json'] = manifest_blob
        source_check = source_proofs(stage, inputs, manifest, files)
    except (KeyError, OSError, ValueError, TypeError, SyntaxError) as exc:
        integrity.append(type(exc).__name__ + ': ' + str(exc))
    smokes, results, envs, driver = {}, {}, [], {}
    if not integrity:
        try:
            driver_blob = read(stage, 'verification/driver.json')
            files['snapshot/verification/driver.json'] = driver_blob
            driver = obj(driver_blob)
        except (OSError, ValueError, KeyError) as exc:
            errors.append('Missing/invalid driver: ' + str(exc))
        tasks = [(mode, tag, method) for mode in ('smoke', 'measure') for tag in TAGS for method in METHODS]
        jobs = driver.get('jobs', [])
        if driver.get('version') != VERSION or driver.get('stage') != str(stage) or driver.get('status') != 'ALL_TWELVE_SMOKES_AND_TWELVE_TRAINING_COSTS_COMPLETED' or len(jobs) != len(tasks):
            errors.append('Driver did not complete all 24 jobs')
        previous_finished, python_path = None, None
        for index, (mode, tag, method) in enumerate(tasks):
            relative = ('verification/smoke_' + tag + '_' + method + '.json') if mode == 'smoke' else ('outputs/' + tag + '_' + method + '.json')
            log = 'verification/' + mode + '_' + tag + '_' + method + '.log'
            name = tag + '_' + method
            try:
                blob = read(stage, relative)
                files['snapshot/' + relative] = blob
                value = obj(blob)
                env = validate_smoke(value, inputs, manifest, manifest_sha, tag, method) if mode == 'smoke' else validate_cost(value, inputs, manifest, manifest_sha, tag, method)
                (smokes if mode == 'smoke' else results)[name] = {'status': 'PASS', 'data': value}
                envs.append(env)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                (smokes if mode == 'smoke' else results)[name] = {'status': 'FAIL', 'error': str(exc)}
                errors.append(mode + ' ' + name + ': ' + str(exc))
            try:
                log_blob = read(stage, log)
                files['snapshot/' + log] = log_blob
            except (OSError, ValueError) as exc:
                errors.append('Missing log ' + log + ': ' + str(exc))
            if index < len(jobs):
                job = jobs[index]
                command = job.get('command')
                if not isinstance(command, list) or not command or any(not isinstance(value, str) for value in command):
                    integrity.append('Driver command identity mismatch: ' + mode + ' ' + name)
                else:
                    if python_path is None:
                        python_path = command[0]
                    expected_command = [python_path, '-u', str(stage / 'scripts/run_x6_train_cost.py'),
                        '--root', manifest['root'], '--run', '--stage', str(stage), '--mode', mode,
                        '--dataset', tag, '--method', method]
                    if not Path(python_path).is_absolute() or command != expected_command:
                        integrity.append('Driver command identity mismatch: ' + mode + ' ' + name)
                if (job.get('mode') != mode or job.get('tag') != tag or job.get('method') != method
                        or job.get('exit_code') != 0 or job.get('log') != str(stage / log)
                        or job.get('proof') != str(stage / relative) or not job.get('started_utc') or not job.get('finished_utc')):
                    errors.append('Driver job proof incomplete: ' + mode + ' ' + name)
                try:
                    started = datetime.datetime.fromisoformat(job['started_utc'])
                    finished = datetime.datetime.fromisoformat(job['finished_utc'])
                    if started.tzinfo is None or started.utcoffset() is None or finished.tzinfo is None or finished.utcoffset() is None:
                        raise ValueError('Timezone is required')
                    if started > finished or (previous_finished is not None and started < previous_finished):
                        raise ValueError('Jobs overlap or finish before starting')
                    previous_finished = finished
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    errors.append('Invalid driver job timestamps: ' + mode + ' ' + name + ': ' + str(exc))
                for key, file in [('proof_sha256', 'snapshot/' + relative), ('log_sha256', 'snapshot/' + log)]:
                    if key in job and file in files and job[key] != digest(files[file]):
                        integrity.append('Driver ' + key + ' mismatch: ' + mode + ' ' + name)
                    elif key not in job:
                        errors.append('Driver missing ' + key + ': ' + mode + ' ' + name)
        if envs:
            fields = ('python', 'torch', 'numpy', 'cuda', 'device', 'total_memory_bytes', 'capability', 'device_count',
                      'cuda_visible_devices', 'device_uuid', 'cublas_workspace_config', 'deterministic_algorithms', 'cuda_matmul_tf32', 'cudnn_tf32')
            if any(any(item.get(key) != envs[0].get(key) for key in fields) for item in envs[1:]):
                errors.append('GPU/software/backend environment differs between jobs')
    complete = not integrity and not errors and len(smokes) == len(results) == 12 and all(v['status'] == 'PASS' for v in list(smokes.values()) + list(results.values()))
    status = 'INTEGRITY_FAILED_NO_BUNDLE' if integrity else 'ALL_TWELVE_TRAINING_COSTS_CHECKED' if complete else 'INCOMPLETE'
    table = []
    for tag in TAGS:
        for method in METHODS:
            record = results.get(tag + '_' + method, {})
            if record.get('status') != 'PASS':
                continue
            value, measured = record['data'], record['data']['epochs'][1:]
            table.append({'dataset': inputs['datasets'][tag]['dataset'], 'method': method, 'seed': 42,
                'training_cost_source': method, 'mean_seconds': value['mean_seconds'], 'sample_sd_seconds': value['sample_sd_seconds'],
                'mean_augmentation_setup_seconds': value['mean_augmentation_setup_seconds'],
                'mean_peak_allocated_bytes': statistics.mean(row['peak_allocated_bytes'] for row in measured),
                'mean_incremental_peak_allocated_bytes': statistics.mean(row['incremental_peak_allocated_bytes'] for row in measured),
                'positions': value['contract']['positions'], 'warmup_epochs': 1, 'measured_epochs': 5})
        control = next((row for row in table if row['dataset'] == inputs['datasets'][tag]['dataset'] and row['method'] == 'control'), None)
        if control:
            table.append(dict(control, method='ensemble', training_cost_source='shared with control; no ensemble training'))
    telemetry_samples = [entry[key] for record in results.values() if record['status'] == 'PASS'
                         for entry in record['data']['epochs']
                         for key in ('device_telemetry_before', 'device_telemetry_after')]
    telemetry_summary = {'samples': len(telemetry_samples),
                         'available': sum(sample['available'] for sample in telemetry_samples),
                         'unavailable': sum(not sample['available'] for sample in telemetry_samples),
                         'scope': TELEMETRY_SCOPE,
                         'interpretation': 'Optional device-wide utilization/memory/temperature/clock snapshots are outside the timed epoch; they include own recent GPU activity and cannot identify another process or prove exclusive access.'}
    report = {'status': status, 'stage': str(stage), 'input_sha256': INPUT_SHA256, 'source_check': source_check,
              'smoke_status': {k: v['status'] for k, v in smokes.items()}, 'measurement_status': {k: v['status'] for k, v in results.items()},
              'reported_gpu_environment': envs[0] if envs else None, 'device_telemetry': telemetry_summary,
              'gpu_identity_limit': 'same reported device/environment; when device_uuid is unavailable exact physical GPU identity is not independently confirmed',
              'errors': errors, 'integrity_failures': integrity,
              'scope': 'Temporary-model full train epochs only; no validation/test/checkpoint I/O. Ensemble shares Control training. Repeated-time SD is not training-seed variability.'}
    files['training_cost_checks.json'] = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    summary = '# X6 training-only epoch cost\n\nStatus: **' + status + '**\n\n' + report['scope'] + '\n\n' + telemetry_summary['interpretation'] + '\n'
    summary += 'Device telemetry available: %d/%d snapshots.\n' % (telemetry_summary['available'], telemetry_summary['samples'])
    if errors or integrity:
        summary += '\n' + '\n'.join('- ' + s for s in integrity + errors) + '\n'
    files['SUMMARY.md'] = summary.encode('utf-8')
    if table:
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
        files['training_cost_summary.csv'] = stream.getvalue().encode('utf-8')
    ledger = io.StringIO(newline='')
    writer = csv.DictWriter(ledger, fieldnames=['path', 'bytes', 'sha256'])
    writer.writeheader()
    for name, blob in sorted(files.items()):
        writer.writerow({'path': name, 'bytes': len(blob), 'sha256': digest(blob)})
    files['REPORT_SHA256.csv'] = ledger.getvalue().encode('utf-8')
    for relative, blob in files.items():
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    bundle = None
    if not integrity:
        bundle = out / 'x6_training_cost_results.zip'
        with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED) as archive:
            for relative, blob in sorted(files.items()):
                archive.writestr(relative, blob)
        with zipfile.ZipFile(bundle) as archive:
            if archive.testzip() is not None or any(archive.read(name) != blob for name, blob in files.items()):
                raise ValueError('Report archive checksum/byte verification failed')
        bundle.with_name(bundle.name + '.sha256.txt').write_text(digest(bundle.read_bytes()) + '  ' + bundle.name + '\n', 'ascii')
    return {'status': status, 'report_dir': str(out), 'zip_path': str(bundle) if bundle else None,
            'exit_code': 2 if integrity else 0 if complete else 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.stage)
    print('CHECKS=' + str(Path(result['report_dir']) / 'training_cost_checks.json'))
    print('SUMMARY=' + str(Path(result['report_dir']) / 'SUMMARY.md'))
    print('X6_TRAIN_REPORT_ZIP=' + (result['zip_path'] or 'NOT_CREATED'))
    print('STATUS=' + result['status'])
    return result['exit_code']


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('COLLECTION_FAILED=' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
