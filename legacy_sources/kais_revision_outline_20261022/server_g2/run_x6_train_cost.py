"""X6 train-only epoch costs on isolated temporary models; plan-only by default."""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import importlib.util
import io
import json
import math
import shutil
import statistics
import subprocess
import sys
import uuid

VERSION = 'x6_train_cost_v1'
INPUT_NAME = 'x6_train_cost_inputs_20261027.json'
INPUT_SHA256 = '1c03866a3b8e0a8c6f6e7a8653db613a2188bef5b929b72671520a054d614f75'
TAGS = ('ml1m', 'beauty', 'sports', 'toys')
METHODS = ('control', 'shuffle', 'lbsp')
TOOLS = ('run_x6_train_cost.py', 'collect_x6_train_cost.py', 'x6_train_cost_prepare.py', 'x3_inference_adapter.py')
PREFIX = 'kais_x6_train_cost_v1_'
TELEMETRY_SCOPE = 'device-wide samples include this process; they do not establish exclusive GPU access'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe(root, relative):
    root = Path(root).resolve()
    path = root / relative
    path.resolve().relative_to(root)
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def verify_data(root, inputs, tag=None):
    data = inputs['data_sha256']
    if tag:
        plan = inputs['datasets'][tag]
        data = {path: data[path] for path in (plan['train_path'], plan['map_path'])}
    for path, expected in dict(inputs['sources_sha256'], **data).items():
        if sha(safe(root, path)) != expected:
            raise ValueError('Reviewed input/source changed: ' + path)


def verify_stage(root, stage):
    root, stage = Path(root).resolve(), Path(stage).resolve()
    if stage.parent != root or not stage.name.startswith(PREFIX):
        raise ValueError('Stage must be an isolated direct child of the project')
    manifest = read_json(stage / 'manifest.json')
    if manifest['version'] != VERSION or manifest['root'] != str(root) or manifest['stage'] != str(stage):
        raise ValueError('Stage identity differs')
    for relative, expected in manifest['files_sha256'].items():
        if sha(safe(stage, relative)) != expected:
            raise ValueError('Generated tool/kernel changed: ' + relative)
    if sha(stage / INPUT_NAME) != INPUT_SHA256:
        raise ValueError('Input seal changed')
    return manifest, read_json(stage / INPUT_NAME)


def prepare(root):
    root, here = Path(root).resolve(), Path(__file__).resolve().parent
    if not root.is_dir() or sha(here / INPUT_NAME) != INPUT_SHA256:
        raise ValueError('Project root or reviewed input seal missing/changed')
    inputs = read_json(here / INPUT_NAME)
    if inputs['version'] != VERSION or inputs['representative_training_seed'] != 42 or set(inputs['datasets']) != set(TAGS):
        raise ValueError('Training cost protocol or fixed tasks changed')
    for name, expected in inputs['helper_sha256'].items():
        if sha(here / name) != expected:
            raise ValueError('Reviewed helper changed: ' + name)
    verify_data(root, inputs)
    science_bytes = {path: safe(root, path).read_bytes() for path in inputs['sources_sha256']}
    for path, blob in science_bytes.items():
        if hashlib.sha256(blob).hexdigest() != inputs['sources_sha256'][path]:
            raise ValueError('Source changed between hash verification and staging: ' + path)
    science = {path: blob.decode('utf-8-sig') for path, blob in science_bytes.items()}
    adapter = module(here / 'x3_inference_adapter.py', 'owned_x6_source_adapter')
    generator = module(here / 'x6_train_cost_prepare.py', 'owned_x6_train_generator')
    model_text = adapter.build_frozen_module(science[inputs['model_control_source']],
        science[inputs['model_lbsp_source']], science[inputs['model_lbsp_forward_source']])
    generated = {'scripts/x6_cost_models.py': model_text}
    specs = {}
    for tag in TAGS:
        plan = inputs['datasets'][tag]
        for method in METHODS:
            source, metadata = generator.build_training_kernel(tag, method,
                science[plan['control_source']], science[plan['shuffle_source']],
                science[plan['lbsp_training_source']], science[inputs['model_lbsp_source']])
            generated['scripts/kernel_' + tag + '_' + method + '.py'] = source
            specs[tag + '_' + method] = metadata
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:10]
    stage = root / (PREFIX + stamp)
    stage.mkdir()
    for name in ('scripts', 'verification', 'outputs'):
        (stage / name).mkdir()
    files = {}
    for name in TOOLS:
        shutil.copyfile(here / name, stage / 'scripts' / name)
        files['scripts/' + name] = sha(stage / 'scripts' / name)
    shutil.copyfile(here / INPUT_NAME, stage / INPUT_NAME)
    files[INPUT_NAME] = sha(stage / INPUT_NAME)
    for relative, blob in science_bytes.items():
        target = safe(stage, 'sources/' + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        files['sources/' + relative] = sha(target)
    for relative, source in generated.items():
        target = safe(stage, relative)
        target.write_text(source, encoding='utf-8', newline='\n')
        files[relative] = sha(target)
    manifest = {'version': VERSION, 'root': str(root), 'stage': str(stage), 'input_sha256': INPUT_SHA256,
                'files_sha256': files, 'sources_sha256': inputs['sources_sha256'], 'data_sha256': inputs['data_sha256'],
                'protocol': inputs['protocol'], 'kernel_specs': specs,
                'tasks': [{'tag': tag, 'method': method, 'seed': 42} for tag in TAGS for method in METHODS]}
    write_json(stage / 'manifest.json', manifest)
    return stage


def environment(torch, np):
    import os
    prop = torch.cuda.get_device_properties(0)
    return {'python': sys.version, 'torch': str(torch.__version__), 'numpy': str(np.__version__),
            'cuda': str(torch.version.cuda), 'device': str(prop.name), 'total_memory_bytes': int(prop.total_memory),
            'capability': list(torch.cuda.get_device_capability(0)), 'device_count': torch.cuda.device_count(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'device_uuid': str(getattr(prop, 'uuid', 'unavailable')),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cuda_matmul_tf32': bool(torch.backends.cuda.matmul.allow_tf32), 'cudnn_tf32': bool(torch.backends.cudnn.allow_tf32)}


def telemetry():
    """Optional device-wide load snapshot, taken strictly outside epoch timing."""
    captured = datetime.datetime.now(datetime.timezone.utc).isoformat()
    command = ['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm',
               '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return {'available': True, 'scope': TELEMETRY_SCOPE, 'recorded_utc': captured,
                    'csv': result.stdout.strip()[:4000], 'exit_code': 0}
        return {'available': False, 'scope': TELEMETRY_SCOPE, 'recorded_utc': captured,
                'reason': ('nvidia-smi failed or returned no device rows: ' + result.stderr.strip())[:400],
                'exit_code': result.returncode}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'scope': TELEMETRY_SCOPE, 'recorded_utc': captured,
                'reason': type(exc).__name__ + ': ' + str(exc)[:300]}


def validate_telemetry(value):
    if not isinstance(value, dict) or type(value.get('available')) is not bool or value.get('scope') != TELEMETRY_SCOPE:
        raise ValueError('Device telemetry scope/availability invalid')
    moment = datetime.datetime.fromisoformat(value['recorded_utc'])
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError('Device telemetry requires an aware UTC timestamp')
    if value['available']:
        if not isinstance(value.get('csv'), str) or not value['csv'].strip() or value.get('exit_code') != 0:
            raise ValueError('Available device telemetry requires nonempty nvidia-smi output')
        rows = list(csv.reader(io.StringIO(value['csv'])))
        if not rows or any(len(row) != 7 or any(not field.strip() for field in row) for row in rows):
            raise ValueError('Device telemetry must contain seven nonempty fields per GPU')
    elif not isinstance(value.get('reason'), str) or not value['reason']:
        raise ValueError('Unavailable device telemetry must provide a reason')


def smoke(kernel, runtime, torch, tag, method):
    dataset = runtime['dataset']
    if method == 'shuffle':
        runtime['augmentation'].set_epoch(1)
    indices = list(range(min(8, len(dataset))))
    batch = torch.utils.data.default_collate([dataset[i] for i in indices])
    device = runtime['device']
    model = runtime['model']
    model.train()
    model.zero_grad(set_to_none=True)
    if method != 'lbsp' and tag in ('ml1m', 'beauty'):
        seq, groups, tokens, pooled, last, target = batch
        seq, groups, tokens, target = [x.long().to(device) for x in (seq, groups, tokens, target)]
        loss = kernel.train_one_batch(model, seq, groups, tokens, target)
    elif method == 'lbsp':
        seq, groups, tokens, target = [x.to(device) for x in batch]
        loss = kernel.training_loss(seq, groups, tokens, target)
    else:
        seq, lengths, target = [x.to(device) for x in batch]
        loss = kernel.loss_for_batch(seq, lengths, target)
    if not bool(torch.isfinite(loss)) or not loss.requires_grad:
        raise ValueError('Smoke loss is nonfinite or has no gradient')
    loss.backward()
    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    if not gradients or not all(bool(torch.isfinite(grad).all()) for grad in gradients):
        raise ValueError('Smoke gradient is nonfinite or absent')
    model.zero_grad(set_to_none=True)
    worker_check, changed_probe_windows = 'not_applicable', 0
    if method == 'shuffle' and tag in ('ml1m', 'beauty'):
        # Pick windows whose histories really change between epochs; otherwise
        # two equal worker results would not demonstrate an epoch-state refresh.
        candidates_count = min(512, len(dataset))
        candidates = [i * (len(dataset) - 1) // max(1, candidates_count - 1) for i in range(candidates_count)]
        runtime['augmentation'].set_epoch(1)
        epoch_one = [dataset[index][0].clone() for index in candidates]
        runtime['augmentation'].set_epoch(2)
        selected = [index for index, first in zip(candidates, epoch_one)
                    if not torch.equal(first, dataset[index][0])][:32]
        if not selected:
            raise ValueError('Sampled Shuffle windows did not change across epochs')
        changed_probe_windows = len(selected)
        generator = torch.Generator().manual_seed(428192)
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, selected), batch_size=8,
            shuffle=False, num_workers=4, persistent_workers=True, generator=generator)
        observed_epochs = []
        for epoch in (1, 2):
            runtime['augmentation'].set_epoch(epoch)
            expected = [torch.utils.data.default_collate([dataset[index] for index in selected[start:start + 8]])
                        for start in range(0, len(selected), 8)]
            observed = list(loader)
            if len(expected) != len(observed) or any(not torch.equal(a, b) for old, new in zip(expected, observed) for a, b in zip(old, new)):
                raise ValueError('Persistent shuffle workers did not observe the epoch change')
            observed_epochs.append([batch[0].clone() for batch in observed])
        if not any(not torch.equal(first, second) for first, second in zip(*observed_epochs)):
            raise ValueError('Persistent worker outputs did not actually change between epochs')
        del loader
        worker_check = 'two_epoch_parent_worker_match'
    return {'finite_loss': float(loss.detach().cpu()), 'finite_gradient_tensors': len(gradients),
            'optimizer_steps': 0, 'real_training_epoch_started': False, 'validation_or_test_inference': False,
            'checkpoint_io': False, 'shuffle_persistent_workers': worker_check,
            'shuffle_changed_probe_windows': changed_probe_windows}


def child(root, stage, tag, method, mode):
    manifest, inputs = verify_stage(root, stage)
    verify_data(root, inputs, tag)
    plan = inputs['datasets'][tag]
    # Explicit server execution only. The stage/scripts directory is Python's
    # script directory, so the generated x6_cost_models module is importable.
    import pandas as pd
    import numpy as np
    import torch
    kernel = module(stage / 'scripts' / ('kernel_' + tag + '_' + method + '.py'), 'x6_kernel')
    frame = pd.read_csv(safe(root, plan['train_path']), dtype={'user_id': str, 'item_id': str})
    mapping_frame = pd.read_csv(safe(root, plan['map_path']), dtype={'raw_item_id': str, 'item_idx': int})
    mapping = dict(zip(mapping_frame['raw_item_id'], mapping_frame['item_idx']))
    if len(mapping_frame) != len(mapping) or len(mapping) != plan['expected_items']:
        raise ValueError('Frozen item mapping identity/count invalid')
    data = kernel.prepare_data(frame, mapping, plan['expected_training_rows'], plan['expected_training_positions'])
    runtime = kernel.initialize(data, plan['expected_items'], seed=42)
    common = {'version': VERSION, 'tag': tag, 'method': method, 'seed': 42,
              'input_sha256': INPUT_SHA256, 'manifest_sha256': sha(stage / 'manifest.json'),
              'files_sha256': manifest['files_sha256'], 'protocol': inputs['protocol'],
              'contract': {'rows': data['rows'], 'positions': data['positions'], 'items': len(mapping)},
              'environment': environment(torch, np), 'training_model': 'fresh temporary seed42 model and Adam optimizer',
              'formal_checkpoint_loaded_or_saved': False}
    if mode == 'smoke':
        proof = smoke(kernel, runtime, torch, tag, method)
        verify_data(root, inputs, tag)
        verify_stage(root, stage)
        write_json(stage / 'verification' / ('smoke_' + tag + '_' + method + '.json'), dict(common, status='SMOKE_PASS', proof=proof))
        print('X6_TRAIN_SMOKE_PASS=' + tag + '_' + method, flush=True)
        return
    for prior_tag in TAGS:
        for prior_method in METHODS:
            validate_smoke(stage, inputs, prior_tag, prior_method)
    measurements = []
    for epoch in range(1, 7):
        telemetry_before = telemetry()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        row = kernel.run_epoch(runtime, epoch)
        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        telemetry_after = telemetry()
        row.update(baseline_allocated_bytes=allocated, baseline_reserved_bytes=reserved,
                   peak_allocated_bytes=peak_alloc, peak_reserved_bytes=peak_reserved,
                   incremental_peak_allocated_bytes=peak_alloc - allocated,
                   incremental_peak_reserved_bytes=peak_reserved - reserved,
                   device_telemetry_before=telemetry_before, device_telemetry_after=telemetry_after)
        measurements.append(row)
        print('X6_TRAIN_EPOCH=' + json.dumps({'tag': tag, 'method': method, **row}), flush=True)
    selected = [row['seconds'] for row in measurements if not row['warmup']]
    setup = [row['augmentation_setup_seconds'] for row in measurements if not row['warmup']]
    verify_data(root, inputs, tag)
    verify_stage(root, stage)
    result = dict(common, status='COMPLETED', epochs=measurements,
        mean_seconds=statistics.mean(selected), sample_sd_seconds=statistics.stdev(selected),
        mean_augmentation_setup_seconds=statistics.mean(setup), training_checkpoint_written=False,
        validation_or_test_inference=False, originals_unchanged=True)
    write_json(stage / 'outputs' / (tag + '_' + method + '.json'), result)
    print('X6_TRAIN_COST_COMPLETED=' + tag + '_' + method, flush=True)


def validate_common(stage, inputs, value, tag, method, status):
    manifest = read_json(stage / 'manifest.json')
    plan = inputs['datasets'][tag]
    expected = {'version': VERSION, 'tag': tag, 'method': method, 'seed': 42, 'status': status,
                'input_sha256': INPUT_SHA256, 'manifest_sha256': sha(stage / 'manifest.json'),
                'files_sha256': manifest['files_sha256'], 'protocol': inputs['protocol'],
                'formal_checkpoint_loaded_or_saved': False}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError('Training cost proof identity/source differs')
    if value['contract'] != {'rows': plan['expected_training_rows'], 'positions': plan['expected_training_positions'], 'items': plan['expected_items']}:
        raise ValueError('Training data coverage differs')
    env = value['environment']
    if (not env['deterministic_algorithms'] or env['cuda_matmul_tf32'] or env['cudnn_tf32']
            or env['cublas_workspace_config'] != ':4096:8'):
        raise ValueError('Controlled numerical backend differs')


def validate_smoke(stage, inputs, tag, method):
    path = stage / 'verification' / ('smoke_' + tag + '_' + method + '.json')
    value = read_json(path)
    validate_common(stage, inputs, value, tag, method, 'SMOKE_PASS')
    proof = value['proof']
    expected_worker = 'two_epoch_parent_worker_match' if method == 'shuffle' and tag in ('ml1m', 'beauty') else 'not_applicable'
    changed = proof.get('shuffle_changed_probe_windows')
    if (not math.isfinite(proof['finite_loss']) or proof['finite_gradient_tensors'] < 1 or proof['optimizer_steps'] != 0
            or proof['real_training_epoch_started'] is not False or proof['validation_or_test_inference'] is not False
            or proof['checkpoint_io'] is not False or proof['shuffle_persistent_workers'] != expected_worker
            or type(changed) is not int or not (1 <= changed <= 32 if expected_worker != 'not_applicable' else changed == 0)):
        raise ValueError('Smoke numerical/scope/worker proof differs')
    return path


def validate_cost(stage, inputs, tag, method):
    path = stage / 'outputs' / (tag + '_' + method + '.json')
    value = read_json(path)
    validate_common(stage, inputs, value, tag, method, 'COMPLETED')
    if any(value.get(key) is not expected for key, expected in
           [('training_checkpoint_written', False), ('validation_or_test_inference', False), ('originals_unchanged', True)]):
        raise ValueError('Training-cost scope proof differs')
    epochs = value['epochs']
    if len(epochs) != 6 or [row['epoch'] for row in epochs] != list(range(1, 7)):
        raise ValueError('Expected one warmup and five measured epochs')
    expected_batches = math.ceil(inputs['datasets'][tag]['expected_training_positions'] / 128)
    for row in epochs:
        validate_telemetry(row['device_telemetry_before'])
        validate_telemetry(row['device_telemetry_after'])
        before = datetime.datetime.fromisoformat(row['device_telemetry_before']['recorded_utc'])
        after = datetime.datetime.fromisoformat(row['device_telemetry_after']['recorded_utc'])
        if before > after:
            raise ValueError('Telemetry samples are not in epoch order')
        if (row['warmup'] != (row['epoch'] == 1) or not math.isfinite(row['seconds']) or row['seconds'] <= 0
                or not 0 <= row['augmentation_setup_seconds'] <= row['seconds']
                or (method != 'shuffle' and row['augmentation_setup_seconds'] != 0)
                or row['positions'] != inputs['datasets'][tag]['expected_training_positions'] or row['batches'] != expected_batches
                or row['validation_included'] is not False or row['checkpoint_io_included'] is not False):
            raise ValueError('Cost epoch timing/coverage/scope invalid')
        for kind in ('allocated', 'reserved'):
            baseline, peak, delta = (row[name + '_' + kind + '_bytes'] for name in ('baseline', 'peak', 'incremental_peak'))
            if baseline < 0 or peak < baseline or delta != peak - baseline:
                raise ValueError('GPU memory baseline/peak invalid')
    selected = [row['seconds'] for row in epochs[1:]]
    setup = [row['augmentation_setup_seconds'] for row in epochs[1:]]
    if (not math.isclose(value['mean_seconds'], statistics.mean(selected), rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(value['sample_sd_seconds'], statistics.stdev(selected), rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(value['mean_augmentation_setup_seconds'], statistics.mean(setup), rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('Five-epoch aggregates differ')
    return path


def run(root):
    stage = prepare(root)
    root = Path(root).resolve()
    print('X6_TRAIN_STAGE=' + str(stage), flush=True)
    driver_path = stage / 'verification/driver.json'
    state = {'version': VERSION, 'stage': str(stage), 'status': 'RUNNING', 'jobs': []}
    write_json(driver_path, state)
    result, process = 1, None
    try:
        tasks = [(mode, tag, method) for mode in ('smoke', 'measure') for tag in TAGS for method in METHODS]
        for mode, tag, method in tasks:
            command = [sys.executable, '-u', str(stage / 'scripts/run_x6_train_cost.py'), '--root', str(root), '--run',
                       '--stage', str(stage), '--mode', mode, '--dataset', tag, '--method', method]
            log = stage / 'verification' / (mode + '_' + tag + '_' + method + '.log')
            record = {'mode': mode, 'tag': tag, 'method': method, 'command': command, 'log': str(log),
                      'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            state['jobs'].append(record)
            write_json(driver_path, state)
            with log.open('x', encoding='utf-8') as handle:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace')
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(line, end='', flush=True)
                record['exit_code'] = process.wait()
            process = None
            if record['exit_code']:
                raise RuntimeError('Failed ' + mode + ' ' + tag + ' ' + method)
            inputs = read_json(stage / INPUT_NAME)
            proof = validate_smoke(stage, inputs, tag, method) if mode == 'smoke' else validate_cost(stage, inputs, tag, method)
            record.update(proof=str(proof), proof_sha256=sha(proof), log_sha256=sha(log),
                          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(driver_path, state)
        result = 0
    except BaseException as exc:
        state['error'] = type(exc).__name__ + ': ' + str(exc)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print('X6_TRAIN_ERROR=' + state['error'], flush=True)
    finally:
        state['status'] = 'ALL_TWELVE_SMOKES_AND_TWELVE_TRAINING_COSTS_COMPLETED' if result == 0 else 'STOPPED_INCOMPLETE'
        write_json(driver_path, state)
        print('X6_TRAIN_DRIVER_STATUS=' + state['status'], flush=True)
        collected = subprocess.run([sys.executable, str(stage / 'scripts/collect_x6_train_cost.py'), '--stage', str(stage)])
        if collected.returncode:
            result = collected.returncode
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--stage', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=('smoke', 'measure'), help=argparse.SUPPRESS)
    parser.add_argument('--dataset', choices=TAGS, help=argparse.SUPPRESS)
    parser.add_argument('--method', choices=METHODS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.run:
        print('PLAN_ONLY_NO_FILES_OR_GPU: four datasets x Control/Shuffle/LBSP; twelve worker/gradient smokes; then twelve isolated temporary-model costs (one warmup + five full training-only epochs); ensemble shares Control training cost; no checkpoint I/O.')
        return 0
    if args.stage:
        if args.mode is None or args.dataset is None or args.method is None:
            parser.error('Internal stage requires mode/dataset/method')
        child(args.root.resolve(), args.stage.resolve(), args.dataset, args.method, args.mode)
        return 0
    return run(args.root)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('ERROR: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
