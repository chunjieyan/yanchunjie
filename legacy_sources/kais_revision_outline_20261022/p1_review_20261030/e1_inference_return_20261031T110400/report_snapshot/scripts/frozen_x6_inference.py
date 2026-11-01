"""X6 inference-cost substage: 4 methods x 4 datasets, seed42, no training.

Plan-only by default. Explicit --run executes private GPU measurements.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import uuid

VERSION = 'x6_inference_v1'
INPUT_NAME = 'x6_inference_inputs_20261027.json'
INPUT_SHA256 = 'd10cf611d84694624526c78cfafa372752d05d8679e40437c813b4ba3e5c2a32'
TOOLS = ('run_x6_inference.py', 'collect_x6_inference.py', 'x3_inference_adapter.py',
         'x3_inference_protocol.py', 'x3_core_helpers.py', 'x6_model_definitions.py', 'x6_cost_protocol.py')
TAGS = ('ml1m', 'beauty', 'sports', 'toys')
METHODS = ('control', 'shuffle', 'ensemble', 'lbsp')
PERMS = (101, 202, 303, 404, 505)
PREFIX = 'kais_x6_inference_v1_'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe(root, relative):
    root = Path(root).resolve()
    path = root / relative
    path.resolve().relative_to(root)
    return path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def verify_root(root, inputs, required=None):
    files = inputs['root_files_sha256']
    if required is not None:
        files = {path: files[path] for path in required}
    for relative, digest in files.items():
        if sha(safe(root, relative)) != digest:
            raise ValueError('Reviewed X6 input/reference changed: ' + relative)


def verify_stage(root, stage):
    root, stage = Path(root).resolve(), Path(stage).resolve()
    if stage.parent != root or not stage.name.startswith(PREFIX):
        raise ValueError('Expected a direct isolated X6 stage')
    manifest = read_json(stage / 'manifest.json')
    if manifest['version'] != VERSION or manifest['root'] != str(root) or manifest['stage'] != str(stage):
        raise ValueError('Stage identity mismatch')
    for name, digest in dict(manifest['tools_sha256'], **manifest['references_sha256']).items():
        if sha(safe(stage, name)) != digest:
            raise ValueError('Stage tool/reference changed: ' + name)
    if sha(stage / INPUT_NAME) != INPUT_SHA256:
        raise ValueError('Input seal differs')
    return manifest, read_json(stage / INPUT_NAME)


def prepare(root):
    root = Path(root).resolve()
    here = Path(__file__).resolve().parent
    if not root.is_dir() or sha(here / INPUT_NAME) != INPUT_SHA256:
        raise ValueError('Project root/input seal missing or changed')
    inputs = read_json(here / INPUT_NAME)
    if inputs['version'] != VERSION or inputs['methods'] != list(METHODS) or inputs['representative_seed'] != 42:
        raise ValueError('Measurement methods/seed differ')
    for name, expected in inputs['helper_sha256'].items():
        if sha(here / name) != expected:
            raise ValueError('Reviewed helper changed: ' + name)
    verify_root(root, inputs)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:10]
    stage = root / (PREFIX + stamp)
    stage.mkdir()
    for name in ('scripts', 'references', 'outputs', 'verification'):
        (stage / name).mkdir()
    tools = {}
    for name in TOOLS:
        shutil.copyfile(here / name, stage / 'scripts' / name)
        tools['scripts/' + name] = sha(stage / 'scripts' / name)
    shutil.copyfile(here / INPUT_NAME, stage / INPUT_NAME)
    tools[INPUT_NAME] = sha(stage / INPUT_NAME)
    references = {}
    for relative, record in inputs['references'].items():
        target = safe(stage, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe(root, record['root_path']), target)
        if sha(target) != record['sha256']:
            raise ValueError('Reference changed during copy')
        references[relative] = record['sha256']
    manifest = {'version': VERSION, 'root': str(root), 'stage': str(stage), 'input_sha256': INPUT_SHA256,
                'tools_sha256': tools, 'references_sha256': references, 'root_files_sha256': inputs['root_files_sha256'],
                'protocol': inputs['protocol'], 'tasks': [{'tag': tag, 'method': method} for tag in TAGS for method in METHODS]}
    write_json(stage / 'manifest.json', manifest)
    return stage


def telemetry():
    """Optional device-wide snapshot. Availability does not alter the experiment."""
    cmd = ['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm',
           '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        return {'available': result.returncode == 0, 'csv': result.stdout.strip()[:1000], 'exit_code': result.returncode}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'error': type(exc).__name__}


def feature_score(model, method, rows, adapter, torch, accounting):
    base = 'lbsp' if method == 'lbsp' else 'control'
    features = [adapter.prepare_history(base, row['_items'], row['_k']) for row in rows]
    seq = torch.tensor([part['seq'] for part in features], dtype=torch.long, device='cuda')
    tokens = torch.tensor([part['num_tokens'] for part in features], dtype=torch.long, device='cuda')
    groups = (torch.tensor([part['groups'] for part in features], dtype=torch.long, device='cuda')
              if base == 'lbsp' else None)
    raw = adapter.score_batch(model, base, seq, tokens, groups=groups, num_tokens=tokens if base == 'lbsp' else None)
    if not bool(torch.isfinite(raw).all()):
        raise ValueError('Nonfinite raw scores')
    accounting['actual_forward_rows'] += len(rows)
    accounting['forward_batch_calls'] += 1
    return raw


def member_rows(rows, seed, dataset, mapping, protocol, np):
    result = []
    for row in rows:
        rngseed = protocol.legacy_seed(dataset, row['sample_id'], seed)
        order = [int(i) for i in np.random.default_rng(rngseed).permutation(row['_k'])]
        items, events, _ = protocol.apply_last_bucket(row['_raw'], row['_events'], None, row['_k'], order)
        result.append(dict(row, _raw=items, _events=events, _items=[mapping[item] for item in items]))
    return result


def ranked_output(scores, rows, torch):
    """Rank/mask on GPU and transfer only fixed ranks+indices to host."""
    scores[:, 0] = -torch.inf
    targets = torch.tensor([row['_target'] for row in rows], dtype=torch.long, device='cuda')
    for index, row in enumerate(rows):
        seen = set(row['_items']) - {row['_target']}
        scores[index, list(seen)] = -torch.inf
    target_scores = scores.gather(1, targets[:, None])
    rank = (scores > target_scores).sum(1) + 1
    top = scores.topk(10, dim=1).indices
    return torch.cat((rank[:, None], top), dim=1).cpu()


def panel(model, method, rows, dataset, mapping, adapter, protocol, np, torch):
    """Entire panel or warmup subset; disk and serialization stay outside."""
    outputs = []
    accounting = {'actual_forward_rows': 0, 'forward_batch_calls': 0, 'cache_hits': 0}
    for start in range(0, len(rows), 256):
        batch = rows[start:start + 256]
        if method == 'ensemble':
            summed = None
            for seed in PERMS:
                candidate = member_rows(batch, seed, dataset, mapping, protocol, np)
                raw = feature_score(model, 'control', candidate, adapter, torch, accounting).to(torch.float64)
                summed = raw if summed is None else summed + raw
            scores = summed / 5.0
        else:
            scores = feature_score(model, method, batch, adapter, torch, accounting)
        outputs.append(ranked_output(scores, batch, torch))
    return outputs, accounting


def predictions_from_host(host_batches, rows, inverse, torch):
    # This conversion and JSON serialization run after the timed interval.
    arrays = torch.cat(host_batches).tolist()
    if len(arrays) != len(rows):
        raise ValueError('Result panel length differs')
    result = []
    for row, values in zip(rows, arrays):
        rank, top = int(values[0]), [int(index) for index in values[1:]]
        seen = set(row['_items']) - {row['_target']}
        if len(top) != 10 or len(set(top)) != 10 or 0 in top or set(top) & seen:
            raise ValueError('Candidate masking/top10 mismatch')
        result.append({'sample_id': row['sample_id'], 'user_id': row['user_id'], 'target_item': row['target_item'],
                       'rank': rank, 'top10_item_idx': json.dumps(top),
                       'top10_raw_item_id': json.dumps([inverse[index] for index in top])})
    return result


def child(root, stage, tag, method):
    manifest, inputs = verify_stage(root, stage)
    plan, run = inputs['datasets'][tag], inputs['datasets'][tag]['runs'][method]
    verify_root(root, inputs, run['required_root_files'])
    adapter = load_module(stage / 'scripts/x3_inference_adapter.py', 'x6_adapter')
    protocol = load_module(stage / 'scripts/x3_inference_protocol.py', 'x6_perm_protocol')
    helpers = load_module(stage / 'scripts/x3_core_helpers.py', 'x6_core_helpers')
    mapping, inverse, original, views, labels = helpers.load_panels(root, stage, plan, run)
    affected_count = sum(row['affected'] for row in labels)
    normalized_map = [[raw, index] for raw, index in sorted(mapping.items(), key=lambda pair: pair[1])]
    normalized_hash = hashlib.sha256(json.dumps(normalized_map, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    if normalized_hash != plan['normalized_mapping_sha256']:
        raise ValueError('Item-ID catalog mapping differs across methods')
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the cost experiment')
    if np.random.default_rng(0).bit_generator.__class__.__name__ != 'PCG64':
        raise RuntimeError('Frozen permutation protocol requires NumPy PCG64')
    # First-group item/event input equality is checked before any timing.
    for seed in PERMS:
        generated = member_rows(original, seed, plan['dataset'], mapping, protocol, np)
        if any(a['_raw'] != b['_raw'] or a['_events'] != b['_events'] for a, b in zip(generated, views[seed])):
            raise ValueError('Legacy first-group permutation does not reproduce frozen inputs')
    del generated, views, labels
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = safe(root, run['model_dir'] + '/' + run['checkpoint'])
    checkpoint_before = sha(checkpoint)
    if run.get('checkpoint_sha256') and checkpoint_before != run['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs from audited X3 inference result')
    value = torch.load(checkpoint, map_location='cpu', weights_only=True)
    expected = {'model_name': 'RecentBucketSetR1SASRec' if method == 'lbsp' else 'SASRecControl',
                'dataset': plan['dataset'], 'seed': 42, 'epoch': run['best_epoch'], 'num_items': len(mapping),
                'max_len': 50, 'hidden_size': 64, 'num_heads': 2, 'num_blocks': 2, 'dropout': 0.2}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError('Checkpoint model/dataset/seed/epoch/architecture mismatch')
    definition = load_module(stage / 'scripts/x6_model_definitions.py', 'x6_models')
    klass = definition.RecentBucketSetR1SASRec if method == 'lbsp' else definition.SASRecControl
    model = klass(num_items=len(mapping), max_len=50, hidden_size=64, num_heads=2, num_blocks=2, dropout=0.2).cuda()
    model.load_state_dict(value['model_state_dict'], strict=True)
    del value
    model.eval()
    output = stage / 'outputs' / (tag + '_' + method)
    output.mkdir(exist_ok=False)
    counts_expected = {'actual_forward_rows': len(original) * (5 if method == 'ensemble' else 1),
                       'forward_batch_calls': math.ceil(len(original) / 256) * (5 if method == 'ensemble' else 1),
                       'cache_hits': 0}
    warmup_n = min(512, len(original))
    repeats = []
    reference = safe(stage, run['reference_predictions'])
    with torch.no_grad():
        reference_output, _ = panel(model, method, original, plan['dataset'], mapping, adapter, protocol, np, torch)
        reference_before_timing = helpers.reproduce(predictions_from_host(reference_output, original, inverse, torch), reference)
        del reference_output
        warmup_output, _ = panel(model, method, original[:warmup_n], plan['dataset'], mapping, adapter, protocol, np, torch)
        del warmup_output
        torch.cuda.synchronize()
        for number in range(1, 6):
            environment = telemetry()
            torch.cuda.synchronize()
            baseline = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            host_batches, accounting = panel(model, method, original, plan['dataset'], mapping, adapter, protocol, np, torch)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            peak = int(torch.cuda.max_memory_allocated())
            reserved = int(torch.cuda.max_memory_reserved())
            if accounting != counts_expected or not math.isfinite(seconds) or seconds <= 0 or peak < baseline:
                raise ValueError('Measurement count/timing/memory mismatch')
            # The reference comparison is intentionally outside the clock.
            predictions = predictions_from_host(host_batches, original, inverse, torch)
            proof = helpers.reproduce(predictions, reference)
            repeats.append({'repeat': number, 'seconds': seconds, 'samples': len(original),
                            'seconds_per_user': seconds / len(original), 'baseline_allocated_bytes': baseline,
                            'peak_allocated_bytes': peak, 'peak_increment_bytes': peak - baseline,
                            'peak_reserved_bytes': reserved, 'accounting': accounting,
                            'matched_reference': proof, 'telemetry_before': environment})
            del host_batches, predictions
            print('X6_REPEAT=' + tag + ' ' + method + ' ' + str(number) + ' seconds=' + str(seconds), flush=True)
    checkpoint_after = sha(checkpoint)
    if checkpoint_before != checkpoint_after:
        raise ValueError('Checkpoint changed during benchmark')
    verify_root(root, inputs, run['required_root_files'])
    verify_stage(root, stage)
    times = [row['seconds'] for row in repeats]
    peaks = [row['peak_allocated_bytes'] for row in repeats]
    result = {'version': VERSION, 'status': 'BENCHMARK_COMPLETED', 'tag': tag, 'method': method, 'seed': 42,
              'input_sha256': INPUT_SHA256, 'manifest_sha256': sha(stage / 'manifest.json'),
              'tools_sha256': manifest['tools_sha256'], 'protocol': inputs['protocol'],
              'environment': {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
                              'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
                              'device_count': torch.cuda.device_count(), 'device_index': torch.cuda.current_device(),
                              'device_capability': list(torch.cuda.get_device_capability(0)),
                              'device_uuid': str(getattr(torch.cuda.get_device_properties(0), 'uuid', '')) or None,
                              'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                              'cuda_device_order': os.environ.get('CUDA_DEVICE_ORDER'),
                              'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                              'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
                              'allow_tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
                              'allow_tf32_cudnn': bool(torch.backends.cudnn.allow_tf32), 'telemetry_after': telemetry()},
              'reference_path': run['reference_predictions'], 'reference_sha256': sha(reference),
              'reference_before_timing': reference_before_timing,
              'normalized_mapping_sha256': normalized_hash,
              'checkpoint_before_sha256': checkpoint_before, 'checkpoint_after_sha256': checkpoint_after,
              'warmup_batches': 2, 'warmup_samples': warmup_n, 'batch_size': 256, 'cache_enabled': False,
              'training_started': False, 'optimizer_steps': 0, 'originals_unchanged': True,
              'samples': len(original), 'affected': affected_count,
              'repeats': repeats, 'summary': {'seconds_mean': statistics.mean(times),
              'seconds_sample_sd': statistics.stdev(times), 'seconds_per_user_mean': statistics.mean(times) / len(original),
              'peak_allocated_bytes_max': max(peaks)}}
    write_json(output / 'benchmark.json', result)
    completion = {'version': VERSION, 'status': 'COMPLETED', 'tag': tag, 'method': method, 'seed': 42,
                  'input_sha256': INPUT_SHA256, 'manifest_sha256': sha(stage / 'manifest.json'),
                  'benchmark_sha256': sha(output / 'benchmark.json'), 'training_started': False,
                  'checkpoint_sha256': checkpoint_after, 'originals_unchanged': True}
    write_json(stage / 'verification' / ('completed_' + tag + '_' + method + '.json'), completion)
    print('X6_INFERENCE_JOB_COMPLETED=' + tag + '_' + method, flush=True)


def verify_completion(stage, tag, method):
    manifest, inputs = verify_stage(Path(read_json(stage / 'manifest.json')['root']), stage)
    proof = stage / 'verification' / ('completed_' + tag + '_' + method + '.json')
    value = read_json(proof)
    expected = {'version': VERSION, 'status': 'COMPLETED', 'tag': tag, 'method': method, 'seed': 42,
                'input_sha256': INPUT_SHA256, 'manifest_sha256': sha(stage / 'manifest.json'),
                'training_started': False, 'originals_unchanged': True}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError('Completion proof identity mismatch')
    result_path = stage / 'outputs' / (tag + '_' + method) / 'benchmark.json'
    if sha(result_path) != value.get('benchmark_sha256'):
        raise ValueError('Benchmark result changed')
    result = read_json(result_path)
    plan = inputs['datasets'][tag]
    if (result.get('status') != 'BENCHMARK_COMPLETED' or result.get('tag') != tag or result.get('method') != method
            or result.get('input_sha256') != INPUT_SHA256 or result.get('manifest_sha256') != sha(stage / 'manifest.json')
            or result.get('tools_sha256') != manifest['tools_sha256'] or result.get('protocol') != inputs['protocol']
            or result.get('checkpoint_before_sha256') != value.get('checkpoint_sha256')
            or result.get('checkpoint_after_sha256') != value.get('checkpoint_sha256')
            or result.get('reference_path') != plan['runs'][method]['reference_predictions']
            or result.get('reference_sha256') != manifest['references_sha256'][result['reference_path']]
            or result.get('normalized_mapping_sha256') != plan['normalized_mapping_sha256']
            or result.get('reference_before_timing', {}).get('status') != 'PASS'
            or result.get('samples') != plan['expected_samples']
            or result.get('affected') != plan['expected_affected'] or len(result.get('repeats', [])) != 5
            or result.get('warmup_batches') != 2 or result.get('training_started') is not False):
        raise ValueError('Benchmark result scope/count failed')
    prior = result['reference_before_timing']
    if (prior.get('reference_path') != str(safe(stage, result['reference_path']))
            or prior.get('reference_sha256') != result['reference_sha256'] or prior.get('samples') != result['samples']
            or any(prior.get(key) != 0 for key in ('rank_mismatches', 'ordered_top10_mismatches', 'raw_top10_mismatches'))):
        raise ValueError('Pre-timing full-panel reference reproduction failed')
    for index, row in enumerate(result['repeats'], 1):
        expected_rows = plan['expected_samples'] * (5 if method == 'ensemble' else 1)
        expected_batches = math.ceil(plan['expected_samples'] / 256) * (5 if method == 'ensemble' else 1)
        match_proof = row['matched_reference']
        if (row['repeat'] != index or not math.isfinite(row['seconds']) or row['seconds'] <= 0
                or row['samples'] != plan['expected_samples'] or row['matched_reference']['status'] != 'PASS'
                or match_proof['reference_path'] != str(safe(stage, result['reference_path']))
                or match_proof['reference_sha256'] != result['reference_sha256'] or match_proof['samples'] != row['samples']
                or any(match_proof.get(key) != 0 for key in ('rank_mismatches', 'ordered_top10_mismatches', 'raw_top10_mismatches'))
                or not math.isclose(row['seconds_per_user'], row['seconds'] / row['samples'], rel_tol=1e-12)
                or row['accounting'] != {'actual_forward_rows': expected_rows, 'forward_batch_calls': expected_batches, 'cache_hits': 0}
                or row['peak_increment_bytes'] != row['peak_allocated_bytes'] - row['baseline_allocated_bytes']
                or row['peak_increment_bytes'] < 0):
            raise ValueError('Repeated timing or reproduction failed')
    return proof


def run(root):
    stage = prepare(root)
    root = Path(root).resolve()
    print('X6_INFERENCE_STAGE=' + str(stage), flush=True)
    path = stage / 'verification/driver.json'
    state = {'version': VERSION, 'stage': str(stage), 'status': 'RUNNING', 'jobs': []}
    write_json(path, state)
    result, process, fingerprint = 1, None, None
    try:
        for tag in TAGS:
            for method in METHODS:
                command = [sys.executable, '-u', str(stage / 'scripts/run_x6_inference.py'), '--root', str(root), '--run',
                           '--stage', str(stage), '--dataset', tag, '--method', method]
                logfile = stage / 'verification' / (tag + '_' + method + '.log')
                record = {'tag': tag, 'method': method, 'seed': 42, 'command': command, 'log': str(logfile),
                          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
                state['jobs'].append(record)
                write_json(path, state)
                with logfile.open('x', encoding='utf-8') as handle:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding='utf-8', errors='replace')
                    for line in process.stdout:
                        handle.write(line)
                        handle.flush()
                        print(line, end='', flush=True)
                    record['exit_code'] = process.wait()
                process = None
                if record['exit_code']:
                    raise RuntimeError('Benchmark failed: ' + tag + ' ' + method)
                proof = verify_completion(stage, tag, method)
                environment = read_json(stage / 'outputs' / (tag + '_' + method) / 'benchmark.json')['environment']
                current = {key: environment.get(key) for key in ('python', 'torch', 'numpy', 'cuda', 'device', 'device_count',
                    'device_index', 'device_capability', 'device_uuid', 'visible_devices', 'cuda_device_order',
                    'cublas_workspace_config', 'deterministic_algorithms', 'cudnn_deterministic',
                    'allow_tf32_matmul', 'allow_tf32_cudnn')}
                if fingerprint is None:
                    fingerprint = current
                    state['environment_fingerprint'] = fingerprint
                elif current != fingerprint:
                    raise ValueError('GPU/runtime environment differs across benchmark jobs')
                record.update(proof=str(proof), proof_sha256=sha(proof), log_sha256=sha(logfile),
                              finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                write_json(path, state)
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
        print('X6_INFERENCE_ERROR=' + state['error'], flush=True)
    finally:
        state['status'] = 'ALL_SIXTEEN_INFERENCE_BENCHMARKS_COMPLETED' if result == 0 else 'STOPPED_INCOMPLETE'
        write_json(path, state)
        print('X6_INFERENCE_DRIVER_STATUS=' + state['status'], flush=True)
        collected = subprocess.run([sys.executable, str(stage / 'scripts/collect_x6_inference.py'), '--stage', str(stage)])
        if collected.returncode:
            result = collected.returncode
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--stage', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--dataset', choices=TAGS, help=argparse.SUPPRESS)
    parser.add_argument('--method', choices=METHODS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.run:
        print('PLAN_ONLY_NO_FILES_OR_GPU: 4 datasets x4 methods seed42; 2 warmup batches; 5 full-panel inference timings; GPU rank/Top10+D2H included, disk/model loading excluded; no training.')
        return 0
    if args.stage:
        if args.dataset is None or args.method is None:
            parser.error('Internal stage requires dataset and method')
        child(args.root.resolve(), args.stage.resolve(), args.dataset, args.method)
        return 0
    return run(args.root)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('ERROR: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
