#!/usr/bin/env python3
"""Collect X6 inference benchmark metadata without reading data or weights."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import statistics
import sys
import uuid
import zipfile

VERSION = 'x6_inference_v1'
INPUT_NAME = 'x6_inference_inputs_20261027.json'
INPUT_SHA256 = 'd10cf611d84694624526c78cfafa372752d05d8679e40437c813b4ba3e5c2a32'
TAGS = ('ml1m', 'beauty', 'sports', 'toys')
METHODS = ('control', 'shuffle', 'ensemble', 'lbsp')
TOOLS = ('run_x6_inference.py', 'collect_x6_inference.py', 'x3_inference_adapter.py',
         'x3_inference_protocol.py', 'x3_core_helpers.py', 'x6_model_definitions.py', 'x6_cost_protocol.py')
FINGERPRINT = ('python', 'torch', 'numpy', 'cuda', 'device', 'device_count', 'device_index',
               'device_capability', 'device_uuid', 'visible_devices', 'cuda_device_order',
               'cublas_workspace_config', 'deterministic_algorithms', 'cudnn_deterministic',
               'allow_tf32_matmul', 'allow_tf32_cudnn')
FILE_LIMIT, TOTAL_LIMIT = 64 * 1024 ** 2, 512 * 1024 ** 2


def absolute_plain(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected an absolute path without ..')
    for component in (path,) + tuple(path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Symbolic link/reparse point rejected')
    return path.resolve(strict=False)


def relative_path(value):
    if (not isinstance(value, str) or not value or '\\' in value or '\x00' in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(x in ('', '.', '..') or x.startswith('.') for x in value.split('/'))):
        raise ValueError('Unsafe relative path')
    return value


def safe(root, relative):
    path = root
    for part in relative_path(relative).split('/'):
        path = path / part
        if path.is_symlink() or (path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Symbolic link/reparse point rejected')
    path.resolve(strict=False).relative_to(root)
    return path


def signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def read_stable(root, relative):
    if PurePosixPath(relative).suffix.lower() in ('.pt', '.pth', '.ckpt') or relative.startswith(('data/', 'processed/')):
        raise ValueError('Checkpoint/interaction access forbidden')
    path = safe(root, relative)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > FILE_LIMIT:
        raise ValueError('Nonregular or oversized file')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    parts, digest, size = [], hashlib.sha256(), 0
    with os.fdopen(os.open(str(path), flags), 'rb') as handle:
        if signature(before) != signature(os.fstat(handle.fileno())):
            raise ValueError('File changed before reading')
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > FILE_LIMIT:
                raise ValueError('File size limit exceeded')
            digest.update(block)
            parts.append(block)
        after = os.fstat(handle.fileno())
    if signature(before) != signature(after) or signature(before) != signature(safe(root, relative).lstat()) or size != before.st_size:
        raise ValueError('File changed while reading')
    return b''.join(parts), digest.hexdigest()


def json_object(blob):
    obj = json.loads(blob.decode('utf-8-sig'))
    if not isinstance(obj, dict):
        raise ValueError('Expected JSON object')
    return obj


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def integer(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not an integer')
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError('Invalid integer')
    return int(number)


def number(value, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a measurement')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError('Invalid measurement')
    return value


def near(a, b):
    return math.isclose(number(a), number(b), rel_tol=1e-10, abs_tol=1e-12)


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


class Collector:
    def __init__(self, stage):
        self.stage = absolute_plain(stage)
        if not self.stage.name.startswith('kais_x6_inference_v1_'):
            raise ValueError('Unexpected X6 stage prefix')
        verification = safe(self.stage, 'verification')
        if not verification.is_dir():
            raise ValueError('Verification directory missing')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
        self.out = verification / ('x6_inference_review_' + stamp)
        self.out.mkdir(exist_ok=False)
        self.files, self.copies, self.warnings, self.integrity_failures = [], {}, [], []
        self.total, self.runs = 0, {}
        self.driver_check = {'status': 'NOT_CHECKED', 'errors': []}

    def copy(self, relative, expected=None):
        entry = {'path': relative, 'status': 'unknown'}
        try:
            blob, value = read_stable(self.stage, relative)
            if expected is not None and value != expected:
                raise ValueError('SHA256 mismatch')
            if self.total + len(blob) > TOTAL_LIMIT:
                raise ValueError('Total size limit exceeded')
            target = safe(self.out, 'snapshot/' + relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(blob)
            self.copies[relative] = blob
            self.total += len(blob)
            entry.update(status='copied', bytes=len(blob), sha256=value)
            return blob
        except (OSError, ValueError) as exc:
            entry.update(status='missing' if isinstance(exc, FileNotFoundError) else 'rejected_or_changed', error=str(exc)[:350])
            self.warnings.append(entry['status'] + ': ' + relative)
            if expected is not None and isinstance(exc, ValueError):
                self.integrity_failures.append('Pinned file rejected/changed: ' + relative)
            return None
        finally:
            self.files.append(entry)

    def sources(self):
        blob, self.manifest_sha = read_stable(self.stage, 'manifest.json')
        self.manifest = manifest = json_object(blob)
        blob, value = read_stable(self.stage, INPUT_NAME)
        if value != INPUT_SHA256:
            raise ValueError('Reviewed input SHA mismatch')
        self.inputs = inputs = json_object(blob)
        self.root = absolute_plain(manifest['root'])
        if (self.stage.parent != self.root or manifest.get('stage') != str(self.stage) or manifest.get('version') != VERSION
                or manifest.get('input_sha256') != INPUT_SHA256 or manifest.get('protocol') != inputs['protocol']
                or manifest.get('root_files_sha256') != inputs['root_files_sha256'] or inputs.get('version') != VERSION
                or inputs.get('methods') != list(METHODS) or inputs.get('representative_seed') != 42
                or set(inputs.get('datasets', {})) != set(TAGS)):
            raise ValueError('Stage/data/protocol identity differs')
        expected_tasks = [{'tag': tag, 'method': method} for tag in TAGS for method in METHODS]
        if manifest.get('tasks') != expected_tasks:
            raise ValueError('Expected 16 fixed dataset/method tasks')
        inventory = {'scripts/' + name for name in TOOLS} | {INPUT_NAME}
        if set(manifest['tools_sha256']) != inventory or manifest['tools_sha256'][INPUT_NAME] != INPUT_SHA256:
            raise ValueError('Expected exactly eight tool/input files')
        helpers = inputs['helper_sha256']
        if set(helpers) != set(TOOLS) - {'run_x6_inference.py', 'collect_x6_inference.py'}:
            raise ValueError('Unexpected non-circular helper pins')
        for name, expected in helpers.items():
            if manifest['tools_sha256']['scripts/' + name] != expected:
                raise ValueError('Reviewed helper differs')
        for name, expected in inputs['root_files_sha256'].items():
            relative_path(name)
            if not valid_hash(expected):
                raise ValueError('Invalid frozen root file SHA')
        references = inputs['references']
        if manifest['references_sha256'] != {p: record['sha256'] for p, record in references.items()}:
            raise ValueError('Stage reference closure differs')
        required_refs = set()
        for tag in TAGS:
            plan = inputs['datasets'][tag]
            if set(plan['runs']) != set(METHODS) or integer(plan['expected_samples']) < 2 or integer(plan['expected_affected']) < 0:
                raise ValueError('Dataset run/sample contract invalid')
            if not valid_hash(plan.get('normalized_mapping_sha256')):
                raise ValueError('Dataset normalized mapping SHA invalid')
            for method in METHODS:
                run = plan['runs'][method]
                relative_path(run['model_dir'])
                relative_path(run['model_dir'] + '/' + run['checkpoint'])
                if integer(run['best_epoch']) < 1 or not set(run['required_root_files']) <= set(inputs['root_files_sha256']):
                    raise ValueError('Run metadata/root hash inventory invalid')
                required_refs.update(run[key] for key in ('reference_config', 'reference_map', 'reference_predictions'))
                if run.get('checkpoint_sha256') is not None and not valid_hash(run['checkpoint_sha256']):
                    raise ValueError('Invalid checkpoint SHA syntax')
            a, b = plan['runs']['control'], plan['runs']['ensemble']
            if a['model_dir'] != b['model_dir'] or a['checkpoint'] != b['checkpoint'] or a.get('checkpoint_sha256') != b.get('checkpoint_sha256'):
                raise ValueError('Ensemble and Control must use the same checkpoint')
        if not required_refs <= set(references):
            raise ValueError('Missing referenced saved model metadata')
        for relative, expected in manifest['tools_sha256'].items():
            if self.copy(relative, expected) is None:
                raise ValueError('Tool/input missing or rejected: ' + relative)
        executing = Path(__file__).absolute()
        _, executing_sha = read_stable(absolute_plain(executing.parent), executing.name)
        if executing_sha != manifest['tools_sha256']['scripts/collect_x6_inference.py']:
            raise ValueError('Executing collector differs from stage collector')
        for relative, record in references.items():
            name = PurePosixPath(relative).name
            if (not relative_path(relative).startswith('references/') or PurePosixPath(relative).suffix not in ('.json', '.csv')
                    or name in ('train.csv', 'test.csv', 'val.csv', 'r1_samples.csv', 'file_order.csv')):
                raise ValueError('Reference is not whitelisted model metadata')
            if inputs['root_files_sha256'].get(record['root_path']) != record['sha256']:
                raise ValueError('Reference source not pinned')
            if self.copy(relative, record['sha256']) is None:
                raise ValueError('Saved model reference missing or rejected')
        if self.copy('manifest.json', self.manifest_sha) is None:
            raise ValueError('Manifest changed')
        self.source_check = {'status': 'PASS', 'stage_tools': len(inventory), 'saved_references': len(references),
                             'root_file_hash_records_not_read': len(inputs['root_files_sha256'])}

    def proof(self, proof, reference, expected_sha, n):
        if (proof.get('status') != 'PASS' or proof.get('reference_path') != str(self.stage / reference)
                or proof.get('reference_sha256') != expected_sha or integer(proof.get('samples')) != n
                or any(integer(proof.get(k)) != 0 for k in ('rank_mismatches', 'ordered_top10_mismatches', 'raw_top10_mismatches'))):
            raise ValueError('Prior prediction reproduction proof differs')

    def mapping_sha(self, relative, expected_items):
        blob = self.copies.get(relative)
        if blob is None:
            raise ValueError('Reference map missing')
        rows = list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig'), newline='')))
        mapping = {row['raw_item_id']: integer(row['item_idx']) for row in rows}
        if (len(rows) != len(mapping) or len(mapping) != expected_items
                or set(mapping.values()) != set(range(1, expected_items + 1))):
            raise ValueError('Reference map is not a complete bijection')
        ordered = [[raw, index] for raw, index in sorted(mapping.items(), key=lambda pair: pair[1])]
        return hashlib.sha256(json.dumps(ordered, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()

    def reference_predictions(self, reference, mapping_reference, n, expected_items):
        maps = list(csv.DictReader(io.StringIO(self.copies[mapping_reference].decode('utf-8-sig'), newline='')))
        mapping = {integer(row['item_idx']): row['raw_item_id'] for row in maps}
        raw_catalog = set(mapping.values())
        rows = list(csv.DictReader(io.StringIO(self.copies[reference].decode('utf-8-sig'), newline='')))
        if len(rows) != n or len({row['sample_id'] for row in rows}) != n:
            raise ValueError('Reference prediction panel population differs')
        for row in rows:
            rank = integer(row.get('target_rank', row.get('rank')))
            top = [integer(value) for value in json.loads(row['top10_item_idx'])]
            raw = [str(value) for value in json.loads(row['top10_raw_item_id'])]
            if (not row['sample_id'] or not row['user_id'] or row['target_item'] not in raw_catalog
                    or not 1 <= rank <= expected_items or len(top) != 10 or len(set(top)) != 10
                    or any(value not in mapping for value in top) or [mapping[value] for value in top] != raw):
                raise ValueError('Reference prediction identity/rank/Top10 invalid')
        return len(rows)

    def run(self, tag, method):
        key = tag + '_' + method
        run = {'status': 'MISSING', 'tag': tag, 'method': method, 'errors': []}
        self.runs[key] = run
        proof_relative = 'verification/completed_' + key + '.json'
        proof_bytes = self.copy(proof_relative)
        if proof_bytes is None:
            run['errors'].append('Completion proof missing')
            return
        try:
            completion = json_object(proof_bytes)
            expected = {'version': VERSION, 'status': 'COMPLETED', 'tag': tag, 'method': method, 'seed': 42,
                        'input_sha256': INPUT_SHA256, 'manifest_sha256': self.manifest_sha,
                        'training_started': False, 'originals_unchanged': True}
            if any(completion.get(k) != v for k, v in expected.items()):
                self.integrity_failures.append('Completion identity/provenance mismatch: ' + key)
                raise ValueError('Completion identity/provenance mismatch')
            if not valid_hash(completion.get('checkpoint_sha256')) or not valid_hash(completion.get('benchmark_sha256')):
                raise ValueError('Invalid checkpoint/benchmark SHA syntax')
            benchmark_relative = 'outputs/' + key + '/benchmark.json'
            blob = self.copy(benchmark_relative, completion['benchmark_sha256'])
            if blob is None:
                raise ValueError('Missing or changed benchmark')
            result = json_object(blob)
            plan, source = self.inputs['datasets'][tag], self.inputs['datasets'][tag]['runs'][method]
            n = integer(plan['expected_samples'])
            reference = source['reference_predictions']
            reference_sha = self.manifest['references_sha256'][reference]
            if (result.get('version') != VERSION or result.get('status') != 'BENCHMARK_COMPLETED'
                    or result.get('tag') != tag or result.get('method') != method or integer(result.get('seed')) != 42
                    or result.get('input_sha256') != INPUT_SHA256 or result.get('manifest_sha256') != self.manifest_sha
                    or result.get('tools_sha256') != self.manifest['tools_sha256'] or result.get('protocol') != self.inputs['protocol']):
                self.integrity_failures.append('Benchmark identity/provenance mismatch: ' + key)
                raise ValueError('Benchmark identity/provenance mismatch')
            checkpoint = completion['checkpoint_sha256']
            if (result.get('checkpoint_before_sha256') != checkpoint or result.get('checkpoint_after_sha256') != checkpoint
                    or (source.get('checkpoint_sha256') and source['checkpoint_sha256'] != checkpoint)):
                self.integrity_failures.append('Checkpoint SHA proof differs: ' + key)
                raise ValueError('Checkpoint SHA proof differs')
            if (result.get('reference_path') != reference or result.get('reference_sha256') != reference_sha
                    or integer(result.get('samples')) != n or integer(result.get('affected')) != integer(plan['expected_affected'])
                    or integer(result.get('warmup_batches')) != 2 or integer(result.get('warmup_samples')) != min(512, n)
                    or integer(result.get('batch_size')) != 256 or result.get('cache_enabled') is not False
                    or result.get('training_started') is not False or integer(result.get('optimizer_steps')) != 0
                    or result.get('originals_unchanged') is not True):
                raise ValueError('Benchmark scope, panel, warmup or reference differs')
            if (result.get('normalized_mapping_sha256') != plan['normalized_mapping_sha256']
                    or self.mapping_sha(source['reference_map'], integer(plan['expected_items'])) != plan['normalized_mapping_sha256']):
                raise ValueError('Normalized model catalog mapping differs')
            checked_reference_samples = self.reference_predictions(reference, source['reference_map'], n, integer(plan['expected_items']))
            self.proof(result['reference_before_timing'], reference, reference_sha, n)
            environment = result['environment']
            if (not set(FINGERPRINT) <= set(environment)
                    or any(not isinstance(environment.get(k), str) or not environment[k] for k in ('python', 'torch', 'numpy', 'cuda', 'device'))):
                raise ValueError('Missing runtime/device identity')
            if integer(environment['device_count']) < 1 or integer(environment['device_index']) != 0:
                raise ValueError('Unexpected CUDA device count/index')
            capability = environment['device_capability']
            if not isinstance(capability, list) or len(capability) != 2 or any(integer(k) < 0 for k in capability):
                raise ValueError('Invalid device capability')
            if environment.get('visible_devices') is not None and not isinstance(environment['visible_devices'], str):
                raise ValueError('Invalid visible devices')
            if environment.get('device_uuid') is not None and not isinstance(environment['device_uuid'], str):
                raise ValueError('Invalid GPU UUID')
            for field in ('cuda_device_order', 'cublas_workspace_config'):
                if environment.get(field) is not None and not isinstance(environment[field], str):
                    raise ValueError('Invalid GPU environment variable')
            if (not isinstance(environment.get('deterministic_algorithms'), bool)
                    or environment.get('cudnn_deterministic') is not True
                    or environment.get('allow_tf32_matmul') is not False or environment.get('allow_tf32_cudnn') is not False
                    or not isinstance(environment.get('telemetry_after'), dict)):
                raise ValueError('GPU precision/determinism/telemetry contract differs')
            expected_accounting = {'actual_forward_rows': n * (5 if method == 'ensemble' else 1),
                'forward_batch_calls': math.ceil(n / 256) * (5 if method == 'ensemble' else 1), 'cache_hits': 0}
            repeats = result.get('repeats')
            if not isinstance(repeats, list) or len(repeats) != 5:
                raise ValueError('Expected five timing repetitions')
            times, peaks = [], []
            for index, row in enumerate(repeats, 1):
                seconds = number(row.get('seconds'), positive=True)
                baseline = integer(row['baseline_allocated_bytes'])
                peak, increment, reserved = map(integer, (row['peak_allocated_bytes'], row['peak_increment_bytes'], row['peak_reserved_bytes']))
                if (integer(row['repeat']) != index or integer(row['samples']) != n or row['accounting'] != expected_accounting
                        or not near(row['seconds_per_user'], seconds / n) or baseline < 0 or peak < baseline
                        or increment != peak - baseline or reserved < peak or not isinstance(row['telemetry_before'], dict)):
                    raise ValueError('Repeat time, workload, memory or telemetry invalid')
                self.proof(row['matched_reference'], reference, reference_sha, n)
                times.append(seconds)
                peaks.append(peak)
            summary = {'seconds_mean': statistics.mean(times), 'seconds_sample_sd': statistics.stdev(times),
                       'seconds_per_user_mean': statistics.mean(times) / n, 'peak_allocated_bytes_max': max(peaks)}
            for name, expected_value in summary.items():
                if not near(result['summary'][name], expected_value):
                    raise ValueError('Summary differs from five repetitions: ' + name)
            run.update(status='VALID_SAVED_BENCHMARK', samples=n, affected=integer(result['affected']),
                       checkpoint_sha256=checkpoint, benchmark_sha256=digest(blob), summary=summary,
                       repeats=repeats, environment=environment, normalized_mapping_sha256=plan['normalized_mapping_sha256'],
                       reference_prediction_samples_checked=checked_reference_samples,
                       gpu_uuid_evidence=('reported_same_across_jobs' if re.fullmatch(r'(?:GPU|MIG)-[A-Za-z0-9-]+', environment.get('device_uuid') or '')
                                          else 'not_available_or_unparseable_device_name_visible_index_only'))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            run['status'] = 'INVALID'
            run['errors'].append(type(exc).__name__ + ': ' + str(exc)[:450])

    def driver(self):
        blob = self.copies.get('verification/driver.json')
        if blob is None:
            self.driver_check = {'status': 'MISSING', 'errors': ['Driver record missing']}
            return
        checks, errors = {}, []
        try:
            value = json_object(blob)
            jobs = value['jobs']
            expected = [(tag, method) for tag in TAGS for method in METHODS]
            checks['stage_version_status'] = (value.get('stage') == str(self.stage) and value.get('version') == VERSION
                and value.get('status') == 'ALL_SIXTEEN_INFERENCE_BENCHMARKS_COMPLETED')
            checks['job_count'] = len(jobs) == len(expected)
            last_end, python = None, None
            for i, job in enumerate(jobs):
                if i >= len(expected):
                    raise ValueError('Unexpected extra jobs')
                tag, method = expected[i]
                label = tag + '_' + method
                checks[label + '_identity'] = (job.get('tag'), job.get('method'), job.get('seed')) == (tag, method, 42)
                command = job['command']
                if python is None:
                    python = command[0]
                checks[label + '_command'] = command == [python, '-u', str(self.stage / 'scripts/run_x6_inference.py'), '--root',
                    str(self.root), '--run', '--stage', str(self.stage), '--dataset', tag, '--method', method]
                checks[label + '_exit'] = integer(job.get('exit_code')) == 0
                proof_path, log_path = 'verification/completed_' + label + '.json', 'verification/' + label + '.log'
                proof, log = self.copies.get(proof_path), self.copies.get(log_path)
                checks[label + '_proof'] = proof is not None and job.get('proof') == str(self.stage / proof_path) and job.get('proof_sha256') == digest(proof)
                checks[label + '_log'] = log is not None and job.get('log') == str(self.stage / log_path) and job.get('log_sha256') == digest(log)
                if proof is not None and job.get('proof_sha256') != digest(proof):
                    self.integrity_failures.append('Driver completion SHA mismatch: ' + label)
                if log is not None and job.get('log_sha256') != digest(log):
                    self.integrity_failures.append('Driver log SHA mismatch: ' + label)
                checks[label + '_saved_benchmark'] = self.runs.get(label, {}).get('status') == 'VALID_SAVED_BENCHMARK'
                if self.runs.get(label, {}).get('status') == 'VALID_SAVED_BENCHMARK':
                    environment = self.runs[label]['environment']
                    checks[label + '_environment'] = {k: environment.get(k) for k in FINGERPRINT} == value.get('environment_fingerprint')
                started = datetime.datetime.fromisoformat(job['started_utc'].replace('Z', '+00:00'))
                finished = datetime.datetime.fromisoformat(job['finished_utc'].replace('Z', '+00:00'))
                checks[label + '_time'] = (started.tzinfo is not None and finished.tzinfo is not None and started <= finished
                                            and (last_end is None or last_end <= started))
                last_end = finished
            for tag in TAGS:
                a = self.runs.get(tag + '_control', {}).get('checkpoint_sha256')
                b = self.runs.get(tag + '_ensemble', {}).get('checkpoint_sha256')
                checks[tag + '_shared_control_checkpoint'] = a is not None and a == b
            if not all(checks.values()):
                errors.append('Driver does not confirm all 16 serial jobs, reproduction proofs and common environment')
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append(type(exc).__name__ + ': ' + str(exc)[:400])
        self.driver_check = {'status': 'PASS' if not errors and all(checks.values()) else 'FAIL', 'checks': checks, 'errors': errors}

    def finish(self):
        complete = (len(self.runs) == 16 and all(r['status'] == 'VALID_SAVED_BENCHMARK' for r in self.runs.values())
                    and self.driver_check['status'] == 'PASS')
        status = 'INTEGRITY_FAILED_NO_BUNDLE' if self.integrity_failures else 'ALL_SIXTEEN_INFERENCE_COSTS_CHECKED' if complete else 'INCOMPLETE'
        report = {'status': status, 'stage': str(self.stage), 'input_sha256': INPUT_SHA256,
            'source_check': getattr(self, 'source_check', {'status': 'NOT_COMPLETED'}), 'run_checks': self.runs,
            'driver_check': self.driver_check, 'files': self.files, 'warnings': self.warnings,
            'integrity_failures': self.integrity_failures,
            'scope': ('Inference-only saved benchmark audit. All five measured full-panel times, per-user times, memory accounting and reproduction records are independently checked; '
                'raw interactions and checkpoint contents are not read/statted. Two batches warmup and one untimed whole-panel reproduction precede timing. '
                'Timing includes CPU feature/permutation preparation, H2D, forward, group1 score aggregation, candidate masking, rank/Top10 and fixed-size D2H; '
                'CSV/JSON/disk I/O, setup and checkpoint loading are excluded. One ensemble prediction is five views, not outer25. '
                'These five timing repetitions are not independent training seeds. Training epoch costs are not included or complete. '
                'Matching device properties/visible index does not prove a physical UUID if device_uuid is unavailable; telemetry is descriptive and does not certify exclusive GPU load.')}
        (self.out / 'x6_inference_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        with (self.out / 'REPORT_SHA256.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=('path', 'status', 'bytes', 'sha256', 'error'))
            writer.writeheader()
            writer.writerows(self.files)
        with (self.out / 'inference_cost_summary.csv').open('w', encoding='utf-8', newline='') as handle:
            fields = ('dataset', 'method', 'seed', 'samples', 'seconds_mean', 'seconds_sample_sd',
                      'seconds_per_user_mean', 'milliseconds_per_user_mean', 'peak_allocated_bytes_max')
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for key, run in self.runs.items():
                if run['status'] == 'VALID_SAVED_BENCHMARK':
                    writer.writerow(dict(dataset=run['tag'], method=run['method'], seed=42, samples=run['samples'],
                                         milliseconds_per_user_mean=run['summary']['seconds_per_user_mean'] * 1000, **run['summary']))
        lines = ['# PRIVATE X6 inference benchmark report', '', 'Status: **' + status + '**', '', report['scope'], '',
                 '| Dataset | Method | Mean seconds | Sample SD | Mean milliseconds/user | Peak allocated bytes |', '|---|---|---:|---:|---:|---:|']
        for run in self.runs.values():
            if run['status'] == 'VALID_SAVED_BENCHMARK':
                s = run['summary']
                lines.append('| %s | %s | %.8f | %.8f | %.8f | %d |' % (run['tag'], run['method'], s['seconds_mean'],
                    s['seconds_sample_sd'], 1000 * s['seconds_per_user_mean'], s['peak_allocated_bytes_max']))
        messages = self.integrity_failures + self.warnings + self.driver_check.get('errors', [])
        messages += [name + ': ' + msg for name, run in self.runs.items() for msg in run['errors']]
        lines += ['', '## Diagnostics', ''] + (['- ' + msg for msg in messages] or ['- No discrepancy found.'])
        (self.out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        bundle = None
        if not self.integrity_failures:
            bundle = self.out / 'x6_inference_results.zip'
            paths = sorted(p for p in self.out.rglob('*') if p.is_file())
            with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for path in paths:
                    if path.is_symlink():
                        raise ValueError('Output symlink rejected')
                    archive.write(path, path.relative_to(self.out).as_posix())
            value = hashlib.sha256()
            with bundle.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    value.update(chunk)
            bundle.with_name(bundle.name + '.sha256.txt').write_text(value.hexdigest() + '  ' + bundle.name + '\n', encoding='ascii')
        return {'status': status, 'report_dir': str(self.out), 'zip_path': str(bundle) if bundle else None,
                'exit_code': 2 if self.integrity_failures else 0 if complete else 1}


def collect(stage):
    collector = Collector(stage)
    try:
        collector.sources()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        collector.integrity_failures.append(type(exc).__name__ + ': ' + str(exc)[:420])
    if not collector.integrity_failures:
        collector.copy('verification/driver.json')
        for tag in TAGS:
            for method in METHODS:
                collector.copy('verification/' + tag + '_' + method + '.log')
                collector.run(tag, method)
        collector.driver()
    return collector.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.stage.expanduser())
    checks = Path(result['report_dir']) / 'x6_inference_checks.json'
    print('CHECKS=' + str(checks))
    print('SUMMARY=' + str(checks.with_name('SUMMARY.md')))
    print('X6_INFERENCE_REPORT_ZIP=' + (result['zip_path'] or 'NOT_CREATED'))
    print('STATUS=' + result['status'])
    value = json_object(checks.read_bytes())
    for message in value['integrity_failures']:
        print('INTEGRITY_FAILURE=' + message)
    for name, run in value['run_checks'].items():
        for message in run['errors']:
            print('RUN_FAILURE=' + name + ': ' + message)
    for message in value['driver_check'].get('errors', []):
        print('DRIVER_FAILURE=' + message)
    for name, passed in value['driver_check'].get('checks', {}).items():
        if not passed:
            print('DRIVER_FAILED_CHECK=' + name)
    return result['exit_code']


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print('COLLECTION_FAILED=' + type(exc).__name__ + ': ' + str(exc)[:400], file=sys.stderr)
        sys.exit(2)
