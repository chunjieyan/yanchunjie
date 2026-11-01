#!/usr/bin/env python3
"""Collect private nine-run Amazon LBSP training evidence without research execution.

Standard library only. Never reads interaction CSV contents or model weights.
Saved metadata and predictions are checked without opening checkpoint contents.
"""
import argparse
import ast
import csv
import datetime
import hashlib
import io
import json
import math
import statistics
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys
import uuid
import zipfile

INPUTS_FILE = Path(__file__).absolute().with_name('lbsp_amazon_train_inputs_20261023.json')
EXPECTED_INPUTS_SHA256 = 'aad9c9a404f04df9b355d6aa6b4ca0b26905d9763ee9fb3915c9f3e102682ff2'
TAGS = {'beauty': 'Amazon_Beauty', 'sports': 'Amazon_Sports', 'toys': 'Amazon_Toys'}
REFERENCE_FILES = ('config.json', 'metrics.json', 'training_log.csv', 'item_id_map.csv', 'test_per_sample_predictions.csv')
MANIFEST = 'manifests/lbsp_amazon_train_manifest.json'
SEEDS = (42, 2026, 3407)
GENERATED = ('scripts/model_definition.py',) + tuple('scripts/train_%s_lbsp.py' % t for t in TAGS)
FILE_LIMIT, TOTAL_LIMIT = 64 * 1024 ** 2, 512 * 1024 ** 2
# Exact source text from the frozen preparer SHA256
# 16842156bcb62059e96c73c96aa97aab640b26a2c713c4dcd5935c75a0426f68.
# Parse reference and emitted code in the SAME interpreter; never execute it.
NORMALIZER_REFERENCE_SOURCE = r'''
def normalize_eval_records(records):
    """Only add missing aliases; reject inconsistent existing fields."""
    result = []
    for record in records:
        row = dict(record)
        present = [row[k] for k in ('last_bucket_size', 'last_history_bucket_size') if k in row]
        if not present or any(not re.fullmatch(r'[0-9]+', str(v)) for v in present):
            raise ValueError('Missing/invalid last-bucket size')
        sizes = [int(v) for v in present]
        history = json.loads(row['sequence_items'])
        if (not isinstance(history, list) or not 1 <= len(history) <= 50
                or len(set(sizes)) != 1 or not 1 <= sizes[0] <= len(history)):
            raise ValueError('Invalid history or inconsistent last-bucket size')
        expected_pooling = int(sizes[0] >= 2)
        if 'direct_pooling' in row and (not re.fullmatch(r'[01]', str(row['direct_pooling']))
                or int(row['direct_pooling']) != expected_pooling):
            raise ValueError('Existing direct_pooling conflicts with last-bucket size')
        if 'last_bucket_size' not in row:
            row['last_bucket_size'] = sizes[0]
        if 'direct_pooling' not in row:
            row['direct_pooling'] = expected_pooling
        result.append(row)
    return result


def normalize_eval_frame(df):
    normalized = normalize_eval_records(df.to_dict(orient='records'))
    df = df.copy()
    for name in ('last_bucket_size', 'direct_pooling'):
        if name not in df.columns:
            df[name] = [row[name] for row in normalized]
    return df
'''


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
            or any(x in ('', '.', '..') or x.startswith('.') or x.lower() == 'notes' for x in value.split('/'))):
        raise ValueError('Unsafe relative path')
    if 'kuairec' in value.lower() or 'step39' in value.lower():
        raise ValueError('Unrelated project path rejected')
    return value


def safe(root, relative):
    path = root
    for part in relative_path(relative).split('/'):
        path = path / part
        if path.is_symlink():
            raise ValueError('Symbolic link rejected')
        if path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError('Reparse point rejected')
    path.resolve(strict=False).relative_to(root)
    return path


def signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def read_stable(root, relative):
    if PurePosixPath(relative).suffix.lower() in ('.pt', '.pth', '.ckpt') or relative.startswith(('data/', 'processed/')):
        raise ValueError('Checkpoint/interaction-input access forbidden')
    path = safe(root, relative)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > FILE_LIMIT:
        raise ValueError('Nonregular or oversized file')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    parts, h, n = [], hashlib.sha256(), 0
    with os.fdopen(os.open(str(path), flags), 'rb') as handle:
        if signature(before) != signature(os.fstat(handle.fileno())):
            raise ValueError('File changed before reading')
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            n += len(block)
            if n > FILE_LIMIT:
                raise ValueError('File size limit exceeded')
            h.update(block)
            parts.append(block)
        after = os.fstat(handle.fileno())
    if signature(before) != signature(after) or signature(before) != signature(safe(root, relative).lstat()) or n != before.st_size:
        raise ValueError('File changed during reading')
    return b''.join(parts), h.hexdigest()


def json_object(blob):
    value = json.loads(blob.decode('utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def csv_rows(blob):
    return list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig'), newline='')))


def integer(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not an integer count')
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError('Invalid integer count')
    return int(number)


def finite(value, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a numeric measurement')
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        raise ValueError('Nonfinite or out-of-range smoke measurement')
    return number


def ast_definitions(blob):
    tree = ast.parse(blob.decode('utf-8-sig'))
    return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted_name(node.value) + '.' + node.attr
    return ''


def close(a, b, tolerance=1e-10):
    return math.isclose(finite(a), finite(b), abs_tol=tolerance, rel_tol=tolerance)


def item_mapping(blob):
    rows = csv_rows(blob)
    mapping = {r['raw_item_id']: integer(r['item_idx']) for r in rows}
    if len(mapping) != len(rows) or set(mapping.values()) != set(range(1, len(rows) + 1)):
        raise ValueError('Item map is not bijective and contiguous')
    return mapping


def predictions(blob, mapping):
    result = {}
    for row in csv_rows(blob):
        sid, user, target = row['sample_id'], row['user_id'], row['target_item']
        rank = integer(row['target_rank'])
        raw = tuple(str(x) for x in json.loads(row['top10_raw_item_id']))
        indices = tuple(integer(x) for x in json.loads(row['top10_item_idx']))
        if not sid or not user or sid in result or target not in mapping or not 1 <= rank <= len(mapping):
            raise ValueError('Prediction identity/rank invalid')
        if len(raw) != 10 or len(indices) != 10 or len(set(raw)) != 10 or len(set(indices)) != 10:
            raise ValueError('Prediction Top10 uniqueness/shape invalid')
        if any(mapping.get(item) != idx for item, idx in zip(raw, indices)):
            raise ValueError('Prediction raw/index map mismatch')
        if (rank <= 10) != (target in raw) or (rank <= 10 and raw[rank - 1] != target):
            raise ValueError('Prediction target rank and ordered Top10 disagree')
        for key, expected in (('ndcg10', 1 / math.log2(rank + 1) if rank <= 10 else 0), ('hit10', int(rank <= 10)), ('hit20', int(rank <= 20))):
            if key in row and not close(row[key], expected):
                raise ValueError('Saved per-row metric disagrees with rank')
        result[sid] = (user, target, rank, raw, indices)
    if not result or len({v[0] for v in result.values()}) != len(result):
        raise ValueError('Empty panel or duplicate user')
    return result


def panel(records):
    return {sid: row[:2] for sid, row in records.items()}


def accuracy(records):
    ranks = [r[2] for r in records.values()]
    return {'samples': len(ranks), 'ndcg10': statistics.mean([1 / math.log2(r + 1) if r <= 10 else 0 for r in ranks]),
            'hr10': statistics.mean([r <= 10 for r in ranks]), 'hr20': statistics.mean([r <= 20 for r in ranks]),
            'mean_rank': statistics.mean(ranks)}


def metric_record(value, expected_samples, num_items):
    if integer(value['samples']) != integer(expected_samples):
        raise ValueError('Saved metric sample count mismatch')
    data = {k: finite(value[k], nonnegative=True) for k in ('ndcg10', 'hr10', 'hr20', 'mean_rank')}
    if not (0 <= data['ndcg10'] <= 1 and 0 <= data['hr10'] <= data['hr20'] <= 1 and 1 <= data['mean_rank'] <= num_items):
        raise ValueError('Saved accuracy outside valid range')
    return data


def training_trace(blob, saved, max_epochs, patience, num_items):
    rows = csv_rows(blob)
    if not rows or len(rows) > max_epochs:
        raise ValueError('Training log empty or exceeds max epochs')
    best, best_epoch, bad = -1.0, -1, 0
    selected = None
    for index, row in enumerate(rows, 1):
        if integer(row['epoch']) != index:
            raise ValueError('Training epoch sequence is not contiguous from one')
        loss, current = finite(row['train_loss'], nonnegative=True), finite(row['val_ndcg10'], nonnegative=True)
        hr, mean_rank = finite(row['val_hr10'], nonnegative=True), finite(row['val_mean_rank'], positive=True)
        finite(row['seconds'], nonnegative=True)
        if not (0 <= current <= 1 and 0 <= hr <= 1 and 1 <= mean_rank <= num_items):
            raise ValueError('Training validation metric range invalid')
        if current > best + 1e-12:
            best, best_epoch, bad, selected = current, index, 0, row
        else:
            bad += 1
        if integer(row['best_epoch']) != best_epoch or not close(row['best_ndcg10'], best, 1e-12) or integer(row['bad_epochs']) != bad:
            raise ValueError('Training best-epoch/bad-epoch trace mismatch')
        if bad >= patience and index != len(rows):
            raise ValueError('Training continued beyond the first early-stopping trigger')
    if len(rows) < max_epochs and bad != patience:
        raise ValueError('Training ended before max epochs without patience early stopping')
    if integer(saved['best_epoch']) != best_epoch:
        raise ValueError('Saved best epoch disagrees with validation selection')
    validation = saved['validation']
    if not close(validation['ndcg10'], best, 1e-8) or not close(validation['hr10'], selected['val_hr10']) or not close(validation['mean_rank'], selected['val_mean_rank']):
        raise ValueError('Saved best validation disagrees with selected training-log epoch')
    return {'completed_epochs': len(rows), 'best_epoch': best_epoch, 'best_validation_ndcg10': best,
            'final_bad_epochs': bad, 'stop_reason': 'patience' if bad >= patience else 'max_epochs',
            'threshold': 1e-12, 'training_loss_last': loss}


class Collector:
    def __init__(self, stage):
        self.stage = absolute_plain(stage)
        if not self.stage.name.startswith('kais_lbsp_amazon_train_v1_'):
            raise ValueError('Unexpected training stage prefix')
        verification = safe(self.stage, 'verification')
        if not verification.is_dir():
            raise ValueError('Training verification directory missing')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
        self.out = verification / ('train_review_' + stamp)
        self.out.mkdir(exist_ok=False)
        self.root, self.inputs, self.manifest, self.manifest_sha = None, None, None, None
        self.files, self.warnings, self.integrity_failures, self.copies = [], [], [], {}
        self.total, self.runs, self.references, self.sources, self.generated = 0, {}, {}, {}, {}
        self.driver_check = {'status': 'NOT_CHECKED'}

    def copy(self, origin, relative, archive_path, expected=None):
        record = {'origin': str(origin), 'path': relative, 'archive_path': archive_path, 'status': 'unknown'}
        try:
            relative_path(archive_path)
            blob, actual = read_stable(origin, relative)
            if expected is not None and actual != expected:
                raise ValueError('Pinned SHA256 mismatch')
            if self.total + len(blob) > TOTAL_LIMIT:
                raise ValueError('Collection exceeds 512 MiB limit')
            target = safe(self.out, 'snapshot/' + archive_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(blob)
            self.copies[archive_path] = blob
            self.total += len(blob)
            record.update(status='copied', bytes=len(blob), sha256=actual)
            return blob
        except (OSError, ValueError) as exc:
            record.update(status='missing' if isinstance(exc, FileNotFoundError) else 'rejected_or_changed', error=str(exc)[:240])
            self.warnings.append(record['status'] + ': ' + relative)
            if expected is not None and isinstance(exc, ValueError):
                self.integrity_failures.append('Pinned file rejected/changed: ' + relative)
            return None
        finally:
            self.files.append(record)

    def verify_sources(self):
        companion = absolute_plain(INPUTS_FILE)
        blob, digest = read_stable(companion.parent, companion.name)
        if digest != EXPECTED_INPUTS_SHA256:
            raise ValueError('Companion training input manifest SHA mismatch')
        self.inputs = json_object(blob)
        manifest_blob, self.manifest_sha = read_stable(self.stage, MANIFEST)
        self.manifest = json_object(manifest_blob)
        m = self.manifest
        if (m.get('version') != 'lbsp_amazon_train_v1' or m.get('inputs') != self.inputs or m.get('inputs_sha256') != digest
                or m.get('sources_sha256') != self.inputs['sources_sha256'] or m.get('data_sha256') != self.inputs['data_sha256']
                or m.get('datasets') != self.inputs['datasets'] or m.get('formal_seeds') != list(SEEDS)
                or m.get('smoke_reference') != self.inputs['smoke_reference']):
            raise ValueError('Preparation/input/source/data/smoke binding mismatch')
        self.root = absolute_plain(m['root'])
        if self.stage.parent != self.root or m['stage'] != str(self.stage):
            raise ValueError('Stage/root binding mismatch')
        if (self.inputs.get('version') != 'lbsp_amazon_train_v1' or self.inputs.get('formal_seeds') != list(SEEDS)
                or set(self.inputs['datasets']) != set(TAGS) or len(self.inputs['sources_sha256']) != 5
                or len(self.inputs['data_sha256']) != 9 or set(m['generated_sha256']) != set(GENERATED)):
            raise ValueError('Unexpected planned source/data/dataset/seed/script inventory')
        expected_references = {'references/%s_seed42/%s' % (tag, name): value
            for tag, plan in self.inputs['datasets'].items() for name, value in plan['control_reference']['files_sha256'].items()}
        if m.get('reference_files_sha256') != expected_references:
            raise ValueError('Stage Control reference manifest inventory/hash mismatch')
        for relative, expected in self.inputs['sources_sha256'].items():
            source, actual = read_stable(self.root, relative_path(relative))
            if not relative.endswith('.py') or actual != expected:
                raise ValueError('Original source SHA/extension mismatch')
            ast.parse(source.decode('utf-8-sig'))
            self.sources[relative] = source
        for relative, expected in m['generated_sha256'].items():
            source, actual = read_stable(self.stage, relative)
            if actual != expected:
                raise ValueError('Generated source SHA mismatch')
            ast.parse(source.decode('utf-8-sig'))
            self.generated[relative] = source
        self.algorithm_ast()
        for relative in self.sources:
            if self.copy(self.root, relative, 'original_sources/' + PurePosixPath(relative).name, self.inputs['sources_sha256'][relative]) is None:
                raise ValueError('Original source changed during collection')
        for relative in self.generated:
            if self.copy(self.stage, relative, relative, m['generated_sha256'][relative]) is None:
                raise ValueError('Generated source changed during collection')
        if self.copy(self.stage, MANIFEST, MANIFEST, self.manifest_sha) is None or self.copy(companion.parent, companion.name, 'manifests/pinned_inputs.json', digest) is None:
            raise ValueError('Manifest changed during collection')

    def algorithm_ast(self):
        template_blob = self.sources[self.inputs['template_source']]
        template = ast_definitions(template_blob)
        model = ast_definitions(self.sources[self.inputs['model_source']])
        frozen = ast_definitions(self.generated['scripts/model_definition.py'])
        for name in ('build_r1_group_positions_from_buckets', 'build_r1_group_positions_from_size', 'canonicalize_last_bucket_items', 'RecentBucketSetR1SASRec'):
            if frozen.get(name) != model.get(name) or name not in model:
                raise ValueError('Frozen model AST differs: ' + name)
        if frozen.get('predict_batch_masked') != template.get('predict_batch_masked'):
            raise ValueError('Frozen masked-forward AST differs')
        reference_normalizers = ast_definitions(NORMALIZER_REFERENCE_SOURCE.encode('utf-8'))
        original_tree = ast.parse(template_blob.decode('utf-8-sig'))
        def formal_parts(tree):
            loops = [n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch']
            mkdir = [i for i, n in enumerate(tree.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and dotted_name(n.value.func) == 'OUTPUT_DIR.mkdir']
            if len(loops) != 1 or len(mkdir) != 1 or mkdir[0] < 4:
                raise ValueError('Formal training loop/RNG reset boundary missing')
            return ast.dump(loops[0], include_attributes=False), [ast.dump(n, include_attributes=False) for n in tree.body[mkdir[0] - 4:mkdir[0]]]
        original_loop, original_rng = formal_parts(original_tree)
        for tag in TAGS:
            blob = self.generated['scripts/train_%s_lbsp.py' % tag]
            emitted = ast_definitions(blob)
            for name in ('AllPositionR1Dataset', 'training_loss', 'evaluate', 'predict_batch_masked'):
                if emitted.get(name) != template.get(name) or name not in template:
                    raise ValueError('Scientific function AST differs: %s/%s' % (tag, name))
            for name, reference in reference_normalizers.items():
                if emitted.get(name) != reference:
                    raise ValueError('Normalization adapter AST differs: %s/%s' % (tag, name))
            tree = ast.parse(blob.decode('utf-8-sig'))
            loop, rng = formal_parts(tree)
            if loop != original_loop or rng != original_rng:
                raise ValueError('Formal training loop or RNG reset AST differs: ' + tag)
            loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_eval')
            insertion = ast.dump(ast.parse('df = normalize_eval_frame(df)').body[0], include_attributes=False)
            insertions = [i for i, n in enumerate(loader.body) if ast.dump(n, include_attributes=False) == insertion]
            if insertions != [1]:
                raise ValueError('Evaluation normalization adapter location changed')
            del loader.body[1]
            if ast.dump(loader, include_attributes=False) != template['load_eval']:
                raise ValueError('Evaluation loader changes exceed approved normalization adapter')
        self.algorithm_check = {'status': 'PASS', 'scope': 'Model/helper, dataset, loss, evaluator, masked-forward, formal epoch-loop and formal RNG-reset AST match the pinned original. Normalizer reference and generated AST are parsed together in the current interpreter. No submitted source executed.'}

    def smoke_proof(self):
        proof = self.inputs['smoke_reference']
        self.smoke_stage = absolute_plain(proof['stage'])
        if self.smoke_stage.parent != self.root or not self.smoke_stage.name.startswith('kais_lbsp_amazon_smoke_v1_'):
            raise ValueError('Pinned smoke stage/root identity invalid')
        if len(proof['files_sha256']) != 13 or proof['files_sha256'].get('manifests/lbsp_amazon_smoke_manifest.json') != proof['manifest_sha256']:
            raise ValueError('Unexpected smoke-proof inventory/binding')
        blobs = {}
        for relative, expected in proof['files_sha256'].items():
            if not relative_path(relative).endswith(('.py', '.json', '.log')):
                raise ValueError('Unexpected smoke-proof extension')
            blobs[relative] = self.copy(self.smoke_stage, relative, 'smoke_reference/' + relative, expected)
        if any(b is None for b in blobs.values()):
            raise ValueError('Pinned smoke proof missing/rejected')
        manifest = json_object(blobs['manifests/lbsp_amazon_smoke_manifest.json'])
        previous = manifest['inputs']
        if (manifest.get('version') != 'lbsp_amazon_smoke_v1' or manifest['root'] != str(self.root)
                or manifest['stage'] != str(self.smoke_stage) or previous['datasets'] != self.inputs['datasets']
                or previous['sources_sha256'] != self.inputs['sources_sha256'] or previous['data_sha256'] != self.inputs['data_sha256']
                or previous['hyperparameters'] != self.inputs['hyperparameters']):
            raise ValueError('Smoke/training frozen protocol closure mismatch')
        drivers = [k for k in blobs if re.fullmatch(r'verification/lbsp_amazon_smoke_driver_[^/]+\.json', k)]
        if len(drivers) != 1:
            raise ValueError('Pinned smoke driver inventory invalid')
        driver = json_object(blobs[drivers[0]])
        if driver.get('status') != 'ALL_THREE_SMOKES_EXITED_ZERO' or driver.get('training_performed') is not False:
            raise ValueError('Pinned smoke driver not all PASS')
        for tag, plan in self.inputs['datasets'].items():
            summary = json_object(blobs['verification/smoke_%s.json' % tag])
            paths = [plan[k] for k in ('train_path', 'validation_path', 'test_path')]
            expected_data = {p: self.inputs['data_sha256'][p] for p in paths}
            if (summary.get('dataset') != TAGS[tag] or summary.get('tag') != tag or summary.get('seed') != 42 or summary.get('status') != 'SMOKE_PASS'
                    or summary.get('input_manifest_sha256') != manifest['inputs_sha256'] or summary.get('preparation_manifest_sha256') != proof['manifest_sha256']
                    or summary.get('sources_sha256') != self.inputs['sources_sha256'] or summary.get('data_sha256') != expected_data
                    or summary.get('reference_files_sha256') != plan['control_reference']['files_sha256']
                    or summary.get('control_reference') != plan['control_reference'] or summary.get('all_gradients_finite') is not True
                    or summary.get('optimizer_steps') != 0 or summary.get('formal_training_started') is not False
                    or summary.get('weights_read') is not False or summary['parity'].get('allclose') is not True
                    or summary['invariance'].get('exact_logits_invariance') is not True):
                raise ValueError('Pinned %s smoke summary protocol mismatch' % tag)
            run = driver['runs'][tag + '_seed42']
            expected_command = [driver['python'], '-u', str(self.smoke_stage / ('scripts/smoke_%s_lbsp.py' % tag)), '--mode', 'smoke']
            if integer(run['exit_code']) != 0 or run.get('command') != expected_command:
                raise ValueError('Pinned smoke driver command/exit mismatch')
        self.smoke_check = {'status': 'PASS', 'manifest_sha256': proof['manifest_sha256'], 'pinned_files': len(blobs),
                            'scope': 'Prior three smoke records and immutable protocol identities verified; no smoke or inference rerun.'}

    def control_references(self, tag):
        plan = self.inputs['datasets'][tag]
        ref = plan['control_reference']
        if set(ref['files_sha256']) != set(REFERENCE_FILES):
            raise ValueError('Unexpected Control reference inventory')
        blobs = {}
        for name in REFERENCE_FILES:
            expected = ref['files_sha256'][name]
            relative = ref['model_dir'] + '/' + name
            original, actual = read_stable(self.root, relative_path(relative))
            if actual != expected:
                raise ValueError('Original Control reference SHA mismatch: ' + relative)
            stage_relative = 'references/%s_seed42/%s' % (tag, name)
            if self.manifest.get('reference_files_sha256', {}).get(stage_relative) != expected:
                raise ValueError('Stage reference manifest SHA mismatch')
            blobs[name] = self.copy(self.stage, stage_relative, stage_relative, expected)
        if any(v is None for v in blobs.values()):
            raise ValueError('Control reference missing/rejected')
        mapping = item_mapping(blobs['item_id_map.csv'])
        baseline = predictions(blobs['test_per_sample_predictions.csv'], mapping)
        config, metrics = json_object(blobs['config.json']), json_object(blobs['metrics.json'])
        if (len(mapping) != integer(plan['expected_items']) or len(baseline) != integer(plan['expected_test'])
                or config.get('dataset') != TAGS[tag] or metrics.get('dataset') != TAGS[tag]
                or integer(config.get('seed')) != 42 or integer(metrics.get('seed')) != 42):
            raise ValueError('Control reference identity/count mismatch')
        for phase, expected in (('validation', plan['expected_val']), ('test', plan['expected_test'])):
            value = metrics[phase]
            count = value['overall']['N'] if 'overall' in value else value['samples']
            if integer(count) != integer(expected):
                raise ValueError('Control reference frozen sample count mismatch')
        self.references[tag] = {'map': mapping, 'panel': panel(baseline)}

    def provenance(self, tag):
        plan = self.inputs['datasets'][tag]
        return {'version': 'lbsp_amazon_train_v1', 'input_manifest_sha256': EXPECTED_INPUTS_SHA256,
            'preparation_manifest_sha256': self.manifest_sha,
            'generated_script_sha256': self.manifest['generated_sha256']['scripts/train_%s_lbsp.py' % tag],
            'model_definition_sha256': self.manifest['generated_sha256']['scripts/model_definition.py'],
            'sources_sha256': self.inputs['sources_sha256'],
            'data_sha256': {plan[k]: self.inputs['data_sha256'][plan[k]] for k in ('train_path', 'validation_path', 'test_path')},
            'reference_files_sha256': plan['control_reference']['files_sha256'], 'control_reference': plan['control_reference'],
            'smoke_reference': self.inputs['smoke_reference'], 'sequence_sort_keys': plan['sequence_sort_keys'],
            'scientific_definitions_unchanged': True, 'formal_rng_reset_preserved': True, 'smoke_execution_omitted': True}

    def run(self, tag, seed):
        name = '%s_seed%d' % (tag, seed)
        output_dir = 'outputs/%s_lbsp_seed%d' % (tag, seed)
        result = {'tag': tag, 'dataset': TAGS[tag], 'seed': seed, 'status': 'INCOMPLETE', 'checks': {}, 'errors': []}
        self.runs[name] = result
        try:
            marker = 'verification/completed_%s_seed%d.json' % (tag, seed)
            completed_blob = self.copy(self.stage, marker, marker)
            if completed_blob is None:
                for filename in REFERENCE_FILES:
                    self.copy(self.stage, output_dir + '/' + filename, output_dir + '/' + filename)
                result['errors'].append('Completion marker absent; partial metadata retained without statistics.')
                return
            completed = json_object(completed_blob)
            expected_provenance = self.provenance(tag)
            if any(completed.get(key) != value for key, value in expected_provenance.items()):
                raise ValueError('Completion provenance/hash/protocol mismatch')
            plan = self.inputs['datasets'][tag]
            if (completed.get('status') != 'COMPLETED' or completed.get('tag') != tag or completed.get('dataset') != TAGS[tag]
                    or integer(completed.get('seed')) != seed or completed.get('output_dir') != str(self.stage / output_dir)
                    or completed.get('all_original_hashes_verified') is not True or completed.get('originals_unchanged') is not True
                    or completed.get('checkpoint') != 'r1_meanpool_best.pt'):
                raise ValueError('Completion identity/original-preservation mismatch')
            hashes = completed['output_metadata_sha256']
            if set(hashes) != set(REFERENCE_FILES):
                raise ValueError('Completion metadata inventory must contain exactly five files')
            blobs = {filename: self.copy(self.stage, output_dir + '/' + filename, output_dir + '/' + filename, hashes[filename]) for filename in REFERENCE_FILES}
            if any(b is None for b in blobs.values()):
                result['errors'].append('Required saved metadata missing/rejected; no statistics.')
                return
            if tag not in self.references:
                raise ValueError('Validated Control map/panel reference unavailable')
            config, metrics = json_object(blobs['config.json']), json_object(blobs['metrics.json'])
            mapping = item_mapping(blobs['item_id_map.csv'])
            rows = predictions(blobs['test_per_sample_predictions.csv'], mapping)
            checks = {'model': config.get('model') == metrics.get('model') == 'RecentBucketSetR1SASRec',
                'method': config.get('method') == metrics.get('method') == 'RecentBucketSet_R1_V1_mean_pooling',
                'dataset': config.get('dataset') == metrics.get('dataset') == TAGS[tag],
                'seed': integer(config.get('seed')) == integer(metrics.get('seed')) == seed,
                'config_provenance': config.get('amazon_lbsp_training') == expected_provenance,
                'item_map_exact_Control': mapping == self.references[tag]['map'],
                'sample_user_target_panel_exact_Control': panel(rows) == self.references[tag]['panel'],
                'item_count': len(mapping) == integer(config.get('num_items')) == integer(plan['expected_items']),
                'legal_positions': integer(config.get('legal_training_positions')) == integer(plan['expected_positions']),
                'sampling': config.get('sampling') == 'all_legal_next_item_positions_reshuffled_every_epoch',
                'loss': config.get('loss') == 'full_item_cross_entropy',
                'pooling': config.get('pooling') == 'last_history_bucket_only_mean',
                'canonicalization': config.get('canonicalization') == 'sort_last_bucket_by_item_idx',
                'optimizer': config.get('optimizer') == 'Adam'}
            for field, expected in self.inputs['hyperparameters'].items():
                checks['hyperparameter_' + field] = config.get(field) == expected
            for field, planned in (('train_path', 'train_path'), ('validation_path', 'validation_path'), ('test_path', 'test_path')):
                checks['path_' + field] = config.get(field) == str(self.root / plan[planned])
            expected_counts = {'num_items': plan['expected_items'], 'training_positions': plan['expected_positions'],
                               'validation_samples': plan['expected_val'], 'test_samples': plan['expected_test']}
            checks['completion_counts'] = completed.get('counts') == expected_counts
            result['checks'] = checks
            if not all(checks.values()):
                raise ValueError('Saved config/protocol mismatch: ' + ','.join(k for k, ok in checks.items() if not ok))
            metric_record(metrics['validation'], plan['expected_val'], len(mapping))
            metric_record(metrics['test'], plan['expected_test'], len(mapping))
            actual = accuracy(rows)
            if any(not close(actual[k], metrics['test'][k]) for k in actual):
                raise ValueError('Saved canonical test accuracy disagrees with target ranks')
            trace = training_trace(blobs['training_log.csv'], metrics, self.inputs['hyperparameters']['max_epochs'], self.inputs['hyperparameters']['patience'], len(mapping))
            if integer(completed.get('best_epoch')) != trace['best_epoch']:
                raise ValueError('Completion best epoch disagrees with saved training trace')
            result.update(status='VALID_SAVED_RUN', best_epoch=trace['best_epoch'], training_trace=trace,
                          canonical_test=actual, validation=metrics['validation'], checkpoint_contents_read=False)
            result['_map'], result['_panel'] = mapping, panel(rows)
            result['_core'] = {k: v for k, v in config.items() if k not in ('seed', 'amazon_lbsp_training')}
        except (OSError, ValueError, KeyError, TypeError, SyntaxError, csv.Error) as exc:
            result.update(status='INVALID_NO_STATISTICS')
            result['errors'].append(type(exc).__name__ + ': ' + str(exc)[:320])

    def summaries(self):
        result = {}
        for tag in TAGS:
            runs = [self.runs['%s_seed%d' % (tag, seed)] for seed in SEEDS]
            summary = {'status': 'INCOMPLETE_NO_THREE_SEED_SUMMARY'}
            result[tag] = summary
            if any(r['status'] != 'VALID_SAVED_RUN' for r in runs):
                continue
            aligned = {'same_item_map': all(r['_map'] == runs[0]['_map'] for r in runs),
                       'same_sample_user_target_panel': all(r['_panel'] == runs[0]['_panel'] for r in runs),
                       'same_scientific_config_except_seed_provenance': all(r['_core'] == runs[0]['_core'] for r in runs)}
            summary['alignment'] = aligned
            if not all(aligned.values()):
                summary['status'] = 'CROSS_SEED_MISMATCH_NO_SUMMARY'
                continue
            summary.update(status='THREE_SEED_SAVED_RUN_SUMMARY', seeds=list(SEEDS),
                canonical_test_samples=runs[0]['canonical_test']['samples'],
                best_epochs=[r['best_epoch'] for r in runs],
                test_mean_sample_SD={metric: {'mean': statistics.mean([r['canonical_test'][metric] for r in runs]),
                    'sample_SD_ddof1': statistics.stdev([r['canonical_test'][metric] for r in runs])}
                    for metric in ('ndcg10', 'hr10', 'hr20', 'mean_rank')},
                validation_mean_sample_SD={metric: {'mean': statistics.mean([r['validation'][metric] for r in runs]),
                    'sample_SD_ddof1': statistics.stdev([r['validation'][metric] for r in runs])}
                    for metric in ('ndcg10', 'hr10', 'hr20', 'mean_rank')})
        return result

    def driver(self):
        names = [k for k in self.copies if re.fullmatch(r'verification/lbsp_amazon_train_driver_[^/]+\.json', k)]
        self.driver_check = {'status': 'MISSING' if not names else 'FAIL', 'errors': []}
        if len(names) != 1:
            self.driver_check['errors'].append('Exactly one training driver JSON is required for all-nine status.')
            return
        try:
            record = json_object(self.copies[names[0]])
            planned = {'%s_seed%d' % (tag, seed) for tag in TAGS for seed in SEEDS}
            checks = {'stage': record.get('stage') == str(self.stage),
                      'version': record.get('version') == 'lbsp_amazon_train_v1',
                      'inputs_sha256': record.get('inputs_sha256') == EXPECTED_INPUTS_SHA256,
                      'formal_seeds': record.get('formal_seeds') == list(SEEDS),
                      'status': record.get('status') == 'ALL_NINE_TRAININGS_EXITED_ZERO',
                      'nine_runs_only': set(record.get('runs', {})) == planned}
            for tag in TAGS:
                for seed in SEEDS:
                    key = '%s_seed%d' % (tag, seed)
                    run = record['runs'][key]
                    command = [record['python'], '-u', str(self.stage / ('scripts/train_%s_lbsp.py' % tag)), '--mode', 'train', '--seed', str(seed)]
                    log = Path(run['log'])
                    checks[key + '_command'] = run.get('command') == command
                    checks[key + '_exit_zero'] = integer(run.get('exit_code')) == 0
                    checks[key + '_completion_verified'] = run.get('completion_verified') is True
                    checks[key + '_saved_run_valid'] = self.runs.get(key, {}).get('status') == 'VALID_SAVED_RUN'
                    checks[key + '_log_collected'] = log.is_absolute() and log.parent == self.stage / 'verification' and 'verification/' + log.name in self.copies
            self.driver_check.update(status='PASS' if all(checks.values()) else 'FAIL', path=names[0], checks=checks)
            if not all(checks.values()):
                self.driver_check['errors'].append('Training driver and saved completion evidence do not agree on all nine successful runs.')
        except (ValueError, KeyError, TypeError) as exc:
            self.driver_check['errors'].append(type(exc).__name__ + ': ' + str(exc)[:320])

    def finish(self, summaries):
        public = {name: {k: v for k, v in run.items() if not k.startswith('_')} for name, run in self.runs.items()}
        complete = (len(public) == 9 and all(r['status'] == 'VALID_SAVED_RUN' for r in public.values())
                    and len(summaries) == 3 and all(s['status'] == 'THREE_SEED_SAVED_RUN_SUMMARY' for s in summaries.values())
                    and self.driver_check.get('status') == 'PASS')
        status = 'INTEGRITY_FAILED_NO_BUNDLE' if self.integrity_failures else 'ALL_NINE_SAVED_RUNS_CHECKED' if complete else 'INCOMPLETE'
        report = {'status': status, 'stage': str(self.stage), 'input_manifest_sha256': EXPECTED_INPUTS_SHA256,
            'algorithm_ast_check': getattr(self, 'algorithm_check', {'status': 'NOT_COMPLETED'}),
            'smoke_reference_check': getattr(self, 'smoke_check', {'status': 'NOT_COMPLETED'}),
            'run_checks': public, 'dataset_summaries': summaries, 'driver_check': self.driver_check,
            'files': self.files, 'warnings': self.warnings, 'integrity_failures': self.integrity_failures,
            'scope': 'PRIVATE saved-file validation of nine Amazon LBSP trainings. No research code executed, no raw interaction input or checkpoint stat/content read. Test metrics are independently rank-derived; validation selection and stopping are checked against saved logs. Three-seed summaries use sample SD (ddof=1) and require all three valid runs per dataset. This does not establish end-to-end reproduction, permutation stability, significance or equivalence.'}
        (self.out / 'training_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
        with (self.out / 'COLLECTED_SHA256.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['origin', 'path', 'archive_path', 'status', 'bytes', 'sha256', 'error'])
            writer.writeheader()
            writer.writerows(self.files)
        with (self.out / 'three_seed_summary.csv').open('w', encoding='utf-8', newline='') as stream:
            fields = ['dataset', 'method', 'samples', 'training_seeds', 'metric', 'mean', 'sample_SD_ddof1']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for tag, summary in summaries.items():
                if summary['status'] == 'THREE_SEED_SAVED_RUN_SUMMARY':
                    for metric, value in summary['test_mean_sample_SD'].items():
                        writer.writerow({'dataset': TAGS[tag], 'method': 'LBSP', 'samples': summary['canonical_test_samples'],
                                         'training_seeds': '42,2026,3407', 'metric': metric, **value})
        lines = ['# PRIVATE Amazon LBSP three-seed training report', '', 'Status: **' + status + '**', '', report['scope'], '',
                 '| Run | Check | Best epoch | Canonical test NDCG@10 |', '|---|---|---:|---:|']
        for name, run in public.items():
            lines.append('| %s | %s | %s | %s |' % (name, run['status'], run.get('best_epoch', ''), run.get('canonical_test', {}).get('ndcg10', '')))
        lines += ['', 'Only a complete valid three-seed dataset receives mean/sample-SD statistics. Incomplete datasets receive no aggregate.',
                  'These are canonical-input accuracy results; five-permutation stability evaluation is a later step.', '', '## Diagnostics', '']
        messages = self.integrity_failures + self.warnings + self.driver_check.get('errors', [])
        messages += [name + ': ' + error for name, run in public.items() for error in run['errors']]
        lines += ['- ' + text for text in messages] or ['- No discrepancy found.']
        lines += ['', 'Private per-user prediction records are included; do not publish this ZIP directly.', '']
        (self.out / 'SUMMARY.md').write_text('\n'.join(lines), 'utf-8')
        bundle = None
        if not self.integrity_failures:
            bundle = self.out / 'lbsp_amazon_three_seed_results.zip'
            paths = sorted(p for p in self.out.rglob('*') if p.is_file())
            with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for path in paths:
                    if path.is_symlink():
                        raise ValueError('Output symlink rejected')
                    archive.write(path, path.relative_to(self.out).as_posix())
            digest = hashlib.sha256()
            with bundle.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            bundle.with_name(bundle.name + '.sha256.txt').write_text(digest.hexdigest() + '  ' + bundle.name + '\n', 'ascii')
        return {'report_dir': str(self.out), 'zip_path': str(bundle) if bundle else None, 'status': status,
                'exit_code': 2 if self.integrity_failures else 0 if complete else 1}


def collect(stage):
    collector = Collector(stage)
    summaries = {}
    try:
        collector.verify_sources()
        collector.smoke_proof()
        for tag in TAGS:
            collector.control_references(tag)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, csv.Error) as exc:
        collector.integrity_failures.append(type(exc).__name__ + ': ' + str(exc)[:320])
    if not collector.integrity_failures:
        eligible = [p for p in safe(collector.stage, 'verification').iterdir() if not p.name.startswith(('.', 'completed_')) and p.suffix.lower() in ('.json', '.log') and p.is_file()]
        if len(eligible) > 128:
            collector.integrity_failures.append('Unexpected verification log/JSON count')
        else:
            for path in sorted(eligible):
                collector.copy(collector.stage, 'verification/' + path.name, 'verification/' + path.name)
            for tag in TAGS:
                for seed in SEEDS:
                    collector.run(tag, seed)
            summaries = collector.summaries()
            collector.driver()
    return collector.finish(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.stage.expanduser())
    path = Path(result['report_dir']) / 'training_checks.json'
    print('SUMMARY=' + str(Path(result['report_dir']) / 'SUMMARY.md'))
    print('CHECKS=' + str(path))
    print('LBSP_TRAIN_REPORT_ZIP=' + (result['zip_path'] or 'NOT_CREATED'))
    print('STATUS=' + result['status'])
    report = json_object(path.read_bytes())
    for error in report['integrity_failures']:
        print('INTEGRITY_FAILURE=' + error)
    for name, run in report['run_checks'].items():
        for error in run['errors']:
            print('RUN_FAILURE=' + name + ': ' + error)
        for key, passed in run['checks'].items():
            if not passed:
                print('RUN_FAILED_CHECK=' + name + ': ' + key)
    for error in report['driver_check'].get('errors', []):
        print('DRIVER_FAILURE=' + error)
    for key, passed in report['driver_check'].get('checks', {}).items():
        if not passed:
            print('DRIVER_FAILED_CHECK=' + key)
    return result['exit_code']


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        print('COLLECTION_FAILED=' + type(exc).__name__ + ': ' + str(exc)[:320], file=sys.stderr)
        sys.exit(2)


