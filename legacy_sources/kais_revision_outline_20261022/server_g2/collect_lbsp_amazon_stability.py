#!/usr/bin/env python3
"""Collect PRIVATE fixed-model stability evidence; Python 3.8+, standard library.

Never executes research code, performs inference, accesses checkpoints, or reads
the eighteen original/permuted interaction inputs. No existing file is modified.
"""
import argparse
import ast
import csv
import datetime
import hashlib
import io
import itertools
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

INPUTS_FILE = Path(__file__).absolute().with_name('lbsp_amazon_stability_inputs_20261025.json')
EXPECTED_INPUTS_SHA256 = 'b98f998d0c169d8c14478795bd119b73aa48f5cdaa1767449e378ce71307667b'
TAGS = {'beauty': 'Amazon_Beauty', 'sports': 'Amazon_Sports', 'toys': 'Amazon_Toys'}
SEEDS, PERMS = (42, 2026, 3407), (101, 202, 303, 404, 505)
REFERENCE_FILES = ('config.json', 'metrics.json', 'training_log.csv', 'item_id_map.csv', 'test_per_sample_predictions.csv')
PREDICTIONS = ('predictions_original.csv',) + tuple('predictions_perm_seed%d.csv' % s for s in PERMS)
CONTROL_RUN_FILES = PREDICTIONS + ('panel_labels.csv', 'metadata.json')
RUN_FILES = CONTROL_RUN_FILES + ('per_seed_invariance.csv',)
MANIFEST = 'manifests/lbsp_amazon_stability_manifest.json'
GENERATED = {'scripts/model_definition.py'} | {'scripts/evaluate_%s_lbsp.py' % tag for tag in TAGS}
FILE_LIMIT, TOTAL_LIMIT = 64 * 1024 ** 2, 512 * 1024 ** 2

# Trusted normalizer source is parsed as data, never executed.
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


def mean(values):
    return statistics.mean(values) if values else None


def integer(value):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError('Invalid integer field')
    return int(number)


def parse_predictions(blob, item_map):
    result = {}
    for row in csv_rows(blob):
        sid, user, target = row['sample_id'], row['user_id'], row['target_item']
        if not sid or not user or sid in result or target not in item_map:
            raise ValueError('Prediction sample/target identity invalid')
        rank = integer(row['target_rank'])
        raw = tuple(str(v) for v in json.loads(row['top10_raw_item_id']))
        idx = tuple(integer(v) for v in json.loads(row['top10_item_idx']))
        if rank < 1 or rank > len(item_map) or len(raw) != 10 or len(idx) != 10 or len(set(raw)) != 10 or len(set(idx)) != 10:
            raise ValueError('Prediction rank/Top10 shape invalid')
        if any(item_map.get(a) != b for a, b in zip(raw, idx)):
            raise ValueError('Top10 raw/index mapping mismatch')
        if ((rank <= 10) != (target in raw)) or (rank <= 10 and raw.index(target) + 1 != rank):
            raise ValueError('Target rank and Top10 disagree')
        for key, expected in (('ndcg10', 1 / math.log2(rank + 1) if rank <= 10 else 0), ('hit10', int(rank <= 10)), ('hit20', int(rank <= 20))):
            if key in row and not math.isclose(float(row[key]), expected, abs_tol=1e-10, rel_tol=1e-10):
                raise ValueError('Saved per-row metric disagrees with rank')
        result[sid] = (user, target, rank, raw, idx)
    if len({v[0] for v in result.values()}) != len(result):
        raise ValueError('Duplicate user in one-target panel')
    return result


def identities(records):
    return {k: v[:2] for k, v in records.items()}


def parse_labels(blob):
    result = {}
    hashes = ['original_sequence_sha256'] + ['perm_seed%d_sequence_sha256' % p for p in PERMS]
    for row in csv_rows(blob):
        sid = row['sample_id']
        if not sid or sid in result:
            raise ValueError('Duplicate/empty label sample')
        affected = str(row['affected']).lower()
        if affected not in ('0', '1', 'false', 'true'):
            raise ValueError('Invalid affected flag')
        affected = affected in ('1', 'true')
        size = integer(row['last_bucket_size'])
        sequence_hashes = tuple(row[k] for k in hashes)
        if any(not re.fullmatch('[a-fA-F0-9]{64}', h) for h in sequence_hashes):
            raise ValueError('Missing/invalid sequence identity hashes')
        sequence_hashes = tuple(h.lower() for h in sequence_hashes)
        distinct = len(set(sequence_hashes[1:]))
        if size < 1 or (affected and size < 2) or affected != (distinct >= 2) or integer(row['distinct_sequences_5perm']) != distinct:
            raise ValueError('Affected/distinct-sequence label mismatch')
        result[sid] = (row['user_id'], row['target_item'], size, affected, sequence_hashes)
    return result


def compare_population(predictions, labels, selected):
    primary_overlap, primary_flip, pair_flip, baseline_overlap, baseline_pairflip, baseline_anyflip, distinct_overlap, distinct_flat = [], [], [], [], [], [], [], []
    for sid in selected:
        rows = [prediction[sid] for prediction in predictions]
        permutation = rows[1:]
        overlaps = [len(set(a[3]) & set(b[3])) / 10 for a, b in itertools.combinations(permutation, 2)]
        flips = [(a[2] <= 10) != (b[2] <= 10) for a, b in itertools.combinations(permutation, 2)]
        states = [p[2] <= 10 for p in permutation]
        primary_overlap.append(mean(overlaps))
        primary_flip.append(any(states) and not all(states))
        pair_flip.append(mean(flips))
        baseline_overlap.append(mean([len(set(rows[0][3]) & set(p[3])) / 10 for p in permutation]))
        base_flips = [(rows[0][2] <= 10) != (p[2] <= 10) for p in permutation]
        baseline_pairflip.append(mean(base_flips))
        baseline_anyflip.append(any(base_flips))
        sequence_hashes = labels[sid][4][1:]
        distinct = [o for (a, b), o in zip(itertools.combinations(range(5), 2), overlaps) if sequence_hashes[a] != sequence_hashes[b]]
        if distinct:
            distinct_overlap.append(mean(distinct))
            distinct_flat.extend(distinct)
    return {'N': len(selected), 'overlap_5perm_pair_mean': mean(primary_overlap),
            'hit_flip_user_fraction': mean(primary_flip),
            'permutation_pairwise_hit_flip_mean_diagnostic': mean(pair_flip),
            'baseline_vs_permutation_overlap_diagnostic': mean(baseline_overlap),
            'baseline_vs_permutation_pairwise_hit_flip_diagnostic': mean(baseline_pairflip),
            'baseline_vs_any_permutation_user_flip_diagnostic': mean(baseline_anyflip),
            'distinct_input_pair_overlap_user_mean_diagnostic': mean(distinct_overlap),
            'distinct_input_pair_overlap_pooled_diagnostic': mean(distinct_flat),
            'distinct_input_pair_N': len(distinct_flat),
            'distinct_pair_user_N': len(distinct_overlap)}


def accuracy(records, selected=None):
    values = list(records.values()) if selected is None else [records[k] for k in selected]
    ranks = [v[2] for v in values]
    return {'N': len(ranks), 'NDCG@10': mean([1 / math.log2(r + 1) if r <= 10 else 0 for r in ranks]),
            'HR@10': mean([r <= 10 for r in ranks]), 'HR@20': mean([r <= 20 for r in ranks]),
            'MeanRank': mean(ranks)}


def same_actual_input_check(predictions, labels):
    for sid in labels:
        seen = {}
        for value, sequence_sha in zip([p[sid] for p in predictions], labels[sid][4]):
            if sequence_sha in seen and seen[sequence_sha] != value[2:]:
                raise ValueError('Identical actual inputs produce different rank/Top10')
            seen[sequence_sha] = value[2:]


def verify_accuracy(metrics, records):
    actual = accuracy(records)
    test = metrics['test']
    if 'overall' in test:
        expected = {k: test['overall'][k] for k in actual}
    else:
        expected = {k: test[v] for k, v in (('N', 'samples'), ('NDCG@10', 'ndcg10'),
                    ('HR@10', 'hr10'), ('HR@20', 'hr20'), ('MeanRank', 'mean_rank'))}
    if any(not math.isclose(actual[k], expected[k], rel_tol=1e-10, abs_tol=1e-10) for k in actual):
        raise ValueError('Canonical rank-derived accuracy disagrees with saved metrics')
    return actual


class Collector:
    def __init__(self, stage):
        self.stage = absolute_plain(stage)
        if not self.stage.name.startswith('kais_lbsp_amazon_stability_v1_'):
            raise ValueError('Unexpected stage prefix')
        verification = safe(self.stage, 'verification')
        if not verification.is_dir():
            raise ValueError('Stage verification directory missing')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_UTC_') + uuid.uuid4().hex[:10]
        self.out = verification / ('stability_review_' + stamp)
        self.out.mkdir(exist_ok=False)
        self.files, self.warnings, self.failures, self.runs = [], [], [], {}
        self.copies, self.total = {}, 0
        self.root, self.manifest, self.inputs = None, None, None
        self.control_copies = {}
        self.training_copies, self.sources, self.generated = {}, {}, {}
        self.driver_check = {'status': 'NOT_CHECKED'}

    def copy(self, origin, relative, archive_path, expected=None):
        record = {'path': relative, 'archive_path': archive_path, 'origin': str(origin), 'status': 'unknown'}
        try:
            relative_path(archive_path)
            blob, actual = read_stable(origin, relative)
            if expected is not None and actual != expected:
                raise ValueError('Pinned SHA256 mismatch')
            if self.total + len(blob) > TOTAL_LIMIT:
                raise ValueError('Collection exceeds 512 MiB limit')
            destination = safe(self.out, 'snapshot/' + archive_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(blob)
            self.total += len(blob)
            self.copies[archive_path] = destination
            record.update(status='copied', bytes=len(blob), sha256=actual)
            return blob
        except (OSError, ValueError) as exc:
            record.update(status='not_observed' if isinstance(exc, FileNotFoundError) else 'rejected_or_changed', error=str(exc)[:200])
            self.warnings.append(record['status'] + ': ' + relative)
            if expected is not None and isinstance(exc, ValueError):
                self.failures.append('Pinned file rejected or changed: ' + relative)
            return None
        finally:
            self.files.append(record)

    def verify_sources(self):
        companion = absolute_plain(INPUTS_FILE)
        blob, sha = read_stable(companion.parent, companion.name)
        if sha != EXPECTED_INPUTS_SHA256:
            raise ValueError('Companion input manifest SHA mismatch')
        self.inputs = json_object(blob)
        manifest_blob, manifest_sha = read_stable(self.stage, MANIFEST)
        self.manifest = json_object(manifest_blob)
        m = self.manifest
        if m.get('version') != 'lbsp_amazon_stability_v1' or m.get('inputs_sha256') != sha or m.get('inputs') != self.inputs:
            raise ValueError('Stage/input manifest binding mismatch')
        self.root = absolute_plain(m['root'])
        if self.stage.parent != self.root or str(self.stage) != m.get('stage'):
            raise ValueError('Stage/root identity mismatch')
        if self.inputs.get('training_seeds') != list(SEEDS) or self.inputs.get('permutation_seeds') != list(PERMS) or set(self.inputs['datasets']) != set(TAGS):
            raise ValueError('Unexpected planned datasets/seeds/permutations')
        if (m.get('sources_sha256') != self.inputs.get('sources_sha256') or m.get('data_sha256') != self.inputs.get('data_sha256')
                or m.get('training_reference') != self.inputs['training_reference'] or m.get('control_reference') != self.inputs['control_reference']
                or len(self.inputs['sources_sha256']) != 3 or len(self.inputs['data_sha256']) != 18):
            raise ValueError('Manifest source/data identities disagree')
        verified = []
        for relative, expected in m['sources_sha256'].items():
            if not relative_path(relative).startswith(('step38_kais_retrain/scripts/', 'audit/')) or not relative.endswith('.py'):
                raise ValueError('Unexpected original source path')
            original, actual = read_stable(self.root, relative)
            if actual != expected:
                raise ValueError('Original source SHA mismatch')
            ast.parse(original.decode('utf-8-sig'), filename=relative)
            self.sources[relative] = original
            verified.append((self.root, relative, 'original_sources/' + PurePosixPath(relative).name, expected))
        generated = m.get('generated_sha256', {})
        if set(generated) != GENERATED:
            raise ValueError('Unexpected generated source count')
        for relative, expected in generated.items():
            if not relative_path(relative).startswith('scripts/') or not relative.endswith('.py'):
                raise ValueError('Unexpected generated source path')
            source, actual = read_stable(self.stage, relative)
            if actual != expected:
                raise ValueError('Generated source SHA mismatch')
            ast.parse(source.decode('utf-8-sig'), filename=relative)
            self.generated[relative] = source
            verified.append((self.stage, relative, relative, expected))
        self.algorithm_ast()
        self.manifest_sha = manifest_sha
        verified.append((self.stage, MANIFEST, MANIFEST, manifest_sha))
        for origin, relative, destination, expected in verified:
            if self.copy(origin, relative, destination, expected) is None:
                raise ValueError('Verified source changed while collecting')
        if self.copy(companion.parent, companion.name, 'manifests/pinned_inputs.json', sha) is None:
            raise ValueError('Companion input manifest changed while collecting')

    def algorithm_ast(self):
        def definitions(blob):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(blob.decode('utf-8-sig')).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        expected_model = definitions(self.sources[self.inputs['model_source']])
        frozen = definitions(self.generated['scripts/model_definition.py'])
        for name in ('build_r1_group_positions_from_size', 'canonicalize_last_bucket_items', 'RecentBucketSetR1SASRec'):
            if name not in expected_model or frozen.get(name) != expected_model[name]:
                raise ValueError('Frozen model AST differs: ' + name)
        expected_forward = definitions(self.sources[self.inputs['forward_source']])
        if frozen.get('predict_batch_masked') != expected_forward.get('predict_batch_masked') or set(frozen) != {'build_r1_group_positions_from_size', 'canonicalize_last_bucket_items', 'RecentBucketSetR1SASRec', 'predict_batch_masked'}:
            raise ValueError('Frozen masked forward/inference definition inventory differs')
        normalizers = definitions(NORMALIZER_REFERENCE_SOURCE.encode('utf-8'))
        for tag, plan in self.inputs['datasets'].items():
            original = definitions(self.sources[plan['eval_source']])
            blob = self.generated['scripts/evaluate_%s_lbsp.py' % tag]
            emitted = definitions(blob)
            for name in ('construct', 'predict_all', 'make_predictions', 'extract'):
                if name not in original or emitted.get(name) != original[name]:
                    raise ValueError('Scientific evaluator AST differs: %s/%s' % (tag, name))
            for name, expected in normalizers.items():
                if emitted.get(name) != expected:
                    raise ValueError('Normalizer AST differs: %s/%s' % (tag, name))
            nodes = [n for n in ast.parse(blob.decode('utf-8-sig')).body if isinstance(n, ast.FunctionDef) and n.name == 'read_csv']
            if len(nodes) != 1:
                raise ValueError('Expected one scientific read_csv function')
            node = nodes[0]
            adapter = ast.dump(ast.parse('df = normalize_eval_frame(df)').body[0], include_attributes=False)
            positions = [i for i, part in enumerate(node.body) if ast.dump(part, include_attributes=False) == adapter]
            if positions != [1]:
                raise ValueError('Approved read_csv alias adapter location differs')
            del node.body[1]
            if ast.dump(node, include_attributes=False) != original['read_csv']:
                raise ValueError('read_csv changes exceed the approved alias adapter')
        self.algorithm_check = {'status': 'PASS', 'scope': 'Frozen model/helpers and construct/predict_all/make_predictions/extract AST match original sources. read_csv permits only one exact alias adapter; normalizer reference and emitted AST parsed together in current interpreter. No research code executed.'}

    def control_reference(self):
        planned = self.inputs['control_reference']
        stage_name = relative_path(planned['stage'])
        if '/' in stage_name or not stage_name.startswith('kais_control_stability_v1_'):
            raise ValueError('Unexpected Control stage identity')
        self.control_stage = absolute_plain(safe(self.root, stage_name))
        if self.control_stage.parent != self.root:
            raise ValueError('Control stage not within root')
        if planned['manifest_path'] != 'manifests/control_stability_manifest.json':
            raise ValueError('Unexpected Control manifest path')
        blob = self.copy(self.control_stage, planned['manifest_path'],
                         'control_reference/' + planned['manifest_path'], planned['manifest_sha256'])
        if blob is None:
            self.warnings.append('Pinned Control manifest unavailable; no method comparison.')
            return
        self.control_manifest = json_object(blob)
        cm = self.control_manifest
        if cm.get('version') != 'control_stability_v1' or cm.get('inputs_sha256') != planned['input_manifest_sha256']:
            raise ValueError('Pinned Control manifest input binding mismatch')
        if cm.get('stage') != str(self.control_stage) or cm.get('root') != str(self.root):
            raise ValueError('Pinned Control stage/root binding mismatch')
        expected_files = {'outputs/%s_seed%d/%s' % (tag, s, name) for tag in TAGS for s in SEEDS for name in CONTROL_RUN_FILES}
        if set(planned['files_sha256']) != expected_files:
            raise ValueError('Unexpected pinned Control output inventory')
        for relative, expected in planned['files_sha256'].items():
            self.control_copies[relative] = self.copy(self.control_stage, relative,
                'control_reference/' + relative, expected)

    def training_reference(self):
        plan = self.inputs['training_reference']
        name = relative_path(plan['stage'])
        if '/' in name or not name.startswith('kais_lbsp_amazon_train_v1_') or plan['manifest_path'] != 'manifests/lbsp_amazon_train_manifest.json':
            raise ValueError('Unexpected training reference path')
        self.training_stage = absolute_plain(safe(self.root, name))
        if self.training_stage.parent != self.root or len(plan['files_sha256']) != 25 or plan['files_sha256'].get(plan['manifest_path']) != plan['manifest_sha256']:
            raise ValueError('Unexpected training reference root/inventory/SHA binding')
        for relative, expected in plan['files_sha256'].items():
            if not relative_path(relative).endswith(('.py', '.json', '.log')):
                raise ValueError('Unexpected training proof extension')
            self.training_copies[relative] = self.copy(self.training_stage, relative, 'training_reference/' + relative, expected)
        if any(b is None for b in self.training_copies.values()):
            raise ValueError('Pinned training proof missing/rejected')
        tm = json_object(self.training_copies[plan['manifest_path']])
        self.training_manifest = tm
        if (tm.get('version') != 'lbsp_amazon_train_v1' or tm.get('inputs_sha256') != plan['input_manifest_sha256']
                or tm.get('root') != str(self.root) or tm.get('stage') != str(self.training_stage)
                or tm['inputs'].get('formal_seeds') != list(SEEDS)):
            raise ValueError('Pinned training manifest identity differs')
        for tag, dataset in self.inputs['datasets'].items():
            training_plan = tm['inputs']['datasets'][tag]
            original = dataset['original_path']
            if training_plan['test_path'] != original or tm['inputs']['data_sha256'].get(original) != self.inputs['data_sha256'][original]:
                raise ValueError('Canonical evaluation input differs from the training frozen test panel')
            for seed in SEEDS:
                run = dataset['runs'][str(seed)]
                completed = json_object(self.training_copies['verification/completed_%s_seed%d.json' % (tag, seed)])
                if (completed.get('status') != 'COMPLETED' or completed.get('dataset') != TAGS[tag] or completed.get('tag') != tag
                        or integer(completed.get('seed')) != seed or completed.get('input_manifest_sha256') != plan['input_manifest_sha256']
                        or completed.get('preparation_manifest_sha256') != plan['manifest_sha256']
                        or completed.get('output_metadata_sha256') != run['files_sha256']
                        or integer(completed.get('best_epoch')) != integer(run['best_epoch'])
                        or completed.get('output_dir') != str(self.root / run['model_dir'])
                        or completed.get('checkpoint') != run['checkpoint']
                        or completed.get('originals_unchanged') is not True or completed.get('all_original_hashes_verified') is not True):
                    raise ValueError('Pinned training completion/model descriptor mismatch')
        drivers = [k for k in self.training_copies if re.fullmatch(r'verification/lbsp_amazon_train_driver_[^/]+\.json', k)]
        if len(drivers) != 1:
            raise ValueError('Pinned training driver inventory invalid')
        driver = json_object(self.training_copies[drivers[0]])
        if driver.get('status') != 'ALL_NINE_TRAININGS_EXITED_ZERO':
            raise ValueError('Pinned training driver did not finish all nine runs')
        for tag in TAGS:
            for seed in SEEDS:
                r = driver['runs']['%s_seed%d' % (tag, seed)]
                if integer(r.get('exit_code')) != 0 or r.get('completion_verified') is not True:
                    raise ValueError('Pinned training driver run incomplete')
        self.training_check = {'status': 'PASS', 'pinned_files': 25, 'manifest_sha256': plan['manifest_sha256'],
                               'scope': 'Prior training completion/driver and model metadata identities bound; no checkpoint stat or content accessed.'}

    def run(self, tag, seed):
        plan = self.inputs['datasets'][tag]
        specification = plan['runs'][str(seed)]
        run_name = '%s_seed%d' % (tag, seed)
        result = {'tag': tag, 'seed': seed, 'status': 'incomplete_no_statistics', 'checks': {}, 'errors': []}
        self.runs[run_name] = result
        try:
            relative_path(specification['model_dir'])
            if specification['model_dir'] != self.inputs['training_reference']['stage'] + '/outputs/%s_lbsp_seed%d' % (tag, seed):
                raise ValueError('Model reference directory outside permitted stages')
            references, output = {}, {}
            for name in REFERENCE_FILES:
                expected = specification['files_sha256'][name]
                references[name] = self.copy(self.root, specification['model_dir'] + '/' + name,
                                              'references/' + run_name + '/' + name, expected)
            for name in RUN_FILES:
                output[name] = self.copy(self.stage, 'outputs/' + run_name + '/' + name, 'outputs/' + run_name + '/' + name)
            if any(v is None for v in list(references.values()) + list(output.values())):
                result['errors'].append('Missing/rejected required files; statistics not computed.')
                return
            config, saved_metrics = json_object(references['config.json']), json_object(references['metrics.json'])
            mapping_rows = csv_rows(references['item_id_map.csv'])
            item_map = {r['raw_item_id']: integer(r['item_idx']) for r in mapping_rows}
            if len(item_map) != len(mapping_rows) or set(item_map.values()) != set(range(1, len(item_map) + 1)):
                raise ValueError('Non-bijective item map')
            predictions = [parse_predictions(output[name], item_map) for name in PREDICTIONS]
            saved = parse_predictions(references['test_per_sample_predictions.csv'], item_map)
            labels = parse_labels(output['panel_labels.csv'])
            panel = identities(saved)
            if any(identities(p) != panel for p in predictions) or {k: v[:2] for k, v in labels.items()} != panel:
                raise ValueError('Sample/user/target panel mismatch')
            if len(panel) != integer(plan['expected_samples']):
                raise ValueError('Unexpected panel sample count')
            rank_mismatch = sum(saved[k][2] != predictions[0][k][2] for k in panel)
            top_mismatch = sum(saved[k][3:] != predictions[0][k][3:] for k in panel)
            if rank_mismatch or top_mismatch:
                raise ValueError('Canonical predictions do not reproduce saved reference')
            identical_mismatch = 0
            for sid in panel:
                seen = {}
                for value, sequence_sha in zip([p[sid] for p in predictions], labels[sid][4]):
                    if sequence_sha in seen and seen[sequence_sha] != value[2:]:
                        identical_mismatch += 1
                    seen[sequence_sha] = value[2:]
            if identical_mismatch:
                raise ValueError('Identical actual inputs produce different rank/Top10')
            actual_accuracy = verify_accuracy(saved_metrics, saved)
            meta = json_object(output['metadata.json'])
            expected_paths = [plan['original_path']] + [plan['permutation_paths'][str(p)] for p in PERMS]
            expected_data = {p: self.inputs['data_sha256'][p] for p in expected_paths}
            expected_checkpoint = specification['model_dir'] + '/' + specification['checkpoint']
            checks = {'dataset': meta.get('dataset') == plan['dataset'] == TAGS[tag] == config.get('dataset') == saved_metrics.get('dataset'),
                'seed': integer(meta.get('model_seed')) == seed == integer(config.get('seed')) == integer(saved_metrics.get('seed')),
                'epoch': integer(meta.get('best_epoch')) == integer(specification['best_epoch']) == integer(saved_metrics.get('best_epoch')),
                'num_items': integer(meta.get('num_items')) == len(item_map) == integer(plan['expected_items']) == integer(config.get('num_items')),
                'sample_count': integer(meta.get('test_samples')) == len(panel),
                'permutation_seeds': meta.get('permutation_seeds') == list(PERMS),
                'input_manifest_sha256': meta.get('input_manifest_sha256') == EXPECTED_INPUTS_SHA256,
                'preparation_manifest_sha256': meta.get('preparation_manifest_sha256') == self.manifest_sha,
                'reference_files_sha256': meta.get('reference_files_sha256') == specification['files_sha256'],
                'data_sha256': meta.get('data_sha256') == expected_data,
                'source_sha256': meta.get('source_sha256') == self.inputs['sources_sha256'],
                'model_dir': meta.get('model_dir') in (specification['model_dir'], str(self.root / specification['model_dir'])),
                'checkpoint_path_identity_only_not_opened': meta.get('checkpoint_path') in (expected_checkpoint, str(self.root / expected_checkpoint)),
                'model': meta.get('model') == config.get('model') == saved_metrics.get('model') == 'RecentBucketSetR1SASRec',
                'method': meta.get('method') == config.get('method') == saved_metrics.get('method') == 'RecentBucketSet_R1_V1_mean_pooling',
                'generated_evaluator_sha256': meta.get('generated_evaluator_sha256') == self.manifest['generated_sha256']['scripts/evaluate_%s_lbsp.py' % tag],
                'model_definition_sha256': meta.get('model_definition_sha256') == self.manifest['generated_sha256']['scripts/model_definition.py'],
                'training_reference_binding': meta.get('training_reference') == self.inputs['training_reference'],
                'training_model_definition_sha256': meta.get('training_model_definition_sha256') == self.training_manifest['generated_sha256']['scripts/model_definition.py'],
                'runtime_hashes_verified_before_after_server_record': meta.get('runtime_hashes_verified_before_after') is True,
                'strict_reproduction': meta.get('strict_reproduction') == 'PASS',
                'reference_test_ndcg': math.isclose(actual_accuracy['NDCG@10'], float(specification['test_ndcg10']), rel_tol=1e-10, abs_tol=1e-10),
                'checkpoint_epoch': integer(meta.get('checkpoint_epoch')) == integer(specification['best_epoch']),
                'original_rank_mismatches': integer(meta.get('original_rank_mismatches')) == rank_mismatch == 0,
                'original_top10_mismatches': integer(meta.get('original_top10_mismatches')) == top_mismatch == 0,
                'original_identity_raw_top10_mismatches': integer(meta.get('original_identity_raw_top10_mismatches')) == 0,
                'identical_input_prediction_consistency': identical_mismatch == 0,
                'exact_raw_logits_invariance_server_report': meta.get('exact_raw_logits_invariance') is True,
                'canonical_input_mismatches_server_report': integer(meta.get('canonical_input_mismatches')) == 0,
                'control_reference_binding': meta.get('control_reference') == self.inputs['control_reference'],
                'control_panel_labels_sha256': meta.get('control_panel_labels_sha256') == self.inputs['control_reference']['files_sha256']['outputs/' + run_name + '/panel_labels.csv'],
                'affected_samples': integer(meta.get('affected_samples')) == sum(v[3] for v in labels.values())}
            control_label = self.control_copies.get('outputs/' + run_name + '/panel_labels.csv')
            checks['raw_event_labels_equal_control'] = control_label is not None and parse_labels(control_label) == labels
            result['checks'] = checks
            before = meta.get('checkpoint_stat_before_inference')
            checks['checkpoint_stat_unchanged_server_record_only'] = (isinstance(before, list) and len(before) == 4
                and all(isinstance(v, int) and not isinstance(v, bool) for v in before) and before[2] > 0
                and before == meta.get('checkpoint_stat_after_inference'))
            training_provenance = config.get('amazon_lbsp_training', {})
            checks['saved_training_provenance_binding'] = (training_provenance.get('input_manifest_sha256') == self.inputs['training_reference']['input_manifest_sha256']
                and training_provenance.get('preparation_manifest_sha256') == self.inputs['training_reference']['manifest_sha256']
                and training_provenance.get('model_definition_sha256') == self.training_manifest['generated_sha256']['scripts/model_definition.py']
                and training_provenance.get('scientific_definitions_unchanged') is True and training_provenance.get('formal_rng_reset_preserved') is True)
            if not all(checks.values()):
                raise ValueError('Metadata identity mismatch: ' + ','.join(k for k, v in checks.items() if not v))
            invariance = csv_rows(output['per_seed_invariance.csv'])
            rows_by_seed = {integer(r['permutation_seed']): r for r in invariance}
            if len(invariance) != 5 or set(rows_by_seed) != set(PERMS):
                raise ValueError('Invariance table must contain exactly five permutation seeds')
            for i, perm in enumerate(PERMS, 1):
                row = rows_by_seed[perm]
                changed = sum(l[4][i] != l[4][0] for l in labels.values())
                if (integer(row['samples']) != len(panel) or integer(row['changed_sequences']) != changed
                        or integer(row['canonical_input_mismatches']) != 0
                        or float(row['max_abs_logit_diff']) != 0.0
                        or integer(row['changed_target_ranks']) != 0 or integer(row['changed_top10']) != 0):
                    raise ValueError('Canonical input/raw-logit invariance table failed')
                if changed and float(row.get('top10_overlap_changed', 1)) != 1.0:
                    raise ValueError('Changed-input Top10 overlap must equal one for invariant LBSP')
                if any(predictions[0][sid][2:] != predictions[i][sid][2:] for sid in panel):
                    raise ValueError('Saved LBSP predictions violate permutation rank/ordered-Top10 invariance')
            result.update(status='valid_saved_predictions', best_epoch=meta['best_epoch'], num_items=len(item_map),
                original_rank_derived_accuracy=actual_accuracy,
                invariance_server_table=invariance,
                populations={name: compare_population(predictions, labels, selected) for name, selected in
                    (('complete', list(panel)), ('affected', [k for k, v in labels.items() if v[3]]))})
            result['accuracy_by_variant'] = {name: accuracy(p) for name, p in zip(PREDICTIONS, predictions)}
            result['_baseline'], result['_labels'], result['_map'] = predictions[0], labels, item_map
            result['_config_core'] = {k: v for k, v in config.items() if k not in ('seed', 'orderfix', 'seed_extension', 'amazon_lbsp_training')}
        except (OSError, ValueError, KeyError, TypeError, SyntaxError, csv.Error) as exc:
            result['status'] = 'invalid_no_statistics'
            result.pop('populations', None)
            result['errors'].append(type(exc).__name__ + ': ' + str(exc)[:240])

    def control_run(self, tag, seed):
        key = 'control_%s_seed%d' % (tag, seed)
        result = {'tag': tag, 'method': 'Control', 'seed': seed,
                  'status': 'incomplete_no_statistics', 'checks': {}, 'errors': []}
        self.runs[key] = result
        try:
            name = '%s_seed%d' % (tag, seed)
            blobs = {n: self.control_copies.get('outputs/' + name + '/' + n) for n in CONTROL_RUN_FILES}
            if any(b is None for b in blobs.values()):
                result['errors'].append('Missing/rejected pinned Control outputs; statistics not computed.')
                return
            lbsp = self.runs[name]
            if lbsp['status'] != 'valid_saved_predictions':
                result['errors'].append('Valid LBSP reference map required for Control raw/index checks.')
                return
            item_map = lbsp['_map']
            labels = parse_labels(blobs['panel_labels.csv'])
            if labels != lbsp['_labels']:
                raise ValueError('Control/LBSP raw event labels differ')
            predictions = [parse_predictions(blobs[n], item_map) for n in PREDICTIONS]
            panel = identities(predictions[0])
            if (any(identities(p) != panel for p in predictions)
                    or {k: v[:2] for k, v in labels.items()} != panel
                    or panel != identities(lbsp['_baseline'])):
                raise ValueError('Control/LBSP panel mismatch')
            same_actual_input_check(predictions, labels)
            meta = json_object(blobs['metadata.json'])
            cm = self.control_manifest
            plan = cm['inputs']['datasets'][tag]
            planned = plan['runs'][str(seed)]
            paths = [plan['original_path']] + [plan['permutation_paths'][str(p)] for p in PERMS]
            data_hashes = {p: cm['inputs']['data_sha256'][p] for p in paths}
            checkpoint = planned['model_dir'] + '/' + planned['checkpoint']
            checks = {
                'dataset': meta.get('dataset') == TAGS[tag],
                'model': meta.get('model') == 'SASRecControl',
                'seed': integer(meta.get('model_seed')) == seed,
                'epoch': integer(meta.get('best_epoch')) == integer(planned['best_epoch']),
                'sample_count': integer(meta.get('test_samples')) == len(panel) == integer(plan['expected_samples']),
                'num_items': integer(meta.get('num_items')) == len(item_map),
                'permutation_seeds': meta.get('permutation_seeds') == list(PERMS),
                'input_manifest_sha256': meta.get('input_manifest_sha256') == self.inputs['control_reference']['input_manifest_sha256'],
                'preparation_manifest_sha256': meta.get('preparation_manifest_sha256') == self.inputs['control_reference']['manifest_sha256'],
                'reference_files_sha256': meta.get('reference_files_sha256') == planned['files_sha256'],
                'data_sha256_and_same_frozen_raw_inputs': meta.get('data_sha256') == data_hashes == {p: self.inputs['data_sha256'][p] for p in paths},
                'generated_evaluator_sha256': meta.get('generated_evaluator_sha256') == cm['generated_sha256']['scripts/eval_%s_stability.py' % tag],
                'model_definition_sha256': meta.get('model_definition_sha256') == cm['generated_sha256']['scripts/model_definition.py'],
                'model_dir': meta.get('model_dir') in (planned['model_dir'], str(self.root / planned['model_dir'])),
                'checkpoint_path_identity_only_not_opened': meta.get('checkpoint_path') in (checkpoint, str(self.root / checkpoint)),
                'raw_event_labels_equal_lbsp': labels == lbsp['_labels'],
                'affected_samples': integer(meta.get('affected_samples')) == sum(v[3] for v in labels.values()),
                'original_rank_mismatches_server_report': integer(meta.get('original_rank_mismatches')) == 0,
                'original_top10_mismatches_server_report': integer(meta.get('original_top10_mismatches')) == 0,
            }
            result['checks'] = checks
            if not all(checks.values()):
                raise ValueError('Pinned Control metadata mismatch: ' + ','.join(k for k, v in checks.items() if not v))
            result.update(status='valid_saved_predictions', best_epoch=meta['best_epoch'], num_items=len(item_map),
                original_rank_derived_accuracy=accuracy(predictions[0]),
                populations={name: compare_population(predictions, labels, selected) for name, selected in
                    (('complete', list(panel)), ('affected', [k for k, v in labels.items() if v[3]]))})
            result['accuracy_by_variant'] = {name: accuracy(p) for name, p in zip(PREDICTIONS, predictions)}
            result['_baseline'], result['_labels'], result['_map'] = predictions[0], labels, item_map
        except (OSError, ValueError, KeyError, TypeError, SyntaxError, csv.Error) as exc:
            result['status'] = 'invalid_no_statistics'
            result.pop('populations', None)
            result['errors'].append(type(exc).__name__ + ': ' + str(exc)[:240])

    def summaries(self, prefix=''):
        summaries = {}
        for tag in TAGS:
            runs = [self.runs['%s%s_seed%d' % (prefix, tag, s)] for s in SEEDS]
            data = {'status': 'incomplete_no_three_seed_summary'}
            summaries[tag] = data
            if not all(r['status'] == 'valid_saved_predictions' for r in runs):
                continue
            aligned = {'labels_equal_across_seeds': all(r['_labels'] == runs[0]['_labels'] for r in runs),
                       'maps_equal_across_seeds': all(r['_map'] == runs[0]['_map'] for r in runs),
                       'baseline_panels_equal_across_seeds': all(identities(r['_baseline']) == identities(runs[0]['_baseline']) for r in runs)}
            if not prefix:
                aligned['core_configs_equal_except_seed_and_provenance'] = all(r['_config_core'] == runs[0]['_config_core'] for r in runs)
            data['alignment'] = aligned
            if not all(aligned.values()):
                data['status'] = 'cross_seed_mismatch_no_summary'
                continue
            data.update(status='three_seed_saved_result_summary', populations={})
            for population in ('complete', 'affected'):
                labels = runs[0]['_labels']
                selected = [k for k in labels if population == 'complete' or labels[k][3]]
                metrics = {}
                for metric in runs[0]['populations'][population]:
                    if metric in ('N', 'distinct_pair_user_N', 'distinct_input_pair_N'):
                        continue
                    values = [r['populations'][population][metric] for r in runs]
                    metrics[metric] = {'mean': statistics.mean(values), 'sample_SD_ddof1': statistics.stdev(values)} if all(v is not None for v in values) else {'mean': None, 'sample_SD_ddof1': None}
                pairs = []
                for a, b in itertools.combinations(range(3), 2):
                    left, right = runs[a]['_baseline'], runs[b]['_baseline']
                    overlap = [len(set(left[k][3]) & set(right[k][3])) / 10 for k in selected]
                    flips = [(left[k][2] <= 10) != (right[k][2] <= 10) for k in selected]
                    pairs.append({'training_seeds': [SEEDS[a], SEEDS[b]], 'N': len(selected),
                                  'baseline_top10_overlap': mean(overlap), 'baseline_pair_hit_flip_fraction': mean(flips)})
                any_seed_flip = [len({r['_baseline'][k][2] <= 10 for r in runs}) > 1 for k in selected]
                data['populations'][population] = {'N': len(selected), 'three_seed_mean_sampleSD': metrics,
                    'canonical_accuracy_three_seed_mean_sampleSD': {
                        metric: {'mean': mean([accuracy(r['_baseline'], selected)[metric] for r in runs]) if selected else None,
                                 'sample_SD_ddof1': statistics.stdev([accuracy(r['_baseline'], selected)[metric] for r in runs])
                                 if selected else None}
                        for metric in ('NDCG@10', 'HR@10', 'HR@20', 'MeanRank')},
                    'distinct_input_pair_N': runs[0]['populations'][population]['distinct_input_pair_N'],
                    'distinct_pair_user_N': runs[0]['populations'][population]['distinct_pair_user_N'],
                    'training_seed_baseline_pairs_reference': pairs,
                    'training_seed_baseline_three_pair_overlap_mean_reference': mean([p['baseline_top10_overlap'] for p in pairs]) if selected else None,
                    'training_seed_baseline_three_pair_hit_flip_mean_reference': mean([p['baseline_pair_hit_flip_fraction'] for p in pairs]) if selected else None,
                    'training_seed_baseline_any_user_flip_across_three_seeds_reference': mean(any_seed_flip)}
        return summaries

    def method_comparison(self, lbsp, control):
        comparisons = {}
        for tag in TAGS:
            result = {'status': 'incomplete_no_method_comparison',
                      'scope': 'Descriptive saved-model comparison on identical samples; no significance, equivalence, or causal claim.'}
            comparisons[tag] = result
            if any(s.get('status') != 'three_seed_saved_result_summary' for s in (lbsp.get(tag, {}), control.get(tag, {}))):
                continue
            all_runs = [self.runs['%s%s_seed%d' % (prefix, tag, seed)] for prefix in ('', 'control_') for seed in SEEDS]
            first = all_runs[0]
            alignment = {'all_six_raw_event_labels_equal': all(r['_labels'] == first['_labels'] for r in all_runs),
                         'all_six_sample_user_target_panels_equal': all(identities(r['_baseline']) == identities(first['_baseline']) for r in all_runs),
                         'lbsp_maps_equal_and_observed_control_indices_validate': all(r['_map'] == first['_map'] for r in all_runs)}
            result['alignment'] = alignment
            if not all(alignment.values()):
                result['status'] = 'population_mismatch_no_method_comparison'
                continue
            result.update(status='two_method_three_seed_saved_result_comparison', populations={})
            for population in ('complete', 'affected'):
                a = lbsp[tag]['populations'][population]
                b = control[tag]['populations'][population]
                if a['N'] != b['N']:
                    raise ValueError('Method comparison population counts disagree')
                result['populations'][population] = {'N': a['N'], 'methods': {'LBSP': a, 'Control': b},
                    'descriptive_mean_difference_LBSP_minus_Control': {
                        metric: (a['three_seed_mean_sampleSD'][metric]['mean'] - b['three_seed_mean_sampleSD'][metric]['mean'])
                                if a['N'] else None
                        for metric in ('overlap_5perm_pair_mean', 'hit_flip_user_fraction')}}
        return comparisons

    def driver(self):
        names = [k for k in self.copies if re.fullmatch(r'verification/lbsp_amazon_stability_driver_[^/]+\.json', k)]
        self.driver_check = {'status': 'MISSING' if not names else 'FAIL', 'errors': []}
        if len(names) != 1:
            self.driver_check['errors'].append('Exactly one fixed-model stability driver record is required for all-nine status.')
            return
        try:
            d = json_object(read_stable(self.out, 'snapshot/' + names[0])[0])
            checks = {'stage': d.get('stage') == str(self.stage), 'version': d.get('version') == 'lbsp_amazon_stability_v1',
                'input_manifest_sha256': d.get('inputs_sha256') == EXPECTED_INPUTS_SHA256,
                'training_seeds': d.get('training_seeds') == list(SEEDS), 'permutation_seeds': d.get('permutation_seeds') == list(PERMS),
                'training_performed_false': d.get('training_performed') is False,
                'all_nine_driver_status': d.get('status') == 'ALL_NINE_EVALUATIONS_EXITED_ZERO',
                'exact_nine_runs': set(d.get('runs', {})) == {'%s_seed%d' % (tag, seed) for tag in TAGS for seed in SEEDS}}
            for tag in TAGS:
                for seed in SEEDS:
                    name = '%s_seed%d' % (tag, seed)
                    r = d['runs'][name]
                    expected = [d['python'], '-u', str(self.stage / ('scripts/evaluate_%s_lbsp.py' % tag)), '--seed', str(seed)]
                    log = Path(r['log'])
                    checks[name + '_command'] = r.get('command') == expected
                    checks[name + '_exit_zero'] = integer(r.get('exit_code')) == 0
                    checks[name + '_metadata_verified'] = r.get('metadata_verified') is True
                    checks[name + '_saved_predictions_valid'] = self.runs[name]['status'] == 'valid_saved_predictions'
                    checks[name + '_log_collected'] = log.is_absolute() and log.parent == self.stage / 'verification' and 'verification/' + log.name in self.copies
            self.driver_check.update(status='PASS' if all(checks.values()) else 'FAIL', path=names[0], checks=checks)
            if not all(checks.values()):
                self.driver_check['errors'].append('Driver exits/metadata and the saved evidence do not consistently establish all nine evaluations.')
        except (KeyError, TypeError, ValueError) as exc:
            self.driver_check['errors'].append(type(exc).__name__ + ': ' + str(exc)[:240])

    def finish(self, summaries, control_summaries=None, comparison=None):
        control_summaries = control_summaries or {}
        comparison = comparison or {}
        public_runs = {name: {k: v for k, v in run.items() if not k.startswith('_')} for name, run in self.runs.items()}
        complete = (len(public_runs) == 18 and all(r['status'] == 'valid_saved_predictions' for r in public_runs.values())
                    and len(comparison) == 3 and all(v.get('status') == 'two_method_three_seed_saved_result_comparison' for v in comparison.values())
                    and self.driver_check.get('status') == 'PASS')
        status = 'INTEGRITY_FAILED_NO_BUNDLE' if self.failures else ('ALL_NINE_LBSP_RUNS_CHECKED' if complete else 'INCOMPLETE')
        result = {'status': status, 'stage': str(self.stage), 'input_manifest_sha256': EXPECTED_INPUTS_SHA256,
            'run_checks': public_runs, 'dataset_summaries': summaries, 'control_summaries': control_summaries,
            'method_comparison': comparison, 'files': self.files, 'warnings': self.warnings,
            'training_reference_check': getattr(self, 'training_check', {'status': 'NOT_COMPLETED'}),
            'algorithm_ast_check': getattr(self, 'algorithm_check', {'status': 'NOT_COMPLETED'}), 'driver_check': self.driver_check,
            'failures': self.failures, 'formulas': {
                'overlap_5perm_pair_mean': 'For each sample average all 10 unordered pairs of five permuted Top10 sets (intersection/10), then average samples; repeated input pairs retained.',
                'hit_flip_user_fraction': 'Fraction of samples with at least one hit@10 and one nonhit among FIVE PERMUTATIONS; original prediction excluded.',
                'affected': 'Five actual permuted input sequence hashes contain at least two distinct values; original is excluded from this definition.',
                'baseline_diagnostics': 'Canonical-versus-five-permutation metrics are diagnostics, separate from primary five-permutation metrics.',
                'training_seed_reference': 'Three unordered pairs of canonical-input model predictions, using the same complete/affected sample population; never pooled with input-permutation results.',
                'empty_population': 'N=0 yields null means/SD; no division by zero.',
                'three_seed_SD': 'Sample standard deviation across the three fixed model seeds, ddof=1.'},
            'scope': 'PRIVATE saved-file verification; no training, inference, checkpoint access, or interaction-data reads. Raw-event sequence hashes and raw-logit/canonical-input invariance are generated server provenance, not independently reconstructed here. Control outputs/manifest are pinned to the earlier audited report. No full reproduction, significance, equivalence or best-seed claim.'}
        (self.out / 'stability_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        fields = ['origin', 'path', 'archive_path', 'status', 'bytes', 'sha256', 'error']
        with (self.out / 'COLLECTED_SHA256.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.files)
        summary_fields = ['dataset', 'population', 'N', 'method', 'overlap_mean', 'overlap_sample_SD',
                          'hit_flip_mean', 'hit_flip_sample_SD', 'NDCG10_mean', 'NDCG10_sample_SD', 'HR10_mean', 'HR10_sample_SD']
        with (self.out / 'method_comparison_summary.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fields)
            writer.writeheader()
            for tag, compared in comparison.items():
                if compared.get('status') != 'two_method_three_seed_saved_result_comparison':
                    continue
                for pop, info in compared['populations'].items():
                    for method, data in info['methods'].items():
                        stability = data['three_seed_mean_sampleSD']
                        acc = data['canonical_accuracy_three_seed_mean_sampleSD']
                        writer.writerow({'dataset': TAGS[tag], 'population': pop, 'N': info['N'], 'method': method,
                            'overlap_mean': stability['overlap_5perm_pair_mean']['mean'],
                            'overlap_sample_SD': stability['overlap_5perm_pair_mean']['sample_SD_ddof1'],
                            'hit_flip_mean': stability['hit_flip_user_fraction']['mean'],
                            'hit_flip_sample_SD': stability['hit_flip_user_fraction']['sample_SD_ddof1'],
                            'NDCG10_mean': acc['NDCG@10']['mean'], 'NDCG10_sample_SD': acc['NDCG@10']['sample_SD_ddof1'],
                            'HR10_mean': acc['HR@10']['mean'], 'HR10_sample_SD': acc['HR@10']['sample_SD_ddof1']})
        lines = ['# PRIVATE Amazon LBSP stability and Control comparison evidence review', '', 'Status: **' + status + '**', '', result['scope'], '',
            'Primary overlap averages ten pairs per sample, then samples. Primary hit_flip_user_fraction is ANY hit-status variation across the FIVE permutations, not a pair-flip mean.',
            'Affected masks use distinct FIVE permutation inputs and are checked identical across model seeds. Empty affected populations produce null metrics.',
            'For each dataset three LBSP and three pinned prior Control outputs use identical raw-event labels and complete/affected populations. Method differences are descriptive only.',
            'LBSP exact raw-logit/canonical-input invariance is a checked server-generated table; this collector independently checks all saved target ranks and ordered Top10, not logits.',
            'Missing/invalid runs are collected where possible; they receive no statistics and prevent their dataset three-seed summary.', '', '## Runs', '']
        lines += ['- ' + k + ': ' + v['status'] for k, v in public_runs.items()]
        lines += ['', '## Findings', ''] + ['- ' + x for x in self.failures + self.warnings]
        lines += ['', 'See stability_checks.json for all populations, three-seed SD and separately named training-seed references.',
                  'Prediction CSVs/labels/references contain user-level material. Do not publish this private ZIP directly.', '']
        (self.out / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
        bundle = None
        if not self.failures:
            bundle = self.out / 'lbsp_amazon_stability_results.zip'
            paths = sorted(p for p in self.out.rglob('*') if p.is_file())
            with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for path in paths:
                    if path.is_symlink():
                        raise ValueError('Output symbolic link rejected')
                    archive.write(path, path.relative_to(self.out).as_posix())
            h = hashlib.sha256()
            with bundle.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    h.update(block)
            (self.out / 'lbsp_amazon_stability_results.zip.sha256.txt').write_text(h.hexdigest() + '  ' + bundle.name + '\n', encoding='ascii')
        return {'report_dir': str(self.out), 'zip_path': str(bundle) if bundle else None, 'status': status, 'exit_code': 2 if self.failures else 0 if complete else 1}


def collect(stage):
    collector = Collector(stage)
    summaries = {}
    try:
        collector.verify_sources()
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        collector.failures.append(type(exc).__name__ + ': ' + str(exc)[:240])
    if not collector.failures:
        try:
            collector.control_reference()
            collector.training_reference()
        except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
            collector.failures.append(type(exc).__name__ + ': ' + str(exc)[:240])
    control_summaries, comparison = {}, {}
    if not collector.failures:
        for path in sorted(safe(collector.stage, 'verification').iterdir()):
            if not path.name.startswith('.') and path.suffix.lower() in ('.log', '.json') and path.is_file():
                collector.copy(collector.stage, 'verification/' + path.name, 'verification/' + path.name)
        for tag in TAGS:
            for seed in SEEDS:
                collector.run(tag, seed)
        for tag in TAGS:
            for seed in SEEDS:
                collector.control_run(tag, seed)
        summaries = collector.summaries()
        control_summaries = collector.summaries('control_')
        comparison = collector.method_comparison(summaries, control_summaries)
        collector.driver()
    return collector.finish(summaries, control_summaries, comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.stage.expanduser())
    checks_path = Path(result['report_dir']) / 'stability_checks.json'
    print('SUMMARY=' + str(Path(result['report_dir']) / 'SUMMARY.md'))
    print('CHECKS=' + str(checks_path))
    print('LBSP_AMAZON_STABILITY_REPORT_ZIP=' + (result['zip_path'] or 'NOT_CREATED'))
    print('STATUS=' + result['status'])
    report = json_object(checks_path.read_bytes())
    for failure in report['failures']:
        print('INTEGRITY_FAILURE=' + failure)
    for name, run in report['run_checks'].items():
        for error in run['errors']:
            print('RUN_FAILURE=' + name + ': ' + error)
        for key, passed in run['checks'].items():
            if not passed:
                print('RUN_FAILED_CHECK=' + name + ': ' + key)
    for failure in report['driver_check'].get('errors', []):
        print('DRIVER_FAILURE=' + failure)
    return result['exit_code']


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        print('COLLECTION_FAILED=' + type(exc).__name__ + ': ' + str(exc)[:240], file=sys.stderr)
        sys.exit(2)
