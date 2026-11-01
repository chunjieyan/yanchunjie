#!/usr/bin/env python3
"""Collect X3 saved inference/ensemble records; standard library and metadata only."""
import argparse
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
import struct
import sys
import uuid
import zipfile

VERSION = 'x3_inference_v1'
INPUT_NAME = 'x3_inference_inputs_20261026.json'
INPUT_SHA256 = '2d0e5cbb7f537fb9989b30fbf60695f8bfa5819f0d7f409ebe9d3786dd9f46a4'
TAGS = ('ml1m', 'beauty', 'sports', 'toys')
METHODS = ('shuffle', 'ensemble')
SEEDS = (42, 2026, 3407)
PERMS = (101, 202, 303, 404, 505)
METRICS = ('ndcg10', 'hr10', 'hr20', 'recall50', 'mrr', 'mrr10', 'mean_rank')
GROUP_SEEDS = {i: tuple(p + 1000 * (i - 1) for p in PERMS) for i in range(1, 6)}
BASE_OUTPUTS = ('config.json', 'item_id_map.csv', 'predictions_original.csv', 'metrics.json', 'stability.json', 'panel_labels.csv', 'input_view_hashes.csv', 'reproduction.json')
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

def close(a, b, tolerance=1e-10):
    return math.isclose(finite(a), finite(b), abs_tol=tolerance, rel_tol=tolerance)

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def mapping(blob):
    rows = csv_rows(blob)
    if not rows or set(rows[0]) != {'raw_item_id', 'item_idx'}:
        raise ValueError('Unexpected item map columns')
    result = {r['raw_item_id']: integer(r['item_idx']) for r in rows}
    if len(result) != len(rows) or set(result.values()) != set(range(1, len(rows) + 1)):
        raise ValueError('Item map is not a contiguous bijection')
    return result

def predictions(blob, item_map, expected_count):
    result = {}
    for row in csv_rows(blob):
        if set(row) != {'sample_id', 'user_id', 'target_item', 'target_idx', 'rank', 'top10_item_idx', 'top10_raw_item_id', 'candidate_count'}:
            raise ValueError('Prediction column inventory differs')
        sid, user, target = row['sample_id'], row['user_id'], row['target_item']
        rank, count, target_idx = integer(row['rank']), integer(row['candidate_count']), integer(row['target_idx'])
        top = tuple(integer(x) for x in json.loads(row['top10_item_idx']))
        raw = tuple(str(x) for x in json.loads(row['top10_raw_item_id']))
        if not sid or not user or sid in result or item_map.get(target) != target_idx:
            raise ValueError('Prediction identity/target mapping invalid')
        if not 10 <= count <= len(item_map) or not 1 <= rank <= count:
            raise ValueError('Prediction rank/candidate bounds invalid')
        if len(top) != 10 or len(raw) != 10 or len(set(top)) != 10 or len(set(raw)) != 10 or any(item_map.get(x) != y for x, y in zip(raw, top)):
            raise ValueError('Prediction ordered Top10 shape/map invalid')
        result[sid] = {'user': user, 'target': target, 'target_idx': target_idx, 'rank': rank, 'candidate_count': count, 'top': top}
    if len(result) != integer(expected_count):
        raise ValueError('Prediction sample count differs from frozen plan')
    return result

def identity(rows):
    return {sid: (r['user'], r['target'], r['target_idx'], r['candidate_count']) for sid, r in rows.items()}

def rank_metrics(rows):
    ranks = [r['rank'] for r in rows.values()]
    if not ranks:
        raise ValueError('Empty accuracy population')
    n = len(ranks)
    return {'ndcg10': math.fsum(1 / math.log2(r + 1) for r in ranks if r <= 10) / n,
            'hr10': sum(r <= 10 for r in ranks) / n, 'hr20': sum(r <= 20 for r in ranks) / n,
            'recall50': sum(r <= 50 for r in ranks) / n, 'mrr': math.fsum(1 / r for r in ranks) / n,
            'mrr10': math.fsum(1 / r for r in ranks if r <= 10) / n, 'mean_rank': sum(ranks) / n}

def rank_top_diagnostics(rows):
    return {'rank_le10_vs_membership_mismatches': sum((r['rank'] <= 10) != (r['target_idx'] in r['top']) for r in rows.values()),
            'target_top10_position_vs_rank_mismatches': sum(r['top'].index(r['target_idx']) + 1 != r['rank'] for r in rows.values() if r['target_idx'] in r['top']),
            'interpretation': 'Strict-greater-plus-one ranks are preserved. Independent torch.topk differences are compatible with score ties but do not establish ties without score counts.'}

def population_stability(panels, affected):
    first = panels[0]
    if len(panels) < 2 or any(identity(p) != identity(first) for p in panels):
        raise ValueError('Stability panels do not share the same population/candidates')
    result = {}
    pairs = len(panels) * (len(panels) - 1) // 2
    for population, ids in (('full', list(first)), ('affected', [sid for sid in first if affected[sid]])):
        users = []
        for sid in ids:
            values = [p[sid] for p in panels]
            tops = [v['top'] for v in values]
            overlap = math.fsum(len(set(a) & set(b)) / 10 for a, b in itertools.combinations(tops, 2)) / pairs
            users.append({'sample_id': sid, 'overlap10': overlap,
                'any_hitflip': int(len({v['rank'] <= 10 for v in values}) > 1),
                'top10_all_identical': int(all(t == tops[0] for t in tops[1:]))})
        result[population] = {'n': len(ids), 'overlap10': math.fsum(u['overlap10'] for u in users) / len(ids) if ids else None,
            'hitflip': sum(u['any_hitflip'] for u in users) / len(ids) if ids else None,
            'unordered_pairs_per_user': pairs, 'users': users}
    return result

def compare_metrics(actual, saved, context):
    if set(saved) != set(METRICS) or any(not close(actual[k], saved[k]) for k in METRICS):
        raise ValueError(context + ': saved metrics disagree with rank-derived seven metrics')

def output_names(method):
    if method == 'shuffle':
        return set(BASE_OUTPUTS) | {'predictions_perm_seed%d.csv' % p for p in PERMS}
    if method == 'ensemble':
        return set(BASE_OUTPUTS) | {'predictions_group%d.csv' % g for g in range(1, 6)} | {'predictions_view_seed%d.csv' % p for p in PERMS} | {'score_mean_probes.json'}
    raise ValueError('Unknown inference method')


def panel_names(method):
    return ('original',) + (tuple('perm_seed%d' % p for p in PERMS) if method == 'shuffle' else tuple('group%d' % g for g in range(1, 6)) + tuple('view_seed%d' % p for p in PERMS))


def reference_predictions(blob, item_map):
    result = {}
    for row in csv_rows(blob):
        sid, user, target = row['sample_id'], row['user_id'], row['target_item']
        rank = integer(row.get('target_rank', row.get('rank')))
        raw = tuple(str(x) for x in json.loads(row['top10_raw_item_id']))
        top = tuple(integer(x) for x in json.loads(row['top10_item_idx']))
        if not sid or not user or sid in result or target not in item_map or not 1 <= rank <= len(item_map):
            raise ValueError('Reference prediction identity/rank invalid')
        if len(top) != len(raw) or len(top) != 10 or len(set(top)) != 10 or any(item_map.get(r) != i for r, i in zip(raw, top)):
            raise ValueError('Reference ordered Top10 map invalid')
        result[sid] = (user, target, rank, top)
    if not result:
        raise ValueError('Empty reference prediction panel')
    return result


def reproduction_compare(rows, saved):
    if set(rows) != set(saved):
        raise ValueError('Reference/new prediction sample populations differ')
    counts = {'identity_mismatches': 0, 'rank_mismatches': 0, 'top10_mismatches': 0}
    for sid, row in rows.items():
        user, target, rank, top = saved[sid]
        counts['identity_mismatches'] += int((row['user'], row['target']) != (user, target))
        counts['rank_mismatches'] += int(row['rank'] != rank)
        counts['top10_mismatches'] += int(row['top'] != top)
    if any(counts.values()):
        raise ValueError('Saved canonical/view predictions not strictly reproduced: ' + repr(counts))
    return counts


def rich_labels(blob, rows, expected_affected):
    required = {'sample_id', 'user_id', 'target_item', 'last_bucket_size', 'affected', 'distinct_sequences_5perm', 'original_sequence_sha256'} | {'perm_seed%d_sequence_sha256' % p for p in PERMS}
    result = {}
    for row in csv_rows(blob):
        sid = row['sample_id']
        if set(row) != required or sid not in rows or sid in result:
            raise ValueError('Affected-label identity/column inventory invalid')
        if (row['user_id'], row['target_item']) != (rows[sid]['user'], rows[sid]['target']):
            raise ValueError('Affected-label user/target identity differs')
        k, affected = integer(row['last_bucket_size']), integer(row['affected'])
        hashes = [row['perm_seed%d_sequence_sha256' % p] for p in PERMS]
        if (not 1 <= k <= 50 or affected not in (0, 1) or any(re.fullmatch('[0-9a-f]{64}', h) is None for h in hashes + [row['original_sequence_sha256']])
                or integer(row['distinct_sequences_5perm']) != len(set(hashes)) or affected != int(len(set(hashes)) >= 2)):
            raise ValueError('Affected is not distinct raw five-permutation histories excluding original')
        result[sid] = dict(row, affected=bool(affected), last_bucket_size=k, distinct_sequences_5perm=len(set(hashes)))
    if set(result) != set(rows) or sum(v['affected'] for v in result.values()) != integer(expected_affected):
        raise ValueError('Affected count/population differs from frozen plan')
    return result


def input_views(blob, labels, method):
    result = {}
    expected = {(sid, 0, 0) for sid in labels}
    groups = (1,) if method == 'shuffle' else range(1, 6)
    expected |= {(sid, g, p) for sid in labels for g in groups for p in GROUP_SEEDS[g]}
    for row in csv_rows(blob):
        if set(row) != {'sample_id', 'group', 'permutation_seed', 'sequence_sha256', 'joint_sha256'}:
            raise ValueError('Input-view hash column inventory differs')
        sid, group = row['sample_id'], integer(row['group'])
        if group == 0 and row['permutation_seed'] != 'original':
            raise ValueError('Canonical input-view row must use the original marker')
        seed = 0 if group == 0 and row['permutation_seed'] == 'original' else integer(row['permutation_seed'])
        key = (sid, group, seed)
        if key not in expected or key in result or any(re.fullmatch('[0-9a-f]{64}', row[k]) is None for k in ('sequence_sha256', 'joint_sha256')):
            raise ValueError('Input-view hash identity/inventory/hash syntax invalid')
        if group in (0, 1):
            field = 'original_sequence_sha256' if group == 0 else 'perm_seed%d_sequence_sha256' % seed
            if labels[sid][field] != row['sequence_sha256']:
                raise ValueError('Canonical/g1 view hash differs from frozen five-permutation labels')
        result[key] = (row['sequence_sha256'], row['joint_sha256'])
    if set(result) != expected:
        raise ValueError('Input-view ledger must retain every logical member, including repeated inputs')
    return result


def same_input_predictions(panels, views, method):
    checked = 0
    names = [('original', 0, 0)] + ([('perm_seed%d' % p, 1, p) for p in PERMS] if method == 'shuffle' else [('view_seed%d' % p, 1, p) for p in PERMS])
    for sid in panels['original']:
        for left, right in itertools.combinations(names, 2):
            a, ag, ap = left;b, bg, bp = right
            if views[(sid, ag, ap)][0] == views[(sid, bg, bp)][0]:
                checked += 1
                if panels[a][sid]['rank'] != panels[b][sid]['rank'] or panels[a][sid]['top'] != panels[b][sid]['top']:
                    raise ValueError('Identical recorded raw Control/Shuffle model input produced different saved rank/orderedTop10')
    if method == 'ensemble':
        for sid in panels['original']:
            for a, b in itertools.combinations(range(1, 6), 2):
                if tuple(views[(sid, a, p)][0] for p in GROUP_SEEDS[a]) == tuple(views[(sid, b, p)][0] for p in GROUP_SEEDS[b]):
                    checked += 1
                    if panels['group%d' % a][sid]['rank'] != panels['group%d' % b][sid]['rank'] or panels['group%d' % a][sid]['top'] != panels['group%d' % b][sid]['top']:
                        raise ValueError('Identical ordered ensemble member inputs produced different saved group prediction')
    return checked


def compare_stability(actual, saved):
    if set(saved) != {'full', 'affected'}:
        raise ValueError('Stability population inventory differs')
    for population, row in actual.items():
        value = saved[population]
        if integer(value['n']) != row['n']:
            raise ValueError('Stability population count differs')
        for k in ('overlap10', 'hitflip'):
            if row[k] is None:
                if value[k] is not None:raise ValueError('Empty population measurement must be null')
            elif not close(row[k], value[k]):raise ValueError('Stability aggregate differs: ' + population + '/' + k)
        by_id = {u['sample_id']: u for u in value['users']}
        if len(by_id) != len(value['users']) or set(by_id) != {r['sample_id'] for r in row['users']}:
            raise ValueError('Stability per-user identities differ')
        for user in row['users']:
            v = by_id[user['sample_id']]
            if not close(v['overlap10'], user['overlap10']) or integer(v['any_hitflip']) != user['any_hitflip'] or integer(v['top10_all_identical']) != user['top10_all_identical']:
                raise ValueError('Stability per-user ten-pair overlap/any-hitflip differs')

def score_probes(value, panels):
    if (value.get('scope') != 'first at most16 canonical samples per group, not all candidate scores'
            or value.get('accumulation_dtype') != 'torch.float64' or value.get('member_score_dtype') != 'torch.float32'
            or value.get('reduction_order') != [list(GROUP_SEEDS[g]) for g in range(1, 6)]
            or value.get('raw_scores_averaged_before_mask') is not True or value.get('scores_averaged_not_ranks') is not True
            or value.get('legacy_first_group_input_reproduced') is not True or value.get('numpy_bit_generator') != 'PCG64'):
        raise ValueError('Score-probe dtype/source/reduction-order/before-mask declaration differs')
    wanted_ids = list(panels['original'])[:16]
    expected = {(sid, group) for sid in wanted_ids for group in range(1, 6)}
    seen = set()
    compared = 0
    for probe in value['probes']:
        sid, group = probe['sample_id'], integer(probe['group'])
        key = (sid, group)
        if key in seen or key not in expected or probe['member_seeds'] != list(GROUP_SEEDS[group]):
            raise ValueError('Bounded score-probe identity/member seed order differs')
        seen.add(key);prediction = panels['group%d' % group][sid]
        target, top = integer(probe['target_idx']), tuple(integer(i) for i in probe['top10_item_idx'])
        indices = [integer(i) for i in probe['candidate_indices']]
        if target != prediction['target_idx'] or top != prediction['top'] or indices != [target] + [i for i in top if i != target]:
            raise ValueError('Bounded score-probe target/Top10 candidate identities differ')
        members, means = probe['member_scores'], probe['mean_scores']
        if len(members) != 5 or len(means) != len(indices) or any(len(v) != len(indices) for v in members):
            raise ValueError('Score-probe member/mean shape invalid')
        for column, saved in enumerate(means):
            numbers = [finite(v[column]) for v in members]
            if any(struct.unpack('!f', struct.pack('!f', n))[0] != n for n in numbers):
                raise ValueError('Declared raw float32 score is not exactly representable as float32')
            actual = math.fsum(numbers) / 5
            if not close(actual, saved, 1e-12):
                raise ValueError('Bounded raw-score mean differs from five-member float64 arithmetic')
            compared += 1
        lookup = {idx: finite(v) for idx, v in zip(indices, means)}
        if any(lookup[a] < lookup[b] for a, b in zip(top, top[1:])):
            raise ValueError('Bounded group Top10 mean scores are not nonincreasing')
        if sum(lookup[i] > lookup[target] for i in top) != min(10, prediction['rank'] - 1):
            raise ValueError('Bounded mean-score target comparisons disagree with strict-greater rank')
    if seen != expected:
        raise ValueError('All five groups require exactly the first min(16,N) canonical score probes')
    return {'status': 'PASS_BOUNDED_SAVED_ARITHMETIC', 'groups': 5, 'samples_per_group': len(wanted_ids),
            'candidate_means_compared': compared,
            'scope': 'Saved first-at-most16 samples/group only. Float32 representability and five-member mean arithmetic checked; no full-catalog/model logits independently recomputed.'}


class Collector:
    def __init__(self, stage):
        self.stage = absolute_plain(stage)
        if not self.stage.name.startswith('kais_x3_inference_v1_'):
            raise ValueError('Unexpected X3 inference stage prefix')
        verification = safe(self.stage, 'verification')
        if not verification.is_dir():raise ValueError('Verification directory missing')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
        self.out = verification / ('x3_inference_review_' + stamp);self.out.mkdir(exist_ok=False)
        self.files, self.copies, self.integrity_failures, self.warnings = [], {}, [], []
        self.total, self.runs = 0, {}
        self.driver_check = {'status': 'NOT_CHECKED', 'errors': []}

    def copy(self, relative, expected=None):
        record = {'path': relative, 'status': 'unknown'}
        try:
            blob, digest = read_stable(self.stage, relative)
            if expected is not None and digest != expected:raise ValueError('SHA256 mismatch')
            if self.total + len(blob) > TOTAL_LIMIT:raise ValueError('Total size limit exceeded')
            path = safe(self.out, 'snapshot/' + relative);path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f:f.write(blob)
            self.copies[relative] = blob;self.total += len(blob);record.update(status='copied', bytes=len(blob), sha256=digest)
            return blob
        except (OSError, ValueError) as exc:
            record.update(status='missing' if isinstance(exc, FileNotFoundError) else 'rejected_or_changed', error=str(exc)[:340])
            self.warnings.append(record['status'] + ': ' + relative)
            if expected is not None and isinstance(exc, ValueError):self.integrity_failures.append('Pinned file rejected/changed: ' + relative)
            return None
        finally:self.files.append(record)

    def sources(self):
        raw, self.manifest_sha = read_stable(self.stage, 'manifest.json');self.manifest = m = json_object(raw)
        raw, digest = read_stable(self.stage, INPUT_NAME)
        if digest != INPUT_SHA256:raise ValueError('Reviewed X3 inference input seal SHA mismatch')
        self.inputs = inputs = json_object(raw);self.root = absolute_plain(m['root'])
        if (m.get('version') != VERSION or self.stage.parent != self.root or m.get('stage') != str(self.stage)
                or m.get('input_sha256') != INPUT_SHA256 or m.get('root_files_sha256') != inputs['root_files_sha256']
                or m.get('protocol') != inputs['protocol'] or inputs.get('version') != VERSION
                or inputs.get('training_seeds') != list(SEEDS) or inputs.get('groups') != [list(GROUP_SEEDS[g]) for g in range(1, 6)]
                or set(inputs['datasets']) != set(TAGS)):
            raise ValueError('Stage/input/root-file/seed/group/protocol closure differs')
        helpers = inputs['helper_sha256']
        if set(helpers) != {'x3_inference_adapter.py', 'x3_inference_protocol.py', 'x3_control_model.py'}:
            raise ValueError('Expected three non-circular helper pins')
        inventory = {'scripts/' + n for n in helpers} | {'scripts/run_x3_inference.py', 'scripts/collect_x3_inference.py', INPUT_NAME}
        if set(m['tools_sha256']) != inventory or m['tools_sha256'][INPUT_NAME] != INPUT_SHA256:
            raise ValueError('Expected exactly six stage tool/input files')
        for relative, expected in inputs['root_files_sha256'].items():
            relative_path(relative)
            if re.fullmatch('[0-9a-f]{64}', expected) is None:raise ValueError('Root-file seal hash syntax invalid')
        references = inputs['references']
        if m['references_sha256'] != {p: r['sha256'] for p, r in references.items()}:
            raise ValueError('Copied reference inventory/hash closure differs')
        expected_tasks = [{'method': method, 'tag': tag, 'seed': seed} for method in METHODS for tag in TAGS for seed in SEEDS]
        if m.get('tasks') != expected_tasks:raise ValueError('Expected exactly twenty-four inference tasks')
        referenced = set()
        for tag, plan in inputs['datasets'].items():
            if set(plan['runs']) != set(METHODS):raise ValueError('Dataset method inventory differs')
            for method in METHODS:
                if set(plan['runs'][method]) != {str(s) for s in SEEDS}:raise ValueError('Exactly three training seeds required')
                for seed in SEEDS:
                    run = plan['runs'][method][str(seed)]
                    for field in ('reference_config', 'reference_map', 'reference_predictions'):referenced.add(run[field])
                    if method == 'ensemble':
                        if set(run['reference_permutations']) != {str(s) for s in PERMS}:raise ValueError('Five sealed old Control view references required')
                        referenced.update(run['reference_permutations'].values())
                    if not set(run['required_root_files']) <= set(inputs['root_files_sha256']):raise ValueError('Run root-file requirements exceed reviewed seal')
                    relative_path(run['model_dir']);relative_path(run['model_dir'] + '/' + run['checkpoint'])
                    if integer(run['best_epoch']) < 1:raise ValueError('Reference best epoch invalid')
        if not referenced <= set(references):raise ValueError('Run reference paths not present in sealed reference inventory')
        for name, expected in helpers.items():
            if m['tools_sha256']['scripts/' + name] != expected:raise ValueError('Reviewed helper differs from prepared byte hash')
        for relative, expected in m['tools_sha256'].items():
            if self.copy(relative, expected) is None:raise ValueError('Tool/input missing/rejected: ' + relative)
        executing = Path(__file__).absolute();blob, digest = read_stable(absolute_plain(executing.parent), executing.name)
        if digest != m['tools_sha256']['scripts/collect_x3_inference.py']:raise ValueError('Executing collector differs from stage collector')
        for relative, record in references.items():
            if (not relative_path(relative).startswith('references/') or PurePosixPath(relative).suffix not in ('.csv', '.json')
                    or PurePosixPath(relative).name in ('train.csv', 'test.csv', 'val.csv', 'r1_samples.csv', 'file_order.csv')):
                raise ValueError('Reference path is not whitelisted saved model metadata')
            if inputs['root_files_sha256'].get(record['root_path']) != record['sha256']:raise ValueError('Stage reference not pinned to a reviewed root metadata seal')
            if self.copy(relative, record['sha256']) is None:raise ValueError('Required saved reference missing/rejected')
        if self.copy('manifest.json', self.manifest_sha) is None:raise ValueError('Manifest changed during collection')
        self.source_check = {'status': 'PASS', 'stage_tools': len(inventory), 'saved_references': len(references),
            'root_file_hash_records_not_read': len(inputs['root_files_sha256']),
            'scope': 'Only stage tools, reviewed input, manifest and copied saved references are read. No original root data/metadata/checkpoint path is opened or statted.'}

    def provenance(self, value, method, tag, seed):
        if (value.get('version') != VERSION or value.get('method') != method or value.get('tag') != tag or integer(value.get('seed')) != seed
                or value.get('input_sha256') != INPUT_SHA256 or value.get('manifest_sha256') != self.manifest_sha
                or value.get('tools_sha256') != self.manifest['tools_sha256']):
            self.integrity_failures.append('Saved provenance/hash mismatch: %s_%s_seed%d' % (method, tag, seed))
            raise ValueError('Saved provenance/hash identity differs')

    def reproduction(self, value, run, panels, item_map, method):
        if (value.get('status') != 'PASS' or value.get('checkpoint_before_sha256') != value.get('checkpoint_after_sha256')
                or not isinstance(value.get('checkpoint_before_sha256'), str) or re.fullmatch('[0-9a-f]{64}', value['checkpoint_before_sha256']) is None):
            raise ValueError('Reproduction/checkpoint-before-after record invalid')
        refs = [('canonical', 'original', run['reference_predictions'])]
        if method == 'ensemble':
            if set(value['fixed_five_views']) != {str(p) for p in PERMS}:raise ValueError('Five old Control view reproduction proofs required')
            refs += [(str(p), 'view_seed%d' % p, run['reference_permutations'][str(p)]) for p in PERMS]
        elif value.get('fixed_five_views') != {}:raise ValueError('Shuffle must not claim Control view reproduction')
        for key, panel, relative in refs:
            saved = reference_predictions(self.copies[relative], item_map);reproduction_compare(panels[panel], saved)
            proof = value['canonical'] if key == 'canonical' else value['fixed_five_views'][key]
            if (proof.get('status') != 'PASS' or proof.get('reference_path') != str(self.stage / relative)
                    or proof.get('reference_sha256') != self.manifest['references_sha256'][relative]
                    or integer(proof.get('samples')) != len(saved) or any(integer(proof.get(k)) != 0 for k in ('rank_mismatches', 'ordered_top10_mismatches', 'raw_top10_mismatches'))):
                raise ValueError('Reproduction proof does not match independent exact reference comparison')
        return {'status': 'PASS', 'canonical_samples': len(panels['original']), 'old_control_views': 5 if method == 'ensemble' else 0,
            'scope': 'Saved ranks/ordered Top10 independently compared to sealed prior model predictions; no checkpoint read or model inference.'}

    def run(self, method, tag, seed):
        name = '%s_%s_seed%d' % (method, tag, seed);output = 'outputs/' + name
        result = {'status': 'INCOMPLETE', 'method': method, 'tag': tag, 'seed': seed, 'errors': []};self.runs[name] = result
        try:
            relative = 'verification/completed_' + name + '.json';blob = self.copy(relative)
            if blob is None:
                for n in sorted(output_names(method)):self.copy(output + '/' + n)
                result['errors'].append('Completion missing; whitelisted partial metadata retained without statistics');return
            complete = json_object(blob);self.provenance(complete, method, tag, seed)
            if (complete.get('status') != 'COMPLETED' or complete.get('canonical_reproduction') is not True
                    or complete.get('old_control_five_views_reproduced') is not (method == 'ensemble')
                    or complete.get('originals_unchanged') is not True or complete.get('training_started') is not False
                    or not isinstance(complete.get('checkpoint_sha256'), str) or re.fullmatch('[0-9a-f]{64}', complete['checkpoint_sha256']) is None):
                raise ValueError('Inference-only completion/reproduction/checkpoint record invalid')
            plan = self.inputs['datasets'][tag];run = plan['runs'][method][str(seed)];n = integer(plan['expected_samples'])
            multiplier = 6 if method == 'shuffle' else 26
            expected_counts = {'samples': n, 'affected': integer(plan['expected_affected']), 'items': integer(plan['expected_items']),
                'logical_view_rows': n * multiplier, 'actual_forward_rows': n * multiplier,
                'forward_batch_calls': math.ceil(n / 256) * multiplier, 'cache_hits': 0}
            if complete.get('counts') != expected_counts:raise ValueError('Inference view multiplicity/actual rows/batch calls/cache accounting differs')
            if set(complete['metadata_sha256']) != output_names(method):
                self.integrity_failures.append('Unexpected sealed output inventory: ' + name);raise ValueError('Whitelisted inference metadata inventory differs')
            blobs = {fn: self.copy(output + '/' + fn, complete['metadata_sha256'][fn]) for fn in sorted(output_names(method))}
            if any(b is None for b in blobs.values()):result['errors'].append('Required sealed metadata missing/rejected; no statistics');return
            config = json_object(blobs['config.json']);self.provenance(config, method, tag, seed)
            prior_config = json_object(self.copies[run['reference_config']])
            if (prior_config.get('dataset') != plan['dataset'] or integer(prior_config.get('seed')) != seed
                    or any(prior_config.get(k) != v for k, v in {'max_len': 50, 'hidden_size': 64, 'num_heads': 2, 'num_blocks': 2, 'dropout': 0.2}.items())):
                raise ValueError('Sealed prior model config identity/architecture differs')
            expected_config = {'protocol': self.inputs['protocol'], 'model_dir': run['model_dir'], 'checkpoint': run['checkpoint'],
                'best_epoch': run['best_epoch'], 'batch_size': 256, 'groups': [list(GROUP_SEEDS[g]) for g in range(1, 6)],
                'no_training': True, 'score_cache_enabled': False}
            if any(config.get(k) != v for k, v in expected_config.items()):raise ValueError('Saved model identity/config/groups/no-training contract differs')
            item_map = mapping(blobs['item_id_map.csv']);reference_map = mapping(self.copies[run['reference_map']])
            if item_map != reference_map or len(item_map) != integer(plan['expected_items']):raise ValueError('Saved map differs from frozen prior model map')
            names = panel_names(method);panels = {p: predictions(blobs['predictions_' + p + '.csv'], item_map, n) for p in names}
            if any(identity(p) != identity(panels['original']) for p in panels.values()):raise ValueError('All saved panels must share sample/user/target/candidate counts')
            actual_metrics = {p: rank_metrics(rows) for p, rows in panels.items()}
            saved_metrics = json_object(blobs['metrics.json']);metric_panels = ('original',) + (tuple('perm_seed%d' % p for p in PERMS) if method == 'shuffle' else tuple('group%d' % g for g in range(1, 6)))
            if set(saved_metrics) != ({'panels'} if method == 'shuffle' else {'panels', 'ensemble_mean_accuracy'}) or set(saved_metrics['panels']) != set(metric_panels):raise ValueError('Saved accuracy panel inventory differs')
            for p in metric_panels:compare_metrics(actual_metrics[p], saved_metrics['panels'][p], p)
            primary = actual_metrics['original'] if method == 'shuffle' else {k: math.fsum(actual_metrics['group%d' % g][k] for g in range(1, 6)) / 5 for k in METRICS}
            if method == 'ensemble':compare_metrics(primary, saved_metrics['ensemble_mean_accuracy'], 'five group accuracy arithmetic mean')
            label_rows = rich_labels(blobs['panel_labels.csv'], panels['original'], plan['expected_affected'])
            views = input_views(blobs['input_view_hashes.csv'], label_rows, method)
            if len(views) != expected_counts['logical_view_rows']:raise ValueError('Input-view rows do not preserve all no-cache logical members')
            same_inputs = same_input_predictions(panels, views, method)
            reproduction = json_object(blobs['reproduction.json']);repro_check = self.reproduction(reproduction, run, panels, item_map, method)
            if reproduction['checkpoint_before_sha256'] != complete['checkpoint_sha256']:raise ValueError('Checkpoint record differs between reproduction and completion')
            if 'checkpoint_sha256' in run and run['checkpoint_sha256'] != complete['checkpoint_sha256']:raise ValueError('Checkpoint hash record differs from reviewed run seal')
            compared = [panels['perm_seed%d' % p] for p in PERMS] if method == 'shuffle' else [panels['group%d' % g] for g in range(1, 6)]
            affected = {sid: r['affected'] for sid, r in label_rows.items()};stability = population_stability(compared, affected)
            compare_stability(stability, json_object(blobs['stability.json']))
            probes = score_probes(json_object(blobs['score_mean_probes.json']), panels) if method == 'ensemble' else {'status': 'NOT_APPLICABLE'}
            diagnostics = {p: rank_top_diagnostics(rows) for p, rows in panels.items()}
            if any(v['rank_le10_vs_membership_mismatches'] or v['target_top10_position_vs_rank_mismatches'] for v in diagnostics.values()):self.warnings.append(name + ': rank/Top10 differences are diagnostics; original ranks preserved, no confirmed score-tie claim')
            if any(not isinstance(complete['environment'].get(k), str) or not complete['environment'][k] for k in ('python', 'torch', 'numpy', 'device', 'cuda')):raise ValueError('Runtime environment record missing')
            result.update(status='VALID_SAVED_INFERENCE', primary_accuracy=primary, canonical_reference_accuracy=actual_metrics['original'],
                primary_accuracy_scope='canonical input' if method == 'shuffle' else 'unweighted arithmetic mean of five final group metrics; not best group or pooled15 training replicates',
                all_panel_metrics=actual_metrics, stability=stability, counts=expected_counts, best_epoch=integer(run['best_epoch']),
                reproduction_check=repro_check, score_probe_check=probes, same_recorded_input_prediction_checks=same_inputs,
                prediction_diagnostics=diagnostics, checkpoint_sha256_record=complete['checkpoint_sha256'], checkpoint_statted_or_read=False)
            result.update(_map=item_map, _identity=identity(panels['original']), _labels=label_rows, _views=views, _panels=panels,
                _config_core={k: v for k, v in config.items() if k not in ('seed', 'model_dir', 'best_epoch')})
        except (OSError, ValueError, KeyError, TypeError, csv.Error, OverflowError, struct.error) as exc:
            result['status'] = 'INVALID_NO_STATISTICS';result['errors'].append(type(exc).__name__ + ': ' + str(exc)[:440])

    def summaries(self):
        result = {}
        for method in METHODS:
            result[method] = {}
            for tag in TAGS:
                rr = [self.runs['%s_%s_seed%d' % (method, tag, s)] for s in SEEDS]
                summary = {'status': 'INCOMPLETE_NO_THREE_SEED_SUMMARY'};result[method][tag] = summary
                if any(r['status'] != 'VALID_SAVED_INFERENCE' for r in rr):continue
                aligned = {'same_map': all(r['_map'] == rr[0]['_map'] for r in rr), 'same_sample_target_candidate_population': all(r['_identity'] == rr[0]['_identity'] for r in rr),
                    'same_frozen_raw_affected_labels': all(r['_labels'] == rr[0]['_labels'] for r in rr), 'same_input_view_hash_ledger': all(r['_views'] == rr[0]['_views'] for r in rr),
                    'same_inference_config_except_seed_modeldir_bestepoch': all(r['_config_core'] == rr[0]['_config_core'] for r in rr)}
                summary['cross_seed_alignment'] = aligned
                if not all(aligned.values()):summary['status'] = 'CROSS_SEED_MISMATCH_NO_SUMMARY';continue
                def stats(v):return {'mean': statistics.mean(v), 'sample_SD_ddof1': statistics.stdev(v)}
                summary.update(status='THREE_SEED_INFERENCE_SUMMARY', seeds=list(SEEDS), samples=len(rr[0]['_identity']), affected=sum(r['affected'] for r in rr[0]['_labels'].values()),
                    primary_accuracy={k: stats([r['primary_accuracy'][k] for r in rr]) for k in METRICS},
                    canonical_reference_accuracy={k: stats([r['canonical_reference_accuracy'][k] for r in rr]) for k in METRICS})
                summary['five_output_stability'] = {}
                for population in ('full', 'affected'):
                    n = rr[0]['stability'][population]['n'];summary['five_output_stability'][population] = {'n': n,
                        **{k: stats([r['stability'][population][k] for r in rr]) if n else {'mean': None, 'sample_SD_ddof1': None} for k in ('overlap10', 'hitflip')}}
                affected = {sid: v['affected'] for sid, v in rr[0]['_labels'].items()}
                reference_panel = 'original' if method == 'shuffle' else 'group1'
                summary['training_seed_reference_stability'] = population_stability([r['_panels'][reference_panel] for r in rr], affected)
                summary['training_seed_reference_panel'] = reference_panel
                summary['canonical_baseline_training_seed_reference'] = population_stability([r['_panels']['original'] for r in rr], affected)
        cross_method = {}
        for tag in TAGS:
            left = self.runs['shuffle_%s_seed42' % tag];right = self.runs['ensemble_%s_seed42' % tag]
            valid = all(result[m][tag]['status'] == 'THREE_SEED_INFERENCE_SUMMARY' for m in METHODS)
            checks = {}
            if valid:
                checks = {'same_map': left['_map'] == right['_map'], 'same_original_identity_candidates': left['_identity'] == right['_identity'],
                    'same_g1_affected_labels': left['_labels'] == right['_labels'],
                    'same_original_g1_input_hashes': all(right['_views'].get(k) == v for k, v in left['_views'].items())}
            cross_method[tag] = {'status': 'ALIGNED_DESCRIPTIVE_COMPARISON' if valid and all(checks.values()) else 'INCOMPLETE_OR_MISALIGNED_NO_COMPARISON', 'checks': checks}
        self.cross_method = cross_method
        return result

    def driver(self):
        self.driver_check = {'status': 'MISSING', 'errors': []};blob = self.copies.get('verification/driver.json')
        if blob is None:self.driver_check['errors'].append('Driver JSON absent');return
        try:
            value = json_object(blob);expected = [(m, t, s) for m in METHODS for t in TAGS for s in SEEDS];jobs = value['jobs']
            checks = {'stage_version': value.get('version') == VERSION and value.get('stage') == str(self.stage),
                'status': value.get('status') == 'ALL_TWELVE_SHUFFLE_AND_TWELVE_ENSEMBLE_EVALUATIONS_COMPLETED', 'twenty_four_jobs': len(jobs) == 24}
            previous, python, logs = None, None, []
            for i, job in enumerate(jobs):
                if i >= 24:raise ValueError('Unexpected extra driver jobs')
                method, tag, seed = expected[i];name = '%s_%s_seed%d' % (method, tag, seed)
                command = job['command'];python = python or command[0]
                checks[name + '_identity'] = (job.get('method'), job.get('tag'), job.get('seed')) == (method, tag, seed)
                checks[name + '_command'] = command == [python, '-u', str(self.stage / 'scripts/run_x3_inference.py'), '--root', str(self.root), '--run', '--stage', str(self.stage), '--method', method, '--dataset', tag, '--seed', str(seed)]
                checks[name + '_exit'] = integer(job.get('exit_code')) == 0
                proof_rel = 'verification/completed_' + name + '.json';log_rel = 'verification/' + name + '.log'
                for key, relative in (('proof', proof_rel), ('log', log_rel)):
                    data = self.copies.get(relative)
                    checks[name + '_' + key] = data is not None and job.get(key) == str(self.stage / relative) and job.get(key + '_sha256') == sha(data)
                    if data is not None and job.get(key + '_sha256') != sha(data):self.integrity_failures.append('Driver ' + key + ' SHA mismatch: ' + name)
                checks[name + '_valid_saved'] = self.runs[name]['status'] == 'VALID_SAVED_INFERENCE';logs.append(log_rel)
                if not isinstance(job.get('started_utc'), str) or not isinstance(job.get('finished_utc'), str):checks[name + '_UTC'] = False;continue
                begin = datetime.datetime.fromisoformat(job['started_utc'].replace('Z', '+00:00'));end = datetime.datetime.fromisoformat(job['finished_utc'].replace('Z', '+00:00'))
                checks[name + '_UTC'] = begin.tzinfo is not None and end.tzinfo is not None and begin <= end and (previous is None or previous <= begin);previous = end
            checks['distinct_twenty_four_logs'] = len(set(logs)) == 24
            self.driver_check.update(status='PASS' if all(checks.values()) else 'FAIL', checks=checks)
            if not all(checks.values()):self.driver_check['errors'].append('Driver does not confirm all twenty-four valid sequential inference-only jobs')
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            self.driver_check.update(status='FAIL');self.driver_check['errors'].append(type(exc).__name__ + ': ' + str(exc)[:440])

    def finish(self, summaries):
        public = {name: {k: v for k, v in run.items() if not k.startswith('_')} for name, run in self.runs.items()}
        complete = (len(public) == 24 and all(r['status'] == 'VALID_SAVED_INFERENCE' for r in public.values())
            and len(summaries) == 2 and all(len(v) == 4 and all(s['status'] == 'THREE_SEED_INFERENCE_SUMMARY' for s in v.values()) for v in summaries.values())
            and all(v['status'] == 'ALIGNED_DESCRIPTIVE_COMPARISON' for v in getattr(self, 'cross_method', {}).values())
            and self.driver_check['status'] == 'PASS')
        status = 'INTEGRITY_FAILED_NO_BUNDLE' if self.integrity_failures else 'ALL_TWENTY_FOUR_INFERENCE_RUNS_CHECKED' if complete else 'INCOMPLETE'
        report = {'status': status, 'stage': str(self.stage), 'input_sha256': INPUT_SHA256, 'source_check': getattr(self, 'source_check', {'status': 'NOT_COMPLETED'}),
            'run_checks': public, 'dataset_summaries': summaries, 'cross_method_population_alignment': getattr(self, 'cross_method', {}),
            'driver_check': self.driver_check, 'files': self.files, 'warnings': self.warnings, 'integrity_failures': self.integrity_failures,
            'scope': 'PRIVATE saved inference records only. No runner/model/protocol executed and no original root input/reference/checkpoint stat/content read. Canonical ranks/ordered Top10 and five g1 Control views are compared to sealed staged prior predictions. Seven metrics, all ten pairs among five Shuffle outputs or five final ensemble groups, user-any hitflip and three-training-seed sample SD are recomputed. Ensemble primary accuracy is the arithmetic mean of five group metrics per model. Canonical Control accuracy is only a reproduction reference. Score probes check the first-at-most16 samples/group saved float32-member/float64-mean arithmetic, not all candidate scores or independent model logits. View ledger/counts retain 6N or26N including canonical, no cache. Outer25-view stability is not one five-view recommendation cost. Affected raw-hash labels/group2-5 input generation/checkpoint content are verified saved records, not locally reread histories/weights. Descriptive statistics do not establish significance/equivalence or confirm score ties.'}
        (self.out / 'x3_inference_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
        with (self.out / 'REPORT_SHA256.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['path', 'status', 'bytes', 'sha256', 'error']);w.writeheader();w.writerows(self.files)
        with (self.out / 'three_seed_accuracy.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['method', 'dataset', 'samples', 'metric', 'mean', 'sample_SD_ddof1']);w.writeheader()
            for method, values in summaries.items():
                for tag, row in values.items():
                    if row['status'] == 'THREE_SEED_INFERENCE_SUMMARY':
                        for metric, v in row['primary_accuracy'].items():w.writerow(dict(method=method, dataset=tag, samples=row['samples'], metric=metric, **v))
        with (self.out / 'three_seed_stability.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['method', 'dataset', 'population', 'samples', 'metric', 'mean', 'sample_SD_ddof1']);w.writeheader()
            for method, values in summaries.items():
                for tag, row in values.items():
                    if row['status'] == 'THREE_SEED_INFERENCE_SUMMARY':
                        for population, r in row['five_output_stability'].items():
                            for metric in ('overlap10', 'hitflip'):w.writerow(dict(method=method, dataset=tag, population=population, samples=r['n'], metric=metric, **r[metric]))
        lines = ['# PRIVATE X3 inference/ensemble results', '', 'Status: **' + status + '**', '', report['scope'], '', '| Run | Status | Primary NDCG@10 |', '|---|---|---:|']
        for name, run in public.items():lines.append('| %s | %s | %s |' % (name, run['status'], run.get('primary_accuracy', {}).get('ndcg10', '')))
        messages = self.integrity_failures + self.warnings + self.driver_check.get('errors', []) + [name + ': ' + e for name, r in public.items() for e in r['errors']]
        lines += ['', '## Diagnostics', ''] + (['- ' + v for v in messages] or ['- No discrepancy found.']) + ['', 'Private per-user metadata. Do not publish this ZIP directly.', '']
        (self.out / 'SUMMARY.md').write_text('\n'.join(lines), 'utf-8')
        bundle = None
        if not self.integrity_failures:
            bundle = self.out / 'x3_inference_results.zip';paths = sorted(p for p in self.out.rglob('*') if p.is_file())
            with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
                for p in paths:
                    if p.is_symlink():raise ValueError('Output symlink rejected')
                    z.write(p, p.relative_to(self.out).as_posix())
            h = hashlib.sha256()
            with bundle.open('rb') as f:
                for b in iter(lambda: f.read(1024 * 1024), b''):h.update(b)
            bundle.with_name(bundle.name + '.sha256.txt').write_text(h.hexdigest() + '  ' + bundle.name + '\n', 'ascii')
        return {'status': status, 'report_dir': str(self.out), 'zip_path': str(bundle) if bundle else None, 'exit_code': 2 if self.integrity_failures else 0 if complete else 1}


def collect(stage):
    c = Collector(stage);summaries = {}
    try:c.sources()
    except (OSError, ValueError, KeyError, TypeError) as exc:c.integrity_failures.append(type(exc).__name__ + ': ' + str(exc)[:440])
    if not c.integrity_failures:
        c.copy('verification/driver.json')
        for method in METHODS:
            for tag in TAGS:
                for seed in SEEDS:
                    c.copy('verification/%s_%s_seed%d.log' % (method, tag, seed));c.run(method, tag, seed)
        summaries = c.summaries();c.driver()
    return c.finish(summaries)


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--stage', required=True, type=Path);args = p.parse_args()
    result = collect(args.stage.expanduser());path = Path(result['report_dir']) / 'x3_inference_checks.json'
    print('CHECKS=' + str(path));print('SUMMARY=' + str(path.with_name('SUMMARY.md')))
    print('X3_INFERENCE_REPORT_ZIP=' + (result['zip_path'] or 'NOT_CREATED'));print('STATUS=' + result['status'])
    report = json_object(path.read_bytes())
    for e in report['integrity_failures']:print('INTEGRITY_FAILURE=' + e)
    for n, r in report['run_checks'].items():
        for e in r['errors']:print('RUN_FAILURE=' + n + ': ' + e)
    for e in report['driver_check'].get('errors', []):print('DRIVER_FAILURE=' + e)
    for k, v in report['driver_check'].get('checks', {}).items():
        if not v:print('DRIVER_FAILED_CHECK=' + k)
    return result['exit_code']


if __name__ == '__main__':
    try:sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print('COLLECTION_FAILED=' + type(exc).__name__ + ': ' + str(exc)[:440], file=sys.stderr);sys.exit(2)
