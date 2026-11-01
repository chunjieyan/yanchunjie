#!/usr/bin/env python3
"""Collect X4 TiSASRec saved metadata with no model, input-history or weight access."""
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
import sys
import uuid
import zipfile

VERSION = 'x4_tisasrec_v1'
INPUT_NAME = 'x4_tisasrec_inputs_20261026.json'
INPUT_SHA256 = '4dfe330abba456fbc471d2b09e85a204ce80727e2a81f28348ebea84ca3392f4'
TAGS = ('ml1m', 'beauty')
SEEDS = (42, 2026, 3407)
PERMS = (101, 202, 303, 404, 505)
PANELS = ('original',) + tuple('perm_seed' + str(s) for s in PERMS)
METRICS = ('ndcg10', 'hr10', 'hr20', 'recall50', 'mrr', 'mrr10', 'mean_rank')
TOOLS = ('run_x4_tisasrec.py', 'collect_x4_tisasrec.py', 'x4_tisasrec_model.py', 'x4_tisasrec_protocol.py', 'X4_UPSTREAM_MIT_LICENSE.txt', 'X4_MODEL_PROVENANCE.json')
OUTPUT_FILES = ('config.json', 'item_id_map.csv', 'training_log.csv', 'validation_best_predictions.csv', 'panel_labels.csv', 'metrics.json', 'stability.json') + tuple('predictions_' + p + '.csv' for p in PANELS)
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


def labels(blob, expected_ids, expected_affected):
    result = {}
    for row in csv_rows(blob):
        if set(row) != {'sample_id', 'affected'} or row['sample_id'] in result:
            raise ValueError('Affected-label shape/identity invalid')
        value = integer(row['affected'])
        if value not in (0, 1):
            raise ValueError('Affected flag must be binary')
        result[row['sample_id']] = bool(value)
    if set(result) != set(expected_ids) or sum(result.values()) != integer(expected_affected):
        raise ValueError('Affected population differs from frozen expected identity/count')
    return result


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


def compare_stability(actual, saved):
    if set(saved) != {'full', 'affected'}:
        raise ValueError('Stability population inventory differs')
    for population, value in actual.items():
        supplied = saved[population]
        if (integer(supplied['n']) != value['n'] or supplied.get('ten_unordered_pairs_per_user') is not True
                or set(supplied) != {'n', 'overlap10', 'hitflip', 'ten_unordered_pairs_per_user', 'users'}):
            raise ValueError('Saved stability population/count/pair convention differs')
        for field in ('overlap10', 'hitflip'):
            if value[field] is None:
                if supplied[field] is not None:
                    raise ValueError('Empty stability population must have null measurements')
            elif not close(value[field], supplied[field]):
                raise ValueError('Saved stability scalar differs: ' + population + '/' + field)
        rows = supplied['users']
        by_id = {r['sample_id']: r for r in rows}
        if len(by_id) != len(rows) or set(by_id) != {r['sample_id'] for r in value['users']}:
            raise ValueError('Saved stability per-user identities differ')
        for row in value['users']:
            r = by_id[row['sample_id']]
            if (set(r) != {'sample_id', 'overlap10', 'any_hitflip', 'top10_all_identical'}
                    or not close(row['overlap10'], r['overlap10']) or integer(r['any_hitflip']) != row['any_hitflip']
                    or integer(r['top10_all_identical']) != row['top10_all_identical']):
                raise ValueError('Saved per-user stability differs')


def training_trace(blob, validation, expected_best, hyperparameters):
    rows = csv_rows(blob)
    maximum, patience = integer(hyperparameters['max_epochs']), integer(hyperparameters['patience'])
    if not rows or len(rows) > maximum:
        raise ValueError('Empty or oversized epoch trace')
    best, best_epoch, bad, selected = -math.inf, 0, 0, None
    for epoch, row in enumerate(rows, 1):
        if integer(row['epoch']) != epoch:
            raise ValueError('Epoch trace is not contiguous from one')
        finite(row['train_loss'], nonnegative=True); finite(row['seconds'], nonnegative=True)
        values = {k: finite(row['val_' + k], nonnegative=True) for k in METRICS}
        if not 0 <= values['ndcg10'] <= values['hr10'] <= values['hr20'] <= values['recall50'] <= 1 or not 0 <= values['mrr10'] <= values['mrr'] <= 1 or values['mean_rank'] < 1:
            raise ValueError('Epoch validation metric range invalid')
        if values['ndcg10'] > best:
            best, best_epoch, bad, selected = values['ndcg10'], epoch, 0, values
        else:
            bad += 1
        if integer(row['best_epoch']) != best_epoch or integer(row['bad_epochs']) != bad:
            raise ValueError('Best/bad trace differs from strict-improvement rule')
        if bad >= patience and epoch != len(rows):
            raise ValueError('Training continued after first patience trigger')
    if len(rows) < maximum and bad != patience:
        raise ValueError('Training stopped before max epochs without patience')
    if integer(expected_best) != best_epoch or any(not close(validation[k], selected[k]) for k in METRICS):
        raise ValueError('Best validation saved ranks/metrics do not match selected epoch')
    return {'completed_epochs': len(rows), 'best_epoch': best_epoch, 'selection_threshold': 0,
            'final_bad_epochs': bad, 'stop_reason': 'patience' if bad >= patience else 'max_epochs',
            'epoch_seconds_total': math.fsum(float(r['seconds']) for r in rows)}


class Collector:
    def __init__(self, stage):
        self.stage = absolute_plain(stage)
        if not self.stage.name.startswith('kais_x4_tisasrec_v1_'):
            raise ValueError('Unexpected X4 stage prefix')
        verification = safe(self.stage, 'verification')
        if not verification.is_dir():
            raise ValueError('Verification directory missing')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
        self.out = verification / ('x4_review_' + stamp); self.out.mkdir(exist_ok=False)
        self.files, self.copies, self.integrity_failures, self.warnings = [], {}, [], []
        self.total, self.runs, self.smokes = 0, {}, {}
        self.driver_check = {'status': 'NOT_CHECKED', 'errors': []}

    def copy(self, relative, expected=None):
        record = {'path': relative, 'status': 'unknown'}
        try:
            blob, digest = read_stable(self.stage, relative)
            if expected is not None and digest != expected:
                raise ValueError('SHA256 mismatch')
            if self.total + len(blob) > TOTAL_LIMIT:
                raise ValueError('Total collection size exceeds limit')
            path = safe(self.out, 'snapshot/' + relative);path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f:f.write(blob)
            self.copies[relative] = blob;self.total += len(blob)
            record.update(status='copied', bytes=len(blob), sha256=digest)
            return blob
        except (OSError, ValueError) as exc:
            record.update(status='missing' if isinstance(exc, FileNotFoundError) else 'rejected_or_changed', error=str(exc)[:320])
            if expected is not None and isinstance(exc, ValueError):
                self.integrity_failures.append('Pinned file rejected/changed: ' + relative)
            self.warnings.append(record['status'] + ': ' + relative)
            return None
        finally:self.files.append(record)

    def sources(self):
        manifest_blob, self.manifest_sha = read_stable(self.stage, 'manifest.json')
        self.manifest = m = json_object(manifest_blob)
        input_blob, digest = read_stable(self.stage, INPUT_NAME)
        if digest != INPUT_SHA256:
            raise ValueError('Reviewed input seal SHA256 mismatch')
        self.inputs = inputs = json_object(input_blob)
        self.root = absolute_plain(m['root'])
        if (self.stage.parent != self.root or m.get('version') != VERSION or m.get('stage') != str(self.stage)
                or m.get('input_sha256') != digest or m.get('data_sha256') != inputs['data_sha256']
                or m.get('protocol') != inputs['protocol'] or inputs.get('version') != VERSION
                or inputs.get('training_seeds') != list(SEEDS) or set(inputs['datasets']) != set(TAGS)):
            raise ValueError('Prepared identity/input/data/protocol binding differs')
        inventory = {'scripts/' + n for n in TOOLS} | {INPUT_NAME}
        if set(m['tool_sha256']) != inventory or m['tool_sha256'][INPUT_NAME] != digest:
            raise ValueError('Expected exactly seven tool/input inventory entries')
        required = {p for plan in inputs['datasets'].values() for p in plan['required_files']}
        if set(inputs['data_sha256']) != required or len(required) != 22 or inputs.get('permutation_seeds') != list(PERMS):
            raise ValueError('Expected twenty-two frozen data/map/config paths and five permutation seeds; no input contents will be opened')
        if m.get('tasks') != [{'dataset': tag, 'seed': seed} for tag in TAGS for seed in SEEDS]:
            raise ValueError('Expected six dataset/seed tasks')
        for p in required:relative_path(p)
        for path, value in inputs['data_sha256'].items():
            if re.fullmatch('[0-9a-f]{64}', value) is None:
                raise ValueError('Invalid frozen data hash syntax')
        helper_names = {'x4_tisasrec_model.py', 'x4_tisasrec_protocol.py', 'X4_UPSTREAM_MIT_LICENSE.txt', 'X4_MODEL_PROVENANCE.json'}
        if set(inputs['helper_sha256']) != helper_names:
            raise ValueError('Unexpected non-circular helper pin inventory')
        for name, expected in inputs['helper_sha256'].items():
            if m['tool_sha256'].get('scripts/' + name) != expected:
                raise ValueError('Pinned helper differs from prepared tool hash')
        for relative, expected in m['tool_sha256'].items():
            if self.copy(relative, expected) is None:
                raise ValueError('Required frozen tool missing/rejected: ' + relative)
        executing = Path(__file__).absolute()
        blob, digest = read_stable(absolute_plain(executing.parent), executing.name)
        if digest != m['tool_sha256']['scripts/collect_x4_tisasrec.py']:
            raise ValueError('Executing collector differs from stage collector')
        if self.copy('manifest.json', self.manifest_sha) is None:
            raise ValueError('Manifest changed during collection')
        self.source_check = {'status': 'PASS', 'tool_files': len(inventory), 'frozen_data_paths_not_read': len(required),
            'scope': 'Stage tools and input seal verified by bytes; four model/protocol/license/provenance helpers bound to the reviewed input. Data hashes are server-provided seals, not locally rehashed raw histories.'}

    def base_provenance(self, tag, seed, value):
        if (value.get('version') != VERSION or value.get('tag') != tag or integer(value.get('seed')) != seed
                or value.get('input_sha256') != INPUT_SHA256 or value.get('manifest_sha256') != self.manifest_sha
                or value.get('tool_sha256') != self.manifest['tool_sha256'] or value.get('protocol') != self.inputs['protocol']):
            self.integrity_failures.append('Provenance/hash/protocol mismatch: %s_seed%d' % (tag, seed))
            raise ValueError('Saved provenance/hash/protocol differs')
        plan, contract = self.inputs['datasets'][tag], value['contract']
        expected = {'training_rows': plan['expected_training_rows'], 'training_positions': plan['expected_training_positions'],
            'num_items': plan['expected_items'], 'validation_samples': plan['expected_validation'],
            'test_samples': plan['expected_test'], 'affected_samples': plan['expected_affected'],
            'trainable_users': plan['expected_trainable_users']}
        if (contract.get('status') != 'PASS' or any(integer(contract[k]) != integer(v) for k, v in expected.items())
                or not 1 <= integer(contract['trainable_users']) <= integer(contract['training_rows']) - integer(contract['training_positions'])
                or any(contract.get(k) is not True for k in ('event_join_complete', 'strict_target_bucket_after_history', 'all_five_event_multisets_and_prefixes_preserved'))
                or contract.get('native_time_unit') != plan['native_time_unit'] or contract.get('target_time_used_as_feature') is not False):
            raise ValueError('Saved data/event/strict chronology contract invalid')
        diagnostic = contract['time_matrix_diagnostic']
        pairs = integer(diagnostic['valid_unordered_pairs'])
        zero, clipped = integer(diagnostic['zero_pairs']), integer(diagnostic['at_clip_256_pairs'])
        if (diagnostic.get('scope') != 'first at most128 validation and128 original test visible histories; no target time'
                or not 0 <= zero <= pairs <= 256 * 1225 or not 0 <= clipped <= pairs or zero + clipped > pairs):
            raise ValueError('Bounded time-matrix diagnostic record invalid')
        env = value['environment']
        if any(not isinstance(env.get(k), str) or not env[k] for k in ('python', 'torch', 'numpy', 'device', 'cuda')):
            raise ValueError('Missing runtime environment record')
        return contract

    def smoke(self, tag):
        result = {'status': 'MISSING', 'errors': []};self.smokes[tag] = result
        relative = 'verification/smoke_' + tag + '.json'
        blob = self.copies.get(relative)
        if blob is None:
            result['errors'].append('Required seed42 smoke summary missing');return
        try:
            value = json_object(blob);self.base_provenance(tag, 42, value)
            proof = value['proof']
            if (value.get('status') != 'SMOKE_PASS' or finite(proof['finite_loss'], positive=True) <= 0
                    or integer(proof['finite_gradient_tensors']) <= 0
                    or any(proof.get(k) is not True for k in ('matrix_exact_integer_reference', 'zero_gap_and_clip', 'causal_and_padding_mask', 'batch_single_allclose', 'save_restore_identical_call'))
                    or integer(proof['optimizer_steps']) != 0 or proof.get('test_inference') is not False or proof.get('formal_weights_saved') is not False):
                raise ValueError('Synthetic smoke model/matrix/mask/gradient/restore proof failed')
            result.update(status='PASS', contract=value['contract'], proof=proof)
        except (ValueError, KeyError, TypeError) as exc:
            result['status'] = 'FAIL';result['errors'].append(type(exc).__name__ + ': ' + str(exc)[:360])

    def run(self, tag, seed):
        name, output = '%s_seed%d' % (tag, seed), 'outputs/%s_seed%d' % (tag, seed)
        result = {'status': 'INCOMPLETE', 'tag': tag, 'seed': seed, 'errors': []};self.runs[name] = result
        try:
            completed_relative = 'verification/completed_' + name + '.json'
            blob = self.copy(completed_relative)
            if blob is None:
                for n in OUTPUT_FILES:self.copy(output + '/' + n)
                result['errors'].append('Completion absent; retained whitelisted partial metadata without statistics');return
            value = json_object(blob);self.base_provenance(tag, seed, value)
            if (value.get('status') != 'COMPLETED' or any(value.get(k) is not True for k in ('original_inputs_unchanged', 'validation_reproduced', 'canonical_and_five_permutations'))
                    or not isinstance(value.get('checkpoint_sha256'), str) or re.fullmatch('[0-9a-f]{64}', value['checkpoint_sha256']) is None):
                raise ValueError('Completion/validation/permutation flags or checkpoint-hash syntax invalid')
            hashes = value['metadata_sha256']
            if set(hashes) != set(OUTPUT_FILES):
                self.integrity_failures.append('Unexpected sealed output metadata inventory: ' + name)
                raise ValueError('Completion must seal exactly thirteen whitelisted metadata files')
            blobs = {n: self.copy(output + '/' + n, hashes[n]) for n in OUTPUT_FILES}
            if any(b is None for b in blobs.values()):
                result['errors'].append('Required saved metadata missing/rejected; no statistics');return
            config = json_object(blobs['config.json']);self.base_provenance(tag, seed, config)
            if (config.get('hyperparameters') != self.inputs['hyperparameters'] or config.get('method') != 'TiSASRec_strict_prefix_time_adaptation'
                    or config['contract'] != value['contract'] or config['environment'] != value['environment']):
                raise ValueError('Saved hyperparameters/method/runtime contract differ')
            item_map = mapping(blobs['item_id_map.csv']);plan = self.inputs['datasets'][tag]
            if len(item_map) != integer(plan['expected_items']):
                raise ValueError('Saved catalog count differs from frozen plan')
            val_rows = predictions(blobs['validation_best_predictions.csv'], item_map, plan['expected_validation'])
            val_metrics = rank_metrics(val_rows)
            panels = {p: predictions(blobs['predictions_' + p + '.csv'], item_map, plan['expected_test']) for p in PANELS}
            original_identity = identity(panels['original'])
            if any(identity(p) != original_identity for p in panels.values()):
                raise ValueError('Original/five-permutation identities or candidate counts differ')
            affected = labels(blobs['panel_labels.csv'], panels['original'], plan['expected_affected'])
            saved_metrics, saved_stability = json_object(blobs['metrics.json']), json_object(blobs['stability.json'])
            if set(saved_metrics) != {'best_epoch', 'best_validation', 'panels'} or set(saved_metrics['panels']) != set(PANELS):
                raise ValueError('Saved metric panel inventory differs')
            compare_metrics(val_metrics, saved_metrics['best_validation'], 'best validation')
            all_metrics = {p: rank_metrics(rows) for p, rows in panels.items()}
            for p in PANELS:compare_metrics(all_metrics[p], saved_metrics['panels'][p], p)
            if integer(value['best_epoch']) != integer(saved_metrics['best_epoch']):
                raise ValueError('Completion/metrics best epoch differ')
            trace = training_trace(blobs['training_log.csv'], val_metrics, saved_metrics['best_epoch'], self.inputs['hyperparameters'])
            actual_stability = population_stability([panels['perm_seed' + str(s)] for s in PERMS], affected)
            compare_stability(actual_stability, saved_stability)
            diagnostics = {p: rank_top_diagnostics(rows) for p, rows in dict(panels, validation=val_rows).items()}
            if any(v['rank_le10_vs_membership_mismatches'] or v['target_top10_position_vs_rank_mismatches'] for v in diagnostics.values()):
                self.warnings.append(name + ': original rank vs independent Top10 differences; ranks preserved, no confirmed tie claim')
            result.update(status='VALID_SAVED_RUN', best_epoch=trace['best_epoch'], training_trace=trace,
                validation=val_metrics, canonical_test=all_metrics['original'], all_panel_metrics=all_metrics,
                stability=actual_stability, prediction_diagnostics=diagnostics,
                server_contract=value['contract'], checkpoint_sha256_format='PASS; checkpoint not statted or read')
            result.update(_map=item_map, _validation_identity=identity(val_rows), _identity=original_identity,
                _labels=affected, _panels=panels, _config_core={k: v for k, v in config.items() if k not in ('seed', 'environment')})
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
            result['status'] = 'INVALID_NO_STATISTICS';result['errors'].append(type(exc).__name__ + ': ' + str(exc)[:420])

    def summaries(self):
        summaries = {}
        for tag in TAGS:
            rr = [self.runs['%s_seed%d' % (tag, s)] for s in SEEDS]
            summary = {'status': 'INCOMPLETE_NO_THREE_SEED_SUMMARY'};summaries[tag] = summary
            if any(r['status'] != 'VALID_SAVED_RUN' for r in rr):continue
            checks = {field: all(r[field] == rr[0][field] for r in rr) for field in ('_map', '_validation_identity', '_identity', '_labels', '_config_core')}
            summary['cross_seed_alignment'] = {k[1:]: v for k, v in checks.items()}
            if not all(checks.values()):
                summary['status'] = 'CROSS_SEED_MISMATCH_NO_SUMMARY';continue
            def stats(values):
                return {'mean': statistics.mean(values), 'sample_SD_ddof1': statistics.stdev(values)}
            summary.update(status='THREE_SEED_SAVED_RUN_SUMMARY', seeds=list(SEEDS),
                canonical_test_samples=len(rr[0]['_identity']), affected_samples=sum(rr[0]['_labels'].values()),
                best_epochs=[r['best_epoch'] for r in rr],
                canonical_accuracy={k: stats([r['canonical_test'][k] for r in rr]) for k in METRICS},
                validation_accuracy={k: stats([r['validation'][k] for r in rr]) for k in METRICS})
            summary['fixed_five_permutation_stability'] = {}
            for population in ('full', 'affected'):
                n = rr[0]['stability'][population]['n']
                summary['fixed_five_permutation_stability'][population] = {'n': n,
                    **{k: stats([r['stability'][population][k] for r in rr]) if n else {'mean': None, 'sample_SD_ddof1': None} for k in ('overlap10', 'hitflip')}}
            summary['training_seed_reference_stability'] = population_stability([r['_panels']['original'] for r in rr], rr[0]['_labels'])
            summary['training_seed_reference_scope'] = 'Canonical predictions from three separately trained models; three unordered pairs per user, with any-user hit flip across all three. Descriptive reference, not the fixed-model five-permutation measurement.'
        return summaries

    def driver(self):
        self.driver_check = {'status': 'MISSING', 'errors': []}
        blob = self.copies.get('verification/driver.json')
        if blob is None:
            self.driver_check['errors'].append('Driver JSON missing');return
        try:
            value = json_object(blob)
            expected = [('smoke', tag, 42) for tag in TAGS] + [('train', tag, s) for tag in TAGS for s in SEEDS]
            jobs = value['jobs']
            checks = {'version_stage': value.get('version') == VERSION and value.get('stage') == str(self.stage),
                'status': value.get('status') == 'ALL_TWO_SMOKES_AND_SIX_TRAININGS_COMPLETED',
                'eight_jobs': len(jobs) == 8}
            last_end, python, logs = None, None, []
            for i, job in enumerate(jobs):
                if i >= len(expected):raise ValueError('Unexpected extra driver jobs')
                mode, tag, seed = expected[i];prefix = '%d_%s_%s_%d' % (i, mode, tag, seed)
                checks[prefix + '_identity'] = (job.get('mode'), job.get('dataset'), job.get('seed')) == (mode, tag, seed)
                command = job['command']
                if python is None:python = command[0]
                checks[prefix + '_command'] = command == [python, '-u', str(self.stage / 'scripts/run_x4_tisasrec.py'), '--root', str(self.root), '--run', '--stage', str(self.stage), '--mode', mode, '--dataset', tag, '--seed', str(seed)]
                checks[prefix + '_exit'] = integer(job.get('exit_code')) == 0
                relative = 'verification/' + ('smoke_' + tag + '.json' if mode == 'smoke' else 'completed_%s_seed%d.json' % (tag, seed))
                proof = self.copies.get(relative)
                checks[prefix + '_proof'] = job.get('proof') == str(self.stage / relative) and proof is not None and job.get('proof_sha256') == sha(proof)
                if proof is not None and job.get('proof_sha256') != sha(proof):
                    self.integrity_failures.append('Driver proof SHA mismatch: ' + prefix)
                checks[prefix + '_saved_valid'] = self.smokes[tag]['status'] == 'PASS' if mode == 'smoke' else self.runs['%s_seed%d' % (tag, seed)]['status'] == 'VALID_SAVED_RUN'
                log_relative = 'verification/%s_%s_seed%d.log' % (mode, tag, seed)
                log = self.copies.get(log_relative);logs.append(log_relative)
                checks[prefix + '_log'] = job.get('log') == str(self.stage / log_relative) and log is not None and job.get('log_sha256') == sha(log)
                if log is not None and job.get('log_sha256') != sha(log):
                    self.integrity_failures.append('Driver log SHA mismatch: ' + prefix)
                if not isinstance(job.get('started_utc'), str) or not isinstance(job.get('finished_utc'), str):
                    checks[prefix + '_time'] = False;continue
                started = datetime.datetime.fromisoformat(job['started_utc'].replace('Z', '+00:00'))
                finished = datetime.datetime.fromisoformat(job['finished_utc'].replace('Z', '+00:00'))
                checks[prefix + '_time'] = started.tzinfo is not None and finished.tzinfo is not None and started <= finished and (last_end is None or last_end <= started)
                last_end = finished
            checks['distinct_eight_logs'] = len(set(logs)) == 8
            self.driver_check.update(status='PASS' if all(checks.values()) else 'FAIL', checks=checks)
            if not all(checks.values()):self.driver_check['errors'].append('Driver records do not confirm two smokes followed by all six completed saved runs')
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            self.driver_check.update(status='FAIL');self.driver_check['errors'].append(type(exc).__name__ + ': ' + str(exc)[:420])

    def finish(self, summaries):
        public = {name: {k: v for k, v in run.items() if not k.startswith('_')} for name, run in self.runs.items()}
        complete = (len(public) == 6 and all(r['status'] == 'VALID_SAVED_RUN' for r in public.values())
            and len(self.smokes) == 2 and all(r['status'] == 'PASS' for r in self.smokes.values())
            and len(summaries) == 2 and all(r['status'] == 'THREE_SEED_SAVED_RUN_SUMMARY' for r in summaries.values())
            and self.driver_check['status'] == 'PASS')
        status = 'INTEGRITY_FAILED_NO_BUNDLE' if self.integrity_failures else 'ALL_SIX_TISASREC_RUNS_CHECKED' if complete else 'INCOMPLETE'
        report = {'status': status, 'stage': str(self.stage), 'input_sha256': INPUT_SHA256,
            'source_check': getattr(self, 'source_check', {'status': 'NOT_COMPLETED'}), 'smoke_checks': self.smokes,
            'run_checks': public, 'dataset_summaries': summaries, 'driver_check': self.driver_check,
            'files': self.files, 'warnings': self.warnings, 'integrity_failures': self.integrity_failures,
            'scope': 'PRIVATE X4 TiSASRec saved-record audit. Only whitelisted stage tools/input/manifest/smoke/completion/logs and saved prediction metadata are read. No model, runner or protocol module executed, no original data/map input opened, no checkpoint stat/content accessed. Seven metrics are rank-derived; validation ranks, strict-improvement selection, stopping, all ten pairs among five permutation Top10 lists, and any-user hit flips are independently recomputed. Affected labels/candidate counts align across saved panels and the frozen expected counts; their raw-history derivation and runtime event/time/masking contracts are server records, not locally reread data. Cross-seed comparisons are descriptive. No score-tie, significance or equivalence claim.'}
        (self.out / 'x4_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
        with (self.out / 'REPORT_SHA256.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['path', 'status', 'bytes', 'sha256', 'error']);w.writeheader();w.writerows(self.files)
        with (self.out / 'three_seed_accuracy.csv').open('w', encoding='utf-8', newline='') as f:
            fields = ['dataset', 'method', 'samples', 'metric', 'mean', 'sample_SD_ddof1'];w = csv.DictWriter(f, fieldnames=fields);w.writeheader()
            for tag, value in summaries.items():
                if value['status'] == 'THREE_SEED_SAVED_RUN_SUMMARY':
                    for metric, record in value['canonical_accuracy'].items():w.writerow(dict(dataset=tag, method='TiSASRec_strict_prefix_time_adaptation', samples=value['canonical_test_samples'], metric=metric, **record))
        with (self.out / 'three_seed_stability.csv').open('w', encoding='utf-8', newline='') as f:
            fields = ['dataset', 'population', 'samples', 'metric', 'mean', 'sample_SD_ddof1'];w = csv.DictWriter(f, fieldnames=fields);w.writeheader()
            for tag, value in summaries.items():
                if value['status'] == 'THREE_SEED_SAVED_RUN_SUMMARY':
                    for population, row in value['fixed_five_permutation_stability'].items():
                        for metric in ('overlap10', 'hitflip'):w.writerow(dict(dataset=tag, population=population, samples=row['n'], metric=metric, **row[metric]))
        lines = ['# PRIVATE X4 TiSASRec saved-record report', '', 'Status: **' + status + '**', '', report['scope'], '', '| Run | Status | Best epoch | Canonical NDCG@10 |', '|---|---|---:|---:|']
        for name, run in public.items():lines.append('| %s | %s | %s | %s |' % (name, run['status'], run.get('best_epoch', ''), run.get('canonical_test', {}).get('ndcg10', '')))
        messages = self.integrity_failures + self.warnings + self.driver_check.get('errors', [])
        messages += [name + ': ' + error for name, run in public.items() for error in run['errors']]
        messages += ['smoke_' + name + ': ' + error for name, run in self.smokes.items() for error in run['errors']]
        lines += ['', '## Diagnostics', ''] + (['- ' + message for message in messages] or ['- No discrepancy found.'])
        lines += ['', 'Private per-user predictions are included. Do not publish this bundle directly.', '']
        (self.out / 'SUMMARY.md').write_text('\n'.join(lines), 'utf-8')
        bundle = None
        if not self.integrity_failures:
            bundle = self.out / 'x4_tisasrec_three_seed_results.zip'
            paths = sorted(p for p in self.out.rglob('*') if p.is_file())
            with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
                for p in paths:
                    if p.is_symlink():raise ValueError('Output symlink rejected')
                    z.write(p, p.relative_to(self.out).as_posix())
            digest = hashlib.sha256()
            with bundle.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):digest.update(block)
            bundle.with_name(bundle.name + '.sha256.txt').write_text(digest.hexdigest() + '  ' + bundle.name + '\n', 'ascii')
        return {'status': status, 'report_dir': str(self.out), 'zip_path': str(bundle) if bundle else None, 'exit_code': 2 if self.integrity_failures else 0 if complete else 1}


def collect(stage):
    c = Collector(stage);summaries = {}
    try:c.sources()
    except (OSError, ValueError, KeyError, TypeError) as exc:c.integrity_failures.append(type(exc).__name__ + ': ' + str(exc)[:420])
    if not c.integrity_failures:
        paths = ['verification/driver.json'] + ['verification/smoke_' + t + '.json' for t in TAGS]
        paths += ['verification/smoke_' + t + '_seed42.log' for t in TAGS]
        paths += ['verification/train_%s_seed%d.log' % (t, s) for t in TAGS for s in SEEDS]
        for path in paths:c.copy(path)
        for tag in TAGS:
            c.smoke(tag)
            for seed in SEEDS:c.run(tag, seed)
        summaries = c.summaries();c.driver()
    return c.finish(summaries)


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--stage', required=True, type=Path);args = p.parse_args()
    result = collect(args.stage.expanduser());checks = Path(result['report_dir']) / 'x4_checks.json'
    print('CHECKS=' + str(checks));print('SUMMARY=' + str(checks.with_name('SUMMARY.md')))
    print('X4_TISASREC_REPORT_ZIP=' + (result['zip_path'] or 'NOT_CREATED'));print('STATUS=' + result['status'])
    report = json_object(checks.read_bytes())
    for message in report['integrity_failures']:print('INTEGRITY_FAILURE=' + message)
    for name, run in report['run_checks'].items():
        for message in run['errors']:print('RUN_FAILURE=' + name + ': ' + message)
    for name, run in report['smoke_checks'].items():
        for message in run['errors']:print('SMOKE_FAILURE=' + name + ': ' + message)
    for message in report['driver_check'].get('errors', []):print('DRIVER_FAILURE=' + message)
    for name, passed in report['driver_check'].get('checks', {}).items():
        if not passed:print('DRIVER_FAILED_CHECK=' + name)
    return result['exit_code']


if __name__ == '__main__':
    try:sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print('COLLECTION_FAILED=' + type(exc).__name__ + ': ' + str(exc)[:420], file=sys.stderr);sys.exit(2)
