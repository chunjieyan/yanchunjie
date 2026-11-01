#!/usr/bin/env python3
"""Collect twelve X3 training-shuffle saved runs without research execution.

Standard library only: no original interaction CSV or checkpoint stat/content
access. New augmentation proofs are provenance, not locally rerun training.
"""
import argparse
import ast
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

INPUTS_FILE = Path(__file__).absolute().with_name('x3_shuffle_inputs_20261025.json')
EXPECTED_INPUTS_SHA256 = '3788d6ce525966c74663a7e2e61b4bc20c9eaac5d2e6c7a5a8e33fc46f4dfce3'
TAGS = {'ml1m': 'MovieLens-1M', 'beauty': 'Amazon_Beauty', 'sports': 'Amazon_Sports', 'toys': 'Amazon_Toys'}
SEEDS = (42, 2026, 3407)
REFERENCE_FILES = ('config.json', 'metrics.json', 'training_log.csv', 'item_id_map.csv', 'test_per_sample_predictions.csv')
METRICS = ('ndcg10', 'hr10', 'hr20', 'recall50', 'mrr', 'mrr10', 'mean_rank')
MANIFEST = 'manifests/x3_shuffle_manifest.json'
GENERATED = {'scripts/model_definition.py'} | {'scripts/train_%s_shuffle.py' % tag for tag in TAGS}
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
        for key, expected in (('ndcg10', 1 / math.log2(rank + 1) if rank <= 10 else 0), ('hit10', int(rank <= 10)), ('hit20', int(rank <= 20))):
            if key in row and not close(row[key], expected):
                raise ValueError('Saved per-row metric disagrees with rank')
        result[sid] = (user, target, rank, raw, indices)
    if not result or len({v[0] for v in result.values()}) != len(result):
        raise ValueError('Empty panel or duplicate user')
    return result


def prediction_diagnostics(records):
    return {'rank_le10_vs_target_membership_mismatches': sum((row[2] <= 10) != (row[1] in row[3]) for row in records.values()),
            'target_top10_position_vs_strict_rank_mismatches': sum(row[3].index(row[1]) + 1 != row[2] for row in records.values() if row[1] in row[3]),
            'interpretation': 'Rank metrics preserve strict-greater-plus-one target ranks. Independent torch.topk membership/position differences are compatible with score ties but do not confirm ties without score counts; ranks are not rewritten.'}


def panel(records):
    return {sid: row[:2] for sid, row in records.items()}


def accuracy(records):
    ranks = [r[2] for r in records.values()]
    return {'samples': len(ranks), 'ndcg10': statistics.mean([1 / math.log2(r + 1) if r <= 10 else 0 for r in ranks]),
            'hr10': statistics.mean([r <= 10 for r in ranks]), 'hr20': statistics.mean([r <= 20 for r in ranks]),
            'recall50': statistics.mean([r <= 50 for r in ranks]), 'mrr': statistics.mean([1 / r for r in ranks]),
            'mrr10': statistics.mean([1 / r if r <= 10 else 0 for r in ranks]),
            'mean_rank': statistics.mean(ranks), 'median_rank': statistics.median(ranks),
            'rank1_ratio': statistics.mean([r == 1 for r in ranks])}


def metric_record(value, expected_samples, num_items):
    if integer(value['samples']) != integer(expected_samples):
        raise ValueError('Saved metric sample count mismatch')
    data = {k: finite(value[k], nonnegative=True) for k in ('ndcg10', 'hr10', 'hr20', 'mean_rank')}
    if not (0 <= data['ndcg10'] <= 1 and 0 <= data['hr10'] <= data['hr20'] <= 1 and 1 <= data['mean_rank'] <= num_items):
        raise ValueError('Saved accuracy outside valid range')
    return data


def normalize_metrics(saved, phase):
    value = saved[phase]
    if 'overall' in value:
        overall = value['overall']
        names = {'samples': 'N', 'ndcg10': 'NDCG@10', 'hr10': 'HR@10', 'hr20': 'HR@20', 'mean_rank': 'MeanRank'}
        normalized = {key: integer(overall[source]) if key == 'samples' else finite(overall[source]) for key, source in names.items()}
        for key, source in (('median_rank', 'MedianRank'), ('rank1_ratio', 'Rank1Ratio')):
            if source in overall:
                normalized[key] = finite(overall[source])
        return normalized
    return {'samples': integer(value['samples']), **{key: finite(value[key]) for key in ('ndcg10', 'hr10', 'hr20', 'mean_rank')}}


def training_trace(blob, saved, tag, max_epochs, patience, num_items):
    rows = csv_rows(blob)
    if tag not in TAGS or not rows or len(rows) > max_epochs:
        raise ValueError('Invalid tag or empty/oversized training log')
    threshold = 0.0 if tag in ('ml1m', 'beauty') else 1e-12
    best, best_epoch, bad, selected = -1.0, -1, 0, None
    for index, row in enumerate(rows, 1):
        if integer(row['epoch']) != index:
            raise ValueError('Training epoch sequence is not contiguous from one')
        loss = finite(row['train_loss'], nonnegative=True)
        current = finite(row['val_ndcg10'], nonnegative=True)
        hr = finite(row['val_hr10'], nonnegative=True)
        hr20 = finite(row['val_hr20'], nonnegative=True)
        rank = finite(row['val_mean_rank'], positive=True)
        seconds_key = 'elapsed_seconds' if tag in ('ml1m', 'beauty') else 'seconds'
        finite(row[seconds_key], nonnegative=True)
        if not 0 <= current <= 1 or not 0 <= hr <= hr20 <= 1 or not 1 <= rank <= num_items:
            raise ValueError('Training validation metric range invalid')
        if current > best + threshold:
            best, best_epoch, bad, selected = current, index, 0, row
        else:
            bad += 1
        if tag in ('sports', 'toys') and (integer(row['best_epoch']) != best_epoch or integer(row['bad_epochs']) != bad or not close(row['best_val_ndcg10'], best, 1e-12)):
            raise ValueError('Saved best/bad trace disagrees with preserved selection rule')
        if bad >= patience and index != len(rows):
            raise ValueError('Training continued after first patience trigger')
    if len(rows) < max_epochs and bad != patience:
        raise ValueError('Training ended early without reaching patience')
    validation = normalize_metrics(saved, 'validation')
    if integer(saved['best_epoch']) != best_epoch or not close(validation['ndcg10'], best, 1e-8):
        raise ValueError('Saved best epoch/validation NDCG disagrees with training log')
    for key, field in (('hr10', 'val_hr10'), ('hr20', 'val_hr20'), ('mean_rank', 'val_mean_rank')):
        if field in selected and not close(validation[key], selected[field]):
            raise ValueError('Restored best validation metric disagrees with selected log row')
    return {'completed_epochs': len(rows), 'best_epoch': best_epoch, 'best_validation_ndcg10': best,
            'final_bad_epochs': bad, 'stop_reason': 'patience' if bad >= patience else 'max_epochs',
            'improvement_threshold': threshold, 'training_loss_last': loss,
            'timing_field': 'elapsed_seconds' if tag in ('ml1m', 'beauty') else 'seconds'}



# Exact trusted helper source, parsed with emitted nodes in the same interpreter.
AUGMENTATION_REFERENCE_SOURCE = 'def event_priority(seed, dataset, user, event_key, epoch):\n    payload = json.dumps([int(seed), str(dataset), str(user), int(event_key), int(epoch)],\n                         separators=(\',\', \':\'), ensure_ascii=False).encode(\'utf-8\')\n    value = hashlib.blake2b(payload, digest_size=16).digest()\n    return int.from_bytes(value[:8], \'big\'), int.from_bytes(value[8:], \'big\')\n\ndef visible_order(items, buckets, event_keys, priorities):\n    n = len(items)\n    if not 1 <= n <= 50 or len(buckets) != n or len(event_keys) != n or len(priorities) != n:\n        raise ValueError(\'Visible history arrays must have the same length in 1..50\')\n    order, start = [], 0\n    while start < n:\n        end = start + 1\n        while end < n and buckets[end] == buckets[start]:\n            end += 1\n        if end == start + 1:\n            order.append(start)\n        else:\n            order.extend(sorted(range(start, end), key=lambda i: (priorities[i], int(event_keys[i]))))\n        start = end\n    return order\n\ndef verify_visible_invariants(items, buckets, event_keys, order, future_keys=()):\n    n = len(items)\n    if sorted(order) != list(range(n)) or len(set(event_keys)) != n or set(event_keys) & set(future_keys):\n        raise ValueError(\'Visible events changed or target/future event leaked\')\n    if [buckets[i] for i in order] != list(buckets):\n        raise ValueError(\'Visible time-bucket positions changed\')\n    start, eligible, segments = 0, 0, 0\n    while start < n:\n        end = start + 1\n        while end < n and buckets[end] == buckets[start]:\n            end += 1\n        before = Counter((int(event_keys[i]), int(items[i])) for i in range(start, end))\n        after = Counter((int(event_keys[order[i]]), int(items[order[i]])) for i in range(start, end))\n        if before != after:\n            raise ValueError(\'Per-bucket event/item multiset changed\')\n        eligible += int(end - start >= 2 and len({int(items[i]) for i in range(start, end)}) >= 2)\n        segments += 1; start = end\n    return {\'visible_events\': n, \'bucket_segments\': segments, \'eligible_bucket_segments\': eligible,\n            \'changed_visible_items\': int([int(items[i]) for i in order] != [int(i) for i in items]),\n            \'all_visible_bucket_event_multisets_preserved\': True, \'target_event_future_excluded\': True}\n\nclass SharedPriorityState:\n    """Epoch priorities shared with the original persistent Linux workers.\n\n    Allocating/filling these standard-library buffers never uses Python\'s\n    global random state, NumPy RNG, Torch RNG, or the sampler generator.\n    """\n    def __init__(self, dataset, seed, metadata, context=None):\n        self.dataset, self.seed = str(dataset), int(seed)\n        self.metadata, self.offsets = {}, {}\n        total = 0\n        for user in sorted(metadata):\n            entry = metadata[user]\n            items, buckets, keys = tuple(entry[\'items\']), tuple(entry[\'buckets\']), tuple(int(k) for k in entry[\'event_keys\'])\n            if len(items) < 2 or len(items) != len(buckets) or len(items) != len(keys) or len(set(keys)) != len(keys):\n                raise ValueError(\'Each retained training user needs unique original event keys and aligned arrays\')\n            self.metadata[str(user)] = {\'items\': items, \'buckets\': buckets, \'event_keys\': keys}\n            self.offsets[str(user)] = total; total += len(items)\n        if not total:\n            raise ValueError(\'No training events available for augmentation\')\n        ctx = context if context is not None else mp.get_context()\n        self.high = ctx.RawArray(ctypes.c_uint64, total)\n        self.low = ctx.RawArray(ctypes.c_uint64, total)\n        self.epoch = ctx.Value(\'q\', -1)\n        self.total_events = total\n\n    def set_epoch(self, epoch):\n        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:\n            raise ValueError(\'Augmentation epoch must be a nonnegative integer\')\n        with self.epoch.get_lock():\n            for user, entry in self.metadata.items():\n                offset = self.offsets[user]\n                for i, event_key in enumerate(entry[\'event_keys\']):\n                    high, low = event_priority(self.seed, self.dataset, user, event_key, epoch)\n                    self.high[offset + i] = high; self.low[offset + i] = low\n            self.epoch.value = epoch\n\n    def window(self, user, start, stop):\n        user = str(user)\n        if user not in self.metadata:\n            raise ValueError(\'Training user is absent from augmentation metadata\')\n        entry = self.metadata[user]\n        if not 0 <= start < stop < len(entry[\'items\']) or stop - start > 50:\n            raise ValueError(\'Original cutoff/visible truncation window is invalid\')\n        with self.epoch.get_lock():\n            if self.epoch.value < 0:\n                raise ValueError(\'Set the shared augmentation epoch before iterating the loader\')\n            offset = self.offsets[user]\n            priorities = [(int(self.high[offset + i]), int(self.low[offset + i])) for i in range(start, stop)]\n        items, buckets, keys = (entry[k][start:stop] for k in (\'items\', \'buckets\', \'event_keys\'))\n        return items, buckets, keys, visible_order(items, buckets, keys, priorities)\n\n    def visible_items(self, user, start, stop, reference_items):\n        items, buckets, keys, order = self.window(user, start, stop)\n        if tuple(reference_items) != items:\n            raise ValueError(\'Dataset visible prefix differs from the original frozen cutoff\')\n        return [items[i] for i in order]\n\n    def proof(self, user, start, stop):\n        items, buckets, keys, order = self.window(user, start, stop)\n        entry = self.metadata[str(user)]\n        result = verify_visible_invariants(items, buckets, keys, order, entry[\'event_keys\'][stop:])\n        target = entry[\'items\'][stop]\n        if sum(item == target for item in items) != sum(items[i] == target for i in order):\n            raise ValueError(\'A legitimate repeated target-item occurrence was changed\')\n        result.update(target_and_cutoff_fixed=True, original_visible_window_and_length_preserved=True,\n                      repeated_target_item_counts_preserved=True)\n        return result\n\ndef _augmentation_smoke_proof():\n    """Server-only small loader probe; separate generator protects global RNG."""\n    python_before = random.getstate()\n    numpy_before = np.random.get_state()\n    torch_before = torch.get_rng_state().clone()\n    cuda_before = [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []\n    n = len(train_dataset)\n    indices = sorted({i * (n - 1) // max(1, min(512, n) - 1) for i in range(min(512, n))})\n    proofs = []\n    for epoch in (1, 2):\n        _AUGMENT.set_epoch(epoch)\n        for index in indices:\n            if _TAG in (\'ml1m\', \'beauty\'):\n                user, end = train_dataset.samples[index]; stop = end - 1\n            else:\n                user_index, stop = train_dataset.positions[index]; user = _AUGMENT_USERS[user_index]\n            proofs.append(_AUGMENT.proof(user, max(0, stop - MAX_LEN), stop))\n    workers = 4 if _TAG in (\'ml1m\', \'beauty\') else 0\n    probe_indices = indices[:min(32, len(indices))]\n    generator = torch.Generator()\n    generator.manual_seed(SEED + 735781)\n    probe = DataLoader(torch.utils.data.Subset(train_dataset, probe_indices), batch_size=8, shuffle=False,\n        num_workers=workers, persistent_workers=bool(workers), generator=generator, pin_memory=False)\n    for epoch in (1, 2):\n        _AUGMENT.set_epoch(epoch)\n        expected = [train_dataset[index] for index in probe_indices]\n        offset = 0\n        for batch in probe:\n            size = len(batch[0])\n            for component, tensor in enumerate(batch):\n                wanted = torch.stack([row[component] for row in expected[offset:offset + size]])\n                if not torch.equal(tensor, wanted):\n                    raise RuntimeError(\'Persistent-worker epoch/window output differs from main process\')\n            offset += size\n        if offset != len(probe_indices):\n            raise RuntimeError(\'Persistent-worker probe sample coverage differs\')\n    del probe\n    _AUGMENT.set_epoch(0)\n    numpy_after = np.random.get_state()\n    if (random.getstate() != python_before or numpy_before[0] != numpy_after[0]\n            or not np.array_equal(numpy_before[1], numpy_after[1]) or numpy_before[2:] != numpy_after[2:]\n            or not torch.equal(torch_before, torch.get_rng_state())\n            or any(not torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all() if cuda_before else []))):\n        raise RuntimeError(\'Augmentation/probe consumed a global RNG stream\')\n    return {\'checked_visible_windows\': len(proofs), \'epoch_checks\': [1, 2], \'worker_epoch_checks\': [1, 2],\n        \'probe_samples_per_epoch\': len(probe_indices), \'probe_workers\': workers,\n        \'persistent_worker_outputs_match_main\': True, \'global_rng_unchanged_by_augmentation_and_probe\': True,\n        \'eligible_bucket_segments\': sum(row[\'eligible_bucket_segments\'] for row in proofs),\n        \'changed_visible_windows\': sum(row[\'changed_visible_items\'] for row in proofs),\n        \'all_visible_bucket_event_multisets_preserved\': True, \'target_and_cutoff_fixed\': True,\n        \'target_event_future_excluded\': True, \'original_visible_window_and_length_preserved\': True,\n        \'repeated_target_item_counts_preserved\': True, \'cross_window_event_priorities_shared\': True}\n\ndef _save_smoke_report(loss, gradients, all_finite, proof):\n    if not math.isfinite(loss) or loss <= 0 or gradients <= 0 or not all_finite:\n        raise RuntimeError(\'Smoke loss/backward gradients failed\')\n    _verify_originals()\n    if OUTPUT_DIR.exists():\n        raise RuntimeError(\'Smoke must not create a formal output directory\')\n    report = dict(_X3_PROVENANCE)\n    report.update(tag=_TAG, dataset=DATASET, seed=SEED, status=\'SMOKE_PASS\', mode=\'smoke\',\n        counts={\'num_items\': NUM_ITEMS, \'training_rows\': len(train_df), \'training_positions\': len(train_dataset),\n                \'trainable_users\': len(_AUGMENT.metadata), \'validation_samples\': len(val_df), \'test_samples\': len(test_df)},\n        smoke_loss=loss, finite_gradient_count=gradients, all_gradients_finite=bool(all_finite), invariants=proof,\n        optimizer_steps=0, formal_training_started=False, weights_read=False, weights_saved=False,\n        test_inference_performed=False, formal_output_created=False, original_hashes_verified_after=True)\n    path = _X3_STAGE / \'verification\' / (\'smoke_\' + _TAG + \'_seed\' + str(SEED) + \'.json\')\n    with path.open(\'x\', encoding=\'utf-8\') as handle:\n        handle.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + \'\\n\')\n    print(\'X3_SHUFFLE_SMOKE_PASS=\' + str(path), flush=True)\n\ndef _canonical_test_mask_proof(reference_rows, prediction_rows, item_map):\n    """Post-save audit only; preserve the protected ranking implementation."""\n    if len(reference_rows) != len(prediction_rows) or not reference_rows:\n        raise RuntimeError(\'Saved test predictions do not cover the canonical test panel\')\n    seen_samples = set()\n    violations = unknown = identity = raw_index_mismatches = 0\n    for reference, prediction in zip(reference_rows, prediction_rows):\n        sample = str(reference[\'sample_id\'])\n        if sample in seen_samples:\n            raise RuntimeError(\'Duplicate canonical test sample identity\')\n        seen_samples.add(sample)\n        identity += int(any(str(reference[key]) != str(prediction[key])\n                            for key in (\'sample_id\', \'user_id\', \'target_item\')))\n        target = str(reference[\'target_item\'])\n        unknown += int(target not in item_map)\n        history = {str(item) for item in json.loads(reference[\'sequence_items\'])}\n        masked = history - {target}\n        raw_top = [str(item) for item in json.loads(prediction[\'top10_raw_item_id\'])]\n        indices = json.loads(prediction[\'top10_item_idx\'])\n        if len(raw_top) != 10 or len(indices) != 10 or len(set(raw_top)) != 10:\n            raise RuntimeError(\'Saved Top10 list is malformed\')\n        for raw, index in zip(raw_top, indices):\n            if isinstance(index, bool) or not isinstance(index, int):\n                raise RuntimeError(\'Saved Top10 index must be an integer\')\n            raw_index_mismatches += int(raw not in item_map or item_map.get(raw) != index)\n            violations += int(raw in masked)\n    if violations or unknown or identity or raw_index_mismatches:\n        raise RuntimeError(\'Canonical test mask/identity/map audit failed: \' + json.dumps({\n            \'seen_item_violations\': violations, \'target_unknown_count\': unknown,\n            \'panel_identity_mismatches\': identity, \'top10_raw_index_mismatches\': raw_index_mismatches}))\n    return {\'canonical_evaluation_mask_verified\': True, \'test_samples_verified\': len(reference_rows),\n        \'test_top10_seen_item_violations\': violations, \'test_target_unknown_count\': unknown,\n        \'test_panel_identity_mismatches\': identity, \'test_top10_raw_index_mismatches\': raw_index_mismatches,\n        \'test_mask_rule\': \'unique canonical history items except target; full frozen Control catalog\'}\n'


class Collector:
    def __init__(self, stage):
        self.stage = absolute_plain(stage)
        if not self.stage.name.startswith('kais_x3_shuffle_v1_'):
            raise ValueError('Unexpected X3 stage prefix')
        verification = safe(self.stage, 'verification')
        if not verification.is_dir():
            raise ValueError('Verification directory missing')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
        self.out = verification / ('x3_train_review_' + stamp)
        self.out.mkdir(exist_ok=False)
        self.files, self.warnings, self.integrity_failures, self.copies = [], [], [], {}
        self.total, self.runs, self.references, self.sources, self.generated, self.smokes = 0, {}, {}, {}, {}, {}
        self.driver_check = {'status': 'NOT_CHECKED'}

    def copy(self, origin, relative, archive_path, expected=None):
        record = {'origin': str(origin), 'path': relative, 'archive_path': archive_path, 'status': 'unknown'}
        try:
            relative_path(archive_path)
            blob, actual = read_stable(origin, relative)
            if expected is not None and actual != expected:
                raise ValueError('Pinned SHA256 mismatch')
            if self.total + len(blob) > TOTAL_LIMIT:
                raise ValueError('Collection exceeds total size limit')
            target = safe(self.out, 'snapshot/' + archive_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(blob)
            self.copies[archive_path] = blob
            self.total += len(blob)
            record.update(status='copied', bytes=len(blob), sha256=actual)
            return blob
        except (OSError, ValueError) as exc:
            record.update(status='missing' if isinstance(exc, FileNotFoundError) else 'rejected_or_changed', error=str(exc)[:320])
            self.warnings.append(record['status'] + ': ' + relative)
            if expected is not None and isinstance(exc, ValueError):
                self.integrity_failures.append('Pinned file rejected/changed: ' + relative)
            return None
        finally:
            self.files.append(record)

    def verify_sources(self):
        companion = absolute_plain(INPUTS_FILE)
        raw, digest = read_stable(companion.parent, companion.name)
        if digest != EXPECTED_INPUTS_SHA256:
            raise ValueError('Companion input JSON differs from reviewed SHA256')
        self.inputs = json_object(raw)
        blob, self.manifest_sha = read_stable(self.stage, MANIFEST)
        self.manifest = m = json_object(blob)
        if (m.get('version') != 'x3_shuffle_v1' or m.get('inputs') != self.inputs or m.get('inputs_sha256') != digest
                or m.get('sources_sha256') != self.inputs['sources_sha256'] or m.get('data_sha256') != self.inputs['data_sha256']
                or m.get('training_seeds') != list(SEEDS) or m.get('augmentation_protocol') != self.inputs['augmentation_protocol']):
            raise ValueError('Preparation/input/source/data/augmentation binding mismatch')
        self.root = absolute_plain(m['root'])
        if self.stage.parent != self.root or m.get('stage') != str(self.stage):
            raise ValueError('Stage/root identity mismatch')
        if (self.inputs.get('version') != 'x3_shuffle_v1' or self.inputs.get('training_seeds') != list(SEEDS)
                or set(self.inputs['datasets']) != set(TAGS) or len(self.inputs['sources_sha256']) != 4
                or len(self.inputs['data_sha256']) != 12 or set(m['generated_sha256']) != GENERATED):
            raise ValueError('Unexpected source/data/dataset/seed/script inventory')
        planned_sources, planned_data = set(), set()
        expected_refs = {}
        for tag, plan in self.inputs['datasets'].items():
            if plan['dataset'] != TAGS[tag] or set(plan['control_reference']['files_sha256']) != set(REFERENCE_FILES):
                raise ValueError('Unexpected dataset or frozen Control metadata inventory')
            planned_sources.add(relative_path(plan['trainer_source']))
            planned_data.update(relative_path(plan[k]) for k in ('train_path', 'validation_path', 'test_path'))
            expected_refs.update({'references/%s_control_seed42/%s' % (tag, n): h for n, h in plan['control_reference']['files_sha256'].items()})
        if (set(self.inputs['sources_sha256']) != planned_sources or set(self.inputs['data_sha256']) != planned_data
                or self.inputs['model_source'] not in planned_sources or m.get('reference_files_sha256') != expected_refs):
            raise ValueError('Source/data/Control reference closure mismatch')
        for relative, expected in self.inputs['sources_sha256'].items():
            source, actual = read_stable(self.root, relative)
            if not relative.endswith('.py') or actual != expected:
                raise ValueError('Original source SHA256/extension mismatch: ' + relative)
            ast.parse(source.decode('utf-8-sig'))
            self.sources[relative] = source
        for relative, expected in m['generated_sha256'].items():
            source, actual = read_stable(self.stage, relative)
            if actual != expected:
                raise ValueError('Generated script SHA256 mismatch: ' + relative)
            ast.parse(source.decode('utf-8-sig'))
            self.generated[relative] = source
        self.algorithm_ast()
        for relative in self.sources:
            if self.copy(self.root, relative, 'original_sources/' + PurePosixPath(relative).name, self.inputs['sources_sha256'][relative]) is None:
                raise ValueError('Source changed during collection')
        for relative in self.generated:
            if self.copy(self.stage, relative, relative, m['generated_sha256'][relative]) is None:
                raise ValueError('Generated script changed during collection')
        if self.copy(self.stage, MANIFEST, MANIFEST, self.manifest_sha) is None or self.copy(companion.parent, companion.name, 'manifests/pinned_inputs.json', digest) is None:
            raise ValueError('Manifest changed during collection')

    def algorithm_ast(self):
        frozen = ast_definitions(self.generated['scripts/model_definition.py'])
        original_model = ast_definitions(self.sources[self.inputs['model_source']])
        if set(frozen) != {'SASRecControl'} or frozen['SASRecControl'] != original_model.get('SASRecControl'):
            raise ValueError('Frozen SASRecControl class AST differs')
        helpers = ast_definitions(AUGMENTATION_REFERENCE_SOURCE.encode('utf-8'))
        def definitions(tree):
            return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        def dump(node):
            return ast.dump(node, include_attributes=False)
        for tag, plan in self.inputs['datasets'].items():
            original_tree = ast.parse(self.sources[plan['trainer_source']].decode('utf-8-sig'))
            generated_tree = ast.parse(self.generated['scripts/train_%s_shuffle.py' % tag].decode('utf-8-sig'))
            old, new = definitions(original_tree), definitions(generated_tree)
            for name, expected in helpers.items():
                if name not in new or dump(new[name]) != expected:
                    raise ValueError('Augmentation/proof helper AST differs: %s/%s' % (tag, name))
            protected = ('load_r1_eval', 'train_one_batch', 'calculate_metrics', 'evaluate', 'build_control_group_positions') if tag in ('ml1m', 'beauty') else ('load_strict_samples', 'loss_for_batch', 'evaluate')
            for name in protected:
                if name not in old or name not in new or dump(old[name]) != dump(new[name]):
                    raise ValueError('Protected load/loss/evaluation AST differs: %s/%s' % (tag, name))
            def loader_flow(tree):
                return [dump(n) for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and dotted_name(n.value.func) == 'DataLoader']
            def rng_flow(tree):
                return [dump(n) for n in tree.body if
                    (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and dotted_name(n.value.func) in ('random.seed', 'np.random.seed', 'torch.manual_seed'))
                    or (isinstance(n, ast.If) and any(isinstance(x, ast.Call) and dotted_name(x.func) == 'torch.cuda.manual_seed_all' for x in ast.walk(n)))]
            if loader_flow(original_tree) != loader_flow(generated_tree) or rng_flow(original_tree) != rng_flow(generated_tree):
                raise ValueError('Original sampler/loader or RNG reset flow changed: ' + tag)
            dataset_name = 'ControlTrainDataset' if tag in ('ml1m', 'beauty') else 'AllPositionsDataset'
            dataset = new[dataset_name]
            getitem = next(n for n in dataset.body if isinstance(n, ast.FunctionDef) and n.name == '__getitem__')
            insertion = ('history_items = np.asarray(_AUGMENT.visible_items(user_id, history_start, target_idx, history_items), dtype=np.int64)'
                         if tag in ('ml1m', 'beauty') else 'history = np.asarray(_AUGMENT.visible_items(_AUGMENT_USERS[u], max(0, p - MAX_LEN), p, history), dtype=np.int64)')
            expected = dump(ast.parse(insertion).body[0])
            positions = [i for i, n in enumerate(getitem.body) if dump(n) == expected]
            anchor_name = 'history_buckets' if tag in ('ml1m', 'beauty') else 'history'
            anchors = [i for i, n in enumerate(getitem.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == anchor_name for t in n.targets) and dump(n) != expected]
            if len(positions) != 1 or len(anchors) != 1 or positions[0] != anchors[0] + 1:
                raise ValueError('Visible-history insertion is not exactly after original truncation: ' + tag)
            del getitem.body[positions[0]]
            if dump(dataset) != dump(old[dataset_name]):
                raise ValueError('Dataset cutoff/target/truncation changed beyond augmentation: ' + tag)
            oldloops = [n for n in original_tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch']
            newloops = [n for n in generated_tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch']
            if len(oldloops) != 1 or len(newloops) != 1:
                raise ValueError('Expected one original epoch loop: ' + tag)
            loop = newloops[0]
            refresh = dump(ast.parse('_AUGMENT.set_epoch(epoch)').body[0])
            positions = [i for i, n in enumerate(loop.body) if dump(n) == refresh]
            timer_name = 'epoch_begin' if tag in ('ml1m', 'beauty') else 'start_time'
            timers = [i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == timer_name for t in n.targets)]
            if len(positions) != 1 or len(timers) != 1 or positions[0] != timers[0] + 1:
                raise ValueError('Epoch setup is not inside original timer before iteration: ' + tag)
            del loop.body[positions[0]]
            if dump(loop) != dump(oldloops[0]):
                raise ValueError('Original training loop changed beyond timed priority refresh: ' + tag)
        self.algorithm_check = {'status': 'PASS', 'scope': 'Frozen model, original load/loss/evaluation, Dataset with only approved visible-window insertion, and epoch loop with only timed priority refresh match exact AST. Trusted augmentation and post-save proof helper source is parsed beside emitted source in the current interpreter. No submitted source executed.'}

    def control_references(self, tag):
        plan, blobs = self.inputs['datasets'][tag], {}
        ref = plan['control_reference']
        for name in REFERENCE_FILES:
            original, actual = read_stable(self.root, ref['model_dir'] + '/' + name)
            if actual != ref['files_sha256'][name]:
                raise ValueError('Original Control reference SHA mismatch: ' + tag + '/' + name)
            relative = 'references/%s_control_seed42/%s' % (tag, name)
            blobs[name] = self.copy(self.stage, relative, relative, actual)
        if any(b is None for b in blobs.values()):
            raise ValueError('Pinned Control reference missing/rejected: ' + tag)
        mapping = item_mapping(blobs['item_id_map.csv'])
        rows = predictions(blobs['test_per_sample_predictions.csv'], mapping)
        config, metrics = json_object(blobs['config.json']), json_object(blobs['metrics.json'])
        if (len(mapping) != integer(plan['expected_items']) or len(rows) != integer(plan['expected_test'])
                or config.get('dataset') != TAGS[tag] or metrics.get('dataset') != TAGS[tag]
                or integer(config.get('seed')) != 42 or integer(metrics.get('seed')) != 42):
            raise ValueError('Control reference identity/count mismatch: ' + tag)
        for phase, expected in (('validation', plan['expected_validation']), ('test', plan['expected_test'])):
            metric_record(normalize_metrics(metrics, phase), expected, len(mapping))
        derived = accuracy(rows)
        normalized = normalize_metrics(metrics, 'test')
        if any(not close(derived[k], v) for k, v in normalized.items()):
            raise ValueError('Control test rank metrics disagree with saved metrics: ' + tag)
        self.references[tag] = {'map': mapping, 'panel': panel(rows), 'diagnostics': prediction_diagnostics(rows)}

    def provenance(self, tag):
        plan = self.inputs['datasets'][tag]
        return {'version': 'x3_shuffle_v1', 'input_manifest_sha256': EXPECTED_INPUTS_SHA256,
            'preparation_manifest_sha256': self.manifest_sha,
            'trainer_sha256': self.manifest['generated_sha256']['scripts/train_%s_shuffle.py' % tag],
            'frozen_model_sha256': self.manifest['generated_sha256']['scripts/model_definition.py'],
            'sources_sha256': self.inputs['sources_sha256'],
            'data_sha256': {plan[k]: self.inputs['data_sha256'][plan[k]] for k in ('train_path', 'validation_path', 'test_path')},
            'control_reference': plan['control_reference'],
            'augmentation': {'visible_history_first': True, 'target_and_cutoff_fixed': True,
                'all_visible_time_bucket_segments': True, 'shared_event_epoch_priorities': True, 'global_rng_consumed': False,
                'event_key_column': plan['event_key_column'], 'priority_algorithm': 'BLAKE2b-128(json([seed,dataset,user,eventkey,epoch]))',
                'worker_epoch_shared': True, 'epoch_setup_included_in_epoch_seconds': True,
                'legitimate_repeated_target_item_ids_retained': True},
            'original_sampler_loader_flow_preserved': True, 'canonical_validation_test': True}

    def counts(self, value, tag):
        plan = self.inputs['datasets'][tag]
        expected = {'num_items': plan['expected_items'], 'training_rows': plan['expected_training_rows'],
            'training_positions': plan['expected_training_positions'], 'validation_samples': plan['expected_validation'], 'test_samples': plan['expected_test']}
        if set(value) != set(expected) | {'trainable_users'} or any(integer(value[k]) != integer(v) for k, v in expected.items()):
            raise ValueError('Saved row/legal-position/catalog/panel counts mismatch')
        if not 1 <= integer(value['trainable_users']) <= integer(value['training_rows']) - integer(value['training_positions']):
            raise ValueError('Trainable-user count invalid')
        return True

    def smoke(self, tag):
        result = {'status': 'MISSING', 'errors': []}
        self.smokes[tag] = result
        relative = 'verification/smoke_%s_seed42.json' % tag
        blob = self.copies.get(relative)
        if blob is None:
            result['errors'].append('Required seed42 augmentation smoke summary missing')
            return
        try:
            value = json_object(blob)
            if any(value.get(k) != v for k, v in self.provenance(tag).items()):
                self.integrity_failures.append('Smoke provenance/hash/protocol mismatch: ' + tag)
                raise ValueError('Smoke provenance/hash/protocol mismatch')
            if (value.get('status') != 'SMOKE_PASS' or value.get('mode') != 'smoke' or value.get('tag') != tag
                    or value.get('dataset') != TAGS[tag] or integer(value.get('seed')) != 42
                    or value.get('all_gradients_finite') is not True or integer(value.get('finite_gradient_count')) <= 0):
                raise ValueError('Smoke identity/finite gradients invalid')
            loss = finite(value['smoke_loss'], positive=True)
            self.counts(value['counts'], tag)
            if any(value.get(k) is not False for k in ('formal_training_started', 'weights_read', 'weights_saved', 'test_inference_performed', 'formal_output_created')) or integer(value.get('optimizer_steps')) != 0 or value.get('original_hashes_verified_after') is not True:
                raise ValueError('Smoke training/checkpoint/original-preservation scope invalid')
            proof = value['invariants']
            flags = ('persistent_worker_outputs_match_main', 'global_rng_unchanged_by_augmentation_and_probe',
                'all_visible_bucket_event_multisets_preserved', 'target_and_cutoff_fixed', 'target_event_future_excluded',
                'original_visible_window_and_length_preserved', 'repeated_target_item_counts_preserved', 'cross_window_event_priorities_shared')
            n = integer(self.inputs['datasets'][tag]['expected_training_positions'])
            if (any(proof.get(k) is not True for k in flags) or proof.get('epoch_checks') != [1, 2] or proof.get('worker_epoch_checks') != [1, 2]
                    or integer(proof.get('checked_visible_windows')) != 2 * min(512, n)
                    or integer(proof.get('probe_samples_per_epoch')) != min(32, n)
                    or integer(proof.get('probe_workers')) != (4 if tag in ('ml1m', 'beauty') else 0)
                    or integer(proof.get('eligible_bucket_segments')) < 0 or integer(proof.get('changed_visible_windows')) < 0
                    or integer(proof['changed_visible_windows']) > integer(proof['checked_visible_windows'])):
                raise ValueError('Smoke visible-event/worker-epoch/RNG proof invalid')
            result.update(status='PASS', smoke_loss=loss, finite_gradient_count=integer(value['finite_gradient_count']), counts=value['counts'], invariants=proof)
        except (ValueError, KeyError, TypeError) as exc:
            result.update(status='FAIL')
            result['errors'].append(type(exc).__name__ + ': ' + str(exc)[:320])

    def run(self, tag, seed):
        key = '%s_seed%d' % (tag, seed)
        output = 'outputs/%s_shuffle_seed%d' % (tag, seed)
        result = {'tag': tag, 'dataset': TAGS[tag], 'method': 'TrainingShuffle', 'seed': seed, 'status': 'INCOMPLETE', 'checks': {}, 'errors': []}
        self.runs[key] = result
        try:
            marker = 'verification/completed_%s_seed%d.json' % (tag, seed)
            blob = self.copy(self.stage, marker, marker)
            if blob is None:
                for filename in REFERENCE_FILES:
                    self.copy(self.stage, output + '/' + filename, output + '/' + filename)
                result['errors'].append('Completion marker absent; partial metadata retained without statistics.')
                return
            completed = json_object(blob)
            provenance = self.provenance(tag)
            if any(completed.get(k) != v for k, v in provenance.items()):
                self.integrity_failures.append('Completion provenance/hash/protocol mismatch: ' + key)
                raise ValueError('Completion provenance/hash/protocol mismatch')
            plan = self.inputs['datasets'][tag]
            if (completed.get('status') != 'COMPLETED' or completed.get('mode') != 'train' or completed.get('tag') != tag
                    or completed.get('dataset') != TAGS[tag] or integer(completed.get('seed')) != seed
                    or completed.get('output_dir') != str(self.stage / output)
                    or completed.get('checkpoint') != 'matched_control_sasrec_best.pt'
                    or completed.get('all_original_hashes_verified') is not True or completed.get('originals_unchanged') is not True):
                raise ValueError('Completion identity/original-preservation mismatch')
            self.counts(completed['counts'], tag)
            if (completed.get('canonical_evaluation_mask_verified') is not True
                    or integer(completed.get('test_samples_verified')) != integer(plan['expected_test'])
                    or any(integer(completed.get(k)) != 0 for k in ('test_top10_seen_item_violations', 'test_target_unknown_count', 'test_panel_identity_mismatches', 'test_top10_raw_index_mismatches'))
                    or completed.get('test_mask_rule') != 'unique canonical history items except target; full frozen Control catalog'):
                raise ValueError('Server canonical-test mask/identity proof failed or absent')
            hashes = completed['output_metadata_sha256']
            if set(hashes) != set(REFERENCE_FILES):
                raise ValueError('Completion must seal exactly five saved metadata files')
            blobs = {n: self.copy(self.stage, output + '/' + n, output + '/' + n, hashes[n]) for n in REFERENCE_FILES}
            if any(b is None for b in blobs.values()):
                result['errors'].append('Saved metadata missing/rejected; no statistics.')
                return
            config, metrics = json_object(blobs['config.json']), json_object(blobs['metrics.json'])
            mapping = item_mapping(blobs['item_id_map.csv'])
            rows = predictions(blobs['test_per_sample_predictions.csv'], mapping)
            checks = {'dataset': config.get('dataset') == metrics.get('dataset') == TAGS[tag],
                'seed': integer(config.get('seed')) == integer(metrics.get('seed')) == seed,
                'config_provenance': config.get('x3_shuffle') == provenance,
                'item_map_exact_Control': mapping == self.references[tag]['map'],
                'canonical_sample_user_target_panel_exact_Control': panel(rows) == self.references[tag]['panel'],
                'num_items': len(mapping) == integer(plan['expected_items']),
                'samples': len(rows) == integer(plan['expected_test'])}
            for field, expected in self.inputs['hyperparameters'].items():
                source_field = {'training_batch_size': 'batch_size', 'max_epochs': 'epochs'}.get(field, field) if tag in ('ml1m', 'beauty') else field
                if tag in ('ml1m', 'beauty') and field == 'evaluation_batch_size':
                    # The original evaluation function AST protects this unsaved default.
                    continue
                checks['hyperparameter_' + field] = config.get(source_field) == expected
            if tag in ('ml1m', 'beauty'):
                checks.update(method=config.get('method') == metrics.get('method') == 'SASRec_Control',
                    legal_positions=config.get('training_positions') == 'all_legal_next_item_positions',
                    loss=config.get('training_objective') == 'target_level_full_ranking_cross_entropy',
                    sampling=config.get('training_cutoff_sampling') == 'all_positions_shuffled_each_epoch',
                    evaluation=config.get('evaluation') == 'full ranking',
                    protocols=config.get('validation_protocol') == config.get('test_protocol') == 'strict')
            else:
                checks.update(model=config.get('model') == metrics.get('model') == 'SASRecControl',
                    legal_positions=integer(config.get('legal_training_positions')) == integer(plan['expected_training_positions']),
                    item_count=integer(config.get('num_items')) == len(mapping),
                    trainable_users=integer(config.get('trainable_users')) == integer(completed['counts']['trainable_users']),
                    loss=config.get('loss') == 'full_item_cross_entropy',
                    sampling=config.get('sampling') == 'all_legal_next_item_positions_reshuffled_every_epoch',
                    optimizer=config.get('optimizer') == 'Adam',
                    early_stop_metric=config.get('early_stop_metric') == 'validation_ndcg10')
                for field in ('train_path', 'validation_path', 'test_path'):
                    checks['path_' + field] = config.get(field) == str(self.root / plan[field])
                checks['item_source_is_frozen_Control_map'] = config.get('item_source_path') == str(self.root / (plan['control_reference']['model_dir'] + '/item_id_map.csv'))
            result['checks'] = checks
            if not checks['config_provenance']:
                self.integrity_failures.append('Saved config provenance/hash/protocol mismatch: ' + key)
            if not all(checks.values()):
                raise ValueError('Saved config/protocol mismatch: ' + ','.join(k for k, ok in checks.items() if not ok))
            validation, saved_test = normalize_metrics(metrics, 'validation'), normalize_metrics(metrics, 'test')
            metric_record(validation, plan['expected_validation'], len(mapping))
            metric_record(saved_test, plan['expected_test'], len(mapping))
            derived = accuracy(rows)
            if any(not close(derived[k], v) for k, v in saved_test.items()):
                raise ValueError('Saved canonical test accuracy disagrees with preserved strict-greater target ranks')
            trace = training_trace(blobs['training_log.csv'], metrics, tag, integer(self.inputs['hyperparameters']['max_epochs']), integer(self.inputs['hyperparameters']['patience']), len(mapping))
            if integer(completed.get('best_epoch')) != trace['best_epoch']:
                raise ValueError('Completion best epoch disagrees with saved trace')
            diagnostics = prediction_diagnostics(rows)
            if any(diagnostics[k] for k in ('rank_le10_vs_target_membership_mismatches', 'target_top10_position_vs_strict_rank_mismatches')):
                self.warnings.append(key + ': strict-greater rank differs from independent Top10 membership/position; see possible score-tie-compatible diagnostics, ranks preserved.')
            result.update(status='VALID_SAVED_RUN', best_epoch=trace['best_epoch'], training_trace=trace,
                canonical_test=derived, validation=validation, prediction_diagnostics=diagnostics,
                server_canonical_mask_record='PASS; audited as server record, raw histories not independently reread', checkpoint_statted_or_read=False)
            result['_map'], result['_panel'] = mapping, panel(rows)
            result['_core'] = {k: v for k, v in config.items() if k not in ('seed', 'x3_shuffle')}
        except (OSError, ValueError, KeyError, TypeError, SyntaxError, csv.Error) as exc:
            result.update(status='INVALID_NO_STATISTICS')
            result['errors'].append(type(exc).__name__ + ': ' + str(exc)[:400])

    def summaries(self):
        result = {}
        for tag in TAGS:
            runs = [self.runs['%s_seed%d' % (tag, seed)] for seed in SEEDS]
            summary = {'status': 'INCOMPLETE_NO_THREE_SEED_SUMMARY'}
            result[tag] = summary
            if any(r['status'] != 'VALID_SAVED_RUN' for r in runs):
                continue
            aligned = {'same_item_map': all(r['_map'] == runs[0]['_map'] for r in runs),
                'same_canonical_sample_user_target_panel': all(r['_panel'] == runs[0]['_panel'] for r in runs),
                'same_scientific_config_except_seed_provenance': all(r['_core'] == runs[0]['_core'] for r in runs)}
            summary['alignment'] = aligned
            if not all(aligned.values()):
                summary['status'] = 'CROSS_SEED_MISMATCH_NO_SUMMARY'
                continue
            summary.update(status='THREE_SEED_SAVED_RUN_SUMMARY', method='TrainingShuffle', seeds=list(SEEDS),
                canonical_test_samples=runs[0]['canonical_test']['samples'], best_epochs=[r['best_epoch'] for r in runs],
                test_mean_sample_SD={metric: {'mean': statistics.mean([r['canonical_test'][metric] for r in runs]),
                    'sample_SD_ddof1': statistics.stdev([r['canonical_test'][metric] for r in runs])} for metric in METRICS},
                validation_mean_sample_SD={metric: {'mean': statistics.mean([r['validation'][metric] for r in runs]),
                    'sample_SD_ddof1': statistics.stdev([r['validation'][metric] for r in runs])} for metric in ('ndcg10', 'hr10', 'hr20', 'mean_rank')})
        return result

    def driver(self):
        names = [k for k in self.copies if re.fullmatch(r'verification/x3_shuffle_driver_[^/]+\.json', k)]
        self.driver_check = {'status': 'MISSING' if not names else 'FAIL', 'errors': []}
        if len(names) != 1:
            self.driver_check['errors'].append('Exactly one driver JSON is required for all-twelve status.')
            return
        try:
            value = json_object(self.copies[names[0]])
            planned = {'%s_seed%d' % (tag, seed) for tag in TAGS for seed in SEEDS}
            checks = {'version': value.get('version') == 'x3_shuffle_v1', 'stage': value.get('stage') == str(self.stage),
                'root': value.get('root') == str(self.root), 'inputs_sha256': value.get('inputs_sha256') == EXPECTED_INPUTS_SHA256,
                'training_seeds': value.get('training_seeds') == list(SEEDS),
                'status': value.get('status') == 'ALL_FOUR_SMOKES_AND_TWELVE_TRAININGS_EXITED_ZERO',
                'four_smokes_only': set(value.get('smokes', {})) == set(TAGS), 'twelve_runs_only': set(value.get('runs', {})) == planned}
            smoke_finishes, train_starts, logs = [], [], []
            for tag in TAGS:
                for phase, seed in [('smoke', 42)] + [('train', s) for s in SEEDS]:
                    key = tag if phase == 'smoke' else '%s_seed%d' % (tag, seed)
                    run = value['smokes' if phase == 'smoke' else 'runs'][key]
                    prefix = phase + '_' + key
                    command = [value['python'], '-u', str(self.stage / ('scripts/train_%s_shuffle.py' % tag)), '--mode', phase, '--seed', str(seed)]
                    log = Path(run['log'])
                    checks[prefix + '_command'] = run.get('command') == command
                    checks[prefix + '_exit_zero'] = integer(run.get('exit_code')) == 0
                    checks[prefix + '_verified'] = run.get('summary_verified' if phase == 'smoke' else 'completion_verified') is True
                    checks[prefix + '_record_valid'] = self.smokes[tag]['status'] == 'PASS' if phase == 'smoke' else self.runs[key]['status'] == 'VALID_SAVED_RUN'
                    checks[prefix + '_log_collected'] = log.is_absolute() and log.parent == self.stage / 'verification' and 'verification/' + log.name in self.copies
                    proof_relative = ('verification/smoke_%s_seed42.json' % tag if phase == 'smoke' else 'verification/completed_%s.json' % key)
                    proof_key = 'summary' if phase == 'smoke' else 'completion'
                    proof_blob = self.copies.get(proof_relative)
                    checks[prefix + '_proof_path_sha256'] = (run.get(proof_key) == str(self.stage / proof_relative)
                        and proof_blob is not None and run.get(proof_key + '_sha256') == hashlib.sha256(proof_blob).hexdigest())
                    if not isinstance(run.get('started_utc'), str) or not isinstance(run.get('finished_utc'), str):
                        raise ValueError('Driver timestamps must be saved UTC strings')
                    started = datetime.datetime.fromisoformat(run['started_utc'].replace('Z', '+00:00'))
                    finished = datetime.datetime.fromisoformat(run['finished_utc'].replace('Z', '+00:00'))
                    checks[prefix + '_timestamps'] = started.tzinfo is not None and finished.tzinfo is not None and started <= finished
                    (smoke_finishes if phase == 'smoke' else train_starts).append(finished if phase == 'smoke' else started)
                    logs.append(str(log))
            checks['distinct_sixteen_logs'] = len(set(logs)) == 16
            checks['four_smokes_finished_before_training'] = max(smoke_finishes) <= min(train_starts)
            self.driver_check.update(status='PASS' if all(checks.values()) else 'FAIL', path=names[0], checks=checks)
            if not all(checks.values()):
                self.driver_check['errors'].append('Driver, four smoke summaries and twelve completed saved runs do not agree.')
        except (ValueError, KeyError, TypeError) as exc:
            self.driver_check['errors'].append(type(exc).__name__ + ': ' + str(exc)[:400])

    def finish(self, summaries):
        public = {name: {k: v for k, v in run.items() if not k.startswith('_')} for name, run in self.runs.items()}
        complete = (len(public) == 12 and all(r['status'] == 'VALID_SAVED_RUN' for r in public.values())
            and len(summaries) == 4 and all(s['status'] == 'THREE_SEED_SAVED_RUN_SUMMARY' for s in summaries.values())
            and len(self.smokes) == 4 and all(s['status'] == 'PASS' for s in self.smokes.values()) and self.driver_check.get('status') == 'PASS')
        status = 'INTEGRITY_FAILED_NO_BUNDLE' if self.integrity_failures else 'ALL_TWELVE_SHUFFLE_SAVED_RUNS_CHECKED' if complete else 'INCOMPLETE'
        report = {'status': status, 'stage': str(self.stage), 'input_manifest_sha256': EXPECTED_INPUTS_SHA256,
            'algorithm_ast_check': getattr(self, 'algorithm_check', {'status': 'NOT_COMPLETED'}),
            'smoke_checks': self.smokes, 'run_checks': public, 'dataset_summaries': summaries, 'driver_check': self.driver_check,
            'files': self.files, 'warnings': self.warnings, 'integrity_failures': self.integrity_failures,
            'scope': 'PRIVATE saved-file audit of twelve TrainingShuffle runs (four datasets, three seeds). No research execution, original interaction CSV or checkpoint stat/content access. Canonical accuracy is derived from original strict-greater-plus-one saved ranks; independent torch.topk membership/position diagnostics never rewrite ranks. Seven derived metrics, validation selection and stopping are checked against saved metadata/logs. Mean and sample SD (ddof=1) require all three valid, aligned seeds. Four smoke summaries and runtime mask verification are audited server records, not locally rerun tensor/inference or raw-history verification. This is not five-permutation stability, end-to-end reproduction, significance or equivalence.'}
        (self.out / 'training_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
        with (self.out / 'COLLECTED_SHA256.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['origin', 'path', 'archive_path', 'status', 'bytes', 'sha256', 'error'])
            writer.writeheader(); writer.writerows(self.files)
        with (self.out / 'three_seed_summary.csv').open('w', encoding='utf-8', newline='') as handle:
            fields = ['dataset', 'method', 'samples', 'training_seeds', 'metric', 'mean', 'sample_SD_ddof1']
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for tag, summary in summaries.items():
                if summary['status'] == 'THREE_SEED_SAVED_RUN_SUMMARY':
                    for metric, result in summary['test_mean_sample_SD'].items():
                        writer.writerow({'dataset': TAGS[tag], 'method': 'TrainingShuffle', 'samples': summary['canonical_test_samples'], 'training_seeds': '42,2026,3407', 'metric': metric, **result})
        lines = ['# PRIVATE X3 TrainingShuffle saved-run report', '', 'Status: **' + status + '**', '', report['scope'], '',
            '| Run | Check | Best epoch | Canonical test NDCG@10 |', '|---|---|---:|---:|']
        for key, run in public.items():
            lines.append('| %s | %s | %s | %s |' % (key, run['status'], run.get('best_epoch', ''), run.get('canonical_test', {}).get('ndcg10', '')))
        lines += ['', 'Rank metrics retain the original strict-greater convention. Top10 diagnostic mismatches, if present, are compatible with score ties but are not confirmed ties without score counts. HR computed from rank must not be described as observed Top10 membership when mismatches are nonzero.', '', '## Diagnostics', '']
        messages = self.integrity_failures + self.warnings + self.driver_check.get('errors', [])
        messages += [key + ': ' + error for key, run in public.items() for error in run['errors']]
        messages += ['smoke_' + key + ': ' + error for key, run in self.smokes.items() for error in run['errors']]
        lines += ['- ' + message for message in messages] or ['- No discrepancy found.']
        lines += ['', 'Private per-user predictions are included. Do not publish this bundle directly.', '']
        (self.out / 'SUMMARY.md').write_text('\n'.join(lines), 'utf-8')
        bundle = None
        if not self.integrity_failures:
            bundle = self.out / 'x3_shuffle_three_seed_results.zip'
            paths = sorted(p for p in self.out.rglob('*') if p.is_file())
            with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for path in paths:
                    if path.is_symlink():
                        raise ValueError('Output symlink rejected')
                    archive.write(path, path.relative_to(self.out).as_posix())
            digest = hashlib.sha256()
            with bundle.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            bundle.with_name(bundle.name + '.sha256.txt').write_text(digest.hexdigest() + '  ' + bundle.name + '\n', 'ascii')
        return {'report_dir': str(self.out), 'zip_path': str(bundle) if bundle else None, 'status': status, 'exit_code': 2 if self.integrity_failures else 0 if complete else 1}


def collect(stage):
    collector = Collector(stage)
    summaries = {}
    try:
        collector.verify_sources()
        for tag in TAGS:
            collector.control_references(tag)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, csv.Error) as exc:
        collector.integrity_failures.append(type(exc).__name__ + ': ' + str(exc)[:400])
    if not collector.integrity_failures:
        # Filter names/suffixes BEFORE stat: unexpected checkpoint names are never statted.
        eligible = [p for p in safe(collector.stage, 'verification').iterdir()
            if not p.name.startswith(('.', 'completed_')) and p.suffix.lower() in ('.json', '.log') and p.is_file()]
        if len(eligible) > 160:
            collector.integrity_failures.append('Unexpected verification JSON/log inventory size')
        else:
            for path in sorted(eligible):
                collector.copy(collector.stage, 'verification/' + path.name, 'verification/' + path.name)
            for tag in TAGS:
                collector.smoke(tag)
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
    print('X3_SHUFFLE_REPORT_ZIP=' + (result['zip_path'] or 'NOT_CREATED'))
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
    for tag, smoke in report['smoke_checks'].items():
        for error in smoke['errors']:
            print('SMOKE_FAILURE=' + tag + ': ' + error)
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
        print('COLLECTION_FAILED=' + type(exc).__name__ + ': ' + str(exc)[:400], file=sys.stderr)
        sys.exit(2)
