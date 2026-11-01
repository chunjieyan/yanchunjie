#!/usr/bin/env python3
# Controlled visible-history augmentation; no old training top-level imported.
import argparse
import ast
from collections import Counter
import ctypes
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import stat
parser=argparse.ArgumentParser(description='Explicit X3 visible-bucket smoke or training')
parser.add_argument('--mode',choices=['smoke','train'],required=True)
parser.add_argument('--seed',type=int,choices=[42,2026,3407],required=True)
args=parser.parse_args()
def plain(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected absolute path without ..')
    for part in (path,) + tuple(path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Symbolic link/reparse point rejected: ' + str(part))
    return path.resolve(strict=False)


def checked_file(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or PurePosixPath(relative).is_absolute() or any(p in ('', '.', '..') for p in relative.split('/'))):
        raise ValueError('Unsafe relative path')
    root = plain(root)
    path = plain(root.joinpath(*relative.split('/')))
    path.relative_to(root)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Expected regular file: ' + relative)
    return path


def file_signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def checked_sha256(root, relative, expected):
    path = checked_file(root, relative)
    before = path.stat()
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    with os.fdopen(os.open(str(path), flags), 'rb') as handle:
        if file_signature(before) != file_signature(os.fstat(handle.fileno())):
            raise RuntimeError('Input changed before hashing: ' + relative)
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if (file_signature(before) != file_signature(after)
            or file_signature(before) != file_signature(checked_file(root, relative).stat())):
        raise RuntimeError('Input changed during hashing: ' + relative)
    if expected is not None and digest.hexdigest() != expected:
        raise RuntimeError('SHA256 mismatch: ' + relative)
    return digest.hexdigest()


def event_priority(seed, dataset, user, event_key, epoch):
    payload = json.dumps([int(seed), str(dataset), str(user), int(event_key), int(epoch)],
                         separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    value = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(value[:8], 'big'), int.from_bytes(value[8:], 'big')


def visible_order(items, buckets, event_keys, priorities):
    n = len(items)
    if not 1 <= n <= 50 or len(buckets) != n or len(event_keys) != n or len(priorities) != n:
        raise ValueError('Visible history arrays must have the same length in 1..50')
    order, start = [], 0
    while start < n:
        end = start + 1
        while end < n and buckets[end] == buckets[start]:
            end += 1
        if end == start + 1:
            order.append(start)
        else:
            order.extend(sorted(range(start, end), key=lambda i: (priorities[i], int(event_keys[i]))))
        start = end
    return order


def verify_visible_invariants(items, buckets, event_keys, order, future_keys=()):
    n = len(items)
    if sorted(order) != list(range(n)) or len(set(event_keys)) != n or set(event_keys) & set(future_keys):
        raise ValueError('Visible events changed or target/future event leaked')
    if [buckets[i] for i in order] != list(buckets):
        raise ValueError('Visible time-bucket positions changed')
    start, eligible, segments = 0, 0, 0
    while start < n:
        end = start + 1
        while end < n and buckets[end] == buckets[start]:
            end += 1
        before = Counter((int(event_keys[i]), int(items[i])) for i in range(start, end))
        after = Counter((int(event_keys[order[i]]), int(items[order[i]])) for i in range(start, end))
        if before != after:
            raise ValueError('Per-bucket event/item multiset changed')
        eligible += int(end - start >= 2 and len({int(items[i]) for i in range(start, end)}) >= 2)
        segments += 1; start = end
    return {'visible_events': n, 'bucket_segments': segments, 'eligible_bucket_segments': eligible,
            'changed_visible_items': int([int(items[i]) for i in order] != [int(i) for i in items]),
            'all_visible_bucket_event_multisets_preserved': True, 'target_event_future_excluded': True}


class SharedPriorityState:
    """Epoch priorities shared with the original persistent Linux workers.

    Allocating/filling these standard-library buffers never uses Python's
    global random state, NumPy RNG, Torch RNG, or the sampler generator.
    """
    def __init__(self, dataset, seed, metadata, context=None):
        self.dataset, self.seed = str(dataset), int(seed)
        self.metadata, self.offsets = {}, {}
        total = 0
        for user in sorted(metadata):
            entry = metadata[user]
            items, buckets, keys = tuple(entry['items']), tuple(entry['buckets']), tuple(int(k) for k in entry['event_keys'])
            if len(items) < 2 or len(items) != len(buckets) or len(items) != len(keys) or len(set(keys)) != len(keys):
                raise ValueError('Each retained training user needs unique original event keys and aligned arrays')
            self.metadata[str(user)] = {'items': items, 'buckets': buckets, 'event_keys': keys}
            self.offsets[str(user)] = total; total += len(items)
        if not total:
            raise ValueError('No training events available for augmentation')
        ctx = context if context is not None else mp.get_context()
        self.high = ctx.RawArray(ctypes.c_uint64, total)
        self.low = ctx.RawArray(ctypes.c_uint64, total)
        self.epoch = ctx.Value('q', -1)
        self.total_events = total

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError('Augmentation epoch must be a nonnegative integer')
        with self.epoch.get_lock():
            for user, entry in self.metadata.items():
                offset = self.offsets[user]
                for i, event_key in enumerate(entry['event_keys']):
                    high, low = event_priority(self.seed, self.dataset, user, event_key, epoch)
                    self.high[offset + i] = high; self.low[offset + i] = low
            self.epoch.value = epoch

    def window(self, user, start, stop):
        user = str(user)
        if user not in self.metadata:
            raise ValueError('Training user is absent from augmentation metadata')
        entry = self.metadata[user]
        if not 0 <= start < stop < len(entry['items']) or stop - start > 50:
            raise ValueError('Original cutoff/visible truncation window is invalid')
        with self.epoch.get_lock():
            if self.epoch.value < 0:
                raise ValueError('Set the shared augmentation epoch before iterating the loader')
            offset = self.offsets[user]
            priorities = [(int(self.high[offset + i]), int(self.low[offset + i])) for i in range(start, stop)]
        items, buckets, keys = (entry[k][start:stop] for k in ('items', 'buckets', 'event_keys'))
        return items, buckets, keys, visible_order(items, buckets, keys, priorities)

    def visible_items(self, user, start, stop, reference_items):
        items, buckets, keys, order = self.window(user, start, stop)
        if tuple(reference_items) != items:
            raise ValueError('Dataset visible prefix differs from the original frozen cutoff')
        return [items[i] for i in order]

    def proof(self, user, start, stop):
        items, buckets, keys, order = self.window(user, start, stop)
        entry = self.metadata[str(user)]
        result = verify_visible_invariants(items, buckets, keys, order, entry['event_keys'][stop:])
        target = entry['items'][stop]
        if sum(item == target for item in items) != sum(items[i] == target for i in order):
            raise ValueError('A legitimate repeated target-item occurrence was changed')
        result.update(target_and_cutoff_fixed=True, original_visible_window_and_length_preserved=True,
                      repeated_target_item_counts_preserved=True)
        return result


def _augmentation_smoke_proof():
    """Server-only small loader probe; separate generator protects global RNG."""
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    cuda_before = [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    n = len(train_dataset)
    indices = sorted({i * (n - 1) // max(1, min(512, n) - 1) for i in range(min(512, n))})
    proofs = []
    for epoch in (1, 2):
        _AUGMENT.set_epoch(epoch)
        for index in indices:
            if _TAG in ('ml1m', 'beauty'):
                user, end = train_dataset.samples[index]; stop = end - 1
            else:
                user_index, stop = train_dataset.positions[index]; user = _AUGMENT_USERS[user_index]
            proofs.append(_AUGMENT.proof(user, max(0, stop - MAX_LEN), stop))
    workers = 4 if _TAG in ('ml1m', 'beauty') else 0
    probe_indices = indices[:min(32, len(indices))]
    generator = torch.Generator()
    generator.manual_seed(SEED + 735781)
    probe = DataLoader(torch.utils.data.Subset(train_dataset, probe_indices), batch_size=8, shuffle=False,
        num_workers=workers, persistent_workers=bool(workers), generator=generator, pin_memory=False)
    for epoch in (1, 2):
        _AUGMENT.set_epoch(epoch)
        expected = [train_dataset[index] for index in probe_indices]
        offset = 0
        for batch in probe:
            size = len(batch[0])
            for component, tensor in enumerate(batch):
                wanted = torch.stack([row[component] for row in expected[offset:offset + size]])
                if not torch.equal(tensor, wanted):
                    raise RuntimeError('Persistent-worker epoch/window output differs from main process')
            offset += size
        if offset != len(probe_indices):
            raise RuntimeError('Persistent-worker probe sample coverage differs')
    del probe
    _AUGMENT.set_epoch(0)
    numpy_after = np.random.get_state()
    if (random.getstate() != python_before or numpy_before[0] != numpy_after[0]
            or not np.array_equal(numpy_before[1], numpy_after[1]) or numpy_before[2:] != numpy_after[2:]
            or not torch.equal(torch_before, torch.get_rng_state())
            or any(not torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all() if cuda_before else []))):
        raise RuntimeError('Augmentation/probe consumed a global RNG stream')
    return {'checked_visible_windows': len(proofs), 'epoch_checks': [1, 2], 'worker_epoch_checks': [1, 2],
        'probe_samples_per_epoch': len(probe_indices), 'probe_workers': workers,
        'persistent_worker_outputs_match_main': True, 'global_rng_unchanged_by_augmentation_and_probe': True,
        'eligible_bucket_segments': sum(row['eligible_bucket_segments'] for row in proofs),
        'changed_visible_windows': sum(row['changed_visible_items'] for row in proofs),
        'all_visible_bucket_event_multisets_preserved': True, 'target_and_cutoff_fixed': True,
        'target_event_future_excluded': True, 'original_visible_window_and_length_preserved': True,
        'repeated_target_item_counts_preserved': True, 'cross_window_event_priorities_shared': True}


def _save_smoke_report(loss, gradients, all_finite, proof):
    if not math.isfinite(loss) or loss <= 0 or gradients <= 0 or not all_finite:
        raise RuntimeError('Smoke loss/backward gradients failed')
    _verify_originals()
    if OUTPUT_DIR.exists():
        raise RuntimeError('Smoke must not create a formal output directory')
    report = dict(_X3_PROVENANCE)
    report.update(tag=_TAG, dataset=DATASET, seed=SEED, status='SMOKE_PASS', mode='smoke',
        counts={'num_items': NUM_ITEMS, 'training_rows': len(train_df), 'training_positions': len(train_dataset),
                'trainable_users': len(_AUGMENT.metadata), 'validation_samples': len(val_df), 'test_samples': len(test_df)},
        smoke_loss=loss, finite_gradient_count=gradients, all_gradients_finite=bool(all_finite), invariants=proof,
        optimizer_steps=0, formal_training_started=False, weights_read=False, weights_saved=False,
        test_inference_performed=False, formal_output_created=False, original_hashes_verified_after=True)
    path = _X3_STAGE / 'verification' / ('smoke_' + _TAG + '_seed' + str(SEED) + '.json')
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print('X3_SHUFFLE_SMOKE_PASS=' + str(path), flush=True)


def _canonical_test_mask_proof(reference_rows, prediction_rows, item_map):
    """Post-save audit only; preserve the protected ranking implementation."""
    if len(reference_rows) != len(prediction_rows) or not reference_rows:
        raise RuntimeError('Saved test predictions do not cover the canonical test panel')
    seen_samples = set()
    violations = unknown = identity = raw_index_mismatches = 0
    for reference, prediction in zip(reference_rows, prediction_rows):
        sample = str(reference['sample_id'])
        if sample in seen_samples:
            raise RuntimeError('Duplicate canonical test sample identity')
        seen_samples.add(sample)
        identity += int(any(str(reference[key]) != str(prediction[key])
                            for key in ('sample_id', 'user_id', 'target_item')))
        target = str(reference['target_item'])
        unknown += int(target not in item_map)
        history = {str(item) for item in json.loads(reference['sequence_items'])}
        masked = history - {target}
        raw_top = [str(item) for item in json.loads(prediction['top10_raw_item_id'])]
        indices = json.loads(prediction['top10_item_idx'])
        if len(raw_top) != 10 or len(indices) != 10 or len(set(raw_top)) != 10:
            raise RuntimeError('Saved Top10 list is malformed')
        for raw, index in zip(raw_top, indices):
            if isinstance(index, bool) or not isinstance(index, int):
                raise RuntimeError('Saved Top10 index must be an integer')
            raw_index_mismatches += int(raw not in item_map or item_map.get(raw) != index)
            violations += int(raw in masked)
    if violations or unknown or identity or raw_index_mismatches:
        raise RuntimeError('Canonical test mask/identity/map audit failed: ' + json.dumps({
            'seen_item_violations': violations, 'target_unknown_count': unknown,
            'panel_identity_mismatches': identity, 'top10_raw_index_mismatches': raw_index_mismatches}))
    return {'canonical_evaluation_mask_verified': True, 'test_samples_verified': len(reference_rows),
        'test_top10_seen_item_violations': violations, 'test_target_unknown_count': unknown,
        'test_panel_identity_mismatches': identity, 'test_top10_raw_index_mismatches': raw_index_mismatches,
        'test_mask_rule': 'unique canonical history items except target; full frozen Control catalog'}

_X3_ROOT=plain(Path('/root/autodl-tmp/time_uncertainty_rec'))
_X3_STAGE=plain(Path('/root/autodl-tmp/time_uncertainty_rec/kais_x3_shuffle_v1_20261026T014332Z_46d05dee7b66'))
_TAG='beauty'
_INPUTS={'version': 'x3_shuffle_v1', 'training_seeds': [42, 2026, 3407], 'model_source': 'step38_kais_retrain/scripts/38_1_train_ml1m_sasrec_bs128.py', 'sources_sha256': {'step38_kais_retrain/scripts/38_1_train_ml1m_sasrec_bs128.py': '56fdd85061943778708a2de7ec18bbe79bd3873a4d9ffbc12b9e693fadb4b27c', 'step38_kais_retrain/scripts/38_4_train_beauty_sasrec_bs128.py': '29c528543757d18b54b18a8c09d4dd4e573e125cc5128e9a13ae702b4be5d48e', 'kais_orderfix_v1_20261022T081312Z_e335970a03a1/scripts/train_sports_orderfix.py': '2fa42816658432c8e85f42b82d715aec4232c4c49d2b52e579b44da8f6a5f55a', 'kais_orderfix_v1_20261022T081312Z_e335970a03a1/scripts/train_toys_orderfix.py': '490269b75a87b5aabdb9195233a958fce4d67a5a8f95df093faca9dc46d8e4bb'}, 'datasets': {'ml1m': {'dataset': 'MovieLens-1M', 'trainer_source': 'step38_kais_retrain/scripts/38_1_train_ml1m_sasrec_bs128.py', 'train_path': 'processed/diagnostic_a_splits/MovieLens-1M/file_order/train.csv', 'validation_path': 'processed/recent_bucketset_r1/MovieLens-1M/val/r1_samples.csv', 'test_path': 'processed/recent_bucketset_r1/MovieLens-1M/test/r1_samples.csv', 'expected_items': 3416, 'expected_validation': 752, 'expected_test': 945, 'expected_training_positions': 794289, 'expected_training_rows': 799689, 'sequence_sort_keys': ['user_id', 'time_bucket', '_source_order'], 'event_key_column': '_source_order', 'control_reference': {'model_dir': 'step38_kais_retrain/outputs/ml1m_sasrec_control_ce_bs128_lr1em3_max200_seed42', 'files_sha256': {'config.json': '04e73d16f5d6f36521355498c9713d7c95b15119fef37f749a6b8a3b09765ad4', 'metrics.json': '1dc82d20bd5c896c6e976abdb0b00b8fd9369eea3df6a6f3dc8d26e63077b6f8', 'training_log.csv': '26ce50511ba2217a03111c3a7969002771dda2d8b54710919ac439d722ca16c2', 'test_per_sample_predictions.csv': '7ef93f95cb57fe9644bd0b59cfeed0a5a03aeaaec577dcc94b3b8479e12c9268', 'item_id_map.csv': '0dd7d6ee37d97e24be33253f5f0836890a818a255a5314519e1678d6c31885d5'}}}, 'beauty': {'dataset': 'Amazon_Beauty', 'trainer_source': 'step38_kais_retrain/scripts/38_4_train_beauty_sasrec_bs128.py', 'train_path': 'processed/diagnostic_a_splits/Amazon_Beauty/file_order/train.csv', 'validation_path': 'processed/recent_bucketset_r1/Amazon_Beauty/val/r1_samples.csv', 'test_path': 'processed/recent_bucketset_r1/Amazon_Beauty/test/r1_samples.csv', 'expected_items': 12101, 'expected_validation': 2856, 'expected_test': 2966, 'expected_training_positions': 138167, 'expected_training_rows': 159008, 'sequence_sort_keys': ['user_id', 'time_bucket', '_source_order'], 'event_key_column': '_source_order', 'control_reference': {'model_dir': 'step38_kais_retrain/outputs/beauty_sasrec_control_ce_bs128_lr1em3_max200_seed42', 'files_sha256': {'config.json': 'a453f393b09d374f25868d8d2a205953b2d0da064c41aad443568bc693ad8425', 'metrics.json': '5e10b55ec8c550344e66c092ef6511ed524527f00d02d006627b8ec9206f9b2b', 'training_log.csv': '2b22f6ad9d8be45e775c053d65ff2f34b4392c14ebaa571d8e0e353341a9d887', 'test_per_sample_predictions.csv': 'c9f0efd243532633594be0ca5b82f81aa1091facd8ad37ea06ffdb478330a40e', 'item_id_map.csv': '259b8725bf180ff02c3f9a7351c43fdf1de22aa51829c40437c2040f95737289'}}}, 'sports': {'dataset': 'Amazon_Sports', 'trainer_source': 'kais_orderfix_v1_20261022T081312Z_e335970a03a1/scripts/train_sports_orderfix.py', 'train_path': 'processed/diagnostic_a_splits/Amazon_Sports/file_order/train.csv', 'validation_path': 'processed/strict_diagnostic_a/Amazon_Sports/file_order/val.csv', 'test_path': 'processed/strict_diagnostic_a/Amazon_Sports/file_order/test.csv', 'expected_items': 18357, 'expected_validation': 4082, 'expected_test': 4659, 'expected_training_positions': 204047, 'expected_training_rows': 237423, 'sequence_sort_keys': ['user_id', 'time_bucket', 'raw_line'], 'event_key_column': 'raw_line', 'control_reference': {'model_dir': 'kais_orderfix_v1_20261022T081312Z_e335970a03a1/outputs/amazon_sports_control_orderfix_seed42', 'files_sha256': {'config.json': '95963620e83b1e06dcda21f947394b2adad4b0c7259b3f7b4f99a759d9493989', 'metrics.json': 'ba3afef8d2451b67d6b4cd3d7cfde2479129288cdfb4bf305b545d3780e67d5b', 'training_log.csv': '798891b74b0df856d8fc5122570a514a3c3819bfa62d3e1f6fb6c676b6284581', 'test_per_sample_predictions.csv': '9ef7554252ff843a1b5bf7e3809aba4827c96427aee66744f06de70852119e5e', 'item_id_map.csv': '8f0e80ba0e996b19409cd72b3ed838242397f548cf38604ffd73d58677e29fee'}}}, 'toys': {'dataset': 'Amazon_Toys', 'trainer_source': 'kais_orderfix_v1_20261022T081312Z_e335970a03a1/scripts/train_toys_orderfix.py', 'train_path': 'processed/diagnostic_a_splits/Amazon_Toys/file_order/train.csv', 'validation_path': 'processed/strict_diagnostic_a/Amazon_Toys/file_order/val.csv', 'test_path': 'processed/strict_diagnostic_a/Amazon_Toys/file_order/test.csv', 'expected_items': 11924, 'expected_validation': 1939, 'expected_test': 2547, 'expected_training_positions': 116686, 'expected_training_rows': 134632, 'sequence_sort_keys': ['user_id', 'time_bucket', 'raw_line'], 'event_key_column': 'raw_line', 'control_reference': {'model_dir': 'kais_orderfix_v1_20261022T081312Z_e335970a03a1/outputs/amazon_toys_control_orderfix_seed42', 'files_sha256': {'config.json': '18d9418f4eb6870929258deb4186a5ee160210d1bf57e55263b076d44a48b64e', 'metrics.json': '2b7ce49edd8c7825f0b1d4f63db4895210e6b7339e640553c4b1924a0ab9b307', 'training_log.csv': '057de30a3015899d2d1ebcfcabfc0731e2f05c864fd6dddfbc9ed97402c1f1ac', 'test_per_sample_predictions.csv': '9b4b5854ff4c05a6b563236e2a6b436d83931f640bf435725294ee2e0fb89cd9', 'item_id_map.csv': 'd6e1135f71ad9749caa6db9743045e1e65860bd62df6f42e0b76642c08645c60'}}}}, 'data_sha256': {'processed/diagnostic_a_splits/MovieLens-1M/file_order/train.csv': '6b0fa970c4b6725a32e718b43883c612624b16127122bb4d9a936eff1dade18e', 'processed/recent_bucketset_r1/MovieLens-1M/val/r1_samples.csv': '4c66e7a33fe1362089cf8aa4bf7144a301803eabed0c6097faa984419346fb32', 'processed/recent_bucketset_r1/MovieLens-1M/test/r1_samples.csv': '99bec555642aca4cd00cdfe3b5d95c7b3ac577c1c1e0457a7c6212c08b9ea303', 'processed/diagnostic_a_splits/Amazon_Beauty/file_order/train.csv': 'c81fd1c1dbcbbe88ed4596e0d952397d1ff74ba70a5ba9d8f9e4038e72e05971', 'processed/recent_bucketset_r1/Amazon_Beauty/val/r1_samples.csv': '0f19f717b9c8a3230c8ce52735960fa48f4fb5e2c08c1787e0a4c52dbfbb5eb9', 'processed/recent_bucketset_r1/Amazon_Beauty/test/r1_samples.csv': 'f9362385825ed79b07e4d262cf9ac7df2591c54cd7d8e95e54ba116bdaad5009', 'processed/diagnostic_a_splits/Amazon_Sports/file_order/train.csv': '8b78398cf350881125d04a15cd0b6181b17d1772d35f428ed61d3e9220a16539', 'processed/strict_diagnostic_a/Amazon_Sports/file_order/val.csv': '8ffe18f11957ba4229aa42d4e1b5d2ffdfbf3ef2adb4ecd0ff3b349469cbe7fb', 'processed/strict_diagnostic_a/Amazon_Sports/file_order/test.csv': '5f2f31713ee7f6c92bb6fde915500d51f1a49328f3f1fc94f146530ad54e0e89', 'processed/diagnostic_a_splits/Amazon_Toys/file_order/train.csv': '26e4e892b1e520fdc213bddad4af3f314e865f53a1c8cbcf1a4e0f60a83ff8c0', 'processed/strict_diagnostic_a/Amazon_Toys/file_order/val.csv': '5b2187611068dbf7f384e55489d3704710387ef42bf86fcf028836ae26e8f14d', 'processed/strict_diagnostic_a/Amazon_Toys/file_order/test.csv': '3e287ca6be6c556d5abfd691666abdc4fde1eca10327d6b8329a8aa55d0df445'}, 'augmentation_protocol': {'target_and_cutoff': 'fixed original reference event and cutoff for each legal next-item training position', 'history_window': 'take the original visible prefix and existing max50 truncation before augmentation', 'scope': 'permute every visible timestamp bucket separately in that retained history; preserve bucket order, per-bucket event multiset and original target identity', 'priority': 'independent deterministic event/epoch priorities shared across overlapping histories; no use of Torch/dropout or original sampler RNG', 'repeated_item_ids': 'retain legitimate repeated items; exclude the target event and future events, not all occurrences of the target raw item ID', 'inference': 'canonical reference order for validation and test'}, 'hyperparameters': {'max_len': 50, 'hidden_size': 64, 'num_heads': 2, 'num_blocks': 2, 'dropout': 0.2, 'training_batch_size': 128, 'evaluation_batch_size': 256, 'learning_rate': 0.001, 'max_epochs': 200, 'patience': 15}, 'scope': 'Twelve new matched Control-architecture models with per-epoch every-visible-bucket training augmentation. Existing24models/splits/inputs remain read only. Canonical validation selects best checkpoint. No test-based seed/epoch selection.'}
_INPUTS_SHA256='3788d6ce525966c74663a7e2e61b4bc20c9eaac5d2e6c7a5a8e33fc46f4dfce3'
_MODEL_SHA256='ac2e47de1cd8cba980be4c049f521a36a08a7d54f3610be6bc687cef7383555e'
_PLAN=_INPUTS['datasets'][_TAG]
_MANIFEST_PATH='manifests/x3_shuffle_manifest.json'
_MANIFEST_BYTES=checked_file(_X3_STAGE,_MANIFEST_PATH).read_bytes()
_MANIFEST_SHA256=hashlib.sha256(_MANIFEST_BYTES).hexdigest()
_MANIFEST=json.loads(_MANIFEST_BYTES.decode('utf-8-sig'))
if (_MANIFEST['inputs']!=_INPUTS or _MANIFEST['inputs_sha256']!=_INPUTS_SHA256
        or _MANIFEST['root']!=str(_X3_ROOT) or _MANIFEST['stage']!=str(_X3_STAGE)):
    raise RuntimeError('Prepared X3 identity differs')
_TRAINER_RELATIVE='scripts/train_'+_TAG+'_shuffle.py'
_TRAINER_SHA256=_MANIFEST['generated_sha256'][_TRAINER_RELATIVE]
_DATA_SHA256={path:_INPUTS['data_sha256'][path] for path in
    (_PLAN['train_path'],_PLAN['validation_path'],_PLAN['test_path'])}
_REFERENCE_SHA256=_PLAN['control_reference']['files_sha256']
_MAP_PATH=checked_file(_X3_ROOT,_PLAN['control_reference']['model_dir']+'/item_id_map.csv')
def _verify_originals():
    for relative,expected in dict(_INPUTS['sources_sha256'],**_DATA_SHA256).items():
        checked_sha256(_X3_ROOT,relative,expected)
    for name,expected in _REFERENCE_SHA256.items():
        checked_sha256(_X3_ROOT,_PLAN['control_reference']['model_dir']+'/'+name,expected)
    checked_sha256(_X3_STAGE,'scripts/model_definition.py',_MODEL_SHA256)
    checked_sha256(_X3_STAGE,_TRAINER_RELATIVE,_TRAINER_SHA256)
    checked_sha256(_X3_STAGE,_MANIFEST_PATH,_MANIFEST_SHA256)
_verify_originals()
if mp.get_context().get_start_method()!='fork':
    raise RuntimeError('These protected top-level Control trainers require the original Linux fork worker context')
_X3_PROVENANCE={'version':'x3_shuffle_v1','input_manifest_sha256':_INPUTS_SHA256,
    'preparation_manifest_sha256':_MANIFEST_SHA256,'trainer_sha256':_TRAINER_SHA256,
    'frozen_model_sha256':_MODEL_SHA256,'sources_sha256':_INPUTS['sources_sha256'],
    'data_sha256':_DATA_SHA256,'control_reference':_PLAN['control_reference'],
    'augmentation':{'visible_history_first':True,'target_and_cutoff_fixed':True,
        'all_visible_time_bucket_segments':True,'shared_event_epoch_priorities':True,
        'global_rng_consumed':False,'event_key_column':_PLAN['event_key_column'],
        'priority_algorithm':'BLAKE2b-128(json([seed,dataset,user,eventkey,epoch]))',
        'worker_epoch_shared':True,'epoch_setup_included_in_epoch_seconds':True,
        'legitimate_repeated_target_item_ids_retained':True},
    'original_sampler_loader_flow_preserved':True,'canonical_validation_test':True}
print('X3 INTEGRITY PASS: frozen sources/data/Control metadata/model; no old checkpoint accessed',flush=True)

from pathlib import Path
import os
import json
import math
import random
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ============================================================
# Step38.4 - Amazon_Beauty SASRec Control
# Unified KAIS retraining protocol:
# BS128, Adam, LR=0.001, MaxEpochs=200, Patience=15
# All legal next-item positions, reshuffled every epoch
# Masked Forward; no LBSP pooling
# ============================================================


# ============================================================
# 0. 固定配置
# ============================================================

ROOT = _X3_ROOT

DATASET = "Amazon_Beauty"
ORDER = "file_order"
SEED = args.seed

MAX_LEN = 50
HIDDEN_SIZE = 64
NUM_HEADS = 2
NUM_BLOCKS = 2
DROPOUT = 0.2

BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 200
PATIENCE = 15
TRAINING_POSITIONS = "all_legal_next_item_positions"


# ============================================================
# 1. 路径
# ============================================================

TRAIN_PATH = ROOT / _PLAN["train_path"]

ITEM_SOURCE_PATH = _MAP_PATH

VAL_PATH = ROOT / _PLAN["validation_path"]

TEST_PATH = ROOT / _PLAN["test_path"]

OUTPUT_DIR = _X3_STAGE / "outputs" / f"beauty_shuffle_seed{SEED}"
if OUTPUT_DIR.exists():
    raise RuntimeError("X3 formal output already exists; refusing to overwrite")





# ============================================================
# 2. 随机种子 / 设备
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 严格确定性
    torch.use_deterministic_algorithms(True)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


set_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

print("=" * 80)
print("Step38.4: Amazon_Beauty SASRec Control")
print("=" * 80)
print("Dataset:", DATASET)
print("Order:", ORDER)
print("Seed:", SEED)
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# ============================================================
# 3. 输入检查
# ============================================================

for path in [
    TRAIN_PATH,
    ITEM_SOURCE_PATH,
    VAL_PATH,
    TEST_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)


# ============================================================
# 4. 统一 item mapping
# ============================================================

mapping_df = pd.read_csv(_MAP_PATH, dtype={"raw_item_id": str})
if not {"raw_item_id", "item_idx"}.issubset(mapping_df.columns):
    raise RuntimeError("Frozen Control mapping columns missing")
if (not mapping_df["raw_item_id"].is_unique or not mapping_df["item_idx"].is_unique
        or mapping_df["item_idx"].astype(int).tolist() != list(range(1, len(mapping_df) + 1))):
    raise RuntimeError("Frozen Control item mapping is not an ordered bijection")
raw_items = mapping_df["raw_item_id"].tolist()
item_map = dict(zip(raw_items, mapping_df["item_idx"].astype(int)))
idx_to_raw_item = {int(index): str(raw) for raw, index in item_map.items()}
idx_to_raw = idx_to_raw_item
NUM_ITEMS = len(item_map)
if NUM_ITEMS != _PLAN["expected_items"]:
    raise RuntimeError("Frozen catalog count differs")

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype={
        "user_id": str,
        "item_id": str,
    },
)

train_df["user_id"] = train_df["user_id"].astype(str)
train_df["item_id"] = train_df["item_id"].astype(str)
train_df["_source_order"] = np.arange(
    len(train_df),
    dtype=np.int64,
)

if "time_bucket" in train_df.columns:
    BUCKET_COLUMN = "time_bucket"
    train_df = train_df.sort_values(
        ["user_id", "time_bucket", "_source_order"],
        kind="mergesort",
    )
elif "timestamp" in train_df.columns:
    # 当 time_bucket 不存在时使用 timestamp 作为兜底。
    BUCKET_COLUMN = "timestamp"
    train_df = train_df.sort_values(
        ["user_id", "timestamp", "_source_order"],
        kind="mergesort",
    )
else:
    raise RuntimeError(
        "训练数据既没有 time_bucket 也没有 timestamp，无法确定最后历史桶。"
    )


# ============================================================
# 6. 用户训练序列：items + buckets
# ============================================================

user_train_sequences = {}

for user_id, group in train_df.groupby("user_id", sort=False):
    mapped_items = []

    for raw_item in group["item_id"].astype(str).tolist():
        if raw_item not in item_map:
            raise RuntimeError(
                f"Train item 不在 mapping 中: {raw_item}"
            )
        mapped_items.append(item_map[raw_item])

    buckets = group[BUCKET_COLUMN].tolist()

    if len(mapped_items) != len(buckets):
        raise RuntimeError("训练 items / buckets 长度不一致")

    if len(mapped_items) >= 2:
        user_train_sequences[str(user_id)] = {
            "items": np.asarray(mapped_items, dtype=np.int64),
            "buckets": np.asarray(buckets),
        }

print("Train records:", len(train_df))
print("Train users:", len(user_train_sequences))
print("Training bucket column:", BUCKET_COLUMN)


# ============================================================
# 7. R=1 分组函数
#
# 对 history：
#   - 最后桶大小 >=2：最后桶全部事件共享一个 token；
#   - 其余事件一个事件一个 token。
# ============================================================

def build_r1_group_positions_from_buckets(
    history_buckets,
    max_len=MAX_LEN,
):
    n = len(history_buckets)

    group_positions = np.full(
        max_len,
        -1,
        dtype=np.int64,
    )

    if n == 0:
        return group_positions, 0, 0, 0

    last_bucket = history_buckets[-1]

    start = n - 1
    while (
        start > 0
        and history_buckets[start - 1] == last_bucket
    ):
        start -= 1

    last_bucket_size = n - start
    direct_pooling = int(last_bucket_size >= 2)

    event_groups = np.empty(n, dtype=np.int64)

    if direct_pooling:
        # prefix 每个事件独立；最后桶共享一个 group。
        for j in range(start):
            event_groups[j] = j
        event_groups[start:n] = start
        num_tokens = start + 1
    else:
        event_groups[:] = np.arange(n, dtype=np.int64)
        num_tokens = n

    group_positions[-n:] = event_groups

    return (
        group_positions,
        int(num_tokens),
        int(direct_pooling),
        int(last_bucket_size),
    )


def build_r1_group_positions_from_size(
    history_len,
    last_bucket_size,
    max_len=MAX_LEN,
):
    if history_len <= 0:
        raise ValueError("history_len must be > 0")

    if last_bucket_size < 1 or last_bucket_size > history_len:
        raise ValueError(
            f"invalid last_bucket_size={last_bucket_size}, "
            f"history_len={history_len}"
        )

    group_positions = np.full(
        max_len,
        -1,
        dtype=np.int64,
    )

    direct_pooling = int(last_bucket_size >= 2)

    if direct_pooling:
        prefix_len = history_len - last_bucket_size
        event_groups = np.empty(history_len, dtype=np.int64)

        for j in range(prefix_len):
            event_groups[j] = j

        event_groups[prefix_len:] = prefix_len
        num_tokens = prefix_len + 1
    else:
        event_groups = np.arange(history_len, dtype=np.int64)
        num_tokens = history_len

    group_positions[-history_len:] = event_groups

    return (
        group_positions,
        int(num_tokens),
        int(direct_pooling),
    )

# ============================================================
# Matched Control
#
# 不进行任何 pooling。
# 每一个历史事件独占一个 token。
# ============================================================

def build_control_group_positions(
    history_len,
    max_len,
):

    if history_len <= 0:
        raise ValueError(
            f"invalid history_len={history_len}"
        )

    if history_len > max_len:
        raise ValueError(
            f"history_len={history_len} > max_len={max_len}"
        )

    group_positions = np.full(
        max_len,
        -1,
        dtype=np.int64,
    )

    group_positions[
        -history_len:
    ] = np.arange(
        history_len,
        dtype=np.int64,
    )

    num_tokens = history_len

    return (
        group_positions,
        int(num_tokens),
    )
# ============================================================
# R=1 deterministic canonicalization
#
# 最后历史桶在 mean pooling 前按 item_idx 固定排序。
# 不增加任何信息，只固定浮点求和顺序。
# ============================================================

def canonicalize_last_bucket_items(
    history_items,
    last_bucket_size,
):

    items = np.asarray(
        history_items,
        dtype=np.int64,
    ).copy()

    if len(items) == 0:
        raise ValueError(
            "canonicalize出现空history"
        )

    if (
        last_bucket_size < 1
        or
        last_bucket_size > len(items)
    ):
        raise ValueError(
            f"invalid last_bucket_size="
            f"{last_bucket_size}, "
            f"history_len={len(items)}"
        )

    if last_bucket_size >= 2:

        start = (
            len(items)
            -
            last_bucket_size
        )

        items[
            start:
        ] = np.sort(
            items[
                start:
            ],
            kind="mergesort",
        )

    return items
# ============================================================
# 8. 训练 Dataset
#
# cutoff 抽样沿用 Step 28.2：
# 全部合法 next-item 训练位置；每轮重新随机打乱。
#
# 与旧 R=1 原型一致，本脚本将每个 cutoff 的最后一个 item
# 作为 target，并对 target 之前的 history 做 R=1 pooling。
# ============================================================

class ControlTrainDataset(Dataset):
    def __init__(self, user_sequences, max_len):
        self.user_sequences = user_sequences
        self.max_len = max_len
        self.users = sorted(user_sequences.keys())
        self.samples = []

        # 每个合法位置恰好生成一个 next-item 样本
        for user_id in self.users:
            seq = self.user_sequences[user_id]["items"]

            if len(seq) < 2:
                continue

            self.samples.extend(
                (user_id, end_idx)
                for end_idx in range(2, len(seq) + 1)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_id, end_idx = self.samples[idx]

        info = self.user_sequences[user_id]
        items = info["items"]
        buckets = info["buckets"]

        target_idx = end_idx - 1
        history_start = max(
            0,
            target_idx - self.max_len,
        )

        history_items = items[
            history_start:target_idx
        ]
        history_buckets = buckets[
            history_start:target_idx
        ]
        history_items = np.asarray(_AUGMENT.visible_items(user_id, history_start, target_idx, history_items), dtype=np.int64)

        if len(history_items) == 0:
            raise RuntimeError("训练样本出现空 history")

        # Step38 Control CPU optimization:
        # 仅计算最后时间桶大小，不构造无用的 R1 分组矩阵。
        history_len = len(history_items)

        last_bucket = history_buckets[-1]
        bucket_start = history_len - 1

        while (
            bucket_start > 0
            and history_buckets[bucket_start - 1] == last_bucket
        ):
            bucket_start -= 1

        last_bucket_size = history_len - bucket_start

        # ----------------------------------------------------
        # Matched Control：
        # 每个历史事件独占一个 token，不做 pooling。
        # ----------------------------------------------------
        (
            group_positions,
            num_tokens,
        ) = build_control_group_positions(
            history_len,
            self.max_len,
        )

        # Control 实际没有 pooling
        direct_pooling = 0

        # ----------------------------------------------------
        # 非常重要：
        # Control 必须保留原始 file_order，
        # 这里绝对不能 canonicalize。
        # ----------------------------------------------------
        seq = np.zeros(
            self.max_len,
            dtype=np.int64,
        )

        seq[
            -history_len:
        ] = np.asarray(
            history_items,
            dtype=np.int64,
        )

        target = int(items[target_idx])

        return (
            torch.from_numpy(seq),
            torch.from_numpy(group_positions),
            torch.tensor(num_tokens, dtype=torch.long),
            torch.tensor(direct_pooling, dtype=torch.long),
            torch.tensor(last_bucket_size, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


_AUGMENT_METADATA = {}
_AUGMENT_USERS = []
for _user, _group in train_df.groupby("user_id", sort=False):
    if len(_group) >= 2:
        _user = str(_user)
        _AUGMENT_USERS.append(_user)
        _AUGMENT_METADATA[_user] = {
            "items": tuple(item_map[str(item)] for item in _group["item_id"]),
            "buckets": tuple(_group[BUCKET_COLUMN]),
            "event_keys": tuple(int(key) for key in _group[_PLAN["event_key_column"]]),
        }
if len(train_df) != _PLAN["expected_training_rows"]:
    raise RuntimeError("Original training row count differs")
_AUGMENT = SharedPriorityState(DATASET, SEED, _AUGMENT_METADATA)

train_dataset = ControlTrainDataset(
    user_sequences=user_train_sequences,
    max_len=MAX_LEN,
)
if len(train_dataset) != _PLAN["expected_training_positions"]:
    raise RuntimeError("Original legal-position coverage differs from the sealed Control protocol")
if _TAG in ('sports', 'toys'):
    if len(user_sequences) != len(_AUGMENT_USERS) or any(tuple(seq) != _AUGMENT.metadata[user]["items"] for seq,user in zip(user_sequences,_AUGMENT_USERS)):
        raise RuntimeError("Parallel Sports/Toys user/event metadata lost original sequence alignment")


expected_positions = sum(
    len(info["items"]) - 1
    for info in user_train_sequences.values()
)

if len(train_dataset) != expected_positions:
    raise RuntimeError("训练位置覆盖数量不匹配")

loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=2,
    generator=loader_generator,
)

print("Train samples:", len(train_dataset))
print("Train batches:", len(train_loader))


# ============================================================
# 9. 读取冻结的 strict val/test
# ============================================================

def load_r1_eval(path):
    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_items": str,
        },
    )

    sequences = []
    group_positions_all = []
    num_tokens_all = []
    targets = []
    direct_pooling_all = []
    last_bucket_sizes = []

    for row in df.itertuples(index=False):
        raw_history = json.loads(row.sequence_items)

        mapped_history = []
        for raw_item in raw_history:
            raw_item = str(raw_item)
            if raw_item not in item_map:
                raise RuntimeError(
                    f"Eval history item 不在 mapping 中: {raw_item}"
                )
            mapped_history.append(item_map[raw_item])

        if len(mapped_history) == 0:
            raise RuntimeError(
                f"Eval 出现空 history: {row.sample_id}"
            )

        if len(mapped_history) > MAX_LEN:
            raise RuntimeError(
                f"Eval history > MAX_LEN: {row.sample_id}"
            )

        target_raw = str(row.target_item)
        if target_raw not in item_map:
            raise RuntimeError(
                f"Eval target 不在 mapping 中: {target_raw}"
            )

        target_idx = item_map[target_raw]

        if target_idx in mapped_history:
            raise RuntimeError(
                f"Target 泄漏进 history: {row.sample_id}"
            )

        history_len = len(
            mapped_history
        )

        last_bucket_size = int(
            row.last_bucket_size
        )

        # ----------------------------------------------------
        # Matched Control：
        # 不做任何 pooling。
        # 每个历史事件独占一个 token。
        #
        # direct_pooling 这里只保留为“R=1会不会处理该样本”
        # 的结构标签，用于后续 Direct-Pooling / Unpooled 分组。
        # ----------------------------------------------------

        direct_pooling = int(
            row.direct_pooling
        )

        (
            group_positions,
            num_tokens,
        ) = build_control_group_positions(
            history_len,
            MAX_LEN,
        )

        # ----------------------------------------------------
        # Control 必须保留原始 file_order。
        # 这里绝对不能 canonicalize。
        # ----------------------------------------------------

        seq = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        seq[
            -history_len:
        ] = np.asarray(
            mapped_history,
            dtype=np.int64,
        )
        seq = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        seq[-history_len:] = np.asarray(
            mapped_history,
            dtype=np.int64,
        )

        seq = np.zeros(MAX_LEN, dtype=np.int64)
        seq[-history_len:] = np.asarray(
            mapped_history,
            dtype=np.int64,
        )
        expected_direct = int(row.direct_pooling)
        if direct_pooling != expected_direct:
            raise RuntimeError(
                f"direct_pooling mismatch: {row.sample_id}"
            )

        seq = np.zeros(MAX_LEN, dtype=np.int64)
        seq[-history_len:] = np.asarray(
            mapped_history,
            dtype=np.int64,
        )

        sequences.append(seq)
        group_positions_all.append(group_positions)
        num_tokens_all.append(num_tokens)
        targets.append(target_idx)
        direct_pooling_all.append(direct_pooling)
        last_bucket_sizes.append(last_bucket_size)

    return (
        df,
        np.stack(sequences),
        np.stack(group_positions_all),
        np.asarray(num_tokens_all, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(direct_pooling_all, dtype=np.int64),
        np.asarray(last_bucket_sizes, dtype=np.int64),
    )


(
    val_df,
    val_sequences,
    val_groups,
    val_num_tokens,
    val_targets,
    val_direct_pooling,
    val_last_bucket_sizes,
) = load_r1_eval(VAL_PATH)

(
    test_df,
    test_sequences,
    test_groups,
    test_num_tokens,
    test_targets,
    test_direct_pooling,
    test_last_bucket_sizes,
) = load_r1_eval(TEST_PATH)

print("Strict Val samples:", len(val_targets))
print("Strict Test samples:", len(test_targets))
print(
    "Val direct-pooling ratio:",
    f"{val_direct_pooling.mean():.6f}",
)
print(
    "Test direct-pooling ratio:",
    f"{test_direct_pooling.mean():.6f}",
)


# ============================================================
# 10. Recent Bucket-Set R=1 SASRec
#
# 核心来自旧 Step 15：
# 多个事件共享 group -> embedding scatter_add -> mean pooling。
#
# 与旧实现不同的关键修正：
# Transformer 实际运行时不使用 padding；
# 每个 compact token 序列使用右对齐绝对位置：
#     MAX_LEN - num_tokens ... MAX_LEN - 1
# ============================================================

_model_tree = ast.parse((_X3_STAGE / "scripts/model_definition.py").read_text(encoding="utf-8"))
_model_nodes = [node for node in _model_tree.body if isinstance(node, ast.ClassDef) and node.name == "SASRecControl"]
if len(_model_nodes) != 1:
    raise RuntimeError("Frozen SASRecControl class missing")
_model_namespace = dict(globals())
exec(compile(ast.Module(body=_model_nodes, type_ignores=[]), "model_definition.py", "exec"), _model_namespace)
SASRecControl = _model_namespace["SASRecControl"]


if len(val_df) != _PLAN["expected_validation"] or len(test_df) != _PLAN["expected_test"]:
    raise RuntimeError("Frozen canonical validation/test panel count differs")
model = SASRecControl(
    num_items=NUM_ITEMS,
    max_len=MAX_LEN,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    dropout=DROPOUT,
).to(device)



criterion = nn.CrossEntropyLoss(
    reduction="sum"
)

print(
    "Model parameters:",
    sum(p.numel() for p in model.parameters()),
)


# ============================================================
# 11. 训练：按 num_tokens 分组，避免 padding
# ============================================================

def train_one_batch(
    model,
    seq,
    groups,
    num_tokens,
    target,
):
    """
    Step38 Control masked training.

    BS保持128，训练目标仍为全物品交叉熵。
    不再按历史长度拆分Transformer计算。

    groups保留在函数接口中，兼容原训练循环。
    该快速路径仅适用于Control。
    """

    if (
        seq.size(0) != target.size(0)
        or num_tokens.size(0) != seq.size(0)
    ):
        raise RuntimeError("Masked batch维度不匹配")

    scores = model.predict_batch_masked(
        seq,
        num_tokens,
    )

    if not torch.isfinite(scores).all():
        raise RuntimeError(
            "Masked training scores出现NaN/Inf"
        )

    scores = scores.clone()
    scores[:, 0] = -1e9

    loss = criterion(scores, target) / target.size(0)

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Masked training loss出现NaN/Inf"
        )

    return loss

# ============================================================
# 12. full-ranking 评价
# ============================================================

def calculate_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)

    if len(ranks) == 0:
        return {
            "N": 0,
            "NDCG@10": float("nan"),
            "HR@10": float("nan"),
            "HR@20": float("nan"),
            "MeanRank": float("nan"),
            "MedianRank": float("nan"),
            "Rank1Ratio": float("nan"),
        }

    ndcg_values = np.where(
        ranks <= 10,
        1.0 / np.log2(ranks + 1),
        0.0,
    )

    return {
        "N": int(len(ranks)),
        "NDCG@10": float(ndcg_values.mean()),
        "HR@10": float((ranks <= 10).mean()),
        "HR@20": float((ranks <= 20).mean()),
        "MeanRank": float(ranks.mean()),
        "MedianRank": float(np.median(ranks)),
        "Rank1Ratio": float((ranks == 1).mean()),
    }


@torch.no_grad()
def evaluate(
    model,
    sequences,
    groups,
    num_tokens,
    targets,
    direct_pooling,
    sample_df=None,
    save_predictions=False,
):
    model.eval()

    total = len(targets)
    all_ranks = np.empty(total, dtype=np.int64)
    all_top10 = np.empty((total, 10), dtype=np.int64)

    for start in range(0, total, EVAL_BATCH_SIZE):
        end = min(start + EVAL_BATCH_SIZE, total)

        seq_np = sequences[start:end]
        group_np = groups[start:end]
        token_np = num_tokens[start:end]
        target_np = targets[start:end]

        # Step38 Control masked evaluation
        # 保持原始 strict val/test 样本及后续 seen-item mask。
        seq_tensor = torch.from_numpy(
            seq_np
        ).long().to(device)

        token_tensor = torch.as_tensor(
            token_np,
            dtype=torch.long,
            device=device,
        )

        batch_scores = model.predict_batch_masked(
            seq_tensor,
            token_tensor,
        )

        if not torch.isfinite(batch_scores).all():
            raise RuntimeError(
                "Masked evaluation scores出现NaN/Inf"
            )

        batch_scores[:, 0] = -1e9

        # seen-item mask 使用原始事件历史，而不是压缩后的 token。
        for b in range(len(seq_np)):
            seen = np.unique(seq_np[b])
            seen = seen[
                (seen != 0)
                & (seen != target_np[b])
            ]

            if len(seen) > 0:
                seen_tensor = torch.as_tensor(
                    seen,
                    device=device,
                    dtype=torch.long,
                )
                batch_scores[b, seen_tensor] = -1e9

        target_tensor = torch.from_numpy(
            target_np
        ).long().to(device)

        row_index = torch.arange(
            len(target_tensor),
            device=device,
        )

        target_scores = batch_scores[
            row_index,
            target_tensor,
        ]

        if not torch.isfinite(target_scores).all():
            raise RuntimeError(
                "Target score 出现 NaN/Inf"
            )

        ranks = (
            (batch_scores > target_scores.unsqueeze(1))
            .sum(dim=1)
            + 1
        )

        top10 = torch.topk(
            batch_scores,
            k=10,
            dim=1,
        ).indices

        all_ranks[start:end] = (
            ranks.detach().cpu().numpy().astype(np.int64)
        )

        all_top10[start:end] = (
            top10.detach().cpu().numpy().astype(np.int64)
        )

    overall = calculate_metrics(all_ranks)

    direct_mask = direct_pooling.astype(bool)
    unpooled_mask = ~direct_mask

    subgroup = {
        "direct_pooling": calculate_metrics(
            all_ranks[direct_mask]
        ),
        "unpooled": calculate_metrics(
            all_ranks[unpooled_mask]
        ),
        "pooling_ratio": float(direct_mask.mean()),
    }

    if save_predictions and sample_df is not None:
        pred_df = sample_df[
            [
                "sample_id",
                "user_id",
                "target_item",
                "ambiguous_last_bucket",
                "direct_pooling",
                "last_bucket_size",
            ]
        ].copy()

        ndcg_values = np.where(
            all_ranks <= 10,
            1.0 / np.log2(all_ranks + 1),
            0.0,
        )

        pred_df["target_rank"] = all_ranks
        pred_df["ndcg10"] = ndcg_values
        pred_df["hit10"] = (all_ranks <= 10).astype(int)
        pred_df["hit20"] = (all_ranks <= 20).astype(int)
        pred_df["num_tokens"] = num_tokens
        pred_df["top10_item_idx"] = [
            json.dumps(row.tolist())
            for row in all_top10
        ]
        pred_df["top10_raw_item_id"] = [
            json.dumps(
                [idx_to_raw_item[int(x)] for x in row],
                ensure_ascii=False,
            )
            for row in all_top10
        ]

        return overall, subgroup, pred_df

    return overall, subgroup

if args.mode == "smoke":
    _SMOKE_PROOF = _augmentation_smoke_proof()
    _batch = torch.utils.data.default_collate([train_dataset[i] for i in range(min(8, len(train_dataset)))])
    _seq, _groups, _tokens, _pool, _last, _targets = [value.to(device) for value in _batch]
    model.train(); model.zero_grad(set_to_none=True)
    _loss = train_one_batch(model, _seq, _groups, _tokens, _targets)
    if not torch.isfinite(_loss).item() or not _loss.requires_grad:
        raise RuntimeError("Smoke loss is nonfinite or has no gradient")
    _loss.backward()
    _all_finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    _nonzero = sum(p.grad is not None and torch.isfinite(p.grad).all().item() and p.grad.abs().sum().item() > 0 for p in model.parameters())
    model.zero_grad(set_to_none=True)
    _save_smoke_report(float(_loss.item()), _nonzero, _all_finite, _SMOKE_PROOF)
    raise SystemExit(0)

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
(OUTPUT_DIR / "item_id_map.csv").write_bytes(_MAP_PATH.read_bytes())
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)




# ============================================================
# 13. 训练 + strict validation early stopping
# ============================================================

best_ndcg = -1.0
bad_epochs = 0
training_log = []

checkpoint_path = (
    OUTPUT_DIR
    / "matched_control_sasrec_best.pt"
)

print("\n" + "=" * 80)
print("开始训练 Step38.4 SASRec Control")
print("=" * 80)

training_begin = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_begin = time.time()
    _AUGMENT.set_epoch(epoch)

    total_loss = 0.0
    valid_batches = 0
    pooled_samples = 0
    total_samples = 0

    for batch_idx, (
        seq,
        groups,
        num_tokens,
        direct_pooling,
        last_bucket_size,
        target,
    ) in enumerate(train_loader, start=1):

        seq = seq.long().to(device)
        groups = groups.long().to(device)
        num_tokens = num_tokens.long().to(device)
        direct_pooling = direct_pooling.long().to(device)
        target = target.long().to(device)

        optimizer.zero_grad()

        loss = train_one_batch(
            model,
            seq,
            groups,
            num_tokens,
            target,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )
        optimizer.step()

        total_loss += float(loss.item())
        valid_batches += 1
        pooled_samples += int(
            direct_pooling.sum().item()
        )
        total_samples += int(len(seq))

        if batch_idx % 200 == 0:
            print(
                f"Epoch {epoch:02d} "
                f"| Batch {batch_idx:04d}/{len(train_loader)} "
                f"| Loss {loss.item():.4f}"
            )

    avg_loss = total_loss / max(valid_batches, 1)
    train_pooling_ratio = (
        pooled_samples / total_samples
        if total_samples > 0
        else 0.0
    )

    val_metrics, val_subgroup = evaluate(
        model,
        val_sequences,
        val_groups,
        val_num_tokens,
        val_targets,
        val_direct_pooling,
    )

    elapsed = time.time() - epoch_begin

    print(
        f"\nEpoch {epoch:02d}"
        f" | Loss {avg_loss:.6f}"
        f" | Val NDCG@10 {val_metrics['NDCG@10']:.6f}"
        f" | HR@10 {val_metrics['HR@10']:.6f}"
        f" | HR@20 {val_metrics['HR@20']:.6f}"
        f" | MeanRank {val_metrics['MeanRank']:.2f}"
        f" | TrainPool {train_pooling_ratio:.4f}"
        f" | ValPool {val_subgroup['pooling_ratio']:.4f}"
        f" | {elapsed:.1f}s"
    )

    training_log.append(
        {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_pooling_ratio": train_pooling_ratio,
            "val_ndcg10": val_metrics["NDCG@10"],
            "val_hr10": val_metrics["HR@10"],
            "val_hr20": val_metrics["HR@20"],
            "val_mean_rank": val_metrics["MeanRank"],
            "val_pooling_ratio": val_subgroup["pooling_ratio"],
            "elapsed_seconds": elapsed,
        }
    )

    if val_metrics["NDCG@10"] > best_ndcg:
        best_ndcg = val_metrics["NDCG@10"]
        bad_epochs = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_name": "SASRecControl",
            "dataset": DATASET,
            "order": ORDER,
            "seed": SEED,
            "num_items": NUM_ITEMS,
            "max_len": MAX_LEN,
            "hidden_size": HIDDEN_SIZE,
            "num_heads": NUM_HEADS,
            "num_blocks": NUM_BLOCKS,
            "dropout": DROPOUT,
            "best_val_ndcg10": best_ndcg,
            "epoch": epoch,
            "protocol": "strict_target_after_last_history_bucket",
            "evaluation": "masked_full_batch_right_aligned_absolute_positions",
            "training_objective": "target_level_full_ranking_cross_entropy",
        }

        torch.save(checkpoint, checkpoint_path)
        print(">>> 保存最佳模型:", checkpoint_path)

    else:
        bad_epochs += 1
        print(
            f">>> Validation 连续 {bad_epochs} 个 epoch 未提升"
        )

    if bad_epochs >= PATIENCE:
        print("\n触发 Early Stopping。")
        break


pd.DataFrame(training_log).to_csv(
    OUTPUT_DIR / "training_log.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. 加载最佳模型并最终评价
# ============================================================

best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)
model.eval()

final_val_metrics, final_val_subgroup = evaluate(
    model,
    val_sequences,
    val_groups,
    val_num_tokens,
    val_targets,
    val_direct_pooling,
)

(
    test_metrics,
    test_subgroup,
    test_predictions,
) = evaluate(
    model,
    test_sequences,
    test_groups,
    test_num_tokens,
    test_targets,
    test_direct_pooling,
    sample_df=test_df,
    save_predictions=True,
)

test_predictions.to_csv(
    OUTPUT_DIR / "test_per_sample_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 15. 保存 metrics / config
# ============================================================

total_training_seconds = time.time() - training_begin

if torch.cuda.is_available():
    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )
else:
    peak_gpu_memory_gb = None

final_metrics = {
    "dataset": DATASET,
    "order": ORDER,
    "seed": SEED,
    "method": "SASRec_Control",
    "best_epoch": int(best_checkpoint["epoch"]),
    "validation": {
        "overall": final_val_metrics,
        "subgroup": final_val_subgroup,
    },
    "test": {
        "overall": test_metrics,
        "subgroup": test_subgroup,
    },
    "runtime": {
        "training_seconds": total_training_seconds,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
    },
}

with open(
    OUTPUT_DIR / "metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        final_metrics,
        f,
        ensure_ascii=False,
        indent=2,
    )

config = {
    "dataset": DATASET,
    "order": ORDER,
    "seed": SEED,
    "method": "SASRec_Control",
    "max_len": MAX_LEN,
    "hidden_size": HIDDEN_SIZE,
    "num_heads": NUM_HEADS,
    "num_blocks": NUM_BLOCKS,
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "training_positions": TRAINING_POSITIONS,
    "validation_protocol": "strict",
    "test_protocol": "strict",
    "evaluation": "full ranking",
    "padding_evaluation": "unpadded",
    "position_encoding": "right_aligned_absolute_control",
    "training_objective": "target_level_full_ranking_cross_entropy",
    "training_cutoff_sampling": "all_positions_shuffled_each_epoch",
}
config["x3_shuffle"] = dict(_X3_PROVENANCE)

with open(
    OUTPUT_DIR / "config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 16. 最终打印
# ============================================================

print("\n" + "=" * 80)
print("Step38.4 completed")
print("=" * 80)
print("Dataset:", DATASET)
print("Order:", ORDER)
print("Seed:", SEED)
print("Best epoch:", best_checkpoint["epoch"])

print("\nStrict Validation - Overall")
print("NDCG@10:", f"{final_val_metrics['NDCG@10']:.6f}")
print("HR@10:", f"{final_val_metrics['HR@10']:.6f}")
print("HR@20:", f"{final_val_metrics['HR@20']:.6f}")
print("MeanRank:", f"{final_val_metrics['MeanRank']:.2f}")
print(
    "Direct-Pooling eligible ratio:",
    f"{final_val_subgroup['pooling_ratio']:.6f}",
)

print("\nStrict Test - Overall")
print("NDCG@10:", f"{test_metrics['NDCG@10']:.6f}")
print("HR@10:", f"{test_metrics['HR@10']:.6f}")
print("HR@20:", f"{test_metrics['HR@20']:.6f}")
print("MeanRank:", f"{test_metrics['MeanRank']:.2f}")
print("Rank=1 ratio:", f"{test_metrics['Rank1Ratio'] * 100:.4f}%")
print("Direct-Pooling eligible ratio:", f"{test_subgroup['pooling_ratio']:.6f}")

print("\nStrict Test - Direct-Pooling")
for k, v in test_subgroup["direct_pooling"].items():
    print(k + ":", v)

print("\nStrict Test - Unpooled")
for k, v in test_subgroup["unpooled"].items():
    print(k + ":", v)

if peak_gpu_memory_gb is not None:
    print(
        "\nPeak GPU memory:",
        f"{peak_gpu_memory_gb:.3f} GB",
    )

print(
    "Training time:",
    f"{total_training_seconds:.1f}s",
)

print("\nOutput directory:", OUTPUT_DIR)
print(
    "\nNOTE: This is a single-seed pilot. "
    "Do not use Step28.2-vs-Step31.2 accuracy difference as a formal paper result yet."
)

_verify_originals()
with (OUTPUT_DIR/'test_per_sample_predictions.csv').open('r',encoding='utf-8-sig',newline='') as _handle:
    _SAVED_TEST_ROWS=list(csv.DictReader(_handle))
_MASK_PROOF=_canonical_test_mask_proof(test_df.to_dict('records'),_SAVED_TEST_ROWS,item_map)
_OUTPUT_METADATA_SHA256={name:checked_sha256(OUTPUT_DIR,name,None) for name in
    ('config.json','metrics.json','training_log.csv','test_per_sample_predictions.csv','item_id_map.csv')}
_COMPLETED=dict(_X3_PROVENANCE)
_COMPLETED.update(tag=_TAG,dataset=DATASET,seed=SEED,mode='train',status='COMPLETED',output_dir=str(OUTPUT_DIR),
    best_epoch=int(best_checkpoint["epoch"]),checkpoint='matched_control_sasrec_best.pt',
    output_metadata_sha256=_OUTPUT_METADATA_SHA256,all_original_hashes_verified=True,originals_unchanged=True,
    counts={'num_items':NUM_ITEMS,'training_rows':len(train_df),'training_positions':len(train_dataset),
        'trainable_users':len(_AUGMENT.metadata),'validation_samples':len(val_df),'test_samples':len(test_df)})
_COMPLETED.update(_MASK_PROOF)
_COMPLETED_PATH=_X3_STAGE/'verification'/('completed_'+_TAG+'_seed'+str(SEED)+'.json')
with _COMPLETED_PATH.open('x',encoding='utf-8') as _handle:
    _handle.write(json.dumps(_COMPLETED,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print('X3_SHUFFLE_TRAIN_COMPLETED='+str(_COMPLETED_PATH),flush=True)
