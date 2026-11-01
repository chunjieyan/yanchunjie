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
_TAG='toys'
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

import argparse
import ast
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler


# ============================================================
# Step38.13 - Amazon Toys SASRec Control
# Unified BS128 / Full legal positions / Strict evaluation
# ============================================================

ROOT = _X3_ROOT
BASE = _X3_STAGE

DATASET = "Amazon_Toys"
ORDER = "file_order"
SEED = args.seed

BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256

LEARNING_RATE = 0.001
MAX_EPOCHS = 200
PATIENCE = 15

MAX_LEN = 50
HIDDEN_SIZE = 64
NUM_HEADS = 2
NUM_BLOCKS = 2
DROPOUT = 0.2

EXPECTED_VAL = 1939
EXPECTED_TEST = 2547
EXPECTED_ITEMS = 11924

TRAIN_PATH = ROOT / _PLAN["train_path"]

VAL_PATH = ROOT / _PLAN["validation_path"]

TEST_PATH = ROOT / _PLAN["test_path"]

ITEM_SOURCE_PATH = _MAP_PATH

MODEL_SCRIPT = _X3_STAGE / "scripts/model_definition.py"

OUTPUT_DIR = _X3_STAGE / "outputs" / f"toys_shuffle_seed{SEED}"
if OUTPUT_DIR.exists():
    raise RuntimeError("X3 formal output already exists; refusing to overwrite")


# Arguments parsed before numerical imports and integrity checks.

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 72)
print("STEP38.13 AMAZON TOYS SASREC CONTROL")
print("=" * 72)
print("Mode:", args.mode)
print("Device:", device)
print("Seed:", SEED)
print("Training BS:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Max epochs:", MAX_EPOCHS)
print("Patience:", PATIENCE)


# ============================================================
# 1. 输入文件检查
# ============================================================

for path in [
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    ITEM_SOURCE_PATH,
    MODEL_SCRIPT,
]:
    if not path.is_file():
        raise FileNotFoundError(path)

if OUTPUT_DIR.exists():
    raise RuntimeError(
        f"输出目录已存在，禁止覆盖: {OUTPUT_DIR}"
    )

print("INPUT FILE AUDIT PASS")


# ============================================================
# 2. 复用 Step38.1 已验证的模型类
# 不执行 ML1M 训练脚本的顶层训练代码
# ============================================================

tree = ast.parse(
    MODEL_SCRIPT.read_text(encoding="utf-8")
)

classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef)
    and node.name == "SASRecControl"
]

if len(classes) != 1:
    raise RuntimeError(
        "SASRecControl 类提取失败"
    )

namespace = {
    "__name__": "step38_sports",
    "torch": torch,
    "nn": nn,
    "F": F,
    "np": np,
    "math": math,
    "MAX_LEN": MAX_LEN,
    "device": device,
}

exec(
    compile(
        ast.Module(
            body=classes,
            type_ignores=[],
        ),
        str(MODEL_SCRIPT),
        "exec",
    ),
    namespace,
)

SASRecControl = namespace["SASRecControl"]

print("Step38 SASRecControl extracted: PASS")


# ============================================================
# 3. 统一物品映射：使用冻结 file_order 全量物品源
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

required_cols = {"user_id", "item_id", "raw_line", "time_bucket"}

if not required_cols.issubset(train_df.columns):
    raise RuntimeError(
        f"训练集缺少列: "
        f"{required_cols - set(train_df.columns)}"
    )

if train_df[
    ["user_id", "item_id", "raw_line"]
].isna().any().any():
    raise RuntimeError("训练集关键字段有缺失值")

# The sole scientific correction: canonical per-user temporal order.
for _key in ("time_bucket", "raw_line"):
    _values = train_df[_key].astype(str)
    if not _values.str.fullmatch(r"[+-]?[0-9]+").all():
        raise RuntimeError("Non-integer or missing training field: " + _key)
    _integers = [int(value) for value in _values]
    if any(value < -(2**63) or value >= 2**63 for value in _integers):
        raise RuntimeError("Training integer outside int64: " + _key)
    train_df[_key] = np.asarray(_integers, dtype=np.int64)

if train_df.duplicated(["user_id", "raw_line"]).any():
    raise RuntimeError("Duplicate (user_id, raw_line) event key")
_event_count = len(train_df)
_event_keys = set(zip(train_df["user_id"], train_df["raw_line"]))
train_df = train_df.sort_values(
    ["user_id", "time_bucket", "raw_line"], kind="stable",
)
if (len(train_df) != _event_count
        or set(zip(train_df["user_id"], train_df["raw_line"])) != _event_keys
        or sorted(train_df.index.tolist()) != list(range(_event_count))):
    raise RuntimeError("Ordering correction added or removed training events")
_same_user = train_df["user_id"].eq(train_df["user_id"].shift())
_bucket = train_df["time_bucket"]
_raw = train_df["raw_line"]
_bad_order = (_bucket.lt(_bucket.shift())
              | (_bucket.eq(_bucket.shift()) & _raw.le(_raw.shift())))
if (_same_user & _bad_order).any():
    raise RuntimeError("Canonical temporal/tie order assertion failed")
print("ORDERFIX PASS: unchanged event keys; user,time_bucket,raw_line stable order")

unknown_train = (
    set(train_df["item_id"]) - set(item_map)
)

if unknown_train:
    raise RuntimeError(
        f"训练集有 {len(unknown_train)} 个物品不在映射中"
    )

user_sequences = []

for _, group in train_df.groupby(
    "user_id",
    sort=False,
):
    seq = [
        item_map[x]
        for x in group["item_id"]
    ]

    if len(seq) >= 2:
        user_sequences.append(
            np.asarray(seq, dtype=np.int64)
        )

LEGAL_POSITIONS = sum(
    len(seq) - 1
    for seq in user_sequences
)

if LEGAL_POSITIONS <= 0:
    raise RuntimeError("无合法训练位置")

print("Training rows:", len(train_df))
print("Trainable users:", len(user_sequences))
print("All legal positions:", LEGAL_POSITIONS)
print(
    "Batches per epoch:",
    math.ceil(LEGAL_POSITIONS / BATCH_SIZE)
)


# ============================================================
# 5. 每一个合法 next-item 位置都构造一个训练样本
#
# position = 1,...,len(user_sequence)-1
# history = 前 position 个交互，最多取最后50个
# target = 第 position 个交互
#
# 每轮 RandomSampler 使用不同 seed 完整打乱
# ============================================================

class AllPositionsDataset(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

        self.positions = [
            (u, p)
            for u, seq in enumerate(sequences)
            for p in range(1, len(seq))
        ]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        u, p = self.positions[idx]

        full = self.sequences[u]

        history = full[
            max(0, p - MAX_LEN):p
        ]
        history = np.asarray(_AUGMENT.visible_items(_AUGMENT_USERS[u], max(0, p - MAX_LEN), p, history), dtype=np.int64)

        seq = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        seq[-len(history):] = history

        return (
            torch.from_numpy(seq),
            torch.tensor(
                len(history),
                dtype=torch.long,
            ),
            torch.tensor(
                int(full[p]),
                dtype=torch.long,
            ),
        )


_AUGMENT_METADATA = {}
_AUGMENT_USERS = []
for _user, _group in train_df.groupby("user_id", sort=False):
    if len(_group) >= 2:
        _user = str(_user)
        _AUGMENT_USERS.append(_user)
        _AUGMENT_METADATA[_user] = {
            "items": tuple(item_map[str(item)] for item in _group["item_id"]),
            "buckets": tuple(_group["time_bucket"]),
            "event_keys": tuple(int(key) for key in _group[_PLAN["event_key_column"]]),
        }
if len(train_df) != _PLAN["expected_training_rows"]:
    raise RuntimeError("Original training row count differs")
_AUGMENT = SharedPriorityState(DATASET, SEED, _AUGMENT_METADATA)

train_dataset = AllPositionsDataset(
    user_sequences
)
if len(train_dataset) != _PLAN["expected_training_positions"]:
    raise RuntimeError("Original legal-position coverage differs from the sealed Control protocol")
if _TAG in ('sports', 'toys'):
    if len(user_sequences) != len(_AUGMENT_USERS) or any(tuple(seq) != _AUGMENT.metadata[user]["items"] for seq,user in zip(user_sequences,_AUGMENT_USERS)):
        raise RuntimeError("Parallel Sports/Toys user/event metadata lost original sequence alignment")


assert len(train_dataset) == LEGAL_POSITIONS


# ============================================================
# 6. 读取严格验证/测试样本
# ============================================================

def load_strict_samples(path, expected_n):
    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_items": str,
        },
    )

    if len(df) != expected_n:
        raise RuntimeError(
            f"样本数异常: {path}, "
            f"actual={len(df)}, expected={expected_n}"
        )

    if not df["sample_id"].is_unique:
        raise RuntimeError(
            f"重复 sample_id: {path}"
        )

    sequences = []
    lengths = []
    targets = []

    for row in df.itertuples(index=False):
        raw_history = json.loads(
            row.sequence_items
        )

        if not (
            1 <= len(raw_history) <= MAX_LEN
        ):
            raise RuntimeError(
                f"严格历史长度异常: {row.sample_id}"
            )

        try:
            mapped = [
                item_map[str(x)]
                for x in raw_history
            ]

            target = item_map[
                str(row.target_item)
            ]

        except KeyError as exc:
            raise RuntimeError(
                f"评价物品不在映射中: "
                f"{row.sample_id}, {exc}"
            )

        if target in mapped:
            raise RuntimeError(
                f"目标物品泄漏进历史: {row.sample_id}"
            )

        seq = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        seq[-len(mapped):] = mapped

        sequences.append(seq)
        lengths.append(len(mapped))
        targets.append(target)

    return (
        df,
        np.stack(sequences),
        np.asarray(lengths, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    )


val_df, val_seq, val_len, val_target = (
    load_strict_samples(
        VAL_PATH,
        EXPECTED_VAL,
    )
)

test_df, test_seq, test_len, test_target = (
    load_strict_samples(
        TEST_PATH,
        EXPECTED_TEST,
    )
)

print("Strict validation:", len(val_target))
print("Strict test:", len(test_target))
print("STRICT DATA AUDIT PASS")


# ============================================================
# 7. 创建模型
# ============================================================

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


# ============================================================
# 8. 训练损失：next-item 全物品交叉熵
# 不使用旧版固定 cutoff + BCE 负采样训练
# ============================================================

def loss_for_batch(seq, lengths, targets):
    scores = model.predict_batch_masked(
        seq,
        lengths,
    )

    if scores.shape != (
        len(targets),
        NUM_ITEMS + 1,
    ):
        raise RuntimeError(
            f"训练 logits 形状异常: {scores.shape}"
        )

    # ID=0 是 padding，不参与 softmax 候选。
    loss = F.cross_entropy(
        scores[:, 1:],
        targets - 1,
    )

    return loss


# ============================================================
# 9. 严格全物品评价 + 已见物品屏蔽
# ============================================================

@torch.inference_mode()
def evaluate(
    sequences,
    lengths,
    targets,
    sample_df=None,
    return_predictions=False,
):
    model.eval()

    all_ranks = []
    all_top10 = []

    total = len(targets)

    for start in range(
        0,
        total,
        EVAL_BATCH_SIZE,
    ):
        end = min(
            start + EVAL_BATCH_SIZE,
            total,
        )

        seq_np = sequences[start:end]
        target_np = targets[start:end]

        seq_tensor = torch.from_numpy(
            seq_np
        ).long().to(device)

        len_tensor = torch.from_numpy(
            lengths[start:end]
        ).long().to(device)

        scores = model.predict_batch_masked(
            seq_tensor,
            len_tensor,
        )

        if scores.shape != (
            len(target_np),
            NUM_ITEMS + 1,
        ):
            raise RuntimeError(
                "评价 logits 维度异常"
            )

        if not torch.isfinite(scores).all():
            raise RuntimeError(
                "评价 logits 出现 NaN/Inf"
            )

        scores = scores.clone()
        scores[:, 0] = -1e9

        # 统一的已见物品屏蔽
        for b in range(len(seq_np)):
            seen = np.unique(
                seq_np[b]
            )

            seen = seen[
                (seen != 0)
                & (seen != target_np[b])
            ]

            if len(seen):
                scores[
                    b,
                    torch.as_tensor(
                        seen,
                        dtype=torch.long,
                        device=device,
                    ),
                ] = -1e9

        target_tensor = torch.as_tensor(
            target_np,
            dtype=torch.long,
            device=device,
        )

        rows = torch.arange(
            len(target_np),
            device=device,
        )

        target_scores = scores[
            rows,
            target_tensor,
        ]

        ranks = (
            (
                scores
                > target_scores[:, None]
            ).sum(dim=1)
            + 1
        )

        top10 = torch.topk(
            scores,
            k=10,
            dim=1,
        ).indices

        all_ranks.append(
            ranks.cpu().numpy()
        )

        if return_predictions:
            all_top10.append(
                top10.cpu().numpy()
            )

    ranks = np.concatenate(all_ranks)

    ndcg = np.where(
        ranks <= 10,
        1.0 / np.log2(ranks + 1),
        0.0,
    )

    metrics = {
        "samples": total,
        "ndcg10": float(ndcg.mean()),
        "hr10": float(
            (ranks <= 10).mean()
        ),
        "hr20": float(
            (ranks <= 20).mean()
        ),
        "mean_rank": float(
            ranks.mean()
        ),
    }

    if not return_predictions:
        return metrics

    if sample_df is None:
        raise RuntimeError(
            "保存预测时必须传入 sample_df"
        )

    top10 = np.concatenate(all_top10)

    pred = sample_df[
        ["sample_id", "user_id", "target_item"]
    ].copy()

    pred["target_rank"] = ranks

    pred["top10_item_idx"] = [
        json.dumps(row.tolist())
        for row in top10
    ]

    pred["top10_raw_item_id"] = [
        json.dumps([
            idx_to_raw[int(x)]
            for x in row
        ])
        for row in top10
    ]

    return metrics, pred


# ============================================================
# 10. Smoke Test：梯度 + 严格评价
# 默认只执行这里，不启动正式训练
# ============================================================

if args.mode == "smoke":
    _SMOKE_PROOF = _augmentation_smoke_proof()
else:
    _AUGMENT.set_epoch(0)
model.train()

smoke_loader = DataLoader(
    train_dataset,
    batch_size=min(
        8,
        len(train_dataset),
    ),
    shuffle=False,
    num_workers=0,
)

seq, lengths, targets = next(
    iter(smoke_loader)
)

seq = seq.to(device)
lengths = lengths.to(device)
targets = targets.to(device)

model.zero_grad(set_to_none=True)

smoke_loss = loss_for_batch(
    seq,
    lengths,
    targets,
)

if not torch.isfinite(smoke_loss):
    raise RuntimeError(
        "Smoke Loss NaN/Inf"
    )

if not smoke_loss.requires_grad:
    raise RuntimeError(
        "训练前向没有梯度，禁止启动正式训练"
    )

smoke_loss.backward()
_SMOKE_ALL_GRADIENTS_FINITE = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
if not _SMOKE_ALL_GRADIENTS_FINITE:
    raise RuntimeError("Smoke backward contains a nonfinite gradient")

gradient_count = sum(
    p.grad is not None
    and torch.isfinite(p.grad).all().item()
    and p.grad.abs().sum().item() > 0
    for p in model.parameters()
)

if gradient_count == 0:
    raise RuntimeError(
        "模型没有有效非零梯度"
    )

model.zero_grad(set_to_none=True)

smoke_val = evaluate(
    val_seq[:32],
    val_len[:32],
    val_target[:32],
)

print()
print("===== SMOKE TEST =====")
print("Training loss:", float(smoke_loss.item()))
print("Parameters with nonzero gradient:", gradient_count)
print("Strict validation smoke:", smoke_val)
print("SMOKE TEST PASS")

if args.mode == "smoke":
    _save_smoke_report(float(smoke_loss.item()), gradient_count, _SMOKE_ALL_GRADIENTS_FINITE, _SMOKE_PROOF)
    raise SystemExit(0)
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
(OUTPUT_DIR / "item_id_map.csv").write_bytes(_MAP_PATH.read_bytes())


# ============================================================
# 11. 正式训练
# ============================================================



mapping_df = pd.DataFrame({
    "raw_item_id": raw_items,
    "item_idx": np.arange(
        1,
        NUM_ITEMS + 1,
    ),
})



config = {'dataset': DATASET, 'model': 'SASRecControl', 'seed': SEED, 'training_batch_size': BATCH_SIZE, 'evaluation_batch_size': EVAL_BATCH_SIZE, 'optimizer': 'Adam', 'learning_rate': LEARNING_RATE, 'max_epochs': MAX_EPOCHS, 'patience': PATIENCE, 'early_stop_metric': 'validation_ndcg10', 'max_len': MAX_LEN, 'hidden_size': HIDDEN_SIZE, 'num_heads': NUM_HEADS, 'num_blocks': NUM_BLOCKS, 'dropout': DROPOUT, 'num_items': NUM_ITEMS, 'trainable_users': len(user_sequences), 'legal_training_positions': LEGAL_POSITIONS, 'sampling': 'all_legal_next_item_positions_reshuffled_every_epoch', 'loss': 'full_item_cross_entropy', 'train_path': str(TRAIN_PATH), 'validation_path': str(VAL_PATH), 'test_path': str(TEST_PATH), 'item_source_path': str(ITEM_SOURCE_PATH)}
config["x3_shuffle"] = dict(_X3_PROVENANCE)

(OUTPUT_DIR / "config.json").write_text(
    json.dumps(
        config,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

# Smoke Test 刚才只检查梯度，没有更新权重。
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_ndcg = -1.0
best_epoch = -1
bad_epochs = 0

training_log = []

checkpoint_path = (
    OUTPUT_DIR
    / "matched_control_sasrec_best.pt"
)

print()
print("=" * 72)
print("FORMAL TRAINING STARTED")
print("=" * 72)

for epoch in range(
    1,
    MAX_EPOCHS + 1,
):
    model.train()

    start_time = time.time()
    _AUGMENT.set_epoch(epoch)

    generator = torch.Generator()
    generator.manual_seed(
        SEED + epoch
    )

    sampler = RandomSampler(
        train_dataset,
        replacement=False,
        generator=generator,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    epoch_loss = 0.0
    total_examples = 0

    for seq, lengths, targets in loader:
        seq = seq.to(
            device,
            non_blocking=True,
        )
        lengths = lengths.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = loss_for_batch(
            seq,
            lengths,
            targets,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Epoch {epoch}: Loss NaN/Inf"
            )

        loss.backward()
        optimizer.step()

        n = len(targets)

        epoch_loss += (
            loss.item() * n
        )

        total_examples += n

    if total_examples != LEGAL_POSITIONS:
        raise RuntimeError(
            "本轮未覆盖全部合法训练位置"
        )

    val_metrics = evaluate(
        val_seq,
        val_len,
        val_target,
    )

    val_ndcg = val_metrics["ndcg10"]

    improved = (
        val_ndcg > best_ndcg + 1e-12
    )

    if improved:
        best_ndcg = val_ndcg
        best_epoch = epoch
        bad_epochs = 0

        torch.save(
            {
                "model_name": "SASRecControl",
                "dataset": DATASET,
                "epoch": epoch,
                "seed": SEED,
                "num_items": NUM_ITEMS,
                "max_len": MAX_LEN,
                "hidden_size": HIDDEN_SIZE,
                "num_heads": NUM_HEADS,
                "num_blocks": NUM_BLOCKS,
                "dropout": DROPOUT,
                "training_batch_size": BATCH_SIZE,
                "model_state_dict": model.state_dict(),
                "validation_metrics": val_metrics,
            },
            checkpoint_path,
        )

    else:
        bad_epochs += 1

    duration = time.time() - start_time

    record = {
        "epoch": epoch,
        "train_loss": (
            epoch_loss / total_examples
        ),
        "val_ndcg10": val_ndcg,
        "val_hr10": val_metrics["hr10"],
        "val_hr20": val_metrics["hr20"],
        "val_mean_rank": val_metrics["mean_rank"],
        "best_epoch": best_epoch,
        "best_val_ndcg10": best_ndcg,
        "bad_epochs": bad_epochs,
        "seconds": duration,
    }

    training_log.append(record)

    # 每轮保存日志，避免中断后没有记录。
    pd.DataFrame(
        training_log
    ).to_csv(
        OUTPUT_DIR / "training_log.csv",
        index=False,
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Loss={record['train_loss']:.6f} | "
        f"Val NDCG@10={val_ndcg:.6f} | "
        f"Best={best_ndcg:.6f} "
        f"(Epoch {best_epoch}) | "
        f"Bad={bad_epochs}/{PATIENCE} | "
        f"{duration:.1f}s",
        flush=True,
    )

    if bad_epochs >= PATIENCE:
        print(
            f"EARLY STOPPING AT EPOCH {epoch}",
            flush=True,
        )
        break


# ============================================================
# 12. 恢复最优 Epoch 模型
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.eval()

print()
print("BEST EPOCH RESTORED:", best_epoch)


# ============================================================
# 13. 严格验证/测试
# ============================================================

final_val = evaluate(
    val_seq,
    val_len,
    val_target,
)

final_test, test_predictions = evaluate(
    test_seq,
    test_len,
    test_target,
    sample_df=test_df,
    return_predictions=True,
)

if abs(
    final_val["ndcg10"] - best_ndcg
) >= 1e-8:
    raise RuntimeError(
        "恢复模型后的验证指标不一致"
    )

test_predictions.to_csv(
    OUTPUT_DIR / "test_per_sample_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

metrics = {
    "dataset": DATASET,
    "model": "SASRecControl",
    "seed": SEED,
    "best_epoch": best_epoch,
    "early_stopped": bad_epochs >= PATIENCE,
    "validation": final_val,
    "test": final_test,
}

(OUTPUT_DIR / "metrics.json").write_text(
    json.dumps(
        metrics,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("=" * 72)
print("STEP38.13 TOYS TRAINING PASS")
print("Best epoch:", best_epoch)
print("Val:", final_val)
print("Test:", final_test)
print("Output:", OUTPUT_DIR)
print("=" * 72)

_verify_originals()
with (OUTPUT_DIR/'test_per_sample_predictions.csv').open('r',encoding='utf-8-sig',newline='') as _handle:
    _SAVED_TEST_ROWS=list(csv.DictReader(_handle))
_MASK_PROOF=_canonical_test_mask_proof(test_df.to_dict('records'),_SAVED_TEST_ROWS,item_map)
_OUTPUT_METADATA_SHA256={name:checked_sha256(OUTPUT_DIR,name,None) for name in
    ('config.json','metrics.json','training_log.csv','test_per_sample_predictions.csv','item_id_map.csv')}
_COMPLETED=dict(_X3_PROVENANCE)
_COMPLETED.update(tag=_TAG,dataset=DATASET,seed=SEED,mode='train',status='COMPLETED',output_dir=str(OUTPUT_DIR),
    best_epoch=int(best_epoch),checkpoint='matched_control_sasrec_best.pt',
    output_metadata_sha256=_OUTPUT_METADATA_SHA256,all_original_hashes_verified=True,originals_unchanged=True,
    counts={'num_items':NUM_ITEMS,'training_rows':len(train_df),'training_positions':len(train_dataset),
        'trainable_users':len(_AUGMENT.metadata),'validation_samples':len(val_df),'test_samples':len(test_df)})
_COMPLETED.update(_MASK_PROOF)
_COMPLETED_PATH=_X3_STAGE/'verification'/('completed_'+_TAG+'_seed'+str(SEED)+'.json')
with _COMPLETED_PATH.open('x',encoding='utf-8') as _handle:
    _handle.write(json.dumps(_COMPLETED,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print('X3_SHUFFLE_TRAIN_COMPLETED='+str(_COMPLETED_PATH),flush=True)
