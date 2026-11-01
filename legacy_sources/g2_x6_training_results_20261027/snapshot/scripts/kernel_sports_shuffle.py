# X6 training-only kernel: selected audited definitions, no original top-level execution.
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (None, ':4096:8'):
    raise RuntimeError('X6 benchmark requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized() and not os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
    raise RuntimeError('Set the cuBLAS deterministic workspace before CUDA initialization')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from x6_cost_models import SASRecControl, RecentBucketSetR1SASRec
MAX_LEN = 50
BATCH_SIZE = 128
TAG = 'sports'
METHOD = 'shuffle'
DATASET = 'Amazon_Sports'


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

def prepare_data(frame, item_map, expected_rows, expected_positions):
    """Build only original visible training prefixes from an already-loaded frame."""
    if set(('user_id','item_id','time_bucket')) - set(frame.columns):
        raise ValueError('Required training columns absent')
    if len(frame) != expected_rows or sorted(item_map.values()) != list(range(1, len(item_map)+1)):
        raise ValueError('Frozen training row count or item map differs')
    frame = frame.copy()
    frame['user_id'] = frame['user_id'].astype(str)
    frame['item_id'] = frame['item_id'].astype(str)
    frame['_source_order'] = np.arange(len(frame), dtype=np.int64)
    if TAG in ('ml1m', 'beauty'):
        keys, event_key = ['user_id', 'time_bucket', '_source_order'], '_source_order'
    else:
        if 'raw_line' not in frame.columns:
            raise ValueError('Corrected Sports/Toys raw_line is required')
        keys, event_key = ['user_id', 'time_bucket', 'raw_line'], 'raw_line'
    frame = frame.sort_values(keys, kind='mergesort')
    sequences, aug_metadata, user_order = {}, {}, []
    for user, group in frame.groupby('user_id', sort=False):
        user = str(user)
        raw_items = group['item_id'].astype(str).tolist()
        try:
            items = np.asarray([item_map[raw] for raw in raw_items], dtype=np.int64)
        except KeyError as exc:
            raise ValueError('Training item outside the frozen item map') from exc
        if len(items) < 2:
            continue
        buckets = np.asarray(group['time_bucket'].tolist())
        events = tuple(int(value) for value in group[event_key].tolist())
        if len(set(events)) != len(events):
            raise ValueError('Duplicate original event key within a user')
        sequences[user] = {'items': items, 'buckets': buckets}
        aug_metadata[user] = {'items': tuple(int(item) for item in items),
                              'buckets': tuple(buckets), 'event_keys': events}
        user_order.append(user)
    positions = sum(len(info['items']) - 1 for info in sequences.values())
    if not positions or positions != expected_positions:
        raise ValueError('Legal next-item position coverage differs')
    return {'sequences': sequences, 'augmentation_metadata': aug_metadata,
            'user_order': user_order, 'rows': len(frame), 'positions': positions}


def initialize(data, num_items, seed=42):
    """Create a fresh temporary model/optimizer; no stored checkpoint is loaded."""
    global model, NUM_ITEMS, criterion, _AUGMENT, _AUGMENT_USERS
    if seed != 42:
        raise ValueError('Representative training seed is fixed to 42')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if METHOD == 'shuffle' and mp.get_context().get_start_method() != 'fork':
        raise RuntimeError('Shared epoch priorities require the original Linux fork workers')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Fixed across all benchmark methods; setup applies before model creation.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    NUM_ITEMS = int(num_items)
    if NUM_ITEMS < 10:
        raise ValueError('Unexpected item catalog size')
    cls = RecentBucketSetR1SASRec if METHOD == 'lbsp' else SASRecControl
    model = cls(num_items=NUM_ITEMS, max_len=50, hidden_size=64, num_heads=2,
                num_blocks=2, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    _AUGMENT = None
    _AUGMENT_USERS = data['user_order']
    if METHOD == 'shuffle':
        _AUGMENT = SharedPriorityState(DATASET, seed, data['augmentation_metadata'])
    if METHOD == 'lbsp':
        dataset = AllPositionR1Dataset(data['sequences'])
    elif TAG in ('ml1m', 'beauty'):
        dataset = ControlTrainDataset(data['sequences'], 50)
    else:
        dataset = AllPositionsDataset([data['sequences'][user]['items'] for user in data['user_order']])
    if len(dataset) != data['positions']:
        raise ValueError('Dataset legal-position count differs from frozen input')
    persistent = None
    if METHOD in ('control', 'shuffle') and TAG in ('ml1m', 'beauty'):
        generator = torch.Generator()
        generator.manual_seed(seed)
        persistent = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4,
                                pin_memory=True, persistent_workers=True, prefetch_factor=2,
                                generator=generator)
    return {'dataset': dataset, 'augmentation': _AUGMENT, 'seed': seed, 'device': device,
            'model': model, 'optimizer': optimizer, 'persistent_loader': persistent,
            'expected_positions': data['positions']}


def run_epoch(runtime, epoch):
    """Measure one complete train-only epoch, including per-epoch setup.

    Caller runs epoch1 as warmup, then epochs2..6 as measurements. Validation,
    checkpoint serialization, dataset read, and model construction are excluded.
    """
    if not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= 6:
        raise ValueError('Expected six fixed benchmark epochs, first warmup')
    if runtime['model'] is not model:
        raise ValueError('Runtime model differs from the loss-function binding')
    device, optimizer = runtime['device'], runtime['optimizer']
    model.train()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    setup_start = time.perf_counter()
    if METHOD == 'shuffle':
        runtime['augmentation'].set_epoch(epoch)
    augmentation_seconds = time.perf_counter() - setup_start if METHOD == 'shuffle' else 0.0
    loader = runtime['persistent_loader']
    if loader is None:
        generator = torch.Generator()
        generator.manual_seed(runtime['seed'] + epoch)
        sampler = RandomSampler(runtime['dataset'], replacement=False, generator=generator)
        loader = DataLoader(runtime['dataset'], batch_size=128, sampler=sampler, num_workers=0,
                            pin_memory=True, drop_last=False)
    total, batches, loss_sum = 0, 0, 0.0
    for batch in loader:
        if METHOD != 'lbsp' and TAG in ('ml1m', 'beauty'):
            seq, groups, tokens, pooled, last_bucket_size, targets = batch
            seq, groups, tokens, pooled, targets = [x.long().to(device) for x in (seq, groups, tokens, pooled, targets)]
            optimizer.zero_grad()
            loss = train_one_batch(model, seq, groups, tokens, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.item())
            int(pooled.sum().item())  # Original Control/Shuffle per-batch accounting/sync.
            total += len(seq)
        elif METHOD == 'lbsp':
            seq, groups, tokens, targets = [x.to(device, non_blocking=True) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(seq, groups, tokens, targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite LBSP loss')
            loss.backward()
            optimizer.step()
            total += len(targets)
            loss_sum += float(loss.item()) * len(targets)
        else:
            seq, lengths, targets = [x.to(device, non_blocking=True) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(seq, lengths, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite Control/Shuffle loss')
            loss.backward()
            optimizer.step()
            total += len(targets)
            loss_sum += float(loss.item()) * len(targets)
        batches += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if total != runtime['expected_positions'] or batches < 1 or not math.isfinite(loss_sum):
        raise ValueError('Incomplete training-position coverage or nonfinite loss')
    return {'epoch': epoch, 'warmup': epoch == 1, 'seconds': elapsed,
            'augmentation_setup_seconds': augmentation_seconds, 'positions': total,
            'batches': batches, 'loss_sum_as_logged': loss_sum,
            'validation_included': False, 'checkpoint_io_included': False,
            'training_state': 'temporary model and optimizer only'}
