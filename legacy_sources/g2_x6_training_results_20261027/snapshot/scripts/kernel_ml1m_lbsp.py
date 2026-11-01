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
TAG = 'ml1m'
METHOD = 'lbsp'
DATASET = 'MovieLens-1M'


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

class AllPositionR1Dataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        self.users = sorted(sequences.keys())

        self.positions = [
            (user_id, p)
            for user_id in self.users
            for p in range(
                1,
                len(sequences[user_id]["items"]),
            )
        ]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        user_id, p = self.positions[idx]

        info = self.sequences[user_id]

        history_items = info["items"][
            max(0, p - MAX_LEN):p
        ]

        history_buckets = info["buckets"][
            max(0, p - MAX_LEN):p
        ]

        (
            groups,
            num_tokens,
            direct_pooling,
            last_bucket_size,
        ) = build_groups_from_buckets(
            history_buckets,
            MAX_LEN,
        )

        history_items = canonicalize(
            history_items,
            last_bucket_size,
        )

        seq = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        seq[-len(history_items):] = history_items

        return (
            torch.from_numpy(seq),
            torch.from_numpy(groups.copy()),
            torch.tensor(num_tokens),
            torch.tensor(int(info["items"][p])),
        )

def training_loss(seq, groups, tokens, targets):
    scores = model.predict_batch_masked(
        seq,
        groups,
        tokens,
    )

    return F.cross_entropy(
        scores[:, 1:],
        targets - 1,
    )

build_groups_from_buckets = build_r1_group_positions_from_buckets
canonicalize = canonicalize_last_bucket_items


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
