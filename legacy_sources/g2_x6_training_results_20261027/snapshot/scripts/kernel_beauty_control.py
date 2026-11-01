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
TAG = 'beauty'
METHOD = 'control'
DATASET = 'Amazon_Beauty'


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
