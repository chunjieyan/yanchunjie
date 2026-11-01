"""Generate X6 training-only cost kernels from previously verified source text.

This module itself uses only the standard library and never reads a file or
executes a research definition. Callers must verify source bytes before using
them. Generated kernels run only on the server, against temporary models and
optimizers; they do not evaluate, save, load, or alter formal checkpoints.
"""
import ast
import hashlib
import json

VERSION = 'x6_train_kernel_v1'
TAGS = ('ml1m', 'beauty', 'sports', 'toys')
METHODS = ('control', 'shuffle', 'lbsp')
DATASETS = {'ml1m': 'MovieLens-1M', 'beauty': 'Amazon_Beauty',
            'sports': 'Amazon_Sports', 'toys': 'Amazon_Toys'}


def _node(text, name):
    candidates = [node for node in ast.parse(text).body
                  if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
    if len(candidates) != 1:
        raise ValueError('Expected exactly one scientific definition: ' + name)
    return candidates[0]


def _ast_digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode('utf-8')).hexdigest()


def build_training_kernel(tag, method, control_text, shuffle_text, lbsp_train_text, lbsp_base_text):
    """Return (generated_python, metadata) for one of the 4x3 kernels.

    Source arguments are trusted text only after the caller has checked their
    fixed SHA256 seals. The caller separately supplies `x6_cost_models.py`,
    generated from the pinned model/forward definitions by the already-audited
    x3_inference_adapter.build_frozen_module(control, lbsp_base, lbsp_forward).
    """
    if tag not in TAGS or method not in METHODS:
        raise ValueError('Unexpected benchmark tag/method')
    ml_style = tag in ('ml1m', 'beauty') and method != 'lbsp'
    selected = []
    def take(source_name, text, name):
        node = _node(text, name)
        selected.append((source_name, text, node))
    if method == 'lbsp':
        for name in ('build_r1_group_positions_from_buckets', 'canonicalize_last_bucket_items'):
            take('lbsp_base', lbsp_base_text, name)
        take('lbsp_training', lbsp_train_text, 'AllPositionR1Dataset')
        take('lbsp_training', lbsp_train_text, 'training_loss')
    else:
        text = shuffle_text if method == 'shuffle' else control_text
        origin = 'shuffle_training' if method == 'shuffle' else 'control_training'
        if method == 'shuffle':
            for name in ('event_priority', 'visible_order', 'verify_visible_invariants', 'SharedPriorityState'):
                take(origin, text, name)
        if ml_style:
            for name in ('build_control_group_positions', 'ControlTrainDataset', 'train_one_batch'):
                take(origin, text, name)
        else:
            for name in ('AllPositionsDataset', 'loss_for_batch'):
                take(origin, text, name)
    head = '''# X6 training-only kernel: selected audited definitions, no original top-level execution.
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
'''
    head += 'TAG = ' + repr(tag) + '\nMETHOD = ' + repr(method) + '\nDATASET = ' + repr(DATASETS[tag]) + '\n'
    selected_source = '\n\n'.join(ast.get_source_segment(text, node) for _, text, node in selected)
    aliases = '\n\nbuild_groups_from_buckets = build_r1_group_positions_from_buckets\ncanonicalize = canonicalize_last_bucket_items\n' if method == 'lbsp' else ''
    tail = r'''

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
'''
    generated = head + '\n\n' + selected_source + aliases + tail
    tree = ast.parse(generated, feature_version=(3, 10))
    generated_nodes = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for _, _, node in selected:
        if _ast_digest(node) != _ast_digest(generated_nodes[node.name]):
            raise RuntimeError('Original scientific definition AST changed: ' + node.name)
    metadata = {'version': VERSION, 'tag': tag, 'dataset': DATASETS[tag], 'method': method,
        'seed': 42, 'training_batch_size': 128, 'epochs': {'warmup': [1], 'measured': [2, 3, 4, 5, 6]},
        'dataset_style': 'control_ml_beauty' if ml_style else 'lbsp_all' if method == 'lbsp' else 'control_sports_toys',
        'scientific_definition_ast_sha256': {node.name: _ast_digest(node) for _, _, node in selected},
        'scientific_definition_source': {node.name: origin for origin, _, node in selected},
        'source_text_sha256_normalized': {name: hashlib.sha256(text.encode('utf-8')).hexdigest()
             for name, text in (('control', control_text), ('shuffle', shuffle_text),
                                 ('lbsp_train', lbsp_train_text), ('lbsp_base', lbsp_base_text))},
        'loader_workers': 4 if ml_style else 0, 'persistent_loader': ml_style,
        'sampler': 'one seed42 persistent-generator sampler' if ml_style else 'new seed(42+epoch) RandomSampler each epoch',
        'gradient_clip_norm': 5.0 if ml_style else None,
        'shuffle_epoch_priority_preparation_in_total_and_separately': method == 'shuffle',
        'timing': 'wall-clock perf_counter after GPU sync through final GPU sync, full legal-position training epoch; excludes data read/model init/validation/checkpoint I/O',
        'backend_policy': 'same deterministic algorithms and TF32 disabled across all methods; explicitly controlled benchmark setting',
        'scope': 'temporary model/optimizer only; no existing checkpoint load/save; no test/validation inference'}
    return generated, metadata
