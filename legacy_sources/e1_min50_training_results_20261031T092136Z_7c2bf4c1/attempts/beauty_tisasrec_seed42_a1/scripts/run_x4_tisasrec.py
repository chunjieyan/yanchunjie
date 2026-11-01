"""X4: six isolated TiSASRec models under the frozen strict protocol.

Default is plan-only. All data access/model execution requires --run.
No downloads, dependency installs, existing checkpoint reads, or overwrites.
"""
from pathlib import Path
import argparse
import collections
import csv
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid

VERSION = 'x4_tisasrec_v1'
PREFIX = 'kais_x4_tisasrec_v1_'
INPUT_NAME = 'x4_tisasrec_inputs_20261026.json'
INPUT_SHA256 = '4dfe330abba456fbc471d2b09e85a204ce80727e2a81f28348ebea84ca3392f4'
TOOLS = ('run_x4_tisasrec.py', 'collect_x4_tisasrec.py', 'x4_tisasrec_model.py', 'x4_tisasrec_protocol.py')
SEEDS = (42, 2026, 3407)
PERMS = (101, 202, 303, 404, 505)
TAGS = ('ml1m', 'beauty')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe(root, relative):
    root = Path(root).resolve()
    path = root / relative
    path.resolve().relative_to(root)
    return path


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_inputs(root, inputs, tag=None):
    entries = inputs['data_sha256']
    if tag is not None:
        keep = set(inputs['datasets'][tag]['required_files'])
        entries = {path: value for path, value in entries.items() if path in keep}
    for relative, expected in entries.items():
        if sha(safe(root, relative)) != expected:
            raise ValueError('Frozen input changed: ' + relative)


def verify_stage(root, stage):
    root, stage = Path(root).resolve(), Path(stage).resolve()
    if stage.parent != root or not stage.name.startswith(PREFIX):
        raise ValueError('Stage must be a direct X4 child of the specified project')
    manifest = read_json(stage / 'manifest.json')
    if manifest['version'] != VERSION or manifest['root'] != str(root) or manifest['stage'] != str(stage):
        raise ValueError('Stage identity mismatch')
    for relative, expected in manifest['tool_sha256'].items():
        if sha(safe(stage, relative)) != expected:
            raise ValueError('Stage tool changed: ' + relative)
    if sha(stage / INPUT_NAME) != INPUT_SHA256:
        raise ValueError('Input seal mismatch')
    return manifest, read_json(stage / INPUT_NAME)


def prepare(root):
    root = Path(root).resolve()
    here = Path(__file__).resolve().parent
    if not root.is_dir():
        raise ValueError('Project root does not exist')
    if list(root.glob(PREFIX + '*')):
        raise ValueError('Existing X4 stage found; inspect it before starting another six-run job')
    if sha(here / INPUT_NAME) != INPUT_SHA256:
        raise ValueError('Reviewed input seal changed')
    inputs = read_json(here / INPUT_NAME)
    if inputs['version'] != VERSION or inputs['training_seeds'] != list(SEEDS):
        raise ValueError('Input version/seeds differ')
    for name, expected in inputs['helper_sha256'].items():
        if sha(here / name) != expected:
            raise ValueError('Reviewed helper changed: ' + name)
    verify_inputs(root, inputs)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:10]
    stage = root / (PREFIX + stamp)
    stage.mkdir()
    (stage / 'scripts').mkdir()
    (stage / 'verification').mkdir()
    (stage / 'outputs').mkdir()
    files = list(TOOLS) + ['X4_UPSTREAM_MIT_LICENSE.txt', 'X4_MODEL_PROVENANCE.json']
    hashes = {}
    for name in files:
        shutil.copyfile(here / name, stage / 'scripts' / name)
        hashes['scripts/' + name] = sha(stage / 'scripts' / name)
    shutil.copyfile(here / INPUT_NAME, stage / INPUT_NAME)
    hashes[INPUT_NAME] = sha(stage / INPUT_NAME)
    manifest = {'version': VERSION, 'root': str(root), 'stage': str(stage), 'created_utc': stamp,
                'input_sha256': INPUT_SHA256, 'tool_sha256': hashes, 'data_sha256': inputs['data_sha256'],
                'tasks': [{'dataset': tag, 'seed': seed} for tag in TAGS for seed in SEEDS],
                'protocol': inputs['protocol']}
    write_json(stage / 'manifest.json', manifest)
    return stage


def metrics(ranks):
    if not ranks or any(not isinstance(r, int) or r < 1 for r in ranks):
        raise ValueError('Invalid ranks')
    n = len(ranks)
    return {'ndcg10': sum(1 / math.log2(r + 1) for r in ranks if r <= 10) / n,
            'hr10': sum(r <= 10 for r in ranks) / n, 'hr20': sum(r <= 20 for r in ranks) / n,
            'recall50': sum(r <= 50 for r in ranks) / n, 'mrr': sum(1 / r for r in ranks) / n,
            'mrr10': sum(1 / r for r in ranks if r <= 10) / n, 'mean_rank': sum(ranks) / n}


def verify_job_proof(stage, inputs, mode, tag, seed):
    path = stage / 'verification' / ('smoke_' + tag + '.json' if mode == 'smoke' else 'completed_' + tag + '_seed' + str(seed) + '.json')
    proof = read_json(path)
    manifest = read_json(stage / 'manifest.json')
    expected = {'status': 'SMOKE_PASS' if mode == 'smoke' else 'COMPLETED', 'version': VERSION,
                'tag': tag, 'seed': seed, 'input_sha256': INPUT_SHA256,
                'manifest_sha256': sha(stage / 'manifest.json'), 'tool_sha256': manifest['tool_sha256']}
    if any(proof.get(key) != value for key, value in expected.items()):
        raise ValueError('Job proof identity/source mismatch: ' + str(path))
    plan = inputs['datasets'][tag]
    contract = proof.get('contract', {})
    expected_contract = {'status': 'PASS', 'training_rows': plan['expected_training_rows'],
                         'training_positions': plan['expected_training_positions'], 'num_items': plan['expected_items'],
                         'trainable_users': plan['expected_trainable_users'],
                         'validation_samples': plan['expected_validation'], 'test_samples': plan['expected_test'],
                         'affected_samples': plan['expected_affected'], 'event_join_complete': True,
                         'strict_target_bucket_after_history': True, 'all_five_event_multisets_and_prefixes_preserved': True,
                         'target_time_used_as_feature': False, 'native_time_unit': plan['native_time_unit']}
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise ValueError('Data contract failed/differs: ' + str(path))
    if mode == 'smoke':
        details = proof.get('proof', {})
        flags = ('matrix_exact_integer_reference', 'zero_gap_and_clip', 'causal_and_padding_mask',
                 'batch_single_allclose', 'save_restore_identical_call')
        if any(details.get(key) is not True for key in flags) or details.get('optimizer_steps') != 0 or details.get('test_inference') is not False or details.get('formal_weights_saved') is not False:
            raise ValueError('Smoke numerical proof/scope failed')
        if not math.isfinite(details.get('finite_loss', math.nan)) or details.get('finite_gradient_tensors', 0) < 1:
            raise ValueError('Smoke loss/gradient proof failed')
    else:
        if any(proof.get(key) is not True for key in ('original_inputs_unchanged', 'validation_reproduced', 'canonical_and_five_permutations')):
            raise ValueError('Training completion scope failed')
    return path


def load_data(root, plan, protocol):
    """Server-only contract validation. No target time is fed to the model."""
    mapping_rows = read_csv(safe(root, plan['map_path']))
    mapping = {row['raw_item_id']: int(row['item_idx']) for row in mapping_rows}
    if len(mapping) != plan['expected_items'] or sorted(mapping.values()) != list(range(1, len(mapping) + 1)):
        raise ValueError('Non-contiguous or duplicate item mapping')
    inverse = {value: key for key, value in mapping.items()}
    panel_paths = {'validation': plan['validation_path'], 'original': plan['test_path']}
    panel_paths.update({'perm_seed' + str(seed): plan['permutation_paths'][str(seed)] for seed in PERMS})
    panels = {key: read_csv(safe(root, value)) for key, value in panel_paths.items()}
    reference_ids = [row['sample_id'] for row in panels['original']]
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError('Duplicate canonical sample ID')
    for seed in PERMS:
        name = 'perm_seed' + str(seed)
        lookup = {row['sample_id']: row for row in panels[name]}
        if len(lookup) != len(panels[name]) or set(lookup) != set(reference_ids):
            raise ValueError('Permutation sample ID population differs')
        # Align rows only, never reorder an event inside a supplied history.
        panels[name] = [lookup[sample] for sample in reference_ids]
    wanted = set()
    for name, rows in panels.items():
        count = plan['expected_validation'] if name == 'validation' else plan['expected_test']
        if len(rows) != count or len({row['sample_id'] for row in rows}) != count:
            raise ValueError('Panel count/unique IDs mismatch: ' + name)
        for row in rows:
            items = [str(item) for item in protocol.parse_list(row['sequence_items'])]
            events = [protocol.native_time(key) for key in protocol.parse_list(row['sequence_raw_lines'])]
            if not 1 <= len(items) <= 50 or len(items) != len(events) or len(set(events)) != len(events):
                raise ValueError('History event identities invalid')
            if row['target_item'] in items or row['target_item'] not in mapping:
                raise ValueError('Target/history mapping or target leakage')
            if any(item not in mapping for item in items):
                raise ValueError('Unknown history item')
            target_event = protocol.native_time(row['target_raw_line'])
            if target_event in events:
                raise ValueError('Target event in history')
            row['_raw'], row['_events'] = items, events
            row['_target_event'] = target_event
            wanted.update((row['user_id'], event) for event in events + [target_event])
    event_index = {}
    with safe(root, plan['event_source_path']).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'user_id', 'item_id', 'raw_line', 'time_bucket', 'timestamp'}.issubset(reader.fieldnames or []):
            raise ValueError('Native event source lacks required columns')
        for row in reader:
            key = (row['user_id'], protocol.native_time(row['raw_line']))
            if key in wanted:
                if key in event_index:
                    raise ValueError('Duplicate event key')
                event_index[key] = protocol.normalize_event(row, plan['dataset'])
    if set(event_index) != wanted:
        raise ValueError('Visible event/target join incomplete')
    for name, rows in panels.items():
        for row in rows:
            times = []
            for raw, event in zip(row['_raw'], row['_events']):
                joined = event_index[(row['user_id'], event)]
                if joined['item_id'] != raw:
                    raise ValueError('Item/event alignment mismatch')
                times.append(joined['time_bucket'])
            target = event_index[(row['user_id'], row['_target_event'])]
            target_bucket = target['time_bucket']
            if target['item_id'] != row['target_item'] or max(times) >= target_bucket:
                raise ValueError('Strict history is not earlier than target bucket')
            if protocol.native_time(row['target_bucket']) != target_bucket or protocol.native_time(row['target_timestamp']) != target['timestamp']:
                raise ValueError('Target time metadata mismatch')
            if times != sorted(times):
                raise ValueError('Panel crosses chronological bucket order')
            k = len(times) - next(i for i, value in enumerate(times) if value == times[-1])
            declared = row.get('last_bucket_size') or row.get('last_history_bucket_size')
            if declared is not None and min(int(declared), len(times)) != k:
                raise ValueError('Last bucket size mismatch')
            row['_times'], row['_last_k'] = times, k
            row['_target_bucket'], row['_target_timestamp'] = target_bucket, target['timestamp']
            row['_items'] = [mapping[raw] for raw in row['_raw']]
            row['_target'] = mapping[row['target_item']]
    original = panels['original']
    affected = []
    for index, reference in enumerate(original):
        histories = []
        for seed in PERMS:
            row = panels['perm_seed' + str(seed)][index]
            for field in ('sample_id', 'user_id', 'target_item', '_target_event', '_target_bucket', '_target_timestamp'):
                if row[field] != reference[field]:
                    raise ValueError('Permutation identity/order mismatch')
            k = reference['_last_k']
            old = list(zip(reference['_raw'], reference['_events'], reference['_times']))
            new = list(zip(row['_raw'], row['_events'], row['_times']))
            if old[:-k] != new[:-k] or collections.Counter(old[-k:]) != collections.Counter(new[-k:]):
                raise ValueError('Permutation changed prefix or last-bucket event multiset')
            histories.append(tuple(row['_raw']))
        affected.append(len(set(histories)) >= 2)
    if sum(affected) != plan['expected_affected']:
        raise ValueError('Affected population differs from established methods')
    groups = collections.defaultdict(list)
    training_rows = 0
    with safe(root, plan['train_path']).open(encoding='utf-8-sig', newline='') as handle:
        for order, row in enumerate(csv.DictReader(handle)):
            if row['item_id'] not in mapping:
                raise ValueError('Unknown training item')
            bucket = protocol.native_time(row['time_bucket'])
            groups[row['user_id']].append((bucket, order, mapping[row['item_id']]))
            training_rows += 1
    sequences = []
    for user in sorted(groups):
        records = sorted(groups[user], key=lambda entry: (entry[0], entry[1]))
        if len(records) >= 2:
            sequences.append(([record[2] for record in records], [record[0] for record in records]))
    positions = sum(len(items) - 1 for items, _ in sequences)
    if positions != plan['expected_training_positions'] or training_rows != plan['expected_training_rows'] or len(sequences) != plan['expected_trainable_users']:
        raise ValueError('Training coverage changed')
    contract = {'status': 'PASS', 'training_rows': training_rows, 'training_positions': positions,
                'trainable_users': len(sequences), 'num_items': len(mapping), 'validation_samples': len(panels['validation']),
                'test_samples': len(original), 'affected_samples': sum(affected), 'event_join_complete': True,
                'strict_target_bucket_after_history': True, 'all_five_event_multisets_and_prefixes_preserved': True,
                'native_time_unit': plan['native_time_unit'], 'target_time_used_as_feature': False}
    # Clearly labelled, bounded contract diagnostics rather than an invented full-population claim.
    matrix_cells, clipped, zero = 0, 0, 0
    for rows in (panels['validation'][:128], original[:128]):
        for row in rows:
            matrix = protocol.time_matrix(row['_times'])
            for a in range(len(row['_times'])):
                for b in range(a + 1, len(row['_times'])):
                    matrix_cells += 1
                    clipped += int(matrix[a][b] == 256)
                    zero += int(matrix[a][b] == 0)
    contract['time_matrix_diagnostic'] = {'scope': 'first at most128 validation and128 original test visible histories; no target time',
        'valid_unordered_pairs': matrix_cells, 'zero_pairs': zero, 'at_clip_256_pairs': clipped}
    return mapping, inverse, panels, sequences, affected, contract


def tensor_batch(histories, times, targets, protocol, torch, device):
    items = torch.zeros((len(histories), 50), dtype=torch.long)
    buckets = torch.zeros_like(items)
    lengths, scales = [], []
    for i, (history, native) in enumerate(zip(histories, times)):
        offsets, scale = protocol.time_features(native)
        if max(offsets) >= 2 ** 62:
            raise ValueError('Native time offsets exceed safe int64 range')
        items[i, :len(history)] = torch.tensor(history, dtype=torch.long)
        buckets[i, :len(history)] = torch.tensor(offsets, dtype=torch.long)
        lengths.append(len(history))
        scales.append(scale)
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    buckets = buckets.to(device)
    scales = torch.tensor(scales, dtype=torch.long, device=device)
    matrix = torch.div((buckets[:, :, None] - buckets[:, None, :]).abs(), scales[:, None, None], rounding_mode='floor').clamp(max=256)
    valid = torch.arange(50, device=device)[None, :] < lengths[:, None]
    matrix = matrix.masked_fill(~(valid[:, :, None] & valid[:, None, :]), 0)
    return items.to(device), lengths, matrix, torch.tensor(targets, dtype=torch.long, device=device)


def evaluate(model, rows, inverse, protocol, torch, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            arguments = tensor_batch([r['_items'] for r in batch], [r['_times'] for r in batch], [r['_target'] for r in batch], protocol, torch, device)
            scores = model.score_all(*arguments[:3])
            if not torch.isfinite(scores).all():
                raise ValueError('Nonfinite unmasked model score')
            scores[:, 0] = -torch.inf
            candidates = []
            for i, row in enumerate(batch):
                seen = set(row['_items']) - {row['_target']}
                scores[i, list(seen)] = -torch.inf
                candidates.append(len(inverse) - len(seen))
            target_scores = scores.gather(1, arguments[3][:, None])
            ranks = ((scores > target_scores).sum(1) + 1).cpu().tolist()
            top10 = scores.topk(10, dim=1).indices.cpu().tolist()
            for row, rank, top, count in zip(batch, ranks, top10, candidates):
                if 0 in top or len(set(top)) != 10 or set(top) & (set(row['_items']) - {row['_target']}):
                    raise ValueError('Top10 violates candidate mask')
                predictions.append({'sample_id': row['sample_id'], 'user_id': row['user_id'], 'target_item': row['target_item'],
                                    'target_idx': row['_target'], 'rank': rank, 'top10_item_idx': json.dumps(top),
                                    'top10_raw_item_id': json.dumps([inverse[item] for item in top]), 'candidate_count': count})
    return metrics([int(row['rank']) for row in predictions]), predictions


def stability(panel_predictions, affected):
    result = {}
    for population, indices in [('full', list(range(len(affected)))), ('affected', [i for i, flag in enumerate(affected) if flag])]:
        if not indices:
            raise ValueError('Empty stability population')
        per_user = []
        for i in indices:
            values = [panel_predictions['perm_seed' + str(seed)][i] for seed in PERMS]
            tops = [json.loads(value['top10_item_idx']) for value in values]
            overlap = sum(len(set(a) & set(b)) / 10 for a, b in itertools.combinations(tops, 2)) / 10
            flips = len({int(value['rank']) <= 10 for value in values}) > 1
            per_user.append({'sample_id': values[0]['sample_id'], 'overlap10': overlap, 'any_hitflip': int(flips),
                             'top10_all_identical': int(all(top == tops[0] for top in tops[1:]))})
        result[population] = {'n': len(indices), 'overlap10': sum(r['overlap10'] for r in per_user) / len(indices),
                              'hitflip': sum(r['any_hitflip'] for r in per_user) / len(indices),
                              'ten_unordered_pairs_per_user': True, 'users': per_user}
    return result


def smoke_model(model, protocol, torch, device, stage, tag):
    """Synthetic model checks; no test-panel inference or optimizer step."""
    model.eval()
    histories = [[1, 2, 3], [2], [1, 3], [1, 4, 2]]
    native = [[0, 0, 10], [0], [0, 100000], [0, 1, 100000]]
    targets = [4, 3, 2, 5]
    batch = tensor_batch(histories, native, targets, protocol, torch, device)
    for i, times in enumerate(native):
        if batch[2][i].cpu().tolist() != protocol.time_matrix(times):
            raise ValueError('GPU time matrix differs from exact integer reference')
    if protocol.time_matrix([0, 1, 100000])[0][2] != 256 or protocol.time_matrix([4, 4])[0][1] != 0:
        raise ValueError('Time clipping/zero gap failed')
    with torch.no_grad():
        scores = model.score_all(*batch[:3])
        singles = []
        for history, buckets, target in zip(histories, native, targets):
            arguments = tensor_batch([history], [buckets], [target], protocol, torch, device)
            singles.append(model.score_all(*arguments[:3]))
        single_scores = torch.cat(singles)
        if not torch.allclose(scores, single_scores, atol=2e-5, rtol=2e-5):
            raise ValueError('Batched/single forward mismatch')
        mask = model.get_attention_mask(batch[0])
        if not bool((mask[0, 0, 0, 1:] < -100).all()) or not bool((mask[0, 0, 2, :3] == 0).all()):
            raise ValueError('Causal/key-padding attention mask mismatch')
    model.train()
    loss = torch.nn.functional.cross_entropy(model.score_all(*batch[:3])[:, 1:], batch[3] - 1)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not math.isfinite(float(loss)) or not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError('Nonfinite loss/gradient')
    model.zero_grad(set_to_none=True)
    model.eval()
    checkpoint = stage / 'verification' / ('temporary_synthetic_' + tag + '.pt')
    try:
        torch.save(model.state_dict(), checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        with torch.no_grad():
            restored = model.score_all(*batch[:3])
        if not torch.equal(restored, scores):
            raise ValueError('Identical-call save/restore scores changed')
    finally:
        checkpoint.unlink(missing_ok=True)
    return {'finite_loss': float(loss), 'finite_gradient_tensors': len(gradients),
            'matrix_exact_integer_reference': True, 'zero_gap_and_clip': True,
            'causal_and_padding_mask': True, 'batch_single_allclose': True, 'save_restore_identical_call': True,
            'optimizer_steps': 0, 'test_inference': False, 'formal_weights_saved': False}


def child(root, stage, mode, tag, seed):
    manifest, inputs = verify_stage(root, stage)
    verify_inputs(root, inputs, tag)
    protocol = load_module(stage / 'scripts/x4_tisasrec_protocol.py', 'x4_protocol')
    mapping, inverse, panels, sequences, affected, contract = load_data(root, inputs['datasets'][tag], protocol)
    # Only this explicit server execution path imports model dependencies.
    import numpy as np
    import torch
    import multiprocessing
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; refusing an accidental CPU training run')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    module = load_module(stage / 'scripts/x4_tisasrec_model.py', 'x4_model')
    model = module.TiSASRecModel(len(mapping)).to(device)
    provenance = {'version': VERSION, 'tag': tag, 'seed': seed, 'input_sha256': INPUT_SHA256,
                  'manifest_sha256': sha(stage / 'manifest.json'), 'tool_sha256': manifest['tool_sha256'],
                  'contract': contract, 'protocol': inputs['protocol'],
                  'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                                  'device': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}}
    if mode == 'smoke':
        proof = smoke_model(model, protocol, torch, device, stage, tag)
        verify_inputs(root, inputs, tag)
        write_json(stage / 'verification' / ('smoke_' + tag + '.json'), dict(provenance, status='SMOKE_PASS', proof=proof))
        print('X4_SMOKE_PASS=' + tag, flush=True)
        return
    for smoke_tag in TAGS:
        verify_job_proof(stage, inputs, 'smoke', smoke_tag, 42)
    if multiprocessing.get_context().get_start_method() != 'fork':
        raise RuntimeError('This four-worker loader requires Linux fork')
    output = stage / 'outputs' / (tag + '_seed' + str(seed))
    output.mkdir(exist_ok=False)
    config = dict(provenance, hyperparameters=inputs['hyperparameters'], method='TiSASRec_strict_prefix_time_adaptation')
    config['e1_minimum_stopping_epochs'] = 50
    write_json(output / 'config.json', config)
    write_csv(output / 'item_id_map.csv', [{'raw_item_id': raw, 'item_idx': index} for raw, index in mapping.items()])

    class TrainDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.positions = [(i, end) for i, (items, _) in enumerate(sequences) for end in range(1, len(items))]

        def __len__(self):
            return len(self.positions)

        def __getitem__(self, index):
            group, end = self.positions[index]
            items, native = sequences[group]
            start = max(0, end - 50)
            offsets, scale = protocol.time_features(native[start:end])
            if max(offsets) >= 2 ** 62:
                raise ValueError('Training time range exceeds safe int64')
            history = torch.zeros(50, dtype=torch.long)
            buckets = torch.zeros(50, dtype=torch.long)
            history[:end - start] = torch.tensor(items[start:end], dtype=torch.long)
            buckets[:end - start] = torch.tensor(offsets, dtype=torch.long)
            return history, end - start, buckets, scale, items[end]

    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(TrainDataset(), batch_size=128, shuffle=True, num_workers=4,
                                        persistent_workers=True, prefetch_factor=2, pin_memory=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best, best_epoch, bad = -math.inf, 0, 0
    fields = ['epoch', 'train_loss', 'val_ndcg10', 'val_hr10', 'val_hr20', 'val_recall50', 'val_mrr', 'val_mrr10',
              'val_mean_rank', 'seconds', 'best_epoch', 'bad_epochs']
    log_path = output / 'training_log.csv'
    with log_path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, 201):
            started = time.perf_counter()
            model.train()
            loss_sum, total = 0.0, 0
            for items, lengths, buckets, scales, targets in loader:
                items, lengths, buckets, scales, targets = [v.to(device, non_blocking=True) for v in (items, lengths, buckets, scales, targets)]
                matrix = torch.div((buckets[:, :, None] - buckets[:, None, :]).abs(), scales[:, None, None], rounding_mode='floor').clamp(max=256)
                valid = torch.arange(50, device=device)[None, :] < lengths[:, None]
                matrix.masked_fill_(~(valid[:, :, None] & valid[:, None, :]), 0)
                optimizer.zero_grad(set_to_none=True)
                scores = model.score_all(items, lengths, matrix)
                loss = torch.nn.functional.cross_entropy(scores[:, 1:], targets - 1)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(items)
                total += len(items)
            if total != contract['training_positions']:
                raise ValueError('Epoch missed/duplicated training positions')
            validation, predictions = evaluate(model, panels['validation'], inverse, protocol, torch, device)
            if validation['ndcg10'] > best:
                best, best_epoch, bad = validation['ndcg10'], epoch, 0
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'seed': seed,
                            'input_sha256': INPUT_SHA256}, output / 'tisasrec_best.pt')
                write_csv(output / 'validation_best_predictions.csv', predictions)
            else:
                bad += 1
            torch.cuda.synchronize()
            record = dict(epoch=epoch, train_loss=loss_sum / total, seconds=time.perf_counter() - started,
                          best_epoch=best_epoch, bad_epochs=bad)
            record.update({'val_' + key: value for key, value in validation.items()})
            writer.writerow(record)
            handle.flush()
            print(tag, seed, json.dumps(record), flush=True)
            if epoch >= 50 and bad >= 15:
                break
    del loader
    saved = torch.load(output / 'tisasrec_best.pt', map_location=device, weights_only=True)
    if saved['seed'] != seed or saved['input_sha256'] != INPUT_SHA256 or saved['epoch'] != best_epoch:
        raise ValueError('Checkpoint identity changed')
    model.load_state_dict(saved['state_dict'])
    selected_validation, selected_predictions = evaluate(model, panels['validation'], inverse, protocol, torch, device)
    old_best = read_csv(output / 'validation_best_predictions.csv')
    if [int(r['rank']) for r in old_best] != [int(r['rank']) for r in selected_predictions] or abs(selected_validation['ndcg10'] - best) > 1e-12:
        raise ValueError('Selected checkpoint does not reproduce best validation ranks')
    all_metrics, all_predictions = {}, {}
    for name in ['original'] + ['perm_seed' + str(seed) for seed in PERMS]:
        values, predictions = evaluate(model, panels[name], inverse, protocol, torch, device)
        all_metrics[name], all_predictions[name] = values, predictions
        write_csv(output / ('predictions_' + name + '.csv'), predictions)
    write_csv(output / 'panel_labels.csv', [{'sample_id': row['sample_id'], 'affected': int(flag)} for row, flag in zip(panels['original'], affected)])
    write_json(output / 'metrics.json', {'best_epoch': best_epoch, 'best_validation': selected_validation, 'panels': all_metrics})
    write_json(output / 'stability.json', stability(all_predictions, affected))
    verify_inputs(root, inputs, tag)
    verify_stage(root, stage)
    metadata = {path.name: sha(path) for path in output.iterdir() if path.is_file() and path.suffix != '.pt'}
    write_json(stage / 'verification' / ('completed_' + tag + '_seed' + str(seed) + '.json'),
               dict(provenance, status='COMPLETED', best_epoch=best_epoch, metadata_sha256=metadata,
                    checkpoint_sha256=sha(output / 'tisasrec_best.pt'), original_inputs_unchanged=True,
                    validation_reproduced=True, canonical_and_five_permutations=True))
    print('X4_TRAIN_COMPLETED=' + tag + '_seed' + str(seed), flush=True)


def run(root):
    stage = prepare(root)
    root = Path(root).resolve()
    print('X4_TISASREC_STAGE=' + str(stage), flush=True)
    state_path = stage / 'verification/driver.json'
    state = {'version': VERSION, 'stage': str(stage), 'status': 'RUNNING', 'jobs': []}
    write_json(state_path, state)
    process = None
    result = 1
    try:
        tasks = [('smoke', tag, 42) for tag in TAGS] + [('train', tag, seed) for tag in TAGS for seed in SEEDS]
        for mode, tag, seed in tasks:
            command = [sys.executable, '-u', str(stage / 'scripts/run_x4_tisasrec.py'), '--root', str(root), '--run',
                       '--stage', str(stage), '--mode', mode, '--dataset', tag, '--seed', str(seed)]
            log = stage / 'verification' / (mode + '_' + tag + '_seed' + str(seed) + '.log')
            record = {'mode': mode, 'dataset': tag, 'seed': seed, 'command': command, 'log': str(log), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            state['jobs'].append(record)
            write_json(state_path, state)
            with log.open('x', encoding='utf-8') as handle:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(line, end='', flush=True)
                record['exit_code'] = process.wait()
            process = None
            if record['exit_code']:
                raise RuntimeError('Failed ' + mode + ' ' + tag + ' seed' + str(seed))
            proof_path = verify_job_proof(stage, read_json(stage / INPUT_NAME), mode, tag, seed)
            record.update(proof_sha256=sha(proof_path), proof=str(proof_path), log_sha256=sha(log),
                          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(state_path, state)
        result = 0
    except BaseException as exc:
        state['error'] = type(exc).__name__ + ': ' + str(exc)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print('X4_ERROR=' + state['error'], flush=True)
    finally:
        state['status'] = 'ALL_TWO_SMOKES_AND_SIX_TRAININGS_COMPLETED' if result == 0 else 'STOPPED_INCOMPLETE'
        write_json(state_path, state)
        print('X4_DRIVER_STATUS=' + state['status'], flush=True)
        report = subprocess.run([sys.executable, str(stage / 'scripts/collect_x4_tisasrec.py'), '--stage', str(stage)])
        if report.returncode:
            result = report.returncode
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--stage', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=('smoke', 'train'), help=argparse.SUPPRESS)
    parser.add_argument('--dataset', choices=TAGS, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, choices=SEEDS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.run:
        print('PLAN_ONLY: ML1M + Beauty; two contract/model smokes first; then six sequential seed42/2026/3407 trainings; canonical + five fixed permutation evaluations; private metadata report. No files/data/GPU accessed.')
        return 0
    if args.stage:
        if args.mode is None or args.dataset is None or args.seed is None:
            parser.error('Internal stage requires mode/dataset/seed')
        child(args.root.resolve(), args.stage.resolve(), args.mode, args.dataset, args.seed)
        return 0
    return run(args.root)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('ERROR: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
