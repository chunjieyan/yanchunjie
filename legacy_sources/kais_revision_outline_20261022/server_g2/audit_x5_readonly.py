#!/usr/bin/env python3
"""Read-only CPU analytic baselines and target training-frequency diagnostics.

Default is a plan with no project file reads. Only --run opens sealed inputs.
Never imports a model library, accesses checkpoint files or changes input data.
"""
import argparse
import ast
from bisect import bisect_right
from collections import Counter
import csv
import datetime
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import statistics
import sys
import uuid
import zipfile

INPUTS_SHA256 = 'dfe843be42325008830b14fb7d3667b072a92cf53220bc1d750f0330b455bc11'
VERSION = 'x5_readonly_v1'
METRICS = ('ndcg10', 'hr10', 'hr20', 'recall50', 'mrr', 'mrr10', 'mean_rank')
SEEDS = (42, 2026, 3407)
DATASETS = {'ml1m': 'MovieLens-1M', 'beauty': 'Amazon_Beauty', 'sports': 'Amazon_Sports', 'toys': 'Amazon_Toys'}
GROUPS = ('overall', 'warm', 'cold')
MAX_FILE_BYTES = 256 * 1024 * 1024


@lru_cache(maxsize=32)
def harmonic_table(n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError('Harmonic table size must be a nonnegative integer')
    values = [0.0]
    for rank in range(1, n + 1):
        values.append(values[-1] + 1.0 / rank)
    return tuple(values)


def build_popularity_context(item_map, train_counts):
    catalog = set(item_map)
    if not catalog or any(item not in catalog for item in train_counts):
        raise ValueError('Training item is outside the frozen catalog or catalog is empty')
    frequencies = {}
    for item in catalog:
        count = train_counts.get(item, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('Training interaction counts must be nonnegative integers')
        frequencies[item] = count
    return {'frequencies': frequencies, 'sorted_frequencies': sorted(frequencies.values()),
            'frequency_histogram': Counter(frequencies.values()), 'harmonics': harmonic_table(len(catalog))}


def candidate_stats(target, history, context):
    frequencies = context['frequencies']
    if target not in frequencies:
        raise ValueError('Target is outside the frozen item map; cannot evaluate')
    unique_history = set(history)
    if any(item not in frequencies for item in unique_history):
        raise ValueError('History contains an item outside the frozen item map')
    masked = unique_history - {target}
    count = frequencies[target]
    candidates = len(frequencies) - len(masked)
    greater = len(frequencies) - bisect_right(context['sorted_frequencies'], count)
    ties = context['frequency_histogram'][count]
    greater -= sum(frequencies[item] > count for item in masked)
    ties -= sum(frequencies[item] == count for item in masked)
    if not (candidates >= 1 and 0 <= greater < candidates and 1 <= ties <= candidates - greater):
        raise ValueError('Invalid eligible candidate/tie counts')
    return {'candidate_count': candidates, 'target_train_count': count, 'pop_gt': greater, 'pop_tie': ties,
            'history_unique_items': len(unique_history), 'masked_unique_items': len(masked),
            'target_in_history': int(target in unique_history), 'target_group': 'warm' if count > 0 else 'cold'}


def analytic_random(candidate_count, harmonics=None):
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 1:
        raise ValueError('Random baseline requires a positive integer candidate count')
    c = candidate_count
    h = harmonic_table(c) if harmonics is None else harmonics
    if len(h) <= c:
        raise ValueError('Harmonic table does not cover the catalog')
    return {'ndcg10': math.fsum(1.0 / math.log2(rank + 1) for rank in range(1, min(10, c) + 1)) / c,
            'hr10': min(10, c) / c, 'hr20': min(20, c) / c, 'recall50': min(50, c) / c,
            'mrr': h[c] / c, 'mrr10': h[min(10, c)] / c, 'mean_rank': (c + 1) / 2.0}


def analytic_popularity(greater, ties, harmonics=None):
    if (isinstance(greater, bool) or isinstance(ties, bool) or not isinstance(greater, int)
            or not isinstance(ties, int) or greater < 0 or ties < 1):
        raise ValueError('Popularity requires G>=0 and T>=1 as integer eligible counts')
    g, t = greater, ties
    h = harmonic_table(g + t) if harmonics is None else harmonics
    if len(h) <= g + t:
        raise ValueError('Harmonic table does not cover the target tie block')
    hit = lambda k: max(0, min(t, k - g)) / t
    end = min(g + t, 10)
    return {'ndcg10': math.fsum(1.0 / math.log2(rank + 1) for rank in range(g + 1, end + 1)) / t,
            'hr10': hit(10), 'hr20': hit(20), 'recall50': hit(50),
            'mrr': (h[g + t] - h[g]) / t,
            'mrr10': (h[end] - h[g]) / t if end > g else 0.0,
            'mean_rank': g + (t + 1) / 2.0}


def rank_metrics(rank):
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError('Saved target rank must be a positive integer')
    return {'ndcg10': 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
            'hr10': float(rank <= 10), 'hr20': float(rank <= 20), 'recall50': float(rank <= 50),
            'mrr': 1.0 / rank, 'mrr10': 1.0 / rank if rank <= 10 else 0.0, 'mean_rank': float(rank)}


def mean_metrics(rows):
    if not rows:
        return {'samples': 0, **{metric: None for metric in METRICS}}
    return {'samples': len(rows), **{metric: math.fsum(row[metric] for row in rows) / len(rows) for metric in METRICS}}


def grouped_metrics(rows, values):
    if len(rows) != len(values):
        raise ValueError('Group labels and metric rows have different sample counts')
    return {group: mean_metrics([value for row, value in zip(rows, values)
                if group == 'overall' or row['target_group'] == group]) for group in GROUPS}


def plain(path):
    path = Path(path)
    if path.suffix.lower() in ('.pt', '.pth', '.ckpt', '.safetensors'):
        raise ValueError('Checkpoint access, including stat, is forbidden')
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Use an absolute project path without ..')
    for part in (path,) + tuple(path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Symbolic link/reparse point rejected: ' + str(part))
    return path.resolve(strict=False)


def safe_file(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or PurePosixPath(relative).is_absolute()
            or any(p in ('', '.', '..') or p.startswith('.') for p in relative.split('/'))):
        raise ValueError('Unsafe project-relative file path')
    # Reject weight extensions BEFORE even statting their paths.
    if PurePosixPath(relative).suffix.lower() not in ('.csv', '.json', '.py', '.log', '.txt', '.md'):
        raise ValueError('File type forbidden for this read-only audit: ' + relative)
    root = plain(root)
    path = plain(root.joinpath(*relative.split('/')))
    path.relative_to(root)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Expected an ordinary file: ' + relative)
    return path


def signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def read_verified(root, relative, expected=None, limit=MAX_FILE_BYTES):
    path = safe_file(root, relative)
    before = path.stat()
    if before.st_size > limit:
        raise ValueError('File exceeds bounded audit size: ' + relative)
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    parts, digest, size = [], hashlib.sha256(), 0
    with os.fdopen(os.open(str(path), flags), 'rb') as stream:
        if signature(before) != signature(os.fstat(stream.fileno())):
            raise ValueError('File changed before read: ' + relative)
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(block)
            if size > limit:
                raise ValueError('File grew beyond bounded audit size: ' + relative)
            digest.update(block); parts.append(block)
        after = os.fstat(stream.fileno())
    if signature(before) != signature(after) or signature(before) != signature(safe_file(root, relative).stat()):
        raise ValueError('File changed during read: ' + relative)
    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        raise ValueError('SHA256 mismatch; recover the audited original file: ' + relative)
    return b''.join(parts), actual


def csv_records(blob, label, required):
    reader = csv.DictReader(io.StringIO(blob.decode('utf-8-sig'), newline=''))
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or not set(required).issubset(reader.fieldnames):
        raise ValueError(label + ': missing/duplicate CSV columns; required=' + ','.join(required))
    for line, row in enumerate(reader, 2):
        if None in row or any(row.get(key) is None for key in required):
            raise ValueError(label + ': malformed CSV row ' + str(line))
        yield line, row


def integer(value, label):
    if isinstance(value, bool) or not re.fullmatch(r'[0-9]+', str(value)):
        raise ValueError(label + ': expected a nonnegative integer')
    return int(value)


def identifier(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + ': empty identifier')
    return value


def read_item_map(blob, expected_items, label):
    mapping = {}
    for line, row in csv_records(blob, label, ('raw_item_id', 'item_idx')):
        raw = identifier(row['raw_item_id'], label + ' row ' + str(line))
        index = integer(row['item_idx'], label + ' item_idx')
        if raw in mapping:
            raise ValueError(label + ': duplicate raw item ID')
        mapping[raw] = index
    if len(mapping) != expected_items or set(mapping.values()) != set(range(1, expected_items + 1)):
        raise ValueError(label + ': frozen item map size or bijection differs')
    return mapping


def read_training_counts(blob, item_map, label):
    counts, users, nrows = Counter(), Counter(), 0
    for line, row in csv_records(blob, label, ('user_id', 'item_id')):
        user = identifier(row['user_id'], label + ' user row ' + str(line))
        item = identifier(row['item_id'], label + ' item row ' + str(line))
        if item not in item_map:
            raise ValueError(label + ': training item outside frozen mapping at row ' + str(line))
        counts[item] += 1; users[user] += 1; nrows += 1
    if not nrows:
        raise ValueError(label + ': empty training interactions')
    return counts, users, nrows


def read_panel(blob, expected, context, users, label):
    rows, ids = [], set()
    for line, row in csv_records(blob, label, ('sample_id', 'user_id', 'target_item', 'sequence_items')):
        sid, user, target = (identifier(row[k], label + ' row ' + str(line)) for k in ('sample_id', 'user_id', 'target_item'))
        if sid in ids:
            raise ValueError(label + ': duplicate sample_id at row ' + str(line))
        ids.add(sid)
        values = json.loads(row['sequence_items'])
        if (not isinstance(values, list) or not 1 <= len(values) <= 50
                or any(isinstance(x, (dict, list, bool)) or x is None for x in values)):
            raise ValueError(label + ': invalid retained history at row ' + str(line))
        history = tuple(str(x) for x in values)
        try:
            stats = candidate_stats(target, history, context)
        except ValueError as exc:
            raise ValueError(label + ' row ' + str(line) + ': ' + str(exc)) from None
        rows.append({'sample_id': sid, 'user_id': user, 'target_item': target, **stats,
            'user_seen_in_training': int(user in users), 'history_length': len(history),
            'history_unique_items_absent_from_training': sum(context['frequencies'][x] == 0 for x in set(history)),
            '_mask': set(history) - {target}})
    if len(rows) != expected:
        raise ValueError(label + ': sample count ' + str(len(rows)) + ' differs from sealed ' + str(expected))
    return rows


def saved_test_metrics(metrics):
    test = metrics.get('test')
    if not isinstance(test, dict):
        raise ValueError('Saved metrics require a test object')
    if 'overall' in test:
        value = test['overall']
        keys = {'samples': 'N', 'ndcg10': 'NDCG@10', 'hr10': 'HR@10', 'hr20': 'HR@20', 'mean_rank': 'MeanRank'}
    elif all(k in test for k in ('samples', 'ndcg10', 'hr10', 'hr20', 'mean_rank')):
        value = test
        keys = {k: k for k in ('samples', 'ndcg10', 'hr10', 'hr20', 'mean_rank')}
    else:
        raise ValueError('Saved test metric schema is neither audited overall-N nor flat-samples format')
    if not isinstance(value, dict) or any(k not in value for k in keys.values()):
        raise ValueError('Saved test metrics miss a required audited field')
    result = {'samples': integer(value[keys['samples']], 'Saved test N')}
    for key in ('ndcg10', 'hr10', 'hr20', 'mean_rank'):
        number = value[keys[key]]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError('Saved metric is nonfinite or nonnumeric: ' + key)
        result[key] = float(number)
    return result


def validate_predictions(blob, rows, mapping, label):
    panel = {row['sample_id']: row for row in rows}
    seen, ranks, top10_hit_disagreements = set(), {}, 0
    for line, pred in csv_records(blob, label, ('sample_id', 'user_id', 'target_item', 'target_rank', 'top10_item_idx', 'top10_raw_item_id')):
        sid = pred['sample_id']
        if sid in seen or sid not in panel:
            raise ValueError(label + ': duplicate or unexpected sample at row ' + str(line))
        seen.add(sid); row = panel[sid]
        if pred['user_id'] != row['user_id'] or pred['target_item'] != row['target_item']:
            raise ValueError(label + ': saved user/target identity differs at row ' + str(line))
        rank = integer(pred['target_rank'], label + ' target_rank')
        if not 1 <= rank <= row['candidate_count']:
            raise ValueError(label + ': rank outside eligible candidate universe')
        indexes, raw = json.loads(pred['top10_item_idx']), json.loads(pred['top10_raw_item_id'])
        if (not isinstance(indexes, list) or not isinstance(raw, list) or len(indexes) != min(10, row['candidate_count'])
                or len(raw) != len(indexes) or any(isinstance(x, bool) or not isinstance(x, int) for x in indexes)):
            raise ValueError(label + ': malformed ordered Top10 lists')
        raw = [str(x) for x in raw]
        if (len(set(indexes)) != len(indexes) or len(set(raw)) != len(raw)
                or any(item not in mapping or mapping[item] != index for item, index in zip(raw, indexes))
                or any(item in row['_mask'] for item in raw)):
            raise ValueError(label + ': Top10 mapping, uniqueness or history masking differs')
        # Existing model ranks use their saved strict-greater rule. Exact-score
        # ties can make rank-HR differ from ordered Top10 membership; diagnose,
        # rather than silently replacing either saved result.
        top10_hit_disagreements += int((rank <= 10) != (row['target_item'] in raw))
        ranks[sid] = rank
    if seen != set(panel):
        raise ValueError(label + ': saved predictions do not cover the exact frozen test panel')
    return [ranks[row['sample_id']] for row in rows], top10_hit_disagreements


def panel_diagnostics(rows):
    n = len(rows)
    warm = sum(row['target_group'] == 'warm' for row in rows)
    return {'samples': n, 'warm_targets': warm, 'cold_targets': n - warm,
        'warm_fraction': warm / n if n else None, 'users_absent_from_training': sum(not row['user_seen_in_training'] for row in rows),
        'target_in_retained_history': sum(row['target_in_history'] for row in rows),
        'samples_with_history_item_absent_from_training': sum(row['history_unique_items_absent_from_training'] > 0 for row in rows),
        'candidate_count_min': min(row['candidate_count'] for row in rows) if rows else None,
        'candidate_count_max': max(row['candidate_count'] for row in rows) if rows else None,
        'unknown_target_or_history_mapping_items': 0}


def three_seed_statistics(model_results):
    output = {}
    for method in ('Control', 'LBSP'):
        selected = [r for r in model_results if r['method'] == method]
        if sorted(r['seed'] for r in selected) != list(SEEDS):
            raise ValueError('Every method must retain all three distinct audited seeds')
        output[method] = {}
        for group in GROUPS:
            sample_counts = {r['groups'][group]['samples'] for r in selected}
            if len(sample_counts) != 1:
                raise ValueError('Training seeds do not use the same warm/cold panel')
            n = next(iter(sample_counts))
            output[method][group] = {'samples_per_seed': n, 'training_seeds': list(SEEDS),
                'mean': {k: statistics.mean(r['groups'][group][k] for r in selected) if n else None for k in METRICS},
                'sample_sd_ddof1': {k: statistics.stdev(r['groups'][group][k] for r in selected) if n else None for k in METRICS}}
    return output


def core_inventory(inputs):
    if (inputs.get('version') != VERSION or inputs.get('training_seeds') != list(SEEDS)
            or set(inputs.get('datasets', {})) != set(DATASETS)):
        raise ValueError('Sealed input schema must contain four datasets and all three training seeds')
    core = dict(inputs['data_sha256'])
    expected_data, model_ids = set(), set()
    def add(path, digest):
        if not re.fullmatch(r'[0-9a-f]{64}', str(digest)) or (path in core and core[path] != digest):
            raise ValueError('Conflicting or invalid core SHA256 descriptor')
        core[path] = digest
    for tag, dataset in inputs['datasets'].items():
        if dataset['dataset'] != DATASETS[tag]:
            raise ValueError('Dataset identity differs: ' + tag)
        expected_data.update(dataset[k] for k in ('train_path', 'validation_path', 'test_path'))
        add(dataset['map_reference']['path'], dataset['map_reference']['sha256'])
        identity = {(m['method'], m['seed']) for m in dataset['models']}
        if len(dataset['models']) != 6 or identity != {(method, seed) for method in ('Control', 'LBSP') for seed in SEEDS}:
            raise ValueError(tag + ': require Control/LBSP for all three seeds')
        for model in dataset['models']:
            if model['id'] in model_ids or not re.fullmatch(r'[a-z0-9_]+', model['id']):
                raise ValueError('Duplicate or unsafe model ID')
            model_ids.add(model['id'])
            if set(model['files_sha256']) != {'config.json', 'metrics.json', 'test_per_sample_predictions.csv'}:
                raise ValueError('Each model requires config, metrics and canonical prediction hashes')
            for name, digest in model['files_sha256'].items():
                add(model['model_dir'] + '/' + name, digest)
    if set(inputs['data_sha256']) != expected_data or len(expected_data) != 12 or len(core) != 88:
        raise ValueError('Expected exactly 12 input CSVs, 4 maps and 72 saved model files')
    for digest in core.values():
        if not re.fullmatch(r'[0-9a-f]{64}', str(digest)):
            raise ValueError('Invalid sealed core SHA256')
    return core


def analyze_dataset(root, tag, dataset, inputs):
    reference = dataset['map_reference']
    mapping = read_item_map(read_verified(root, reference['path'], reference['sha256'])[0], dataset['expected_items'], tag + ' item map')
    counts, users, train_rows = read_training_counts(read_verified(root, dataset['train_path'], inputs['data_sha256'][dataset['train_path']])[0], mapping, tag + ' train')
    context = build_popularity_context(mapping, counts)
    panels = {}
    for phase, expected in (('validation', dataset['expected_validation']), ('test', dataset['expected_test'])):
        path = dataset[phase + '_path']
        panels[phase] = read_panel(read_verified(root, path, inputs['data_sha256'][path])[0], expected, context, users, tag + ' ' + phase)
    rows = panels['test']
    random_values = [analytic_random(row['candidate_count'], context['harmonics']) for row in rows]
    popularity_values = [analytic_popularity(row['pop_gt'], row['pop_tie'], context['harmonics']) for row in rows]
    baseline = {'UniformRandomAnalytic': grouped_metrics(rows, random_values),
                'PopularityUniformTieAnalytic': grouped_metrics(rows, popularity_values)}
    for row, random_result, pop_result in zip(rows, random_values, popularity_values):
        row.update({'random_' + k: v for k, v in random_result.items()})
        row.update({'popularity_' + k: v for k, v in pop_result.items()})
    models = []
    for model in dataset['models']:
        blobs = {name: read_verified(root, model['model_dir'] + '/' + name, pin)[0] for name, pin in model['files_sha256'].items()}
        config, metrics = (json.loads(blobs[name].decode('utf-8-sig')) for name in ('config.json', 'metrics.json'))
        if (config.get('dataset') != dataset['dataset'] or metrics.get('dataset') != dataset['dataset']
                or config.get('seed') != model['seed'] or metrics.get('seed') != model['seed']):
            raise ValueError(model['id'] + ': saved config/metric dataset or seed identity differs')
        ranks, disagreements = validate_predictions(blobs['test_per_sample_predictions.csv'], rows, mapping, model['id'])
        groups = grouped_metrics(rows, [rank_metrics(rank) for rank in ranks])
        saved = saved_test_metrics(metrics)
        if (saved['samples'] != len(rows) or any(not math.isclose(groups['overall'][key], saved[key], rel_tol=0, abs_tol=1e-10)
                for key in ('ndcg10', 'hr10', 'hr20', 'mean_rank'))):
            raise ValueError(model['id'] + ': rank-recomputed canonical accuracy differs from saved test metrics')
        for row, rank in zip(rows, ranks):
            row[model['id'] + '_rank'] = rank
        models.append({'id': model['id'], 'method': model['method'], 'seed': model['seed'], 'groups': groups,
            'canonical_saved_accuracy_reproduced': True, 'ordered_top10_map_and_mask_verified': True,
            'rank_hit10_vs_top10_membership_disagreements': disagreements,
            'saved_metric_schema': 'overall-N' if 'overall' in metrics['test'] else 'flat-samples',
            'files_sha256': model['files_sha256']})
    diagnostics = {'training_rows': train_rows, 'training_users': len(users),
        'training_items_with_positive_interaction_count': len(counts), 'catalog_items': len(mapping),
        'validation': panel_diagnostics(panels['validation']), 'test': panel_diagnostics(rows)}
    # Keep only derived counts/IDs/ranks. Neither sequence histories nor masks
    # are emitted to any report or archive.
    for panel in panels.values():
        for row in panel:
            row.pop('_mask')
    return {'dataset': dataset['dataset'], 'diagnostics': diagnostics, 'analytic_baselines': baseline,
            'models': models, 'model_three_seed': three_seed_statistics(models)}, panels


PROTOCOL_KEYS = {
    'hiddenunits': 'hidden_size', 'hiddensize': 'hidden_size', 'hiddendim': 'hidden_size',
    'numheads': 'num_heads', 'nheads': 'num_heads', 'dropout': 'dropout', 'dropoutrate': 'dropout',
    'patience': 'patience', 'maxlen': 'max_len', 'maxlength': 'max_len', 'lr': 'learning_rate',
    'learningrate': 'learning_rate', 'batchsize': 'training_batch_size', 'trainingbatchsize': 'training_batch_size',
    'maxepochs': 'max_epochs', 'numepochs': 'max_epochs'}


def protocol_evidence(blob, relative):
    text = blob.decode('utf-8-sig')
    result = []
    def visit(value, prefix=''):
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = re.sub(r'[^a-z0-9]', '', str(key).lower())
                if (normalized in PROTOCOL_KEYS and isinstance(child, (int, float, str)) and not isinstance(child, bool)
                        and (not isinstance(child, float) or math.isfinite(child))
                        and (not isinstance(child, str) or len(child) <= 100)):
                    result.append({'field': PROTOCOL_KEYS[normalized], 'value': child, 'location': prefix + str(key)})
                if isinstance(child, dict):
                    visit(child, prefix + str(key) + '.')
    suffix = PurePosixPath(relative).suffix.lower()
    if suffix == '.json':
        visit(json.loads(text))
    elif suffix == '.py':
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        visit({target.id: value}, 'line' + str(node.lineno) + '.')
    return result


def optional_inventory(root, entries):
    records, copies = [], {}
    for entry in entries:
        relative = entry['path']
        record = {'path': relative, 'local_reference_sha256': entry['local_reference_sha256'], 'optional': True}
        try:
            blob, digest = read_verified(root, relative, limit=8 * 1024 * 1024)
            text = blob.decode('utf-8-sig')
            # Optional copies are text-only provenance. Do not package secrets
            # or serialized interaction histories if an old file has changed.
            if re.search(r'(?i)\b(?:api_key|password|secret|access_token)[\x22\x27]?\s*[:=]\s*[\x22\x27][^\x22\x27\r\n]{8,}', text):
                raise ValueError('Possible credential assignment; optional copy omitted')
            if PurePosixPath(relative).suffix.lower() == '.json' and re.search(r'\x22(?:sequence_items|history_items)\x22\s*:', text):
                raise ValueError('Possible interaction histories; optional copy omitted')
            record.update(status='MATCH' if digest == entry['local_reference_sha256'] else 'CHANGED', sha256=digest,
                          bytes=len(blob), copied=True, protocol_evidence=protocol_evidence(blob, relative))
            copies['optional_legacy/' + relative] = blob
        except FileNotFoundError:
            record.update(status='MISSING', copied=False)
        except (OSError, ValueError, SyntaxError, UnicodeError) as exc:
            record.update(status='REJECTED_OR_UNREADABLE', copied=False, reason=str(exc)[:240])
        records.append(record)
    return records, copies


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Expected nonempty derived report rows')
    fields = list(rows[0])
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def run_audit(root, inputs_path):
    if Path(inputs_path).suffix.lower() != '.json':
        raise ValueError('The sealed input plan must be a JSON file')
    root, inputs_path = plain(root), plain(inputs_path)
    if not root.is_dir():
        raise ValueError('Project root is not an ordinary directory')
    raw = inputs_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUTS_SHA256:
        raise ValueError('Input JSON differs from the reviewed X5 seal; upload the matching JSON')
    inputs = json.loads(raw.decode('utf-8-sig'))
    core = core_inventory(inputs)
    print('VERIFYING_CORE_FILES=' + str(len(core)), flush=True)
    for relative, digest in core.items():
        read_verified(root, relative, digest)
    results, panels = {}, {}
    for tag, dataset in inputs['datasets'].items():
        print('ANALYZING_CPU_READONLY=' + dataset['dataset'], flush=True)
        results[tag], panels[tag] = analyze_dataset(root, tag, dataset, inputs)
    optional, optional_copies = optional_inventory(root, inputs.get('optional_legacy_files', []))
    # A second exact hash check binds all conclusions to unchanged core files.
    # Any core hash/panel/metric failure occurs before a report ZIP is created.
    for relative, digest in core.items():
        read_verified(root, relative, digest)
    now = datetime.datetime.now(datetime.timezone.utc)
    stage = root / ('kais_x5_readonly_v1_' + now.strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:12])
    stage.mkdir(exist_ok=False)
    for directory in ('derived', 'manifests', 'optional_legacy'):
        (stage / directory).mkdir(exist_ok=False)
    status = 'READONLY_BASELINES_AND_DIAGNOSTICS_COMPLETED'
    limits = [
        'Analytic baselines use the frozen full catalog after unique-history masking, preserving the target; no candidates are sampled.',
        'Popularity counts every training interaction only. Uniform ordering within target-frequency ties is averaged exactly; optimistic strict-greater tie ranks are not used for this baseline.',
        'Random/Popularity are analytic expectations without invented training-seed SD. Model summaries retain all three saved seeds and use sample SD (ddof=1).',
        'Warm/cold means target training-interaction count positive/zero. A cold target is not necessarily an untrained embedding: full-softmax CE negatives can update it.',
        'Warm/cold diagnostics are descriptive and do not establish a sole cause of low accuracy.',
        'Model metrics are recomputed from saved ranks with the existing rank convention. Ordered Top10 identity/mapping/masking is checked separately; any rank-hit/Top10-membership disagreement is reported.',
        'Recall@50 equals Hit@50 for these single-relevant-item panels. Full MRR is mean(1/rank); MRR@10 is zero beyond rank10.',
        'Optional legacy calibration evidence may be absent or changed. Its inventory does not complete current-protocol calibration, and this status does not mean every X5 requirement is complete.',
        'No model training/inference or checkpoint stat/read was performed. No raw histories, interaction files, item maps or original prediction files are included in the report ZIP.']
    tool_blob = Path(__file__).read_bytes()
    report = {'version': VERSION, 'status': status, 'created_utc': now.isoformat(), 'root': str(root), 'stage': str(stage),
        'input_manifest_sha256': INPUTS_SHA256, 'core_sha256': core, 'core_hashes_verified_before_after': True,
        'tool_sha256': hashlib.sha256(tool_blob).hexdigest(),
        'baseline_rules': inputs['baseline_rules'], 'datasets': results, 'optional_legacy_inventory': optional,
        'protocol_comparison_profiles': {'legacy_reference_profile': {'hidden_size': 50, 'num_heads': 1, 'dropout': 0.5, 'patience': 25},
            'current_step38_profile': {'hidden_size': 64, 'num_heads': 2, 'dropout': 0.2, 'patience': 15},
            'interpretation': 'Compare observed optional evidence with these distinct profiles; no protocol equivalence or new calibration-success claim.'},
        'training_performed': False, 'inference_performed': False, 'checkpoints_accessed_or_statted': False,
        'raw_histories_in_archive': False, 'all_x5_requirements_completed': False, 'limitations': limits}
    write_json(stage / 'x5_readonly_results.json', report)
    write_json(stage / 'manifests/x5_readonly_inputs.json', inputs)
    with (stage / 'manifests/audit_x5_readonly.py').open('xb') as handle:
        handle.write(tool_blob)
    write_csv(stage / 'manifests/CORE_SHA256.csv', [{'path': p, 'sha256': h} for p, h in core.items()])
    baselines, model_rows, aggregate = [], [], []
    for tag, result in results.items():
        for phase, rows in panels[tag].items():
            write_csv(stage / 'derived' / (tag + '_' + phase + '.csv'), rows)
        for method, groups in result['analytic_baselines'].items():
            for group, metrics in groups.items():
                baselines.append({'dataset': result['dataset'], 'method': method, 'group': group, **metrics,
                                  'training_seed_sd': 'not_applicable_analytic_expectation'})
        for model in result['models']:
            for group, metrics in model['groups'].items():
                model_rows.append({'dataset': result['dataset'], 'model_id': model['id'], 'method': model['method'],
                                   'seed': model['seed'], 'group': group, **metrics})
        for method, groups in result['model_three_seed'].items():
            for group, value in groups.items():
                aggregate.append({'dataset': result['dataset'], 'method': method, 'group': group, 'seeds': '42,2026,3407',
                    'samples_per_seed': value['samples_per_seed'], **{k + '_mean': v for k, v in value['mean'].items()},
                    **{k + '_sample_sd': v for k, v in value['sample_sd_ddof1'].items()}})
    write_csv(stage / 'analytic_baselines.csv', baselines)
    write_csv(stage / 'saved_model_per_seed.csv', model_rows)
    write_csv(stage / 'saved_model_three_seed.csv', aggregate)
    for relative, blob in optional_copies.items():
        path = stage.joinpath(*relative.split('/'))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(blob)
    lines = ['# Read-only analytic baselines and target-frequency diagnostics', '', 'Status: ' + status, '',
             '| Dataset | Test N | Warm | Cold | Unknown map IDs |', '|---|---:|---:|---:|---:|']
    for result in results.values():
        d = result['diagnostics']['test']
        lines.append('| %s | %d | %d | %d | 0 |' % (result['dataset'], d['samples'], d['warm_targets'], d['cold_targets']))
    lines += ['', 'All 24 saved model canonical metrics reproduce from the identical test panels. '
        'See analytic_baselines.csv and saved_model_three_seed.csv for the complete seven-metric results.', '',
        'Popularity target ties occupy ranks G+1 through G+T and are averaged uniformly. '
        'Analytic Random averages the target over all C eligible ranks. No Monte Carlo seeds or fake SD are introduced.', '',
        '## Interpretation and remaining scope', ''] + ['- ' + text for text in limits]
    lines += ['', 'This private report contains derived sample/user/target identifiers and ranks; it is not a public data release.', '']
    with (stage / 'SUMMARY.md').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(lines))
    files = sorted(p for p in stage.rglob('*') if p.is_file())
    write_csv(stage / 'REPORT_SHA256.csv', [{'path': p.relative_to(stage).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                           'bytes': p.stat().st_size} for p in files])
    bundle = stage / 'x5_readonly_reports.zip'
    with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(p for p in stage.rglob('*') if p.is_file() and p != bundle):
            archive.write(path, path.relative_to(stage).as_posix())
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    with bundle.with_name(bundle.name + '.sha256.txt').open('x', encoding='ascii') as handle:
        handle.write(digest + '  ' + bundle.name + '\n')
    return {'stage': str(stage), 'status': status, 'zip_path': str(bundle), 'sha256': digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).with_name('x5_readonly_inputs_20261025.json'))
    parser.add_argument('--run', action='store_true', help='Read sealed inputs and create a new CPU-only report stage')
    args = parser.parse_args()
    if not args.run:
        print('STATUS=PLAN_ONLY_NO_PROJECT_READS')
        print('PLAN=Verify 12 input CSVs, 4 item maps and 72 saved metadata/prediction files; compute analytic Random/Popularity and saved-model diagnostics; inventory optional legacy calibration evidence.')
        print('RUN_COMMAND=python audit_x5_readonly.py --root ' + str(args.root) + ' --run')
        print('X5_READONLY_REPORT_ZIP=NOT_CREATED')
        return 0
    result = run_audit(args.root, args.inputs)
    print('WORKDIR=' + result['stage'])
    print('X5_READONLY_REPORT_ZIP=' + result['zip_path'])
    print('STATUS=' + result['status'])
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, csv.Error) as exc:
        print('READONLY_AUDIT_FAILED=' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        print('X5_READONLY_REPORT_ZIP=NOT_CREATED')
        print('STATUS=CORE_AUDIT_FAILED_NO_BUNDLE')
        sys.exit(2)
