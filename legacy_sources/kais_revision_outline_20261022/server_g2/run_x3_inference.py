"""Finish X3 Shuffle stability and five-group Control score ensembles.

Plan-only by default. --run performs inference only on existing checkpoints.
"""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
import csv
import math
import shutil
import subprocess
import sys
import uuid

VERSION = 'x3_inference_v1'
INPUT_NAME = 'x3_inference_inputs_20261026.json'
INPUT_SHA256 = '2d0e5cbb7f537fb9989b30fbf60695f8bfa5819f0d7f409ebe9d3786dd9f46a4'
TOOLS = ('run_x3_inference.py', 'collect_x3_inference.py', 'x3_inference_adapter.py',
         'x3_inference_protocol.py', 'x3_control_model.py')
TAGS = ('ml1m', 'beauty', 'sports', 'toys')
SEEDS = (42, 2026, 3407)
PERMS = (101, 202, 303, 404, 505)
METHODS = ('shuffle', 'ensemble')
GROUPS = (PERMS, (1101, 1202, 1303, 1404, 1505), (2101, 2202, 2303, 2404, 2505),
          (3101, 3202, 3303, 3404, 3505), (4101, 4202, 4303, 4404, 4505))
PREFIX = 'kais_x3_inference_v1_'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def safe(root, relative):
    root = Path(root).resolve()
    path = root / relative
    path.resolve().relative_to(root)
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refusing an empty prediction/identity table')
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_root_files(root, inputs, required=None):
    paths = inputs['root_files_sha256'] if required is None else {p: inputs['root_files_sha256'][p] for p in required}
    for relative, expected in paths.items():
        if sha(safe(root, relative)) != expected:
            raise ValueError('Reviewed input/reference changed: ' + relative)


def verify_stage(root, stage):
    root, stage = Path(root).resolve(), Path(stage).resolve()
    if stage.parent != root or not stage.name.startswith(PREFIX):
        raise ValueError('Expected an isolated direct-child X3 inference stage')
    manifest = read_json(stage / 'manifest.json')
    if manifest['version'] != VERSION or manifest['root'] != str(root) or manifest['stage'] != str(stage):
        raise ValueError('Stage identity mismatch')
    for relative, expected in dict(manifest['tools_sha256'], **manifest['references_sha256']).items():
        if sha(safe(stage, relative)) != expected:
            raise ValueError('Stage tool/reference changed: ' + relative)
    if sha(stage / INPUT_NAME) != INPUT_SHA256:
        raise ValueError('Input seal mismatch')
    return manifest, read_json(stage / INPUT_NAME)


def prepare(root):
    root = Path(root).resolve()
    here = Path(__file__).resolve().parent
    if not root.is_dir() or sha(here / INPUT_NAME) != INPUT_SHA256:
        raise ValueError('Missing project or changed reviewed input seal')
    inputs = read_json(here / INPUT_NAME)
    if inputs['version'] != VERSION or inputs['training_seeds'] != list(SEEDS) or inputs['groups'] != [list(g) for g in GROUPS]:
        raise ValueError('Protocol/seeds changed')
    for name, expected in inputs['helper_sha256'].items():
        if sha(here / name) != expected:
            raise ValueError('Reviewed helper changed: ' + name)
    verify_root_files(root, inputs)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:10]
    stage = root / (PREFIX + stamp)
    stage.mkdir()
    for name in ('scripts', 'references', 'outputs', 'verification'):
        (stage / name).mkdir()
    tools = {}
    for name in TOOLS:
        shutil.copyfile(here / name, stage / 'scripts' / name)
        tools['scripts/' + name] = sha(stage / 'scripts' / name)
    shutil.copyfile(here / INPUT_NAME, stage / INPUT_NAME)
    tools[INPUT_NAME] = sha(stage / INPUT_NAME)
    references = {}
    for relative, record in inputs['references'].items():
        target = safe(stage, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe(root, record['root_path']), target)
        if sha(target) != record['sha256']:
            raise ValueError('Reference changed during copy')
        references[relative] = record['sha256']
    manifest = {'version': VERSION, 'root': str(root), 'stage': str(stage), 'input_sha256': INPUT_SHA256,
                'tools_sha256': tools, 'references_sha256': references,
                'root_files_sha256': inputs['root_files_sha256'], 'protocol': inputs['protocol'],
                'tasks': [{'method': method, 'tag': tag, 'seed': seed} for method in METHODS for tag in TAGS for seed in SEEDS]}
    write_json(stage / 'manifest.json', manifest)
    return stage


def metrics(ranks):
    if not ranks or any(not isinstance(r, int) or r < 1 for r in ranks):
        raise ValueError('Invalid target ranks')
    n = len(ranks)
    return {'ndcg10': math.fsum(1 / math.log2(r + 1) for r in ranks if r <= 10) / n,
            'hr10': sum(r <= 10 for r in ranks) / n, 'hr20': sum(r <= 20 for r in ranks) / n,
            'recall50': sum(r <= 50 for r in ranks) / n, 'mrr': math.fsum(1 / r for r in ranks) / n,
            'mrr10': math.fsum(1 / r for r in ranks if r <= 10) / n, 'mean_rank': sum(ranks) / n}


def parse_panel(path, mapping):
    rows = read_csv(path)
    if not rows or len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Empty panel or duplicate sample IDs')
    for row in rows:
        raw = [str(item) for item in json.loads(row['sequence_items'])]
        events = [int(key) for key in json.loads(row['sequence_raw_lines'])]
        k = int(row.get('last_bucket_size') or row['last_history_bucket_size'])
        k = min(k, len(raw))
        if not 1 <= k <= len(raw) <= 50 or len(raw) != len(events) or len(set(events)) != len(events):
            raise ValueError('History or last-bucket event identities invalid')
        if row['target_item'] not in mapping or any(item not in mapping for item in raw):
            raise ValueError('Item outside frozen catalog')
        if 'target_raw_line' in row and int(row['target_raw_line']) in events:
            raise ValueError('Target event leaked into history')
        row.update(_raw=raw, _events=events, _k=k, _items=[mapping[item] for item in raw], _target=mapping[row['target_item']])
    return rows


def load_panels(root, stage, plan, run):
    mapping_rows = read_csv(safe(stage, run['reference_map']))
    mapping = {row['raw_item_id']: int(row['item_idx']) for row in mapping_rows}
    if len(mapping) != plan['expected_items'] or sorted(mapping.values()) != list(range(1, len(mapping) + 1)):
        raise ValueError('Mapping differs from complete frozen catalog')
    original = parse_panel(safe(root, plan['original_path']), mapping)
    if len(original) != plan['expected_samples']:
        raise ValueError('Canonical panel sample count differs')
    ids = [row['sample_id'] for row in original]
    views = {}
    for seed in PERMS:
        rows = parse_panel(safe(root, plan['permutation_paths'][str(seed)]), mapping)
        lookup = {row['sample_id']: row for row in rows}
        if set(lookup) != set(ids):
            raise ValueError('Permutation population differs')
        rows = [lookup[sample] for sample in ids]
        for base, row in zip(original, rows):
            if any(row[key] != base[key] for key in ('user_id', 'target_item')) or row['_k'] != base['_k']:
                raise ValueError('Permutation sample/target/bucket identity changed')
            for key in ('target_raw_line', 'target_timestamp', 'target_bucket'):
                if key in base and str(row.get(key)) != str(base[key]):
                    raise ValueError('Permutation changed target metadata')
            k = base['_k']
            a, b = list(zip(base['_raw'], base['_events'])), list(zip(row['_raw'], row['_events']))
            if a[:-k] != b[:-k] or collections.Counter(a[-k:]) != collections.Counter(b[-k:]):
                raise ValueError('Permutation changed prefix or joint item/event multiset')
        views[seed] = rows
    labels = []
    for i, row in enumerate(original):
        histories = [views[seed][i]['_raw'] for seed in PERMS]
        distinct = len({tuple(history) for history in histories})
        label = {'sample_id': row['sample_id'], 'user_id': row['user_id'], 'target_item': row['target_item'],
                 'last_bucket_size': row['_k'], 'affected': int(distinct >= 2), 'distinct_sequences_5perm': distinct,
                 'original_sequence_sha256': sequence_hash(row['_raw'])}
        label.update({'perm_seed' + str(seed) + '_sequence_sha256': sequence_hash(history) for seed, history in zip(PERMS, histories)})
        labels.append(label)
    if sum(row['affected'] for row in labels) != plan['expected_affected']:
        raise ValueError('Affected population differs from existing methods')
    return mapping, {index: raw for raw, index in mapping.items()}, original, views, labels


def sequence_hash(items):
    return hashlib.sha256(json.dumps(list(map(str, items)), separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


def joint_hash(row):
    payload = {'items': list(map(str, row['_raw'])), 'raw_lines': list(map(int, row['_events'])), 'time_buckets': None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def view_ledger(rows, group, seed):
    return [{'sample_id': row['sample_id'], 'group': group, 'permutation_seed': seed,
             'sequence_sha256': sequence_hash(row['_raw']), 'joint_sha256': joint_hash(row)} for row in rows]


def reproduce(predictions, path):
    rows = read_csv(path)
    lookup = {row['sample_id']: row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != {row['sample_id'] for row in predictions}:
        raise ValueError('Saved prediction sample population differs')
    proof = {'reference_path': str(path), 'reference_sha256': sha(path), 'samples': len(rows),
             'rank_mismatches': 0, 'ordered_top10_mismatches': 0, 'raw_top10_mismatches': 0}
    for row in predictions:
        prior = lookup[row['sample_id']]
        if any(str(row[key]) != str(prior[key]) for key in ('user_id', 'target_item')):
            raise ValueError('Saved prediction identity changed')
        proof['rank_mismatches'] += int(int(row['rank']) != int(prior.get('target_rank', prior.get('rank'))))
        proof['ordered_top10_mismatches'] += int(json.loads(row['top10_item_idx']) != json.loads(prior['top10_item_idx']))
        proof['raw_top10_mismatches'] += int(list(map(str, json.loads(row['top10_raw_item_id']))) != list(map(str, json.loads(prior['top10_raw_item_id']))))
    if any(proof[key] for key in ('rank_mismatches', 'ordered_top10_mismatches', 'raw_top10_mismatches')):
        raise ValueError('Canonical/fixed-five exact rank and Top10 reproduction failed: ' + str(proof))
    proof['status'] = 'PASS'
    return proof


def predict_rows(raw, rows, inverse, torch):
    if not bool(torch.isfinite(raw).all()):
        raise ValueError('Nonfinite unmasked model/mean score')
    scores = raw.clone()
    scores[:, 0] = -torch.inf
    targets = torch.tensor([row['_target'] for row in rows], dtype=torch.long, device=scores.device)
    counts = []
    for i, row in enumerate(rows):
        seen = set(row['_items']) - {row['_target']}
        scores[i, list(seen)] = -torch.inf
        counts.append(len(inverse) - len(seen))
    target_scores = scores.gather(1, targets[:, None])
    ranks = ((scores > target_scores).sum(1) + 1).cpu().tolist()
    tops = scores.topk(10, dim=1).indices.cpu().tolist()
    result = []
    for row, rank, top, count in zip(rows, ranks, tops, counts):
        if len(set(top)) != 10 or 0 in top or set(top) & (set(row['_items']) - {row['_target']}):
            raise ValueError('Top10 candidate mask violated')
        result.append({'sample_id': row['sample_id'], 'user_id': row['user_id'], 'target_item': row['target_item'],
                       'target_idx': row['_target'], 'rank': int(rank), 'top10_item_idx': json.dumps(top),
                       'top10_raw_item_id': json.dumps([inverse[index] for index in top]), 'candidate_count': count})
    return result


def score(model, rows, adapter, torch, counts):
    features = [adapter.prepare_history('control', row['_items'], row['_k']) for row in rows]
    seq = torch.tensor([value['seq'] for value in features], dtype=torch.long, device='cuda')
    lengths = torch.tensor([value['num_tokens'] for value in features], dtype=torch.long, device='cuda')
    raw = adapter.score_batch(model, 'control', seq, lengths)
    if not bool(torch.isfinite(raw).all()):
        raise ValueError('Nonfinite raw model logits')
    counts['actual_forward_rows'] += len(rows)
    counts['forward_batch_calls'] += 1
    return raw


def evaluate(model, rows, inverse, adapter, torch, counts):
    predictions = []
    for start in range(0, len(rows), 256):
        batch = rows[start:start + 256]
        predictions.extend(predict_rows(score(model, batch, adapter, torch, counts), batch, inverse, torch))
    return predictions


def stability(predictions, labels):
    import itertools
    result = {}
    for population, indices in [('full', range(len(labels))), ('affected', [i for i, row in enumerate(labels) if int(row['affected'])])]:
        users = []
        for i in indices:
            values = [panel[i] for panel in predictions]
            tops = [json.loads(row['top10_item_idx']) for row in values]
            overlap = sum(len(set(a) & set(b)) / 10 for a, b in itertools.combinations(tops, 2)) / 10
            users.append({'sample_id': values[0]['sample_id'], 'overlap10': overlap,
                          'any_hitflip': int(len({int(row['rank']) <= 10 for row in values}) > 1),
                          'top10_all_identical': int(all(top == tops[0] for top in tops[1:]))})
        if not users:
            raise ValueError('Empty stability population')
        result[population] = {'n': len(users), 'overlap10': sum(r['overlap10'] for r in users) / len(users),
                              'hitflip': sum(r['any_hitflip'] for r in users) / len(users), 'users': users}
    return result


def child(root, stage, method, tag, seed):
    manifest, inputs = verify_stage(root, stage)
    plan, run = inputs['datasets'][tag], inputs['datasets'][tag]['runs'][method][str(seed)]
    verify_root_files(root, inputs, run['required_root_files'])
    adapter = load_module(stage / 'scripts/x3_inference_adapter.py', 'x3_adapter')
    protocol = load_module(stage / 'scripts/x3_inference_protocol.py', 'x3_protocol')
    mapping, inverse, original, views, labels = load_panels(root, stage, plan, run)
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing accidental CPU inference')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_path = safe(root, run['model_dir'] + '/' + run['checkpoint'])
    checkpoint_before = sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    expected = {'model_name': 'SASRecControl', 'dataset': plan['dataset'], 'seed': seed, 'epoch': run['best_epoch'], 'num_items': len(mapping),
                'max_len': 50, 'hidden_size': 64, 'num_heads': 2, 'num_blocks': 2, 'dropout': 0.2}
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError('Checkpoint model/dataset/seed/epoch/architecture identity differs')
    definition = load_module(stage / 'scripts/x3_control_model.py', 'x3_control_model')
    model = definition.SASRecControl(num_items=len(mapping), max_len=50, hidden_size=64, num_heads=2, num_blocks=2, dropout=0.2).cuda()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    model.eval()
    output = stage / 'outputs' / (method + '_' + tag + '_seed' + str(seed))
    output.mkdir(exist_ok=False)
    counts = {'samples': len(original), 'affected': sum(row['affected'] for row in labels), 'items': len(mapping),
              'logical_view_rows': 0, 'actual_forward_rows': 0, 'forward_batch_calls': 0, 'cache_hits': 0}
    config = {'version': VERSION, 'method': method, 'tag': tag, 'seed': seed, 'input_sha256': INPUT_SHA256,
              'manifest_sha256': sha(stage / 'manifest.json'), 'tools_sha256': manifest['tools_sha256'],
              'protocol': inputs['protocol'], 'model_dir': run['model_dir'], 'checkpoint': run['checkpoint'],
              'best_epoch': run['best_epoch'], 'batch_size': 256, 'groups': [list(g) for g in GROUPS],
              'no_training': True, 'score_cache_enabled': False}
    write_json(output / 'config.json', config)
    write_csv(output / 'item_id_map.csv', [{'raw_item_id': raw, 'item_idx': index} for raw, index in mapping.items()])
    write_csv(output / 'panel_labels.csv', labels)
    ledgers, results, probes = view_ledger(original, 0, 'original'), {}, []
    fixed_reproduction = {}
    with torch.no_grad():
        canonical = evaluate(model, original, inverse, adapter, torch, counts)
        reproduction = reproduce(canonical, safe(stage, run['reference_predictions']))
        write_csv(output / 'predictions_original.csv', canonical)
        results['original'] = canonical
        if method == 'shuffle':
            for view in PERMS:
                rows = views[view]
                predictions = evaluate(model, rows, inverse, adapter, torch, counts)
                results['perm_seed' + str(view)] = predictions
                write_csv(output / ('predictions_perm_seed' + str(view) + '.csv'), predictions)
                ledgers.extend(view_ledger(rows, 1, view))
            compared = [results['perm_seed' + str(view)] for view in PERMS]
        else:
            bit_generator = np.random.default_rng(0).bit_generator.__class__.__name__
            if bit_generator != 'PCG64':
                raise ValueError('Legacy permutation protocol requires NumPy PCG64')
            for view in PERMS:
                for base, frozen in zip(original, views[view]):
                    key = protocol.legacy_seed(plan['dataset'], base['sample_id'], view)
                    indices = [int(value) for value in np.random.default_rng(key).permutation(base['_k'])]
                    items, events, _ = protocol.apply_last_bucket(base['_raw'], base['_events'], None, base['_k'], indices)
                    if items != frozen['_raw'] or events != frozen['_events']:
                        raise ValueError('Legacy PCG64 first-group item/event inputs do not reproduce sealed panels')
            # Existing group1 CSVs are used directly. New groups follow legacy PCG64.
            for group in GROUPS[1:]:
                for view in group:
                    rows = []
                    for base in original:
                        key = protocol.legacy_seed(plan['dataset'], base['sample_id'], view)
                        indices = [int(value) for value in np.random.default_rng(key).permutation(base['_k'])]
                        items, events, _ = protocol.apply_last_bucket(base['_raw'], base['_events'], None, base['_k'], indices)
                        row = dict(base, _raw=items, _events=events, _items=[mapping[item] for item in items])
                        rows.append(row)
                    views[view] = rows
            for group_index, group in enumerate(GROUPS, 1):
                predictions = []
                view_predictions = {view: [] for view in group} if group_index == 1 else {}
                for start in range(0, len(original), 256):
                    size = len(original[start:start + 256])
                    accumulator, member_probe = None, []
                    take = max(0, min(16 - start, size))
                    for view in group:
                        batch = views[view][start:start + 256]
                        raw = score(model, batch, adapter, torch, counts)
                        if take:
                            member_probe.append(raw[:take].clone())
                        if group_index == 1:
                            view_predictions[view].extend(predict_rows(raw, batch, inverse, torch))
                        value = raw.to(torch.float64)
                        accumulator = value if accumulator is None else accumulator + value
                    mean = accumulator / 5.0
                    batch_predictions = predict_rows(mean, original[start:start + 256], inverse, torch)
                    predictions.extend(batch_predictions)
                    for i in range(take):
                        pred = batch_predictions[i]
                        indices = [int(pred['target_idx'])] + [index for index in json.loads(pred['top10_item_idx']) if index != int(pred['target_idx'])]
                        probes.append({'sample_id': pred['sample_id'], 'group': group_index, 'member_seeds': list(group),
                            'target_idx': pred['target_idx'], 'top10_item_idx': json.loads(pred['top10_item_idx']),
                            'candidate_indices': indices, 'member_scores': [member[i, indices].cpu().tolist() for member in member_probe],
                            'mean_scores': mean[i, indices].cpu().tolist()})
                results['group' + str(group_index)] = predictions
                write_csv(output / ('predictions_group' + str(group_index) + '.csv'), predictions)
                for view in group:
                    ledgers.extend(view_ledger(views[view], group_index, view))
                    if group_index == 1:
                        write_csv(output / ('predictions_view_seed' + str(view) + '.csv'), view_predictions[view])
                        fixed_reproduction[str(view)] = reproduce(view_predictions[view], safe(stage, run['reference_permutations'][str(view)]))
            compared = [results['group' + str(group)] for group in range(1, 6)]
            write_json(output / 'score_mean_probes.json', {'scope': 'first at most16 canonical samples per group, not all candidate scores',
                'accumulation_dtype': 'torch.float64', 'member_score_dtype': 'torch.float32', 'reduction_order': [list(g) for g in GROUPS],
                'raw_scores_averaged_before_mask': True, 'scores_averaged_not_ranks': True,
                'legacy_first_group_input_reproduced': True, 'numpy_bit_generator': bit_generator, 'probes': probes})
    panel_metrics = {name: metrics([int(row['rank']) for row in values]) for name, values in results.items()}
    metric_report = {'panels': panel_metrics}
    if method == 'ensemble':
        metric_report['ensemble_mean_accuracy'] = {key: math.fsum(panel_metrics['group' + str(g)][key] for g in range(1, 6)) / 5 for key in panel_metrics['original']}
    write_json(output / 'metrics.json', metric_report)
    write_json(output / 'stability.json', stability(compared, labels))
    write_csv(output / 'input_view_hashes.csv', ledgers)
    checkpoint_after = sha(checkpoint_path)
    if checkpoint_after != checkpoint_before:
        raise ValueError('Checkpoint changed during inference')
    write_json(output / 'reproduction.json', {'status': 'PASS', 'canonical': reproduction, 'fixed_five_views': fixed_reproduction,
        'checkpoint_before_sha256': checkpoint_before, 'checkpoint_after_sha256': checkpoint_after})
    counts['logical_view_rows'] = len(original) * (6 if method == 'shuffle' else 26)
    expected_calls = math.ceil(len(original) / 256) * (6 if method == 'shuffle' else 26)
    if counts['actual_forward_rows'] != counts['logical_view_rows'] or counts['forward_batch_calls'] != expected_calls:
        raise ValueError('Declared/actual inference accounting differs')
    verify_root_files(root, inputs, run['required_root_files'])
    verify_stage(root, stage)
    metadata = {path.name: sha(path) for path in output.iterdir() if path.is_file()}
    completion = {'version': VERSION, 'method': method, 'tag': tag, 'seed': seed, 'input_sha256': INPUT_SHA256,
                  'manifest_sha256': sha(stage / 'manifest.json'), 'tools_sha256': manifest['tools_sha256'],
                  'metadata_sha256': metadata, 'status': 'COMPLETED', 'checkpoint_sha256': checkpoint_after,
                  'canonical_reproduction': True, 'old_control_five_views_reproduced': method == 'ensemble',
                  'originals_unchanged': True, 'training_started': False, 'counts': counts,
                  'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                                  'device': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}}
    write_json(stage / 'verification' / ('completed_' + method + '_' + tag + '_seed' + str(seed) + '.json'), completion)
    print('X3_INFERENCE_JOB_COMPLETED=' + method + '_' + tag + '_seed' + str(seed), flush=True)


def verify_completion(stage, method, tag, seed):
    path = stage / 'verification' / ('completed_' + method + '_' + tag + '_seed' + str(seed) + '.json')
    value = read_json(path)
    manifest, inputs = verify_stage(Path(read_json(stage / 'manifest.json')['root']), stage)
    plan = inputs['datasets'][tag]
    expected = {'version': VERSION, 'method': method, 'tag': tag, 'seed': seed, 'input_sha256': INPUT_SHA256,
                'manifest_sha256': sha(stage / 'manifest.json'), 'tools_sha256': manifest['tools_sha256'],
                'status': 'COMPLETED', 'canonical_reproduction': True, 'old_control_five_views_reproduced': method == 'ensemble',
                'originals_unchanged': True, 'training_started': False}
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError('Inference completion identity/proof invalid')
    count = value['counts']
    multiplier = 6 if method == 'shuffle' else 26
    expected_counts = {'samples': plan['expected_samples'], 'affected': plan['expected_affected'], 'items': plan['expected_items'],
        'logical_view_rows': plan['expected_samples'] * multiplier, 'actual_forward_rows': plan['expected_samples'] * multiplier,
        'forward_batch_calls': math.ceil(plan['expected_samples'] / 256) * multiplier, 'cache_hits': 0}
    if count != expected_counts:
        raise ValueError('Inference work/population counts invalid')
    output = stage / 'outputs' / (method + '_' + tag + '_seed' + str(seed))
    names = {'config.json', 'item_id_map.csv', 'predictions_original.csv', 'metrics.json', 'stability.json',
             'panel_labels.csv', 'input_view_hashes.csv', 'reproduction.json'}
    if method == 'shuffle':
        names.update('predictions_perm_seed' + str(view) + '.csv' for view in PERMS)
    else:
        names.update('predictions_group' + str(group) + '.csv' for group in range(1, 6))
        names.update('predictions_view_seed' + str(view) + '.csv' for view in PERMS)
        names.add('score_mean_probes.json')
    if set(value['metadata_sha256']) != names or {path.name for path in output.iterdir()} != names:
        raise ValueError('Inference output inventory is missing or unexpected')
    for name, expected_sha in value['metadata_sha256'].items():
        if sha(safe(output, name)) != expected_sha:
            raise ValueError('Saved inference output changed')
    reproduction = read_json(output / 'reproduction.json')
    run = plan['runs'][method][str(seed)]
    checks = [(reproduction['canonical'], run['reference_predictions'])]
    if set(reproduction['fixed_five_views']) != ({str(view) for view in PERMS} if method == 'ensemble' else set()):
        raise ValueError('Missing fixed-five reproduction proofs')
    checks.extend((reproduction['fixed_five_views'][str(view)], run['reference_permutations'][str(view)]) for view in PERMS if method == 'ensemble')
    for proof, relative in checks:
        if (proof.get('status') != 'PASS' or proof.get('samples') != plan['expected_samples']
                or proof.get('reference_path') != str(safe(stage, relative))
                or proof.get('reference_sha256') != manifest['references_sha256'][relative]
                or any(proof.get(key) != 0 for key in ('rank_mismatches', 'ordered_top10_mismatches', 'raw_top10_mismatches'))):
            raise ValueError('Prediction reproduction proof invalid')
    if (reproduction.get('status') != 'PASS' or reproduction.get('checkpoint_before_sha256') != value['checkpoint_sha256']
            or reproduction.get('checkpoint_after_sha256') != value['checkpoint_sha256']):
        raise ValueError('Checkpoint/inference provenance changed')
    return path


def run(root):
    stage = prepare(root)
    root = Path(root).resolve()
    print('X3_INFERENCE_STAGE=' + str(stage), flush=True)
    state = {'version': VERSION, 'stage': str(stage), 'status': 'RUNNING', 'jobs': []}
    path = stage / 'verification/driver.json'
    write_json(path, state)
    result, process = 1, None
    try:
        for method in METHODS:
            for tag in TAGS:
                for seed in SEEDS:
                    command = [sys.executable, '-u', str(stage / 'scripts/run_x3_inference.py'), '--root', str(root), '--run',
                               '--stage', str(stage), '--method', method, '--dataset', tag, '--seed', str(seed)]
                    logfile = stage / 'verification' / (method + '_' + tag + '_seed' + str(seed) + '.log')
                    record = {'method': method, 'tag': tag, 'seed': seed, 'command': command, 'log': str(logfile),
                              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
                    state['jobs'].append(record)
                    write_json(path, state)
                    with logfile.open('x', encoding='utf-8') as handle:
                        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace')
                        for line in process.stdout:
                            handle.write(line)
                            handle.flush()
                            print(line, end='', flush=True)
                        record['exit_code'] = process.wait()
                    process = None
                    if record['exit_code']:
                        raise RuntimeError('Inference job failed: ' + method + ' ' + tag + ' seed' + str(seed))
                    proof = verify_completion(stage, method, tag, seed)
                    record.update(proof=str(proof), proof_sha256=sha(proof), log_sha256=sha(logfile),
                                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    write_json(path, state)
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
        print('X3_INFERENCE_ERROR=' + state['error'], flush=True)
    finally:
        state['status'] = 'ALL_TWELVE_SHUFFLE_AND_TWELVE_ENSEMBLE_EVALUATIONS_COMPLETED' if result == 0 else 'STOPPED_INCOMPLETE'
        write_json(path, state)
        print('X3_INFERENCE_DRIVER_STATUS=' + state['status'], flush=True)
        collected = subprocess.run([sys.executable, str(stage / 'scripts/collect_x3_inference.py'), '--stage', str(stage)])
        if collected.returncode:
            result = collected.returncode
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--stage', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--method', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--dataset', choices=TAGS, help=argparse.SUPPRESS)
    parser.add_argument('--seed', choices=SEEDS, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.run:
        print('PLAN_ONLY_NO_FILES_OR_GPU: existing12Shuffle original+5perm; existing12Control original+5groups x5raw-score members; exact old predictions first; no training; sequential; collect metadata.')
        return 0
    if args.stage:
        if args.method is None or args.dataset is None or args.seed is None:
            parser.error('Internal stage requires method/dataset/seed')
        child(args.root.resolve(), args.stage.resolve(), args.method, args.dataset, args.seed)
        return 0
    return run(args.root)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('ERROR: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
