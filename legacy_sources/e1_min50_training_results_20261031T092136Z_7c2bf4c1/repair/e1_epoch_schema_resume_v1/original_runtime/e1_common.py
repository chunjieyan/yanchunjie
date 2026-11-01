"""E1 standard-library preparation and saved-output checks; no research imports."""
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import uuid

HERE = Path(__file__).resolve().parent
PREFIX = 'kais_e1_min50_v1_'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex[:10] + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def safe(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in str(relative) and ':' not in str(relative), 'Unsafe relative path')
    result = root / p
    require(result.resolve().is_relative_to(root), 'Path leaves root')
    for part in (result,) + tuple(result.parents):
        if part == root.parent:
            break
        require(not part.is_symlink(), 'Symlink rejected: ' + str(part))
    return result


def lock():
    result = read_json(HERE / 'e1_policy.json')
    require(result['minimum_epochs'] == 50 and result['patience'] == 15 and result['max_epochs'] == 200, 'Policy changed')
    require(len(result['jobs']) == 19, 'Exactly19 selected runs required')
    require(len({j['id'] for j in result['jobs']}) == 19, 'Duplicate selected run')
    for relative, expected in result['bundle_sha256'].items():
        require(sha(safe(HERE, relative)) == expected, 'Bundle source changed: ' + relative)
    audited = read_csv(HERE / 'all42_training_audit.csv')
    require(len(audited) == 42, 'Audit inventory must contain42 models')
    expected = {(r['dataset'], r['method'], int(r['seed'])) for r in audited if int(r['stop_epoch']) < 50}
    require({(j['dataset'], j['method'], j['seed']) for j in result['jobs']} == expected, 'Selected runs differ from all42 audited stop<50 rule')
    return result


def assignment(source, name, expression):
    matches = [n for n in ast.parse(source).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    require(len(matches) == 1, 'Expected one top-level assignment: ' + name)
    node = matches[0]
    old = ast.get_source_segment(source, node)
    return once(source, old, name + ' = ' + expression)


def once(source, old, new):
    require(source.count(old) == 1, 'Unexpected source shape: ' + old[:100])
    return source.replace(old, new, 1)


def transform(source, family, root, stage, seed=None):
    """Only infrastructure/path/config edits and the single stop gate change."""
    original = source
    edits = []
    def assign(name, expression):
        nonlocal source
        before = source
        source = assignment(source, name, expression)
        edits.append({'kind': 'assignment', 'name': name})
        return before
    def change(old, new, kind):
        nonlocal source
        source = once(source, old, new)
        edits.append({'kind': kind, 'old': old, 'new': new})
    if family == 'control':
        assign('_ORDERFIX_ROOT', 'plain_path(Path(' + repr(str(root)) + '))')
        assign('_ORDERFIX_STAGE', 'plain_path(Path(' + repr(str(stage)) + '))')
    elif family == 'shuffle':
        assign('_X3_ROOT', 'plain(Path(' + repr(str(root)) + '))')
        assign('_X3_STAGE', 'plain(Path(' + repr(str(stage)) + '))')
    elif family == 'amazon_lbsp':
        assign('_TRAIN_ROOT', 'Path(' + repr(str(root)) + ')')
        assign('_TRAIN_STAGE', 'Path(' + repr(str(stage)) + ')')
        change("choices=['train']", "choices=['smoke','train']", 'smoke_cli')
        marker = '# Smoke 不更新参数；这里重置训练随机状态。'
        probe = '''if args.mode == 'smoke':
    _probe = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)
    _batch = [value.to(device) for value in next(iter(_probe))]
    model.train()
    model.zero_grad(set_to_none=True)
    _loss = training_loss(*_batch)
    require(bool(torch.isfinite(_loss)), 'E1 smoke nonfinite loss')
    _loss.backward()
    _grads = [p.grad for p in model.parameters() if p.grad is not None]
    require(bool(_grads) and all(bool(torch.isfinite(g).all()) for g in _grads), 'E1 smoke invalid gradients')
    require(any(bool(g.abs().sum() > 0) for g in _grads), 'E1 smoke zero gradients')
    model.zero_grad(set_to_none=True)
    print('E1_AMAZON_LBSP_SMOKE_PASS optimizer_steps=0', flush=True)
    raise SystemExit(0)

'''
        change(marker, probe + marker, 'smoke_probe')
    elif family == 'ml_lbsp':
        assign('ROOT', 'Path(' + repr(str(root)) + ')')
        assign('SEED', str(seed))
        assign('OUTPUT_DIR', 'Path(' + repr(str(stage / 'outputs' / ('ml1m_lbsp_seed' + str(seed)))) + ')')
    elif family == 'tisasrec':
        change('if bad >= 15:\n                break', 'if epoch >= 50 and bad >= 15:\n                break', 'minimum_stop_gate')
        change("write_json(output / 'config.json', config)", "config['e1_minimum_stopping_epochs'] = 50\n    write_json(output / 'config.json', config)", 'config_metadata')
    else:
        raise ValueError('Unknown source family')
    if family != 'tisasrec':
        change('if bad_epochs >= PATIENCE:\n', 'if epoch >= 50 and bad_epochs >= PATIENCE:\n', 'minimum_stop_gate')
        # Insert metadata directly before serialization, after the config is complete.
        if '(OUTPUT_DIR / "config.json").write_text(' in source:
            change('(OUTPUT_DIR / "config.json").write_text(', 'config["e1_minimum_stopping_epochs"] = 50\n(OUTPUT_DIR / "config.json").write_text(', 'config_metadata')
        else:
            marker = 'with open(\n    OUTPUT_DIR / "config.json",'
            change(marker, 'config["e1_minimum_stopping_epochs"] = 50\n' + marker, 'config_metadata')
    ast.parse(source)
    # Every scientific function/class is byte/AST-identical except X4's full child,
    # whose difference is restricted to the gate and config insertion above.
    old_defs = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(original).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    reversed_text = source
    for e in reversed(edits):
        if e['kind'] in ('minimum_stop_gate', 'config_metadata'):
            reversed_text = once(reversed_text, e['new'], e['old'])
    new_defs = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(reversed_text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    require(old_defs == new_defs, 'Scientific function/class changed')
    return source, {'edits': edits, 'scientific_definition_count': len(old_defs), 'scientific_ast_unchanged_after_gate_undo': True}


def preflight(root, policy):
    root = Path(root).resolve()
    require(root.is_dir(), 'Project root missing')
    for relative, expected in policy['server_readonly_sha256'].items():
        path = safe(root, relative)
        require(path.is_file() and sha(path) == expected, 'Frozen server input missing/changed: ' + relative)
    return {'status': 'PASS_FROZEN_INPUTS', 'checked_files': len(policy['server_readonly_sha256'])}


def make_attempt(root, outer, job, number, policy):
    """A failed partial attempt is preserved; retries get a new output directory."""
    family = job['family']
    # X4's accepted worker requires its private stage be a direct project child.
    native = root / ('kais_x4_tisasrec_v1_e1_' + outer.name[len(PREFIX):] + '_' + job['id'] + '_a' + str(number)) if family == 'tisasrec' else outer / 'attempts' / (job['id'] + '_a' + str(number))
    require(not native.exists(), 'Attempt exists; refusing overwrite')
    native.mkdir(parents=True)
    for folder in ('scripts', 'verification', 'outputs', 'manifests'):
        (native / folder).mkdir()
    sources = policy['families'][family]
    generated = {}
    for name, vendor in sources['scripts'].items():
        if name in sources.get('patch_scripts', []):
            source = safe(HERE, vendor).read_text(encoding='utf-8')
            source, audit = transform(source, family, root, native, job['seed'])
            write_json(native / 'verification' / ('transformation_' + name + '.json'), audit)
            (native / 'scripts' / name).write_text(source, encoding='utf-8', newline='\n')
        else:
            shutil.copyfile(safe(HERE, vendor), native / 'scripts' / name)
        generated['scripts/' + name] = sha(native / 'scripts' / name)
    if family in ('shuffle', 'amazon_lbsp'):
        manifest = read_json(safe(HERE, sources['manifest']))
        manifest.update(root=str(root), stage=str(native), generated_sha256=generated, e1_minimum_stopping_epochs=50)
        write_json(native / sources['manifest_name'], manifest)
        if family == 'amazon_lbsp':
            plan = manifest['inputs']['datasets'][job['dataset']]
            target = native / 'references' / (job['dataset'] + '_seed42')
            target.mkdir(parents=True)
            for name, expected in plan['control_reference']['files_sha256'].items():
                old = safe(root, plan['control_reference']['model_dir'] + '/' + name)
                require(sha(old) == expected, 'Old reference changed')
                shutil.copyfile(old, target / name)
    if family == 'tisasrec':
        inputs = read_json(safe(HERE, sources['inputs']))
        input_name = 'x4_tisasrec_inputs_20261026.json'
        shutil.copyfile(safe(HERE, sources['inputs']), native / input_name)
        generated[input_name] = sha(native / input_name)
        manifest = {'version': 'x4_tisasrec_v1', 'root': str(root), 'stage': str(native),
                    'input_sha256': sha(native / input_name), 'tool_sha256': generated,
                    'data_sha256': inputs['data_sha256'], 'tasks': [{'dataset': job['dataset'], 'seed': job['seed']}],
                    'protocol': inputs['protocol'], 'e1_minimum_stopping_epochs': 50}
        write_json(native / 'manifest.json', manifest)
    return native


def output_path(stage, job):
    if job['family'] == 'control':
        name = 'amazon_' + job['dataset'] + '_control_orderfix_seed' + str(job['seed'])
    elif job['family'] == 'tisasrec':
        name = job['dataset'] + '_seed' + str(job['seed'])
    else:
        name = job['dataset'] + '_' + ('shuffle' if job['family'] == 'shuffle' else 'lbsp') + '_seed' + str(job['seed'])
    return stage / 'outputs' / name


def checked_native(root, outer, value):
    native = Path(value)
    require(native.is_absolute() and native.is_relative_to(root), 'Native path leaves project')
    native = safe(root, native.relative_to(root).as_posix())
    require(native.is_relative_to(outer / 'attempts') or (native.parent == root and native.name.startswith('kais_x4_tisasrec_v1_e1_')), 'Unexpected native stage')
    return native


def metric_values(ranks):
    n = len(ranks)
    require(n > 0 and all(r >= 1 for r in ranks), 'Invalid saved ranks')
    return {'ndcg10': sum(1 / math.log2(r + 1) for r in ranks if r <= 10) / n,
            'hr10': sum(r <= 10 for r in ranks) / n, 'hr20': sum(r <= 20 for r in ranks) / n,
            'mean_rank': sum(ranks) / n, 'mrr': sum(1 / r for r in ranks) / n,
            'mrr10': sum(1 / r for r in ranks if r <= 10) / n, 'recall50': sum(r <= 50 for r in ranks) / n}


def validate_output(stage, job):
    output = output_path(stage, job)
    config, metrics = read_json(output / 'config.json'), read_json(output / 'metrics.json')
    require(config.get('seed') == job['seed'] and config.get('e1_minimum_stopping_epochs') == 50, 'Output config identity/policy differs')
    hyper = config.get('hyperparameters', config)
    require(hyper.get('patience') == 15 and hyper.get('max_epochs') == 200, 'Output stopping budget differs')
    trace = read_csv(output / 'training_log.csv')
    require(50 <= len(trace) <= 200, 'Run ended outside minimum/max epochs')
    best, best_epoch, bad = -math.inf, 0, 0
    for epoch, row in enumerate(trace, 1):
        require(int(row['epoch']) == epoch, 'Missing epoch')
        val, loss = float(row['val_ndcg10']), float(row['train_loss'])
        require(math.isfinite(val) and 0 <= val <= 1 and math.isfinite(loss) and loss >= 0, 'Nonfinite trajectory')
        eps = 1e-12 if job['family'] in ('ml_lbsp', 'amazon_lbsp') or (job['dataset'] in ('sports', 'toys') and job['family'] in ('control', 'shuffle')) else 0
        if val > best + eps:
            best, best_epoch, bad = val, epoch, 0
        else:
            bad += 1
        if 'bad_epochs' in row:
            require(int(row['bad_epochs']) == bad, 'Saved patience state differs')
        if 'best_epoch' in row:
            require(int(row['best_epoch']) == best_epoch, 'Saved best epoch differs')
        require(not(epoch >= 50 and bad >= 15) or epoch == len(trace), 'Run continued after new stopping rule')
    require(len(trace) == 200 or bad >= 15, 'Premature/incomplete training')
    require(metrics['best_epoch'] == best_epoch, 'Wrong selected checkpoint epoch')
    saved_validation = metrics.get('best_validation', metrics.get('validation'))
    if 'overall' in saved_validation:
        saved_validation = saved_validation['overall']
    require(abs(saved_validation.get('ndcg10', saved_validation.get('NDCG@10')) - best) < 1e-11, 'Wrong best validation value')
    pred_name = 'predictions_original.csv' if job['family'] == 'tisasrec' else 'test_per_sample_predictions.csv'
    predictions = read_csv(output / pred_name)
    require(len(predictions) == job['expected_test'], 'Saved test count differs')
    require(len({p['sample_id'] for p in predictions}) == len(predictions), 'Duplicate saved sample')
    baseline = read_csv(safe(HERE, job['baseline_predictions']))
    require([(p['sample_id'], p['user_id'], p['target_item']) for p in baseline] == [(p['sample_id'], p['user_id'], p['target_item']) for p in predictions], 'Canonical test sample/target identities differ')
    mapping_rows = read_csv(output / 'item_id_map.csv')
    old_mapping = read_csv(safe(HERE, job['baseline_map']))
    require(mapping_rows == old_mapping, 'Frozen item mapping differs')
    mapping = {r['raw_item_id']: int(r['item_idx']) for r in mapping_rows}
    require(len(mapping) == len(mapping_rows) and sorted(mapping.values()) == list(range(1, len(mapping) + 1)), 'Invalid item map')
    for row in predictions:
        indices = json.loads(row['top10_item_idx'])
        raw = json.loads(row['top10_raw_item_id'])
        require(len(indices) == len(raw) == 10 and len(set(indices)) == len(set(raw)) == 10, 'Malformed saved Top10')
        require(all(isinstance(i, int) and not isinstance(i, bool) and mapping.get(str(r)) == i for r, i in zip(raw, indices)), 'Top10 index/raw identity differs')
        require(1 <= int(row.get('rank', row.get('target_rank'))) <= len(mapping), 'Rank outside catalogue')
    current = metric_values([int(p.get('rank', p.get('target_rank'))) for p in predictions])
    saved_test = metrics['panels']['original'] if job['family'] == 'tisasrec' else metrics['test']
    saved_test = saved_test.get('overall', saved_test)
    alias = {'NDCG@10': 'ndcg10', 'HR@10': 'hr10', 'HR@20': 'hr20', 'MRR': 'mrr', 'MRR@10': 'mrr10', 'Recall@50': 'recall50', 'MeanRank': 'mean_rank', 'mean_target_rank': 'mean_rank'}
    require({'ndcg10', 'hr10', 'hr20', 'mean_rank'} <= {alias.get(k, k) for k in saved_test}, 'Required saved test metrics missing')
    if job['family'] == 'tisasrec':
        require(set(current) <= set(saved_test), 'Required time-aware test metrics missing')
    for key, value in saved_test.items():
        key = alias.get(key, key)
        if key in current:
            require(math.isclose(float(value), current[key], rel_tol=1e-10, abs_tol=1e-11), 'Test metric differs: ' + key)
    old_trace = read_csv(safe(HERE, job['baseline_log']))
    prefix_deltas = {field: max(abs(float(a[field]) - float(b[field])) for a, b in zip(trace, old_trace)) for field in ('val_ndcg10', 'train_loss')}
    weights = [p for p in output.iterdir() if p.is_file() and p.suffix in ('.pt', '.pth')]
    require(len(weights) == 1, 'Expected exactly one retained best checkpoint')
    if job['family'] == 'tisasrec':
        original_proof = read_json(stage / 'verification' / ('completed_' + job['dataset'] + '_seed' + str(job['seed']) + '.json'))
        require(original_proof['status'] == 'COMPLETED' and original_proof['canonical_and_five_permutations'] is True and original_proof['validation_reproduced'] is True, 'Time-aware completion scope differs')
        require(original_proof['checkpoint_sha256'] == sha(weights[0]), 'Time-aware best checkpoint changed')
        for name in ['predictions_original.csv'] + ['predictions_perm_seed' + str(s) + '.csv' for s in (101, 202, 303, 404, 505)] + ['stability.json']:
            require(name in original_proof['metadata_sha256'] and sha(output / name) == original_proof['metadata_sha256'][name], 'Time-aware permutation output missing/changed')
    return {'status': 'PASS_COMPLETED_TRAINING_AND_CANONICAL_OUTPUTS', 'id': job['id'], 'best_epoch': best_epoch,
            'stop_epoch': len(trace), 'best_validation_ndcg10': best, 'canonical_metrics': current,
            'old_prefix_max_abs_differences': prefix_deltas,
            'prefix_note': 'A fresh run is not optimizer continuation. Prefix differences are retained; nonzero differences limit attribution to the stopping gate alone.',
            'five_permutation_inference_complete': job['family'] == 'tisasrec',
            'checkpoint_sha256': {p.name: sha(p) for p in weights},
            'output_dir': str(output),
            'metadata_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.suffix in ('.json', '.csv')}}
