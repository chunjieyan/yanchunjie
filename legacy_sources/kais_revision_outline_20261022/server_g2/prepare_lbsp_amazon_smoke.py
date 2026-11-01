#!/usr/bin/env python3
"""Prepare three isolated Amazon LBSP smoke checks; standard library only.

No research definitions are executed during preparation. No weight file is
opened or statted. Generated smoke scripts cannot enter formal training.
"""
import argparse
import ast
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import uuid

INPUTS_SHA256 = 'c65912d443b01d2f76ed164e57638a99ad1b1972394d542b5f22fe1eacd92265'
VERSION = 'lbsp_amazon_smoke_v1'
MANIFEST_PATH = 'manifests/lbsp_amazon_smoke_manifest.json'
DATASETS = {'beauty': 'Amazon_Beauty', 'sports': 'Amazon_Sports', 'toys': 'Amazon_Toys'}
SAVED_FILES = {'config.json', 'metrics.json', 'training_log.csv', 'test_per_sample_predictions.csv', 'item_id_map.csv'}
FROZEN_NAMES = ('build_r1_group_positions_from_buckets', 'build_r1_group_positions_from_size',
                'canonicalize_last_bucket_items', 'RecentBucketSetR1SASRec', 'predict_batch_masked')


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
    if digest.hexdigest() != expected:
        raise RuntimeError('SHA256 mismatch: ' + relative)
    return digest.hexdigest()


def normalize_eval_records(records):
    """Only add missing aliases; reject inconsistent existing fields."""
    result = []
    for record in records:
        row = dict(record)
        present = [row[k] for k in ('last_bucket_size', 'last_history_bucket_size') if k in row]
        if not present or any(not re.fullmatch(r'[0-9]+', str(v)) for v in present):
            raise ValueError('Missing/invalid last-bucket size')
        sizes = [int(v) for v in present]
        history = json.loads(row['sequence_items'])
        if (not isinstance(history, list) or not 1 <= len(history) <= 50
                or len(set(sizes)) != 1 or not 1 <= sizes[0] <= len(history)):
            raise ValueError('Invalid history or inconsistent last-bucket size')
        expected_pooling = int(sizes[0] >= 2)
        if 'direct_pooling' in row and (not re.fullmatch(r'[01]', str(row['direct_pooling']))
                or int(row['direct_pooling']) != expected_pooling):
            raise ValueError('Existing direct_pooling conflicts with last-bucket size')
        if 'last_bucket_size' not in row:
            row['last_bucket_size'] = sizes[0]
        if 'direct_pooling' not in row:
            row['direct_pooling'] = expected_pooling
        result.append(row)
    return result


def normalize_eval_frame(df):
    normalized = normalize_eval_records(df.to_dict(orient='records'))
    df = df.copy()
    for name in ('last_bucket_size', 'direct_pooling'):
        if name not in df.columns:
            df[name] = [row[name] for row in normalized]
    return df


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Unexpected template structure: ' + old[:70])
    return source.replace(old, new, 1)


def replace_assignment(source, name, expression):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(nodes) != 1:
        raise RuntimeError('Expected one assignment: ' + name)
    return replace_once(source, ast.get_source_segment(source, nodes[0]), name + ' = ' + expression)


def definition(source, name):
    matches = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    if len(matches) != 1:
        raise RuntimeError('Expected one definition: ' + name)
    return matches[0]


def frozen_model(model_source, forward_source):
    selected = [(model_source, definition(model_source, name)) for name in FROZEN_NAMES[:-1]]
    selected.append((forward_source, definition(forward_source, FROZEN_NAMES[-1])))
    result = '# Frozen scientific definitions only; original training top-level is excluded.\n' + '\n\n'.join(
        ast.get_source_segment(text, node) for text, node in selected) + '\n'
    if [ast.dump(n) for _, n in selected] != [ast.dump(n) for n in ast.parse(result).body]:
        raise RuntimeError('Frozen algorithm AST changed')
    return result


def sorting_block(control_source, tag):
    """Extract the pinned Control ordering statements, never execute them."""
    if tag == 'beauty':
        start = 'train_df["user_id"] = train_df["user_id"].astype(str)'
        end = '\n\n\n# ============================================================\n# 6.'
        if control_source.count(start) != 1 or control_source.count(end) != 1:
            raise RuntimeError('Unexpected Beauty ordering source')
        block = control_source.split(start, 1)[1].split(end, 1)[0]
        block = start + block
        if '["user_id", "time_bucket", "_source_order"]' not in block or 'kind="mergesort"' not in block:
            raise RuntimeError('Beauty stable temporal ordering missing')
    elif tag in ('sports', 'toys'):
        start = 'required_cols = {"user_id", "item_id", "raw_line", "time_bucket"}'
        end = '\nunknown_train = ('
        if control_source.count(start) != 1 or control_source.count(end) != 1:
            raise RuntimeError('Unexpected corrected ordering source')
        block = start + control_source.split(start, 1)[1].split(end, 1)[0]
        if ('["user_id", "time_bucket", "raw_line"], kind="stable"' not in block
                or 'Duplicate (user_id, raw_line) event key' not in block
                or 'Ordering correction added or removed training events' not in block
                or 'Canonical temporal/tie order assertion failed' not in block):
            raise RuntimeError('Corrected Control ordering guards missing')
        block += '\nBUCKET_COLUMN = "time_bucket"\n'
    else:
        raise ValueError('Unknown dataset tag')
    ast.parse(block)
    return block.rstrip() + '\n'


def generated_smoke(source, control_source, tag, inputs, root, stage, model_hash):
    plan = inputs['datasets'][tag]
    original_tree = ast.parse(source)
    protected = {name: ast.dump(definition(source, name)) for name in
                 ('AllPositionR1Dataset', 'training_loss', 'evaluate', 'predict_batch_masked')}
    # Cut before the original smoke exit: no formal training statements survive.
    exits = [n for n in original_tree.body if isinstance(n, ast.If)
             and ast.unparse(n.test) == "args.mode == 'smoke'"]
    if len(exits) != 1:
        raise RuntimeError('Original smoke exit boundary is ambiguous')
    source = '\n'.join(source.splitlines()[:exits[0].lineno - 1]) + '\n'
    for name, expression in {
            'ROOT': '_SMOKE_ROOT', 'BASE': '_SMOKE_STAGE', 'DATASET': repr(plan['dataset']),
            'CONTROL_SCRIPT': 'ROOT / _PLAN["control_trainer_source"]',
            'R1_SOURCE': '_SMOKE_STAGE / "scripts/model_definition.py"',
            'CONTROL_DIR': 'ROOT / _PLAN["control_reference"]["model_dir"]',
            'OUTPUT_DIR': '_SMOKE_STAGE / "outputs" / ' + repr(tag + '_smoke_unused'),
            'EXPECTED_ITEMS': repr(plan['expected_items']), 'EXPECTED_VAL': repr(plan['expected_val']),
            'EXPECTED_TEST': repr(plan['expected_test']), 'EXPECTED_POSITIONS': repr(plan['expected_positions'])}.items():
        source = replace_assignment(source, name, expression)
    source = replace_once(source, 'choices=["smoke", "train"]', 'choices=["smoke"]')
    start, end = 'control_tree = ast.parse(', 'for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:'
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError('Original path extraction boundary changed')
    a, tail = source.split(start, 1)
    _, b = tail.split(end, 1)
    source = a + ('TRAIN_PATH = ROOT / _PLAN["train_path"]\n'
                  'VAL_PATH = ROOT / _PLAN["validation_path"]\n'
                  'TEST_PATH = ROOT / _PLAN["test_path"]\n\n') + end + b
    begin = 'require(\n    {"user_id", "item_id"}.issubset('
    finish = 'user_sequences = {}'
    if source.count(begin) != 1 or source.count(finish) != 1:
        raise RuntimeError('Original training sorting boundary changed')
    a, tail = source.split(begin, 1)
    _, b = tail.split(finish, 1)
    source = a + sorting_block(control_source, tag) + '\n' + finish + b
    # Only insert an alias/consistency normalization after the original CSV read.
    loader = definition(source, 'load_eval')
    first = loader.body[0]
    if not isinstance(first, ast.Assign) or ast.unparse(first.targets[0]) != 'df':
        raise RuntimeError('load_eval no longer begins with a CSV read')
    old = ast.get_source_segment(source, first)
    source = replace_once(source, old, old + '\n    df = normalize_eval_frame(df)')
    for name, expected in protected.items():
        if ast.dump(definition(source, name)) != expected:
            raise RuntimeError('Scientific definition changed: ' + name)
    original_loader = definition('\n'.join(source_line for source_line in source.splitlines()
        if source_line != '    df = normalize_eval_frame(df)'), 'load_eval')
    # Independently compare the controlled loader insertion against the source AST.
    reference_loader = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_eval')
    if ast.dump(original_loader) != ast.dump(reference_loader):
        raise RuntimeError('Unexpected load_eval change beyond alias insertion')
    source = replace_once(source, 'loss.backward()', '''loss.backward()
_ALL_GRADIENTS_FINITE = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
require(_ALL_GRADIENTS_FINITE,
        "Smoke backward produced a non-finite gradient")''')
    source = source.replace('STEP38.20 ML1M R1 MEAN POOLING', 'AMAZON LBSP SMOKE ONLY: ' + tag)
    prelude = runtime_prelude(tag, inputs, root, stage, model_hash)
    result = prelude + '\n' + source + '\n' + smoke_report_code()
    tree = ast.parse(result, feature_version=(3, 10))
    forbidden = {'torch.load', 'torch.save', 'torch.optim.Adam', 'torch.optim.AdamW', 'optimizer.step', 'optimizer.zero_grad'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in forbidden:
            raise RuntimeError('Forbidden training/weight call in smoke script')
    if any(isinstance(n, ast.FunctionDef) and n.name == 'read_path_assignment' for n in tree.body):
        raise RuntimeError('Old Control path interpretation survived')
    return result


def runtime_prelude(tag, inputs, root, stage, model_hash):
    helpers = '\n\n'.join(inspect.getsource(fn) for fn in (plain, checked_file, file_signature, checked_sha256,
                                      normalize_eval_records, normalize_eval_frame))
    text = '''# Generated smoke-only script. Hash checks run before NumPy/Torch imports.
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import math
'''
    text += helpers + '\n'
    text += '_SMOKE_ROOT = Path(' + repr(str(root)) + ')\n_SMOKE_STAGE = Path(' + repr(str(stage)) + ')\n'
    text += '_TAG = ' + repr(tag) + '\n_INPUTS = json.loads(' + repr(json.dumps(inputs, ensure_ascii=False)) + ')\n'
    text += '_INPUTS_SHA256 = ' + repr(INPUTS_SHA256) + '\n_MODEL_SHA256 = ' + repr(model_hash) + '\n'
    text += '''_PLAN = _INPUTS['datasets'][_TAG]
_MANIFEST_RELATIVE = 'manifests/lbsp_amazon_smoke_manifest.json'
_MANIFEST_FILE = checked_file(_SMOKE_STAGE, _MANIFEST_RELATIVE)
_MANIFEST_BYTES = _MANIFEST_FILE.read_bytes()
_MANIFEST_HASH = hashlib.sha256(_MANIFEST_BYTES).hexdigest()
_MANIFEST = json.loads(_MANIFEST_BYTES.decode('utf-8-sig'))
if (_MANIFEST['version'] != 'lbsp_amazon_smoke_v1' or _MANIFEST['inputs'] != _INPUTS
        or _MANIFEST['inputs_sha256'] != _INPUTS_SHA256 or _MANIFEST['root'] != str(_SMOKE_ROOT)
        or _MANIFEST['stage'] != str(_SMOKE_STAGE)):
    raise RuntimeError('Preparation manifest identity differs')
_GENERATED_RELATIVE = 'scripts/smoke_' + _TAG + '_lbsp.py'
_GENERATED_HASH = _MANIFEST['generated_sha256'][_GENERATED_RELATIVE]
checked_sha256(_SMOKE_STAGE, _GENERATED_RELATIVE, _GENERATED_HASH)
checked_sha256(_SMOKE_STAGE, 'scripts/model_definition.py', _MODEL_SHA256)
_DATA_HASHES = {path: _INPUTS['data_sha256'][path] for path in
                (_PLAN['train_path'], _PLAN['validation_path'], _PLAN['test_path'])}
_REF_HASHES = _PLAN['control_reference']['files_sha256']
_OLD_HASHES = dict(_INPUTS['sources_sha256'])
_OLD_HASHES.update(_DATA_HASHES)
_OLD_HASHES.update({_PLAN['control_reference']['model_dir'] + '/' + name: digest
                    for name, digest in _REF_HASHES.items()})
for _relative, _expected in _OLD_HASHES.items():
    checked_sha256(_SMOKE_ROOT, _relative, _expected)
_OUTPUT_ROOT = plain(_SMOKE_STAGE / 'outputs')
if any(_OUTPUT_ROOT.iterdir()):
    raise RuntimeError('Smoke outputs directory must remain empty')
_REPORT_RELATIVE = 'verification/smoke_' + _TAG + '.json'
_REPORT_FILE = plain(_SMOKE_STAGE / 'verification') / ('smoke_' + _TAG + '.json')
if _REPORT_FILE.exists():
    raise RuntimeError('Smoke report already exists; refusing to overwrite')
'''
    return text


def smoke_report_code():
    return '''# Complete smoke-only verification, with no formal result or weight output.
require(float(loss.item()) > 0.0, "Smoke loss must be positive")
require(smoke_val['samples'] == 32 and all(math.isfinite(float(v)) for v in smoke_val.values()),
        "Validation smoke metrics must be finite for exactly 32 samples")
for _relative, _expected in _OLD_HASHES.items():
    checked_sha256(_SMOKE_ROOT, _relative, _expected)
checked_sha256(_SMOKE_STAGE, _GENERATED_RELATIVE, _GENERATED_HASH)
checked_sha256(_SMOKE_STAGE, 'scripts/model_definition.py', _MODEL_SHA256)
checked_sha256(_SMOKE_STAGE, _MANIFEST_RELATIVE, _MANIFEST_HASH)
require(not any(plain(_SMOKE_STAGE / 'outputs').iterdir()), "Smoke produced forbidden formal outputs")
_SUMMARY = {
    'version': 'lbsp_amazon_smoke_v1', 'dataset': DATASET, 'tag': _TAG, 'seed': SEED, 'status': 'SMOKE_PASS',
    'input_manifest_sha256': _INPUTS_SHA256, 'preparation_manifest_sha256': _MANIFEST_HASH,
    'generated_script_sha256': _GENERATED_HASH, 'model_definition_sha256': _MODEL_SHA256,
    'sources_sha256': _INPUTS['sources_sha256'], 'data_sha256': _DATA_HASHES,
    'reference_files_sha256': _REF_HASHES, 'control_reference': _PLAN['control_reference'],
    'counts': {'num_items': NUM_ITEMS, 'training_rows': len(train_df), 'trainable_users': len(user_sequences),
        'training_positions': LEGAL_POSITIONS, 'validation_samples': len(val_df), 'test_samples': len(test_df)},
    'smoke_loss': float(loss.item()), 'finite_gradient_count': int(grad_count), 'all_gradients_finite': _ALL_GRADIENTS_FINITE,
    'parity': {'max_abs_diff': float(max_diff), 'atol': 1e-4, 'rtol': 1e-5, 'allclose': True, 'samples': sample_count},
    'invariance': {'tested': tested, 'arrangements_per_sample': 6, 'exact_logits_invariance': True},
    'validation_smoke': smoke_val, 'formal_training_started': False, 'optimizer_steps': 0,
    'weights_read': False, 'weights_statted': False, 'test_inference_performed': False,
    'random_initialization': True, 'outputs_empty': True, 'original_hashes_unchanged': True,
    'sequence_sort_keys': _PLAN['sequence_sort_keys'], 'fixed_hyperparameters': _INPUTS['hyperparameters'],
}
with _REPORT_FILE.open('x', encoding='utf-8', newline='\\n') as _handle:
    _handle.write(json.dumps(_SUMMARY, indent=2, ensure_ascii=False, allow_nan=False) + '\\n')
print('SMOKE_SUMMARY=' + str(_REPORT_FILE))
print('SMOKE_PASS: no optimizer update, formal training, saved weights or test inference')
raise SystemExit(0)
'''


def write_new(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)


def prepare(root, inputs_path):
    root, inputs_path = plain(root), plain(inputs_path)
    if not root.is_dir() or not inputs_path.is_file():
        raise ValueError('Project root or sealed input JSON is missing')
    raw = inputs_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUTS_SHA256:
        raise RuntimeError('Input plan differs from the reviewed Amazon smoke JSON')
    inputs = json.loads(raw.decode('utf-8-sig'))
    if (inputs['version'] != VERSION or inputs['seed'] != 42 or set(inputs['datasets']) != set(DATASETS)
            or len(inputs['sources_sha256']) != 5 or len(inputs['data_sha256']) != 9):
        raise ValueError('Unexpected Amazon smoke plan')
    expected_sources = {inputs['template_source'], inputs['model_source']}
    expected_data, reference_hashes = set(), {}
    for tag, plan in inputs['datasets'].items():
        if plan['dataset'] != DATASETS[tag] or plan['seed'] != 42 or set(plan['control_reference']['files_sha256']) != SAVED_FILES:
            raise ValueError('Unexpected Control reference descriptor')
        expected_sources.add(plan['control_trainer_source'])
        expected_data.update((plan['train_path'], plan['validation_path'], plan['test_path']))
        reference_hashes.update({plan['control_reference']['model_dir'] + '/' + name: digest
                                for name, digest in plan['control_reference']['files_sha256'].items()})
    if set(inputs['sources_sha256']) != expected_sources or set(inputs['data_sha256']) != expected_data or len(reference_hashes) != 15:
        raise ValueError('Unexpected source/data/reference closure')
    verified = {}
    for relative, digest in {**inputs['sources_sha256'], **inputs['data_sha256'], **reference_hashes}.items():
        verified[relative] = checked_sha256(root, relative, digest)
    sources = {}
    for relative, digest in inputs['sources_sha256'].items():
        blob = checked_file(root, relative).read_bytes()
        if hashlib.sha256(blob).hexdigest() != digest:
            raise RuntimeError('Source changed after hashing: ' + relative)
        sources[relative] = blob.decode('utf-8-sig')
        ast.parse(sources[relative])
    now = datetime.datetime.now(datetime.timezone.utc)
    stage = root / ('kais_lbsp_amazon_smoke_v1_' + now.strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:12])
    model = frozen_model(sources[inputs['model_source']], sources[inputs['template_source']])
    model_hash = hashlib.sha256(model.encode('utf-8')).hexdigest()
    generated = {'scripts/model_definition.py': model}
    for tag, plan in inputs['datasets'].items():
        generated['scripts/smoke_' + tag + '_lbsp.py'] = generated_smoke(sources[inputs['template_source']],
            sources[plan['control_trainer_source']], tag, inputs, root, stage, model_hash)
    for text in generated.values():
        ast.parse(text, feature_version=(3, 10))
    hashes = {name: hashlib.sha256(text.encode('utf-8')).hexdigest() for name, text in generated.items()}
    manifest = {'version': VERSION, 'status': 'PREPARED_ONLY_NO_SMOKE_EXECUTED', 'created_utc': now.isoformat(),
        'root': str(root), 'stage': str(stage), 'inputs': inputs, 'inputs_sha256': INPUTS_SHA256,
        'generated_sha256': hashes, 'sources_sha256': inputs['sources_sha256'], 'data_sha256': inputs['data_sha256'],
        'reference_files_sha256': reference_hashes, 'checkpoint_contents_read': False, 'checkpoint_statted': False,
        'scope': inputs['scope'], 'formal_training_started': False,
        'preserved': 'Five frozen scientific ASTs; AllPositionR1Dataset/training_loss/evaluate and safe forward ASTs; pinned Control ordering blocks.',
        'load_eval_change': 'In-memory checked size alias and direct_pooling only; original CSV files remain unchanged.',
        'limits': ['Smoke is random initialization only, without optimizer updates or test inference.',
                   'No trained effectiveness or full-panel stability claims follow from smoke checks.']}
    stage.mkdir(exist_ok=False)
    for child in ('scripts', 'outputs', 'verification', 'manifests'):
        (stage / child).mkdir(exist_ok=False)
    for relative, text in generated.items():
        write_new(stage / relative, text)
    write_new(stage / MANIFEST_PATH, json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    write_new(stage / 'verification/PREPARED.json', json.dumps({'status': 'PREPARED_ONLY', 'hash_verified': verified,
        'research_code_executed': False, 'weights_read': False, 'weights_statted': False}, indent=2) + '\n')
    print('WORKDIR=' + str(stage)); print('STATUS=PREPARED_ONLY_NO_SMOKE_EXECUTED')
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).with_name('lbsp_amazon_smoke_inputs_20261023.json'))
    args = parser.parse_args()
    prepare(args.root, args.inputs)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, SyntaxError) as exc:
        print('PREPARATION_FAILED=' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
