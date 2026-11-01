#!/usr/bin/env python3
"""Prepare nine isolated Amazon LBSP trainings; standard-library preparation only.

Preparation hashes pinned input bytes without parsing interactions or executing
research definitions. It never accesses checkpoints. The explicit training CLI
runs only when the user launches a generated trainer on the server.
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

INPUTS_SHA256 = 'aad9c9a404f04df9b355d6aa6b4ca0b26905d9763ee9fb3915c9f3e102682ff2'
VERSION = 'lbsp_amazon_train_v1'
MANIFEST_PATH = 'manifests/lbsp_amazon_train_manifest.json'
SEEDS = (42, 2026, 3407)
DATASETS = {'beauty': 'Amazon_Beauty', 'sports': 'Amazon_Sports', 'toys': 'Amazon_Toys'}
SAVED_FILES = {'config.json', 'metrics.json', 'training_log.csv', 'test_per_sample_predictions.csv', 'item_id_map.csv'}
FROZEN_NAMES = ('build_r1_group_positions_from_buckets', 'build_r1_group_positions_from_size', 'canonicalize_last_bucket_items', 'RecentBucketSetR1SASRec', 'predict_batch_masked')

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


def write_new(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)


def formal_tail(template):
    marker = '# Smoke 不更新参数；这里重置训练随机状态。'
    if template.count(marker) != 1:
        raise RuntimeError('Original formal RNG-reset boundary changed')
    return marker + template.split(marker, 1)[1]


def generated_trainer(smoke_source, template, control_source, tag, inputs, root, stage, model_hash):
    """Keep the proved loader/model body; append the original formal tail."""
    smoke_tree = ast.parse(smoke_source)
    imports = [n for n in smoke_tree.body if isinstance(n, ast.Import) and any(a.name == 'argparse' for a in n.names)]
    if len(imports) != 1:
        raise RuntimeError('Original template start is ambiguous in smoke source')
    body = '\n'.join(smoke_source.splitlines()[imports[0].lineno - 1:]) + '\n'
    # evaluate is the final definition before any Smoke A/B/C execution.
    end = definition(body, 'evaluate').end_lineno
    body = '\n'.join(body.splitlines()[:end]) + '\n'
    remove = []
    for node in ast.parse(body).body:
        is_parser_assign = isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ('parser', 'args') for t in node.targets)
        is_parser_call = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                          and isinstance(node.value.func, ast.Attribute)
                          and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'parser')
        if is_parser_assign or is_parser_call:
            remove.append(node)
    if len(remove) != 3:
        raise RuntimeError('Unexpected original smoke CLI')
    lines = body.splitlines()
    for node in reversed(remove):
        del lines[node.lineno - 1:node.end_lineno]
    body = '\n'.join(lines) + '\n'
    for name, expression in {
            'ROOT': '_TRAIN_ROOT', 'BASE': '_TRAIN_STAGE', 'SEED': 'args.seed',
            'R1_SOURCE': '_TRAIN_STAGE / "scripts/model_definition.py"',
            'OUTPUT_DIR': '_TRAIN_STAGE / "outputs" / f"' + tag + '_lbsp_seed{args.seed}"'}.items():
        body = replace_assignment(body, name, expression)
    body = replace_once(body, 'AMAZON LBSP SMOKE ONLY: ' + tag, 'AMAZON LBSP FORMAL TRAINING: ' + tag)
    for name in ('AllPositionR1Dataset', 'training_loss', 'evaluate', 'predict_batch_masked'):
        if ast.dump(definition(body, name)) != ast.dump(definition(template, name)):
            raise RuntimeError('Scientific definition changed: ' + name)
    loader = definition(body, 'load_eval')
    insertions = [n for n in loader.body if ast.unparse(n) == 'df = normalize_eval_frame(df)']
    if len(insertions) != 1 or loader.body[1] is not insertions[0]:
        raise RuntimeError('Approved loader normalization insertion changed')
    loader.body.remove(insertions[0])
    if ast.dump(loader) != ast.dump(definition(template, 'load_eval')):
        raise RuntimeError('Loader changed beyond the approved normalization adapter')
    if sorting_block(control_source, tag).strip() not in body:
        raise RuntimeError('Pinned Control temporal ordering block changed')
    tail = formal_tail(template)
    marker = '(OUTPUT_DIR / "config.json").write_text('
    tail = replace_once(tail, marker, 'config["amazon_lbsp_training"] = dict(_PROVENANCE)\n\n' + marker)
    # After removing this operational provenance assignment, every original
    # formal-training statement, including all four RNG resets, stays identical.
    rebuilt = ast.parse(tail)
    rebuilt.body = [n for n in rebuilt.body if not (isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "config['amazon_lbsp_training']" for t in n.targets))]
    if ast.dump(rebuilt) != ast.dump(ast.parse(formal_tail(template))):
        raise RuntimeError('Original formal-training AST changed')
    result = runtime_prelude(tag, inputs, root, stage, model_hash) + '\n' + body + '\n' + tail + '\n' + completion_code()
    tree = ast.parse(result, feature_version=(3, 10))
    if any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'smoke_loader' for t in n.targets) for n in tree.body):
        raise RuntimeError('Smoke execution survived in the formal trainer')
    seeds = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SEED' for t in n.targets)]
    if len(seeds) != 1 or ast.unparse(seeds[0].value) != 'args.seed':
        raise RuntimeError('Training seed is not selected by the explicit CLI')
    return result


def runtime_prelude(tag, inputs, root, stage, model_hash):
    helpers = '\n\n'.join(inspect.getsource(fn) for fn in (plain, checked_file, file_signature, checked_sha256,
                                      normalize_eval_records, normalize_eval_frame))
    text = '''# Generated formal trainer. Standard-library identity checks precede Torch imports.
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import math
parser = argparse.ArgumentParser(description='Explicit Amazon LBSP formal training')
parser.add_argument('--mode', choices=['train'], required=True)
parser.add_argument('--seed', type=int, choices=[42, 2026, 3407], required=True)
args = parser.parse_args()
'''
    text += helpers + '\n'
    text += '_TRAIN_ROOT = Path(' + repr(str(root)) + ')\n_TRAIN_STAGE = Path(' + repr(str(stage)) + ')\n'
    text += '_TAG = ' + repr(tag) + '\n_INPUTS = json.loads(' + repr(json.dumps(inputs, ensure_ascii=False)) + ')\n'
    text += '_INPUTS_SHA256 = ' + repr(INPUTS_SHA256) + '\n_MODEL_SHA256 = ' + repr(model_hash) + '\n'
    text += '''_PLAN = _INPUTS['datasets'][_TAG]
_MANIFEST_RELATIVE = 'manifests/lbsp_amazon_train_manifest.json'
_MANIFEST_BYTES = checked_file(_TRAIN_STAGE, _MANIFEST_RELATIVE).read_bytes()
_MANIFEST_HASH = hashlib.sha256(_MANIFEST_BYTES).hexdigest()
_MANIFEST = json.loads(_MANIFEST_BYTES.decode('utf-8-sig'))
if (_MANIFEST['version'] != 'lbsp_amazon_train_v1' or _MANIFEST['inputs'] != _INPUTS
        or _MANIFEST['inputs_sha256'] != _INPUTS_SHA256 or _MANIFEST['root'] != str(_TRAIN_ROOT)
        or _MANIFEST['stage'] != str(_TRAIN_STAGE)):
    raise RuntimeError('Formal preparation manifest identity differs')
_GENERATED_RELATIVE = 'scripts/train_' + _TAG + '_lbsp.py'
_GENERATED_HASH = _MANIFEST['generated_sha256'][_GENERATED_RELATIVE]
checked_sha256(_TRAIN_STAGE, _GENERATED_RELATIVE, _GENERATED_HASH)
checked_sha256(_TRAIN_STAGE, 'scripts/model_definition.py', _MODEL_SHA256)
_DATA_HASHES = {path: _INPUTS['data_sha256'][path] for path in
                (_PLAN['train_path'], _PLAN['validation_path'], _PLAN['test_path'])}
_REF_HASHES = _PLAN['control_reference']['files_sha256']
_OLD_HASHES = dict(_INPUTS['sources_sha256'])
_OLD_HASHES.update(_DATA_HASHES)
_OLD_HASHES.update({_PLAN['control_reference']['model_dir'] + '/' + name: digest
                    for name, digest in _REF_HASHES.items()})
_SMOKE_STAGE = plain(Path(_INPUTS['smoke_reference']['stage']))
_SMOKE_STAGE.relative_to(_TRAIN_ROOT)
for _relative, _expected in _INPUTS['smoke_reference']['files_sha256'].items():
    _OLD_HASHES[(_SMOKE_STAGE.relative_to(_TRAIN_ROOT) / _relative).as_posix()] = _expected
for _relative, _expected in _OLD_HASHES.items():
    checked_sha256(_TRAIN_ROOT, _relative, _expected)
for _name, _expected in _REF_HASHES.items():
    checked_sha256(_TRAIN_STAGE, 'references/' + _TAG + '_seed42/' + _name, _expected)
_COMPLETION_FILE = plain(_TRAIN_STAGE / 'verification') / ('completed_' + _TAG + '_seed' + str(args.seed) + '.json')
if _COMPLETION_FILE.exists():
    raise RuntimeError('Completion record exists; refusing to rerun this training')
_PROVENANCE = {
    'version': 'lbsp_amazon_train_v1', 'input_manifest_sha256': _INPUTS_SHA256,
    'preparation_manifest_sha256': _MANIFEST_HASH, 'generated_script_sha256': _GENERATED_HASH,
    'model_definition_sha256': _MODEL_SHA256, 'sources_sha256': _INPUTS['sources_sha256'],
    'data_sha256': _DATA_HASHES, 'reference_files_sha256': _REF_HASHES,
    'control_reference': _PLAN['control_reference'], 'smoke_reference': _INPUTS['smoke_reference'],
    'sequence_sort_keys': _PLAN['sequence_sort_keys'], 'scientific_definitions_unchanged': True,
    'formal_rng_reset_preserved': True, 'smoke_execution_omitted': True,
}
'''
    return text


def completion_code():
    return '''# Hash only the five saved metadata files; never hash checkpoint contents.
for _relative, _expected in _OLD_HASHES.items():
    checked_sha256(_TRAIN_ROOT, _relative, _expected)
checked_sha256(_TRAIN_STAGE, _GENERATED_RELATIVE, _GENERATED_HASH)
checked_sha256(_TRAIN_STAGE, 'scripts/model_definition.py', _MODEL_SHA256)
checked_sha256(_TRAIN_STAGE, _MANIFEST_RELATIVE, _MANIFEST_HASH)
for _name, _expected in _REF_HASHES.items():
    checked_sha256(_TRAIN_STAGE, 'references/' + _TAG + '_seed42/' + _name, _expected)
_OUTPUT_METADATA = {name: checked_sha256(OUTPUT_DIR, name, None) for name in
    ('config.json', 'metrics.json', 'training_log.csv', 'test_per_sample_predictions.csv', 'item_id_map.csv')}
_COMPLETION = dict(_PROVENANCE)
_COMPLETION.update({
    'tag': _TAG, 'dataset': DATASET, 'seed': SEED, 'status': 'COMPLETED', 'output_dir': str(OUTPUT_DIR),
    'all_original_hashes_verified': True, 'originals_unchanged': True, 'output_metadata_sha256': _OUTPUT_METADATA,
    'best_epoch': best_epoch, 'checkpoint': 'r1_meanpool_best.pt',
    'counts': {'num_items': NUM_ITEMS, 'training_positions': LEGAL_POSITIONS,
               'validation_samples': len(val_df), 'test_samples': len(test_df)},
})
with _COMPLETION_FILE.open('x', encoding='utf-8', newline='\\n') as _handle:
    _handle.write(json.dumps(_COMPLETION, indent=2, ensure_ascii=False, allow_nan=False) + '\\n')
print('LBSP_AMAZON_TRAIN_COMPLETED=' + str(_COMPLETION_FILE), flush=True)
'''


def prepare(root, inputs_path):
    root, inputs_path = plain(root), plain(inputs_path)
    if not root.is_dir() or not inputs_path.is_file():
        raise ValueError('Project root or sealed input JSON is missing')
    raw = inputs_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUTS_SHA256:
        raise RuntimeError('Input plan differs from the reviewed Amazon formal-training JSON')
    inputs = json.loads(raw.decode('utf-8-sig'))
    if (inputs['version'] != VERSION or inputs['formal_seeds'] != list(SEEDS)
            or set(inputs['datasets']) != set(DATASETS) or len(inputs['sources_sha256']) != 5
            or len(inputs['data_sha256']) != 9):
        raise ValueError('Unexpected Amazon formal training plan')
    expected_sources = {inputs['template_source'], inputs['model_source']}
    expected_data, references = set(), {}
    for tag, plan in inputs['datasets'].items():
        if (plan['dataset'] != DATASETS[tag] or plan['seed'] != 42
                or set(plan['control_reference']['files_sha256']) != SAVED_FILES):
            raise ValueError('Unexpected fixed Control reference')
        expected_sources.add(plan['control_trainer_source'])
        expected_data.update((plan['train_path'], plan['validation_path'], plan['test_path']))
        references.update({plan['control_reference']['model_dir'] + '/' + name: digest
                           for name, digest in plan['control_reference']['files_sha256'].items()})
    if (expected_sources != set(inputs['sources_sha256']) or expected_data != set(inputs['data_sha256'])
            or len(references) != 15):
        raise ValueError('Unexpected fixed source/data/reference closure')
    smoke_reference = inputs['smoke_reference']
    smoke_stage = plain(Path(smoke_reference['stage']))
    if smoke_stage.parent != root or not smoke_stage.name.startswith('kais_lbsp_amazon_smoke_v1_'):
        raise ValueError('Audited smoke stage is outside the project root')
    smoke_manifest_path = 'manifests/lbsp_amazon_smoke_manifest.json'
    if (len(smoke_reference['files_sha256']) != 13
            or smoke_reference['files_sha256'].get(smoke_manifest_path) != smoke_reference['manifest_sha256']):
        raise ValueError('Unexpected smoke reference inventory')
    verified = {}
    for relative, digest in {**inputs['sources_sha256'], **inputs['data_sha256'], **references}.items():
        verified[relative] = checked_sha256(root, relative, digest)
    for relative, digest in smoke_reference['files_sha256'].items():
        checked_sha256(smoke_stage, relative, digest)
        verified[(smoke_stage.relative_to(root) / relative).as_posix()] = digest
    smoke_manifest = json.loads(checked_file(smoke_stage, smoke_manifest_path).read_text(encoding='utf-8-sig'))
    old_inputs = smoke_manifest['inputs']
    common = set(old_inputs) - {'version', 'seed', 'scope'}
    if (smoke_manifest['version'] != 'lbsp_amazon_smoke_v1' or old_inputs['seed'] != 42
            or smoke_manifest['root'] != str(root) or smoke_manifest['stage'] != str(smoke_stage)
            or any(inputs.get(k) != old_inputs[k] for k in common)):
        raise ValueError('Audited smoke and formal input identities differ')
    for tag, plan in inputs['datasets'].items():
        summary = json.loads(checked_file(smoke_stage, 'verification/smoke_' + tag + '.json').read_text(encoding='utf-8-sig'))
        if (summary.get('status') != 'SMOKE_PASS' or summary.get('dataset') != plan['dataset']
                or summary.get('seed') != 42 or summary.get('all_gradients_finite') is not True
                or summary.get('original_hashes_unchanged') is not True
                or summary.get('preparation_manifest_sha256') != smoke_reference['manifest_sha256']):
            raise ValueError('Required audited smoke summary is inconsistent: ' + tag)
    sources = {}
    for relative, digest in inputs['sources_sha256'].items():
        blob = checked_file(root, relative).read_bytes()
        if hashlib.sha256(blob).hexdigest() != digest:
            raise RuntimeError('Source changed after hashing: ' + relative)
        sources[relative] = blob.decode('utf-8-sig')
        ast.parse(sources[relative])
    now = datetime.datetime.now(datetime.timezone.utc)
    stage = root / ('kais_lbsp_amazon_train_v1_' + now.strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:12])
    model = frozen_model(sources[inputs['model_source']], sources[inputs['template_source']])
    model_hash = hashlib.sha256(model.encode('utf-8')).hexdigest()
    if model_hash != smoke_reference['files_sha256']['scripts/model_definition.py']:
        raise RuntimeError('Frozen scientific model differs from the audited smoke module')
    generated = {'scripts/model_definition.py': model}
    for tag, plan in inputs['datasets'].items():
        relative = 'scripts/smoke_' + tag + '_lbsp.py'
        blob = checked_file(smoke_stage, relative).read_bytes()
        if hashlib.sha256(blob).hexdigest() != smoke_reference['files_sha256'][relative]:
            raise RuntimeError('Audited smoke source changed after hashing')
        generated['scripts/train_' + tag + '_lbsp.py'] = generated_trainer(blob.decode('utf-8-sig'),
            sources[inputs['template_source']], sources[plan['control_trainer_source']], tag, inputs, root, stage, model_hash)
    hashes = {name: hashlib.sha256(text.encode('utf-8')).hexdigest() for name, text in generated.items()}
    for text in generated.values():
        ast.parse(text, feature_version=(3, 10))
    reference_copies = {}
    for tag, plan in inputs['datasets'].items():
        for name, digest in plan['control_reference']['files_sha256'].items():
            blob = checked_file(root, plan['control_reference']['model_dir'] + '/' + name).read_bytes()
            if hashlib.sha256(blob).hexdigest() != digest:
                raise RuntimeError('Control reference changed after verification')
            reference_copies['references/' + tag + '_seed42/' + name] = blob
    manifest = {'version': VERSION, 'status': 'PREPARED_ONLY_NO_TRAINING_EXECUTED', 'created_utc': now.isoformat(),
        'root': str(root), 'stage': str(stage), 'inputs': inputs, 'inputs_sha256': INPUTS_SHA256,
        'generated_sha256': hashes, 'sources_sha256': inputs['sources_sha256'], 'data_sha256': inputs['data_sha256'],
        'smoke_reference': smoke_reference, 'datasets': inputs['datasets'], 'formal_seeds': list(SEEDS),
        'reference_files_sha256': {p: hashlib.sha256(b).hexdigest() for p, b in reference_copies.items()},
        'scientific_definitions_unchanged': True, 'formal_rng_reset_preserved': True, 'smoke_execution_omitted': True,
        'checkpoint_contents_read': False, 'checkpoint_statted': False, 'scope': inputs['scope']}
    stage.mkdir(exist_ok=False)
    for child in ('scripts', 'outputs', 'verification', 'manifests', 'references'):
        (stage / child).mkdir(exist_ok=False)
    for relative, text in generated.items():
        write_new(stage / relative, text)
    for relative, blob in reference_copies.items():
        destination = stage / relative
        destination.parent.mkdir(exist_ok=True)
        with destination.open('xb') as handle:
            handle.write(blob)
    write_new(stage / MANIFEST_PATH, json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    write_new(stage / 'verification/PREPARED.json', json.dumps({'status': 'PREPARED_ONLY', 'hash_verified': verified,
        'research_code_executed': False, 'weights_read': False, 'weights_statted': False}, indent=2) + '\n')
    print('WORKDIR=' + str(stage)); print('STATUS=PREPARED_ONLY_NO_TRAINING_EXECUTED')
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).with_name('lbsp_amazon_train_inputs_20261023.json'))
    args = parser.parse_args()
    prepare(args.root, args.inputs)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, SyntaxError) as exc:
        print('PREPARATION_FAILED=' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)


