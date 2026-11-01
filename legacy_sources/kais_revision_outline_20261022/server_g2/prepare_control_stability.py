#!/usr/bin/env python3
"""Prepare fixed-model Control stability evaluation; standard library only.

Preparation hashes inputs and saved metadata, inspects checkpoint presence,
and writes a unique stage. It never executes research code or reads weights.
"""
import argparse
import ast
from collections import Counter
import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import uuid

INPUTS_SHA256 = '8e8d13ded1ded7729e7f951b97d65ef3bda84614f4f4a9ad881bec68f7c81458'
SEEDS = (42, 2026, 3407)
PERMUTATIONS = (101, 202, 303, 404, 505)
SAVED_FILES = {'config.json', 'metrics.json', 'training_log.csv', 'test_per_sample_predictions.csv', 'item_id_map.csv'}
DATASETS = {'ml1m': 'MovieLens-1M', 'beauty': 'Amazon_Beauty', 'sports': 'Amazon_Sports', 'toys': 'Amazon_Toys'}


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
    if file_signature(before) != file_signature(after) or file_signature(before) != file_signature(checked_file(root, relative).stat()):
        raise RuntimeError('Input changed during hashing: ' + relative)
    if digest.hexdigest() != expected:
        raise RuntimeError('SHA256 mismatch: ' + relative)
    return digest.hexdigest()


def row_sequence(row):
    values = json.loads(row['sequence_items'])
    if not isinstance(values, list) or not values or any(isinstance(v, (dict, list, bool)) or v is None for v in values):
        raise ValueError('Invalid input sequence')
    return tuple(str(v) for v in values)


def last_bucket_size(row):
    present = [row[k] for k in ('last_bucket_size', 'last_history_bucket_size') if k in row]
    if not present or any(not re.fullmatch(r'[0-9]+', str(v)) for v in present):
        raise ValueError('Missing/invalid last-bucket size')
    sizes = [int(v) for v in present]
    if len(set(sizes)) != 1 or not 1 <= sizes[0] <= len(row_sequence(row)):
        raise ValueError('Last-bucket size outside history or inconsistent')
    return sizes[0]


def indexed_rows(records):
    result = {}
    for row in records:
        sid = str(row['sample_id'])
        if not sid or sid in result or not str(row['user_id']) or not str(row['target_item']):
            raise ValueError('Duplicate/empty sample identity')
        sequence = row_sequence(row)
        if len(sequence) > 50 or str(row['target_item']) in sequence:
            raise ValueError('Invalid history length or target in history')
        last_bucket_size(row)
        result[sid] = row
    if not result:
        raise ValueError('Empty input panel')
    return result


def verify_permutation(original, records):
    variant = indexed_rows(records)
    if set(variant) != set(original):
        raise ValueError('Permutation sample IDs differ')
    for sid, base in original.items():
        row = variant[sid]
        if any(str(row[k]) != str(base[k]) for k in ('user_id', 'target_item')):
            raise ValueError('Permutation user/target changed: ' + sid)
        for key in ('target_raw_line', 'target_timestamp', 'target_bucket'):
            if key in base and key in row and str(base[key]) != str(row[key]):
                raise ValueError('Permutation target metadata changed: ' + sid)
        a, b = row_sequence(base), row_sequence(row)
        k = last_bucket_size(base)
        if last_bucket_size(row) != k or len(a) != len(b) or a[:-k] != b[:-k] or Counter(a[-k:]) != Counter(b[-k:]):
            raise ValueError('Permutation prefix/length/last-bucket multiset changed: ' + sid)
    return variant


def sequence_sha256(sequence):
    return hashlib.sha256(json.dumps(sequence, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


def panel_labels(original, variants):
    if set(variants) != {str(seed) for seed in PERMUTATIONS}:
        raise ValueError('Exactly five permutation panels are required')
    labels = []
    for sid, row in original.items():
        sequences = [row_sequence(variants[str(seed)][sid]) for seed in PERMUTATIONS]
        distinct = len(set(sequences))
        label = {'sample_id': sid, 'user_id': str(row['user_id']), 'target_item': str(row['target_item']),
                 'last_bucket_size': last_bucket_size(row), 'affected': int(distinct >= 2),
                 'distinct_sequences_5perm': distinct, 'original_sequence_sha256': sequence_sha256(row_sequence(row))}
        label.update({'perm_seed' + str(seed) + '_sequence_sha256': sequence_sha256(seq)
                      for seed, seq in zip(PERMUTATIONS, sequences)})
        labels.append(label)
    return labels


def verify_original_top10(new_records, saved_records):
    saved = {}
    for row in saved_records:
        sid = str(row['sample_id'])
        if sid in saved:
            raise ValueError('Duplicate saved prediction sample')
        saved[sid] = row
    if set(saved) != {str(r['sample_id']) for r in new_records} or len(saved) != len(new_records):
        raise ValueError('Saved/base prediction sample panel differs')
    mismatches = 0
    for row in new_records:
        prior = saved[str(row['sample_id'])]
        if any(str(row[k]) != str(prior[k]) for k in ('user_id', 'target_item')):
            raise ValueError('Saved/base user or target differs')
        a, b = json.loads(row['top10_item_idx']), json.loads(prior['top10_item_idx'])
        ra = [str(x) for x in json.loads(row['top10_raw_item_id'])]
        rb = [str(x) for x in json.loads(prior['top10_raw_item_id'])]
        if len(a) != 10 or len(b) != 10 or len(set(a)) != 10 or len(set(b)) != 10:
            raise ValueError('Invalid base/saved Top10')
        mismatches += int(a != b or ra != rb)
    if mismatches:
        raise RuntimeError('Saved Top10 exact reproduction failed: ' + str(mismatches))
    return mismatches


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Unexpected template structure: ' + old[:65])
    return source.replace(old, new, 1)


def replace_assignment(source, name, expression):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(nodes) != 1:
        raise RuntimeError('Expected one assignment: ' + name)
    return replace_once(source, ast.get_source_segment(source, nodes[0]), name + ' = ' + expression)


def frozen_model(source):
    found = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'SASRecControl']
    if len(found) != 1:
        raise RuntimeError('Expected one SASRecControl class')
    result = '# Frozen class only; no original training statements.\n' + ast.get_source_segment(source, found[0]) + '\n'
    if ast.dump(found[0]) != ast.dump(ast.parse(result).body[0]):
        raise RuntimeError('Model class AST changed')
    return result


def generated_evaluator(source, tag, plan, root, stage, inputs_hash, model_hash):
    protected = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in ('read_samples', 'evaluate')}
    for name, value in {'ROOT': '_STABILITY_ROOT', 'BASE': '_STABILITY_STAGE', 'MODEL_SEED': 'args.seed',
                        'MODEL_DIR': 'ROOT / _RUN["model_dir"]', 'MODEL_SCRIPT': '_STABILITY_STAGE / "scripts/model_definition.py"',
                        'CHECKPOINT_PATH': '_CHECKPOINT_PATH', 'ORIGINAL_PATH': 'ROOT / _PLAN["original_path"]',
                        'OUTPUT_DIR': '_STABILITY_STAGE / "outputs" / f"{_TAG}_seed{MODEL_SEED}"'}.items():
        source = replace_assignment(source, name, value)
    source, count = re.subn(r'if int\(checkpoint\["epoch"\]\) != [0-9]+:', 'if int(checkpoint["epoch"]) != _RUN["best_epoch"]:', source)
    if count != 1:
        raise RuntimeError('Expected one hardcoded checkpoint epoch guard')
    source, count = re.subn(r'if abs\(ndcg - [0-9.eE+-]+\) >= 1e-6:', 'if abs(ndcg - _RUN["test_ndcg10"]) >= 1e-6:', source)
    if count != 1:
        raise RuntimeError('Expected one hardcoded NDCG guard')
    source = replace_once(source, 'PERM_ROOT / f"perm_seed{seed}" / "test.csv"', 'ROOT / _PLAN["permutation_paths"][str(seed)]')
    source = replace_once(source, '''input_path = (
        PERM_ROOT
        / f"perm_seed{seed}"
        / "test.csv"
    )''', 'input_path = ROOT / _PLAN["permutation_paths"][str(seed)]')
    source = replace_once(source, 'model = SASRecControl(', '''if file_signature(checked_file(ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"]).stat()) != _CHECKPOINT_SIGNATURE:
    raise RuntimeError("Checkpoint changed during loading")
if checkpoint.get("dataset") != DATASET:
    raise RuntimeError("Checkpoint dataset differs from reviewed run")
for _name in ("max_len", "hidden_size", "num_heads", "num_blocks", "dropout"):
    if checkpoint.get(_name) != _REFERENCE_CONFIG.get(_name):
        raise RuntimeError("Checkpoint architecture differs from saved config: " + _name)

model = SASRecControl(''')
    source = replace_once(source, '''reference_ids = (
    original_df["sample_id"].tolist()
)''', '''reference_ids = (
    original_df["sample_id"].tolist()
)
_ORIGINAL_ROWS = indexed_rows(original_df.to_dict(orient="records"))
_VARIANTS = {}''')
    source = replace_once(source, 'print("ORIGINAL STRICT REPRODUCTION PASS")', '''_TOP10_MISMATCHES = verify_original_top10(
    original_pred.to_dict(orient="records"), saved.to_dict(orient="records"))
print("ORIGINAL STRICT REPRODUCTION PASS: ranks and exact ordered Top10")''')
    source = replace_once(source, '''    result = evaluate(
        df,''', '''    _VARIANTS[str(seed)] = verify_permutation(_ORIGINAL_ROWS, df.to_dict(orient="records"))
    result = evaluate(
        df,''')
    source = replace_once(source, '''original_pred.to_csv(''', '''_LABELS = panel_labels(_ORIGINAL_ROWS, _VARIANTS)
pd.DataFrame(_LABELS).to_csv(OUTPUT_DIR / "panel_labels.csv", index=False, encoding="utf-8-sig")

original_pred.to_csv(''')
    source = replace_once(source, 'metadata = {\n    "dataset": DATASET,', '''metadata = {
    "input_manifest_sha256": _INPUTS_SHA256,
    "preparation_manifest_sha256": hashlib.sha256(
        (_STABILITY_STAGE / "manifests/control_stability_manifest.json").read_bytes()).hexdigest(),
    "generated_evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "model_dir": str(MODEL_DIR),
    "checkpoint_path": str(CHECKPOINT_PATH),
    "checkpoint_stat_before_inference": list(_CHECKPOINT_SIGNATURE),
    "reference_files_sha256": _RUN["files_sha256"],
    "data_sha256": _DATA_SHA256,
    "model_definition_sha256": _MODEL_SHA256,
    "original_rank_mismatches": mismatch,
    "original_top10_mismatches": _TOP10_MISMATCHES,
    "affected_definition": "at least two distinct actual input sequences among five permutations; original excluded",
    "affected_samples": sum(row["affected"] for row in _LABELS),
    "sequence_hash_definition": "SHA256(json.dumps(tuple(str(item) for item in sequence), separators=(',', ':'), ensure_ascii=False).encode('utf-8'))",
    "dataset": DATASET,''')
    helpers = (plain, checked_file, file_signature, checked_sha256, row_sequence, last_bucket_size,
               indexed_rows, verify_permutation, sequence_sha256, panel_labels, verify_original_top10)
    prefix = ('#!/usr/bin/env python3\n# Evaluation only: no training/optimizer steps.\n'
              'import argparse\nfrom collections import Counter\nimport hashlib\nimport json\nimport os\nimport re\nimport stat\nfrom pathlib import Path, PurePosixPath\n\n'
              + '\n\n'.join(inspect.getsource(f) for f in helpers) + '\n\n'
              + 'PERMUTATIONS = ' + repr(PERMUTATIONS) + '\n'
              + "parser = argparse.ArgumentParser(description='Fixed-model last-bucket stability evaluation')\n"
              + "parser.add_argument('--seed', type=int, choices=[42, 2026, 3407], required=True)\nargs = parser.parse_args()\n"
              + '_STABILITY_ROOT = plain(Path(' + repr(str(root)) + '))\n_STABILITY_STAGE = plain(Path(' + repr(str(stage)) + '))\n'
              + "if Path(__file__).resolve(strict=False).parent != _STABILITY_STAGE / 'scripts':\n    raise RuntimeError('Use the original generated evaluator stage')\n"
              + '_TAG = ' + repr(tag) + '\n_PLAN = ' + repr(plan) + '\n_RUN = _PLAN["runs"][str(args.seed)]\n'
              + '_INPUTS_SHA256 = ' + repr(inputs_hash) + '\n_MODEL_SHA256 = ' + repr(model_hash) + '\n'
              + '_DATA_SHA256 = ' + repr(plan['_data_sha256']) + '\n'
              + 'for _relative, _expected in _DATA_SHA256.items():\n    checked_sha256(_STABILITY_ROOT, _relative, _expected)\n'
              + 'for _name, _expected in _RUN["files_sha256"].items():\n    checked_sha256(_STABILITY_ROOT, _RUN["model_dir"] + "/" + _name, _expected)\n'
              + "checked_sha256(_STABILITY_STAGE, 'scripts/model_definition.py', _MODEL_SHA256)\n"
              + '_CHECKPOINT_PATH = checked_file(_STABILITY_ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"])\n'
              + '_CHECKPOINT_SIGNATURE = file_signature(_CHECKPOINT_PATH.stat())\n'
              + '_REFERENCE_CONFIG = json.loads(checked_file(_STABILITY_ROOT, _RUN["model_dir"] + "/config.json").read_text(encoding="utf-8-sig"))\n'
              + "print('INTEGRITY PASS: six fixed input CSVs, five reviewed run files and frozen class; weights not yet read')\n\n")
    result = prefix + source
    tree = ast.parse(result)
    after = {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in protected}
    if protected != after or set(protected) != {'read_samples', 'evaluate'}:
        raise RuntimeError('Original sample loader/inference function AST changed')
    return result


def write_new(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)


def prepare(root, inputs_path):
    root, inputs_path = plain(root), plain(inputs_path)
    if not root.is_dir() or not inputs_path.is_file():
        raise ValueError('Project root/input JSON missing')
    raw = inputs_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUTS_SHA256:
        raise RuntimeError('Input plan differs from the reviewed JSON')
    inputs = json.loads(raw.decode('utf-8-sig'))
    if (inputs.get('version') != 'control_stability_v1' or inputs.get('training_seeds') != list(SEEDS)
            or inputs.get('permutation_seeds') != list(PERMUTATIONS) or set(inputs.get('datasets', {})) != set(DATASETS)):
        raise ValueError('Unexpected evaluation plan schema')
    expected_sources = {inputs['model_source']} | {d['eval_source'] for d in inputs['datasets'].values()}
    if set(inputs['sources_sha256']) != expected_sources or len(expected_sources) != 5:
        raise ValueError('Source set differs from four evaluators plus one model')
    all_data = set()
    for tag, plan in inputs['datasets'].items():
        if (plan['dataset'] != DATASETS[tag] or set(plan['runs']) != {str(s) for s in SEEDS}
                or set(plan['permutation_paths']) != {str(s) for s in PERMUTATIONS} or plan['expected_samples'] < 1):
            raise ValueError('Dataset/run/permutation identity differs')
        all_data.update([plan['original_path']] + list(plan['permutation_paths'].values()))
        for run in plan['runs'].values():
            if (set(run['files_sha256']) != SAVED_FILES or run['checkpoint'] != 'matched_control_sasrec_best.pt'
                    or not isinstance(run['best_epoch'], int) or run['best_epoch'] < 1
                    or not math.isfinite(run['test_ndcg10']) or not 0 <= run['test_ndcg10'] <= 1):
                raise ValueError('Unexpected saved-run descriptor')
    if set(inputs['data_sha256']) != all_data or len(all_data) != 24:
        raise ValueError('Exactly twenty-four fixed input CSVs are required')
    verified, sources, checkpoint_presence = {}, {}, {}
    # Complete all hashing and metadata checks before creating any stage.
    for relative, digest in dict(inputs['sources_sha256'], **inputs['data_sha256']).items():
        verified[relative] = checked_sha256(root, relative, digest)
    for relative, digest in inputs['sources_sha256'].items():
        blob = checked_file(root, relative).read_bytes()
        if hashlib.sha256(blob).hexdigest() != digest:
            raise RuntimeError('Source changed after hashing')
        sources[relative] = blob.decode('utf-8-sig')
        ast.parse(sources[relative])
    for tag, plan in inputs['datasets'].items():
        for seed, run in plan['runs'].items():
            for name, digest in run['files_sha256'].items():
                relative = run['model_dir'] + '/' + name
                verified[relative] = checked_sha256(root, relative, digest)
            config = json.loads(checked_file(root, run['model_dir'] + '/config.json').read_text(encoding='utf-8-sig'))
            metrics = json.loads(checked_file(root, run['model_dir'] + '/metrics.json').read_text(encoding='utf-8-sig'))
            test = metrics['test'].get('overall', metrics['test'])
            if (config.get('seed') != int(seed) or metrics.get('seed') != int(seed)
                    or config.get('dataset') != plan['dataset'] or metrics.get('dataset') != plan['dataset']
                    or metrics.get('best_epoch') != run['best_epoch']
                    or not math.isclose(test.get('NDCG@10', test.get('ndcg10', -1)), run['test_ndcg10'], abs_tol=1e-12)
                    or test.get('N', test.get('samples')) != plan['expected_samples']):
                raise ValueError('Saved metadata disagrees with the audited input plan')
            cp = checked_file(root, run['model_dir'] + '/' + run['checkpoint'])
            info = cp.stat()
            if info.st_size < 1:
                raise ValueError('Empty checkpoint')
            checkpoint_presence[tag + '_seed' + seed] = {'path': str(cp), 'bytes': info.st_size,
                                                        'mtime_ns': info.st_mtime_ns, 'contents_checked': False}
    now = datetime.datetime.now(datetime.timezone.utc)
    stage = root / ('kais_control_stability_v1_' + now.strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:12])
    model = frozen_model(sources[inputs['model_source']])
    model_hash = hashlib.sha256(model.encode('utf-8')).hexdigest()
    generated = {'scripts/model_definition.py': model}
    for tag, plan in inputs['datasets'].items():
        copy = dict(plan)
        paths = [plan['original_path']] + list(plan['permutation_paths'].values())
        copy['_data_sha256'] = {relative: inputs['data_sha256'][relative] for relative in paths}
        generated['scripts/eval_' + tag + '_stability.py'] = generated_evaluator(
            sources[plan['eval_source']], tag, copy, root, stage, INPUTS_SHA256, model_hash)
    hashes = {relative: hashlib.sha256(source.encode('utf-8')).hexdigest() for relative, source in generated.items()}
    manifest = {'version': 'control_stability_v1', 'status': 'PREPARED_ONLY_NO_INFERENCE', 'created_utc': now.isoformat(),
                'root': str(root), 'stage': str(stage), 'inputs': inputs, 'inputs_sha256': INPUTS_SHA256,
                'generated_sha256': hashes, 'sources_sha256': inputs['sources_sha256'], 'data_sha256': inputs['data_sha256'],
                'checkpoint_presence': checkpoint_presence,
                'scope': 'Fixed-model evaluation only; no training, data rewriting or checkpoint-content access during preparation.',
                'preserved': 'Original read_samples/evaluate function ASTs, model class AST, masking, ranks, Top10, strict panels and five permutations.',
                'added_checks': ['Dynamic reviewed epoch/NDCG guards', 'Exact baseline ranks and ordered Top10 reproduction',
                                 'User/target/length/prefix/last-bucket multiset invariants before each permutation inference',
                                 'Affected labels from five distinct input sequences, with sequence hashes and no histories saved']}
    stage.mkdir(exist_ok=False)
    for child in ('scripts', 'outputs', 'verification', 'manifests'):
        (stage / child).mkdir(exist_ok=False)
    for relative, source in generated.items():
        write_new(stage / relative, source)
    write_new(stage / 'manifests/control_stability_manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    write_new(stage / 'verification/PREPARED.json', json.dumps({'status': 'PREPARED_ONLY', 'hash_verified': verified,
              'inference_function_ast': 'UNCHANGED', 'model_class_ast': 'UNCHANGED', 'checkpoint_contents_read': False}, indent=2) + '\n')
    print('WORKDIR=' + str(stage))
    print('STATUS=PREPARED_ONLY_NO_INFERENCE')
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).with_name('control_stability_inputs_20261023.json'))
    args = parser.parse_args()
    prepare(args.root, args.inputs)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, SyntaxError) as exc:
        print('PREPARATION_FAILED=' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
