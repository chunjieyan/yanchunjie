#!/usr/bin/env python3
"""Prepare fixed-model ML1M LBSP evaluation; stdlib only, no research execution.
Raw-history labels retain the audited Control population before canonicalization.
"""
import argparse
import ast
from collections import Counter
import csv
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

INPUTS_SHA256 = 'a34890d187617f445fb20d6aba70ddf5bd1d81d01c9ffb6099b6e46486675541'
SEEDS = (42, 2026, 3407)
PERMUTATIONS = (101, 202, 303, 404, 505)
SAVED_FILES = {'config.json', 'metrics.json', 'training_log.csv', 'test_per_sample_predictions.csv', 'item_id_map.csv'}
MODEL_NAME = 'RecentBucketSetR1SASRec'
MANIFEST_PATH = 'manifests/lbsp_ml1m_stability_manifest.json'


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

def write_new(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)


def frozen_model(model_source, forward_source):
    requested = ('build_r1_group_positions_from_size', 'canonicalize_last_bucket_items', 'RecentBucketSetR1SASRec')
    selected = []
    for name in requested:
        matches = [n for n in ast.parse(model_source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
        if len(matches) != 1:
            raise RuntimeError('Expected one frozen definition: ' + name)
        selected.append((model_source, matches[0]))
    matches = [n for n in ast.parse(forward_source).body if isinstance(n, ast.FunctionDef) and n.name == 'predict_batch_masked']
    if len(matches) != 1:
        raise RuntimeError('Expected one safe masked forward definition')
    selected.append((forward_source, matches[0]))
    result = '# Frozen definitions only; never import original training top-level code.\n' + '\n\n'.join(
        ast.get_source_segment(text, node) for text, node in selected) + '\n'
    rebuilt = [n for n in ast.parse(result).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    if [ast.dump(node) for _, node in selected] != [ast.dump(node) for node in rebuilt]:
        raise RuntimeError('Frozen model/grouping/canonicalization/forward AST changed')
    return result


def control_labels_match(labels, path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        records = list(csv.DictReader(handle))
    expected = {str(r['sample_id']): {k: str(v) for k, v in r.items()} for r in labels}
    actual = {}
    for row in records:
        sid = row['sample_id']
        if sid in actual:
            raise ValueError('Duplicate Control population sample')
        actual[sid] = row
    if actual != expected:
        raise ValueError('Raw-history labels differ from the reviewed Control population')
    return True


def generated_evaluator(source, plan, control_reference, root, stage, inputs_hash, model_hash):
    names = {'construct', 'read_csv', 'predict_all', 'make_predictions', 'extract'}
    protected = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names}
    if set(protected) != names:
        raise RuntimeError('Original evaluator functions missing')
    assignments = {'ROOT': '_LBSP_ROOT', 'BASE': '_LBSP_STAGE', 'MODEL_DIR': 'ROOT / _RUN["model_dir"]',
                   'CHECKPOINT': '_CHECKPOINT_PATH', 'ORIGINAL_PATH': 'ROOT / _PLAN["original_path"]',
                   'R1_SOURCE': '_LBSP_STAGE / "scripts/model_definition.py"',
                   'FORWARD_SOURCE': '_LBSP_STAGE / "scripts/model_definition.py"',
                   'OUTPUT_DIR': '_LBSP_STAGE / "outputs" / f"ml1m_seed{args.seed}"'}
    for name, expression in assignments.items():
        source = replace_assignment(source, name, expression)
    source = replace_once(source, 'int(ckpt["epoch"]) == 33', 'int(ckpt["epoch"]) == _RUN["best_epoch"]')
    source = replace_once(source, 'int(ckpt["seed"]) == 42', 'int(ckpt["seed"]) == args.seed')
    source = replace_once(source, 'PERM_ROOT / f"perm_seed{s}/test.csv"', 'ROOT / _PLAN["permutation_paths"][str(s)]')
    source = replace_once(source, 'PERM_ROOT / f"perm_seed{seed}/test.csv"', 'ROOT / _PLAN["permutation_paths"][str(seed)]')
    source = replace_once(source, 'model = Model(', '''require(file_signature(checked_file(ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"]).stat()) == _CHECKPOINT_SIGNATURE,
        "Checkpoint stat changed while loading")
for _name in ("max_len", "hidden_size", "num_heads", "num_blocks", "dropout", "training_batch_size"):
    require(ckpt.get(_name) == _REFERENCE_CONFIG.get(_name), "Checkpoint/config architecture mismatch: " + _name)

model = Model(''')
    source = replace_once(source, 'ids = original_df["sample_id"].tolist()', '''ids = original_df["sample_id"].tolist()
_ORIGINAL_ROWS = indexed_rows(original_df.to_dict(orient="records"))
_VARIANTS = {}''')
    source = replace_once(source, '''saved = (
    saved.set_index("sample_id")''', '''require(set(saved["sample_id"]) == set(ids), "Saved baseline sample panel differs")
saved = (
    saved.set_index("sample_id")''')
    source = replace_once(source, 'print("ORIGINAL STRICT REPRODUCTION PASS")', '''_IDENTITY_TOP10_MISMATCHES = verify_original_top10(
    make_predictions(original_df, base_ranks, base_top10).to_dict(orient="records"), saved.to_dict(orient="records"))
require(abs(ndcg - _RUN["test_ndcg10"]) < 1e-10, "Baseline NDCG differs from the audited run")
print("ORIGINAL STRICT REPRODUCTION PASS: ranks, ordered Top10, raw IDs and sample identity")''')
    source = replace_once(source, '''    seq, gp, tok, target, raw_histories = construct(''', '''    _VARIANTS[str(seed)] = verify_permutation(_ORIGINAL_ROWS, perm_df.to_dict(orient="records"))
    seq, gp, tok, target, raw_histories = construct(''')
    source = replace_once(source, '''        "seed": seed,
        "samples": EXPECTED_N,''', '''        "permutation_seed": seed,
        "samples": EXPECTED_N,''')
    source = replace_once(source, '''OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)''', '''_LABELS = panel_labels(_ORIGINAL_ROWS, _VARIANTS)
control_labels_match(_LABELS, _CONTROL_LABEL_PATH)
_CHECKPOINT_AFTER = file_signature(checked_file(ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"]).stat())
require(_CHECKPOINT_AFTER == _CHECKPOINT_SIGNATURE, "Checkpoint stat changed during inference")
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
# Copy only after raw-event labels/hashes/affected flags exactly match Control.
(OUTPUT_DIR / "panel_labels.csv").write_bytes(_CONTROL_LABEL_PATH.read_bytes())''')
    metadata = '''manifest = {
    "dataset": "MovieLens-1M",
    "model": "RecentBucketSetR1SASRec",
    "method": "RecentBucketSet_R1_V1_mean_pooling",
    "model_seed": args.seed,
    "best_epoch": int(ckpt["epoch"]),
    "checkpoint_epoch": int(ckpt["epoch"]),
    "num_items": NUM_ITEMS,
    "test_samples": EXPECTED_N,
    "training_batch_size": 128,
    "evaluation_batch_size": BATCH_SIZE,
    "permutation_seeds": SEEDS,
    "strict_reproduction": "PASS",
    "canonicalization": "last_bucket_sorted_by_item_idx",
    "exact_raw_logits_invariance": True,
    "canonical_input_mismatches": 0,
    "original_rank_mismatches": rank_mismatches,
    "original_top10_mismatches": top10_mismatches,
    "original_identity_raw_top10_mismatches": _IDENTITY_TOP10_MISMATCHES,
    "input_manifest_sha256": _INPUTS_SHA256,
    "preparation_manifest_sha256": hashlib.sha256((_LBSP_STAGE / "manifests/lbsp_ml1m_stability_manifest.json").read_bytes()).hexdigest(),
    "generated_evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "model_definition_sha256": _MODEL_SHA256,
    "reference_files_sha256": _RUN["files_sha256"],
    "data_sha256": _DATA_SHA256,
    "source_sha256": _SOURCE_SHA256,
    "model_dir": str(MODEL_DIR),
    "checkpoint_path": str(CHECKPOINT),
    "checkpoint_stat_before_inference": list(_CHECKPOINT_SIGNATURE),
    "checkpoint_stat_after_inference": list(_CHECKPOINT_AFTER),
    "control_reference": _CONTROL_REFERENCE,
    "control_panel_labels_sha256": hashlib.sha256(_CONTROL_LABEL_PATH.read_bytes()).hexdigest(),
    "affected_definition": "at least two distinct raw-event sequences among five permutations, before LBSP canonicalization; original excluded",
    "affected_samples": sum(row["affected"] for row in _LABELS),
    "sequence_hash_definition": "SHA256(json.dumps(tuple(str(item) for item in sequence), separators=(',', ':'), ensure_ascii=False).encode('utf-8'))",
    "per_seed": seed_summaries,
}'''
    source = replace_assignment(source, 'manifest', metadata.split(' = ', 1)[1])
    source = replace_once(source, '(OUTPUT_DIR / "manifest.json").write_text(', '(OUTPUT_DIR / "metadata.json").write_text(')
    helpers = (plain, checked_file, file_signature, checked_sha256, row_sequence, last_bucket_size, indexed_rows,
               verify_permutation, sequence_sha256, panel_labels, verify_original_top10, control_labels_match)
    prefix = ('#!/usr/bin/env python3\n# ML1M LBSP fixed-model inference only; no training.\n'
        'import argparse\nfrom collections import Counter\nimport csv\nimport hashlib\nimport json\nimport os\nimport re\nimport stat\nfrom pathlib import Path, PurePosixPath\n\n'
        + '\n\n'.join(inspect.getsource(f) for f in helpers) + '\n\n'
        + 'PERMUTATIONS = ' + repr(PERMUTATIONS) + '\n'
        + "parser = argparse.ArgumentParser(description='ML1M LBSP fixed-model five-permutation evaluation')\n"
        + "parser.add_argument('--seed', type=int, choices=[42, 2026, 3407], required=True)\nargs = parser.parse_args()\n"
        + '_LBSP_ROOT = plain(Path(' + repr(str(root)) + '))\n_LBSP_STAGE = plain(Path(' + repr(str(stage)) + '))\n'
        + "if Path(__file__).resolve(strict=False).parent != _LBSP_STAGE / 'scripts':\n    raise RuntimeError('Use the original generated evaluator stage')\n"
        + '_PLAN = ' + repr(plan) + '\n_RUN = _PLAN["runs"][str(args.seed)]\n'
        + '_INPUTS_SHA256 = ' + repr(inputs_hash) + '\n_MODEL_SHA256 = ' + repr(model_hash) + '\n'
        + '_SOURCE_SHA256 = ' + repr(plan['_source_sha256']) + '\n_DATA_SHA256 = ' + repr(plan['_data_sha256']) + '\n'
        + '_CONTROL_REFERENCE = ' + repr(control_reference) + '\n'
        + 'for _relative, _expected in dict(_SOURCE_SHA256, **_DATA_SHA256).items():\n    checked_sha256(_LBSP_ROOT, _relative, _expected)\n'
        + 'for _name, _expected in _RUN["files_sha256"].items():\n    checked_sha256(_LBSP_ROOT, _RUN["model_dir"] + "/" + _name, _expected)\n'
        + "checked_sha256(_LBSP_STAGE, 'scripts/model_definition.py', _MODEL_SHA256)\n"
        + 'checked_sha256(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + "/" + _CONTROL_REFERENCE["manifest_path"], _CONTROL_REFERENCE["manifest_sha256"])\n'
        + 'for _relative, _expected in _CONTROL_REFERENCE["files_sha256"].items():\n    checked_sha256(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + "/" + _relative, _expected)\n'
        + '_CONTROL_LABEL_PATH = checked_file(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + f"/outputs/ml1m_seed{args.seed}/panel_labels.csv")\n'
        + '_CHECKPOINT_PATH = checked_file(_LBSP_ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"])\n'
        + '_CHECKPOINT_SIGNATURE = file_signature(_CHECKPOINT_PATH.stat())\n'
        + '_REFERENCE_CONFIG = json.loads(checked_file(_LBSP_ROOT, _RUN["model_dir"] + "/config.json").read_text(encoding="utf-8-sig"))\n'
        + "print('INTEGRITY PASS: three original sources, six input CSVs, five run files, audited Control reference and frozen model; weights not yet read')\n\n")
    result = prefix + source
    tree = ast.parse(result)
    after = {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}
    if protected != after:
        raise RuntimeError('Original construct/read_csv/predict_all/make_predictions/extract AST changed')
    return result


def prepare(root, inputs_path):
    root, inputs_path = plain(root), plain(inputs_path)
    if not root.is_dir() or not inputs_path.is_file():
        raise ValueError('Project root or input JSON is missing')
    raw = inputs_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUTS_SHA256:
        raise RuntimeError('Input plan differs from reviewed ML1M LBSP JSON')
    inputs = json.loads(raw.decode('utf-8-sig'))
    if (inputs.get('version') != 'lbsp_ml1m_stability_v1' or inputs.get('training_seeds') != list(SEEDS)
            or inputs.get('permutation_seeds') != list(PERMUTATIONS) or set(inputs.get('datasets', {})) != {'ml1m'}):
        raise ValueError('Unexpected LBSP plan schema')
    plan = inputs['datasets']['ml1m']
    if (plan['dataset'] != 'MovieLens-1M' or plan['expected_samples'] != 945 or plan['expected_items'] != 3416
            or set(plan['runs']) != {str(s) for s in SEEDS}
            or set(plan['permutation_paths']) != {str(s) for s in PERMUTATIONS}):
        raise ValueError('Unexpected dataset/seeds/permutations')
    sources_expected = {plan['eval_source'], inputs['model_source'], inputs['forward_source']}
    data_expected = {plan['original_path']} | set(plan['permutation_paths'].values())
    if set(inputs['sources_sha256']) != sources_expected or len(sources_expected) != 3 or set(inputs['data_sha256']) != data_expected or len(data_expected) != 6:
        raise ValueError('Expected three source files and six fixed input CSVs')
    control = inputs['control_reference']
    expected_control_files = {'outputs/ml1m_seed' + str(seed) + '/' + name for seed in SEEDS
        for name in ['predictions_original.csv', 'panel_labels.csv', 'metadata.json']
            + ['predictions_perm_seed' + str(p) + '.csv' for p in PERMUTATIONS]}
    if set(control['files_sha256']) != expected_control_files:
        raise ValueError('Expected24 audited Control reference outputs')
    verified, sources, checkpoint_presence = {}, {}, {}
    for relative, expected in dict(inputs['sources_sha256'], **inputs['data_sha256']).items():
        verified[relative] = checked_sha256(root, relative, expected)
    for relative, expected in inputs['sources_sha256'].items():
        blob = checked_file(root, relative).read_bytes()
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError('Source changed after hashing')
        sources[relative] = blob.decode('utf-8-sig')
        ast.parse(sources[relative])
    cp_rel = control['stage'] + '/' + control['manifest_path']
    verified[cp_rel] = checked_sha256(root, cp_rel, control['manifest_sha256'])
    control_manifest = json.loads(checked_file(root, cp_rel).read_text(encoding='utf-8-sig'))
    cplan = control_manifest['inputs']['datasets']['ml1m']
    if (control_manifest['root'] != str(root) or control_manifest['stage'] != str(root / control['stage'])
            or control_manifest['inputs_sha256'] != control['input_manifest_sha256']
            or cplan['original_path'] != plan['original_path'] or cplan['permutation_paths'] != plan['permutation_paths']
            or any(control_manifest['data_sha256'].get(p) != h for p, h in inputs['data_sha256'].items())):
        raise ValueError('Control/LBSP raw input identities differ')
    for relative, expected in control['files_sha256'].items():
        cp_rel = control['stage'] + '/' + relative
        verified[cp_rel] = checked_sha256(root, cp_rel, expected)
    for seed, run in plan['runs'].items():
        if (set(run['files_sha256']) != SAVED_FILES or run['checkpoint'] != 'r1_meanpool_best.pt'
                or not isinstance(run['best_epoch'], int) or run['best_epoch'] < 1
                or not math.isfinite(run['test_ndcg10']) or not 0 <= run['test_ndcg10'] <= 1):
            raise ValueError('Unexpected saved LBSP run descriptor')
        for name, expected in run['files_sha256'].items():
            relative = run['model_dir'] + '/' + name
            verified[relative] = checked_sha256(root, relative, expected)
        config = json.loads(checked_file(root, run['model_dir'] + '/config.json').read_text(encoding='utf-8-sig'))
        metrics = json.loads(checked_file(root, run['model_dir'] + '/metrics.json').read_text(encoding='utf-8-sig'))
        if (config.get('dataset') != plan['dataset'] or metrics.get('dataset') != plan['dataset']
                or config.get('seed') != int(seed) or metrics.get('seed') != int(seed)
                or config.get('model') != MODEL_NAME or metrics.get('model') != MODEL_NAME
                or metrics.get('best_epoch') != run['best_epoch'] or metrics['test']['samples'] != plan['expected_samples']
                or not math.isclose(metrics['test']['ndcg10'], run['test_ndcg10'], abs_tol=1e-12)):
            raise ValueError('Saved LBSP metadata differs from the audited plan')
        checkpoint = checked_file(root, run['model_dir'] + '/' + run['checkpoint'])
        info = checkpoint.stat()
        if info.st_size < 1:
            raise ValueError('Empty LBSP checkpoint')
        checkpoint_presence['ml1m_seed' + seed] = {'path': str(checkpoint), 'bytes': info.st_size,
                                                  'mtime_ns': info.st_mtime_ns, 'contents_checked': False}
    now = datetime.datetime.now(datetime.timezone.utc)
    stage = root / ('kais_lbsp_ml1m_stability_v1_' + now.strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:12])
    model = frozen_model(sources[inputs['model_source']], sources[inputs['forward_source']])
    model_hash = hashlib.sha256(model.encode('utf-8')).hexdigest()
    copy = dict(plan); copy['_source_sha256'] = inputs['sources_sha256']; copy['_data_sha256'] = inputs['data_sha256']
    evaluator = generated_evaluator(sources[plan['eval_source']], copy, control, root, stage, INPUTS_SHA256, model_hash)
    generated = {'scripts/model_definition.py': model, 'scripts/eval_ml1m_lbsp_stability.py': evaluator}
    hashes = {name: hashlib.sha256(text.encode('utf-8')).hexdigest() for name, text in generated.items()}
    manifest = {'version': 'lbsp_ml1m_stability_v1', 'status': 'PREPARED_ONLY_NO_INFERENCE', 'created_utc': now.isoformat(),
        'root': str(root), 'stage': str(stage), 'inputs': inputs, 'inputs_sha256': INPUTS_SHA256,
        'generated_sha256': hashes, 'sources_sha256': inputs['sources_sha256'], 'data_sha256': inputs['data_sha256'],
        'control_reference': control, 'checkpoint_presence': checkpoint_presence,
        'scope': 'ML1M LBSP fixed-model evaluation only; no training, hyperparameter changes or original file writes.',
        'preserved': 'Original construct/read_csv/predict_all/make_predictions/extract ASTs and four model/group/canonical/forward ASTs.',
        'labels': 'Raw-event sequence identity before canonicalization, five-permutation distinct>=2 affected definition; audited Control labels copied only after semantic equality.',
        'limits': ['Preparation never opens or hashes checkpoint contents.', 'No unobserved dataset/method results or significance/equivalence claims.']}
    stage.mkdir(exist_ok=False)
    for child in ('scripts', 'outputs', 'verification', 'manifests'):
        (stage / child).mkdir(exist_ok=False)
    for relative, text in generated.items():
        write_new(stage / relative, text)
    write_new(stage / MANIFEST_PATH, json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    write_new(stage / 'verification/PREPARED.json', json.dumps({'status': 'PREPARED_ONLY', 'hash_verified': verified,
        'inference_and_model_asts': 'UNCHANGED', 'checkpoint_contents_read': False}, indent=2) + '\n')
    print('WORKDIR=' + str(stage)); print('STATUS=PREPARED_ONLY_NO_INFERENCE')
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).with_name('lbsp_ml1m_stability_inputs_20261023.json'))
    args = parser.parse_args()
    prepare(args.root, args.inputs)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, SyntaxError) as exc:
        print('PREPARATION_FAILED=' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
