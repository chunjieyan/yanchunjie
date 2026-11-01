#!/usr/bin/env python3
"""Prepare nine fixed-model Amazon LBSP stability evaluations; stdlib only.
Preparation hashes inputs/metadata and stats checkpoint identities without
opening weights or executing research. Raw labels precede canonicalization.
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

INPUTS_SHA256 = 'b98f998d0c169d8c14478795bd119b73aa48f5cdaa1767449e378ce71307667b'
SEEDS = (42, 2026, 3407)
PERMUTATIONS = (101, 202, 303, 404, 505)
SAVED_FILES = {'config.json', 'metrics.json', 'training_log.csv', 'test_per_sample_predictions.csv', 'item_id_map.csv'}
MODEL_NAME = 'RecentBucketSetR1SASRec'
VERSION = 'lbsp_amazon_stability_v1'
MANIFEST_PATH = 'manifests/lbsp_amazon_stability_manifest.json'
DATASETS = {'beauty': 'Amazon_Beauty', 'sports': 'Amazon_Sports', 'toys': 'Amazon_Toys'}
FIXED_CONFIG = {'model': MODEL_NAME, 'method': 'RecentBucketSet_R1_V1_mean_pooling',
    'training_batch_size':128,'evaluation_batch_size':256,'optimizer':'Adam','learning_rate':0.001,
    'max_epochs':200,'patience':15,'max_len':50,'hidden_size':64,'num_heads':2,'num_blocks':2,'dropout':0.2,
    'sampling':'all_legal_next_item_positions_reshuffled_every_epoch','loss':'full_item_cross_entropy',
    'pooling':'last_history_bucket_only_mean','canonicalization':'sort_last_bucket_by_item_idx'}

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


def _legacy_evaluator(source, plan, control_reference, root, stage, inputs_hash, model_hash):
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


def validate_training_binding(inputs, training_manifest, root):
    reference = inputs['training_reference']
    if (training_manifest['version'] != 'lbsp_amazon_train_v1'
            or training_manifest['root'] != str(root) or training_manifest['stage'] != str(root / reference['stage'])
            or training_manifest['inputs_sha256'] != reference['input_manifest_sha256']):
        raise ValueError('Training stage/manifest binding differs')
    for source in (inputs['model_source'], inputs['forward_source']):
        if training_manifest['sources_sha256'].get(source) != inputs['sources_sha256'][source]:
            raise ValueError('Shared training model/forward source differs')
    for tag, plan in inputs['datasets'].items():
        original = training_manifest['inputs']['datasets'][tag]
        if (original['dataset'] != plan['dataset'] or original['expected_items'] != plan['expected_items']
                or original['expected_test'] != plan['expected_samples'] or original['test_path'] != plan['original_path']
                or training_manifest['data_sha256'].get(plan['original_path']) != inputs['data_sha256'][plan['original_path']]):
            raise ValueError('Original test input differs from audited training: ' + tag)
        for seed, run in plan['runs'].items():
            if run['model_dir'] != reference['stage'] + '/outputs/' + tag + '_lbsp_seed' + seed:
                raise ValueError('Checkpoint run path differs from audited training')
    return True


def validate_run_metadata(inputs, training_manifest, tag, seed, config, metrics, completion, root):
    plan = inputs['datasets'][tag]
    run = plan['runs'][str(seed)]
    original = training_manifest['inputs']['datasets'][tag]
    if (config.get('dataset') != plan['dataset'] or config.get('seed') != seed
            or config.get('num_items') != plan['expected_items']
            or config.get('test_path') != str(root / plan['original_path'])
            or config.get('legal_training_positions') != original['expected_positions']
            or any(config.get(k) != v for k, v in FIXED_CONFIG.items())):
        raise ValueError('Audited training config identity/hyperparameters differ')
    provenance = config['amazon_lbsp_training']
    reference = inputs['training_reference']
    if (provenance.get('input_manifest_sha256') != reference['input_manifest_sha256']
            or provenance.get('preparation_manifest_sha256') != reference['manifest_sha256']
            or provenance.get('generated_script_sha256') != training_manifest['generated_sha256']['scripts/train_' + tag + '_lbsp.py']
            or provenance.get('model_definition_sha256') != training_manifest['generated_sha256']['scripts/model_definition.py']
            or provenance.get('sources_sha256') != training_manifest['sources_sha256']
            or provenance.get('data_sha256', {}).get(plan['original_path']) != inputs['data_sha256'][plan['original_path']]
            or any(provenance.get(k) is not True for k in ('scientific_definitions_unchanged', 'formal_rng_reset_preserved', 'smoke_execution_omitted'))):
        raise ValueError('Audited training config provenance differs')
    if (metrics.get('dataset') != plan['dataset'] or metrics.get('seed') != seed
            or metrics.get('model') != MODEL_NAME or metrics.get('method') != FIXED_CONFIG['method']
            or metrics.get('best_epoch') != run['best_epoch'] or metrics['test']['samples'] != plan['expected_samples']
            or not math.isclose(metrics['test']['ndcg10'], run['test_ndcg10'], rel_tol=0, abs_tol=1e-12)):
        raise ValueError('Saved metrics differ from the audited checkpoint run')
    if (completion.get('status') != 'COMPLETED' or completion.get('tag') != tag or completion.get('dataset') != plan['dataset']
            or completion.get('seed') != seed or completion.get('best_epoch') != run['best_epoch']
            or completion.get('output_dir') != str(root / run['model_dir'])
            or completion.get('checkpoint') != run['checkpoint'] or completion.get('output_metadata_sha256') != run['files_sha256']
            or completion.get('all_original_hashes_verified') is not True or completion.get('originals_unchanged') is not True
            or any(completion.get(k) != v for k, v in provenance.items())):
        raise ValueError('Completed training record differs from saved metadata')
    return True


def generated_evaluator(source, tag, inputs, root, stage, inputs_hash, model_hash):
    plan = dict(inputs['datasets'][tag])
    plan['_source_sha256'] = inputs['sources_sha256']
    plan['_data_sha256'] = {p: inputs['data_sha256'][p] for p in [plan['original_path']] + list(plan['permutation_paths'].values())}
    result = _legacy_evaluator(source, plan, inputs['control_reference'], root, stage, inputs_hash, model_hash)
    result = replace_assignment(result, 'EXPECTED_N', repr(plan['expected_samples']))
    result = replace_assignment(result, 'EXPECTED_ITEMS', repr(plan['expected_items']))
    result = replace_assignment(result, 'OUTPUT_DIR', '_LBSP_STAGE / "outputs" / f"' + tag + '_seed{args.seed}"')
    result = replace_once(result, 'ckpt["dataset"] == "MovieLens-1M"', 'ckpt["dataset"] == _PLAN["dataset"]')
    result = replace_once(result, 'f"/outputs/ml1m_seed{args.seed}/panel_labels.csv"', 'f"/outputs/' + tag + '_seed{args.seed}/panel_labels.csv"')
    result = result.replace('manifests/lbsp_ml1m_stability_manifest.json', MANIFEST_PATH)
    # The one accepted scientific loader adaptation normalizes aliases only in
    # memory. All original construction, inference and prediction ASTs remain.
    loader = next(n for n in ast.parse(result).body if isinstance(n, ast.FunctionDef) and n.name == 'read_csv')
    first = ast.get_source_segment(result, loader.body[0])
    result = replace_once(result, first, first + '\n    df = normalize_eval_frame(df)')
    helpers = '\n\n'.join(inspect.getsource(f) for f in
        (normalize_eval_records, normalize_eval_frame, validate_training_binding, validate_run_metadata))
    result = replace_once(result, 'PERMUTATIONS = ' + repr(PERMUTATIONS), helpers + '\n\n'
        + 'MODEL_NAME = ' + repr(MODEL_NAME) + '\nFIXED_CONFIG = ' + repr(FIXED_CONFIG) + '\n'
        + 'PERMUTATIONS = ' + repr(PERMUTATIONS))
    runtime = ('\nimport math\n_TAG = ' + repr(tag) + '\n_INPUTS_FULL = ' + repr(inputs)
        + '\n_TRAINING_REFERENCE = _INPUTS_FULL["training_reference"]\n'
        + '_TRAINING_MANIFEST = json.loads(checked_file(_LBSP_ROOT, _TRAINING_REFERENCE["stage"] + "/" + _TRAINING_REFERENCE["manifest_path"]).read_text(encoding="utf-8-sig"))\n'
        + '_PREPARATION_BYTES = checked_file(_LBSP_STAGE, ' + repr(MANIFEST_PATH) + ').read_bytes()\n'
        + '_PREPARATION_SHA256 = hashlib.sha256(_PREPARATION_BYTES).hexdigest()\n'
        + '_PREPARATION = json.loads(_PREPARATION_BYTES.decode("utf-8-sig"))\n'
        + 'if (_PREPARATION["inputs"] != _INPUTS_FULL or _PREPARATION["inputs_sha256"] != _INPUTS_SHA256\n'
        + '        or _PREPARATION["version"] != "lbsp_amazon_stability_v1" or _PREPARATION["root"] != str(_LBSP_ROOT)\n'
        + '        or _PREPARATION["stage"] != str(_LBSP_STAGE)):\n    raise RuntimeError("Preparation identity differs")\n'
        + '_GENERATED_RELATIVE = "scripts/evaluate_" + _TAG + "_lbsp.py"\n'
        + '_GENERATED_SHA256 = _PREPARATION["generated_sha256"][_GENERATED_RELATIVE]\n'
        + '_TRAINING_MODEL_SHA256 = _TRAINING_MANIFEST["generated_sha256"]["scripts/model_definition.py"]\n'
        + 'def _verify_closure():\n'
        + '    for relative, expected in dict(_SOURCE_SHA256, **_DATA_SHA256).items():\n        checked_sha256(_LBSP_ROOT, relative, expected)\n'
        + '    for name, expected in _RUN["files_sha256"].items():\n        checked_sha256(_LBSP_ROOT, _RUN["model_dir"] + "/" + name, expected)\n'
        + '    checked_sha256(_LBSP_STAGE, "scripts/model_definition.py", _MODEL_SHA256)\n'
        + '    checked_sha256(_LBSP_STAGE, _GENERATED_RELATIVE, _GENERATED_SHA256)\n'
        + '    checked_sha256(_LBSP_STAGE, ' + repr(MANIFEST_PATH) + ', _PREPARATION_SHA256)\n'
        + '    for relative, expected in _CONTROL_REFERENCE["files_sha256"].items():\n        checked_sha256(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + "/" + relative, expected)\n'
        + '    checked_sha256(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + "/" + _CONTROL_REFERENCE["manifest_path"], _CONTROL_REFERENCE["manifest_sha256"])\n'
        + '    for relative, expected in _TRAINING_REFERENCE["files_sha256"].items():\n        checked_sha256(_LBSP_ROOT, _TRAINING_REFERENCE["stage"] + "/" + relative, expected)\n'
        + '_verify_closure()\nvalidate_training_binding(_INPUTS_FULL, _TRAINING_MANIFEST, _LBSP_ROOT)\n'
        + '_COMPLETED_RUN = json.loads(checked_file(_LBSP_ROOT, _TRAINING_REFERENCE["stage"] + f"/verification/completed_{_TAG}_seed{args.seed}.json").read_text(encoding="utf-8-sig"))\n'
        + '_REFERENCE_METRICS = json.loads(checked_file(_LBSP_ROOT, _RUN["model_dir"] + "/metrics.json").read_text(encoding="utf-8-sig"))\n'
        + 'validate_run_metadata(_INPUTS_FULL, _TRAINING_MANIFEST, _TAG, args.seed, _REFERENCE_CONFIG, _REFERENCE_METRICS, _COMPLETED_RUN, _LBSP_ROOT)\n'
        + 'print("AMAZON LBSP INTEGRITY PASS: sources, six inputs, five saved files, full training and Control references verified before Torch")\n')
    result = replace_once(result, '\nimport ast\n', runtime + '\nimport ast\n')
    result = replace_once(result, 'model = Model(', '''require(ckpt.get("method") == _REFERENCE_CONFIG["method"], "Checkpoint method differs")
model = Model(''')
    result = replace_once(result, '_LABELS = panel_labels(_ORIGINAL_ROWS, _VARIANTS)', '_verify_closure()\n_LABELS = panel_labels(_ORIGINAL_ROWS, _VARIANTS)')
    manifest = next(n for n in ast.parse(result).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'manifest' for t in n.targets))
    if not isinstance(manifest.value, ast.Dict):
        raise RuntimeError('Expected a metadata dictionary')
    metadata = ast.get_source_segment(result, manifest.value)
    metadata = replace_once(metadata, '"dataset": "MovieLens-1M"', '"dataset": _PLAN["dataset"]')
    metadata = metadata[:-1] + '''    "training_reference": _TRAINING_REFERENCE,
    "training_model_definition_sha256": _TRAINING_MODEL_SHA256,
    "runtime_hashes_verified_before_after": True,
}'''
    result = replace_assignment(result, 'manifest', metadata)
    # Check the original four science definitions exactly, and the read_csv
    # definition after removing precisely the approved alias statement.
    for name in ('construct', 'predict_all', 'make_predictions', 'extract', 'read_csv'):
        original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        emitted = next(n for n in ast.parse(result).body if isinstance(n, ast.FunctionDef) and n.name == name)
        if name == 'read_csv':
            inserts = [n for n in emitted.body if ast.unparse(n) == 'df = normalize_eval_frame(df)']
            if len(inserts) != 1 or emitted.body[1] is not inserts[0]:
                raise RuntimeError('CSV adapter is outside the approved position')
            emitted.body.remove(inserts[0])
        if ast.dump(original) != ast.dump(emitted):
            raise RuntimeError('Original scientific evaluator AST changed: ' + name)
    ast.parse(result, feature_version=(3, 10))
    return result


def prepare(root, inputs_path):
    root, inputs_path = plain(root), plain(inputs_path)
    raw = inputs_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUTS_SHA256:
        raise RuntimeError('Input plan differs from reviewed Amazon stability seal')
    inputs = json.loads(raw.decode('utf-8-sig'))
    if (inputs['version'] != VERSION or inputs['training_seeds'] != list(SEEDS)
            or inputs['permutation_seeds'] != list(PERMUTATIONS) or set(inputs['datasets']) != set(DATASETS)):
        raise ValueError('Unexpected Amazon stability schema')
    expected_data, expected_sources = set(), {inputs['model_source'], inputs['forward_source']}
    for tag, plan in inputs['datasets'].items():
        if (plan['dataset'] != DATASETS[tag] or set(plan['runs']) != {str(s) for s in SEEDS}
                or set(plan['permutation_paths']) != {str(s) for s in PERMUTATIONS}):
            raise ValueError('Unexpected dataset run/permutation plan')
        expected_data.update([plan['original_path']] + list(plan['permutation_paths'].values()))
        expected_sources.add(plan['eval_source'])
    if (set(inputs['sources_sha256']) != expected_sources or len(expected_sources) != 3
            or set(inputs['data_sha256']) != expected_data or len(expected_data) != 18):
        raise ValueError('Expected three sources and eighteen inputs')
    verified, sources, checkpoint_presence = {}, {}, {}
    for relative, digest in {**inputs['sources_sha256'], **inputs['data_sha256']}.items():
        verified[relative] = checked_sha256(root, relative, digest)
    for relative, digest in inputs['sources_sha256'].items():
        blob = checked_file(root, relative).read_bytes()
        if hashlib.sha256(blob).hexdigest() != digest:
            raise RuntimeError('Source changed after hashing')
        sources[relative] = blob.decode('utf-8-sig')
        ast.parse(sources[relative])
    control, training = inputs['control_reference'], inputs['training_reference']
    expected_control = {'outputs/' + tag + '_seed' + str(seed) + '/' + filename for tag in DATASETS for seed in SEEDS
        for filename in ['metadata.json', 'panel_labels.csv', 'predictions_original.csv']
            + ['predictions_perm_seed' + str(p) + '.csv' for p in PERMUTATIONS]}
    if set(control['files_sha256']) != expected_control or len(training['files_sha256']) != 25:
        raise ValueError('Unexpected Control/training reference inventory')
    for reference in (control, training):
        relative = reference['stage'] + '/' + reference['manifest_path']
        verified[relative] = checked_sha256(root, relative, reference['manifest_sha256'])
        for name, digest in reference['files_sha256'].items():
            relative = reference['stage'] + '/' + name
            verified[relative] = checked_sha256(root, relative, digest)
    control_manifest = json.loads(checked_file(root, control['stage'] + '/' + control['manifest_path']).read_text(encoding='utf-8-sig'))
    if (control_manifest['root'] != str(root) or control_manifest['stage'] != str(root / control['stage'])
            or control_manifest['inputs_sha256'] != control['input_manifest_sha256']):
        raise ValueError('Control reference manifest identity differs')
    training_manifest = json.loads(checked_file(root, training['stage'] + '/' + training['manifest_path']).read_text(encoding='utf-8-sig'))
    validate_training_binding(inputs, training_manifest, root)
    for tag, plan in inputs['datasets'].items():
        cplan = control_manifest['inputs']['datasets'][tag]
        relevant_data = [plan['original_path']] + list(plan['permutation_paths'].values())
        if (cplan['original_path'] != plan['original_path'] or cplan['permutation_paths'] != plan['permutation_paths']
                or any(control_manifest['data_sha256'].get(p) != inputs['data_sha256'][p] for p in relevant_data)):
            raise ValueError('Control/LBSP raw test panel identities differ')
        for seed, run in plan['runs'].items():
            if (set(run['files_sha256']) != SAVED_FILES or run['checkpoint'] != 'r1_meanpool_best.pt'
                    or not isinstance(run['best_epoch'], int) or not 1 <= run['best_epoch'] <= 200
                    or not math.isfinite(run['test_ndcg10']) or not 0 <= run['test_ndcg10'] <= 1):
                raise ValueError('Unexpected saved checkpoint descriptor')
            for name, digest in run['files_sha256'].items():
                relative = run['model_dir'] + '/' + name
                verified[relative] = checked_sha256(root, relative, digest)
            config = json.loads(checked_file(root, run['model_dir'] + '/config.json').read_text(encoding='utf-8-sig'))
            metrics = json.loads(checked_file(root, run['model_dir'] + '/metrics.json').read_text(encoding='utf-8-sig'))
            completion = json.loads(checked_file(root, training['stage'] + '/verification/completed_' + tag + '_seed' + seed + '.json').read_text(encoding='utf-8-sig'))
            validate_run_metadata(inputs, training_manifest, tag, int(seed), config, metrics, completion, root)
            checkpoint = checked_file(root, run['model_dir'] + '/' + run['checkpoint'])
            info = checkpoint.stat()
            if info.st_size < 1:
                raise ValueError('Checkpoint is empty')
            checkpoint_presence[tag + '_seed' + seed] = {'path': str(checkpoint), 'bytes': info.st_size,
                'mtime_ns': info.st_mtime_ns, 'contents_checked': False}
    now = datetime.datetime.now(datetime.timezone.utc)
    stage = root / ('kais_lbsp_amazon_stability_v1_' + now.strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:12])
    model = frozen_model(sources[inputs['model_source']], sources[inputs['forward_source']])
    model_hash = hashlib.sha256(model.encode()).hexdigest()
    generated = {'scripts/model_definition.py': model}
    for tag, plan in inputs['datasets'].items():
        generated['scripts/evaluate_' + tag + '_lbsp.py'] = generated_evaluator(sources[plan['eval_source']], tag,
            inputs, root, stage, INPUTS_SHA256, model_hash)
    for text in generated.values():
        ast.parse(text, feature_version=(3, 10))
    manifest = {'version': VERSION, 'status': 'PREPARED_ONLY_NO_INFERENCE', 'created_utc': now.isoformat(),
        'root': str(root), 'stage': str(stage), 'inputs': inputs, 'inputs_sha256': INPUTS_SHA256,
        'generated_sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in generated.items()},
        'sources_sha256': inputs['sources_sha256'], 'data_sha256': inputs['data_sha256'],
        'control_reference': control, 'training_reference': training, 'checkpoint_presence': checkpoint_presence,
        'scope': inputs['scope'], 'preserved': 'Four frozen model/scientific ASTs; construct/predict_all/make_predictions/extract and read_csv except the checked in-memory size/pooling alias.',
        'labels': 'Five-permutation raw-event distinct>=2 before canonicalization; original excluded. Control labels copied only after semantic equality.',
        'limits': ['Preparation never reads checkpoint contents or executes research.', 'Recorded tensor/logit results require the later explicit evaluator runs.']}
    stage.mkdir(exist_ok=False)
    for name in ('scripts', 'outputs', 'verification', 'manifests'):
        (stage / name).mkdir(exist_ok=False)
    for relative, text in generated.items():
        write_new(stage / relative, text)
    write_new(stage / MANIFEST_PATH, json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    write_new(stage / 'verification/PREPARED.json', json.dumps({'status': 'PREPARED_ONLY', 'hash_verified': verified,
        'research_code_executed': False, 'checkpoint_contents_read': False}, indent=2) + '\n')
    print('WORKDIR=' + str(stage)); print('STATUS=PREPARED_ONLY_NO_INFERENCE')
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).with_name('lbsp_amazon_stability_inputs_20261025.json'))
    args = parser.parse_args()
    prepare(args.root, args.inputs)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, SyntaxError) as exc:
        print('PREPARATION_FAILED=' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)


