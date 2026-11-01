#!/usr/bin/env python3
# ML1M LBSP fixed-model inference only; no training.
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath

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


PERMUTATIONS = (101, 202, 303, 404, 505)
parser = argparse.ArgumentParser(description='ML1M LBSP fixed-model five-permutation evaluation')
parser.add_argument('--seed', type=int, choices=[42, 2026, 3407], required=True)
args = parser.parse_args()
_LBSP_ROOT = plain(Path('/root/autodl-tmp/time_uncertainty_rec'))
_LBSP_STAGE = plain(Path('/root/autodl-tmp/time_uncertainty_rec/kais_lbsp_ml1m_stability_v1_20261023T035832Z_ecc8773184d1'))
if Path(__file__).resolve(strict=False).parent != _LBSP_STAGE / 'scripts':
    raise RuntimeError('Use the original generated evaluator stage')
_PLAN = {'dataset': 'MovieLens-1M', 'expected_samples': 945, 'expected_items': 3416, 'eval_source': 'step38_kais_retrain/scripts/38_22_eval_ml1m_r1_stability.py', 'original_path': 'processed/recent_bucketset_r1/MovieLens-1M/test/r1_samples.csv', 'permutation_paths': {'101': 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed101/test.csv', '202': 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed202/test.csv', '303': 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed303/test.csv', '404': 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed404/test.csv', '505': 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed505/test.csv'}, 'runs': {'42': {'model_dir': 'step38_kais_retrain/outputs/ml1m_r1_meanpool_ce_bs128_lr1em3_max200_seed42', 'files_sha256': {'config.json': 'f6113519090c766bb656a6346ea02317aa40c169863fd9abe6135eddbd2ce1fa', 'metrics.json': '1bf4cfe31603ccd7007e24db1aa79d0cb93f421e675a17c81716e746456de0ce', 'training_log.csv': 'bbafc1299163bc777a35aec57f7984ebced51efaf9535132ecb2b192bd358364', 'test_per_sample_predictions.csv': '766c063096bd782fbb6bfc3c916c05e231419c2c1bdcab73d5c3b3a8fbb48ced', 'item_id_map.csv': '5eced6a0a70be8bb2fd44649ab0cf7a27db4f8b8092e15b19c56765dae685f53'}, 'best_epoch': 33, 'test_ndcg10': 0.05202619536074009, 'checkpoint': 'r1_meanpool_best.pt'}, '2026': {'model_dir': 'step38_kais_retrain/outputs/ml1m_r1_meanpool_ce_bs128_lr1em3_max200_seed2026', 'files_sha256': {'config.json': '0dc37e5425c97ae409c3f618d6c262aa46031172f4ebe9cb9d198f10341680b5', 'metrics.json': '8bbd49cc3164bc0046fec02020264613fa89106ee8f17664e107a005dd1cd63b', 'training_log.csv': 'ab9314c4fc7cd124b69a5906e8a0638ff2bcdd259fbe56390107c3f22a0f1de8', 'test_per_sample_predictions.csv': 'b7de338b4bb0ffe42c619d0bf399750a6acd08659d4b88d26cc559119f94df66', 'item_id_map.csv': '5eced6a0a70be8bb2fd44649ab0cf7a27db4f8b8092e15b19c56765dae685f53'}, 'best_epoch': 29, 'test_ndcg10': 0.05294391851062197, 'checkpoint': 'r1_meanpool_best.pt'}, '3407': {'model_dir': 'step38_kais_retrain/outputs/ml1m_r1_meanpool_ce_bs128_lr1em3_max200_seed3407', 'files_sha256': {'config.json': '66b6dac3a8d9d0d1aa2e58d065220343375f1c9dd65f18e3d8adcecc8173093a', 'metrics.json': 'beb05f9a51ae89311f0f741b7747f54555fc1eb8ba7265f3afde35802e6f1b89', 'training_log.csv': '1379775a15ad0404c9efe1316d011e873cd8d7b5cdc945b01c3c370b2829e7d6', 'test_per_sample_predictions.csv': 'e3f04d102fcdb50ba888c4b863f14db35d9bad60026e9bdad9f38f6e71c6e08f', 'item_id_map.csv': '5eced6a0a70be8bb2fd44649ab0cf7a27db4f8b8092e15b19c56765dae685f53'}, 'best_epoch': 30, 'test_ndcg10': 0.05086777398458645, 'checkpoint': 'r1_meanpool_best.pt'}}, '_source_sha256': {'step38_kais_retrain/scripts/38_22_eval_ml1m_r1_stability.py': '6695ccae462f786fdda4d14100d676d184fa613487fc62b0c5bd1ebe5e815080', 'step38_kais_retrain/scripts/38_20_train_ml1m_r1_bs128.py': 'c718592e65a3d66f2ec1ce6d22c8b9abc3b8d4a393dde49ae014247414741b38', 'audit/33_2_train_movielens_r1_seed42.py': 'e8ae647f3fa3cc2ef9f3087a0ae44cfbd4da2706b2ce33fc558fff0678a04ffb'}, '_data_sha256': {'processed/recent_bucketset_r1/MovieLens-1M/test/r1_samples.csv': '99bec555642aca4cd00cdfe3b5d95c7b3ac577c1c1e0457a7c6212c08b9ea303', 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed101/test.csv': '32dfbfaedc7e79418d6283ba4fb020e7d36f56e92368bda27bb44761f0fcb84d', 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed202/test.csv': '23fb6ef556787dc7c3174c719cac2640de015ef905141fa7d509fd11b1387be1', 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed303/test.csv': 'a3b7a6b1b8463e771843bab97d4aeb7fac3d401c0a190e0e2da35e4a2803e03e', 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed404/test.csv': 'ff39558c3805cce5990961588614a1c76c89626f0765dc26ee263a995247b29c', 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed505/test.csv': 'f863476ab258233b33081a9f476b043847a992349218db692c2b9a938f19dfb7'}}
_RUN = _PLAN["runs"][str(args.seed)]
_INPUTS_SHA256 = 'a34890d187617f445fb20d6aba70ddf5bd1d81d01c9ffb6099b6e46486675541'
_MODEL_SHA256 = 'b2bb38a5a080589d8d38dfe4485149f2c4c0b3da06922f141cfa483b488165bc'
_SOURCE_SHA256 = {'step38_kais_retrain/scripts/38_22_eval_ml1m_r1_stability.py': '6695ccae462f786fdda4d14100d676d184fa613487fc62b0c5bd1ebe5e815080', 'step38_kais_retrain/scripts/38_20_train_ml1m_r1_bs128.py': 'c718592e65a3d66f2ec1ce6d22c8b9abc3b8d4a393dde49ae014247414741b38', 'audit/33_2_train_movielens_r1_seed42.py': 'e8ae647f3fa3cc2ef9f3087a0ae44cfbd4da2706b2ce33fc558fff0678a04ffb'}
_DATA_SHA256 = {'processed/recent_bucketset_r1/MovieLens-1M/test/r1_samples.csv': '99bec555642aca4cd00cdfe3b5d95c7b3ac577c1c1e0457a7c6212c08b9ea303', 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed101/test.csv': '32dfbfaedc7e79418d6283ba4fb020e7d36f56e92368bda27bb44761f0fcb84d', 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed202/test.csv': '23fb6ef556787dc7c3174c719cac2640de015ef905141fa7d509fd11b1387be1', 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed303/test.csv': 'a3b7a6b1b8463e771843bab97d4aeb7fac3d401c0a190e0e2da35e4a2803e03e', 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed404/test.csv': 'ff39558c3805cce5990961588614a1c76c89626f0765dc26ee263a995247b29c', 'processed/strict_last_bucket_only/MovieLens-1M/perm_seed505/test.csv': 'f863476ab258233b33081a9f476b043847a992349218db692c2b9a938f19dfb7'}
_CONTROL_REFERENCE = {'stage': 'kais_control_stability_v1_20261023T032552Z_c9383ff8a16c', 'manifest_path': 'manifests/control_stability_manifest.json', 'manifest_sha256': '7bab86b2638dc73368da5103bf3fcf90e588a6e63d1e6b66749565350efdfd4c', 'input_manifest_sha256': '8e8d13ded1ded7729e7f951b97d65ef3bda84614f4f4a9ad881bec68f7c81458', 'files_sha256': {'outputs/ml1m_seed42/predictions_original.csv': 'e3dcf095cba1b8ed6cb47804089335d1acdbbf0f6710a0585ce4ace42c095822', 'outputs/ml1m_seed42/predictions_perm_seed101.csv': '05f2fbd29919b0c9a76610397bf59ca8f637c0e339fdb81a522ee3e1d9a91bd0', 'outputs/ml1m_seed42/predictions_perm_seed202.csv': '069cb960f3ed6c5022677451d3b4645946f37c3381d25020e9ad69986a41711a', 'outputs/ml1m_seed42/predictions_perm_seed303.csv': '66bf46179a6942c2f91f4ed876125bd84bc77f44bce9f02cf9015d03b5cd0e88', 'outputs/ml1m_seed42/predictions_perm_seed404.csv': 'c5b506b47666403163119833264665af30e80f2f92009ada1f0751228b44406c', 'outputs/ml1m_seed42/predictions_perm_seed505.csv': 'f40e12ab3b238f9e4ff54f836364b947801590ad362bf2010bab3a5d23a2f4a3', 'outputs/ml1m_seed42/panel_labels.csv': 'ebaa7d9ff9039b0446bf5304367f64d85377dff40e76e83f127ca93377c7cac2', 'outputs/ml1m_seed42/metadata.json': 'af560cf0000da2f6f40bc0ee06bc8d0b62abcb8b17436bc150440c68a61025bb', 'outputs/ml1m_seed2026/predictions_original.csv': 'ffffcba167c1ffe3f5ab19b9d594504d7d1bc400ef946064acc53e8e8928eff0', 'outputs/ml1m_seed2026/predictions_perm_seed101.csv': '4d6bec69c04752b32c7bcbdc86c602cf13ac04a12f25e92b60599ca8de0f115e', 'outputs/ml1m_seed2026/predictions_perm_seed202.csv': '7e56d276335a96d24a408ad0b41f9c2a6ad59f11d7c59b1a812ea1e85a8b1d0f', 'outputs/ml1m_seed2026/predictions_perm_seed303.csv': '710c1db6516785bb970f8e96140cc723e568238075a403bb40971255a003ef5a', 'outputs/ml1m_seed2026/predictions_perm_seed404.csv': '82161cb67dcea5a5a9849c5fa1a84afd96547405bd2146591f22bfa57422acdb', 'outputs/ml1m_seed2026/predictions_perm_seed505.csv': '8e0558975042ab924d75d78079a8ddcd9994e88821ec407acaacc6ed92cce6e8', 'outputs/ml1m_seed2026/panel_labels.csv': 'ebaa7d9ff9039b0446bf5304367f64d85377dff40e76e83f127ca93377c7cac2', 'outputs/ml1m_seed2026/metadata.json': '44f8382a77c4362fa2259e971d1d76fd96820f0cba7980abfa3accb14df80c88', 'outputs/ml1m_seed3407/predictions_original.csv': 'aa28ca63138a85220b3d274a2921aaf899e9e1f4aad324c335df6e1116fb9b31', 'outputs/ml1m_seed3407/predictions_perm_seed101.csv': '740e8120e140e164f8fe005b6c8fdd01ee45754ce83d949fcffa7658a56e7cef', 'outputs/ml1m_seed3407/predictions_perm_seed202.csv': 'c1f630f98e362b7aec8f9a81fc531b2dc2542a1faa852257f4e057c6d31330bb', 'outputs/ml1m_seed3407/predictions_perm_seed303.csv': '40212d0a42e9bf1febf4e417fea8b18c980e8894bded31246f558e3fd5fe5d80', 'outputs/ml1m_seed3407/predictions_perm_seed404.csv': 'a4ad948454ffce337f14c5cfe5e3b1d20e35d4c3845779fa357f5af1fd4314a6', 'outputs/ml1m_seed3407/predictions_perm_seed505.csv': 'e02df8cbef3114d33647a269b42ab73bcd087744f3a585ad355316b3c3a875b0', 'outputs/ml1m_seed3407/panel_labels.csv': 'ebaa7d9ff9039b0446bf5304367f64d85377dff40e76e83f127ca93377c7cac2', 'outputs/ml1m_seed3407/metadata.json': '70bbb74216f54ebeb23c8e9ebed507554aab073b9e13a8212adf6545c1245554'}}
for _relative, _expected in dict(_SOURCE_SHA256, **_DATA_SHA256).items():
    checked_sha256(_LBSP_ROOT, _relative, _expected)
for _name, _expected in _RUN["files_sha256"].items():
    checked_sha256(_LBSP_ROOT, _RUN["model_dir"] + "/" + _name, _expected)
checked_sha256(_LBSP_STAGE, 'scripts/model_definition.py', _MODEL_SHA256)
checked_sha256(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + "/" + _CONTROL_REFERENCE["manifest_path"], _CONTROL_REFERENCE["manifest_sha256"])
for _relative, _expected in _CONTROL_REFERENCE["files_sha256"].items():
    checked_sha256(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + "/" + _relative, _expected)
_CONTROL_LABEL_PATH = checked_file(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + f"/outputs/ml1m_seed{args.seed}/panel_labels.csv")
_CHECKPOINT_PATH = checked_file(_LBSP_ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"])
_CHECKPOINT_SIGNATURE = file_signature(_CHECKPOINT_PATH.stat())
_REFERENCE_CONFIG = json.loads(checked_file(_LBSP_ROOT, _RUN["model_dir"] + "/config.json").read_text(encoding="utf-8-sig"))
print('INTEGRITY PASS: three original sources, six input CSVs, five run files, audited Control reference and frozen model; weights not yet read')


import ast
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Step38.22 ML1M R1 Fixed-Checkpoint Stability
# ============================================================

ROOT = _LBSP_ROOT
BASE = _LBSP_STAGE

SEEDS = [101, 202, 303, 404, 505]
MAX_LEN = 50
BATCH_SIZE = 256
EXPECTED_N = 945
EXPECTED_ITEMS = 3416

MODEL_DIR = ROOT / _RUN["model_dir"]

CHECKPOINT = _CHECKPOINT_PATH
MAP_PATH = MODEL_DIR / "item_id_map.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"
SAVED_PRED = MODEL_DIR / "test_per_sample_predictions.csv"

ORIGINAL_PATH = ROOT / _PLAN["original_path"]

PERM_ROOT = (
    ROOT / "processed/strict_last_bucket_only/"
    "MovieLens-1M"
)

R1_SOURCE = _LBSP_STAGE / "scripts/model_definition.py"

FORWARD_SOURCE = _LBSP_STAGE / "scripts/model_definition.py"

OUTPUT_DIR = _LBSP_STAGE / "outputs" / f"ml1m_seed{args.seed}"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


print("=" * 72)
print("STEP38.22 ML1M R1 FIXED MODEL STABILITY")
print("=" * 72)
print("Device:", device)
print("Permutation seeds:", SEEDS)

required = [
    CHECKPOINT,
    MAP_PATH,
    METRICS_PATH,
    SAVED_PRED,
    ORIGINAL_PATH,
    R1_SOURCE,
    FORWARD_SOURCE,
]

required += [
    ROOT / _PLAN["permutation_paths"][str(s)]
    for s in SEEDS
]

for path in required:
    require(path.is_file(), f"缺少文件: {path}")

require(
    not OUTPUT_DIR.exists(),
    f"输出目录已存在，禁止覆盖: {OUTPUT_DIR}",
)

print("INPUT FILE AUDIT PASS")


# ============================================================
# 1. 从已验证脚本提取 R1 模型、分组函数和安全前向
# 不执行训练脚本
# ============================================================

r1_ast = ast.parse(
    R1_SOURCE.read_text(encoding="utf-8")
)

forward_ast = ast.parse(
    FORWARD_SOURCE.read_text(encoding="utf-8")
)


def extract(tree, name, cls=False):
    node_type = ast.ClassDef if cls else ast.FunctionDef

    matches = [
        node for node in tree.body
        if isinstance(node, node_type)
        and node.name == name
    ]

    require(
        len(matches) == 1,
        f"代码定义异常: {name}",
    )

    return matches[0]


nodes = [
    extract(
        r1_ast,
        "build_r1_group_positions_from_size",
    ),
    extract(
        r1_ast,
        "canonicalize_last_bucket_items",
    ),
    extract(
        r1_ast,
        "RecentBucketSetR1SASRec",
        cls=True,
    ),
    extract(
        forward_ast,
        "predict_batch_masked",
    ),
]

namespace = {
    "__name__": "step38_r1_stability",
    "torch": torch,
    "nn": nn,
    "F": F,
    "np": np,
    "math": math,
    "MAX_LEN": MAX_LEN,
    "device": device,
    "require": require,
}

exec(
    compile(
        ast.Module(
            body=nodes,
            type_ignores=[],
        ),
        str(FORWARD_SOURCE),
        "exec",
    ),
    namespace,
)

Model = namespace["RecentBucketSetR1SASRec"]

Model.predict_batch_masked = namespace[
    "predict_batch_masked"
]

build_groups = namespace[
    "build_r1_group_positions_from_size"
]

canonicalize = namespace[
    "canonicalize_last_bucket_items"
]

print("R1 model and safe forward loaded: PASS")


# ============================================================
# 2. 加载固定物品映射
# ============================================================

mapping = pd.read_csv(
    MAP_PATH,
    dtype={"raw_item_id": str},
)

require(
    mapping["raw_item_id"].is_unique,
    "原始物品ID重复",
)

require(
    mapping["item_idx"].is_unique,
    "物品索引重复",
)

item_map = dict(zip(
    mapping["raw_item_id"],
    mapping["item_idx"].astype(int),
))

idx_to_raw = {
    int(idx): str(raw)
    for raw, idx in item_map.items()
}

NUM_ITEMS = len(item_map)

require(
    NUM_ITEMS == EXPECTED_ITEMS,
    f"物品数异常: {NUM_ITEMS}",
)

require(
    set(item_map.values())
    == set(range(1, NUM_ITEMS + 1)),
    "物品映射不连续",
)


# ============================================================
# 3. 加载 Epoch 33 最佳 checkpoint
# ============================================================

ckpt = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

require(
    ckpt["model_name"] == "RecentBucketSetR1SASRec",
    "Checkpoint 模型身份错误",
)

require(
    ckpt["dataset"] == "MovieLens-1M",
    "Checkpoint 数据集错误",
)

require(
    int(ckpt["epoch"]) == _RUN["best_epoch"],
    "Checkpoint 不是 Epoch33",
)

require(
    int(ckpt["seed"]) == args.seed,
    "Checkpoint seed 错误",
)

require(
    int(ckpt["num_items"]) == NUM_ITEMS,
    "Checkpoint 物品数错误",
)

require(
    int(ckpt["training_batch_size"]) == 128,
    "Checkpoint 训练 BS 不是128",
)

require(file_signature(checked_file(ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"]).stat()) == _CHECKPOINT_SIGNATURE,
        "Checkpoint stat changed while loading")
for _name in ("max_len", "hidden_size", "num_heads", "num_blocks", "dropout", "training_batch_size"):
    require(ckpt.get(_name) == _REFERENCE_CONFIG.get(_name), "Checkpoint/config architecture mismatch: " + _name)

model = Model(
    num_items=NUM_ITEMS,
    max_len=int(ckpt["max_len"]),
    hidden_size=int(ckpt["hidden_size"]),
    num_heads=int(ckpt["num_heads"]),
    num_blocks=int(ckpt["num_blocks"]),
    dropout=float(ckpt["dropout"]),
).to(device)

model.load_state_dict(
    ckpt["model_state_dict"],
    strict=True,
)

model.eval()

print("Checkpoint epoch:", ckpt["epoch"])
print("Checkpoint loading PASS")


# ============================================================
# 4. 读取冻结原始样本
# ============================================================

def read_csv(path, reference_ids=None):
    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_items": str,
        },
    )

    require(
        len(df) == EXPECTED_N,
        f"样本数量异常: {path}",
    )

    require(
        df["sample_id"].is_unique,
        f"重复样本ID: {path}",
    )

    if reference_ids is not None:
        require(
            set(df["sample_id"]) == set(reference_ids),
            f"样本集合不同: {path}",
        )

        df = (
            df.set_index("sample_id")
            .loc[reference_ids]
            .reset_index()
        )

    return df


original_df = read_csv(ORIGINAL_PATH)
ids = original_df["sample_id"].tolist()
_ORIGINAL_ROWS = indexed_rows(original_df.to_dict(orient="records"))
_VARIANTS = {}

original_raw = [
    [str(x) for x in json.loads(v)]
    for v in original_df["sequence_items"]
]

bucket_sizes = (
    original_df["last_bucket_size"]
    .astype(int)
    .to_numpy()
)

targets_raw = original_df[
    "target_item"
].astype(str).tolist()


# ============================================================
# 5. R1 规范输入构造
# ============================================================

def construct(df, use_permutation=False):
    seq_list = []
    group_list = []
    token_list = []
    targets = []
    raw_histories = []

    for i, row in enumerate(
        df.itertuples(index=False)
    ):
        raw = [
            str(x)
            for x in json.loads(row.sequence_items)
        ]

        k = int(bucket_sizes[i])

        require(
            1 <= k <= len(raw) <= MAX_LEN,
            f"历史长度或桶大小异常: {row.sample_id}",
        )

        require(
            str(row.target_item) == targets_raw[i],
            f"Target 不一致: {row.sample_id}",
        )

        if use_permutation:
            require(
                int(row.last_history_bucket_size) == k,
                f"排列桶大小不一致: {row.sample_id}",
            )

            base_raw = original_raw[i]

            require(
                len(base_raw) == len(raw),
                f"历史长度变化: {row.sample_id}",
            )

            require(
                base_raw[:-k] == raw[:-k],
                f"历史前缀变化: {row.sample_id}",
            )

            require(
                Counter(base_raw[-k:]) == Counter(raw[-k:]),
                f"最后桶物品集合变化: {row.sample_id}",
            )

        mapped = np.asarray(
            [item_map[x] for x in raw],
            dtype=np.int64,
        )

        target = item_map[targets_raw[i]]

        require(
            target not in mapped,
            f"目标泄漏: {row.sample_id}",
        )

        gp, num_tokens, pooled = build_groups(
            len(mapped),
            k,
            MAX_LEN,
        )

        if not use_permutation:
            require(
                pooled == int(row.direct_pooling),
                f"Pooling 标签不一致: {row.sample_id}",
            )

        canonical = canonicalize(
            mapped,
            k,
        )

        padded = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        padded[-len(canonical):] = canonical

        seq_list.append(padded)
        group_list.append(gp)
        token_list.append(num_tokens)
        targets.append(target)
        raw_histories.append(raw)

    return (
        np.stack(seq_list),
        np.stack(group_list),
        np.asarray(token_list, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        raw_histories,
    )


base_seq, base_gp, base_tok, base_target, _ = (
    construct(original_df)
)


# ============================================================
# 6. 固定模型严格推理
# 返回未屏蔽 logits，用于真正的数值不变性核验
# ============================================================

@torch.inference_mode()
def predict_all(seq_array, gp_array, tok_array, targets):
    model.eval()

    logits_chunks = []
    ranks_chunks = []
    top10_chunks = []

    for start in range(
        0,
        len(targets),
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            len(targets),
        )

        seq_np = seq_array[start:end]
        target_np = targets[start:end]

        seq = torch.from_numpy(
            seq_np
        ).long().to(device)

        groups = torch.from_numpy(
            gp_array[start:end]
        ).long().to(device)

        tokens = torch.from_numpy(
            tok_array[start:end]
        ).long().to(device)

        raw_scores = model.predict_batch_masked(
            seq,
            groups,
            tokens,
        )

        require(
            torch.isfinite(raw_scores).all().item(),
            "模型输出包含 NaN/Inf",
        )

        logits_chunks.append(
            raw_scores.cpu().numpy().copy()
        )

        scores = raw_scores.clone()
        scores[:, 0] = -1e9

        for b in range(len(seq_np)):
            seen = np.unique(seq_np[b])

            seen = seen[
                (seen != 0)
                & (seen != target_np[b])
            ]

            if len(seen):
                seen_tensor = torch.as_tensor(
                    seen,
                    dtype=torch.long,
                    device=device,
                )

                scores[b, seen_tensor] = -1e9

        target_tensor = torch.as_tensor(
            target_np,
            dtype=torch.long,
            device=device,
        )

        index = torch.arange(
            len(target_np),
            device=device,
        )

        target_scores = scores[
            index,
            target_tensor,
        ]

        ranks = (
            (
                scores > target_scores[:, None]
            ).sum(dim=1)
            + 1
        )

        top10 = torch.topk(
            scores,
            k=10,
            dim=1,
        ).indices

        ranks_chunks.append(
            ranks.cpu().numpy()
        )

        top10_chunks.append(
            top10.cpu().numpy()
        )

    return (
        np.concatenate(logits_chunks),
        np.concatenate(ranks_chunks),
        np.concatenate(top10_chunks),
    )


def make_predictions(df, ranks, top10):
    pred = df[
        ["sample_id", "user_id", "target_item"]
    ].copy()

    pred["target_rank"] = ranks

    pred["top10_item_idx"] = [
        json.dumps(row.tolist())
        for row in top10
    ]

    pred["top10_raw_item_id"] = [
        json.dumps(
            [idx_to_raw[int(x)] for x in row]
        )
        for row in top10
    ]

    return pred


# ============================================================
# 7. 复现原始严格测试结果
# ============================================================

base_logits, base_ranks, base_top10 = predict_all(
    base_seq,
    base_gp,
    base_tok,
    base_target,
)

saved = read_csv_predictions = pd.read_csv(
    SAVED_PRED,
    dtype={"sample_id": str},
)

require(
    saved["sample_id"].is_unique,
    "已保存预测中出现重复样本",
)

require(set(saved["sample_id"]) == set(ids), "Saved baseline sample panel differs")
saved = (
    saved.set_index("sample_id")
    .loc[ids]
    .reset_index()
)

saved_ranks = (
    saved["target_rank"]
    .to_numpy(dtype=np.int64)
)

saved_top10 = np.stack([
    np.asarray(
        json.loads(v),
        dtype=np.int64,
    )
    for v in saved["top10_item_idx"]
])

rank_mismatches = int(
    np.count_nonzero(
        base_ranks != saved_ranks
    )
)

top10_mismatches = int(
    np.count_nonzero(
        np.any(
            base_top10 != saved_top10,
            axis=1,
        )
    )
)

metrics = json.loads(
    METRICS_PATH.read_text(encoding="utf-8")
)

ndcg = float(
    np.where(
        base_ranks <= 10,
        1.0 / np.log2(base_ranks + 1),
        0.0,
    ).mean()
)

require(
    rank_mismatches == 0,
    "原始目标排名未能复现",
)

require(
    top10_mismatches == 0,
    "原始 Top-10 未能复现",
)

require(
    abs(ndcg - metrics["test"]["ndcg10"]) < 1e-10,
    "原始 NDCG 未能复现",
)

print()
print("===== ORIGINAL STRICT REPRODUCTION =====")
print("Samples:", len(base_ranks))
print("Rank mismatches:", rank_mismatches)
print("Top10 mismatches:", top10_mismatches)
print("Original Test NDCG@10:", f"{ndcg:.9f}")
_IDENTITY_TOP10_MISMATCHES = verify_original_top10(
    make_predictions(original_df, base_ranks, base_top10).to_dict(orient="records"), saved.to_dict(orient="records"))
require(abs(ndcg - _RUN["test_ndcg10"]) < 1e-10, "Baseline NDCG differs from the audited run")
print("ORIGINAL STRICT REPRODUCTION PASS: ranks, ordered Top10, raw IDs and sample identity")


# ============================================================
# 8. 五组真实排列推理
# ============================================================

original_pred = make_predictions(
    original_df,
    base_ranks,
    base_top10,
)

results = {}
seed_summaries = []

for seed in SEEDS:
    perm_df = read_csv(
        ROOT / _PLAN["permutation_paths"][str(seed)],
        reference_ids=ids,
    )

    _VARIANTS[str(seed)] = verify_permutation(_ORIGINAL_ROWS, perm_df.to_dict(orient="records"))
    seq, gp, tok, target, raw_histories = construct(
        perm_df,
        use_permutation=True,
    )

    changed = np.asarray([
        a != b
        for a, b in zip(
            original_raw,
            raw_histories,
        )
    ])

    changed_n = int(changed.sum())

    require(
        changed_n > 0,
        f"Seed {seed}: 未发现扰动样本",
    )

    # Input invariant: grouped canonicalized tensors
    require(
        np.array_equal(seq, base_seq)
        and np.array_equal(gp, base_gp)
        and np.array_equal(tok, base_tok)
        and np.array_equal(target, base_target),
        f"Seed {seed}: R1 规范化输入不一致",
    )

    logits, ranks, top10 = predict_all(
        seq,
        gp,
        tok,
        target,
    )

    max_abs_diff = float(
        np.max(
            np.abs(
                logits.astype(np.float64)
                - base_logits.astype(np.float64)
            )
        )
    )

    ranks_changed = int(
        np.count_nonzero(
            ranks != base_ranks
        )
    )

    top10_changed = int(
        np.count_nonzero(
            np.any(
                top10 != base_top10,
                axis=1,
            )
        )
    )

    require(
        np.array_equal(logits, base_logits),
        f"Seed {seed}: Logits 不完全一致，"
        f"最大差异={max_abs_diff}",
    )

    require(
        ranks_changed == 0
        and top10_changed == 0,
        f"Seed {seed}: 排名或 Top10 发生变化",
    )

    results[seed] = make_predictions(
        perm_df,
        ranks,
        top10,
    )

    summary = {
        "permutation_seed": seed,
        "samples": EXPECTED_N,
        "changed_sequences": changed_n,
        "canonical_input_mismatches": 0,
        "max_abs_logit_diff": max_abs_diff,
        "changed_target_ranks": ranks_changed,
        "changed_top10": top10_changed,
        "top10_overlap_changed": 1.0,
    }

    seed_summaries.append(summary)

    print(
        f"Seed {seed}: "
        f"ChangedSeq={changed_n} "
        f"ChangedRank={ranks_changed} "
        f"ChangedTop10={top10_changed} "
        f"MaxLogitDiff={max_abs_diff:.12g} "
        f"OverlapChanged=1.000000",
        flush=True,
    )


# ============================================================
# 9. 全部通过后保存，避免部分成功文件污染
# ============================================================

_LABELS = panel_labels(_ORIGINAL_ROWS, _VARIANTS)
control_labels_match(_LABELS, _CONTROL_LABEL_PATH)
_CHECKPOINT_AFTER = file_signature(checked_file(ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"]).stat())
require(_CHECKPOINT_AFTER == _CHECKPOINT_SIGNATURE, "Checkpoint stat changed during inference")
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
# Copy only after raw-event labels/hashes/affected flags exactly match Control.
(OUTPUT_DIR / "panel_labels.csv").write_bytes(_CONTROL_LABEL_PATH.read_bytes())

original_pred.to_csv(
    OUTPUT_DIR / "predictions_original.csv",
    index=False,
    encoding="utf-8-sig",
)

for seed, pred in results.items():
    pred.to_csv(
        OUTPUT_DIR / f"predictions_perm_seed{seed}.csv",
        index=False,
        encoding="utf-8-sig",
    )

pd.DataFrame(seed_summaries).to_csv(
    OUTPUT_DIR / "per_seed_invariance.csv",
    index=False,
)

manifest = {
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
}

(OUTPUT_DIR / "metadata.json").write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("=" * 72)
print("STEP38.22 ML1M R1 STABILITY PASS")
print("All five permutations: exact logits/ranks/Top10 invariant")
print("Output:", OUTPUT_DIR)
print("=" * 72)
