#!/usr/bin/env python3
# Evaluation only: no training/optimizer steps.
import argparse
from collections import Counter
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


PERMUTATIONS = (101, 202, 303, 404, 505)
parser = argparse.ArgumentParser(description='Fixed-model last-bucket stability evaluation')
parser.add_argument('--seed', type=int, choices=[42, 2026, 3407], required=True)
args = parser.parse_args()
_STABILITY_ROOT = plain(Path('/root/autodl-tmp/time_uncertainty_rec'))
_STABILITY_STAGE = plain(Path('/root/autodl-tmp/time_uncertainty_rec/kais_control_stability_v1_20261023T032552Z_c9383ff8a16c'))
if Path(__file__).resolve(strict=False).parent != _STABILITY_STAGE / 'scripts':
    raise RuntimeError('Use the original generated evaluator stage')
_TAG = 'sports'
_PLAN = {'dataset': 'Amazon_Sports', 'expected_samples': 4659, 'eval_source': 'step38_kais_retrain/scripts/38_14_eval_sports_sasrec_stability.py', 'original_path': 'processed/strict_diagnostic_a/Amazon_Sports/file_order/test.csv', 'permutation_paths': {'101': 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed101/test.csv', '202': 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed202/test.csv', '303': 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed303/test.csv', '404': 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed404/test.csv', '505': 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed505/test.csv'}, 'runs': {'42': {'model_dir': 'kais_orderfix_v1_20261022T081312Z_e335970a03a1/outputs/amazon_sports_control_orderfix_seed42', 'files_sha256': {'config.json': '95963620e83b1e06dcda21f947394b2adad4b0c7259b3f7b4f99a759d9493989', 'metrics.json': 'ba3afef8d2451b67d6b4cd3d7cfde2479129288cdfb4bf305b545d3780e67d5b', 'training_log.csv': '798891b74b0df856d8fc5122570a514a3c3819bfa62d3e1f6fb6c676b6284581', 'test_per_sample_predictions.csv': '9ef7554252ff843a1b5bf7e3809aba4827c96427aee66744f06de70852119e5e', 'item_id_map.csv': '8f0e80ba0e996b19409cd72b3ed838242397f548cf38604ffd73d58677e29fee'}, 'best_epoch': 47, 'test_ndcg10': 0.008110770095710542, 'checkpoint': 'matched_control_sasrec_best.pt'}, '2026': {'model_dir': 'kais_orderfix_v1_20261022T081312Z_e335970a03a1/outputs/amazon_sports_control_orderfix_seed2026', 'files_sha256': {'config.json': '989375ab094dfab8b6e70bc49d9f7c5a86d4b9b2218f0e83b2a617b21e203862', 'metrics.json': '8316ebaf8ecc0a15aead0edaf15432d2fb29395543c7cd72ff63687972279c34', 'training_log.csv': '9fa39f1a9b16426ac08c38f8da5af4aff6aae9699c4d050a6e956e974690521c', 'test_per_sample_predictions.csv': '9c275fced2d98cda43257f1337908991e8c2ea970a494e16d0a0db06d24c9f53', 'item_id_map.csv': '8f0e80ba0e996b19409cd72b3ed838242397f548cf38604ffd73d58677e29fee'}, 'best_epoch': 4, 'test_ndcg10': 0.004824070488883555, 'checkpoint': 'matched_control_sasrec_best.pt'}, '3407': {'model_dir': 'kais_orderfix_v1_20261022T081312Z_e335970a03a1/outputs/amazon_sports_control_orderfix_seed3407', 'files_sha256': {'config.json': '0234af09fccef24382d9b19c6da32faefb553140606822da2efcb9fe75aa8858', 'metrics.json': 'a5cbcb983e152db7a7c9d599ca203eeeb5b58a03868827dd37c183b9faf7ed26', 'training_log.csv': 'c25c05321bf6a86aa4beed56e31f937d8d4646b4c7c566908c5fb71f9e53211f', 'test_per_sample_predictions.csv': '68db693ced7634bbb1fd18faf2152b62114ecaea80f3a5a5474b80ec431f7474', 'item_id_map.csv': '8f0e80ba0e996b19409cd72b3ed838242397f548cf38604ffd73d58677e29fee'}, 'best_epoch': 37, 'test_ndcg10': 0.009272276553271529, 'checkpoint': 'matched_control_sasrec_best.pt'}}, '_data_sha256': {'processed/strict_diagnostic_a/Amazon_Sports/file_order/test.csv': '5f2f31713ee7f6c92bb6fde915500d51f1a49328f3f1fc94f146530ad54e0e89', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed101/test.csv': 'ae1cc356e685451a400db4266f53e0cce42db12ede88f590de418a5c5a7ba3db', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed202/test.csv': '88d1d557a1c4f079d261aae9a01326f836caf67fa8b51ab877e87110bd42e1f0', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed303/test.csv': 'c284d34cf4d3193826e9a489dd31943e29f1fef86be74753e497f831620ddd63', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed404/test.csv': 'c5cf2bdf1b8b87beb1db2397038d1a546938a764c321696ab2ae6b062e5f3bf8', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed505/test.csv': 'dadca45252214e7b94bac0c0aedf9129182c2133be875fba77b3c748d3a0bff8'}}
_RUN = _PLAN["runs"][str(args.seed)]
_INPUTS_SHA256 = '8e8d13ded1ded7729e7f951b97d65ef3bda84614f4f4a9ad881bec68f7c81458'
_MODEL_SHA256 = 'ee30419e7c3b2086353d1d815207ef488a8b7eb980266a02115bf9a9cd1d1434'
_DATA_SHA256 = {'processed/strict_diagnostic_a/Amazon_Sports/file_order/test.csv': '5f2f31713ee7f6c92bb6fde915500d51f1a49328f3f1fc94f146530ad54e0e89', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed101/test.csv': 'ae1cc356e685451a400db4266f53e0cce42db12ede88f590de418a5c5a7ba3db', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed202/test.csv': '88d1d557a1c4f079d261aae9a01326f836caf67fa8b51ab877e87110bd42e1f0', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed303/test.csv': 'c284d34cf4d3193826e9a489dd31943e29f1fef86be74753e497f831620ddd63', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed404/test.csv': 'c5cf2bdf1b8b87beb1db2397038d1a546938a764c321696ab2ae6b062e5f3bf8', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed505/test.csv': 'dadca45252214e7b94bac0c0aedf9129182c2133be875fba77b3c748d3a0bff8'}
for _relative, _expected in _DATA_SHA256.items():
    checked_sha256(_STABILITY_ROOT, _relative, _expected)
for _name, _expected in _RUN["files_sha256"].items():
    checked_sha256(_STABILITY_ROOT, _RUN["model_dir"] + "/" + _name, _expected)
checked_sha256(_STABILITY_STAGE, 'scripts/model_definition.py', _MODEL_SHA256)
_CHECKPOINT_PATH = checked_file(_STABILITY_ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"])
_CHECKPOINT_SIGNATURE = file_signature(_CHECKPOINT_PATH.stat())
_REFERENCE_CONFIG = json.loads(checked_file(_STABILITY_ROOT, _RUN["model_dir"] + "/config.json").read_text(encoding="utf-8-sig"))
print('INTEGRITY PASS: six fixed input CSVs, five reviewed run files and frozen class; weights not yet read')

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Step38.14 - SPORTS SASRec Control Stability Evaluation
#
# Fixed checkpoint: Step38.1, best epoch 66
# Permutation seeds: 101, 202, 303, 404, 505
# No training; no parameter updates
# ============================================================

ROOT = _STABILITY_ROOT
BASE = _STABILITY_STAGE

DATASET = "Amazon_Sports"
MODEL_SEED = args.seed
PERMUTATION_SEEDS = [101, 202, 303, 404, 505]

MAX_LEN = 50
EVAL_BATCH_SIZE = 256
EXPECTED_SAMPLES = 4659

MODEL_SCRIPT = _STABILITY_STAGE / "scripts/model_definition.py"

MODEL_DIR = ROOT / _RUN["model_dir"]

CHECKPOINT_PATH = _CHECKPOINT_PATH

ITEM_MAP_PATH = MODEL_DIR / "item_id_map.csv"

ORIGINAL_PATH = ROOT / _PLAN["original_path"]

PERM_ROOT = (
    ROOT
    / "processed/strict_last_bucket_only"
    / DATASET
)

OUTPUT_DIR = _STABILITY_STAGE / "outputs" / f"{_TAG}_seed{MODEL_SEED}"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 72)
print("STEP38.14 SPORTS SASREC STABILITY")
print("=" * 72)
print("Device:", device)
print("Model seed:", MODEL_SEED)
print("Permutation seeds:", PERMUTATION_SEEDS)


# ============================================================
# 1. 文件检查
# ============================================================

required_files = [
    MODEL_SCRIPT,
    CHECKPOINT_PATH,
    ITEM_MAP_PATH,
    ORIGINAL_PATH,
]

for seed in PERMUTATION_SEEDS:
    required_files.append(
        ROOT / _PLAN["permutation_paths"][str(seed)]
    )

for path in required_files:
    if not path.is_file():
        raise FileNotFoundError(path)

if OUTPUT_DIR.exists():
    raise RuntimeError(
        f"输出目录已存在，禁止覆盖: {OUTPUT_DIR}"
    )

print("Input files: PASS")


# ============================================================
# 2. 提取 Step38 原始模型类
#
# 只提取 SASRecControl 类，不执行训练脚本。
# 保证与正式训练的模型定义一致。
# ============================================================

tree = ast.parse(
    MODEL_SCRIPT.read_text(encoding="utf-8")
)

classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef)
    and node.name == "SASRecControl"
]

if len(classes) != 1:
    raise RuntimeError("SASRecControl 类数量异常")

namespace = {
    "__name__": "step38_stability",
    "torch": torch,
    "nn": nn,
    "F": F,
    "np": np,
    "math": math,
    "MAX_LEN": MAX_LEN,
    "device": device,
}

exec(
    compile(
        ast.Module(
            body=classes,
            type_ignores=[]
        ),
        str(MODEL_SCRIPT),
        "exec",
    ),
    namespace,
)

SASRecControl = namespace["SASRecControl"]


# ============================================================
# 3. 加载物品映射
# ============================================================

mapping = pd.read_csv(
    ITEM_MAP_PATH,
    dtype={"raw_item_id": str}
)

if not mapping["raw_item_id"].is_unique:
    raise RuntimeError("物品原始ID重复")

if not mapping["item_idx"].is_unique:
    raise RuntimeError("物品内部ID重复")

item_map = dict(zip(
    mapping["raw_item_id"],
    mapping["item_idx"].astype(int),
))

idx_to_raw = {
    int(idx): str(raw)
    for raw, idx in item_map.items()
}

NUM_ITEMS = len(item_map)

if NUM_ITEMS != 18357:
    raise RuntimeError(
        f"物品数异常: {NUM_ITEMS}"
    )

if set(item_map.values()) != set(
    range(1, NUM_ITEMS + 1)
):
    raise RuntimeError("物品映射不连续")

print("NUM_ITEMS:", NUM_ITEMS)


# ============================================================
# 4. 加载 Step38.1 最佳 checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
)

if checkpoint["model_name"] != "SASRecControl":
    raise RuntimeError("模型身份不一致")

if int(checkpoint["epoch"]) != _RUN["best_epoch"]:
    raise RuntimeError("Checkpoint最佳轮次不是66")

if int(checkpoint["num_items"]) != NUM_ITEMS:
    raise RuntimeError("Checkpoint物品数不一致")

if int(checkpoint["seed"]) != MODEL_SEED:
    raise RuntimeError("Checkpoint随机种子不一致")

if file_signature(checked_file(ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"]).stat()) != _CHECKPOINT_SIGNATURE:
    raise RuntimeError("Checkpoint changed during loading")
if checkpoint.get("dataset") != DATASET:
    raise RuntimeError("Checkpoint dataset differs from reviewed run")
for _name in ("max_len", "hidden_size", "num_heads", "num_blocks", "dropout"):
    if checkpoint.get(_name) != _REFERENCE_CONFIG.get(_name):
        raise RuntimeError("Checkpoint architecture differs from saved config: " + _name)

model = SASRecControl(
    num_items=NUM_ITEMS,
    max_len=int(checkpoint["max_len"]),
    hidden_size=int(checkpoint["hidden_size"]),
    num_heads=int(checkpoint["num_heads"]),
    num_blocks=int(checkpoint["num_blocks"]),
    dropout=float(checkpoint["dropout"]),
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.eval()

print("Checkpoint epoch:", checkpoint["epoch"])
print("Checkpoint loaded: PASS")


# ============================================================
# 5. 冻结样本读取
# ============================================================

def read_samples(path, reference_ids=None):
    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_items": str,
        },
    )

    if len(df) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"样本数量异常: {path}, {len(df)}"
        )

    if not df["sample_id"].is_unique:
        raise RuntimeError(
            f"存在重复 sample_id: {path}"
        )

    if reference_ids is not None:
        if set(df["sample_id"]) != set(reference_ids):
            raise RuntimeError(
                f"样本ID与冻结测试集不一致: {path}"
            )

        df = (
            df.set_index("sample_id")
            .loc[reference_ids]
            .reset_index()
        )

    sequences = []
    lengths = []
    targets = []

    for row in df.itertuples(index=False):
        raw_history = json.loads(
            row.sequence_items
        )

        history = np.asarray(
            [item_map[str(x)] for x in raw_history],
            dtype=np.int64,
        )

        if not (1 <= len(history) <= MAX_LEN):
            raise RuntimeError(
                f"历史长度异常: {row.sample_id}"
            )

        target = item_map[str(row.target_item)]

        if target in history:
            raise RuntimeError(
                f"目标泄漏进历史: {row.sample_id}"
            )

        seq = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        seq[-len(history):] = history

        sequences.append(seq)
        lengths.append(len(history))
        targets.append(target)

    return (
        df,
        np.stack(sequences),
        np.asarray(lengths, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    )


# ============================================================
# 6. 固定模型推理
#
# 与 Step38.1 保持一致：
# - Masked Forward
# - Padding ID=0 屏蔽
# - 历史已见物品屏蔽
# - 排名按严格大于目标分数计算
# ============================================================

@torch.inference_mode()
def evaluate(df, sequences, lengths, targets):
    total = len(df)

    all_ranks = np.empty(
        total,
        dtype=np.int64,
    )

    all_top10 = np.empty(
        (total, 10),
        dtype=np.int64,
    )

    model.eval()

    for start in range(
        0, total, EVAL_BATCH_SIZE
    ):
        end = min(
            start + EVAL_BATCH_SIZE,
            total
        )

        seq_np = sequences[start:end]
        target_np = targets[start:end]

        seq_tensor = torch.from_numpy(
            seq_np
        ).long().to(device)

        token_tensor = torch.from_numpy(
            lengths[start:end]
        ).long().to(device)

        scores = model.predict_batch_masked(
            seq_tensor,
            token_tensor,
        )

        if not torch.isfinite(scores).all():
            raise RuntimeError(
                "推理分数出现 NaN/Inf"
            )

        scores[:, 0] = -1e9

        # 与 Step38 严格评价一致
        for b in range(len(seq_np)):
            seen = np.unique(
                seq_np[b]
            )

            seen = seen[
                (seen != 0)
                & (seen != target_np[b])
            ]

            if len(seen) > 0:
                seen_tensor = torch.as_tensor(
                    seen,
                    device=device,
                    dtype=torch.long,
                )

                scores[b, seen_tensor] = -1e9

        target_tensor = torch.from_numpy(
            target_np
        ).long().to(device)

        row_index = torch.arange(
            len(target_tensor),
            device=device,
        )

        target_scores = scores[
            row_index,
            target_tensor,
        ]

        if not torch.isfinite(target_scores).all():
            raise RuntimeError(
                "目标分数异常"
            )

        ranks = (
            (
                scores
                > target_scores.unsqueeze(1)
            ).sum(dim=1)
            + 1
        )

        top10 = torch.topk(
            scores,
            k=10,
            dim=1,
        ).indices

        all_ranks[start:end] = (
            ranks.cpu().numpy()
        )

        all_top10[start:end] = (
            top10.cpu().numpy()
        )

    result = df[
        [
            "sample_id",
            "user_id",
            "target_item",
        ]
    ].copy()

    result["target_rank"] = all_ranks

    result["top10_item_idx"] = [
        json.dumps(row.tolist())
        for row in all_top10
    ]

    result["top10_raw_item_id"] = [
        json.dumps(
            [
                idx_to_raw[int(x)]
                for x in row
            ]
        )
        for row in all_top10
    ]

    return result


# ============================================================
# 7. 原始测试集复算
# ============================================================

original_df, seq, lens, targets = read_samples(
    ORIGINAL_PATH
)

reference_ids = (
    original_df["sample_id"].tolist()
)
_ORIGINAL_ROWS = indexed_rows(original_df.to_dict(orient="records"))
_VARIANTS = {}

original_pred = evaluate(
    original_df,
    seq,
    lens,
    targets,
)

# 与 Step38.1 已保存的严格测试排名逐样本比较
saved_path = (
    MODEL_DIR
    / "test_per_sample_predictions.csv"
)

saved = pd.read_csv(
    saved_path,
    dtype={"sample_id": str}
)

comparison = original_pred[
    ["sample_id", "target_rank"]
].merge(
    saved[["sample_id", "target_rank"]],
    on="sample_id",
    validate="one_to_one",
    suffixes=("_new", "_saved"),
)

if len(comparison) != EXPECTED_SAMPLES:
    raise RuntimeError(
        "严格测试复算样本数量异常"
    )

mismatch = int(
    (
        comparison["target_rank_new"]
        != comparison["target_rank_saved"]
    ).sum()
)

print()
print("===== ORIGINAL STRICT REPRODUCTION =====")
print("Samples:", len(comparison))
print("Rank mismatches:", mismatch)

if mismatch != 0:
    raise RuntimeError(
        "Step38.1 严格排名复算失败"
    )

ranks = original_pred[
    "target_rank"
].to_numpy()

ndcg = np.where(
    ranks <= 10,
    1.0 / np.log2(ranks + 1),
    0.0,
).mean()

print("Original Test NDCG@10:", f"{ndcg:.6f}")

if abs(ndcg - _RUN["test_ndcg10"]) >= 1e-6:
    raise RuntimeError(
        "严格 NDCG@10 与 Step38.1 不一致"
    )

_TOP10_MISMATCHES = verify_original_top10(
    original_pred.to_dict(orient="records"), saved.to_dict(orient="records"))
print("ORIGINAL STRICT REPRODUCTION PASS: ranks and exact ordered Top10")


# ============================================================
# 8. 评价五组最后时间桶排列
# ============================================================

predictions = {}

for seed in PERMUTATION_SEEDS:
    input_path = ROOT / _PLAN["permutation_paths"][str(seed)]

    print()
    print("=" * 60)
    print("Evaluating permutation seed:", seed)

    df, seq, lens, targets = read_samples(
        input_path,
        reference_ids=reference_ids,
    )

    # 检查目标物品没有发生变化
    if not (
        df["target_item"].astype(str).to_numpy()
        == original_df["target_item"].astype(str).to_numpy()
    ).all():
        raise RuntimeError(
            f"Seed {seed} 目标不一致"
        )

    _VARIANTS[str(seed)] = verify_permutation(_ORIGINAL_ROWS, df.to_dict(orient="records"))
    result = evaluate(
        df,
        seq,
        lens,
        targets,
    )

    predictions[seed] = result

    rank = result[
        "target_rank"
    ].to_numpy()

    changed_rank = int(
        (
            rank
            != original_pred["target_rank"].to_numpy()
        ).sum()
    )

    print("Samples:", len(result))
    print("Changed target ranks:", changed_rank)
    print("MeanRank:", f"{rank.mean():.2f}")


# ============================================================
# 9. 全部计算成功后保存
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

_LABELS = panel_labels(_ORIGINAL_ROWS, _VARIANTS)
pd.DataFrame(_LABELS).to_csv(OUTPUT_DIR / "panel_labels.csv", index=False, encoding="utf-8-sig")

original_pred.to_csv(
    OUTPUT_DIR / "predictions_original.csv",
    index=False,
    encoding="utf-8-sig",
)

for seed, result in predictions.items():
    output_path = (
        OUTPUT_DIR
        / f"predictions_perm_seed{seed}.csv"
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved:", output_path.name)

metadata = {
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
    "dataset": DATASET,
    "model": "SASRecControl",
    "model_seed": MODEL_SEED,
    "best_epoch": int(checkpoint["epoch"]),
    "num_items": NUM_ITEMS,
    "test_samples": EXPECTED_SAMPLES,
    "permutation_seeds": PERMUTATION_SEEDS,
    "protocol": "Step38_last_bucket_only_fixed_model",
    "strict_reproduction": "PASS",
    "training_batch_size": 128,
    "evaluation_batch_size": EVAL_BATCH_SIZE,
}

(OUTPUT_DIR / "metadata.json").write_text(
    json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("=" * 72)
print("STEP38.14 SPORTS STABILITY INFERENCE PASS")
print("Output:", OUTPUT_DIR)
print("=" * 72)
