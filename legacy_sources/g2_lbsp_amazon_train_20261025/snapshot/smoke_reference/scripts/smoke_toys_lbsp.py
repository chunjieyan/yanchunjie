# Generated smoke-only script. Hash checks run before NumPy/Torch imports.
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import math
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

_SMOKE_ROOT = Path('/root/autodl-tmp/time_uncertainty_rec')
_SMOKE_STAGE = Path('/root/autodl-tmp/time_uncertainty_rec/kais_lbsp_amazon_smoke_v1_20261023T073215Z_9eeed6c3de08')
_TAG = 'toys'
_INPUTS = json.loads('{"version": "lbsp_amazon_smoke_v1", "seed": 42, "datasets": {"beauty": {"dataset": "Amazon_Beauty", "seed": 42, "expected_items": 12101, "expected_val": 2856, "expected_test": 2966, "expected_positions": 138167, "train_path": "processed/diagnostic_a_splits/Amazon_Beauty/file_order/train.csv", "validation_path": "processed/recent_bucketset_r1/Amazon_Beauty/val/r1_samples.csv", "test_path": "processed/recent_bucketset_r1/Amazon_Beauty/test/r1_samples.csv", "control_trainer_source": "step38_kais_retrain/scripts/38_4_train_beauty_sasrec_bs128.py", "sequence_sort_keys": ["user_id", "time_bucket", "_source_order"], "control_reference": {"model_dir": "step38_kais_retrain/outputs/beauty_sasrec_control_ce_bs128_lr1em3_max200_seed42", "files_sha256": {"config.json": "a453f393b09d374f25868d8d2a205953b2d0da064c41aad443568bc693ad8425", "metrics.json": "5e10b55ec8c550344e66c092ef6511ed524527f00d02d006627b8ec9206f9b2b", "training_log.csv": "2b22f6ad9d8be45e775c053d65ff2f34b4392c14ebaa571d8e0e353341a9d887", "test_per_sample_predictions.csv": "c9f0efd243532633594be0ca5b82f81aa1091facd8ad37ea06ffdb478330a40e", "item_id_map.csv": "259b8725bf180ff02c3f9a7351c43fdf1de22aa51829c40437c2040f95737289"}}}, "sports": {"dataset": "Amazon_Sports", "seed": 42, "expected_items": 18357, "expected_val": 4082, "expected_test": 4659, "expected_positions": 204047, "train_path": "processed/diagnostic_a_splits/Amazon_Sports/file_order/train.csv", "validation_path": "processed/strict_diagnostic_a/Amazon_Sports/file_order/val.csv", "test_path": "processed/strict_diagnostic_a/Amazon_Sports/file_order/test.csv", "control_trainer_source": "kais_orderfix_v1_20261022T081312Z_e335970a03a1/scripts/train_sports_orderfix.py", "sequence_sort_keys": ["user_id", "time_bucket", "raw_line"], "control_reference": {"model_dir": "kais_orderfix_v1_20261022T081312Z_e335970a03a1/outputs/amazon_sports_control_orderfix_seed42", "files_sha256": {"config.json": "95963620e83b1e06dcda21f947394b2adad4b0c7259b3f7b4f99a759d9493989", "metrics.json": "ba3afef8d2451b67d6b4cd3d7cfde2479129288cdfb4bf305b545d3780e67d5b", "training_log.csv": "798891b74b0df856d8fc5122570a514a3c3819bfa62d3e1f6fb6c676b6284581", "test_per_sample_predictions.csv": "9ef7554252ff843a1b5bf7e3809aba4827c96427aee66744f06de70852119e5e", "item_id_map.csv": "8f0e80ba0e996b19409cd72b3ed838242397f548cf38604ffd73d58677e29fee"}}}, "toys": {"dataset": "Amazon_Toys", "seed": 42, "expected_items": 11924, "expected_val": 1939, "expected_test": 2547, "expected_positions": 116686, "train_path": "processed/diagnostic_a_splits/Amazon_Toys/file_order/train.csv", "validation_path": "processed/strict_diagnostic_a/Amazon_Toys/file_order/val.csv", "test_path": "processed/strict_diagnostic_a/Amazon_Toys/file_order/test.csv", "control_trainer_source": "kais_orderfix_v1_20261022T081312Z_e335970a03a1/scripts/train_toys_orderfix.py", "sequence_sort_keys": ["user_id", "time_bucket", "raw_line"], "control_reference": {"model_dir": "kais_orderfix_v1_20261022T081312Z_e335970a03a1/outputs/amazon_toys_control_orderfix_seed42", "files_sha256": {"config.json": "18d9418f4eb6870929258deb4186a5ee160210d1bf57e55263b076d44a48b64e", "metrics.json": "2b7ce49edd8c7825f0b1d4f63db4895210e6b7339e640553c4b1924a0ab9b307", "training_log.csv": "057de30a3015899d2d1ebcfcabfc0731e2f05c864fd6dddfbc9ed97402c1f1ac", "test_per_sample_predictions.csv": "9b4b5854ff4c05a6b563236e2a6b436d83931f640bf435725294ee2e0fb89cd9", "item_id_map.csv": "d6e1135f71ad9749caa6db9743045e1e65860bd62df6f42e0b76642c08645c60"}}}}, "template_source": "step38_kais_retrain/scripts/38_20_train_ml1m_r1_bs128.py", "model_source": "audit/33_2_train_movielens_r1_seed42.py", "sources_sha256": {"step38_kais_retrain/scripts/38_20_train_ml1m_r1_bs128.py": "c718592e65a3d66f2ec1ce6d22c8b9abc3b8d4a393dde49ae014247414741b38", "audit/33_2_train_movielens_r1_seed42.py": "e8ae647f3fa3cc2ef9f3087a0ae44cfbd4da2706b2ce33fc558fff0678a04ffb", "step38_kais_retrain/scripts/38_4_train_beauty_sasrec_bs128.py": "29c528543757d18b54b18a8c09d4dd4e573e125cc5128e9a13ae702b4be5d48e", "kais_orderfix_v1_20261022T081312Z_e335970a03a1/scripts/train_sports_orderfix.py": "2fa42816658432c8e85f42b82d715aec4232c4c49d2b52e579b44da8f6a5f55a", "kais_orderfix_v1_20261022T081312Z_e335970a03a1/scripts/train_toys_orderfix.py": "490269b75a87b5aabdb9195233a958fce4d67a5a8f95df093faca9dc46d8e4bb"}, "data_sha256": {"processed/diagnostic_a_splits/Amazon_Beauty/file_order/train.csv": "c81fd1c1dbcbbe88ed4596e0d952397d1ff74ba70a5ba9d8f9e4038e72e05971", "processed/recent_bucketset_r1/Amazon_Beauty/val/r1_samples.csv": "0f19f717b9c8a3230c8ce52735960fa48f4fb5e2c08c1787e0a4c52dbfbb5eb9", "processed/recent_bucketset_r1/Amazon_Beauty/test/r1_samples.csv": "f9362385825ed79b07e4d262cf9ac7df2591c54cd7d8e95e54ba116bdaad5009", "processed/diagnostic_a_splits/Amazon_Sports/file_order/train.csv": "8b78398cf350881125d04a15cd0b6181b17d1772d35f428ed61d3e9220a16539", "processed/strict_diagnostic_a/Amazon_Sports/file_order/val.csv": "8ffe18f11957ba4229aa42d4e1b5d2ffdfbf3ef2adb4ecd0ff3b349469cbe7fb", "processed/strict_diagnostic_a/Amazon_Sports/file_order/test.csv": "5f2f31713ee7f6c92bb6fde915500d51f1a49328f3f1fc94f146530ad54e0e89", "processed/diagnostic_a_splits/Amazon_Toys/file_order/train.csv": "26e4e892b1e520fdc213bddad4af3f314e865f53a1c8cbcf1a4e0f60a83ff8c0", "processed/strict_diagnostic_a/Amazon_Toys/file_order/val.csv": "5b2187611068dbf7f384e55489d3704710387ef42bf86fcf028836ae26e8f14d", "processed/strict_diagnostic_a/Amazon_Toys/file_order/test.csv": "3e287ca6be6c556d5abfd691666abdc4fde1eca10327d6b8329a8aa55d0df445"}, "hyperparameters": {"max_len": 50, "hidden_size": 64, "num_heads": 2, "num_blocks": 2, "dropout": 0.2, "training_batch_size": 128, "evaluation_batch_size": 256, "learning_rate": 0.001, "max_epochs": 200, "patience": 15}, "scope": "Three datasets LBSP smoke checks only; random initialized models, finite gradients, masked/unpadded parity, last-bucket permutation invariant logits, 32 validation samples. No optimizer update, formal training, saved model weights or test inference; no effectiveness claims."}')
_INPUTS_SHA256 = 'c65912d443b01d2f76ed164e57638a99ad1b1972394d542b5f22fe1eacd92265'
_MODEL_SHA256 = '500244e2a920b3d256176b6cad0c79031d9dc327b9b2b812214005f5764a6a6a'
_PLAN = _INPUTS['datasets'][_TAG]
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


import argparse
import ast
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler


# ============================================================
# Step38.20 MovieLens-1M R1 Mean Pooling
# Full legal positions / BS128 / Strict evaluation
# ============================================================

ROOT = _SMOKE_ROOT
BASE = _SMOKE_STAGE

DATASET = 'Amazon_Toys'
ORDER = "file_order"
SEED = 42

BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
LR = 0.001
MAX_EPOCHS = 200
PATIENCE = 15

MAX_LEN = 50
HIDDEN_SIZE = 64
NUM_HEADS = 2
NUM_BLOCKS = 2
DROPOUT = 0.2

EXPECTED_ITEMS = 11924
EXPECTED_VAL = 1939
EXPECTED_TEST = 2547
EXPECTED_POSITIONS = 116686

CONTROL_SCRIPT = ROOT / _PLAN["control_trainer_source"]

R1_SOURCE = _SMOKE_STAGE / "scripts/model_definition.py"

CONTROL_DIR = ROOT / _PLAN["control_reference"]["model_dir"]

MAP_PATH = CONTROL_DIR / "item_id_map.csv"

OUTPUT_DIR = _SMOKE_STAGE / "outputs" / 'toys_smoke_unused'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["smoke"],
    default="smoke",
)
args = parser.parse_args()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def require(condition, msg):
    if not condition:
        raise RuntimeError(msg)


print("=" * 72)
print("AMAZON LBSP SMOKE ONLY: toys")
print("=" * 72)
print("Mode:", args.mode)
print("Device:", device)
print("Training BS:", BATCH_SIZE)
print("LR:", LR)
print("Patience:", PATIENCE)

for p in [CONTROL_SCRIPT, R1_SOURCE, MAP_PATH]:
    require(p.is_file(), f"输入不存在: {p}")

require(
    not OUTPUT_DIR.exists(),
    f"输出目录已存在，禁止覆盖: {OUTPUT_DIR}",
)


# ============================================================
# 1. 读取 Step38 Control 原始路径定义
# 不执行其顶层训练程序
# ============================================================

TRAIN_PATH = ROOT / _PLAN["train_path"]
VAL_PATH = ROOT / _PLAN["validation_path"]
TEST_PATH = ROOT / _PLAN["test_path"]

for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    require(p.is_file(), f"冻结数据不存在: {p}")

print("Train:", TRAIN_PATH)
print("Val:", VAL_PATH)
print("Test:", TEST_PATH)


# ============================================================
# 2. 精确复用旧 R1 函数与模型类
# ============================================================

r1_tree = ast.parse(
    R1_SOURCE.read_text(encoding="utf-8")
)

required_functions = [
    "build_r1_group_positions_from_buckets",
    "build_r1_group_positions_from_size",
    "canonicalize_last_bucket_items",
]

required_classes = [
    "RecentBucketSetR1SASRec",
]

selected = []

for name in required_functions:
    found = [
        n for n in r1_tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == name
    ]

    require(
        len(found) == 1,
        f"R1 函数提取异常: {name}",
    )
    selected.append(found[0])

for name in required_classes:
    found = [
        n for n in r1_tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == name
    ]

    require(
        len(found) == 1,
        f"R1 模型类提取异常: {name}",
    )
    selected.append(found[0])

namespace = {
    "__name__": "step38_r1",
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
            body=selected,
            type_ignores=[],
        ),
        str(R1_SOURCE),
        "exec",
    ),
    namespace,
)

build_groups_from_buckets = namespace[
    "build_r1_group_positions_from_buckets"
]

build_groups_from_size = namespace[
    "build_r1_group_positions_from_size"
]

canonicalize = namespace[
    "canonicalize_last_bucket_items"
]

R1Model = namespace[
    "RecentBucketSetR1SASRec"
]

print("Original R1 model/functions extracted: PASS")


# ============================================================
# 3. 新增 BS128 Masked Forward
#
# R1 原始 Token 为左对齐。
# 这里转为右对齐后加入相同的位置编码，
# 与原始不含 padding 的 Forward 核对。
# ============================================================

def predict_batch_masked(
    self,
    seq,
    group_positions,
    num_tokens,
):
    batch_size, width = seq.shape

    require(
        width == self.max_len,
        "输入序列长度异常",
    )

    pooled = self.build_token_embeddings(
        seq,
        group_positions,
    )

    positions = torch.arange(
        width,
        device=seq.device,
    ).unsqueeze(0).expand(batch_size, width)

    source_index = (
        positions
        - (width - num_tokens).unsqueeze(1)
    )

    valid = source_index >= 0

    gather_index = (
        source_index
        .clamp(min=0)
        .unsqueeze(-1)
        .expand(-1, -1, self.hidden_size)
    )

    x = torch.gather(
        pooled,
        dim=1,
        index=gather_index,
    )

    x = x * valid.unsqueeze(-1)

    x = x + self.position_embedding(positions)
    x = self.dropout(x)

    # ----------------------------------------------------
    # Safe masked causal attention for left-padded tokens.
    #
    # 有效 Query：
    #   只能关注真实 Token，且保持因果约束。
    #
    # Padding Query：
    #   只允许关注自身，避免整行全部被屏蔽产生 NaN。
    #
    # Padding Query 不能影响任何有效 Query。
    # ----------------------------------------------------

    causal_mask = torch.triu(
        torch.ones(
            width,
            width,
            dtype=torch.bool,
            device=seq.device,
        ),
        diagonal=1,
    )

    # [B, L, L]
    attention_mask = (
        causal_mask.unsqueeze(0)
        | (~valid).unsqueeze(1)
    )

    # 仅为 Padding Query 放开自身这一位置。
    pad_self = (
        (~valid).unsqueeze(-1)
        & torch.eye(
            width,
            dtype=torch.bool,
            device=seq.device,
        ).unsqueeze(0)
    )

    attention_mask = (
        attention_mask & (~pad_self)
    )

    # nn.MultiheadAttention 使用 [B * Heads, L, L]
    num_heads = (
        self.encoder.layers[0].self_attn.num_heads
    )

    attention_mask = (
        attention_mask.repeat_interleave(
            num_heads,
            dim=0,
        )
    )

    x = self.encoder(
        x,
        mask=attention_mask,
    )

    x = self.final_norm(x)

    final_feature = x[:, -1, :]

    return torch.matmul(
        final_feature,
        self.item_embedding.weight.T,
    )


R1Model.predict_batch_masked = predict_batch_masked


# ============================================================
# 4. Control 冻结 item map
# ============================================================

mapping_df = pd.read_csv(
    MAP_PATH,
    dtype={"raw_item_id": str},
)

require(
    {"raw_item_id", "item_idx"}.issubset(mapping_df.columns),
    "物品映射字段异常",
)

require(
    mapping_df["raw_item_id"].is_unique,
    "raw_item_id 重复",
)

require(
    mapping_df["item_idx"].is_unique,
    "item_idx 重复",
)

item_map = dict(zip(
    mapping_df["raw_item_id"],
    mapping_df["item_idx"].astype(int),
))

idx_to_raw = {
    int(v): str(k)
    for k, v in item_map.items()
}

NUM_ITEMS = len(item_map)

require(
    NUM_ITEMS == EXPECTED_ITEMS,
    f"物品数异常: {NUM_ITEMS}",
)

require(
    set(item_map.values())
    == set(range(1, NUM_ITEMS + 1)),
    "物品索引不是1..NUM_ITEMS",
)

print("Control item map reused:", NUM_ITEMS)


# ============================================================
# 5. 训练序列
# 与 Step38 Control 一样：
# user + time_bucket + source_order
# ============================================================

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype={
        "user_id": str,
        "item_id": str,
    },
)

required_cols = {"user_id", "item_id", "raw_line", "time_bucket"}

if not required_cols.issubset(train_df.columns):
    raise RuntimeError(
        f"训练集缺少列: "
        f"{required_cols - set(train_df.columns)}"
    )

if train_df[
    ["user_id", "item_id", "raw_line"]
].isna().any().any():
    raise RuntimeError("训练集关键字段有缺失值")

# The sole scientific correction: canonical per-user temporal order.
for _key in ("time_bucket", "raw_line"):
    _values = train_df[_key].astype(str)
    if not _values.str.fullmatch(r"[+-]?[0-9]+").all():
        raise RuntimeError("Non-integer or missing training field: " + _key)
    _integers = [int(value) for value in _values]
    if any(value < -(2**63) or value >= 2**63 for value in _integers):
        raise RuntimeError("Training integer outside int64: " + _key)
    train_df[_key] = np.asarray(_integers, dtype=np.int64)

if train_df.duplicated(["user_id", "raw_line"]).any():
    raise RuntimeError("Duplicate (user_id, raw_line) event key")
_event_count = len(train_df)
_event_keys = set(zip(train_df["user_id"], train_df["raw_line"]))
train_df = train_df.sort_values(
    ["user_id", "time_bucket", "raw_line"], kind="stable",
)
if (len(train_df) != _event_count
        or set(zip(train_df["user_id"], train_df["raw_line"])) != _event_keys
        or sorted(train_df.index.tolist()) != list(range(_event_count))):
    raise RuntimeError("Ordering correction added or removed training events")
_same_user = train_df["user_id"].eq(train_df["user_id"].shift())
_bucket = train_df["time_bucket"]
_raw = train_df["raw_line"]
_bad_order = (_bucket.lt(_bucket.shift())
              | (_bucket.eq(_bucket.shift()) & _raw.le(_raw.shift())))
if (_same_user & _bad_order).any():
    raise RuntimeError("Canonical temporal/tie order assertion failed")
print("ORDERFIX PASS: unchanged event keys; user,time_bucket,raw_line stable order")

BUCKET_COLUMN = "time_bucket"

user_sequences = {}

for user_id, group in train_df.groupby(
    "user_id",
    sort=False,
):
    raw_items = group["item_id"].tolist()

    require(
        all(x in item_map for x in raw_items),
        "训练 item 未出现在 Control 映射中",
    )

    items = np.asarray(
        [item_map[x] for x in raw_items],
        dtype=np.int64,
    )

    buckets = group[
        BUCKET_COLUMN
    ].to_numpy()

    if len(items) >= 2:
        user_sequences[user_id] = {
            "items": items,
            "buckets": buckets,
        }

LEGAL_POSITIONS = sum(
    len(info["items"]) - 1
    for info in user_sequences.values()
)

print("Training rows:", len(train_df))
print("Trainable users:", len(user_sequences))
print("Legal next-item positions:", LEGAL_POSITIONS)
print("Bucket column:", BUCKET_COLUMN)

require(
    LEGAL_POSITIONS == EXPECTED_POSITIONS,
    "合法位置数量与 Step38 Control 不一致；"
    "禁止开始 R1 训练",
)


# ============================================================
# 6. 全部合法位置 R1 Dataset
# ============================================================

class AllPositionR1Dataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        self.users = sorted(sequences.keys())

        self.positions = [
            (user_id, p)
            for user_id in self.users
            for p in range(
                1,
                len(sequences[user_id]["items"]),
            )
        ]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        user_id, p = self.positions[idx]

        info = self.sequences[user_id]

        history_items = info["items"][
            max(0, p - MAX_LEN):p
        ]

        history_buckets = info["buckets"][
            max(0, p - MAX_LEN):p
        ]

        (
            groups,
            num_tokens,
            direct_pooling,
            last_bucket_size,
        ) = build_groups_from_buckets(
            history_buckets,
            MAX_LEN,
        )

        history_items = canonicalize(
            history_items,
            last_bucket_size,
        )

        seq = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        seq[-len(history_items):] = history_items

        return (
            torch.from_numpy(seq),
            torch.from_numpy(groups.copy()),
            torch.tensor(num_tokens),
            torch.tensor(int(info["items"][p])),
        )


train_dataset = AllPositionR1Dataset(
    user_sequences
)

require(
    len(train_dataset) == LEGAL_POSITIONS,
    "训练位置数量不一致",
)


# ============================================================
# 7. 严格冻结评价样本
# ============================================================

def load_eval(path, expected_n):
    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_items": str,
        },
    )
    df = normalize_eval_frame(df)

    require(
        len(df) == expected_n,
        f"评价样本数异常: {path}, {len(df)}",
    )

    require(
        df["sample_id"].is_unique,
        f"重复 sample_id: {path}",
    )

    require(
        {"last_bucket_size", "direct_pooling"}
        .issubset(df.columns),
        f"R1 冻结分组字段缺失: {path}",
    )

    sequences = []
    groups_all = []
    tokens_all = []
    targets = []

    for row in df.itertuples(index=False):
        history = [
            item_map[str(x)]
            for x in json.loads(row.sequence_items)
        ]

        target = item_map[str(row.target_item)]

        require(
            1 <= len(history) <= MAX_LEN,
            f"历史长度异常: {row.sample_id}",
        )

        require(
            target not in history,
            f"目标泄漏: {row.sample_id}",
        )

        k = int(row.last_bucket_size)

        (
            groups,
            tokens,
            pooled,
        ) = build_groups_from_size(
            len(history),
            k,
            MAX_LEN,
        )

        require(
            pooled == int(row.direct_pooling),
            f"Pooling 标签不一致: {row.sample_id}",
        )

        history = canonicalize(
            np.asarray(history, dtype=np.int64),
            k,
        )

        seq = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        seq[-len(history):] = history

        sequences.append(seq)
        groups_all.append(groups)
        tokens_all.append(tokens)
        targets.append(target)

    return (
        df,
        np.stack(sequences),
        np.stack(groups_all),
        np.asarray(tokens_all, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    )


val_df, val_seq, val_groups, val_tokens, val_targets = (
    load_eval(VAL_PATH, EXPECTED_VAL)
)

test_df, test_seq, test_groups, test_tokens, test_targets = (
    load_eval(TEST_PATH, EXPECTED_TEST)
)

print("Frozen validation:", len(val_targets))
print("Frozen test:", len(test_targets))
print("STRICT INPUT AUDIT PASS")


# ============================================================
# 8. 创建真正的 R1 模型
# ============================================================

model = R1Model(
    num_items=NUM_ITEMS,
    max_len=MAX_LEN,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    dropout=DROPOUT,
).to(device)


def training_loss(seq, groups, tokens, targets):
    scores = model.predict_batch_masked(
        seq,
        groups,
        tokens,
    )

    return F.cross_entropy(
        scores[:, 1:],
        targets - 1,
    )


@torch.inference_mode()
def evaluate(
    sequences,
    groups,
    tokens,
    targets,
    sample_df=None,
    save_predictions=False,
):
    model.eval()

    ranks_list = []
    top10_list = []

    for start in range(
        0,
        len(targets),
        EVAL_BATCH_SIZE,
    ):
        end = min(
            start + EVAL_BATCH_SIZE,
            len(targets),
        )

        seq_np = sequences[start:end]
        target_np = targets[start:end]

        seq = torch.from_numpy(
            seq_np
        ).long().to(device)

        group = torch.from_numpy(
            groups[start:end]
        ).long().to(device)

        token_count = torch.from_numpy(
            tokens[start:end]
        ).long().to(device)

        scores = model.predict_batch_masked(
            seq,
            group,
            token_count,
        )

        require(
            torch.isfinite(scores).all().item(),
            "评价分数 NaN/Inf",
        )

        scores[:, 0] = -1e9

        for b in range(len(seq_np)):
            seen = np.unique(seq_np[b])
            seen = seen[
                (seen != 0)
                & (seen != target_np[b])
            ]

            if len(seen):
                scores[
                    b,
                    torch.as_tensor(
                        seen,
                        dtype=torch.long,
                        device=device,
                    )
                ] = -1e9

        target_tensor = torch.as_tensor(
            target_np,
            dtype=torch.long,
            device=device,
        )

        row_idx = torch.arange(
            len(target_np),
            device=device,
        )

        target_scores = scores[
            row_idx,
            target_tensor,
        ]

        ranks = (
            (
                scores
                > target_scores.unsqueeze(1)
            ).sum(dim=1)
            + 1
        )

        ranks_list.append(
            ranks.cpu().numpy()
        )

        if save_predictions:
            top10 = torch.topk(
                scores,
                k=10,
                dim=1,
            ).indices

            top10_list.append(
                top10.cpu().numpy()
            )

    ranks = np.concatenate(ranks_list)

    result = {
        "samples": int(len(ranks)),
        "ndcg10": float(
            np.where(
                ranks <= 10,
                1.0 / np.log2(ranks + 1),
                0.0,
            ).mean()
        ),
        "hr10": float((ranks <= 10).mean()),
        "hr20": float((ranks <= 20).mean()),
        "mean_rank": float(ranks.mean()),
    }

    if not save_predictions:
        return result

    require(
        sample_df is not None,
        "预测输出缺少 sample_df",
    )

    top10 = np.concatenate(top10_list)

    pred = sample_df[
        ["sample_id", "user_id", "target_item"]
    ].copy()

    pred["target_rank"] = ranks

    pred["top10_item_idx"] = [
        json.dumps(x.tolist())
        for x in top10
    ]

    pred["top10_raw_item_id"] = [
        json.dumps([
            idx_to_raw[int(i)] for i in row
        ])
        for row in top10
    ]

    return result, pred


# ============================================================
# 9. Smoke A：训练梯度
# ============================================================

smoke_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

seq, groups, tokens, targets = next(
    iter(smoke_loader)
)

seq = seq.to(device)
groups = groups.to(device)
tokens = tokens.to(device)
targets = targets.to(device)

model.train()
model.zero_grad(set_to_none=True)

loss = training_loss(
    seq, groups, tokens, targets
)

require(
    torch.isfinite(loss).item(),
    "Smoke Loss 异常",
)

loss.backward()
_ALL_GRADIENTS_FINITE = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
require(_ALL_GRADIENTS_FINITE,
        "Smoke backward produced a non-finite gradient")

grad_count = sum(
    p.grad is not None
    and torch.isfinite(p.grad).all().item()
    and p.grad.abs().sum().item() > 0
    for p in model.parameters()
)

require(
    grad_count > 0,
    "模型没有有效梯度",
)

model.zero_grad(set_to_none=True)

print()
print("===== SMOKE A: TRAINING =====")
print("Loss:", float(loss.item()))
print("Nonzero gradient parameters:", grad_count)
print("SMOKE A PASS")


# ============================================================
# 10. Smoke B：新旧 R1 Forward 数值一致性
# ============================================================

model.eval()
max_diff = 0.0

with torch.inference_mode():
    sample_count = min(32, len(val_targets))

    seq32 = torch.from_numpy(
        val_seq[:sample_count]
    ).long().to(device)

    groups32 = torch.from_numpy(
        val_groups[:sample_count]
    ).long().to(device)

    tokens32 = torch.from_numpy(
        val_tokens[:sample_count]
    ).long().to(device)

    new_scores = model.predict_batch_masked(
        seq32,
        groups32,
        tokens32,
    )

    old_scores = torch.empty_like(new_scores)

    for token_count in torch.unique(tokens32):
        n = int(token_count.item())
        index = torch.where(tokens32 == n)[0]

        old_scores[index] = model.predict_group_unpadded(
            seq32[index],
            groups32[index],
            n,
        )

    max_diff = (
        new_scores - old_scores
    ).abs().max().item()

    require(
        torch.allclose(
            new_scores,
            old_scores,
            atol=1e-4,
            rtol=1e-5,
        ),
        f"Masked Forward 与旧 R1 不一致: {max_diff}",
    )

print()
print("===== SMOKE B: FORWARD PARITY =====")
print("Max absolute score difference:", max_diff)
print("SMOKE B PASS")


# ============================================================
# 11. Smoke C：最后桶排列不变性
# ============================================================

pool_candidates = val_df.index[
    (val_df["last_bucket_size"].astype(int) >= 2)
].tolist()

require(
    len(pool_candidates) > 0,
    "找不到可聚合评价样本",
)

tested = 0

with torch.inference_mode():
    rng = np.random.RandomState(2026)

    for i in pool_candidates:
        row = val_df.iloc[i]
        raw = [
            item_map[str(x)]
            for x in json.loads(row.sequence_items)
        ]

        k = int(row.last_bucket_size)

        if len(set(raw[-k:])) < 2:
            continue

        (
            gp,
            nt,
            pooled,
        ) = build_groups_from_size(
            len(raw),
            k,
            MAX_LEN,
        )

        require(
            pooled == 1,
            "可聚合样本标记异常",
        )

        baseline = None

        for attempt in range(6):
            items = list(raw)

            if attempt > 0:
                perm = rng.permutation(k)
                suffix = items[-k:]
                items[-k:] = [
                    suffix[j]
                    for j in perm
                ]

            items = canonicalize(
                np.asarray(items, dtype=np.int64),
                k,
            )

            padded = np.zeros(
                MAX_LEN,
                dtype=np.int64,
            )

            padded[-len(items):] = items

            seq_tensor = torch.from_numpy(
                padded
            ).long().unsqueeze(0).to(device)

            group_tensor = torch.from_numpy(
                gp
            ).long().unsqueeze(0).to(device)

            token_tensor = torch.tensor(
                [nt],
                dtype=torch.long,
                device=device,
            )

            scores = model.predict_batch_masked(
                seq_tensor,
                group_tensor,
                token_tensor,
            )

            if baseline is None:
                baseline = scores
            else:
                require(
                    torch.equal(baseline, scores),
                    "R1 桶内排列不变性失败",
                )

        tested += 1

        if tested == 16:
            break

require(
    tested > 0,
    "无有效排列不变性检查样本",
)

print()
print("===== SMOKE C: INVARIANCE =====")
print("Distinct eligible samples tested:", tested)
print("Randomized arrangements per sample: 6")
print("Exact logits equality: PASS")
print("SMOKE C PASS")


# ============================================================
# 12. Smoke 严格验证推理
# ============================================================

smoke_val = evaluate(
    val_seq[:32],
    val_groups[:32],
    val_tokens[:32],
    val_targets[:32],
)

print()
print("Strict validation smoke:", smoke_val)


# Complete smoke-only verification, with no formal result or weight output.
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
with _REPORT_FILE.open('x', encoding='utf-8', newline='\n') as _handle:
    _handle.write(json.dumps(_SUMMARY, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
print('SMOKE_SUMMARY=' + str(_REPORT_FILE))
print('SMOKE_PASS: no optimizer update, formal training, saved weights or test inference')
raise SystemExit(0)
