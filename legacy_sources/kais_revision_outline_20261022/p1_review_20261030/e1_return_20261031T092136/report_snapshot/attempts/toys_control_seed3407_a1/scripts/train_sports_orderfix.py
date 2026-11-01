#!/usr/bin/env python3
# Generated private order correction. Preparation never runs this script.
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat

def plain_path(path):
    """Reject symbolic links and Windows reparse points in every component."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected an absolute path without '..': " + str(path))
    for part in reversed((path,) + tuple(path.parents)):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError("Symbolic link/reparse point rejected: " + str(part))
    return path


def checked_file(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or any(p in ("", ".", "..") for p in relative.parts):
        raise ValueError("Unsafe relative input path: " + str(relative))
    if "\\" in str(relative) or ":" in str(relative):
        raise ValueError("Unsafe relative input path: " + str(relative))
    root = plain_path(root)
    path = plain_path(root.joinpath(*relative.parts))
    path.resolve(strict=False).relative_to(root.resolve(strict=False))
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Expected a regular file: " + str(path))
    return path


def checked_sha256(root, relative, expected):
    path = checked_file(root, relative)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError("Input changed while opening: " + relative)
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = checked_file(root, relative).stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError("Input changed during hashing: " + relative)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError("SHA256 mismatch; no training is allowed: " + relative)
    return actual


parser = argparse.ArgumentParser(description='Isolated KAIS order correction')
parser.add_argument('--mode', choices=['smoke', 'train'], default='smoke')
parser.add_argument('--seed', type=int, choices=[42, 2026, 3407], default=42)
args = parser.parse_args()
_ORDERFIX_ROOT = plain_path(Path('/root/autodl-tmp/time_uncertainty_rec'))
_ORDERFIX_STAGE = plain_path(Path('/root/autodl-tmp/time_uncertainty_rec/kais_e1_min50_v1_20261030T045527Z_3fca1af177/attempts/toys_control_seed3407_a1'))
if Path(__file__).resolve().parent != _ORDERFIX_STAGE / 'scripts':
    raise RuntimeError('Run the original generated trainer in its original stage')
_INPUT_SHA256 = {'processed/diagnostic_a/Amazon_Sports/file_order.csv': 'e4619ab4f9e74f98b817a6a485b8453c1519bf380bc248fd1d2238fb8bed3a9c', 'processed/diagnostic_a/Amazon_Toys/file_order.csv': '8129322bddc716bfca6877a8c0e7cbf2b7ad3baf236fccc58ee95af1d08cf541', 'processed/diagnostic_a_splits/Amazon_Sports/file_order/train.csv': '8b78398cf350881125d04a15cd0b6181b17d1772d35f428ed61d3e9220a16539', 'processed/diagnostic_a_splits/Amazon_Toys/file_order/train.csv': '26e4e892b1e520fdc213bddad4af3f314e865f53a1c8cbcf1a4e0f60a83ff8c0', 'processed/strict_diagnostic_a/Amazon_Sports/file_order/test.csv': '5f2f31713ee7f6c92bb6fde915500d51f1a49328f3f1fc94f146530ad54e0e89', 'processed/strict_diagnostic_a/Amazon_Sports/file_order/val.csv': '8ffe18f11957ba4229aa42d4e1b5d2ffdfbf3ef2adb4ecd0ff3b349469cbe7fb', 'processed/strict_diagnostic_a/Amazon_Toys/file_order/test.csv': '3e287ca6be6c556d5abfd691666abdc4fde1eca10327d6b8329a8aa55d0df445', 'processed/strict_diagnostic_a/Amazon_Toys/file_order/val.csv': '5b2187611068dbf7f384e55489d3704710387ef42bf86fcf028836ae26e8f14d'}
for _relative, _expected in _INPUT_SHA256.items():
    checked_sha256(_ORDERFIX_ROOT, _relative, _expected)
_FROZEN_MODEL_SHA256 = '74758f49cb95ee74cb68c60d44c7f5beacdcaf2bc8173b26bd5862434b786dc9'
checked_sha256(_ORDERFIX_STAGE, 'scripts/model_definition.py', _FROZEN_MODEL_SHA256)
print('INTEGRITY PASS: eight fixed CSV inputs and frozen model definition')

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
# Step38.12 - Amazon Sports SASRec Control
# Unified BS128 / Full legal positions / Strict evaluation
# ============================================================

ROOT = _ORDERFIX_ROOT
BASE = _ORDERFIX_STAGE

DATASET = "Amazon_Sports"
ORDER = "file_order"
SEED = args.seed

BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256

LEARNING_RATE = 0.001
MAX_EPOCHS = 200
PATIENCE = 15

MAX_LEN = 50
HIDDEN_SIZE = 64
NUM_HEADS = 2
NUM_BLOCKS = 2
DROPOUT = 0.2

EXPECTED_VAL = 4082
EXPECTED_TEST = 4659
EXPECTED_ITEMS = 18357

TRAIN_PATH = (
    ROOT / "processed/diagnostic_a_splits"
    / DATASET / ORDER / "train.csv"
)

VAL_PATH = (
    ROOT / "processed/strict_diagnostic_a"
    / DATASET / ORDER / "val.csv"
)

TEST_PATH = (
    ROOT / "processed/strict_diagnostic_a"
    / DATASET / ORDER / "test.csv"
)

ITEM_SOURCE_PATH = (
    ROOT / "processed/diagnostic_a"
    / DATASET / "file_order.csv"
)

MODEL_SCRIPT = _ORDERFIX_STAGE / "scripts/model_definition.py"

OUTPUT_DIR = (BASE / "outputs" / f"{DATASET.lower()}_control_orderfix_seed{SEED}")

# Arguments parsed before numerical imports and integrity checks.

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 72)
print("STEP38.12 AMAZON SPORTS SASREC CONTROL")
print("=" * 72)
print("Mode:", args.mode)
print("Device:", device)
print("Seed:", SEED)
print("Training BS:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Max epochs:", MAX_EPOCHS)
print("Patience:", PATIENCE)


# ============================================================
# 1. 输入文件检查
# ============================================================

for path in [
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    ITEM_SOURCE_PATH,
    MODEL_SCRIPT,
]:
    if not path.is_file():
        raise FileNotFoundError(path)

if OUTPUT_DIR.exists():
    raise RuntimeError(
        f"输出目录已存在，禁止覆盖: {OUTPUT_DIR}"
    )

print("INPUT FILE AUDIT PASS")


# ============================================================
# 2. 复用 Step38.1 已验证的模型类
# 不执行 ML1M 训练脚本的顶层训练代码
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
    raise RuntimeError(
        "SASRecControl 类提取失败"
    )

namespace = {
    "__name__": "step38_sports",
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
            type_ignores=[],
        ),
        str(MODEL_SCRIPT),
        "exec",
    ),
    namespace,
)

SASRecControl = namespace["SASRecControl"]

print("Step38 SASRecControl extracted: PASS")


# ============================================================
# 3. 统一物品映射：使用冻结 file_order 全量物品源
# ============================================================

item_source = pd.read_csv(
    ITEM_SOURCE_PATH,
    usecols=["item_id"],
    dtype={"item_id": str},
)

raw_items = (
    item_source["item_id"]
    .dropna()
    .drop_duplicates()
    .tolist()
)

item_map = {
    raw: idx
    for idx, raw in enumerate(raw_items, start=1)
}

idx_to_raw = {
    idx: raw
    for raw, idx in item_map.items()
}

NUM_ITEMS = len(item_map)

print("NUM_ITEMS:", NUM_ITEMS)

if NUM_ITEMS != EXPECTED_ITEMS:
    raise RuntimeError(
        f"Sports 物品数异常: {NUM_ITEMS}, "
        f"expected={EXPECTED_ITEMS}"
    )


# ============================================================
# 4. 读取原始冻结交互，按用户、时间桶、原始行号构造序列
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

unknown_train = (
    set(train_df["item_id"]) - set(item_map)
)

if unknown_train:
    raise RuntimeError(
        f"训练集有 {len(unknown_train)} 个物品不在映射中"
    )

user_sequences = []

for _, group in train_df.groupby(
    "user_id",
    sort=False,
):
    seq = [
        item_map[x]
        for x in group["item_id"]
    ]

    if len(seq) >= 2:
        user_sequences.append(
            np.asarray(seq, dtype=np.int64)
        )

LEGAL_POSITIONS = sum(
    len(seq) - 1
    for seq in user_sequences
)

if LEGAL_POSITIONS <= 0:
    raise RuntimeError("无合法训练位置")

print("Training rows:", len(train_df))
print("Trainable users:", len(user_sequences))
print("All legal positions:", LEGAL_POSITIONS)
print(
    "Batches per epoch:",
    math.ceil(LEGAL_POSITIONS / BATCH_SIZE)
)


# ============================================================
# 5. 每一个合法 next-item 位置都构造一个训练样本
#
# position = 1,...,len(user_sequence)-1
# history = 前 position 个交互，最多取最后50个
# target = 第 position 个交互
#
# 每轮 RandomSampler 使用不同 seed 完整打乱
# ============================================================

class AllPositionsDataset(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

        self.positions = [
            (u, p)
            for u, seq in enumerate(sequences)
            for p in range(1, len(seq))
        ]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        u, p = self.positions[idx]

        full = self.sequences[u]

        history = full[
            max(0, p - MAX_LEN):p
        ]

        seq = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        seq[-len(history):] = history

        return (
            torch.from_numpy(seq),
            torch.tensor(
                len(history),
                dtype=torch.long,
            ),
            torch.tensor(
                int(full[p]),
                dtype=torch.long,
            ),
        )


train_dataset = AllPositionsDataset(
    user_sequences
)

assert len(train_dataset) == LEGAL_POSITIONS


# ============================================================
# 6. 读取严格验证/测试样本
# ============================================================

def load_strict_samples(path, expected_n):
    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_items": str,
        },
    )

    if len(df) != expected_n:
        raise RuntimeError(
            f"样本数异常: {path}, "
            f"actual={len(df)}, expected={expected_n}"
        )

    if not df["sample_id"].is_unique:
        raise RuntimeError(
            f"重复 sample_id: {path}"
        )

    sequences = []
    lengths = []
    targets = []

    for row in df.itertuples(index=False):
        raw_history = json.loads(
            row.sequence_items
        )

        if not (
            1 <= len(raw_history) <= MAX_LEN
        ):
            raise RuntimeError(
                f"严格历史长度异常: {row.sample_id}"
            )

        try:
            mapped = [
                item_map[str(x)]
                for x in raw_history
            ]

            target = item_map[
                str(row.target_item)
            ]

        except KeyError as exc:
            raise RuntimeError(
                f"评价物品不在映射中: "
                f"{row.sample_id}, {exc}"
            )

        if target in mapped:
            raise RuntimeError(
                f"目标物品泄漏进历史: {row.sample_id}"
            )

        seq = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        seq[-len(mapped):] = mapped

        sequences.append(seq)
        lengths.append(len(mapped))
        targets.append(target)

    return (
        df,
        np.stack(sequences),
        np.asarray(lengths, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    )


val_df, val_seq, val_len, val_target = (
    load_strict_samples(
        VAL_PATH,
        EXPECTED_VAL,
    )
)

test_df, test_seq, test_len, test_target = (
    load_strict_samples(
        TEST_PATH,
        EXPECTED_TEST,
    )
)

print("Strict validation:", len(val_target))
print("Strict test:", len(test_target))
print("STRICT DATA AUDIT PASS")


# ============================================================
# 7. 创建模型
# ============================================================

model = SASRecControl(
    num_items=NUM_ITEMS,
    max_len=MAX_LEN,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    dropout=DROPOUT,
).to(device)


# ============================================================
# 8. 训练损失：next-item 全物品交叉熵
# 不使用旧版固定 cutoff + BCE 负采样训练
# ============================================================

def loss_for_batch(seq, lengths, targets):
    scores = model.predict_batch_masked(
        seq,
        lengths,
    )

    if scores.shape != (
        len(targets),
        NUM_ITEMS + 1,
    ):
        raise RuntimeError(
            f"训练 logits 形状异常: {scores.shape}"
        )

    # ID=0 是 padding，不参与 softmax 候选。
    loss = F.cross_entropy(
        scores[:, 1:],
        targets - 1,
    )

    return loss


# ============================================================
# 9. 严格全物品评价 + 已见物品屏蔽
# ============================================================

@torch.inference_mode()
def evaluate(
    sequences,
    lengths,
    targets,
    sample_df=None,
    return_predictions=False,
):
    model.eval()

    all_ranks = []
    all_top10 = []

    total = len(targets)

    for start in range(
        0,
        total,
        EVAL_BATCH_SIZE,
    ):
        end = min(
            start + EVAL_BATCH_SIZE,
            total,
        )

        seq_np = sequences[start:end]
        target_np = targets[start:end]

        seq_tensor = torch.from_numpy(
            seq_np
        ).long().to(device)

        len_tensor = torch.from_numpy(
            lengths[start:end]
        ).long().to(device)

        scores = model.predict_batch_masked(
            seq_tensor,
            len_tensor,
        )

        if scores.shape != (
            len(target_np),
            NUM_ITEMS + 1,
        ):
            raise RuntimeError(
                "评价 logits 维度异常"
            )

        if not torch.isfinite(scores).all():
            raise RuntimeError(
                "评价 logits 出现 NaN/Inf"
            )

        scores = scores.clone()
        scores[:, 0] = -1e9

        # 统一的已见物品屏蔽
        for b in range(len(seq_np)):
            seen = np.unique(
                seq_np[b]
            )

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
                    ),
                ] = -1e9

        target_tensor = torch.as_tensor(
            target_np,
            dtype=torch.long,
            device=device,
        )

        rows = torch.arange(
            len(target_np),
            device=device,
        )

        target_scores = scores[
            rows,
            target_tensor,
        ]

        ranks = (
            (
                scores
                > target_scores[:, None]
            ).sum(dim=1)
            + 1
        )

        top10 = torch.topk(
            scores,
            k=10,
            dim=1,
        ).indices

        all_ranks.append(
            ranks.cpu().numpy()
        )

        if return_predictions:
            all_top10.append(
                top10.cpu().numpy()
            )

    ranks = np.concatenate(all_ranks)

    ndcg = np.where(
        ranks <= 10,
        1.0 / np.log2(ranks + 1),
        0.0,
    )

    metrics = {
        "samples": total,
        "ndcg10": float(ndcg.mean()),
        "hr10": float(
            (ranks <= 10).mean()
        ),
        "hr20": float(
            (ranks <= 20).mean()
        ),
        "mean_rank": float(
            ranks.mean()
        ),
    }

    if not return_predictions:
        return metrics

    if sample_df is None:
        raise RuntimeError(
            "保存预测时必须传入 sample_df"
        )

    top10 = np.concatenate(all_top10)

    pred = sample_df[
        ["sample_id", "user_id", "target_item"]
    ].copy()

    pred["target_rank"] = ranks

    pred["top10_item_idx"] = [
        json.dumps(row.tolist())
        for row in top10
    ]

    pred["top10_raw_item_id"] = [
        json.dumps([
            idx_to_raw[int(x)]
            for x in row
        ])
        for row in top10
    ]

    return metrics, pred


# ============================================================
# 10. Smoke Test：梯度 + 严格评价
# 默认只执行这里，不启动正式训练
# ============================================================

model.train()

smoke_loader = DataLoader(
    train_dataset,
    batch_size=min(
        8,
        len(train_dataset),
    ),
    shuffle=False,
    num_workers=0,
)

seq, lengths, targets = next(
    iter(smoke_loader)
)

seq = seq.to(device)
lengths = lengths.to(device)
targets = targets.to(device)

model.zero_grad(set_to_none=True)

smoke_loss = loss_for_batch(
    seq,
    lengths,
    targets,
)

if not torch.isfinite(smoke_loss):
    raise RuntimeError(
        "Smoke Loss NaN/Inf"
    )

if not smoke_loss.requires_grad:
    raise RuntimeError(
        "训练前向没有梯度，禁止启动正式训练"
    )

smoke_loss.backward()

gradient_count = sum(
    p.grad is not None
    and torch.isfinite(p.grad).all().item()
    and p.grad.abs().sum().item() > 0
    for p in model.parameters()
)

if gradient_count == 0:
    raise RuntimeError(
        "模型没有有效非零梯度"
    )

model.zero_grad(set_to_none=True)

smoke_val = evaluate(
    val_seq[:32],
    val_len[:32],
    val_target[:32],
)

print()
print("===== SMOKE TEST =====")
print("Training loss:", float(smoke_loss.item()))
print("Parameters with nonzero gradient:", gradient_count)
print("Strict validation smoke:", smoke_val)
print("SMOKE TEST PASS")

if args.mode == "smoke":
    print()
    print("STEP38.12 SPORTS SMOKE PASS")
    print("No training started.")
    raise SystemExit(0)


# ============================================================
# 11. 正式训练
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

mapping_df = pd.DataFrame({
    "raw_item_id": raw_items,
    "item_idx": np.arange(
        1,
        NUM_ITEMS + 1,
    ),
})

mapping_df.to_csv(
    OUTPUT_DIR / "item_id_map.csv",
    index=False,
)

config = {
    "orderfix": {
        "version": "kais_orderfix_v1",
        "sequence_sort_keys": ["user_id", "time_bucket", "raw_line"],
        "sort_kind": "stable",
        "original_input_files_unchanged": True,
        "input_sha256": _INPUT_SHA256,
        "frozen_model_sha256": _FROZEN_MODEL_SHA256,
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preparation_manifest_sha256": hashlib.sha256(
            (_ORDERFIX_STAGE / "manifests/orderfix_manifest.json").read_bytes()
        ).hexdigest(),
    },
    "dataset": DATASET,
    "model": "SASRecControl",
    "seed": SEED,
    "training_batch_size": BATCH_SIZE,
    "evaluation_batch_size": EVAL_BATCH_SIZE,
    "optimizer": "Adam",
    "learning_rate": LEARNING_RATE,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "early_stop_metric": "validation_ndcg10",
    "max_len": MAX_LEN,
    "hidden_size": HIDDEN_SIZE,
    "num_heads": NUM_HEADS,
    "num_blocks": NUM_BLOCKS,
    "dropout": DROPOUT,
    "num_items": NUM_ITEMS,
    "trainable_users": len(user_sequences),
    "legal_training_positions": LEGAL_POSITIONS,
    "sampling": (
        "all_legal_next_item_positions_"
        "reshuffled_every_epoch"
    ),
    "loss": "full_item_cross_entropy",
    "train_path": str(TRAIN_PATH),
    "validation_path": str(VAL_PATH),
    "test_path": str(TEST_PATH),
    "item_source_path": str(ITEM_SOURCE_PATH),
}

config["e1_minimum_stopping_epochs"] = 50
(OUTPUT_DIR / "config.json").write_text(
    json.dumps(
        config,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

# Smoke Test 刚才只检查梯度，没有更新权重。
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_ndcg = -1.0
best_epoch = -1
bad_epochs = 0

training_log = []

checkpoint_path = (
    OUTPUT_DIR
    / "matched_control_sasrec_best.pt"
)

print()
print("=" * 72)
print("FORMAL TRAINING STARTED")
print("=" * 72)

for epoch in range(
    1,
    MAX_EPOCHS + 1,
):
    model.train()

    start_time = time.time()

    generator = torch.Generator()
    generator.manual_seed(
        SEED + epoch
    )

    sampler = RandomSampler(
        train_dataset,
        replacement=False,
        generator=generator,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    epoch_loss = 0.0
    total_examples = 0

    for seq, lengths, targets in loader:
        seq = seq.to(
            device,
            non_blocking=True,
        )
        lengths = lengths.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = loss_for_batch(
            seq,
            lengths,
            targets,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Epoch {epoch}: Loss NaN/Inf"
            )

        loss.backward()
        optimizer.step()

        n = len(targets)

        epoch_loss += (
            loss.item() * n
        )

        total_examples += n

    if total_examples != LEGAL_POSITIONS:
        raise RuntimeError(
            "本轮未覆盖全部合法训练位置"
        )

    val_metrics = evaluate(
        val_seq,
        val_len,
        val_target,
    )

    val_ndcg = val_metrics["ndcg10"]

    improved = (
        val_ndcg > best_ndcg + 1e-12
    )

    if improved:
        best_ndcg = val_ndcg
        best_epoch = epoch
        bad_epochs = 0

        torch.save(
            {
                "model_name": "SASRecControl",
                "dataset": DATASET,
                "epoch": epoch,
                "seed": SEED,
                "num_items": NUM_ITEMS,
                "max_len": MAX_LEN,
                "hidden_size": HIDDEN_SIZE,
                "num_heads": NUM_HEADS,
                "num_blocks": NUM_BLOCKS,
                "dropout": DROPOUT,
                "training_batch_size": BATCH_SIZE,
                "model_state_dict": model.state_dict(),
                "validation_metrics": val_metrics,
            },
            checkpoint_path,
        )

    else:
        bad_epochs += 1

    duration = time.time() - start_time

    record = {
        "epoch": epoch,
        "train_loss": (
            epoch_loss / total_examples
        ),
        "val_ndcg10": val_ndcg,
        "val_hr10": val_metrics["hr10"],
        "val_hr20": val_metrics["hr20"],
        "val_mean_rank": val_metrics["mean_rank"],
        "best_epoch": best_epoch,
        "best_val_ndcg10": best_ndcg,
        "bad_epochs": bad_epochs,
        "seconds": duration,
    }

    training_log.append(record)

    # 每轮保存日志，避免中断后没有记录。
    pd.DataFrame(
        training_log
    ).to_csv(
        OUTPUT_DIR / "training_log.csv",
        index=False,
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Loss={record['train_loss']:.6f} | "
        f"Val NDCG@10={val_ndcg:.6f} | "
        f"Best={best_ndcg:.6f} "
        f"(Epoch {best_epoch}) | "
        f"Bad={bad_epochs}/{PATIENCE} | "
        f"{duration:.1f}s",
        flush=True,
    )

    if epoch >= 50 and bad_epochs >= PATIENCE:
        print(
            f"EARLY STOPPING AT EPOCH {epoch}",
            flush=True,
        )
        break


# ============================================================
# 12. 恢复最优 Epoch 模型
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.eval()

print()
print("BEST EPOCH RESTORED:", best_epoch)


# ============================================================
# 13. 严格验证/测试
# ============================================================

final_val = evaluate(
    val_seq,
    val_len,
    val_target,
)

final_test, test_predictions = evaluate(
    test_seq,
    test_len,
    test_target,
    sample_df=test_df,
    return_predictions=True,
)

if abs(
    final_val["ndcg10"] - best_ndcg
) >= 1e-8:
    raise RuntimeError(
        "恢复模型后的验证指标不一致"
    )

test_predictions.to_csv(
    OUTPUT_DIR / "test_per_sample_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

metrics = {
    "dataset": DATASET,
    "model": "SASRecControl",
    "seed": SEED,
    "best_epoch": best_epoch,
    "early_stopped": bad_epochs >= PATIENCE,
    "validation": final_val,
    "test": final_test,
}

(OUTPUT_DIR / "metrics.json").write_text(
    json.dumps(
        metrics,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("=" * 72)
print("STEP38.12 SPORTS TRAINING PASS")
print("Best epoch:", best_epoch)
print("Val:", final_val)
print("Test:", final_test)
print("Output:", OUTPUT_DIR)
print("=" * 72)
