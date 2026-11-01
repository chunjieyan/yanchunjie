
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

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "step38_kais_retrain"

DATASET = "MovieLens-1M"
ORDER = "file_order"
SEED = 2026

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

EXPECTED_ITEMS = 3416
EXPECTED_VAL = 752
EXPECTED_TEST = 945
EXPECTED_POSITIONS = 794289

CONTROL_SCRIPT = (
    BASE / "scripts/"
    "38_1_train_ml1m_sasrec_bs128.py"
)

R1_SOURCE = (
    ROOT / "audit/"
    "33_2_train_movielens_r1_seed42.py"
)

CONTROL_DIR = (
    BASE / "outputs/"
    "ml1m_sasrec_control_ce_bs128_lr1em3_max200_seed42"
)

MAP_PATH = CONTROL_DIR / "item_id_map.csv"

OUTPUT_DIR = (
    BASE / "outputs/"
    "ml1m_r1_meanpool_ce_bs128_lr1em3_max200_seed2026"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["smoke", "train"],
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
print("STEP38.20 ML1M R1 MEAN POOLING")
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

control_tree = ast.parse(
    CONTROL_SCRIPT.read_text(encoding="utf-8")
)


def read_path_assignment(name):
    found = []

    for node in control_tree.body:
        if not isinstance(node, ast.Assign):
            continue

        if any(
            isinstance(target, ast.Name)
            and target.id == name
            for target in node.targets
        ):
            found.append(node.value)

    require(
        len(found) == 1,
        f"Control 路径定义数量异常: {name}",
    )

    env = {
        "ROOT": ROOT,
        "BASE": BASE,
        "DATASET": DATASET,
        "ORDER": ORDER,
        "Path": Path,
    }

    try:
        value = eval(
            compile(
                ast.Expression(found[0]),
                str(CONTROL_SCRIPT),
                "eval",
            ),
            {"__builtins__": {}},
            env,
        )

    except Exception as exc:
        raise RuntimeError(
            f"无法解析原始 {name}，禁止猜测路径: {exc}"
        )

    return Path(value)


TRAIN_PATH = read_path_assignment("TRAIN_PATH")
VAL_PATH = read_path_assignment("VAL_PATH")
TEST_PATH = read_path_assignment("TEST_PATH")

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

require(
    {"user_id", "item_id"}.issubset(train_df.columns),
    "训练字段不足",
)

train_df["_source_order"] = np.arange(
    len(train_df)
)

if "time_bucket" in train_df.columns:
    BUCKET_COLUMN = "time_bucket"
elif "timestamp" in train_df.columns:
    BUCKET_COLUMN = "timestamp"
else:
    raise RuntimeError(
        "训练数据没有 time_bucket/timestamp"
    )

require(
    not train_df[
        ["user_id", "item_id", BUCKET_COLUMN]
    ].isna().any().any(),
    "训练数据关键字段存在空值",
)

train_df = train_df.sort_values(
    ["user_id", BUCKET_COLUMN, "_source_order"],
    kind="stable",
)

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

if args.mode == "smoke":
    print()
    print("=" * 72)
    print("STEP38.20 ML1M R1 SMOKE PASS")
    print("No formal training started.")
    print("=" * 72)
    raise SystemExit(0)


# ============================================================
# 13. 正式训练
# ============================================================

# Smoke 不更新参数；这里重置训练随机状态。
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

mapping_df.to_csv(
    OUTPUT_DIR / "item_id_map.csv",
    index=False,
)

config = {
    "dataset": DATASET,
    "model": "RecentBucketSetR1SASRec",
    "method": "RecentBucketSet_R1_V1_mean_pooling",
    "seed": SEED,
    "training_batch_size": BATCH_SIZE,
    "evaluation_batch_size": EVAL_BATCH_SIZE,
    "optimizer": "Adam",
    "learning_rate": LR,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "max_len": MAX_LEN,
    "hidden_size": HIDDEN_SIZE,
    "num_heads": NUM_HEADS,
    "num_blocks": NUM_BLOCKS,
    "dropout": DROPOUT,
    "num_items": NUM_ITEMS,
    "legal_training_positions": LEGAL_POSITIONS,
    "sampling": (
        "all_legal_next_item_positions_"
        "reshuffled_every_epoch"
    ),
    "loss": "full_item_cross_entropy",
    "pooling": "last_history_bucket_only_mean",
    "canonicalization": "sort_last_bucket_by_item_idx",
    "train_path": str(TRAIN_PATH),
    "validation_path": str(VAL_PATH),
    "test_path": str(TEST_PATH),
}

(OUTPUT_DIR / "config.json").write_text(
    json.dumps(config, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)

best_ndcg = -1.0
best_epoch = -1
bad_epochs = 0
training_log = []

checkpoint_path = (
    OUTPUT_DIR / "r1_meanpool_best.pt"
)

print()
print("FORMAL R1 TRAINING STARTED", flush=True)

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    t0 = time.time()

    generator = torch.Generator()
    generator.manual_seed(SEED + epoch)

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

    total_loss = 0.0
    total_n = 0

    for seq, groups, tokens, targets in loader:
        seq = seq.to(device, non_blocking=True)
        groups = groups.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        loss = training_loss(
            seq,
            groups,
            tokens,
            targets,
        )

        require(
            torch.isfinite(loss).item(),
            f"Epoch {epoch}: Loss NaN/Inf",
        )

        loss.backward()
        optimizer.step()

        n = len(targets)

        total_loss += loss.item() * n
        total_n += n

    require(
        total_n == LEGAL_POSITIONS,
        "本轮训练位置覆盖不完整",
    )

    val_metrics = evaluate(
        val_seq,
        val_groups,
        val_tokens,
        val_targets,
    )

    current = val_metrics["ndcg10"]
    improved = current > best_ndcg + 1e-12

    if improved:
        best_ndcg = current
        best_epoch = epoch
        bad_epochs = 0

        torch.save(
            {
                "model_name": "RecentBucketSetR1SASRec",
                "method": "RecentBucketSet_R1_V1_mean_pooling",
                "dataset": DATASET,
                "seed": SEED,
                "epoch": epoch,
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

    seconds = time.time() - t0

    record = {
        "epoch": epoch,
        "train_loss": total_loss / total_n,
        "val_ndcg10": current,
        "val_hr10": val_metrics["hr10"],
        "val_mean_rank": val_metrics["mean_rank"],
        "best_epoch": best_epoch,
        "best_ndcg10": best_ndcg,
        "bad_epochs": bad_epochs,
        "seconds": seconds,
    }

    training_log.append(record)

    pd.DataFrame(training_log).to_csv(
        OUTPUT_DIR / "training_log.csv",
        index=False,
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Loss={record['train_loss']:.6f} | "
        f"Val NDCG@10={current:.6f} | "
        f"Best={best_ndcg:.6f} "
        f"(Epoch {best_epoch}) | "
        f"Bad={bad_epochs}/{PATIENCE} | "
        f"{seconds:.1f}s",
        flush=True,
    )

    if bad_epochs >= PATIENCE:
        print(
            f"EARLY STOPPING AT EPOCH {epoch}",
            flush=True,
        )
        break


# ============================================================
# 14. 恢复最佳模型并评价
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

final_val = evaluate(
    val_seq,
    val_groups,
    val_tokens,
    val_targets,
)

final_test, predictions = evaluate(
    test_seq,
    test_groups,
    test_tokens,
    test_targets,
    sample_df=test_df,
    save_predictions=True,
)

require(
    abs(final_val["ndcg10"] - best_ndcg) < 1e-8,
    "最佳 checkpoint 恢复失败",
)

predictions.to_csv(
    OUTPUT_DIR / "test_per_sample_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

metrics = {
    "dataset": DATASET,
    "model": "RecentBucketSetR1SASRec",
    "method": "RecentBucketSet_R1_V1_mean_pooling",
    "seed": SEED,
    "best_epoch": best_epoch,
    "validation": final_val,
    "test": final_test,
}

(OUTPUT_DIR / "metrics.json").write_text(
    json.dumps(metrics, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("=" * 72)
print("STEP38.20 ML1M R1 TRAINING PASS")
print("Best epoch:", best_epoch)
print("Val:", final_val)
print("Test:", final_test)
print("Output:", OUTPUT_DIR)
print("=" * 72)
