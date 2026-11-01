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
# Step38.13 - Amazon Toys SASRec Control
# Unified BS128 / Full legal positions / Strict evaluation
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "step38_kais_retrain"

DATASET = "Amazon_Toys"
ORDER = "file_order"
SEED = 42

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

EXPECTED_VAL = 1939
EXPECTED_TEST = 2547
EXPECTED_ITEMS = 11924

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

MODEL_SCRIPT = (
    BASE / "scripts/38_1_train_ml1m_sasrec_bs128.py"
)

OUTPUT_DIR = (
    BASE / "outputs"
    / "toys_sasrec_control_ce_bs128_lr1em3_max200_seed42"
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

print("=" * 72)
print("STEP38.13 AMAZON TOYS SASREC CONTROL")
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
        f"Toys 物品数异常: {NUM_ITEMS}, "
        f"expected={EXPECTED_ITEMS}"
    )


# ============================================================
# 4. 读取训练交互，按冻结 file_order 排序
# ============================================================

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype={
        "user_id": str,
        "item_id": str,
    },
)

required_cols = {
    "user_id",
    "item_id",
    "raw_line",
}

if not required_cols.issubset(train_df.columns):
    raise RuntimeError(
        f"训练集缺少列: "
        f"{required_cols - set(train_df.columns)}"
    )

if train_df[
    ["user_id", "item_id", "raw_line"]
].isna().any().any():
    raise RuntimeError("训练集关键字段有缺失值")

train_df["raw_line"] = pd.to_numeric(
    train_df["raw_line"],
    errors="raise",
)

train_df = train_df.sort_values(
    "raw_line",
    kind="stable",
)

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
    print("STEP38.13 TOYS SMOKE PASS")
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

    if bad_epochs >= PATIENCE:
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
print("STEP38.13 TOYS TRAINING PASS")
print("Best epoch:", best_epoch)
print("Val:", final_val)
print("Test:", final_test)
print("Output:", OUTPUT_DIR)
print("=" * 72)
