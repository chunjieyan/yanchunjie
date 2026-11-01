import ast
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/root/autodl-tmp/time_uncertainty_rec")
SCRIPTS = ROOT / "step38_kais_retrain/scripts"
DEVICE = torch.device("cuda")

SEED = 42
BS = 128
MAX_LEN = 50
WARMUP = 5
TEST_BATCHES = 100

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_model_class(filename):
    path = SCRIPTS / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))

    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "RecentBucketSetR1SASRec"
    )

    ns = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
    }

    exec(
        compile(
            ast.Module(body=[cls], type_ignores=[]),
            str(path),
            "exec",
        ),
        ns,
    )

    return ns["RecentBucketSetR1SASRec"]


# ==================================================
# 1. 读取真实训练数据，保持原始排序协议
# ==================================================

print("===== LOADING REAL TRAIN DATA =====", flush=True)

mapping_path = ROOT / (
    "runs/phase2_natural_timestamp/matched_control/"
    "sasrec/MovieLens-1M/file_order/seed42/item_id_map.csv"
)

mapping = pd.read_csv(
    mapping_path,
    dtype={"raw_item_id": str},
)

item_map = dict(
    zip(mapping["raw_item_id"], mapping["item_idx"])
)

train_path = ROOT / (
    "processed/diagnostic_a_splits/"
    "MovieLens-1M/file_order/train.csv"
)

df = pd.read_csv(
    train_path,
    dtype={"user_id": str, "item_id": str},
)

df["_source_order"] = np.arange(len(df))

df = df.sort_values(
    ["user_id", "time_bucket", "_source_order"],
    kind="mergesort",
)

mapped = df["item_id"].map(item_map)

if mapped.isna().any():
    raise RuntimeError("存在未映射的训练物品")

df["item_idx"] = mapped.astype(np.int64)

sequences = []

for _, group in df.groupby("user_id", sort=False):
    items = group["item_idx"].to_numpy(dtype=np.int64)

    if len(items) >= 2:
        sequences.append(items)

counts = np.asarray(
    [len(seq) - 1 for seq in sequences],
    dtype=np.int64,
)

cumulative = np.cumsum(counts)
total_positions = int(cumulative[-1])

print("Train records:", len(df))
print("Eligible users:", len(sequences))
print("All legal positions:", total_positions)

if total_positions != 794289:
    raise RuntimeError(
        f"训练位置数量异常: {total_positions}"
    )


# ==================================================
# 2. 从全部合法位置生成相同的真实训练 Batch
# ==================================================

rng = np.random.default_rng(SEED)

num_samples = (WARMUP + TEST_BATCHES) * BS

selected = rng.permutation(total_positions)[:num_samples]

user_indices = np.searchsorted(
    cumulative,
    selected,
    side="right",
)

previous = np.concatenate((
    np.array([0], dtype=np.int64),
    cumulative[:-1],
))

end_indices = (
    selected - previous[user_indices] + 2
)

batches = []

prep_start = time.perf_counter()

for start in range(0, num_samples, BS):
    seq_batch = np.zeros(
        (BS, MAX_LEN), dtype=np.int64
    )

    group_batch = np.full(
        (BS, MAX_LEN), -1, dtype=np.int64
    )

    lengths = np.zeros(BS, dtype=np.int64)
    targets = np.zeros(BS, dtype=np.int64)

    for j in range(BS):
        k = start + j
        items = sequences[int(user_indices[k])]
        end_idx = int(end_indices[k])

        target_idx = end_idx - 1

        history = items[
            max(0, target_idx - MAX_LEN):target_idx
        ]

        n = len(history)

        seq_batch[j, -n:] = history
        group_batch[j, -n:] = np.arange(n)

        lengths[j] = n
        targets[j] = int(items[target_idx])

    batches.append((
        torch.from_numpy(seq_batch),
        torch.from_numpy(group_batch),
        torch.from_numpy(lengths),
        torch.from_numpy(targets),
    ))

prep_seconds = time.perf_counter() - prep_start

print("Prepared batches:", len(batches))
print(f"Batch preparation: {prep_seconds:.2f}s")


# ==================================================
# 3. 创建相同初始权重的两种模型
# ==================================================

Grouped = load_model_class(
    "38_0_ml1m_control_fast.py"
)

Masked = load_model_class(
    "38_0_ml1m_control_masked.py"
)

config = dict(
    num_items=len(item_map),
    max_len=MAX_LEN,
    hidden_size=64,
    num_heads=2,
    num_blocks=2,
    dropout=0.2,
)

grouped = Grouped(**config).to(DEVICE)
masked = Masked(**config).to(DEVICE)

masked.load_state_dict(grouped.state_dict())


# ==================================================
# 4. 训练步骤：BS128、Adam、LR0.001
# ==================================================

def train_step(model, optimizer, batch, method):
    seq, groups, lengths, targets = [
        tensor.to(DEVICE)
        for tensor in batch
    ]

    model.train()
    optimizer.zero_grad(set_to_none=True)

    if method == "grouped":
        loss_sum = None

        for n in torch.unique(lengths, sorted=True):
            ids = torch.where(lengths == n)[0]

            scores = model.predict_group_unpadded(
                seq[ids],
                groups[ids],
                int(n.item()),
            )

            scores = scores.clone()
            scores[:, 0] = -1e9

            part_loss = F.cross_entropy(
                scores,
                targets[ids],
                reduction="sum",
            )

            loss_sum = (
                part_loss
                if loss_sum is None
                else loss_sum + part_loss
            )

        loss = loss_sum / len(targets)

    else:
        scores = model.predict_batch_masked(
            seq,
            lengths,
        )

        scores = scores.clone()
        scores[:, 0] = -1e9

        loss = F.cross_entropy(scores, targets)

    if not torch.isfinite(loss):
        raise RuntimeError("Loss NaN/Inf")

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 5.0
    )

    optimizer.step()


def benchmark(model, method):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    for batch in batches[:WARMUP]:
        train_step(model, optimizer, batch, method)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    for batch in batches[WARMUP:]:
        train_step(model, optimizer, batch, method)

    torch.cuda.synchronize()

    seconds = time.perf_counter() - start
    memory_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    return seconds, memory_gb


# ==================================================
# 5. 运行测试
# ==================================================

print("\n===== REAL BATCH SPEED TEST =====", flush=True)
print("GPU:", torch.cuda.get_device_name(0))
print("Batch Size:", BS)
print("Optimizer: Adam")
print("Learning Rate: 0.001")
print("Dropout: 0.2")
print("Timed batches:", TEST_BATCHES, flush=True)

grouped_time, grouped_memory = benchmark(
    grouped, "grouped"
)

print(
    f"Grouped completed: {grouped_time:.2f}s",
    flush=True,
)

masked_time, masked_memory = benchmark(
    masked, "masked"
)

print("\n===== FINAL BENCHMARK =====")
print(f"Grouped total: {grouped_time:.3f}s")
print(f"Masked total:  {masked_time:.3f}s")
print(f"Grouped per batch: {grouped_time*10:.2f}ms")
print(f"Masked per batch:  {masked_time*10:.2f}ms")
print(f"Speedup: {grouped_time/masked_time:.2f}x")
print(f"Grouped peak GPU: {grouped_memory:.3f}GB")
print(f"Masked peak GPU:  {masked_memory:.3f}GB")
print("REAL BATCH BENCHMARK COMPLETE")
