from pathlib import Path
import os
import json
import math
import random
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ============================================================
# Step 31.2
# Amazon Beauty + SASRec + Recent Bucket-Set R=1
# Single-seed pilot (seed=42)
#
# 固定：
#   dataset = Amazon_Beauty
#   order   = file_order
#   seed    = 42
#
# 方法：
#   - 历史中除最后时间桶外，每个事件保留一个 token；
#   - 若最后时间桶包含 >=2 个事件，则将其 item embedding 做 mean pooling，
#     压缩为 1 个 token；
#   - 若最后时间桶只有 1 个事件，则退化为普通 SASRec 输入。
#
# 评价：
#   - 使用 Step 31.1 固定的严格 val/test 样本；
#   - full ranking；
#   - seen-item mask；
#   - 无 padding Transformer 推理；
#   - 压缩后的 token 使用 MAX_LEN=50 的右对齐绝对位置编码；
#   - 强制 NaN / Inf 检查。
#
# 重要：
#   本脚本用于 31.2 单种子可行性验证。
#   训练目标沿用旧 Recent Bucket-Set 原型的 target-level full-ranking CE，
#   而 Step 28.2 Control 使用 all-position BCE。
#   因此本步结果不能直接作为论文中的正式 Control-vs-R1 准确率差值；
#   后续主表必须使用匹配训练目标的 Control 与 R1。
# ============================================================


# ============================================================
# 0. 固定配置
# ============================================================

ROOT = Path(r"D:\time_uncertainty_rec")

DATASET = "MovieLens-1M"
ORDER = "file_order"
SEED = 42

MAX_LEN = 50
HIDDEN_SIZE = 64
NUM_HEADS = 2
NUM_BLOCKS = 2
DROPOUT = 0.2

BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 5e-4
EPOCHS = 20
PATIENCE = 4
SAMPLES_PER_USER = 10


# ============================================================
# 1. 路径
# ============================================================

TRAIN_PATH = (
    ROOT
    / "processed"
    / "diagnostic_a_splits"
    / DATASET
    / ORDER
    / "train.csv"
)

ITEM_SOURCE_PATH = (
    ROOT
    / "processed"
    / "diagnostic_a"
    / DATASET
    / "file_order.csv"
)

VAL_PATH = (
    ROOT
    / "processed"
    / "recent_bucketset_r1"
    / DATASET
    / "val"
    / "r1_samples.csv"
)

TEST_PATH = (
    ROOT
    / "processed"
    / "recent_bucketset_r1"
    / DATASET
    / "test"
    / "r1_samples.csv"
)

OUTPUT_DIR = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "matched_control"
    / "sasrec"
    / DATASET
    / ORDER
    / f"seed{SEED}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 随机种子 / 设备
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 严格确定性
    torch.use_deterministic_algorithms(True)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


set_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

print("=" * 80)
print("Step 31.2: Amazon Beauty Recent Bucket-Set R=1")
print("=" * 80)
print("Dataset:", DATASET)
print("Order:", ORDER)
print("Seed:", SEED)
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# ============================================================
# 3. 输入检查
# ============================================================

for path in [
    TRAIN_PATH,
    ITEM_SOURCE_PATH,
    VAL_PATH,
    TEST_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)


# ============================================================
# 4. 统一 item mapping
# ============================================================

item_source = pd.read_csv(
    ITEM_SOURCE_PATH,
    dtype={"item_id": str},
    usecols=["item_id"],
)

raw_items = sorted(
    item_source["item_id"]
    .astype(str)
    .unique()
    .tolist()
)

item_map = {
    raw_item: idx + 1
    for idx, raw_item in enumerate(raw_items)
}

idx_to_raw_item = {
    idx + 1: raw_item
    for idx, raw_item in enumerate(raw_items)
}

NUM_ITEMS = len(item_map)

print("NUM_ITEMS:", NUM_ITEMS)

pd.DataFrame(
    {
        "raw_item_id": raw_items,
        "item_idx": np.arange(1, NUM_ITEMS + 1),
    }
).to_csv(
    OUTPUT_DIR / "item_id_map.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 5. 读取训练数据，并保留时间桶
# ============================================================

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype={
        "user_id": str,
        "item_id": str,
    },
)

train_df["user_id"] = train_df["user_id"].astype(str)
train_df["item_id"] = train_df["item_id"].astype(str)
train_df["_source_order"] = np.arange(
    len(train_df),
    dtype=np.int64,
)

if "time_bucket" in train_df.columns:
    BUCKET_COLUMN = "time_bucket"
    train_df = train_df.sort_values(
        ["user_id", "time_bucket", "_source_order"],
        kind="mergesort",
    )
elif "timestamp" in train_df.columns:
    # 仅作为兜底；Amazon Beauty 正常应有 time_bucket。
    BUCKET_COLUMN = "timestamp"
    train_df = train_df.sort_values(
        ["user_id", "timestamp", "_source_order"],
        kind="mergesort",
    )
else:
    raise RuntimeError(
        "训练数据既没有 time_bucket 也没有 timestamp，无法确定最后历史桶。"
    )


# ============================================================
# 6. 用户训练序列：items + buckets
# ============================================================

user_train_sequences = {}

for user_id, group in train_df.groupby("user_id", sort=False):
    mapped_items = []

    for raw_item in group["item_id"].astype(str).tolist():
        if raw_item not in item_map:
            raise RuntimeError(
                f"Train item 不在 mapping 中: {raw_item}"
            )
        mapped_items.append(item_map[raw_item])

    buckets = group[BUCKET_COLUMN].tolist()

    if len(mapped_items) != len(buckets):
        raise RuntimeError("训练 items / buckets 长度不一致")

    if len(mapped_items) >= 2:
        user_train_sequences[str(user_id)] = {
            "items": np.asarray(mapped_items, dtype=np.int64),
            "buckets": np.asarray(buckets),
        }

print("Train records:", len(train_df))
print("Train users:", len(user_train_sequences))
print("Training bucket column:", BUCKET_COLUMN)


# ============================================================
# 7. R=1 分组函数
#
# 对 history：
#   - 最后桶大小 >=2：最后桶全部事件共享一个 token；
#   - 其余事件一个事件一个 token。
# ============================================================

def build_r1_group_positions_from_buckets(
    history_buckets,
    max_len=MAX_LEN,
):
    n = len(history_buckets)

    group_positions = np.full(
        max_len,
        -1,
        dtype=np.int64,
    )

    if n == 0:
        return group_positions, 0, 0, 0

    last_bucket = history_buckets[-1]

    start = n - 1
    while (
        start > 0
        and history_buckets[start - 1] == last_bucket
    ):
        start -= 1

    last_bucket_size = n - start
    direct_pooling = int(last_bucket_size >= 2)

    event_groups = np.empty(n, dtype=np.int64)

    if direct_pooling:
        # prefix 每个事件独立；最后桶共享一个 group。
        for j in range(start):
            event_groups[j] = j
        event_groups[start:n] = start
        num_tokens = start + 1
    else:
        event_groups[:] = np.arange(n, dtype=np.int64)
        num_tokens = n

    group_positions[-n:] = event_groups

    return (
        group_positions,
        int(num_tokens),
        int(direct_pooling),
        int(last_bucket_size),
    )


def build_r1_group_positions_from_size(
    history_len,
    last_bucket_size,
    max_len=MAX_LEN,
):
    if history_len <= 0:
        raise ValueError("history_len must be > 0")

    if last_bucket_size < 1 or last_bucket_size > history_len:
        raise ValueError(
            f"invalid last_bucket_size={last_bucket_size}, "
            f"history_len={history_len}"
        )

    group_positions = np.full(
        max_len,
        -1,
        dtype=np.int64,
    )

    direct_pooling = int(last_bucket_size >= 2)

    if direct_pooling:
        prefix_len = history_len - last_bucket_size
        event_groups = np.empty(history_len, dtype=np.int64)

        for j in range(prefix_len):
            event_groups[j] = j

        event_groups[prefix_len:] = prefix_len
        num_tokens = prefix_len + 1
    else:
        event_groups = np.arange(history_len, dtype=np.int64)
        num_tokens = history_len

    group_positions[-history_len:] = event_groups

    return (
        group_positions,
        int(num_tokens),
        int(direct_pooling),
    )

# ============================================================
# Matched Control
#
# 不进行任何 pooling。
# 每一个历史事件独占一个 token。
# ============================================================

def build_control_group_positions(
    history_len,
    max_len,
):

    if history_len <= 0:
        raise ValueError(
            f"invalid history_len={history_len}"
        )

    if history_len > max_len:
        raise ValueError(
            f"history_len={history_len} > max_len={max_len}"
        )

    group_positions = np.full(
        max_len,
        -1,
        dtype=np.int64,
    )

    group_positions[
        -history_len:
    ] = np.arange(
        history_len,
        dtype=np.int64,
    )

    num_tokens = history_len

    return (
        group_positions,
        int(num_tokens),
    )
# ============================================================
# R=1 deterministic canonicalization
#
# 最后历史桶在 mean pooling 前按 item_idx 固定排序。
# 不增加任何信息，只固定浮点求和顺序。
# ============================================================

def canonicalize_last_bucket_items(
    history_items,
    last_bucket_size,
):

    items = np.asarray(
        history_items,
        dtype=np.int64,
    ).copy()

    if len(items) == 0:
        raise ValueError(
            "canonicalize出现空history"
        )

    if (
        last_bucket_size < 1
        or
        last_bucket_size > len(items)
    ):
        raise ValueError(
            f"invalid last_bucket_size="
            f"{last_bucket_size}, "
            f"history_len={len(items)}"
        )

    if last_bucket_size >= 2:

        start = (
            len(items)
            -
            last_bucket_size
        )

        items[
            start:
        ] = np.sort(
            items[
                start:
            ],
            kind="mergesort",
        )

    return items
# ============================================================
# 8. 训练 Dataset
#
# cutoff 抽样沿用 Step 28.2：
# 每个用户固定抽 SAMPLES_PER_USER 个 end_idx。
#
# 与旧 R=1 原型一致，本脚本将每个 cutoff 的最后一个 item
# 作为 target，并对 target 之前的 history 做 R=1 pooling。
# ============================================================

class R1TrainDataset(Dataset):
    def __init__(
        self,
        user_sequences,
        max_len,
        samples_per_user,
        seed,
    ):
        self.user_sequences = user_sequences
        self.max_len = max_len
        self.users = sorted(user_sequences.keys())
        self.samples = []

        rng = np.random.RandomState(seed)

        for user_id in self.users:
            seq = self.user_sequences[user_id]["items"]

            if len(seq) < 2:
                continue

            for _ in range(samples_per_user):
                end_idx = int(
                    rng.randint(2, len(seq) + 1)
                )
                self.samples.append((user_id, end_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_id, end_idx = self.samples[idx]

        info = self.user_sequences[user_id]
        items = info["items"]
        buckets = info["buckets"]

        target_idx = end_idx - 1
        history_start = max(
            0,
            target_idx - self.max_len,
        )

        history_items = items[
            history_start:target_idx
        ]
        history_buckets = buckets[
            history_start:target_idx
        ]

        if len(history_items) == 0:
            raise RuntimeError("训练样本出现空 history")

        # ----------------------------------------------------
        # 先读取该样本在 R=1 设定下是否属于可 pooling 样本，
        # 这里只保留为结构标签，不改变 Control 的输入表示。
        # ----------------------------------------------------
        (
            _,
            _,
            eligible_direct_pooling,
            last_bucket_size,
        ) = build_r1_group_positions_from_buckets(
            history_buckets,
            self.max_len,
        )

        history_len = len(
            history_items
        )

        # ----------------------------------------------------
        # Matched Control：
        # 每个历史事件独占一个 token，不做 pooling。
        # ----------------------------------------------------
        (
            group_positions,
            num_tokens,
        ) = build_control_group_positions(
            history_len,
            self.max_len,
        )

        # Control 实际没有 pooling
        direct_pooling = 0

        # ----------------------------------------------------
        # 非常重要：
        # Control 必须保留原始 file_order，
        # 这里绝对不能 canonicalize。
        # ----------------------------------------------------
        seq = np.zeros(
            self.max_len,
            dtype=np.int64,
        )

        seq[
            -history_len:
        ] = np.asarray(
            history_items,
            dtype=np.int64,
        )

        target = int(items[target_idx])

        return (
            torch.from_numpy(seq),
            torch.from_numpy(group_positions),
            torch.tensor(num_tokens, dtype=torch.long),
            torch.tensor(direct_pooling, dtype=torch.long),
            torch.tensor(last_bucket_size, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


train_dataset = R1TrainDataset(
    user_sequences=user_train_sequences,
    max_len=MAX_LEN,
    samples_per_user=SAMPLES_PER_USER,
    seed=SEED,
)

loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    generator=loader_generator,
)

print("Train samples:", len(train_dataset))
print("Train batches:", len(train_loader))


# ============================================================
# 9. 读取 Step 31.1 固定的 strict val/test
# ============================================================

def load_r1_eval(path):
    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_items": str,
        },
    )

    sequences = []
    group_positions_all = []
    num_tokens_all = []
    targets = []
    direct_pooling_all = []
    last_bucket_sizes = []

    for row in df.itertuples(index=False):
        raw_history = json.loads(row.sequence_items)

        mapped_history = []
        for raw_item in raw_history:
            raw_item = str(raw_item)
            if raw_item not in item_map:
                raise RuntimeError(
                    f"Eval history item 不在 mapping 中: {raw_item}"
                )
            mapped_history.append(item_map[raw_item])

        if len(mapped_history) == 0:
            raise RuntimeError(
                f"Eval 出现空 history: {row.sample_id}"
            )

        if len(mapped_history) > MAX_LEN:
            raise RuntimeError(
                f"Eval history > MAX_LEN: {row.sample_id}"
            )

        target_raw = str(row.target_item)
        if target_raw not in item_map:
            raise RuntimeError(
                f"Eval target 不在 mapping 中: {target_raw}"
            )

        target_idx = item_map[target_raw]

        if target_idx in mapped_history:
            raise RuntimeError(
                f"Target 泄漏进 history: {row.sample_id}"
            )

        history_len = len(
            mapped_history
        )

        last_bucket_size = int(
            row.last_bucket_size
        )

        # ----------------------------------------------------
        # Matched Control：
        # 不做任何 pooling。
        # 每个历史事件独占一个 token。
        #
        # direct_pooling 这里只保留为“R=1会不会处理该样本”
        # 的结构标签，用于后续 Direct-Pooling / Unpooled 分组。
        # ----------------------------------------------------

        direct_pooling = int(
            row.direct_pooling
        )

        (
            group_positions,
            num_tokens,
        ) = build_control_group_positions(
            history_len,
            MAX_LEN,
        )

        # ----------------------------------------------------
        # Control 必须保留原始 file_order。
        # 这里绝对不能 canonicalize。
        # ----------------------------------------------------

        seq = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        seq[
            -history_len:
        ] = np.asarray(
            mapped_history,
            dtype=np.int64,
        )
        seq = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        seq[-history_len:] = np.asarray(
            mapped_history,
            dtype=np.int64,
        )

        seq = np.zeros(MAX_LEN, dtype=np.int64)
        seq[-history_len:] = np.asarray(
            mapped_history,
            dtype=np.int64,
        )
        expected_direct = int(row.direct_pooling)
        if direct_pooling != expected_direct:
            raise RuntimeError(
                f"direct_pooling mismatch: {row.sample_id}"
            )

        seq = np.zeros(MAX_LEN, dtype=np.int64)
        seq[-history_len:] = np.asarray(
            mapped_history,
            dtype=np.int64,
        )

        sequences.append(seq)
        group_positions_all.append(group_positions)
        num_tokens_all.append(num_tokens)
        targets.append(target_idx)
        direct_pooling_all.append(direct_pooling)
        last_bucket_sizes.append(last_bucket_size)

    return (
        df,
        np.stack(sequences),
        np.stack(group_positions_all),
        np.asarray(num_tokens_all, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(direct_pooling_all, dtype=np.int64),
        np.asarray(last_bucket_sizes, dtype=np.int64),
    )


(
    val_df,
    val_sequences,
    val_groups,
    val_num_tokens,
    val_targets,
    val_direct_pooling,
    val_last_bucket_sizes,
) = load_r1_eval(VAL_PATH)

(
    test_df,
    test_sequences,
    test_groups,
    test_num_tokens,
    test_targets,
    test_direct_pooling,
    test_last_bucket_sizes,
) = load_r1_eval(TEST_PATH)

print("Strict Val samples:", len(val_targets))
print("Strict Test samples:", len(test_targets))
print(
    "Val direct-pooling ratio:",
    f"{val_direct_pooling.mean():.6f}",
)
print(
    "Test direct-pooling ratio:",
    f"{test_direct_pooling.mean():.6f}",
)


# ============================================================
# 10. Recent Bucket-Set R=1 SASRec
#
# 核心来自旧 Step 15：
# 多个事件共享 group -> embedding scatter_add -> mean pooling。
#
# 与旧实现不同的关键修正：
# Transformer 实际运行时不使用 padding；
# 每个 compact token 序列使用右对齐绝对位置：
#     MAX_LEN - num_tokens ... MAX_LEN - 1
# ============================================================

class RecentBucketSetR1SASRec(nn.Module):
    def __init__(
        self,
        num_items,
        max_len,
        hidden_size,
        num_heads,
        num_blocks,
        dropout,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len
        self.hidden_size = hidden_size

        self.item_embedding = nn.Embedding(
            num_items + 1,
            hidden_size,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            max_len,
            hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_blocks,
        )

        self.final_norm = nn.LayerNorm(hidden_size)

    def build_token_embeddings(
        self,
        seq,
        group_positions,
    ):
        """原始事件 embedding -> group/token 的 mean embedding。"""

        batch_size = seq.size(0)

        item_embeddings = (
            self.item_embedding(seq)
            * math.sqrt(self.hidden_size)
        )

        valid_event_mask = (
            group_positions >= 0
        )

        safe_positions = (
            group_positions
            .clamp(min=0)
            .long()
        )

        # ====================================================
        # Deterministic group mean pooling
        #
        # 不使用 CUDA scatter_add_。
        # 每个 event 通过 one-hot assignment 映射到 group，
        # 再使用固定矩阵乘法完成 mean pooling。
        # ====================================================

        assignment = F.one_hot(
            safe_positions,
            num_classes=self.max_len,
        ).to(
            dtype=item_embeddings.dtype,
            device=item_embeddings.device,
        )

        assignment = (
            assignment
            *
            valid_event_mask
            .unsqueeze(-1)
            .to(dtype=item_embeddings.dtype)
        )

        # [B, group]
        token_count = (
            assignment
            .sum(dim=1)
            .clamp(min=1.0)
        )

        # [B, event, group]
        weights = (
            assignment
            /
            token_count.unsqueeze(1)
        )

        # [B, group, event] @ [B, event, hidden]
        # -> [B, group, hidden]
        token_embeddings = torch.bmm(
            weights.transpose(1, 2),
            item_embeddings,
        )

        return token_embeddings

    def predict_group_unpadded(
        self,
        seq,
        group_positions,
        num_tokens,
    ):
        """
        输入这一组样本的 num_tokens 必须完全相同。
        Transformer 不接收 padding；位置编码右对齐。
        """

        if seq.size(0) == 0:
            raise ValueError("empty batch")

        num_tokens = int(num_tokens)

        if num_tokens <= 0 or num_tokens > self.max_len:
            raise ValueError(
                f"invalid num_tokens={num_tokens}"
            )

        # Step38 Control fast path:
        # 每个历史事件对应唯一 token，不需要 One-hot + BMM。
        # 仅适用于 Control，不适用于 LBSP / R1 pooling。
        x = (
            self.item_embedding(seq[:, -num_tokens:])
            * math.sqrt(self.hidden_size)
        )

        position_offset = self.max_len - num_tokens

        positions = (
            torch.arange(
                position_offset,
                self.max_len,
                device=seq.device,
            )
            .unsqueeze(0)
            .expand(seq.size(0), num_tokens)
        )

        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(
                num_tokens,
                num_tokens,
                device=seq.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        x = self.encoder(
            x,
            mask=causal_mask,
        )

        x = self.final_norm(x)
        final_feature = x[:, -1, :]

        scores = torch.matmul(
            final_feature,
            self.item_embedding.weight.T,
        )

        return scores


model = RecentBucketSetR1SASRec(
    num_items=NUM_ITEMS,
    max_len=MAX_LEN,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    dropout=DROPOUT,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

criterion = nn.CrossEntropyLoss(
    reduction="sum"
)

print(
    "Model parameters:",
    sum(p.numel() for p in model.parameters()),
)


# ============================================================
# 11. 训练：按 num_tokens 分组，避免 padding
# ============================================================

def train_one_batch(
    model,
    seq,
    groups,
    num_tokens,
    target,
):
    unique_token_counts = torch.unique(
        num_tokens,
        sorted=True,
    )

    loss_sum = None
    sample_count = 0

    for token_count_tensor in unique_token_counts:
        token_count = int(token_count_tensor.item())
        mask = num_tokens == token_count_tensor
        indices = torch.where(mask)[0]

        scores = model.predict_group_unpadded(
            seq[indices],
            groups[indices],
            token_count,
        )

        if not torch.isfinite(scores).all():
            raise RuntimeError(
                "训练 scores 出现 NaN/Inf"
            )

        # padding item 0 不参与分类。
        scores[:, 0] = -1e9

        group_loss = criterion(
            scores,
            target[indices],
        )

        if not torch.isfinite(group_loss):
            raise RuntimeError(
                "训练 loss 出现 NaN/Inf"
            )

        loss_sum = (
            group_loss
            if loss_sum is None
            else loss_sum + group_loss
        )

        sample_count += len(indices)

    if sample_count == 0:
        raise RuntimeError("训练 batch 没有有效样本")

    return loss_sum / sample_count


# ============================================================
# 12. full-ranking 评价
# ============================================================

def calculate_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)

    if len(ranks) == 0:
        return {
            "N": 0,
            "NDCG@10": float("nan"),
            "HR@10": float("nan"),
            "HR@20": float("nan"),
            "MeanRank": float("nan"),
            "MedianRank": float("nan"),
            "Rank1Ratio": float("nan"),
        }

    ndcg_values = np.where(
        ranks <= 10,
        1.0 / np.log2(ranks + 1),
        0.0,
    )

    return {
        "N": int(len(ranks)),
        "NDCG@10": float(ndcg_values.mean()),
        "HR@10": float((ranks <= 10).mean()),
        "HR@20": float((ranks <= 20).mean()),
        "MeanRank": float(ranks.mean()),
        "MedianRank": float(np.median(ranks)),
        "Rank1Ratio": float((ranks == 1).mean()),
    }


@torch.no_grad()
def evaluate(
    model,
    sequences,
    groups,
    num_tokens,
    targets,
    direct_pooling,
    sample_df=None,
    save_predictions=False,
):
    model.eval()

    total = len(targets)
    all_ranks = np.empty(total, dtype=np.int64)
    all_top10 = np.empty((total, 10), dtype=np.int64)

    for start in range(0, total, EVAL_BATCH_SIZE):
        end = min(start + EVAL_BATCH_SIZE, total)

        seq_np = sequences[start:end]
        group_np = groups[start:end]
        token_np = num_tokens[start:end]
        target_np = targets[start:end]

        batch_scores = torch.empty(
            (len(seq_np), NUM_ITEMS + 1),
            device=device,
            dtype=torch.float32,
        )

        for token_count in np.unique(token_np):
            token_count = int(token_count)
            indices_np = np.where(
                token_np == token_count
            )[0]

            indices = torch.as_tensor(
                indices_np,
                device=device,
                dtype=torch.long,
            )

            seq_group = torch.from_numpy(
                seq_np[indices_np]
            ).long().to(device)

            group_group = torch.from_numpy(
                group_np[indices_np]
            ).long().to(device)

            group_scores = model.predict_group_unpadded(
                seq_group,
                group_group,
                token_count,
            )

            if not torch.isfinite(group_scores).all():
                raise RuntimeError(
                    "Strict evaluation 出现 NaN/Inf"
                )

            batch_scores[indices] = group_scores

        batch_scores[:, 0] = -1e9

        # seen-item mask 使用原始事件历史，而不是压缩后的 token。
        for b in range(len(seq_np)):
            seen = np.unique(seq_np[b])
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
                batch_scores[b, seen_tensor] = -1e9

        target_tensor = torch.from_numpy(
            target_np
        ).long().to(device)

        row_index = torch.arange(
            len(target_tensor),
            device=device,
        )

        target_scores = batch_scores[
            row_index,
            target_tensor,
        ]

        if not torch.isfinite(target_scores).all():
            raise RuntimeError(
                "Target score 出现 NaN/Inf"
            )

        ranks = (
            (batch_scores > target_scores.unsqueeze(1))
            .sum(dim=1)
            + 1
        )

        top10 = torch.topk(
            batch_scores,
            k=10,
            dim=1,
        ).indices

        all_ranks[start:end] = (
            ranks.detach().cpu().numpy().astype(np.int64)
        )

        all_top10[start:end] = (
            top10.detach().cpu().numpy().astype(np.int64)
        )

    overall = calculate_metrics(all_ranks)

    direct_mask = direct_pooling.astype(bool)
    unpooled_mask = ~direct_mask

    subgroup = {
        "direct_pooling": calculate_metrics(
            all_ranks[direct_mask]
        ),
        "unpooled": calculate_metrics(
            all_ranks[unpooled_mask]
        ),
        "pooling_ratio": float(direct_mask.mean()),
    }

    if save_predictions and sample_df is not None:
        pred_df = sample_df[
            [
                "sample_id",
                "user_id",
                "target_item",
                "ambiguous_last_bucket",
                "direct_pooling",
                "last_bucket_size",
            ]
        ].copy()

        ndcg_values = np.where(
            all_ranks <= 10,
            1.0 / np.log2(all_ranks + 1),
            0.0,
        )

        pred_df["target_rank"] = all_ranks
        pred_df["ndcg10"] = ndcg_values
        pred_df["hit10"] = (all_ranks <= 10).astype(int)
        pred_df["hit20"] = (all_ranks <= 20).astype(int)
        pred_df["num_tokens"] = num_tokens
        pred_df["top10_item_idx"] = [
            json.dumps(row.tolist())
            for row in all_top10
        ]
        pred_df["top10_raw_item_id"] = [
            json.dumps(
                [idx_to_raw_item[int(x)] for x in row],
                ensure_ascii=False,
            )
            for row in all_top10
        ]

        return overall, subgroup, pred_df

    return overall, subgroup


# ============================================================
# 13. 训练 + strict validation early stopping
# ============================================================

best_ndcg = -1.0
bad_epochs = 0
training_log = []

checkpoint_path = (
    OUTPUT_DIR
    / "matched_control_sasrec_best.pt"
)

print("\n" + "=" * 80)
print("开始训练 R=1")
print("=" * 80)

training_begin = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_begin = time.time()

    total_loss = 0.0
    valid_batches = 0
    pooled_samples = 0
    total_samples = 0

    for batch_idx, (
        seq,
        groups,
        num_tokens,
        direct_pooling,
        last_bucket_size,
        target,
    ) in enumerate(train_loader, start=1):

        seq = seq.long().to(device)
        groups = groups.long().to(device)
        num_tokens = num_tokens.long().to(device)
        direct_pooling = direct_pooling.long().to(device)
        target = target.long().to(device)

        optimizer.zero_grad()

        loss = train_one_batch(
            model,
            seq,
            groups,
            num_tokens,
            target,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )
        optimizer.step()

        total_loss += float(loss.item())
        valid_batches += 1
        pooled_samples += int(
            direct_pooling.sum().item()
        )
        total_samples += int(len(seq))

        if batch_idx % 200 == 0:
            print(
                f"Epoch {epoch:02d} "
                f"| Batch {batch_idx:04d}/{len(train_loader)} "
                f"| Loss {loss.item():.4f}"
            )

    avg_loss = total_loss / max(valid_batches, 1)
    train_pooling_ratio = (
        pooled_samples / total_samples
        if total_samples > 0
        else 0.0
    )

    val_metrics, val_subgroup = evaluate(
        model,
        val_sequences,
        val_groups,
        val_num_tokens,
        val_targets,
        val_direct_pooling,
    )

    elapsed = time.time() - epoch_begin

    print(
        f"\nEpoch {epoch:02d}"
        f" | Loss {avg_loss:.6f}"
        f" | Val NDCG@10 {val_metrics['NDCG@10']:.6f}"
        f" | HR@10 {val_metrics['HR@10']:.6f}"
        f" | HR@20 {val_metrics['HR@20']:.6f}"
        f" | MeanRank {val_metrics['MeanRank']:.2f}"
        f" | TrainPool {train_pooling_ratio:.4f}"
        f" | ValPool {val_subgroup['pooling_ratio']:.4f}"
        f" | {elapsed:.1f}s"
    )

    training_log.append(
        {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_pooling_ratio": train_pooling_ratio,
            "val_ndcg10": val_metrics["NDCG@10"],
            "val_hr10": val_metrics["HR@10"],
            "val_hr20": val_metrics["HR@20"],
            "val_mean_rank": val_metrics["MeanRank"],
            "val_pooling_ratio": val_subgroup["pooling_ratio"],
            "elapsed_seconds": elapsed,
        }
    )

    if val_metrics["NDCG@10"] > best_ndcg:
        best_ndcg = val_metrics["NDCG@10"]
        bad_epochs = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_name": "RecentBucketSetR1SASRec",
            "dataset": DATASET,
            "order": ORDER,
            "seed": SEED,
            "recent_r": 1,
            "num_items": NUM_ITEMS,
            "max_len": MAX_LEN,
            "hidden_size": HIDDEN_SIZE,
            "num_heads": NUM_HEADS,
            "num_blocks": NUM_BLOCKS,
            "dropout": DROPOUT,
            "best_val_ndcg10": best_ndcg,
            "epoch": epoch,
            "protocol": "strict_target_after_last_history_bucket",
            "evaluation": "unpadded_right_aligned_compact_tokens",
            "training_objective": "target_level_full_ranking_cross_entropy",
            "pilot_only": True,
        }

        torch.save(checkpoint, checkpoint_path)
        print(">>> 保存最佳模型:", checkpoint_path)

    else:
        bad_epochs += 1
        print(
            f">>> Validation 连续 {bad_epochs} 个 epoch 未提升"
        )

    if bad_epochs >= PATIENCE:
        print("\n触发 Early Stopping。")
        break


pd.DataFrame(training_log).to_csv(
    OUTPUT_DIR / "training_log.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. 加载最佳模型并最终评价
# ============================================================

try:
    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
except TypeError:
    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)
model.eval()

final_val_metrics, final_val_subgroup = evaluate(
    model,
    val_sequences,
    val_groups,
    val_num_tokens,
    val_targets,
    val_direct_pooling,
)

(
    test_metrics,
    test_subgroup,
    test_predictions,
) = evaluate(
    model,
    test_sequences,
    test_groups,
    test_num_tokens,
    test_targets,
    test_direct_pooling,
    sample_df=test_df,
    save_predictions=True,
)

test_predictions.to_csv(
    OUTPUT_DIR / "test_per_sample_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 15. 保存 metrics / config
# ============================================================

total_training_seconds = time.time() - training_begin

if torch.cuda.is_available():
    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )
else:
    peak_gpu_memory_gb = None

final_metrics = {
    "dataset": DATASET,
    "order": ORDER,
    "seed": SEED,
    "method": "RecentBucketSet_R1_V1_mean_pooling",
    "pilot_only": True,
    "best_epoch": int(best_checkpoint["epoch"]),
    "validation": {
        "overall": final_val_metrics,
        "subgroup": final_val_subgroup,
    },
    "test": {
        "overall": test_metrics,
        "subgroup": test_subgroup,
    },
    "runtime": {
        "training_seconds": total_training_seconds,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
    },
}

with open(
    OUTPUT_DIR / "metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        final_metrics,
        f,
        ensure_ascii=False,
        indent=2,
    )

config = {
    "dataset": DATASET,
    "order": ORDER,
    "seed": SEED,
    "method": "RecentBucketSet_R1_V1_mean_pooling",
    "recent_r": 1,
    "max_len": MAX_LEN,
    "hidden_size": HIDDEN_SIZE,
    "num_heads": NUM_HEADS,
    "num_blocks": NUM_BLOCKS,
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "samples_per_user": SAMPLES_PER_USER,
    "validation_protocol": "strict",
    "test_protocol": "strict",
    "evaluation": "full ranking",
    "padding_evaluation": "unpadded",
    "position_encoding": "right_aligned_after_r1_token_compression",
    "training_objective": "target_level_full_ranking_cross_entropy",
    "training_cutoff_sampling": "same_random_cutoff_rule_as_step28_2",
    "pilot_only": True,
    "formal_comparison_warning": (
        "Step31.2 is a single-seed feasibility run. "
        "Step28.2 Control uses all-position BCE, while this R1 pilot uses "
        "target-level full-ranking CE transferred from the validated old R1 prototype. "
        "Do not report a formal Control-vs-R1 accuracy delta until a matched-control run is completed."
    ),
}

with open(
    OUTPUT_DIR / "config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 16. 最终打印
# ============================================================

print("\n" + "=" * 80)
print("Step 31.2 completed")
print("=" * 80)
print("Dataset:", DATASET)
print("Order:", ORDER)
print("Seed:", SEED)
print("Best epoch:", best_checkpoint["epoch"])

print("\nStrict Validation - Overall")
print("NDCG@10:", f"{final_val_metrics['NDCG@10']:.6f}")
print("HR@10:", f"{final_val_metrics['HR@10']:.6f}")
print("HR@20:", f"{final_val_metrics['HR@20']:.6f}")
print("MeanRank:", f"{final_val_metrics['MeanRank']:.2f}")
print(
    "Direct-Pooling eligible ratio:",
    f"{final_val_subgroup['pooling_ratio']:.6f}",
)

print("\nStrict Test - Overall")
print("NDCG@10:", f"{test_metrics['NDCG@10']:.6f}")
print("HR@10:", f"{test_metrics['HR@10']:.6f}")
print("HR@20:", f"{test_metrics['HR@20']:.6f}")
print("MeanRank:", f"{test_metrics['MeanRank']:.2f}")
print("Rank=1 ratio:", f"{test_metrics['Rank1Ratio'] * 100:.4f}%")
print("Direct-Pooling eligible ratio:", f"{test_subgroup['pooling_ratio']:.6f}")

print("\nStrict Test - Direct-Pooling")
for k, v in test_subgroup["direct_pooling"].items():
    print(k + ":", v)

print("\nStrict Test - Unpooled")
for k, v in test_subgroup["unpooled"].items():
    print(k + ":", v)

if peak_gpu_memory_gb is not None:
    print(
        "\nPeak GPU memory:",
        f"{peak_gpu_memory_gb:.3f} GB",
    )

print(
    "Training time:",
    f"{total_training_seconds:.1f}s",
)

print("\nOutput directory:", OUTPUT_DIR)
print(
    "\nNOTE: This is a single-seed pilot. "
    "Do not use Step28.2-vs-Step31.2 accuracy difference as a formal paper result yet."
)
