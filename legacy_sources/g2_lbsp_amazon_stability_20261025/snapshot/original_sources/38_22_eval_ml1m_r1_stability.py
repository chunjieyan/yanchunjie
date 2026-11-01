
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

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "step38_kais_retrain"

SEEDS = [101, 202, 303, 404, 505]
MAX_LEN = 50
BATCH_SIZE = 256
EXPECTED_N = 945
EXPECTED_ITEMS = 3416

MODEL_DIR = (
    BASE / "outputs/"
    "ml1m_r1_meanpool_ce_bs128_lr1em3_max200_seed42"
)

CHECKPOINT = MODEL_DIR / "r1_meanpool_best.pt"
MAP_PATH = MODEL_DIR / "item_id_map.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"
SAVED_PRED = MODEL_DIR / "test_per_sample_predictions.csv"

ORIGINAL_PATH = (
    ROOT / "processed/recent_bucketset_r1/"
    "MovieLens-1M/test/r1_samples.csv"
)

PERM_ROOT = (
    ROOT / "processed/strict_last_bucket_only/"
    "MovieLens-1M"
)

R1_SOURCE = (
    ROOT / "audit/33_2_train_movielens_r1_seed42.py"
)

FORWARD_SOURCE = (
    BASE / "scripts/38_20_train_ml1m_r1_bs128.py"
)

OUTPUT_DIR = (
    BASE / "outputs/ml1m_r1_stability_seed42"
)

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
    PERM_ROOT / f"perm_seed{s}/test.csv"
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
    int(ckpt["epoch"]) == 33,
    "Checkpoint 不是 Epoch33",
)

require(
    int(ckpt["seed"]) == 42,
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
print("ORIGINAL STRICT REPRODUCTION PASS")


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
        PERM_ROOT / f"perm_seed{seed}/test.csv",
        reference_ids=ids,
    )

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
        "seed": seed,
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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

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
    "method": "RecentBucketSet_R1_V1_mean_pooling",
    "model_seed": 42,
    "checkpoint_epoch": 33,
    "training_batch_size": 128,
    "evaluation_batch_size": BATCH_SIZE,
    "test_samples": EXPECTED_N,
    "permutation_seeds": SEEDS,
    "strict_reproduction": "PASS",
    "canonicalization": "last_bucket_sorted_by_item_idx",
    "exact_raw_logits_invariance": True,
    "per_seed": seed_summaries,
}

(OUTPUT_DIR / "manifest.json").write_text(
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
