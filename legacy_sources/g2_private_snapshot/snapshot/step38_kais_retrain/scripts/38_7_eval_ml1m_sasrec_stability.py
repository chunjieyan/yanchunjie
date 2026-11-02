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
# Step38.7 - ML1M SASRec Control Stability Evaluation
#
# Fixed checkpoint: Step38.1, best epoch 66
# Permutation seeds: 101, 202, 303, 404, 505
# No training; no parameter updates
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "step38_kais_retrain"

DATASET = "MovieLens-1M"
MODEL_SEED = 42
PERMUTATION_SEEDS = [101, 202, 303, 404, 505]

MAX_LEN = 50
EVAL_BATCH_SIZE = 256
EXPECTED_SAMPLES = 945

MODEL_SCRIPT = (
    BASE / "scripts"
    / "38_1_train_ml1m_sasrec_bs128.py"
)

MODEL_DIR = (
    BASE / "outputs"
    / "ml1m_sasrec_control_ce_bs128_lr1em3_max200_seed42"
)

CHECKPOINT_PATH = (
    MODEL_DIR / "matched_control_sasrec_best.pt"
)

ITEM_MAP_PATH = MODEL_DIR / "item_id_map.csv"

ORIGINAL_PATH = (
    ROOT
    / "processed/recent_bucketset_r1"
    / DATASET
    / "test/r1_samples.csv"
)

PERM_ROOT = (
    ROOT
    / "processed/strict_last_bucket_only"
    / DATASET
)

OUTPUT_DIR = (
    BASE / "outputs"
    / "ml1m_sasrec_stability_seed42"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 72)
print("STEP38.7 ML1M SASREC STABILITY")
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
        PERM_ROOT / f"perm_seed{seed}" / "test.csv"
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

if NUM_ITEMS != 3416:
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

if int(checkpoint["epoch"]) != 66:
    raise RuntimeError("Checkpoint最佳轮次不是66")

if int(checkpoint["num_items"]) != NUM_ITEMS:
    raise RuntimeError("Checkpoint物品数不一致")

if int(checkpoint["seed"]) != MODEL_SEED:
    raise RuntimeError("Checkpoint随机种子不一致")

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

if abs(ndcg - 0.046118) >= 1e-6:
    raise RuntimeError(
        "严格 NDCG@10 与 Step38.1 不一致"
    )

print("ORIGINAL STRICT REPRODUCTION PASS")


# ============================================================
# 8. 评价五组最后时间桶排列
# ============================================================

predictions = {}

for seed in PERMUTATION_SEEDS:
    input_path = (
        PERM_ROOT
        / f"perm_seed{seed}"
        / "test.csv"
    )

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
print("STEP38.7 ML1M STABILITY INFERENCE PASS")
print("Output:", OUTPUT_DIR)
print("=" * 72)
