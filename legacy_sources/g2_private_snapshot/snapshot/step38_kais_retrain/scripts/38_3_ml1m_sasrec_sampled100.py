import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "step38_kais_retrain"

MODEL_SCRIPT = BASE / "scripts/38_1_train_ml1m_sasrec_bs128.py"

MODEL_DIR = (
    BASE / "outputs"
    / "ml1m_sasrec_control_ce_bs128_lr1em3_max200_seed42"
)

CHECKPOINT = MODEL_DIR / "matched_control_sasrec_best.pt"
MAP_PATH = MODEL_DIR / "item_id_map.csv"

EVAL_ROOT = (
    ROOT / "processed/recent_bucketset_r1/MovieLens-1M"
)

OUT = BASE / "outputs/ml1m_sasrec_sampled100_seed42"

MAX_LEN = 50
EVAL_BATCH_SIZE = 256
NUM_NEGATIVES = 100
REPEATS = 10
SEED = 42

EXPECTED_NDCG = {
    "val": 0.079358,
    "test": 0.046118,
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if OUT.exists():
    raise RuntimeError(f"输出目录已存在，禁止覆盖: {OUT}")


# ============================================================
# 1. 只提取模型类，绝不执行原训练脚本
# ============================================================

tree = ast.parse(
    MODEL_SCRIPT.read_text(encoding="utf-8")
)

classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef)
    and node.name == "SASRecControl"
]

assert len(classes) == 1

namespace = {
    "__name__": "step38_checkpoint_evaluator",
    "torch": torch,
    "nn": nn,
    "F": F,
    "np": np,
    "math": math,
    "MAX_LEN": MAX_LEN,
}

exec(
    compile(
        ast.Module(body=classes, type_ignores=[]),
        str(MODEL_SCRIPT),
        "exec",
    ),
    namespace,
)

SASRecControl = namespace["SASRecControl"]


# ============================================================
# 2. 加载最佳 checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

assert checkpoint["model_name"] == "SASRecControl"
assert int(checkpoint["epoch"]) == 66
assert int(checkpoint["max_len"]) == MAX_LEN

mapping = pd.read_csv(
    MAP_PATH,
    dtype={"raw_item_id": str},
)

item_map = dict(zip(
    mapping["raw_item_id"],
    mapping["item_idx"].astype(int),
))

N = len(item_map)

assert int(checkpoint["num_items"]) == N
assert set(item_map.values()) == set(range(1, N + 1))

model = SASRecControl(
    num_items=N,
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
print("NUM_ITEMS:", N)


# ============================================================
# 3. 读取冻结评价样本
# ============================================================

def load_split(split):
    path = EVAL_ROOT / split / "r1_samples.csv"

    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "sequence_items": str,
            "target_item": str,
        },
    )

    expected = 752 if split == "val" else 945

    assert len(df) == expected
    assert df["sample_id"].is_unique

    sequences = []
    lengths = []
    targets = []
    histories = []

    for row in df.itertuples(index=False):
        raw_history = json.loads(row.sequence_items)

        history = np.asarray(
            [item_map[str(x)] for x in raw_history],
            dtype=np.int64,
        )

        target = item_map[str(row.target_item)]

        assert 0 < len(history) <= MAX_LEN
        assert target not in history

        seq = np.zeros(MAX_LEN, dtype=np.int64)
        seq[-len(history):] = history

        sequences.append(seq)
        lengths.append(len(history))
        targets.append(target)
        histories.append(history)

    return (
        df,
        np.stack(sequences),
        np.asarray(lengths, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        histories,
    )


# ============================================================
# 4. 一次前向计算得到全部物品分数
# ============================================================

@torch.inference_mode()
def get_scores(sequences, lengths, targets, histories):
    all_scores = []
    all_ranks = []

    for start in range(0, len(targets), EVAL_BATCH_SIZE):
        end = min(start + EVAL_BATCH_SIZE, len(targets))

        seq = torch.from_numpy(
            sequences[start:end]
        ).long().to(device)

        lens = torch.from_numpy(
            lengths[start:end]
        ).long().to(device)

        scores = model.predict_batch_masked(seq, lens)

        assert torch.isfinite(scores).all()

        scores[:, 0] = -1e9

        for j in range(end - start):
            target = targets[start + j]
            seen = np.unique(histories[start + j])
            seen = seen[seen != target]

            if len(seen):
                scores[
                    j,
                    torch.as_tensor(
                        seen,
                        dtype=torch.long,
                        device=device,
                    ),
                ] = -1e9

        target_ids = torch.as_tensor(
            targets[start:end],
            dtype=torch.long,
            device=device,
        )

        target_scores = scores[
            torch.arange(end - start, device=device),
            target_ids,
        ]

        ranks = (
            (scores > target_scores.unsqueeze(1))
            .sum(dim=1)
            + 1
        )

        all_scores.append(scores.cpu().numpy())
        all_ranks.append(ranks.cpu().numpy())

    return (
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_ranks, axis=0),
    )


def calculate_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)

    return {
        "N": int(len(ranks)),
        "NDCG@10": float(
            np.where(
                ranks <= 10,
                1.0 / np.log2(ranks + 1),
                0.0,
            ).mean()
        ),
        "HR@10": float((ranks <= 10).mean()),
        "HR@20": float((ranks <= 20).mean()),
        "MeanRank": float(ranks.mean()),
    }


# ============================================================
# 5. 100个随机负样本 + 1个目标物品
# ============================================================

def evaluate_split(split):
    df, seq, lengths, targets, histories = load_split(split)

    scores, strict_ranks = get_scores(
        seq, lengths, targets, histories
    )

    strict_metrics = calculate_metrics(strict_ranks)

    print(
        f"{split.upper()} strict NDCG@10:",
        f"{strict_metrics['NDCG@10']:.6f}",
    )

    # 检查 checkpoint 与原严格评价结果一致
    assert abs(
        strict_metrics["NDCG@10"]
        - EXPECTED_NDCG[split]
    ) < 1e-6, "严格评价复算不一致，停止采样校准"

    all_items = np.arange(1, N + 1)
    negative_pools = []

    for target, history in zip(targets, histories):
        available = (
            (all_items != target)
            & (~np.isin(all_items, history))
        )

        pool = all_items[available]

        assert len(pool) >= NUM_NEGATIVES
        negative_pools.append(pool)

    repeat_summaries = []
    prediction_rows = []

    split_offset = 0 if split == "val" else 100000

    for repeat in range(REPEATS):
        rng = np.random.default_rng(
            SEED + split_offset + repeat
        )

        sampled_ranks = np.empty(
            len(targets), dtype=np.int64
        )

        for i, target in enumerate(targets):
            negative_items = rng.choice(
                negative_pools[i],
                size=NUM_NEGATIVES,
                replace=False,
            )

            target_score = scores[i, target]

            sampled_ranks[i] = (
                1 + np.count_nonzero(
                    scores[i, negative_items] > target_score
                )
            )

            prediction_rows.append({
                "split": split,
                "repeat": repeat,
                "sample_id": df.iloc[i]["sample_id"],
                "target_idx": int(target),
                "strict_rank": int(strict_ranks[i]),
                "sampled100_rank": int(sampled_ranks[i]),
            })

        metrics = calculate_metrics(sampled_ranks)
        metrics["repeat"] = repeat
        repeat_summaries.append(metrics)

        print(
            f"{split.upper()} repeat={repeat:02d} "
            f"NDCG@10={metrics['NDCG@10']:.6f} "
            f"HR@10={metrics['HR@10']:.6f}"
        )

    mean_ndcg = float(np.mean([
        x["NDCG@10"] for x in repeat_summaries
    ]))

    std_ndcg = float(np.std([
        x["NDCG@10"] for x in repeat_summaries
    ], ddof=1))

    mean_hr10 = float(np.mean([
        x["HR@10"] for x in repeat_summaries
    ]))

    print(
        f"{split.upper()} sampled100: "
        f"NDCG@10={mean_ndcg:.6f} ± {std_ndcg:.6f} "
        f"HR@10={mean_hr10:.6f}"
    )

    return {
        "strict": strict_metrics,
        "sampled100": {
            "ndcg10_mean": mean_ndcg,
            "ndcg10_std": std_ndcg,
            "hr10_mean": mean_hr10,
            "repeat_metrics": repeat_summaries,
        },
    }, prediction_rows


print("\n===== STRICT CHECK + SAMPLED100 =====")

val_report, val_rows = evaluate_split("val")
test_report, test_rows = evaluate_split("test")


# ============================================================
# 6. 所有审计通过后保存
# ============================================================

OUT.mkdir(parents=True, exist_ok=False)

report = {
    "dataset": "MovieLens-1M",
    "model": "SASRecControl",
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "checkpoint_path": str(CHECKPOINT),
    "num_items": N,
    "num_negatives": NUM_NEGATIVES,
    "total_candidates_per_sample": NUM_NEGATIVES + 1,
    "sample_without_replacement": True,
    "negative_pool": "all_items_excluding_history_and_target",
    "repeats": REPEATS,
    "seed": SEED,
    "sampling_protocol": "sampled100_calibration_not_strict_ranking",
    "val": val_report,
    "test": test_report,
}

(OUT / "metrics.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

pd.DataFrame(
    val_rows + test_rows
).to_csv(
    OUT / "sampled100_predictions.csv",
    index=False,
)

print()
print("STEP38.3 SAMPLED100 CALIBRATION PASS")
print("Output:", OUT)
