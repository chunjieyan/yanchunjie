import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "step38_kais_retrain"

TRAIN = (
    ROOT / "processed/diagnostic_a_splits"
    / "MovieLens-1M/file_order/train.csv"
)

MAP_PATH = (
    BASE / "outputs"
    / "ml1m_sasrec_control_ce_bs128_lr1em3_max200_seed42"
    / "item_id_map.csv"
)

EVAL_DIR = (
    ROOT / "processed/recent_bucketset_r1"
    / "MovieLens-1M"
)

OUT = BASE / "outputs/ml1m_popularity_random_strict_seed42"

SEED = 42
MAX_LEN = 50


def calculate_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)

    if len(ranks) == 0:
        raise RuntimeError("评价子组为空")

    ndcg = np.where(
        ranks <= 10,
        1.0 / np.log2(ranks + 1),
        0.0,
    )

    return {
        "N": int(len(ranks)),
        "NDCG@10": float(ndcg.mean()),
        "HR@10": float((ranks <= 10).mean()),
        "HR@20": float((ranks <= 20).mean()),
        "MeanRank": float(ranks.mean()),
        "MedianRank": float(np.median(ranks)),
        "Rank1Ratio": float((ranks == 1).mean()),
    }


# 直接复用正式 SASRec 的物品编号
mapping = pd.read_csv(
    MAP_PATH,
    dtype={"raw_item_id": str},
)

assert mapping["raw_item_id"].is_unique
assert mapping["item_idx"].is_unique

item_map = dict(zip(
    mapping["raw_item_id"],
    mapping["item_idx"].astype(int),
))

N = len(item_map)

assert N == 3416
assert set(item_map.values()) == set(range(1, N + 1))

print("NUM_ITEMS:", N)


# Popularity 只使用训练集频次
train = pd.read_csv(
    TRAIN,
    usecols=["item_id"],
    dtype={"item_id": str},
)

mapped_train = train["item_id"].map(item_map)

assert mapped_train.notna().all()

counts = np.bincount(
    mapped_train.to_numpy(dtype=np.int64),
    minlength=N + 1,
)

assert counts.sum() == len(train)


# 排名规则：
# 先按训练频次降序；相同频次按 item_idx 升序。
# 为每个物品建立唯一且可复现的排序优先级。
ids = np.arange(1, N + 1)

order = np.lexsort((
    ids,
    -counts[1:],
))

priority = np.full(
    N + 1,
    N + 1,
    dtype=np.int64,
)

priority[ids[order]] = np.arange(N)


def evaluate_split(split, random_seed):
    path = EVAL_DIR / split / "r1_samples.csv"

    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_items": str,
        },
    )

    expected_n = 752 if split == "val" else 945

    assert len(df) == expected_n, (
        f"{split} 样本数异常: {len(df)}"
    )

    rng = np.random.default_rng(random_seed)

    result_rows = []

    for row in df.itertuples(index=False):
        history_raw = json.loads(row.sequence_items)

        assert 0 < len(history_raw) <= MAX_LEN

        history = np.asarray(
            [item_map[str(x)] for x in history_raw],
            dtype=np.int64,
        )

        target = item_map[str(row.target_item)]

        # 与 SASRec strict eval 一致：禁止目标泄漏
        assert target not in history

        # 屏蔽 Padding 0 和历史已见物品
        available = np.ones(N + 1, dtype=bool)

        available[0] = False
        available[history] = False

        assert available[target]

        # Popularity 排名，包含确定性同频次打破平局
        pop_rank = 1 + np.count_nonzero(
            available
            & (priority < priority[target])
        )

        # Random：每个样本独立生成随机分数
        scores = rng.random(N + 1)

        scores[~available] = -np.inf

        random_rank = 1 + np.count_nonzero(
            scores > scores[target]
        )

        result_rows.append({
            "sample_id": row.sample_id,
            "user_id": row.user_id,
            "target_idx": int(target),
            "direct_pooling": int(row.direct_pooling),
            "last_bucket_size": int(row.last_bucket_size),
            "popularity_rank": int(pop_rank),
            "random_rank": int(random_rank),
        })

    result = pd.DataFrame(result_rows)

    assert result["sample_id"].is_unique

    summary = {}

    for method, column in [
        ("Popularity", "popularity_rank"),
        ("Random", "random_rank"),
    ]:
        ranks = result[column].to_numpy()
        direct = result["direct_pooling"].to_numpy().astype(bool)

        assert ranks.min() >= 1
        assert ranks.max() <= N

        summary[method] = {
            "overall": calculate_metrics(ranks),
            "direct_pooling": calculate_metrics(ranks[direct]),
            "unpooled": calculate_metrics(ranks[~direct]),
        }

    return result, summary


val_rows, val_metrics = evaluate_split(
    "val", random_seed=SEED
)

test_rows, test_metrics = evaluate_split(
    "test", random_seed=SEED + 1
)

# 完成全部检查后才创建新的输出目录
if OUT.exists():
    raise RuntimeError(
        f"输出目录已经存在，禁止覆盖: {OUT}"
    )

OUT.mkdir(parents=True, exist_ok=False)

val_rows.to_csv(OUT / "val_ranks.csv", index=False)
test_rows.to_csv(OUT / "test_ranks.csv", index=False)

report = {
    "dataset": "MovieLens-1M",
    "protocol": "Step38_B1_strict_full_ranking",
    "num_items": N,
    "train_events": int(len(train)),
    "random_val_seed": SEED,
    "random_test_seed": SEED + 1,
    "popularity_tie_break": "smaller_item_idx_first",
    "seen_mask": "exclude_seen_except_target",
    "val": val_metrics,
    "test": test_metrics,
}

(OUT / "metrics.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("===== STEP38.2 STRICT BASELINES =====")

for method in ("Popularity", "Random"):
    print(f"\n{method}")

    for split, metrics in [
        ("Val", val_metrics),
        ("Test", test_metrics),
    ]:
        m = metrics[method]["overall"]

        print(
            f"{split}: "
            f"NDCG@10={m['NDCG@10']:.6f} "
            f"HR@10={m['HR@10']:.6f} "
            f"HR@20={m['HR@20']:.6f} "
            f"MeanRank={m['MeanRank']:.2f}"
        )

print()
print("STEP38.2 BASELINE EVALUATION PASS")
print("Output:", OUT)
