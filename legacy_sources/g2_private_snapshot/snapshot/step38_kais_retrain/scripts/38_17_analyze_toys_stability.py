import json
from pathlib import Path
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "step38_kais_retrain"

DATASET = "Amazon_Toys"
SEEDS = [101, 202, 303, 404, 505]
N = 2547

PRED_DIR = (
    BASE / "outputs/toys_sasrec_stability_seed42"
)

PERM_DIR = (
    ROOT / "processed/strict_last_bucket_only"
    / DATASET
)

ORIGINAL = (
    ROOT / "processed/strict_diagnostic_a"
    / DATASET / "file_order/test.csv"
)

OUT = (
    BASE / "outputs/toys_sasrec_stability_summary_seed42"
)

if OUT.exists():
    raise RuntimeError("结果目录已存在，禁止覆盖")


def read_indexed(path, ids=None):
    df = pd.read_csv(
        path,
        dtype={"sample_id": str, "target_item": str},
    )
    assert len(df) == N
    assert df["sample_id"].is_unique

    df = df.set_index("sample_id")

    if ids is not None:
        assert set(df.index) == set(ids)
        df = df.loc[ids]

    return df


def parse_seq(value):
    return tuple(str(x) for x in json.loads(value))


def parse_top10(value):
    values = tuple(int(x) for x in json.loads(value))
    assert len(values) == 10
    assert len(set(values)) == 10
    return values


def overlap10(a, b):
    return len(set(a) & set(b)) / 10.0


# ============================================================
# 1. 原始测试集与预测
# ============================================================

original = read_indexed(ORIGINAL)
ids = original.index.tolist()

base_pred = read_indexed(
    PRED_DIR / "predictions_original.csv",
    ids,
)

original_sequences = [
    parse_seq(x)
    for x in original["sequence_items"]
]

original_ranks = base_pred[
    "target_rank"
].to_numpy(dtype=np.int64)

original_top10 = [
    parse_top10(x)
    for x in base_pred["top10_item_idx"]
]

assert abs(original_ranks.mean() - 6930.95) < 0.02

# ============================================================
# 2. 五组排列：预测与输入对齐
# ============================================================

all_ranks = []
all_top10 = []
all_sequences = []

per_seed_rows = []
per_sample_changed = np.zeros(N, dtype=np.int64)

for seed in SEEDS:
    perm = read_indexed(
        PERM_DIR / f"perm_seed{seed}/test.csv",
        ids,
    )

    pred = read_indexed(
        PRED_DIR / f"predictions_perm_seed{seed}.csv",
        ids,
    )

    assert (
        perm["target_item"].astype(str).to_numpy()
        == original["target_item"].astype(str).to_numpy()
    ).all()

    assert (
        pred["target_item"].astype(str).to_numpy()
        == original["target_item"].astype(str).to_numpy()
    ).all()

    sequences = [
        parse_seq(x)
        for x in perm["sequence_items"]
    ]

    ranks = pred[
        "target_rank"
    ].to_numpy(dtype=np.int64)

    top10 = [
        parse_top10(x)
        for x in pred["top10_item_idx"]
    ]

    changed = np.asarray([
        a != b
        for a, b in zip(
            original_sequences,
            sequences,
        )
    ])

    rank_changed = ranks != original_ranks

    top10_changed = np.asarray([
        set(a) != set(b)
        for a, b in zip(
            original_top10,
            top10,
        )
    ])

    # 未扰动的历史不能产生不同的预测
    assert not np.any(rank_changed & ~changed)
    assert not np.any(top10_changed & ~changed)

    # 再次核验只改变最后时间桶内部顺序
    for i in range(N):
        k = int(perm.iloc[i]["last_history_bucket_size"])
        a = original_sequences[i]
        b = sequences[i]

        assert len(a) == len(b)
        assert 1 <= k <= len(a)
        assert a[:-k] == b[:-k]
        assert Counter(a[-k:]) == Counter(b[-k:])

    per_sample_changed += changed.astype(np.int64)

    changed_n = int(changed.sum())
    rank_changed_n = int(rank_changed.sum())
    top10_changed_n = int(top10_changed.sum())

    overlaps = [
        overlap10(a, b)
        for a, b in zip(original_top10, top10)
    ]

    changed_overlaps = [
        value
        for value, flag in zip(overlaps, changed)
        if flag
    ]

    row = {
        "seed": seed,
        "samples": N,
        "changed_sequences": changed_n,
        "changed_ranks": rank_changed_n,
        "changed_top10_sets": top10_changed_n,
        "rank_changed_rate_all": rank_changed_n / N,
        "rank_changed_rate_among_changed":
            rank_changed_n / changed_n if changed_n else 0.0,
        "top10_changed_rate_among_changed":
            top10_changed_n / changed_n if changed_n else 0.0,
        "overlap10_all": float(np.mean(overlaps)),
        "overlap10_changed": (
            float(np.mean(changed_overlaps))
            if changed_overlaps else None
        ),
        "mean_rank": float(ranks.mean()),
    }

    per_seed_rows.append(row)

    all_ranks.append(ranks)
    all_top10.append(top10)
    all_sequences.append(sequences)

    print(
        f"Seed {seed}: "
        f"ChangedSeq={changed_n} "
        f"ChangedRank={rank_changed_n} "
        f"ChangedTop10={top10_changed_n} "
        f"OverlapChanged={row['overlap10_changed']:.6f}"
    )


# ============================================================
# 3. 五组排列之间，两两比较
# ============================================================

rank_matrix = np.stack(all_ranks, axis=1)

pair_overlaps_all = []
pair_overlaps_distinct = []

pair_rank_diffs_all = []
pair_rank_diffs_distinct = []

pair_rank_change_all = []
pair_rank_change_distinct = []

sample_rows = []

for i, sid in enumerate(ids):
    ranks = rank_matrix[i]

    local_overlap = []
    local_distinct_overlap = []

    for a, b in combinations(range(5), 2):
        same_sequence = (
            all_sequences[a][i]
            == all_sequences[b][i]
        )

        overlap = overlap10(
            all_top10[a][i],
            all_top10[b][i],
        )

        abs_rank_diff = abs(
            int(ranks[a]) - int(ranks[b])
        )

        rank_changed = int(ranks[a] != ranks[b])

        if same_sequence:
            assert abs_rank_diff == 0
            assert set(all_top10[a][i]) == set(
                all_top10[b][i]
            )

        local_overlap.append(overlap)
        pair_overlaps_all.append(overlap)
        pair_rank_diffs_all.append(abs_rank_diff)
        pair_rank_change_all.append(rank_changed)

        if not same_sequence:
            local_distinct_overlap.append(overlap)
            pair_overlaps_distinct.append(overlap)
            pair_rank_diffs_distinct.append(abs_rank_diff)
            pair_rank_change_distinct.append(rank_changed)

    sample_rows.append({
        "sample_id": sid,
        "original_rank": int(original_ranks[i]),
        "changed_permutations_vs_original":
            int(per_sample_changed[i]),
        "rank_mean_5perm": float(np.mean(ranks)),
        "rank_std_5perm": float(np.std(ranks, ddof=0)),
        "rank_min_5perm": int(np.min(ranks)),
        "rank_max_5perm": int(np.max(ranks)),
        "pairwise_overlap10_mean":
            float(np.mean(local_overlap)),
        "distinct_pair_overlap10_mean": (
            float(np.mean(local_distinct_overlap))
            if local_distinct_overlap else None
        ),
    })


# ============================================================
# 4. 汇总结果
# ============================================================

def safe_mean(values):
    return float(np.mean(values)) if values else None


summary = {
    "dataset": DATASET,
    "model": "SASRecControl",
    "model_seed": 42,
    "checkpoint_best_epoch": 72,
    "test_samples": N,
    "permutation_seeds": SEEDS,
    "overlap10_definition":
        "intersection_of_top10_sets_divided_by_10",
    "rank_std_definition":
        "population_std_across_five_permutations_ddof0",
    "per_seed": per_seed_rows,
    "overall": {
        "samples_changed_at_least_once":
            int((per_sample_changed > 0).sum()),
        "samples_never_changed":
            int((per_sample_changed == 0).sum()),
        "mean_rank_std_5perm": float(
            np.std(rank_matrix, axis=1, ddof=0).mean()
        ),
        "mean_pairwise_overlap10_all":
            safe_mean(pair_overlaps_all),
        "mean_pairwise_overlap10_distinct_sequences":
            safe_mean(pair_overlaps_distinct),
        "mean_pairwise_abs_rank_diff_all":
            safe_mean(pair_rank_diffs_all),
        "mean_pairwise_abs_rank_diff_distinct_sequences":
            safe_mean(pair_rank_diffs_distinct),
        "pairwise_rank_change_rate_all":
            safe_mean(pair_rank_change_all),
        "pairwise_rank_change_rate_distinct_sequences":
            safe_mean(pair_rank_change_distinct),
        "total_pairwise_comparisons":
            len(pair_overlaps_all),
        "distinct_sequence_pairwise_comparisons":
            len(pair_overlaps_distinct),
    },
}

assert len(pair_overlaps_all) == N * 10

# ============================================================
# 5. 全部审计通过后保存
# ============================================================

OUT.mkdir(parents=True, exist_ok=False)

pd.DataFrame(per_seed_rows).to_csv(
    OUT / "per_seed_summary.csv",
    index=False,
)

pd.DataFrame(sample_rows).to_csv(
    OUT / "per_sample_stability.csv",
    index=False,
)

(OUT / "stability_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("===== STEP38.17 OVERALL STABILITY =====")

for key, value in summary["overall"].items():
    print(f"{key}: {value}")

print()
print("STEP38.17 TOYS STABILITY SUMMARY PASS")
print("Output:", OUT)
