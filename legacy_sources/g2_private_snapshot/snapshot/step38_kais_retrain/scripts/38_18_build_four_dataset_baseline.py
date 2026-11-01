import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "step38_kais_retrain"
INPUT = BASE / "outputs"

OUTPUT = (
    INPUT / "step38_18_four_dataset_sasrec_control_seed42"
)

SEEDS = [101, 202, 303, 404, 505]

DATASETS = [
    {
        "name": "MovieLens-1M",
        "model": "ml1m_sasrec_control_ce_bs128_lr1em3_max200_seed42",
        "stability": "ml1m_sasrec_stability_summary_seed42",
        "val_n": 752,
        "test_n": 945,
        "epoch": 66,
        "eval_protocol": "recent_bucketset_r1",
    },
    {
        "name": "Amazon_Beauty",
        "model": "beauty_sasrec_control_ce_bs128_lr1em3_max200_seed42",
        "stability": "beauty_sasrec_stability_summary_seed42",
        "val_n": 2856,
        "test_n": 2966,
        "epoch": 77,
        "eval_protocol": "recent_bucketset_r1",
    },
    {
        "name": "Amazon_Sports",
        "model": "sports_sasrec_control_ce_bs128_lr1em3_max200_seed42",
        "stability": "sports_sasrec_stability_summary_seed42",
        "val_n": 4082,
        "test_n": 4659,
        "epoch": 8,
        "eval_protocol": "strict_diagnostic_a",
    },
    {
        "name": "Amazon_Toys",
        "model": "toys_sasrec_control_ce_bs128_lr1em3_max200_seed42",
        "stability": "toys_sasrec_stability_summary_seed42",
        "val_n": 1939,
        "test_n": 2547,
        "epoch": 72,
        "eval_protocol": "strict_diagnostic_a",
    },
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    require(path.is_file(), f"缺少文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_metrics(data, split):
    values = data[split]

    # ML1M / Beauty: test.overall
    # Sports / Toys: test
    if "overall" in values:
        values = values["overall"]

    if "NDCG@10" in values:
        return {
            "n": int(values["N"]),
            "ndcg10": float(values["NDCG@10"]),
            "hr10": float(values["HR@10"]),
            "hr20": float(values["HR@20"]),
            "mean_rank": float(values["MeanRank"]),
        }

    return {
        "n": int(values["samples"]),
        "ndcg10": float(values["ndcg10"]),
        "hr10": float(values["hr10"]),
        "hr20": float(values["hr20"]),
        "mean_rank": float(values["mean_rank"]),
    }


accuracy_rows = []
stability_rows = []
combined_rows = []
sources = []

for spec in DATASETS:
    dataset = spec["name"]

    metrics_path = (
        INPUT / spec["model"] / "metrics.json"
    )
    stability_path = (
        INPUT / spec["stability"] / "stability_summary.json"
    )

    metrics = load_json(metrics_path)
    stability = load_json(stability_path)

    # --------------------------------------------------------
    # 1. 模型与数据一致性检查
    # --------------------------------------------------------

    require(
        metrics["dataset"] == dataset,
        f"{dataset}: metrics 数据集不一致",
    )

    require(
        int(metrics["seed"]) == 42,
        f"{dataset}: 训练随机种子不一致",
    )

    require(
        int(metrics["best_epoch"]) == spec["epoch"],
        f"{dataset}: 最佳 Epoch 不一致",
    )

    val = normalize_metrics(metrics, "validation")
    test = normalize_metrics(metrics, "test")

    require(
        val["n"] == spec["val_n"],
        f"{dataset}: 验证样本数不一致",
    )

    require(
        test["n"] == spec["test_n"],
        f"{dataset}: 测试样本数不一致",
    )

    for split_name, result in [
        ("validation", val),
        ("test", test),
    ]:
        for metric in ["ndcg10", "hr10", "hr20"]:
            require(
                0 <= result[metric] <= 1,
                f"{dataset}: {split_name} {metric} 超出范围",
            )

        require(
            result["mean_rank"] >= 1,
            f"{dataset}: {split_name} MeanRank 异常",
        )

    # --------------------------------------------------------
    # 2. 稳定性结果一致性检查
    # --------------------------------------------------------

    require(
        stability["dataset"] == dataset,
        f"{dataset}: 稳定性数据集不一致",
    )

    require(
        stability["model"] == "SASRecControl",
        f"{dataset}: 稳定性模型不一致",
    )

    require(
        int(stability["model_seed"]) == 42,
        f"{dataset}: 稳定性模型种子不一致",
    )

    require(
        int(stability["checkpoint_best_epoch"])
        == spec["epoch"],
        f"{dataset}: 稳定性 checkpoint 不一致",
    )

    require(
        int(stability["test_samples"]) == test["n"],
        f"{dataset}: 稳定性测试样本数不一致",
    )

    require(
        stability["permutation_seeds"] == SEEDS,
        f"{dataset}: 排列种子不一致",
    )

    require(
        stability["overlap10_definition"]
        == "intersection_of_top10_sets_divided_by_10",
        f"{dataset}: Top-10 Overlap 定义不一致",
    )

    require(
        stability["rank_std_definition"]
        == "population_std_across_five_permutations_ddof0",
        f"{dataset}: Rank Std 定义不一致",
    )

    require(
        len(stability["per_seed"]) == 5,
        f"{dataset}: 排列实验数量异常",
    )

    require(
        [int(x["seed"]) for x in stability["per_seed"]]
        == SEEDS,
        f"{dataset}: 排列结果与种子不一致",
    )

    for result in stability["per_seed"]:
        require(
            int(result["samples"]) == test["n"],
            f"{dataset}: 排列样本数量异常",
        )

    s = stability["overall"]

    changed = int(s["samples_changed_at_least_once"])
    unchanged = int(s["samples_never_changed"])

    require(
        changed + unchanged == test["n"],
        f"{dataset}: 样本数量无法对齐",
    )

    total_pairs = int(s["total_pairwise_comparisons"])
    distinct_pairs = int(
        s["distinct_sequence_pairwise_comparisons"]
    )

    require(
        total_pairs == test["n"] * 10,
        f"{dataset}: 两两排列对数不一致",
    )

    require(
        0 < distinct_pairs <= total_pairs,
        f"{dataset}: 不同序列排列对数异常",
    )

    for key in [
        "mean_pairwise_overlap10_all",
        "mean_pairwise_overlap10_distinct_sequences",
        "pairwise_rank_change_rate_distinct_sequences",
    ]:
        require(
            0 <= float(s[key]) <= 1,
            f"{dataset}: {key} 超出范围",
        )

    require(
        float(s["mean_rank_std_5perm"]) >= 0,
        f"{dataset}: Rank Std 异常",
    )

    # --------------------------------------------------------
    # 3. 准确率结果
    # --------------------------------------------------------

    accuracy = {
        "dataset": dataset,
        "model": "SASRec_Control",
        "seed": 42,
        "best_epoch": spec["epoch"],
        "validation_n": val["n"],
        "test_n": test["n"],
        "val_ndcg10": val["ndcg10"],
        "test_ndcg10": test["ndcg10"],
        "test_hr10": test["hr10"],
        "test_hr20": test["hr20"],
        "test_mean_rank": test["mean_rank"],
        "evaluation_protocol": spec["eval_protocol"],
    }

    # --------------------------------------------------------
    # 4. 稳定性结果
    # --------------------------------------------------------

    stable = {
        "dataset": dataset,
        "model": "SASRec_Control",
        "seed": 42,
        "test_n": test["n"],
        "changed_samples_any": changed,
        "unchanged_samples": unchanged,
        "top10_overlap_all": float(
            s["mean_pairwise_overlap10_all"]
        ),
        "top10_overlap_distinct": float(
            s["mean_pairwise_overlap10_distinct_sequences"]
        ),
        "rank_change_rate_distinct": float(
            s["pairwise_rank_change_rate_distinct_sequences"]
        ),
        "abs_rank_diff_distinct": float(
            s["mean_pairwise_abs_rank_diff_distinct_sequences"]
        ),
        "mean_rank_std_5perm": float(
            s["mean_rank_std_5perm"]
        ),
        "total_pairwise_comparisons": total_pairs,
        "distinct_sequence_pairwise_comparisons":
            distinct_pairs,
        "evaluation_protocol": spec["eval_protocol"],
    }

    accuracy_rows.append(accuracy)
    stability_rows.append(stable)

    combined_rows.append({
        **accuracy,
        **{
            k: v
            for k, v in stable.items()
            if k not in accuracy
        },
    })

    sources.append({
        "dataset": dataset,
        "metrics_path": str(metrics_path),
        "stability_path": str(stability_path),
    })

    print(
        f"{dataset}: "
        f"N={test['n']} "
        f"Epoch={spec['epoch']} "
        f"NDCG={test['ndcg10']:.6f} "
        f"DistinctOverlap="
        f"{stable['top10_overlap_distinct']:.6f}"
    )


# ============================================================
# 5. 全部检查通过后才创建输出
# ============================================================

require(
    not OUTPUT.exists(),
    f"输出目录已存在，禁止覆盖: {OUTPUT}",
)

accuracy_df = pd.DataFrame(accuracy_rows)
stability_df = pd.DataFrame(stability_rows)
combined_df = pd.DataFrame(combined_rows)

OUTPUT.mkdir(
    parents=True,
    exist_ok=False,
)

accuracy_df.to_csv(
    OUTPUT / "accuracy_table.csv",
    index=False,
    encoding="utf-8-sig",
)

stability_df.to_csv(
    OUTPUT / "stability_table.csv",
    index=False,
    encoding="utf-8-sig",
)

combined_df.to_csv(
    OUTPUT / "full_baseline_table.csv",
    index=False,
    encoding="utf-8-sig",
)

manifest = {
    "step": "Step38.18",
    "model": "SASRecControl",
    "training_seed": 42,
    "training_batch_size": 128,
    "permutation_seeds": SEEDS,
    "method": "fixed_checkpoint_last_bucket_only",
    "scope": "four_dataset_single_seed_control_baseline",
    "sources": sources,
    "notes": [
        "Training protocol was established in earlier Step38 runs.",
        "This script verifies metric identity and sample consistency.",
        "Evaluation sample construction differs by dataset.",
        "Absolute ranking metrics are not directly comparable across item universes.",
        "These are seed42 results, not multi-seed estimates.",
    ],
}

(OUTPUT / "manifest.json").write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# 6. 打印正式汇总
# ============================================================

print()
print("=" * 72)
print("STEP38.18 FOUR-DATASET ACCURACY")
print("=" * 72)

print(
    accuracy_df[
        [
            "dataset",
            "best_epoch",
            "test_n",
            "test_ndcg10",
            "test_hr10",
            "test_hr20",
            "test_mean_rank",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print("=" * 72)
print("STEP38.18 FOUR-DATASET STABILITY")
print("=" * 72)

print(
    stability_df[
        [
            "dataset",
            "changed_samples_any",
            "top10_overlap_all",
            "top10_overlap_distinct",
            "rank_change_rate_distinct",
            "abs_rank_diff_distinct",
            "mean_rank_std_5perm",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print("STEP38.18 FOUR-DATASET BASELINE PASS")
print("Saved:", OUTPUT)
