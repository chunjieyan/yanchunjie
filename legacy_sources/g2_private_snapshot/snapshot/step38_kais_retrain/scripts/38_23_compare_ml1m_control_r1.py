import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "step38_kais_retrain/outputs"

CONTROL = BASE / "ml1m_sasrec_stability_seed42"
R1 = BASE / "ml1m_r1_stability_seed42"

CONTROL_METRICS = (
    BASE
    / "ml1m_sasrec_control_ce_bs128_lr1em3_max200_seed42"
    / "metrics.json"
)

R1_METRICS = (
    BASE
    / "ml1m_r1_meanpool_ce_bs128_lr1em3_max200_seed42"
    / "metrics.json"
)

CONTROL_SUMMARY = (
    BASE
    / "ml1m_sasrec_stability_summary_seed42"
    / "stability_summary.json"
)

R1_MANIFEST = R1 / "manifest.json"

OUTPUT = BASE / "step38_23_ml1m_control_vs_r1_seed42"

N = 945
SEEDS = [101, 202, 303, 404, 505]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load_json(path):
    require(path.is_file(), f"缺少文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path):
    require(path.is_file(), f"缺少预测: {path}")

    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
        },
    )

    require(len(df) == N, f"样本数异常: {path}")
    require(df["sample_id"].is_unique, "重复 sample_id")

    return df.set_index("sample_id")


def ndcg_per_sample(ranks):
    return np.where(
        ranks <= 10,
        1.0 / np.log2(ranks + 1),
        0.0,
    )


def top10_lists(df):
    return [
        tuple(json.loads(x))
        for x in df["top10_item_idx"]
    ]


require(
    not OUTPUT.exists(),
    f"结果目录已存在，禁止覆盖: {OUTPUT}",
)

control_metrics = load_json(CONTROL_METRICS)
r1_metrics = load_json(R1_METRICS)
control_stability = load_json(CONTROL_SUMMARY)
r1_manifest = load_json(R1_MANIFEST)

require(
    control_metrics["dataset"] == "MovieLens-1M"
    and r1_metrics["dataset"] == "MovieLens-1M",
    "数据集不一致",
)

require(
    control_metrics["seed"] == 42
    and r1_metrics["seed"] == 42,
    "训练种子不一致",
)

require(
    control_metrics["best_epoch"] == 66
    and r1_metrics["best_epoch"] == 33,
    "最佳 Epoch 不一致",
)

require(
    r1_manifest["permutation_seeds"] == SEEDS,
    "R1 排列种子不一致",
)

require(
    control_stability["permutation_seeds"] == SEEDS,
    "Control 排列种子不一致",
)

ctrl = load_predictions(CONTROL / "predictions_original.csv")
r1 = load_predictions(R1 / "predictions_original.csv")

require(
    set(ctrl.index) == set(r1.index),
    "两个模型的测试样本集合不同",
)

r1 = r1.loc[ctrl.index]

require(
    (ctrl["target_item"] == r1["target_item"]).all(),
    "两个模型的目标物品不一致",
)

control_rank = ctrl["target_rank"].to_numpy(dtype=np.int64)
r1_rank = r1["target_rank"].to_numpy(dtype=np.int64)

require(
    ((control_rank >= 1) & (control_rank <= 3416)).all()
    and ((r1_rank >= 1) & (r1_rank <= 3416)).all(),
    "目标排名超出候选物品范围",
)

control_ndcg = ndcg_per_sample(control_rank)
r1_ndcg = ndcg_per_sample(r1_rank)

control_test = control_metrics["test"]["overall"]
r1_test = r1_metrics["test"]

require(
    abs(
        control_ndcg.mean()
        - control_test["NDCG@10"]
    ) < 1e-10,
    "Control NDCG 不能复现",
)

require(
    abs(
        r1_ndcg.mean()
        - r1_test["ndcg10"]
    ) < 1e-10,
    "R1 NDCG 不能复现",
)


# ============================================================
# 五组排列预测逐样本核验
# ============================================================

base_r1_top10 = top10_lists(r1)

for seed, control_seed_result, r1_seed_result in zip(
    SEEDS,
    control_stability["per_seed"],
    r1_manifest["per_seed"],
):
    require(
        control_seed_result["seed"] == seed
        and r1_seed_result["seed"] == seed,
        "排列种子顺序不一致",
    )

    cp = load_predictions(
        CONTROL / f"predictions_perm_seed{seed}.csv"
    ).loc[ctrl.index]

    rp = load_predictions(
        R1 / f"predictions_perm_seed{seed}.csv"
    ).loc[ctrl.index]

    require(
        (cp["target_item"] == ctrl["target_item"]).all()
        and (rp["target_item"] == r1["target_item"]).all(),
        f"Seed {seed}: 目标物品不一致",
    )

    changed_control = int(
        np.count_nonzero(
            cp["target_rank"].to_numpy(dtype=np.int64)
            != control_rank
        )
    )

    require(
        changed_control
        == int(control_seed_result["changed_ranks"]),
        f"Seed {seed}: Control 排名变化数不一致",
    )

    require(
        np.array_equal(
            rp["target_rank"].to_numpy(dtype=np.int64),
            r1_rank,
        ),
        f"Seed {seed}: R1 排名发生变化",
    )

    require(
        top10_lists(rp) == base_r1_top10,
        f"Seed {seed}: R1 Top10 发生变化",
    )

    require(
        r1_seed_result["max_abs_logit_diff"] == 0,
        f"Seed {seed}: R1 Logits 不完全一致",
    )

    require(
        int(control_seed_result["changed_sequences"])
        == int(r1_seed_result["changed_sequences"]),
        f"Seed {seed}: 扰动样本数不一致",
    )

print("FIVE-SEED PREDICTION ALIGNMENT PASS")


# ============================================================
# 测试准确率的逐样本配对比较
# ============================================================

deltas = r1_ndcg - control_ndcg

relative_gain = (
    (r1_ndcg.mean() / control_ndcg.mean() - 1) * 100
)

# 同一批945个样本配对重采样
rng = np.random.default_rng(2026)

bootstrap_deltas = []

for _ in range(20):
    indices = rng.integers(
        0,
        N,
        size=(500, N),
    )

    bootstrap_deltas.append(
        deltas[indices].mean(axis=1)
    )

bootstrap_deltas = np.concatenate(bootstrap_deltas)

ci_low, ci_high = np.quantile(
    bootstrap_deltas,
    [0.025, 0.975],
)


# ============================================================
# 汇总
# ============================================================

overlap_control = float(
    control_stability["overall"][
        "mean_pairwise_overlap10_distinct_sequences"
    ]
)

comparison = pd.DataFrame([
    {
        "method": "SASRec_Control",
        "training_seed": 42,
        "best_epoch": 66,
        "test_n": N,
        "test_ndcg10": float(control_ndcg.mean()),
        "test_hr10": float((control_rank <= 10).mean()),
        "test_hr20": float((control_rank <= 20).mean()),
        "test_mean_rank": float(control_rank.mean()),
        "distinct_top10_overlap": overlap_control,
        "distinct_rank_change_rate": float(
            control_stability["overall"][
                "pairwise_rank_change_rate_distinct_sequences"
            ]
        ),
    },
    {
        "method": "R1_MeanPooling",
        "training_seed": 42,
        "best_epoch": 33,
        "test_n": N,
        "test_ndcg10": float(r1_ndcg.mean()),
        "test_hr10": float((r1_rank <= 10).mean()),
        "test_hr20": float((r1_rank <= 20).mean()),
        "test_mean_rank": float(r1_rank.mean()),
        "distinct_top10_overlap": 1.0,
        "distinct_rank_change_rate": 0.0,
    },
])

paired_samples = pd.DataFrame({
    "sample_id": ctrl.index.to_numpy(),
    "target_item": ctrl["target_item"].to_numpy(),
    "control_rank": control_rank,
    "r1_rank": r1_rank,
    "control_ndcg10": control_ndcg,
    "r1_ndcg10": r1_ndcg,
    "delta_ndcg10": deltas,
})

bootstrap_summary = {
    "dataset": "MovieLens-1M",
    "training_seed": 42,
    "test_samples": N,
    "bootstrap_seed": 2026,
    "bootstrap_repeats": 10000,
    "resampling_unit": "paired_test_sample",
    "delta_ndcg10": float(deltas.mean()),
    "relative_gain_percent": float(relative_gain),
    "ci95_lower": float(ci_low),
    "ci95_upper": float(ci_high),
    "ci95_excludes_zero": bool(
        ci_low > 0 or ci_high < 0
    ),
    "limitations": (
        "Confidence interval describes test-sample "
        "resampling conditional on training seed42; "
        "it does not quantify training-seed variability."
    ),
}

OUTPUT.mkdir(parents=True, exist_ok=False)

comparison.to_csv(
    OUTPUT / "comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

paired_samples.to_csv(
    OUTPUT / "paired_sample_deltas.csv",
    index=False,
    encoding="utf-8-sig",
)

(OUTPUT / "bootstrap_summary.json").write_text(
    json.dumps(
        bootstrap_summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("===== STEP38.23 ML1M CONTROL VS R1 =====")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print("===== PAIRED BOOTSTRAP =====")
print("Delta NDCG@10:", f"{deltas.mean():.9f}")
print("Relative gain:", f"{relative_gain:.2f}%")
print(
    "95% CI:",
    f"[{ci_low:.9f}, {ci_high:.9f}]",
)
print("CI excludes zero:", bootstrap_summary["ci95_excludes_zero"])

print()
print("STEP38.23 PAIRED COMPARISON PASS")
print("Saved:", OUTPUT)
