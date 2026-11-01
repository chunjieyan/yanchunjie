import json
from pathlib import Path


BASE = Path(
    "/root/autodl-tmp/time_uncertainty_rec/"
    "step38_kais_retrain/outputs"
)


R1_FILE = BASE / "ml1m_r1_three_seed_summary.json"
CTRL_FILE = BASE / "ml1m_control_three_seed_summary.json"


OUTPUT = BASE / "ml1m_control_vs_r1_three_seed_final.json"


with open(R1_FILE, "r") as f:
    r1 = json.load(f)

with open(CTRL_FILE, "r") as f:
    ctrl = json.load(f)


def get_mean_std(data, key):

    mapping = {
        "ndcg10": ["ndcg10", "NDCG@10"],
        "hr10": ["hr10", "HR@10"],
        "hr20": ["hr20", "HR@20"],
        "mean_rank": ["mean_rank", "MeanRank"],
    }

    for name in mapping[key]:
        if name in data["summary"]:
            return (
                data["summary"][name]["mean"],
                data["summary"][name]["std"]
            )

    raise KeyError(
        f"Metric {key} not found. Available: {list(data['summary'].keys())}"
    )


metrics = [
    "ndcg10",
    "hr10",
    "hr20",
    "mean_rank",
]


result = {
    "dataset": "MovieLens-1M",
    "comparison": "R1_MeanPooling_vs_SASRec_Control",
    "R1": {},
    "Control": {},
    "relative_gain": {},
}


for m in metrics:

    r1_mean, r1_std = get_mean_std(
        r1,
        m
    )

    ctrl_mean, ctrl_std = get_mean_std(
        ctrl,
        m
    )

    result["R1"][m] = {
        "mean": r1_mean,
        "std": r1_std
    }

    result["Control"][m] = {
        "mean": ctrl_mean,
        "std": ctrl_std
    }

    if m != "mean_rank":
        result["relative_gain"][m] = (
            (r1_mean - ctrl_mean)
            / ctrl_mean
        )


with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False
    )


print("="*70)
print("STEP38.30 ML1M CONTROL VS R1 FINAL")
print("="*70)

for m in metrics:
    r = result["R1"][m]
    c = result["Control"][m]

    print(
        m,
        "\n R1:",
        f"{r['mean']:.6f} ± {r['std']:.6f}",
        "\n Control:",
        f"{c['mean']:.6f} ± {c['std']:.6f}"
    )

print("\nRelative Gain:")

for k,v in result["relative_gain"].items():
    print(
        k,
        ":",
        f"{v*100:.2f}%"
    )

print()
print("Saved:", OUTPUT)
