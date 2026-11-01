import json
from pathlib import Path
import numpy as np


BASE = Path(
    "/root/autodl-tmp/time_uncertainty_rec/"
    "step38_kais_retrain/outputs"
)

OUTPUT = BASE / "ml1m_r1_three_seed_summary.json"


dirs = [
    BASE / "ml1m_r1_meanpool_ce_bs128_lr1em3_max200_seed42",
    BASE / "ml1m_r1_meanpool_ce_bs128_lr1em3_max200_seed2026",
    BASE / "ml1m_r1_meanpool_ce_bs128_lr1em3_max200_seed3407",
]


results = []

for d in dirs:
    path = d / "metrics.json"

    if not path.exists():
        raise RuntimeError(
            f"Missing: {path}"
        )

    with open(path, "r") as f:
        metrics = json.load(f)

    results.append(metrics)


keys = [
    "ndcg10",
    "hr10",
    "hr20",
    "mean_rank",
]


summary = {}

for k in keys:
    values = [
        r["test"][k]
        for r in results
    ]

    summary[k] = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": values,
    }


output = {
    "method": "R1_MeanPooling",
    "seeds": [42, 2026, 3407],
    "summary": summary,
}


with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("=" * 70)
print("STEP38.27 ML1M R1 THREE SEED SUMMARY")
print("=" * 70)

for k,v in summary.items():
    print(
        k,
        ":",
        f"{v['mean']:.6f} ± {v['std']:.6f}"
    )

print()
print("Saved:", OUTPUT)
