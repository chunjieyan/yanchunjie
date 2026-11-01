import json
from pathlib import Path
import numpy as np


BASE = Path(
    "/root/autodl-tmp/time_uncertainty_rec/"
    "step38_kais_retrain/outputs"
)


dirs = [
    BASE / "ml1m_sasrec_control_ce_bs128_lr1em3_max200_seed42",
    BASE / "ml1m_sasrec_control_ce_bs128_lr1em3_max200_seed2026",
    BASE / "ml1m_sasrec_control_ce_bs128_lr1em3_max200_seed3407",
]


OUTPUT = BASE / "ml1m_control_three_seed_summary.json"


results = []

for d in dirs:
    path = d / "metrics.json"

    if not path.exists():
        raise RuntimeError(f"Missing {path}")

    with open(path, "r") as f:
        results.append(json.load(f))


metrics = [
    "NDCG@10",
    "HR@10",
    "HR@20",
    "MeanRank",
]


summary = {}

for m in metrics:

    values = [
        x["test"]["overall"][m]
        for x in results
    ]

    summary[m] = {
        "values": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


output = {
    "method": "SASRec_Control",
    "seeds": [42,2026,3407],
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


print("="*70)
print("STEP38.29 ML1M CONTROL THREE SEED SUMMARY")
print("="*70)

for k,v in summary.items():
    print(
        k,
        ":",
        f"{v['mean']:.6f} ± {v['std']:.6f}"
    )

print()
print("Saved:", OUTPUT)
