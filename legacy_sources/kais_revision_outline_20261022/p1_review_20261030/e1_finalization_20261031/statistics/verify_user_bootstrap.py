"""Independently reproduce one ensemble bootstrap cell using count weights."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
METRICS = ("ndcg10", "hr10", "hr20", "recall50", "mrr", "mrr10", "mean_rank")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def vector(rank):
    return np.array([1 / math.log2(rank + 1) if rank <= 10 else 0, int(rank <= 10), int(rank <= 20),
                     int(rank <= 50), 1 / rank, 1 / rank if rank <= 10 else 0, rank], dtype=np.float64)


def main():
    all_rows = read(OUT / "paired_per_seed_user_bootstrap.csv")
    checks = 0
    keys = set()
    for row in all_rows:
        k = tuple(row[v] for v in ("dataset", "method", "training_seed", "population", "metric"))
        assert k not in keys
        keys.add(k)
        assert int(row["bootstrap_replicates"]) == 2000 and int(row["master_rng_seed"]) == 20261027
        assert float(row["ci95_lower"]) <= float(row["ci95_upper"])
        expected = int.from_bytes(hashlib.sha256(f"20261027|{row['dataset']}|{row['method']}|{row['training_seed']}|{row['population']}".encode("ascii")).digest()[:8], "big")
        assert int(row["rng_seed"]) == expected
        if row["method"] == "Control":
            assert float(row["mean_delta"]) == float(row["ci95_lower"]) == float(row["ci95_upper"]) == 0
        checks += 5
    assert len(all_rows) == 924
    chosen = [r for r in all_rows if (r["dataset"], r["method"], r["training_seed"], r["population"]) == ("beauty", "Ensemble", "2026", "affected")]
    labels = read(ROOT / "g2_lbsp_amazon_stability_20261025/snapshot/outputs/beauty_seed42/panel_labels.csv")
    assert len({r["user_id"] for r in labels}) == len(labels)
    ids = [r["sample_id"] for r in labels if int(r["affected"])]
    control = {r["sample_id"]: r for r in read(ROOT / "g2_lbsp_amazon_stability_20261025/snapshot/control_reference/outputs/beauty_seed2026/predictions_original.csv")}
    groups = [{r["sample_id"]: r for r in read(ROOT / "g2_x3_inference_results_20261027_e635848880/snapshot/outputs/ensemble_beauty_seed2026" / f"predictions_group{g}.csv")} for g in range(1, 6)]
    delta = []
    for sid in ids:
        c = vector(int(control[sid]["target_rank"]))
        members = [vector(int(group[sid]["rank"])) for group in groups]
        delta.append(np.array([math.fsum(v[i] for v in members) / 5 for i in range(7)]) - c)
    delta = np.asarray(delta, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(int(chosen[0]["rng_seed"])))
    distribution = np.empty((2000, 7))
    for i in range(2000):
        indices = generator.integers(0, len(ids), size=len(ids))
        # Independent implementation: count sampled users, then weighted sum.
        counts = np.bincount(indices, minlength=len(ids))
        distribution[i] = counts @ delta / len(ids)
    lo, hi = np.quantile(distribution, (0.025, 0.975), axis=0, method="linear")
    numeric = 0
    for i, metric in enumerate(METRICS):
        row = next(r for r in chosen if r["metric"] == metric)
        for a, b in ((delta[:, i].mean(), float(row["mean_delta"])), (lo[i], float(row["ci95_lower"])), (hi[i], float(row["ci95_upper"]))):
            assert math.isclose(a, b, abs_tol=3e-12, rel_tol=1e-10), (metric, a, b)
            numeric += 1
    report = {"status": "PASS", "schema_rng_zero_effect_checks": checks,
              "independent_reproduced_cell": "beauty/Ensemble/2026/affected", "cell_users": len(ids),
              "replicates": 2000, "independent_numeric_reproductions": numeric,
              "method": "Pure-math per-user rank metrics, then five-group average; bootstrap via user multiplicity count weights rather than indexed per-user averaging.",
              "raw_interactions_checkpoints_or_original_research_code_accessed": False}
    (OUT / "independent_bootstrap_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
