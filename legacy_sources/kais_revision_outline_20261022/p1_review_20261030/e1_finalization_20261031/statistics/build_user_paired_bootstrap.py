"""Conditional paired-user intervals; does not execute any model/research code."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_unified_statistics import ACCURACY, Builder, METHODS, SEEDS, TAGS


def per_user_metrics(pred, ids):
    ranks = np.asarray([pred[sid]["rank"] for sid in ids], dtype=np.float64)
    hit = ranks <= 10
    return np.column_stack((np.where(hit, 1 / np.log2(ranks + 1), 0), hit, ranks <= 20,
                            ranks <= 50, 1 / ranks, np.where(hit, 1 / ranks, 0), ranks))


def seed_for(tag, method, seed, pop):
    value = f"20261027|{tag}|{method}|{seed}|{pop}".encode("ascii")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def main(root):
    b = Builder(root)
    plan = b.read(b.out / "user_bootstrap_analysis_plan.json")
    b.check(plan["fixed_before_this_bootstrap_computation"] and plan["replicates"] == 2000, "fixed bootstrap analysis plan")
    b.prepare_labels()
    # Independent fixture for the vectorized metric mapping used by bootstrap.
    fixture = {str(i): {"rank": rank} for i, rank in enumerate((1, 10, 11, 50, 51))}
    arr = per_user_metrics(fixture, list(fixture))
    b.check(tuple(arr[:, 1]) == (1, 1, 0, 0, 0) and tuple(arr[:, 3]) == (1, 1, 1, 1, 0), "metric cutoffs")
    b.near(float(arr[0, 0]), 1.0, "first rank NDCG")
    b.near(float(arr[1, 0]), 1 / np.log2(11), "rank10 NDCG")
    b.near(float(arr[2, 4]), 1 / 11, "untruncated MRR")
    b.near(float(arr[2, 5]), 0.0, "MRR10 truncation")
    reference = {(r["dataset"], r["method"], int(r["training_seed"]), r["population"], r["metric"]): r
                 for r in b.read(b.out / "paired_differences_per_seed.csv")}
    rows, probes = [], []
    for tag in TAGS:
        labels = b.labels[tag]
        for seed in SEEDS:
            control_dir = b.run_output(tag, "Control", seed)
            control = b.predictions(control_dir / "predictions_original.csv", labels)
            for method in METHODS:
                if method == "TiSASRec" and tag not in ("ml1m", "beauty"):
                    continue
                if method == "Control":
                    primary = [control]
                elif method == "LBSP":
                    path = b.run_output(tag, "LBSP", seed)
                    primary = [b.predictions(path / "predictions_original.csv", labels)]
                elif method == "Shuffle":
                    path = b.run_output(tag, "Shuffle", seed)
                    primary = [b.predictions(path / "predictions_original.csv", labels)]
                elif method == "Ensemble":
                    path = b.run_output(tag, "Ensemble", seed)
                    primary = [b.predictions(path / f"predictions_group{g}.csv", labels) for g in range(1, 6)]
                else:
                    path = b.run_output(tag, "TiSASRec", seed)
                    primary = [b.predictions(path / "predictions_original.csv", labels)]
                populations = ("full", "affected", "pooled", "singleton") if method == "LBSP" else ("full", "affected")
                for pop in populations:
                    ids = b.population(labels, pop)
                    n = len(ids)
                    delta = sum(per_user_metrics(p, ids) for p in primary) / len(primary) - per_user_metrics(control, ids)
                    point = delta.mean(axis=0)
                    rng_seed = seed_for(tag, method, seed, pop)
                    generator = np.random.Generator(np.random.PCG64(rng_seed))
                    distribution = np.empty((2000, 7), dtype=np.float64)
                    zero = bool(np.all(delta == 0))
                    if zero:
                        distribution.fill(0)
                    else:
                        for start in range(0, 2000, 32):
                            end = min(2000, start + 32)
                            indices = generator.integers(0, n, size=(end - start, n))
                            distribution[start:end] = delta[indices].mean(axis=1)
                    lower, upper = np.quantile(distribution, (0.025, 0.975), axis=0, method="linear")
                    for i, metric in enumerate(ACCURACY):
                        expected = reference[(tag, method, seed, pop, metric)]
                        b.near(float(point[i]), float(expected["difference"]), "paired-user point matches separately aggregated paired seed difference")
                        b.check(np.isfinite(lower[i]) and np.isfinite(upper[i]) and lower[i] <= upper[i], "finite ordered user interval")
                        rows.append({"dataset": tag, "method": method, "training_seed": seed, "population": pop,
                            "n": n, "affected_n": len(b.population(labels, "affected")), "metric": metric,
                            "mean_delta": float(point[i]), "ci95_lower": float(lower[i]), "ci95_upper": float(upper[i]),
                            "bootstrap_replicates": 2000, "master_rng_seed": 20261027, "rng_seed": rng_seed,
                            "rng": "PCG64", "numpy_version": np.__version__, "interval_method": "percentile_linear_quantile",
                            "independent_unit": "paired_user_conditional_on_fixed_checkpoints",
                            "accuracy_view": "within_user_five_final_group_metric_mean" if method == "Ensemble" else "canonical",
                            "zero_distribution_exact": zero})
                    probes.append({"dataset": tag, "method": method, "training_seed": seed, "population": pop,
                                   "rng_seed": rng_seed, "mean_distribution_sha256": hashlib.sha256(distribution.tobytes(order="C")).hexdigest(),
                                   "first_three_bootstrap_means": distribution[:3].tolist()})
    b.write("paired_per_seed_user_bootstrap.csv", rows)
    report = {"status": "PASS", "checks": dict(b.counts), "interval_rows": len(rows), "bootstrap_cells": len(probes),
              "B": 2000, "rng": "PCG64", "master_rng_seed": 20261027, "numpy_version": np.__version__,
              "all_points_match_independently_aggregated_seed_differences": True,
              "scope": "Separate conditional paired-user intervals; never substitute for training-seed intervals or claim independent model repeats.",
              "raw_data_or_checkpoint_read_or_stat": False, "original_research_code_or_torch_executed": False,
              "probes": probes}
    (b.out / "paired_user_bootstrap_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (b.out / "bootstrap_source_manifest.json").write_text(json.dumps({"files": b.manifest,
        "bootstrap_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prediction_reader_sha256": hashlib.sha256(Path(__file__).with_name("build_unified_statistics.py").read_bytes()).hexdigest()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "probes"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:/time_uncertainty_rec"))
    args = parser.parse_args()
    main(args.root)
