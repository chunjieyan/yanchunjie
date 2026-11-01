"""Independent identities, weighting and interval checks on the unified outputs."""
import csv
from collections import defaultdict
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
COUNTS = defaultdict(int)
TC = 4.302652729749464
SEEDS = (42, 2026, 3407)


def read(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check(value, label):
    COUNTS["assertions"] += 1
    if not value:
        raise AssertionError(label)


def number(row, key):
    return float(row[key]) if row[key] else None


def near(a, b, label):
    COUNTS["numeric_identities"] += 1
    check(math.isclose(a, b, rel_tol=1e-10, abs_tol=3e-12), label)


def key(row):
    return tuple(row[x] for x in ("dataset", "method", "population", "metric"))


def main():
    seed = read("per_seed_metrics.csv")
    summary = read("method_summary.csv")
    delta = read("paired_differences_per_seed.csv")
    ds = read("paired_difference_summary.csv")
    seed_index = {key(r) + (int(r["training_seed"]),): r for r in seed}
    summary_index = {key(r): r for r in summary}
    check(len(seed_index) == len(seed) == 5832, "unique per-seed keys and full coverage")
    check(len(summary_index) == len(summary) == 1944, "unique summaries and coverage")
    check(not any(r["method"] == "TiSASRec" and r["dataset"] in ("sports", "toys") for r in seed), "no invented TiSASRec datasets")
    for rows, value_name in ((seed, "value"), (delta, "difference")):
        group = defaultdict(list)
        for row in rows:
            group[key(row)].append(row)
        target = summary_index if value_name == "value" else {key(r): r for r in ds}
        for k, values in group.items():
            check(tuple(int(v["training_seed"]) for v in values) == SEEDS, "three chosen seeds in order")
            r = target[k]
            nums = [number(v, value_name) for v in values]
            if int(r["n"]) == 0:
                check(all(v is None for v in nums) and all(not r[v] for v in ("mean", "sample_sd", "ci95_lower", "ci95_upper")), "empty strata remain missing")
                continue
            mean = sum(nums) / 3
            sd = math.sqrt(sum((v - mean) ** 2 for v in nums) / 2)
            near(float(r["mean"]), mean, "independent seed mean")
            near(float(r["sample_sd"]), sd, "independent sample SD denominator two")
            near(float(r["ci95_lower"]), mean - TC * sd / math.sqrt(3), "lower CI")
            near(float(r["ci95_upper"]), mean + TC * sd / math.sqrt(3), "upper CI")
    for row in delta:
        k = key(row)
        s = int(row["training_seed"])
        m = seed_index[k + (s,)]
        c = seed_index[(k[0], "Control", k[2], k[3], s)]
        if row["difference"]:
            near(float(row["difference"]), float(m["value"]) - float(c["value"]), "same-training-seed Control pairing")
    for row in seed:
        n = int(row["n"])
        value = number(row, "value")
        check((n == 0) == (value is None), "missing estimates iff no users")
        if value is not None:
            check(value >= 0 and (row["metric"] == "mean_rank" or value <= 1 + 1e-12), "metric domain")
        pop = row["population"]
        strata = {"full": ("pooled", "singleton"), "pooled": ("bucket2", "bucket3", "bucket4", "bucket5plus"),
                  "affected": ("affected_bucket2", "affected_bucket3", "affected_bucket4", "affected_bucket5plus")}.get(pop)
        if strata:
            parts = [seed_index[(row["dataset"], row["method"], p, row["metric"], int(row["training_seed"]))] for p in strata]
            check(sum(int(p["n"]) for p in parts) == n, "exhaustive strata counts")
            weighted = sum(float(p["value"]) * int(p["n"]) for p in parts if int(p["n"])) / n
            near(value, weighted, "weighted stratum reconstruction")
    ensemble = read("ensemble_group_accuracy.csv")
    groups = defaultdict(list)
    for row in ensemble:
        groups[key(row) + (int(row["training_seed"]),)].append(row)
    for k, vals in groups.items():
        check(tuple(int(v["group"]) for v in vals) == (1, 2, 3, 4, 5), "exact five final groups within seed")
        if vals[0]["value"]:
            near(sum(float(v["value"]) for v in vals) / 5, float(seed_index[k]["value"]), "ensemble group metrics averaged within training seed")
    pairs = read("seed_reference_pairs.csv")
    ref = read("seed_reference_summary.csv")
    g = defaultdict(list)
    for row in pairs:
        g[(row["dataset"], row["method"], row["population"])].append(row)
    check(len(pairs) == 648 and len(ref) == 216, "all seed-reference populations")
    for row in ref:
        vals = g[(row["dataset"], row["method"], row["population"])]
        check([(int(r["seed_a"]), int(r["seed_b"])) for r in vals] == [(42, 2026), (42, 3407), (2026, 3407)], "all three dependent seed pairs")
        if row["mean_overlap10"]:
            near(float(row["mean_overlap10"]), sum(float(v["overlap10"]) for v in vals) / 3, "seed-pair mean reference")
    baselines = read("analytic_baselines.csv")
    check(len(baselines) == 784, "all analytic baseline groups")
    check(all(r["n_training_seeds"] == "0" and not any(r[k] for k in ("sample_sd", "ci95_lower", "ci95_upper")) for r in baselines), "no fabricated analytic training-seed variance")
    report = {"status": "PASS", "checks": dict(COUNTS),
              "coverage": "Independent CSV arithmetic, paired identities, nonempty/missing strata, ensemble within-seed aggregation, weighted population reconstruction, all three seed pairs, no invented methods/seed variation.",
              "raw_data_or_research_code_executed": False}
    (OUT / "independent_output_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
