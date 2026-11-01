from pathlib import Path
import json
import pandas as pd


ROOT = Path(
    r"D:\time_uncertainty_rec"
)

REP = (
    ROOT
    / "reproducibility"
    / "step35_5"
)

REQUIRED = [
    REP / "last_bucket_only_permutation_detail.csv",
    REP / "last_bucket_only_permutation_manifest.json",
    REP / "last_bucket_only_per_sample_stability.csv",
    REP / "last_bucket_only_inference_integrity.csv",
    REP / "last_bucket_only_stability_summary.csv",
    REP / "last_bucket_only_stability_manifest.json",
    REP / "last_bucket_only_bootstrap_ci.csv",
    REP / "last_bucket_only_paper_main_table.csv",
    REP / "last_bucket_only_paper_main_table.md",
    REP / "last_bucket_only_bootstrap_manifest.json",
]


print("=" * 100)
print("35.5-E Evidence Freeze")
print("=" * 100)


missing = [
    str(p)
    for p in REQUIRED
    if not p.exists()
]


if missing:

    print("STATUS: FAIL")
    print("Missing:")

    for x in missing:
        print(" ", x)

    raise SystemExit(1)


stability_manifest = json.loads(
    (
        REP
        / "last_bucket_only_stability_manifest.json"
    ).read_text(
        encoding="utf-8"
    )
)


bootstrap_manifest = json.loads(
    (
        REP
        / "last_bucket_only_bootstrap_manifest.json"
    ).read_text(
        encoding="utf-8"
    )
)


if stability_manifest.get("status") != "PASS":

    raise RuntimeError(
        "35.5-C is not PASS"
    )


if bootstrap_manifest.get("status") != "PASS":

    raise RuntimeError(
        "35.5-D is not PASS"
    )


paper = pd.read_csv(
    REP
    / "last_bucket_only_paper_main_table.csv"
)


required_datasets = {
    "Amazon_Beauty",
    "MovieLens-1M",
}


if set(
    paper["dataset"]
) != required_datasets:

    raise RuntimeError(
        "Unexpected datasets in paper table"
    )


expected_n = {
    "Amazon_Beauty": 922,
    "MovieLens-1M": 163,
}


for _, row in paper.iterrows():

    dataset = row["dataset"]

    if int(
        row["n_primary"]
    ) != expected_n[
        dataset
    ]:

        raise RuntimeError(
            f"{dataset}: unexpected primary N"
        )


summary = """# Step 35.5 — Last-Bucket-Only Fixed-Model Sensitivity

## Status

PASS

## Experimental isolation

The experiment fixes:

- the trained SASRec checkpoint (model seed 42);
- the test sample and target item;
- the history event set;
- the candidate / full-ranking item universe;
- all history events before the final tied timestamp bucket.

The only manipulated factor is the within-bucket order of the final
ambiguous timestamp-tied history group.

No model retraining is performed across the five permutation seeds.

## Integrity

- Five permutation seeds: 101, 202, 303, 404, 505.
- Identical input sequences produced identical predictions.
- Non-ambiguous final-bucket samples showed zero ordering-induced change.
- Source-file row order is not treated as ground-truth chronology.

## Primary group

Primary inference-sensitivity analysis is restricted to:

ambiguous_last_bucket == True
AND
actually_permuted == True

Primary sample counts:

- Amazon Beauty: 922
- MovieLens-1M: 163

## Main results

Amazon Beauty:

- Rank changed: 99.89% [99.39%, 99.98%]
- Mean RankStd: 460.81 [424.64, 495.79]
- Mean RankRange: 1078.36 [992.05, 1162.35]
- Mean absolute rank difference: 810.13 [745.74, 874.23]
- Top-10 changed: 100.00% [99.59%, 100.00%]
- Mean Top-10 overlap across distinct sequences:
  0.459 [0.449, 0.470]

MovieLens-1M:

- Rank changed: 98.77% [95.64%, 99.66%]
- Mean RankStd: 30.93 [26.61, 35.36]
- Mean RankRange: 71.75 [61.62, 82.20]
- Mean absolute rank difference: 56.00 [47.97, 64.34]
- Top-10 changed: 99.39% [96.61%, 99.89%]
- Mean Top-10 overlap across distinct sequences:
  0.778 [0.765, 0.791]

## Statistical protocol

Continuous sample-level statistics use:

- 10,000 percentile bootstrap repetitions;
- test sample as the resampling unit;
- 95% confidence intervals.

Binary proportions use:

- Wilson score 95% confidence intervals.

## Interpretation boundary

The result supports inference-time sensitivity to ordering within the
final timestamp-tied history bucket.

It does not establish that source-file row order is the true
chronological order.

Absolute rank variability should not be compared directly across
datasets because their item universes differ.

Cross-dataset discussion should emphasize dimensionless quantities such
as rank-changed ratio, Top-10-changed ratio, and Top-10 overlap.
"""


summary_path = (
    REP
    / "STEP35_5_EVIDENCE_FREEZE.md"
)


summary_path.write_text(
    summary,
    encoding="utf-8"
)


manifest = {
    "step": "35.5-E",
    "status": "PASS",
    "depends_on": [
        "35.5-A",
        "35.5-B",
        "35.5-C",
        "35.5-D",
    ],
    "datasets": [
        "Amazon_Beauty",
        "MovieLens-1M",
    ],
    "model_seed": 42,
    "permutation_seeds": [
        101,
        202,
        303,
        404,
        505,
    ],
    "primary_group":
        "ambiguous_last_bucket AND actually_permuted",
    "primary_sample_counts": expected_n,
    "claim_scope":
        "fixed-model last-tied-bucket inference sensitivity",
}


manifest_path = (
    REP
    / "step35_5_final_manifest.json"
)


manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8"
)


print()
print("Required artifacts:", len(REQUIRED))
print("35.5-C:", stability_manifest["status"])
print("35.5-D:", bootstrap_manifest["status"])
print("Evidence freeze:", summary_path)
print("Final manifest:", manifest_path)
print()
print("STATUS: PASS")