from pathlib import Path
from itertools import combinations
import ast
import json

import numpy as np
import pandas as pd


ROOT = Path(
    r"D:\time_uncertainty_rec"
)

INPUT_ROOT = (
    ROOT
    / "processed"
    / "strict_last_bucket_only"
)

CANONICAL_ROOT = (
    ROOT
    / "processed"
    / "strict_diagnostic_a"
)

PRED_ROOT = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "step35_5_last_bucket_only"
    / "fixed_model_eval"
)

OUT_DIR = (
    ROOT
    / "reproducibility"
    / "step35_5"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DATASETS = [
    "Amazon_Beauty",
    "MovieLens-1M",
]

PERMUTATION_SEEDS = [
    101,
    202,
    303,
    404,
    505,
]


def parse_list(value):

    if isinstance(
        value,
        list,
    ):
        return value

    if pd.isna(value):
        return []

    text = str(
        value
    ).strip()

    try:
        return json.loads(
            text
        )

    except Exception:
        return ast.literal_eval(
            text
        )


def normalize_sequence(value):

    return tuple(
        int(x)
        for x in parse_list(
            value
        )
    )


def normalize_top10(value):

    values = parse_list(
        value
    )

    return tuple(
        int(x)
        for x in values[:10]
    )


def overlap10(a, b):

    if (
        len(a) == 0
        and
        len(b) == 0
    ):
        return 1.0

    return (
        len(
            set(a)
            &
            set(b)
        )
        /
        10.0
    )


print(
    "=" * 104
)

print(
    "35.5-C Last-Bucket-Only "
    "Per-Sample Stability Audit"
)

print(
    "=" * 104
)


all_sample_rows = []
integrity_rows = []
summary_rows = []


for dataset in DATASETS:

    print()
    print(
        "=" * 104
    )
    print(
        dataset
    )
    print(
        "=" * 104
    )


    # ========================================================
    # Canonical file-order test set
    # ========================================================

    canonical_path = (
        CANONICAL_ROOT
        / dataset
        / "file_order"
        / "test.csv"
    )


    if not canonical_path.exists():

        raise FileNotFoundError(
            canonical_path
        )


    canonical = pd.read_csv(
        canonical_path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_raw_lines": str,
        },
    )


    if not canonical[
        "sample_id"
    ].is_unique:

        raise RuntimeError(
            f"{dataset}: duplicate canonical sample_id"
        )


    canonical = (
        canonical
        .set_index(
            "sample_id",
            drop=False,
        )
    )


    # ========================================================
    # Load five input permutations + predictions
    # ========================================================

    input_frames = {}
    pred_frames = {}


    for seed in PERMUTATION_SEEDS:

        input_path = (
            INPUT_ROOT
            / dataset
            / f"perm_seed{seed}"
            / "test.csv"
        )

        pred_path = (
            PRED_ROOT
            / dataset
            / "seed42"
            / "test"
            / f"per_sample_preds_seed{seed}.csv"
        )


        if not input_path.exists():

            raise FileNotFoundError(
                input_path
            )


        if not pred_path.exists():

            raise FileNotFoundError(
                pred_path
            )


        inp = pd.read_csv(
            input_path,
            dtype={
                "sample_id": str,
                "user_id": str,
                "target_item": str,
                "sequence_raw_lines": str,
            },
        )


        pred = pd.read_csv(
            pred_path,
            dtype={
                "sample_id": str,
                "user_id": str,
                "target_item": str,
            },
        )


        required_pred = {
            "sample_id",
            "user_id",
            "target_item",
            "ambiguous_last_bucket",
            "target_rank",
            "ndcg10",
            "hit10",
            "hit20",
            "top20",
        }


        missing = (
            required_pred
            -
            set(
                pred.columns
            )
        )


        if missing:

            raise RuntimeError(
                f"{dataset}/seed{seed}: "
                f"missing prediction columns "
                f"{sorted(missing)}"
            )


        if not inp[
            "sample_id"
        ].is_unique:

            raise RuntimeError(
                f"{dataset}/seed{seed}: "
                "duplicate input sample_id"
            )


        if not pred[
            "sample_id"
        ].is_unique:

            raise RuntimeError(
                f"{dataset}/seed{seed}: "
                "duplicate prediction sample_id"
            )


        inp = inp.set_index(
            "sample_id",
            drop=False,
        )

        pred = pred.set_index(
            "sample_id",
            drop=False,
        )


        input_frames[
            seed
        ] = inp

        pred_frames[
            seed
        ] = pred


    # ========================================================
    # Global identity integrity
    # ========================================================

    canonical_ids = list(
        canonical.index
    )


    dataset_integrity_pass = True


    for seed in PERMUTATION_SEEDS:

        inp = input_frames[
            seed
        ]

        pred = pred_frames[
            seed
        ]


        same_input_ids = (
            list(
                inp.index
            )
            ==
            canonical_ids
        )


        same_pred_ids = (
            list(
                pred.index
            )
            ==
            canonical_ids
        )


        same_targets_input = (
            inp[
                "target_item"
            ]
            .astype(str)
            .tolist()
            ==
            canonical[
                "target_item"
            ]
            .astype(str)
            .tolist()
        )


        same_targets_pred = (
            pred[
                "target_item"
            ]
            .astype(str)
            .tolist()
            ==
            canonical[
                "target_item"
            ]
            .astype(str)
            .tolist()
        )


        seed_pass = bool(
            same_input_ids
            and
            same_pred_ids
            and
            same_targets_input
            and
            same_targets_pred
        )


        dataset_integrity_pass &= (
            seed_pass
        )


        integrity_rows.append(
            {
                "dataset":
                    dataset,

                "permutation_seed":
                    seed,

                "samples":
                    len(
                        canonical
                    ),

                "same_input_ids":
                    same_input_ids,

                "same_prediction_ids":
                    same_pred_ids,

                "same_input_targets":
                    same_targets_input,

                "same_prediction_targets":
                    same_targets_pred,

                "status":
                    (
                        "PASS"
                        if seed_pass
                        else "FAIL"
                    ),
            }
        )


    if not dataset_integrity_pass:

        raise RuntimeError(
            f"{dataset}: integrity check failed"
        )


    # ========================================================
    # Per-sample analysis
    # ========================================================

    identical_sequence_pairs = 0
    identical_sequence_prediction_mismatches = 0

    distinct_sequence_pairs = 0
    distinct_sequence_rank_changed_pairs = 0


    for sample_id in canonical_ids:

        canonical_row = canonical.loc[
            sample_id
        ]


        canonical_seq = (
            normalize_sequence(
                canonical_row[
                    "sequence_raw_lines"
                ]
            )
        )


        sequences = []

        ranks = []

        ndcgs = []

        hr10s = []

        hr20s = []

        top10s = []


        ambiguous = int(
            pred_frames[
                PERMUTATION_SEEDS[0]
            ].loc[
                sample_id,
                "ambiguous_last_bucket",
            ]
        )


        target_item = str(
            canonical_row[
                "target_item"
            ]
        )


        for seed in PERMUTATION_SEEDS:

            inp_row = (
                input_frames[
                    seed
                ].loc[
                    sample_id
                ]
            )

            pred_row = (
                pred_frames[
                    seed
                ].loc[
                    sample_id
                ]
            )


            sequences.append(
                normalize_sequence(
                    inp_row[
                        "sequence_raw_lines"
                    ]
                )
            )


            ranks.append(
                int(
                    pred_row[
                        "target_rank"
                    ]
                )
            )


            ndcgs.append(
                float(
                    pred_row[
                        "ndcg10"
                    ]
                )
            )


            hr10s.append(
                int(
                    pred_row[
                        "hit10"
                    ]
                )
            )


            hr20s.append(
                int(
                    pred_row[
                        "hit20"
                    ]
                )
            )


            top10s.append(
                normalize_top10(
                    pred_row[
                        "top20"
                    ]
                )
            )


        unique_sequence_count = len(
            set(
                sequences
            )
        )


        changed_from_canonical_count = sum(
            seq
            !=
            canonical_seq

            for seq
            in sequences
        )


        # At least two of the five tested inputs differ.
        actually_permuted = (
            unique_sequence_count
            >
            1
        )


        rank_array = np.asarray(
            ranks,
            dtype=float,
        )

        ndcg_array = np.asarray(
            ndcgs,
            dtype=float,
        )


        pair_overlaps = []

        distinct_pair_overlaps = []

        distinct_pair_abs_rank_diff = []

        sample_identical_mismatch = False


        for i, j in combinations(
            range(
                len(
                    PERMUTATION_SEEDS
                )
            ),
            2,
        ):

            same_sequence = (
                sequences[i]
                ==
                sequences[j]
            )


            same_rank = (
                ranks[i]
                ==
                ranks[j]
            )


            same_top10 = (
                top10s[i]
                ==
                top10s[j]
            )


            overlap = overlap10(
                top10s[i],
                top10s[j],
            )


            pair_overlaps.append(
                overlap
            )


            if same_sequence:

                identical_sequence_pairs += 1


                if not (
                    same_rank
                    and
                    same_top10
                ):

                    identical_sequence_prediction_mismatches += 1
                    sample_identical_mismatch = True


            else:

                distinct_sequence_pairs += 1


                if not same_rank:

                    distinct_sequence_rank_changed_pairs += 1


                distinct_pair_overlaps.append(
                    overlap
                )


                distinct_pair_abs_rank_diff.append(
                    abs(
                        ranks[i]
                        -
                        ranks[j]
                    )
                )


        all_sample_rows.append(
            {
                "dataset":
                    dataset,

                "sample_id":
                    sample_id,

                "user_id":
                    str(
                        canonical_row[
                            "user_id"
                        ]
                    ),

                "target_item":
                    target_item,

                "ambiguous_last_bucket":
                    ambiguous,

                "canonical_history_len":
                    len(
                        canonical_seq
                    ),

                "unique_sequence_variants":
                    unique_sequence_count,

                "changed_from_canonical_count":
                    changed_from_canonical_count,

                "actually_permuted":
                    bool(
                        actually_permuted
                    ),

                "rank_seed101":
                    ranks[0],

                "rank_seed202":
                    ranks[1],

                "rank_seed303":
                    ranks[2],

                "rank_seed404":
                    ranks[3],

                "rank_seed505":
                    ranks[4],

                "rank_std":
                    float(
                        np.std(
                            rank_array,
                            ddof=0,
                        )
                    ),

                "rank_min":
                    int(
                        rank_array.min()
                    ),

                "rank_max":
                    int(
                        rank_array.max()
                    ),

                "rank_range":
                    int(
                        rank_array.max()
                        -
                        rank_array.min()
                    ),

                "rank_changed":
                    bool(
                        rank_array.max()
                        !=
                        rank_array.min()
                    ),

                "ndcg10_min":
                    float(
                        ndcg_array.min()
                    ),

                "ndcg10_max":
                    float(
                        ndcg_array.max()
                    ),

                "ndcg10_range":
                    float(
                        ndcg_array.max()
                        -
                        ndcg_array.min()
                    ),

                "ndcg10_changed":
                    bool(
                        ndcg_array.max()
                        !=
                        ndcg_array.min()
                    ),

                "hr10_range":
                    int(
                        max(
                            hr10s
                        )
                        -
                        min(
                            hr10s
                        )
                    ),

                "hr20_range":
                    int(
                        max(
                            hr20s
                        )
                        -
                        min(
                            hr20s
                        )
                    ),

                "mean_pairwise_top10_overlap":
                    float(
                        np.mean(
                            pair_overlaps
                        )
                    ),

                "min_pairwise_top10_overlap":
                    float(
                        np.min(
                            pair_overlaps
                        )
                    ),

                "top10_changed":
                    bool(
                        len(
                            set(
                                top10s
                            )
                        )
                        >
                        1
                    ),

                "mean_top10_overlap_distinct_sequences":
                    (
                        float(
                            np.mean(
                                distinct_pair_overlaps
                            )
                        )
                        if
                        distinct_pair_overlaps
                        else
                        1.0
                    ),

                "mean_abs_rank_diff_distinct_sequences":
                    (
                        float(
                            np.mean(
                                distinct_pair_abs_rank_diff
                            )
                        )
                        if
                        distinct_pair_abs_rank_diff
                        else
                        0.0
                    ),

                "identical_sequence_prediction_mismatch":
                    bool(
                        sample_identical_mismatch
                    ),
            }
        )


    print(
        "Samples:",
        len(
            canonical
        )
    )

    print(
        "Integrity:",
        "PASS"
    )

    print(
        "Identical-sequence pairs:",
        identical_sequence_pairs
    )

    print(
        "Identical-sequence prediction mismatches:",
        identical_sequence_prediction_mismatches
    )

    print(
        "Distinct-sequence pairs:",
        distinct_sequence_pairs
    )

    print(
        "Distinct-sequence rank-changed pairs:",
        distinct_sequence_rank_changed_pairs
    )


# ============================================================
# Save per-sample detail
# ============================================================

sample_df = pd.DataFrame(
    all_sample_rows
)


detail_path = (
    OUT_DIR
    / "last_bucket_only_per_sample_stability.csv"
)


sample_df.to_csv(
    detail_path,
    index=False,
    encoding="utf-8-sig",
)


integrity_df = pd.DataFrame(
    integrity_rows
)


integrity_path = (
    OUT_DIR
    / "last_bucket_only_inference_integrity.csv"
)


integrity_df.to_csv(
    integrity_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Group summaries
# ============================================================

for dataset in DATASETS:

    d = sample_df[
        sample_df[
            "dataset"
        ]
        ==
        dataset
    ].copy()


    masks = {
        "overall":
            np.ones(
                len(d),
                dtype=bool,
            ),

        "ambiguous_last_bucket":
            (
                d[
                    "ambiguous_last_bucket"
                ]
                ==
                1
            ).to_numpy(),

        "nonambiguous_last_bucket":
            (
                d[
                    "ambiguous_last_bucket"
                ]
                ==
                0
            ).to_numpy(),

        "actually_permuted":
            d[
                "actually_permuted"
            ].astype(bool).to_numpy(),

        "ambiguous_and_actually_permuted":
            (
                (
                    d[
                        "ambiguous_last_bucket"
                    ]
                    ==
                    1
                )
                &
                d[
                    "actually_permuted"
                ].astype(bool)
            ).to_numpy(),
    }


    for group, mask in masks.items():

        g = d[
            mask
        ]


        if len(g) == 0:

            continue


        summary_rows.append(
            {
                "dataset":
                    dataset,

                "group":
                    group,

                "samples":
                    len(g),

                "mean_unique_sequence_variants":
                    float(
                        g[
                            "unique_sequence_variants"
                        ].mean()
                    ),

                "mean_rank_std":
                    float(
                        g[
                            "rank_std"
                        ].mean()
                    ),

                "median_rank_std":
                    float(
                        g[
                            "rank_std"
                        ].median()
                    ),

                "mean_rank_range":
                    float(
                        g[
                            "rank_range"
                        ].mean()
                    ),

                "median_rank_range":
                    float(
                        g[
                            "rank_range"
                        ].median()
                    ),

                "rank_changed_samples":
                    int(
                        g[
                            "rank_changed"
                        ].sum()
                    ),

                "rank_changed_ratio":
                    float(
                        g[
                            "rank_changed"
                        ].mean()
                    ),

                "mean_ndcg10_range":
                    float(
                        g[
                            "ndcg10_range"
                        ].mean()
                    ),

                "ndcg10_changed_samples":
                    int(
                        g[
                            "ndcg10_changed"
                        ].sum()
                    ),

                "ndcg10_changed_ratio":
                    float(
                        g[
                            "ndcg10_changed"
                        ].mean()
                    ),

                "mean_pairwise_top10_overlap":
                    float(
                        g[
                            "mean_pairwise_top10_overlap"
                        ].mean()
                    ),

                "median_pairwise_top10_overlap":
                    float(
                        g[
                            "mean_pairwise_top10_overlap"
                        ].median()
                    ),

                "mean_min_pairwise_top10_overlap":
                    float(
                        g[
                            "min_pairwise_top10_overlap"
                        ].mean()
                    ),

                "top10_changed_samples":
                    int(
                        g[
                            "top10_changed"
                        ].sum()
                    ),

                "top10_changed_ratio":
                    float(
                        g[
                            "top10_changed"
                        ].mean()
                    ),

                "mean_top10_overlap_distinct_sequences":
                    float(
                        g[
                            "mean_top10_overlap_distinct_sequences"
                        ].mean()
                    ),

                "mean_abs_rank_diff_distinct_sequences":
                    float(
                        g[
                            "mean_abs_rank_diff_distinct_sequences"
                        ].mean()
                    ),

                "identical_sequence_prediction_mismatch_samples":
                    int(
                        g[
                            "identical_sequence_prediction_mismatch"
                        ].sum()
                    ),
            }
        )


summary_df = pd.DataFrame(
    summary_rows
)


summary_path = (
    OUT_DIR
    / "last_bucket_only_stability_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Final integrity decision
# ============================================================

input_integrity_pass = bool(
    (
        integrity_df[
            "status"
        ]
        ==
        "PASS"
    ).all()
)


prediction_determinism_pass = bool(
    (
        sample_df[
            "identical_sequence_prediction_mismatch"
        ]
        ==
        False
    ).all()
)


final_status = (
    "PASS"
    if (
        input_integrity_pass
        and
        prediction_determinism_pass
    )
    else
    "REVIEW"
)


manifest = {
    "step":
        "35.5-C",

    "status":
        final_status,

    "model":
        (
            "fixed seed42 strict SASRec; "
            "no retraining between permutations"
        ),

    "perturbation":
        (
            "only within-order of the final "
            "ambiguous tied history bucket"
        ),

    "permutation_seeds":
        PERMUTATION_SEEDS,

    "integrity": {
        "sample_target_identity":
            input_integrity_pass,

        "identical_sequence_gives_identical_prediction":
            prediction_determinism_pass,
    },

    "primary_stability_metrics": [
        "rank_std",
        "rank_range",
        "rank_changed_ratio",
        "mean_pairwise_top10_overlap",
        "min_pairwise_top10_overlap",
        "top10_changed_ratio",
    ],

    "interpretation_boundary":
        (
            "This experiment isolates inference-time "
            "sensitivity to ordering inside the final "
            "timestamp-tied history bucket. It does not "
            "treat source-file row order as ground-truth "
            "chronology."
        ),
}


manifest_path = (
    OUT_DIR
    / "last_bucket_only_stability_manifest.json"
)


with open(
    manifest_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print(
    "=" * 104
)

print(
    "35.5-C Stability Summary"
)

print(
    "=" * 104
)


show_cols = [
    "dataset",
    "group",
    "samples",
    "mean_rank_std",
    "median_rank_std",
    "mean_rank_range",
    "rank_changed_ratio",
    "mean_pairwise_top10_overlap",
    "top10_changed_ratio",
    "mean_top10_overlap_distinct_sequences",
]


print(
    summary_df[
        show_cols
    ].to_string(
        index=False
    )
)


print()
print(
    "=" * 104
)

print(
    "35.5-C completed"
)

print(
    "=" * 104
)

print(
    "Input integrity PASS:",
    input_integrity_pass
)

print(
    "Deterministic identical-input PASS:",
    prediction_determinism_pass
)

print(
    "STATUS:",
    final_status
)

print()

print(
    "Saved:",
    detail_path
)

print(
    "Saved:",
    summary_path
)

print(
    "Saved:",
    manifest_path
)