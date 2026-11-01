from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


ROOT = Path(
    r"D:\time_uncertainty_rec"
)

INPUT_PATH = (
    ROOT
    / "reproducibility"
    / "step35_5"
    / "last_bucket_only_per_sample_stability.csv"
)

C_MANIFEST_PATH = (
    ROOT
    / "reproducibility"
    / "step35_5"
    / "last_bucket_only_stability_manifest.json"
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


BOOTSTRAP_REPS = 10000
BASE_SEED = 355
CI_LOW_Q = 0.025
CI_HIGH_Q = 0.975

Z_95 = 1.959963984540054


DATASETS = [
    "Amazon_Beauty",
    "MovieLens-1M",
]


EXPECTED = {
    "Amazon_Beauty": {
        "all": 2966,
        "ambiguous": 949,
        "primary": 922,
    },

    "MovieLens-1M": {
        "all": 945,
        "ambiguous": 167,
        "primary": 163,
    },
}


CONTINUOUS_METRICS = {
    "rank_std":
        "mean_rank_std",

    "rank_range":
        "mean_rank_range",

    "mean_abs_rank_diff_distinct_sequences":
        "mean_abs_rank_diff_distinct_sequences",

    "mean_top10_overlap_distinct_sequences":
        "mean_top10_overlap_distinct_sequences",

    "ndcg10_range":
        "mean_ndcg10_range",
}


PROPORTION_METRICS = {
    "rank_changed":
        "rank_changed_ratio",

    "top10_changed":
        "top10_changed_ratio",
}


# ============================================================
# Helpers
# ============================================================

def to_bool_series(series):

    if pd.api.types.is_bool_dtype(
        series
    ):
        return (
            series
            .fillna(False)
            .astype(bool)
        )

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    invalid = ~normalized.isin(
        mapping.keys()
    )

    if invalid.any():

        bad = (
            normalized[
                invalid
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Cannot parse boolean values: "
            f"{bad[:10]}"
        )

    return normalized.map(
        mapping
    ).astype(bool)


def bootstrap_mean_matrix(
    frame,
    columns,
    seed,
    reps=10000,
    chunk_size=250,
):

    values = (
        frame[
            columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    n, k = values.shape

    if n <= 0:

        raise RuntimeError(
            "Bootstrap received empty group"
        )

    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            "Non-finite values detected "
            "before bootstrap"
        )

    point = values.mean(
        axis=0
    )

    rng = np.random.default_rng(
        seed
    )

    boot = np.empty(
        (
            reps,
            k,
        ),
        dtype=np.float64,
    )

    done = 0

    while done < reps:

        current = min(
            chunk_size,
            reps - done,
        )

        indices = rng.integers(
            low=0,
            high=n,
            size=(
                current,
                n,
            ),
        )

        sampled = values[
            indices
        ]

        boot[
            done:
            done + current
        ] = sampled.mean(
            axis=1
        )

        done += current

    low = np.quantile(
        boot,
        CI_LOW_Q,
        axis=0,
    )

    high = np.quantile(
        boot,
        CI_HIGH_Q,
        axis=0,
    )

    return (
        point,
        low,
        high,
    )


def wilson_interval(
    successes,
    n,
    z=Z_95,
):

    if n <= 0:

        raise RuntimeError(
            "Wilson interval received n <= 0"
        )

    p = (
        successes
        /
        n
    )

    z2 = z * z

    denominator = (
        1.0
        +
        z2 / n
    )

    center = (
        p
        +
        z2 / (2.0 * n)
    ) / denominator

    half = (
        z
        *
        math.sqrt(
            (
                p * (1.0 - p) / n
            )
            +
            (
                z2
                /
                (4.0 * n * n)
            )
        )
        /
        denominator
    )

    low = max(
        0.0,
        center - half,
    )

    high = min(
        1.0,
        center + half,
    )

    return (
        p,
        low,
        high,
    )


def format_continuous(
    estimate,
    low,
    high,
    digits=2,
):

    return (
        f"{estimate:.{digits}f} "
        f"[{low:.{digits}f}, "
        f"{high:.{digits}f}]"
    )


def format_proportion(
    estimate,
    low,
    high,
):

    return (
        f"{100.0 * estimate:.2f}% "
        f"[{100.0 * low:.2f}%, "
        f"{100.0 * high:.2f}%]"
    )


# ============================================================
# Input checks
# ============================================================

print(
    "=" * 110
)

print(
    "35.5-D Bootstrap CI + Paper Main Table"
)

print(
    "=" * 110
)


if not INPUT_PATH.exists():

    raise FileNotFoundError(
        INPUT_PATH
    )


if not C_MANIFEST_PATH.exists():

    raise FileNotFoundError(
        C_MANIFEST_PATH
    )


with open(
    C_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as f:

    c_manifest = json.load(
        f
    )


if c_manifest.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "35.5-C manifest is not PASS"
    )


df = pd.read_csv(
    INPUT_PATH,
    dtype={
        "sample_id": str,
        "user_id": str,
        "target_item": str,
    },
)


required_columns = {
    "dataset",
    "sample_id",
    "ambiguous_last_bucket",
    "actually_permuted",
    "rank_std",
    "rank_range",
    "rank_changed",
    "ndcg10_range",
    "top10_changed",
    "mean_pairwise_top10_overlap",
    "mean_top10_overlap_distinct_sequences",
    "mean_abs_rank_diff_distinct_sequences",
    "identical_sequence_prediction_mismatch",
}


missing = (
    required_columns
    -
    set(
        df.columns
    )
)


if missing:

    raise RuntimeError(
        "Missing columns: "
        f"{sorted(missing)}"
    )


# Parse booleans safely.

for col in [
    "ambiguous_last_bucket",
    "actually_permuted",
    "rank_changed",
    "top10_changed",
    "identical_sequence_prediction_mismatch",
]:

    df[
        col
    ] = to_bool_series(
        df[
            col
        ]
    )


# ============================================================
# Integrity
# ============================================================

if df[
    "identical_sequence_prediction_mismatch"
].any():

    raise RuntimeError(
        "Identical-input prediction "
        "mismatch detected"
    )


if (
    df[
        "actually_permuted"
    ]
    &
    ~df[
        "ambiguous_last_bucket"
    ]
).any():

    raise RuntimeError(
        "Found actually_permuted sample "
        "outside ambiguous subgroup"
    )


for dataset in DATASETS:

    d = df[
        df[
            "dataset"
        ]
        ==
        dataset
    ]

    n_all = len(
        d
    )

    n_ambiguous = int(
        d[
            "ambiguous_last_bucket"
        ].sum()
    )

    n_primary = int(
        (
            d[
                "ambiguous_last_bucket"
            ]
            &
            d[
                "actually_permuted"
            ]
        ).sum()
    )


    exp = EXPECTED[
        dataset
    ]


    if (
        n_all
        !=
        exp[
            "all"
        ]
    ):

        raise RuntimeError(
            f"{dataset}: unexpected "
            f"all samples {n_all}"
        )


    if (
        n_ambiguous
        !=
        exp[
            "ambiguous"
        ]
    ):

        raise RuntimeError(
            f"{dataset}: unexpected "
            f"ambiguous samples "
            f"{n_ambiguous}"
        )


    if (
        n_primary
        !=
        exp[
            "primary"
        ]
    ):

        raise RuntimeError(
            f"{dataset}: unexpected "
            f"primary samples "
            f"{n_primary}"
        )


    print()
    print(
        dataset
    )

    print(
        "  all samples:",
        n_all
    )

    print(
        "  ambiguous:",
        n_ambiguous
    )

    print(
        "  ambiguous & actually permuted:",
        n_primary
    )

    print(
        "  integrity: PASS"
    )


# ============================================================
# Groups
# ============================================================

ci_records = []


for dataset_idx, dataset in enumerate(
    DATASETS
):

    d = (
        df[
            df[
                "dataset"
            ]
            ==
            dataset
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    group_masks = {
        "overall":
            pd.Series(
                True,
                index=d.index,
            ),

        "ambiguous_last_bucket":
            d[
                "ambiguous_last_bucket"
            ],

        "nonambiguous_negative_control":
            ~d[
                "ambiguous_last_bucket"
            ],

        "ambiguous_and_actually_permuted":
            (
                d[
                    "ambiguous_last_bucket"
                ]
                &
                d[
                    "actually_permuted"
                ]
            ),
    }


    for group_idx, (
        group_name,
        mask,
    ) in enumerate(
        group_masks.items()
    ):

        g = (
            d[
                mask
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        if len(
            g
        ) == 0:

            continue


        # ----------------------------------------
        # Continuous sample-level metrics
        # ----------------------------------------

        source_columns = list(
            CONTINUOUS_METRICS.keys()
        )


        bootstrap_seed = (
            BASE_SEED
            +
            dataset_idx * 1000
            +
            group_idx * 100
        )


        (
            point,
            low,
            high,
        ) = bootstrap_mean_matrix(
            frame=g,
            columns=source_columns,
            seed=bootstrap_seed,
            reps=BOOTSTRAP_REPS,
        )


        for metric_idx, source_col in enumerate(
            source_columns
        ):

            metric_name = (
                CONTINUOUS_METRICS[
                    source_col
                ]
            )


            ci_records.append(
                {
                    "dataset":
                        dataset,

                    "group":
                        group_name,

                    "n":
                        len(
                            g
                        ),

                    "metric":
                        metric_name,

                    "estimate":
                        float(
                            point[
                                metric_idx
                            ]
                        ),

                    "ci_low":
                        float(
                            low[
                                metric_idx
                            ]
                        ),

                    "ci_high":
                        float(
                            high[
                                metric_idx
                            ]
                        ),

                    "ci_level":
                        0.95,

                    "ci_method":
                        (
                            "sample-level percentile "
                            "bootstrap"
                        ),

                    "bootstrap_reps":
                        BOOTSTRAP_REPS,

                    "bootstrap_seed":
                        bootstrap_seed,
                }
            )


        # ----------------------------------------
        # Binary sample-level metrics
        # ----------------------------------------

        for source_col, metric_name in (
            PROPORTION_METRICS.items()
        ):

            successes = int(
                g[
                    source_col
                ].sum()
            )

            (
                estimate,
                ci_low,
                ci_high,
            ) = wilson_interval(
                successes=
                    successes,

                n=
                    len(
                        g
                    ),
            )


            ci_records.append(
                {
                    "dataset":
                        dataset,

                    "group":
                        group_name,

                    "n":
                        len(
                            g
                        ),

                    "metric":
                        metric_name,

                    "estimate":
                        float(
                            estimate
                        ),

                    "ci_low":
                        float(
                            ci_low
                        ),

                    "ci_high":
                        float(
                            ci_high
                        ),

                    "ci_level":
                        0.95,

                    "ci_method":
                        "Wilson score",

                    "bootstrap_reps":
                        0,

                    "bootstrap_seed":
                        None,
                }
            )


# ============================================================
# Save detailed CI table
# ============================================================

ci_df = pd.DataFrame(
    ci_records
)


ci_path = (
    OUT_DIR
    / "last_bucket_only_bootstrap_ci.csv"
)


ci_df.to_csv(
    ci_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Primary paper table
# ============================================================

PRIMARY_GROUP = (
    "ambiguous_and_actually_permuted"
)


def get_ci(
    dataset,
    metric,
):

    rows = ci_df[
        (
            ci_df[
                "dataset"
            ]
            ==
            dataset
        )
        &
        (
            ci_df[
                "group"
            ]
            ==
            PRIMARY_GROUP
        )
        &
        (
            ci_df[
                "metric"
            ]
            ==
            metric
        )
    ]


    if len(
        rows
    ) != 1:

        raise RuntimeError(
            f"Cannot uniquely find "
            f"{dataset} / {metric}"
        )


    return rows.iloc[
        0
    ]


paper_rows_raw = []
paper_rows_formatted = []


for dataset in DATASETS:

    primary_n = EXPECTED[
        dataset
    ][
        "primary"
    ]


    rank_changed = get_ci(
        dataset,
        "rank_changed_ratio",
    )

    rank_std = get_ci(
        dataset,
        "mean_rank_std",
    )

    rank_range = get_ci(
        dataset,
        "mean_rank_range",
    )

    abs_rank_diff = get_ci(
        dataset,
        "mean_abs_rank_diff_distinct_sequences",
    )

    top10_changed = get_ci(
        dataset,
        "top10_changed_ratio",
    )

    top10_overlap = get_ci(
        dataset,
        "mean_top10_overlap_distinct_sequences",
    )


    paper_rows_raw.append(
        {
            "dataset":
                dataset,

            "n_primary":
                primary_n,

            "rank_changed_estimate":
                rank_changed[
                    "estimate"
                ],

            "rank_changed_ci_low":
                rank_changed[
                    "ci_low"
                ],

            "rank_changed_ci_high":
                rank_changed[
                    "ci_high"
                ],

            "mean_rank_std":
                rank_std[
                    "estimate"
                ],

            "mean_rank_std_ci_low":
                rank_std[
                    "ci_low"
                ],

            "mean_rank_std_ci_high":
                rank_std[
                    "ci_high"
                ],

            "mean_rank_range":
                rank_range[
                    "estimate"
                ],

            "mean_rank_range_ci_low":
                rank_range[
                    "ci_low"
                ],

            "mean_rank_range_ci_high":
                rank_range[
                    "ci_high"
                ],

            "mean_abs_rank_diff":
                abs_rank_diff[
                    "estimate"
                ],

            "mean_abs_rank_diff_ci_low":
                abs_rank_diff[
                    "ci_low"
                ],

            "mean_abs_rank_diff_ci_high":
                abs_rank_diff[
                    "ci_high"
                ],

            "top10_changed_estimate":
                top10_changed[
                    "estimate"
                ],

            "top10_changed_ci_low":
                top10_changed[
                    "ci_low"
                ],

            "top10_changed_ci_high":
                top10_changed[
                    "ci_high"
                ],

            "mean_top10_overlap":
                top10_overlap[
                    "estimate"
                ],

            "mean_top10_overlap_ci_low":
                top10_overlap[
                    "ci_low"
                ],

            "mean_top10_overlap_ci_high":
                top10_overlap[
                    "ci_high"
                ],
        }
    )


    paper_rows_formatted.append(
        {
            "Dataset":
                (
                    "Amazon Beauty"
                    if dataset
                    ==
                    "Amazon_Beauty"
                    else
                    "MovieLens-1M"
                ),

            "N":
                str(
                    primary_n
                ),

            "Rank changed":
                format_proportion(
                    rank_changed[
                        "estimate"
                    ],
                    rank_changed[
                        "ci_low"
                    ],
                    rank_changed[
                        "ci_high"
                    ],
                ),

            "Mean RankStd":
                format_continuous(
                    rank_std[
                        "estimate"
                    ],
                    rank_std[
                        "ci_low"
                    ],
                    rank_std[
                        "ci_high"
                    ],
                    digits=2,
                ),

            "Mean RankRange":
                format_continuous(
                    rank_range[
                        "estimate"
                    ],
                    rank_range[
                        "ci_low"
                    ],
                    rank_range[
                        "ci_high"
                    ],
                    digits=2,
                ),

            "Mean abs rank diff":
                format_continuous(
                    abs_rank_diff[
                        "estimate"
                    ],
                    abs_rank_diff[
                        "ci_low"
                    ],
                    abs_rank_diff[
                        "ci_high"
                    ],
                    digits=2,
                ),

            "Top-10 changed":
                format_proportion(
                    top10_changed[
                        "estimate"
                    ],
                    top10_changed[
                        "ci_low"
                    ],
                    top10_changed[
                        "ci_high"
                    ],
                ),

            "Mean Top-10 overlap":
                format_continuous(
                    top10_overlap[
                        "estimate"
                    ],
                    top10_overlap[
                        "ci_low"
                    ],
                    top10_overlap[
                        "ci_high"
                    ],
                    digits=3,
                ),
        }
    )


paper_raw_df = pd.DataFrame(
    paper_rows_raw
)


paper_raw_path = (
    OUT_DIR
    / "last_bucket_only_paper_main_table.csv"
)


paper_raw_df.to_csv(
    paper_raw_path,
    index=False,
    encoding="utf-8-sig",
)


paper_formatted_df = pd.DataFrame(
    paper_rows_formatted
)


# ============================================================
# Markdown table without external tabulate dependency
# ============================================================

headers = list(
    paper_formatted_df.columns
)


md_lines = [
    "| "
    +
    " | ".join(
        headers
    )
    +
    " |",

    "| "
    +
    " | ".join(
        [
            "---"
            for _
            in headers
        ]
    )
    +
    " |",
]


for _, row in (
    paper_formatted_df
    .iterrows()
):

    md_lines.append(
        "| "
        +
        " | ".join(
            str(
                row[
                    h
                ]
            )
            for h
            in headers
        )
        +
        " |"
    )


md_path = (
    OUT_DIR
    / "last_bucket_only_paper_main_table.md"
)


md_path.write_text(
    "\n".join(
        md_lines
    )
    +
    "\n",
    encoding="utf-8",
)


# ============================================================
# Manifest
# ============================================================

manifest = {
    "step":
        "35.5-D",

    "status":
        "PASS",

    "input":
        str(
            INPUT_PATH
        ),

    "primary_group":
        (
            "ambiguous_last_bucket "
            "AND actually_permuted"
        ),

    "expected_primary_samples": {
        "Amazon_Beauty":
            922,

        "MovieLens-1M":
            163,
    },

    "continuous_ci": {
        "method":
            (
                "sample-level percentile "
                "bootstrap"
            ),

        "repetitions":
            BOOTSTRAP_REPS,

        "confidence_level":
            0.95,

        "base_seed":
            BASE_SEED,

        "resampling_unit":
            "test sample",
    },

    "proportion_ci": {
        "method":
            "Wilson score",

        "confidence_level":
            0.95,

        "reason":
            (
                "avoids degenerate percentile "
                "bootstrap intervals when the "
                "observed proportion equals 1"
            ),
    },

    "paper_table_metrics": [
        "rank_changed_ratio",
        "mean_rank_std",
        "mean_rank_range",
        "mean_abs_rank_diff_distinct_sequences",
        "top10_changed_ratio",
        "mean_top10_overlap_distinct_sequences",
    ],

    "interpretation_boundary": [
        (
            "The experiment isolates inference-time "
            "sensitivity to ordering within the final "
            "timestamp-tied history bucket."
        ),

        (
            "Source-file row order is not interpreted "
            "as ground-truth chronology."
        ),

        (
            "Absolute rank variability should not be "
            "directly compared across datasets because "
            "their item universes differ."
        ),
    ],
}


manifest_path = (
    OUT_DIR
    / "last_bucket_only_bootstrap_manifest.json"
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


# ============================================================
# Console output
# ============================================================

print()
print(
    "=" * 110
)

print(
    "35.5-D Primary Paper Table"
)

print(
    "=" * 110
)


print(
    paper_formatted_df.to_string(
        index=False
    )
)


print()
print(
    "=" * 110
)

print(
    "Integrity / methodology"
)

print(
    "=" * 110
)

print(
    "35.5-C status:",
    c_manifest.get(
        "status"
    )
)

print(
    "Bootstrap repetitions:",
    BOOTSTRAP_REPS
)

print(
    "Bootstrap unit:",
    "test sample"
)

print(
    "Continuous CI:",
    "percentile bootstrap 95%"
)

print(
    "Binary proportion CI:",
    "Wilson 95%"
)

print(
    "STATUS: PASS"
)


print()
print(
    "Saved:",
    ci_path
)

print(
    "Saved:",
    paper_raw_path
)

print(
    "Saved:",
    md_path
)

print(
    "Saved:",
    manifest_path
)