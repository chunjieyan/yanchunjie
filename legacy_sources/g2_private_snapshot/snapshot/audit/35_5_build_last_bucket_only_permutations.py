from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(r"D:\time_uncertainty_rec")

INPUT_ROOT = (
    ROOT
    / "processed"
    / "strict_diagnostic_a"
)

OUTPUT_ROOT = (
    ROOT
    / "processed"
    / "strict_last_bucket_only"
)

REPRO_DIR = (
    ROOT
    / "reproducibility"
    / "step35_5"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

REPRO_DIR.mkdir(
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

    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    return json.loads(
        str(value)
    )


def deterministic_rng_seed(
    dataset,
    sample_id,
    permutation_seed,
):

    key = (
        f"{dataset}|"
        f"{sample_id}|"
        f"{permutation_seed}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def same_multiset(a, b):

    return Counter(
        map(str, a)
    ) == Counter(
        map(str, b)
    )


print("=" * 104)
print("35.5-A Build Last-Bucket-Only Permutations")
print("=" * 104)


summary = {}
detail_rows = []


for dataset in DATASETS:

    print()
    print("=" * 104)
    print(dataset)
    print("=" * 104)


    source_path = (
        INPUT_ROOT
        / dataset
        / "file_order"
        / "test.csv"
    )


    if not source_path.exists():

        raise FileNotFoundError(
            source_path
        )


    source = pd.read_csv(
        source_path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_items": str,
            "sequence_raw_lines": str,
        },
    )


    if not source[
        "sample_id"
    ].is_unique:

        raise RuntimeError(
            f"{dataset}: duplicate sample_id"
        )


    required = {
        "sample_id",
        "user_id",
        "target_item",
        "target_raw_line",
        "target_timestamp",
        "ambiguous_last_bucket",
        "last_history_bucket_size",
        "fixed_history_len",
        "sequence_items",
        "sequence_raw_lines",
    }


    missing = (
        required
        -
        set(source.columns)
    )


    if missing:

        raise RuntimeError(
            f"{dataset}: missing columns "
            f"{sorted(missing)}"
        )


    ambiguous_source = (
        source[
            "ambiguous_last_bucket"
        ].astype(int)
        ==
        1
    )


    summary[
        dataset
    ] = {
        "source_samples":
            int(
                len(source)
            ),

        "ambiguous_samples":
            int(
                ambiguous_source.sum()
            ),

        "permutations":
            {},
    }


    print(
        "Source samples:",
        len(source)
    )

    print(
        "Ambiguous last-bucket samples:",
        int(
            ambiguous_source.sum()
        )
    )


    for permutation_seed in PERMUTATION_SEEDS:

        print()
        print("-" * 104)

        print(
            "Permutation seed:",
            permutation_seed
        )

        print("-" * 104)


        output = source.copy()


        changed_samples = 0
        changed_ambiguous = 0

        prefix_changed = 0
        content_changed = 0
        nonambiguous_changed = 0
        target_changed = 0
        target_leakage = 0

        permutation_identity = 0


        new_items_col = []
        new_raw_col = []


        for row in source.itertuples(
            index=False
        ):

            seq_items = parse_list(
                row.sequence_items
            )

            seq_raw = parse_list(
                row.sequence_raw_lines
            )


            if len(seq_items) != len(
                seq_raw
            ):

                raise RuntimeError(
                    f"{dataset}/"
                    f"{row.sample_id}: "
                    "item/raw length mismatch"
                )


            fixed_history_len = int(
                row.fixed_history_len
            )


            if len(
                seq_raw
            ) != fixed_history_len:

                raise RuntimeError(
                    f"{dataset}/"
                    f"{row.sample_id}: "
                    "history length mismatch"
                )


            ambiguous = (
                int(
                    row.ambiguous_last_bucket
                )
                ==
                1
            )


            bucket_size = int(
                row.last_history_bucket_size
            )


            if bucket_size < 1:

                raise RuntimeError(
                    f"{dataset}/"
                    f"{row.sample_id}: "
                    "invalid bucket size"
                )


            if bucket_size > len(
                seq_raw
            ):

                raise RuntimeError(
                    f"{dataset}/"
                    f"{row.sample_id}: "
                    "bucket larger than history"
                )


            original_items = list(
                seq_items
            )

            original_raw = [
                int(x)
                for x in seq_raw
            ]


            new_items = list(
                original_items
            )

            new_raw = list(
                original_raw
            )


            if (
                ambiguous
                and
                bucket_size > 1
            ):

                prefix_items = (
                    original_items[
                        :-bucket_size
                    ]
                )

                prefix_raw = (
                    original_raw[
                        :-bucket_size
                    ]
                )

                tail_items = (
                    original_items[
                        -bucket_size:
                    ]
                )

                tail_raw = (
                    original_raw[
                        -bucket_size:
                    ]
                )


                rng_seed = (
                    deterministic_rng_seed(
                        dataset,
                        row.sample_id,
                        permutation_seed,
                    )
                )


                rng = (
                    np.random.default_rng(
                        rng_seed
                    )
                )


                permutation = (
                    rng.permutation(
                        bucket_size
                    )
                )


                shuffled_items = [
                    tail_items[i]
                    for i in permutation
                ]

                shuffled_raw = [
                    tail_raw[i]
                    for i in permutation
                ]


                new_items = (
                    prefix_items
                    +
                    shuffled_items
                )

                new_raw = (
                    prefix_raw
                    +
                    shuffled_raw
                )


                if (
                    new_raw
                    ==
                    original_raw
                ):

                    permutation_identity += 1


                if (
                    new_raw[
                        :-bucket_size
                    ]
                    !=
                    original_raw[
                        :-bucket_size
                    ]
                ):

                    prefix_changed += 1


            exact_same = (
                new_raw
                ==
                original_raw
            )


            if not exact_same:

                changed_samples += 1

                if ambiguous:
                    changed_ambiguous += 1

                else:
                    nonambiguous_changed += 1


            if not same_multiset(
                new_raw,
                original_raw,
            ):

                content_changed += 1


            target_raw = int(
                row.target_raw_line
            )


            if target_raw in new_raw:

                target_leakage += 1


            new_items_col.append(
                json.dumps(
                    new_items,
                    ensure_ascii=False,
                )
            )

            new_raw_col.append(
                json.dumps(
                    new_raw,
                    ensure_ascii=False,
                )
            )


            detail_rows.append(
                {
                    "dataset":
                        dataset,

                    "permutation_seed":
                        permutation_seed,

                    "sample_id":
                        row.sample_id,

                    "ambiguous_last_bucket":
                        int(
                            ambiguous
                        ),

                    "last_history_bucket_size":
                        bucket_size,

                    "sequence_changed":
                        bool(
                            not exact_same
                        ),

                    "history_content_same":
                        bool(
                            same_multiset(
                                new_raw,
                                original_raw,
                            )
                        ),

                    "prefix_unchanged":
                        bool(
                            (
                                new_raw[
                                    :-bucket_size
                                ]
                                ==
                                original_raw[
                                    :-bucket_size
                                ]
                            )
                            if bucket_size > 0
                            else True
                        ),

                    "target_absent_from_history":
                        bool(
                            target_raw
                            not in new_raw
                        ),
                }
            )


        output[
            "sequence_items"
        ] = new_items_col


        output[
            "sequence_raw_lines"
        ] = new_raw_col


        # ====================================================
        # Target metadata must remain byte-for-byte identical
        # ====================================================

        target_cols = [
            "sample_id",
            "user_id",
            "target_item",
            "target_raw_line",
            "target_timestamp",
            "target_bucket",
            "ambiguous_last_bucket",
            "last_history_bucket_size",
            "fixed_history_len",
        ]


        for col in target_cols:

            if col not in source.columns:
                continue

            same = (
                source[col]
                .fillna("__NA__")
                .astype(str)
                .equals(
                    output[col]
                    .fillna("__NA__")
                    .astype(str)
                )
            )

            if not same:

                target_changed += 1


        out_dir = (
            OUTPUT_ROOT
            / dataset
            / f"perm_seed{permutation_seed}"
        )


        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        out_path = (
            out_dir
            / "test.csv"
        )


        output.to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig",
        )


        pass_flag = (
            len(output)
            ==
            len(source)
            and
            content_changed
            ==
            0
            and
            prefix_changed
            ==
            0
            and
            nonambiguous_changed
            ==
            0
            and
            target_changed
            ==
            0
            and
            target_leakage
            ==
            0
        )


        summary[
            dataset
        ][
            "permutations"
        ][
            str(
                permutation_seed
            )
        ] = {
            "samples":
                int(
                    len(output)
                ),

            "changed_samples":
                int(
                    changed_samples
                ),

            "changed_ambiguous_samples":
                int(
                    changed_ambiguous
                ),

            "identity_permutations":
                int(
                    permutation_identity
                ),

            "prefix_changed":
                int(
                    prefix_changed
                ),

            "history_content_changed":
                int(
                    content_changed
                ),

            "nonambiguous_changed":
                int(
                    nonambiguous_changed
                ),

            "target_metadata_changed":
                int(
                    target_changed
                ),

            "target_leakage":
                int(
                    target_leakage
                ),

            "pass":
                bool(
                    pass_flag
                ),

            "output":
                str(
                    out_path.relative_to(
                        ROOT
                    )
                ),
        }


        print(
            "Samples:",
            len(output)
        )

        print(
            "Changed samples:",
            changed_samples
        )

        print(
            "Changed ambiguous samples:",
            changed_ambiguous
        )

        print(
            "Identity permutations:",
            permutation_identity
        )

        print()

        print(
            "Prefix changed:",
            prefix_changed
        )

        print(
            "History content changed:",
            content_changed
        )

        print(
            "Non-ambiguous changed:",
            nonambiguous_changed
        )

        print(
            "Target metadata changed:",
            target_changed
        )

        print(
            "Target leakage:",
            target_leakage
        )

        print(
            "PASS:",
            pass_flag
        )


# ============================================================
# Final audit
# ============================================================

all_pass = all(
    p[
        "pass"
    ]

    for dataset_info
    in summary.values()

    for p
    in dataset_info[
        "permutations"
    ].values()
)


detail_df = pd.DataFrame(
    detail_rows
)


detail_path = (
    REPRO_DIR
    / "last_bucket_only_permutation_detail.csv"
)


detail_df.to_csv(
    detail_path,
    index=False,
    encoding="utf-8-sig",
)


manifest = {
    "step":
        "35.5-A",

    "status":
        (
            "PASS"
            if all_pass
            else "REVIEW"
        ),

    "purpose":
        (
            "Create deterministic inference-time "
            "perturbations that change only the "
            "within-order of the final ambiguous "
            "history bucket."
        ),

    "constraints": {
        "model":
            "fixed later in 35.5-B",

        "target":
            "unchanged",

        "history_event_set":
            "unchanged",

        "history_prefix":
            "unchanged",

        "nonambiguous_samples":
            "unchanged",

        "changed_component":
            (
                "order inside final "
                "ambiguous history bucket only"
            ),
    },

    "permutation_seeds":
        PERMUTATION_SEEDS,

    "datasets":
        summary,

    "important_interpretation":
        (
            "raw_line is used only as a stable "
            "event identifier. The experiment "
            "does not treat source-file order "
            "as ground-truth chronology."
        ),

    "next_step":
        (
            "35.5-B evaluate all five "
            "last-bucket-only permutations "
            "with one fixed SASRec checkpoint."
        ),
}


manifest_path = (
    REPRO_DIR
    / "last_bucket_only_permutation_manifest.json"
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
print("=" * 104)
print("35.5-A completed")
print("=" * 104)

print(
    "STATUS:",
    manifest[
        "status"
    ]
)

print(
    "Datasets:",
    len(
        DATASETS
    )
)

print(
    "Permutation seeds:",
    PERMUTATION_SEEDS
)

print()

print(
    "Saved:",
    manifest_path
)

print(
    "Saved:",
    detail_path
)