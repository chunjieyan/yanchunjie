from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

SPLIT_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_a_splits"
)

STRICT_ROOT = (
    ROOT
    / "processed"
    / "strict_protocol"
)

REP_ROOT = (
    ROOT
    / "reproducibility"
    / "step36"
)

REP_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MIN_HISTORY = 5

DATASETS = [
    "Amazon_Sports",
    "Amazon_Toys",
]

EXPECTED_COLUMNS = [
    "sample_id",
    "dataset",
    "split",
    "user_id",
    "target_item",
    "target_raw_line",
    "target_timestamp",
    "target_bucket",
    "history_len",
    "last_history_timestamp",
    "last_history_bucket",
    "last_history_bucket_size",
    "ambiguous_last_bucket",
    "bucket_gap",
]


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_dataset_with_split(dataset):

    frames = []

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        path = (
            SPLIT_ROOT
            / dataset
            / "file_order"
            / f"{split_name}.csv"
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        df = pd.read_csv(
            path,
            dtype={
                "user_id": str,
                "item_id": str,
            },
        )

        required = {
            "user_id",
            "item_id",
            "rating",
            "timestamp",
            "time_bucket",
            "raw_line",
        }

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            raise RuntimeError(
                f"{dataset}/{split_name}: "
                f"missing columns {missing}"
            )

        df["_split"] = split_name

        frames.append(
            df
        )

    full = pd.concat(
        frames,
        ignore_index=True,
    )

    # raw_line must identify an event uniquely.
    if not full["raw_line"].is_unique:
        raise RuntimeError(
            f"{dataset}: raw_line not unique"
        )

    return full


def build_strict_samples(
    full,
    dataset,
    split_name,
):

    # Frozen Beauty protocol:
    #
    # 1. deterministic user sequence
    # 2. bucket tie broken by raw_line
    # 3. target = final interaction in current split
    # 4. history = all events before that fixed target
    # 5. same-bucket target excluded
    # 6. min history = 5

    df = (
        full
        .sort_values(
            [
                "user_id",
                "time_bucket",
                "raw_line",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    samples = []

    candidate_users = 0
    excluded_no_history = 0
    excluded_same_bucket = 0
    excluded_short_history = 0

    for user_id, group in df.groupby(
        "user_id",
        sort=True,
    ):

        group = (
            group
            .reset_index(drop=True)
        )

        split_positions = np.flatnonzero(
            group["_split"]
            .to_numpy()
            == split_name
        )

        if len(split_positions) == 0:
            continue

        candidate_users += 1

        # Canonical leave-one-out target:
        # final interaction in this split.
        target_idx = int(
            split_positions[-1]
        )

        target = group.iloc[
            target_idx
        ]

        history = group.iloc[
            :target_idx
        ]

        if len(history) == 0:
            excluded_no_history += 1
            continue

        last_history = history.iloc[-1]

        last_history_bucket = int(
            last_history["time_bucket"]
        )

        target_bucket = int(
            target["time_bucket"]
        )

        # Frozen strict condition:
        # target must be strictly later than
        # the final history bucket.
        if (
            target_bucket
            <= last_history_bucket
        ):
            excluded_same_bucket += 1
            continue

        if len(history) < MIN_HISTORY:
            excluded_short_history += 1
            continue

        last_bucket_history = history[
            history["time_bucket"]
            == last_history_bucket
        ]

        last_bucket_size = int(
            len(last_bucket_history)
        )

        ambiguous = int(
            last_bucket_size >= 2
        )

        raw_line = int(
            target["raw_line"]
        )

        sample_id = (
            f"{dataset}"
            f"__{split_name}"
            f"__{raw_line}"
        )

        samples.append({
            "sample_id":
                sample_id,

            "dataset":
                dataset,

            "split":
                split_name,

            "user_id":
                str(user_id),

            "target_item":
                str(
                    target["item_id"]
                ),

            "target_raw_line":
                raw_line,

            "target_timestamp":
                int(
                    target["timestamp"]
                ),

            "target_bucket":
                target_bucket,

            "history_len":
                int(
                    len(history)
                ),

            "last_history_timestamp":
                int(
                    last_history[
                        "timestamp"
                    ]
                ),

            "last_history_bucket":
                last_history_bucket,

            "last_history_bucket_size":
                last_bucket_size,

            "ambiguous_last_bucket":
                ambiguous,

            "bucket_gap":
                int(
                    target_bucket
                    - last_history_bucket
                ),
        })

    samples_df = pd.DataFrame(
        samples,
        columns=EXPECTED_COLUMNS,
    )

    summary = {
        "dataset":
            dataset,

        "split":
            split_name,

        "candidate_users":
            int(candidate_users),

        "kept_samples":
            int(len(samples_df)),

        "excluded_same_bucket":
            int(
                excluded_same_bucket
            ),

        "excluded_short_history":
            int(
                excluded_short_history
            ),

        "excluded_no_history":
            int(
                excluded_no_history
            ),

        "keep_ratio":
            (
                float(
                    len(samples_df)
                    / candidate_users
                )
                if candidate_users > 0
                else 0.0
            ),

        "ambiguous_last_bucket_samples":
            int(
                samples_df[
                    "ambiguous_last_bucket"
                ].sum()
            )
            if len(samples_df)
            else 0,

        "ambiguous_last_bucket_ratio":
            (
                float(
                    samples_df[
                        "ambiguous_last_bucket"
                    ].mean()
                )
                if len(samples_df)
                else 0.0
            ),
    }

    accounted = (
        summary["kept_samples"]
        + summary["excluded_same_bucket"]
        + summary["excluded_short_history"]
        + summary["excluded_no_history"]
    )

    if accounted != candidate_users:
        raise RuntimeError(
            f"{dataset}/{split_name}: "
            "candidate accounting mismatch"
        )

    if len(samples_df):

        if not (
            samples_df[
                "target_bucket"
            ]
            >
            samples_df[
                "last_history_bucket"
            ]
        ).all():
            raise RuntimeError(
                f"{dataset}/{split_name}: "
                "strict bucket condition failed"
            )

        if not (
            samples_df[
                "history_len"
            ]
            >= MIN_HISTORY
        ).all():
            raise RuntimeError(
                f"{dataset}/{split_name}: "
                "minimum history failed"
            )

        if (
            samples_df["sample_id"]
            .duplicated()
            .any()
        ):
            raise RuntimeError(
                f"{dataset}/{split_name}: "
                "duplicate sample_id"
            )

    return (
        samples_df,
        summary,
    )


def reproduce_beauty():

    print()
    print("=" * 80)
    print(
        "BEAUTY FROZEN STRICT-PROTOCOL "
        "REPRODUCTION CHECK"
    )
    print("=" * 80)

    full = load_dataset_with_split(
        "Amazon_Beauty"
    )

    expected_counts = {
        "val": 2856,
        "test": 2966,
    }

    expected_ambiguous = {
        "val": 931,
        "test": 949,
    }

    for split_name in [
        "val",
        "test",
    ]:

        generated, summary = (
            build_strict_samples(
                full,
                "Amazon_Beauty",
                split_name,
            )
        )

        frozen_path = (
            STRICT_ROOT
            / "Amazon_Beauty"
            / (
                "validation_samples.csv"
                if split_name == "val"
                else "test_samples.csv"
            )
        )

        if not frozen_path.is_file():
            raise FileNotFoundError(
                frozen_path
            )

        frozen = pd.read_csv(
            frozen_path,
            dtype={
                "sample_id": str,
                "dataset": str,
                "split": str,
                "user_id": str,
                "target_item": str,
            },
        )

        if (
            generated.columns.tolist()
            != frozen.columns.tolist()
        ):
            raise RuntimeError(
                f"Beauty {split_name}: "
                "column mismatch"
            )

        if len(generated) != expected_counts[
            split_name
        ]:
            raise RuntimeError(
                f"Beauty {split_name}: "
                f"generated count "
                f"{len(generated)} != "
                f"{expected_counts[split_name]}"
            )

        amb = int(
            generated[
                "ambiguous_last_bucket"
            ].sum()
        )

        if amb != expected_ambiguous[
            split_name
        ]:
            raise RuntimeError(
                f"Beauty {split_name}: "
                f"ambiguous count "
                f"{amb} != "
                f"{expected_ambiguous[split_name]}"
            )

        # Exact row/column/value comparison.
        try:
            pd.testing.assert_frame_equal(
                generated.reset_index(
                    drop=True
                ),
                frozen.reset_index(
                    drop=True
                ),
                check_dtype=False,
                check_like=False,
            )
        except AssertionError as e:
            print(
                f"Beauty {split_name} "
                "exact reproduction: FAIL"
            )
            raise

        print(
            f"Beauty {split_name:4s}: "
            f"N={len(generated):,} | "
            f"ambiguous={amb:,} | "
            "EXACT FRAME PASS"
        )

    print(
        "BEAUTY STRICT-PROTOCOL "
        "REPRODUCTION: PASS"
    )


print("=" * 80)
print(
    "STEP36.7 SPORTS/TOYS STRICT PROTOCOL"
)
print("=" * 80)

# ------------------------------------------------------------
# Critical guard:
# new datasets are written only after Beauty
# frozen protocol is reproduced exactly.
# ------------------------------------------------------------

reproduce_beauty()


all_summaries = []
manifest_datasets = []


for dataset in DATASETS:

    print()
    print("=" * 80)
    print("DATASET:", dataset)
    print("=" * 80)

    full = load_dataset_with_split(
        dataset
    )

    out_dir = (
        STRICT_ROOT
        / dataset
    )

    if out_dir.exists():

        existing = list(
            out_dir.glob("*.csv")
        )

        if existing:
            raise RuntimeError(
                f"Refusing to overwrite existing "
                f"strict files: {out_dir}"
            )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_manifest = {
        "dataset":
            dataset,

        "min_history":
            MIN_HISTORY,

        "target_rule":
            (
                "final interaction inside "
                "the current validation/test split"
            ),

        "same_bucket_rule":
            (
                "exclude sample; "
                "do not modify target"
            ),

        "ordering_rule":
            "time_bucket + raw_line",

        "splits": {},
    }

    for split_name in [
        "val",
        "test",
    ]:

        samples, summary = (
            build_strict_samples(
                full,
                dataset,
                split_name,
            )
        )

        filename = (
            "validation_samples.csv"
            if split_name == "val"
            else "test_samples.csv"
        )

        out_path = (
            out_dir
            / filename
        )

        samples.to_csv(
            out_path,
            index=False,
        )

        summary[
            "output_path"
        ] = str(
            out_path.relative_to(ROOT)
        )

        summary[
            "sha256"
        ] = sha256(
            out_path
        )

        all_summaries.append(
            summary
        )

        dataset_manifest[
            "splits"
        ][split_name] = summary

        print()
        print(
            split_name.upper()
        )

        print(
            "Candidate users:",
            f"{summary['candidate_users']:,}"
        )

        print(
            "Kept samples:",
            f"{summary['kept_samples']:,}"
        )

        print(
            "Excluded same bucket:",
            f"{summary['excluded_same_bucket']:,}"
        )

        print(
            "Excluded short history:",
            f"{summary['excluded_short_history']:,}"
        )

        print(
            "Excluded no history:",
            f"{summary['excluded_no_history']:,}"
        )

        print(
            "Ambiguous last bucket:",
            f"{summary['ambiguous_last_bucket_samples']:,}",
            (
                f"("
                f"{summary['ambiguous_last_bucket_ratio']:.6f}"
                f")"
            ),
        )

        print(
            "STRICT INTEGRITY: PASS"
        )

    manifest_datasets.append(
        dataset_manifest
    )


summary_df = pd.DataFrame(
    all_summaries
)

summary_path = (
    REP_ROOT
    / "step36_7_strict_protocol_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


manifest = {
    "step":
        "36.7",

    "protocol":
        "Frozen Beauty strict protocol extension",

    "beauty_reproduction":
        "exact dataframe PASS",

    "min_history":
        MIN_HISTORY,

    "ordering_rule":
        "user_id + time_bucket + raw_line",

    "target_rule":
        (
            "canonical final interaction "
            "inside each validation/test split"
        ),

    "strict_condition":
        (
            "target_bucket > "
            "last_history_bucket"
        ),

    "datasets":
        manifest_datasets,
}


manifest_path = (
    REP_ROOT
    / "step36_7_strict_protocol_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)


print()
print("=" * 80)
print("STEP36.7 SUMMARY")
print("=" * 80)

for row in all_summaries:

    print(
        f"{row['dataset']:16s} "
        f"{row['split']:4s} "
        f"N={row['kept_samples']:,} "
        f"ambiguous="
        f"{row['ambiguous_last_bucket_samples']:,} "
        f"("
        f"{row['ambiguous_last_bucket_ratio']:.4f}"
        f")"
    )

print()
print(
    "SUMMARY:",
    summary_path
)

print(
    "MANIFEST:",
    manifest_path
)

print(
    "STEP36.7 STATUS: PASS"
)
