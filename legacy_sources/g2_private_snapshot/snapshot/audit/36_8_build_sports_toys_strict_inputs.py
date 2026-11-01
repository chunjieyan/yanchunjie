from pathlib import Path
import pandas as pd
import numpy as np
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]

REP = (
    ROOT
    / "reproducibility"
    / "step36"
)

REP.mkdir(
    parents=True,
    exist_ok=True,
)

MAX_LEN = 50

DATASETS = [
    "Amazon_Sports",
    "Amazon_Toys",
]

OUTPUT_COLUMNS = [
    "sample_id",
    "user_id",
    "target_item",
    "target_raw_line",
    "target_timestamp",
    "target_bucket",
    "ambiguous_last_bucket",
    "last_history_bucket_size",
    "fixed_history_len",
    "sequence_items",
    "sequence_raw_lines",
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


def load_full_dataset(dataset):

    path = (
        ROOT
        / "processed"
        / "diagnostic_a"
        / dataset
        / "file_order.csv"
    )

    assert path.is_file(), path

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
            f"{dataset}: missing columns "
            f"{missing}"
        )

    df["raw_line"] = (
        df["raw_line"]
        .astype(np.int64)
    )

    df["time_bucket"] = (
        df["time_bucket"]
        .astype(np.int64)
    )

    return df, path


def load_strict_samples(
    dataset,
    split_name,
):

    filename = (
        "validation_samples.csv"
        if split_name == "val"
        else "test_samples.csv"
    )

    path = (
        ROOT
        / "processed"
        / "strict_protocol"
        / dataset
        / filename
    )

    assert path.is_file(), path

    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
        },
    )

    return df, path


def build_user_groups(full_df):

    ordered = (
        full_df
        .sort_values(
            [
                "user_id",
                "time_bucket",
                "raw_line",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    groups = {}

    for user_id, group in ordered.groupby(
        "user_id",
        sort=False,
    ):

        groups[
            str(user_id)
        ] = (
            group
            .reset_index(drop=True)
        )

    return groups


def build_inputs(
    dataset,
    split_name,
):

    full_df, full_path = (
        load_full_dataset(
            dataset
        )
    )

    samples, strict_path = (
        load_strict_samples(
            dataset,
            split_name,
        )
    )

    groups = build_user_groups(
        full_df
    )

    records = []

    max_last_bucket_size = 0
    truncated_histories = 0

    for row in samples.itertuples(
        index=False
    ):

        user_id = str(
            row.user_id
        )

        if user_id not in groups:
            raise RuntimeError(
                f"{row.sample_id}: "
                f"user missing"
            )

        group = groups[
            user_id
        ]

        target_raw = int(
            row.target_raw_line
        )

        positions = np.flatnonzero(
            group["raw_line"]
            .to_numpy(dtype=np.int64)
            == target_raw
        )

        if len(positions) != 1:
            raise RuntimeError(
                f"{row.sample_id}: "
                f"target raw_line match="
                f"{len(positions)}"
            )

        target_idx = int(
            positions[0]
        )

        target = group.iloc[
            target_idx
        ]

        history = (
            group.iloc[
                :target_idx
            ]
            .copy()
        )

        if len(history) != int(
            row.history_len
        ):
            raise RuntimeError(
                f"{row.sample_id}: "
                f"history_len mismatch "
                f"{len(history)} != "
                f"{row.history_len}"
            )

        if len(history) < 5:
            raise RuntimeError(
                f"{row.sample_id}: "
                f"history shorter than 5"
            )

        if str(
            target["item_id"]
        ) != str(
            row.target_item
        ):
            raise RuntimeError(
                f"{row.sample_id}: "
                "target item mismatch"
            )

        if int(
            target["timestamp"]
        ) != int(
            row.target_timestamp
        ):
            raise RuntimeError(
                f"{row.sample_id}: "
                "target timestamp mismatch"
            )

        if int(
            target["time_bucket"]
        ) != int(
            row.target_bucket
        ):
            raise RuntimeError(
                f"{row.sample_id}: "
                "target bucket mismatch"
            )

        last_history_bucket = int(
            history.iloc[-1][
                "time_bucket"
            ]
        )

        if last_history_bucket != int(
            row.last_history_bucket
        ):
            raise RuntimeError(
                f"{row.sample_id}: "
                "last history bucket mismatch"
            )

        last_bucket_count = int(
            (
                history[
                    "time_bucket"
                ]
                ==
                last_history_bucket
            ).sum()
        )

        if last_bucket_count != int(
            row.last_history_bucket_size
        ):
            raise RuntimeError(
                f"{row.sample_id}: "
                "last bucket size mismatch "
                f"{last_bucket_count} != "
                f"{row.last_history_bucket_size}"
            )

        max_last_bucket_size = max(
            max_last_bucket_size,
            last_bucket_count,
        )

        # Same protocol as Beauty:
        # keep only the most recent MAX_LEN events.
        fixed_history = (
            history
            .tail(MAX_LEN)
            .copy()
        )

        if len(history) > MAX_LEN:
            truncated_histories += 1

        fixed_history_len = int(
            len(fixed_history)
        )

        # Required by later last-bucket-only permutation.
        # The complete final bucket must still fit in
        # the fixed history.
        if (
            last_bucket_count
            > fixed_history_len
        ):
            raise RuntimeError(
                f"{row.sample_id}: "
                f"last bucket size "
                f"{last_bucket_count} "
                f"> fixed history "
                f"{fixed_history_len}"
            )

        sequence_items = [
            str(x)
            for x in
            fixed_history[
                "item_id"
            ].tolist()
        ]

        sequence_raw_lines = [
            int(x)
            for x in
            fixed_history[
                "raw_line"
            ].tolist()
        ]

        if target_raw in (
            sequence_raw_lines
        ):
            raise RuntimeError(
                f"{row.sample_id}: "
                "target leakage"
            )

        if len(
            sequence_items
        ) != fixed_history_len:
            raise RuntimeError(
                "sequence length mismatch"
            )

        records.append({
            "sample_id":
                str(row.sample_id),

            "user_id":
                user_id,

            "target_item":
                str(row.target_item),

            "target_raw_line":
                target_raw,

            "target_timestamp":
                int(
                    row.target_timestamp
                ),

            "target_bucket":
                int(
                    row.target_bucket
                ),

            "ambiguous_last_bucket":
                int(
                    row.ambiguous_last_bucket
                ),

            "last_history_bucket_size":
                last_bucket_count,

            "fixed_history_len":
                fixed_history_len,

            "sequence_items":
                json.dumps(
                    sequence_items,
                    ensure_ascii=False,
                ),

            "sequence_raw_lines":
                json.dumps(
                    sequence_raw_lines,
                ),
        })

    out = pd.DataFrame(
        records,
        columns=OUTPUT_COLUMNS,
    )

    if len(out) != len(samples):
        raise RuntimeError(
            f"{dataset} {split_name}: "
            "sample count mismatch"
        )

    if out["sample_id"].duplicated().any():
        raise RuntimeError(
            f"{dataset} {split_name}: "
            "duplicate sample_id"
        )

    return {
        "df": out,
        "full_path": full_path,
        "strict_path": strict_path,
        "max_last_bucket_size":
            int(max_last_bucket_size),
        "truncated_histories":
            int(truncated_histories),
    }


def compare_exact(
    generated,
    existing_path,
    label,
):

    assert existing_path.is_file(), (
        existing_path
    )

    existing = pd.read_csv(
        existing_path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_items": str,
            "sequence_raw_lines": str,
        },
    )

    if list(
        existing.columns
    ) != OUTPUT_COLUMNS:
        raise RuntimeError(
            f"{label}: existing schema mismatch"
        )

    if len(
        generated
    ) != len(
        existing
    ):
        raise RuntimeError(
            f"{label}: row count mismatch "
            f"{len(generated)} != "
            f"{len(existing)}"
        )

    for col in OUTPUT_COLUMNS:

        a = (
            generated[col]
            .astype(str)
            .tolist()
        )

        b = (
            existing[col]
            .astype(str)
            .tolist()
        )

        if a != b:

            mismatch = next(
                i
                for i, (x, y)
                in enumerate(zip(a, b))
                if x != y
            )

            raise RuntimeError(
                f"{label}: column {col} "
                f"mismatch at row "
                f"{mismatch}: "
                f"{a[mismatch]!r} != "
                f"{b[mismatch]!r}"
            )

    print(
        f"{label}: N={len(existing):,} "
        "| EXACT FRAME PASS"
    )


def safe_write(
    df,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():

        old = pd.read_csv(
            path,
            dtype=str,
        )

        new = df.astype(str)

        if (
            list(old.columns)
            == list(new.columns)
            and
            old.equals(new)
        ):
            print(
                "Existing identical file: SKIP",
                path.relative_to(ROOT),
            )
            return

        raise RuntimeError(
            f"Refusing to overwrite "
            f"non-identical file: "
            f"{path}"
        )

    df.to_csv(
        path,
        index=False,
    )


print("=" * 80)
print("STEP36.8 SPORTS/TOYS STRICT FIXED-HISTORY INPUTS")
print("=" * 80)


# ============================================================
# Beauty frozen reproduction
# ============================================================

print()
print("=" * 80)
print("BEAUTY STRICT INPUT REPRODUCTION CHECK")
print("=" * 80)

beauty_results = {}

for split_name in [
    "val",
    "test",
]:

    built = build_inputs(
        "Amazon_Beauty",
        split_name,
    )

    existing_path = (
        ROOT
        / "processed"
        / "strict_diagnostic_a"
        / "Amazon_Beauty"
        / "file_order"
        / f"{split_name}.csv"
    )

    compare_exact(
        built["df"],
        existing_path,
        f"Beauty {split_name}",
    )

    beauty_results[
        split_name
    ] = {
        "samples":
            int(
                len(
                    built["df"]
                )
            ),
        "max_last_bucket_size":
            built[
                "max_last_bucket_size"
            ],
        "truncated_histories":
            built[
                "truncated_histories"
            ],
    }

print(
    "BEAUTY STRICT INPUT REPRODUCTION: PASS"
)


# ============================================================
# Sports / Toys
# ============================================================

summary = []

for dataset in DATASETS:

    print()
    print("=" * 80)
    print("DATASET:", dataset)
    print("=" * 80)

    for split_name in [
        "val",
        "test",
    ]:

        built = build_inputs(
            dataset,
            split_name,
        )

        out_path = (
            ROOT
            / "processed"
            / "strict_diagnostic_a"
            / dataset
            / "file_order"
            / f"{split_name}.csv"
        )

        safe_write(
            built["df"],
            out_path,
        )

        df = built["df"]

        ambiguous = int(
            df[
                "ambiguous_last_bucket"
            ].sum()
        )

        record = {
            "dataset":
                dataset,

            "split":
                split_name,

            "samples":
                int(len(df)),

            "ambiguous_last_bucket_samples":
                ambiguous,

            "ambiguous_last_bucket_ratio":
                float(
                    ambiguous
                    / len(df)
                ),

            "max_last_bucket_size":
                built[
                    "max_last_bucket_size"
                ],

            "truncated_histories":
                built[
                    "truncated_histories"
                ],

            "output_path":
                str(
                    out_path
                    .relative_to(ROOT)
                ),

            "output_sha256":
                sha256(
                    out_path
                ),
        }

        summary.append(
            record
        )

        print()
        print(
            split_name.upper(),
        )

        print(
            "Samples:",
            f"{len(df):,}",
        )

        print(
            "Ambiguous:",
            f"{ambiguous:,}",
            (
                f"({ambiguous / len(df):.6f})"
            ),
        )

        print(
            "Max last bucket size:",
            built[
                "max_last_bucket_size"
            ],
        )

        print(
            "Histories truncated to 50:",
            f"{built['truncated_histories']:,}",
        )

        print(
            "OUTPUT:",
            out_path,
        )

        print(
            "STRICT INPUT INTEGRITY: PASS"
        )


summary_df = pd.DataFrame(
    summary
)

summary_path = (
    REP
    / "step36_8_strict_input_summary.csv"
)

if summary_path.exists():
    raise RuntimeError(
        f"Refusing to overwrite: "
        f"{summary_path}"
    )

summary_df.to_csv(
    summary_path,
    index=False,
)

manifest = {
    "step": "36.8",

    "max_len": MAX_LEN,

    "ordering":
        "time_bucket + raw_line",

    "history":
        (
            "all events before fixed target; "
            "retain most recent 50"
        ),

    "beauty_reproduction":
        {
            "status": "PASS",
            "splits":
                beauty_results,
        },

    "datasets":
        summary,
}

manifest_path = (
    REP
    / "step36_8_strict_input_manifest.json"
)

if manifest_path.exists():
    raise RuntimeError(
        f"Refusing to overwrite: "
        f"{manifest_path}"
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
print("STEP36.8 SUMMARY")
print("=" * 80)

for r in summary:

    print(
        f"{r['dataset']:16s} "
        f"{r['split']:4s} "
        f"N={r['samples']:,} "
        f"ambiguous="
        f"{r['ambiguous_last_bucket_samples']:,} "
        f"({r['ambiguous_last_bucket_ratio']:.4f}) "
        f"max_bucket="
        f"{r['max_last_bucket_size']}"
    )

print()
print(
    "SUMMARY:",
    summary_path,
)

print(
    "MANIFEST:",
    manifest_path,
)

print(
    "STEP36.8 STATUS: PASS"
)
