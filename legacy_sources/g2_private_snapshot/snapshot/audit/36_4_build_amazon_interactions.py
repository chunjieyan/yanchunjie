from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import gzip
import ast
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]

OUT_ROOT = ROOT / "processed" / "step36_amazon_interactions"
REP = ROOT / "reproducibility" / "step36"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "Amazon_Sports": (
        ROOT / "data" / "Amazon_Sports"
        / "reviews_Sports_and_Outdoors_5.json.gz"
    ),
    "Amazon_Toys": (
        ROOT / "data" / "Amazon_Toys"
        / "reviews_Toys_and_Games_5.json.gz"
    ),
}


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_amazon(path):
    records = []

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as f:

        for source_row, line in enumerate(f):

            line = line.strip()

            if not line:
                continue

            obj = ast.literal_eval(line)

            records.append({
                "user_id": str(obj["reviewerID"]),
                "item_id": str(obj["asin"]),
                "rating": float(obj["overall"]),
                "timestamp": int(obj["unixReviewTime"]),
                "source_row": int(source_row),
            })

    return pd.DataFrame(records)


def audit_dataset(df):
    assert len(df) > 0

    assert not df[
        ["user_id", "item_id", "timestamp", "source_row"]
    ].isna().any().any()

    assert df["source_row"].is_unique

    user_counts = df.groupby("user_id").size()
    item_counts = df.groupby("item_id").size()

    # Published Amazon 5-core files should satisfy >=5
    min_user_interactions = int(user_counts.min())
    min_item_interactions = int(item_counts.min())

    # Number of user-timestamp groups.
    bucket_sizes = (
        df.groupby(["user_id", "timestamp"])
        .size()
    )

    tied_buckets = bucket_sizes[bucket_sizes > 1]

    tied_interactions = int(
        tied_buckets.sum()
    )

    users_with_any_tie = int(
        tied_buckets
        .reset_index()["user_id"]
        .nunique()
    )

    # Latest timestamp bucket for each user.
    latest_ts = (
        df.groupby("user_id")["timestamp"]
        .max()
        .rename("latest_timestamp")
    )

    latest = df.merge(
        latest_ts,
        left_on="user_id",
        right_index=True,
        how="left",
    )

    latest = latest[
        latest["timestamp"]
        == latest["latest_timestamp"]
    ]

    latest_bucket_sizes = (
        latest.groupby("user_id").size()
    )

    users_with_tied_last_bucket = int(
        (latest_bucket_sizes > 1).sum()
    )

    last_bucket_size_distribution = {
        str(int(k)): int(v)
        for k, v in
        latest_bucket_sizes.value_counts()
        .sort_index()
        .items()
    }

    return {
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "items": int(df["item_id"].nunique()),

        "min_user_interactions":
            min_user_interactions,

        "min_item_interactions":
            min_item_interactions,

        "timestamp_min":
            int(df["timestamp"].min()),

        "timestamp_max":
            int(df["timestamp"].max()),

        "unique_user_timestamp_buckets":
            int(len(bucket_sizes)),

        "tied_user_timestamp_buckets":
            int(len(tied_buckets)),

        "tied_interactions":
            tied_interactions,

        "tied_interaction_ratio":
            float(tied_interactions / len(df)),

        "users_with_any_timestamp_tie":
            users_with_any_tie,

        "users_with_any_timestamp_tie_ratio":
            float(
                users_with_any_tie
                / df["user_id"].nunique()
            ),

        "users_with_tied_last_bucket":
            users_with_tied_last_bucket,

        "users_with_tied_last_bucket_ratio":
            float(
                users_with_tied_last_bucket
                / df["user_id"].nunique()
            ),

        "last_bucket_size_distribution":
            last_bucket_size_distribution,
    }


print("=" * 78)
print("STEP36.4 AMAZON INTERACTION NORMALIZATION")
print("=" * 78)

summary = []

for dataset, raw_path in DATASETS.items():

    print()
    print("=" * 78)
    print("DATASET:", dataset)
    print("=" * 78)

    assert raw_path.is_file(), (
        f"Missing raw file: {raw_path}"
    )

    print("Loading:", raw_path)

    df = load_amazon(raw_path)

    stats = audit_dataset(df)

    # Confirm published 5-core property.
    assert stats["min_user_interactions"] >= 5
    assert stats["min_item_interactions"] >= 5

    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_csv = out_dir / "interactions.csv"

    if out_csv.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {out_csv}"
        )

    df.to_csv(
        out_csv,
        index=False,
    )

    record = {
        "dataset": dataset,
        "raw_path":
            str(raw_path.relative_to(ROOT)),

        "raw_sha256":
            sha256(raw_path),

        "output_path":
            str(out_csv.relative_to(ROOT)),

        "output_sha256":
            sha256(out_csv),

        **stats,
    }

    summary.append(record)

    print("Rows:", f"{stats['rows']:,}")
    print("Users:", f"{stats['users']:,}")
    print("Items:", f"{stats['items']:,}")

    print(
        "Min interactions/user:",
        stats["min_user_interactions"],
    )

    print(
        "Min interactions/item:",
        stats["min_item_interactions"],
    )

    print(
        "Tied interaction ratio:",
        f"{stats['tied_interaction_ratio']:.6f}",
    )

    print(
        "Users with any timestamp tie:",
        f"{stats['users_with_any_timestamp_tie_ratio']:.6f}",
    )

    print(
        "Users with tied last bucket:",
        stats["users_with_tied_last_bucket"],
        "/",
        stats["users"],
        "=",
        f"{stats['users_with_tied_last_bucket_ratio']:.6f}",
    )

    print("SAVED:", out_csv)


manifest = {
    "step": "36.4",
    "operation":
        "Normalize Amazon 2014 5-core reviews "
        "without changing original ordering",

    "columns": [
        "user_id",
        "item_id",
        "rating",
        "timestamp",
        "source_row",
    ],

    "datasets": summary,
}

manifest_path = (
    REP /
    "step36_4_amazon_interaction_manifest.json"
)

if manifest_path.exists():
    raise RuntimeError(
        f"Refusing to overwrite: {manifest_path}"
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
print("=" * 78)
print("STEP36.4 SUMMARY")
print("=" * 78)

for r in summary:

    print(
        f"{r['dataset']:16s} "
        f"N={r['rows']:,} "
        f"users={r['users']:,} "
        f"items={r['items']:,} "
        f"last_tie_users="
        f"{r['users_with_tied_last_bucket']:,} "
        f"({r['users_with_tied_last_bucket_ratio']:.4f})"
    )

print()
print("MANIFEST:", manifest_path)
print("STEP36.4 STATUS: PASS")
