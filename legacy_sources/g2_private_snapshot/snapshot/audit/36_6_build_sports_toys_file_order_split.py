from pathlib import Path
import importlib.util
import hashlib
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

OLD_SPLIT_SCRIPT = (
    ROOT
    / "audit"
    / "25_2_build_diagnostic_a_splits.py"
)

REP = (
    ROOT
    / "reproducibility"
    / "step36"
)

REP.mkdir(
    parents=True,
    exist_ok=True,
)

DATASETS = {
    "Amazon_Sports": (
        ROOT
        / "processed"
        / "step36_amazon_interactions"
        / "Amazon_Sports"
        / "interactions.csv"
    ),

    "Amazon_Toys": (
        ROOT
        / "processed"
        / "step36_amazon_interactions"
        / "Amazon_Toys"
        / "interactions.csv"
    ),
}


# ============================================================
# SHA256
# ============================================================

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


# ============================================================
# Load frozen Step25.2 implementation
# ============================================================

assert OLD_SPLIT_SCRIPT.is_file(), (
    f"Missing old split script: "
    f"{OLD_SPLIT_SCRIPT}"
)

spec = importlib.util.spec_from_file_location(
    "step25_2_frozen",
    OLD_SPLIT_SCRIPT,
)

step25 = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    step25
)

assert hasattr(
    step25,
    "get_global_time_cutoffs",
)

assert hasattr(
    step25,
    "split_by_global_time",
)

print("=" * 80)
print("STEP36.6 SPORTS/TOYS FORMAL GLOBAL-TIME SPLIT")
print("=" * 80)

print(
    "Frozen split source:",
    OLD_SPLIT_SCRIPT.relative_to(ROOT),
)

print(
    "Frozen split source SHA256:",
    sha256(OLD_SPLIT_SCRIPT),
)


# ============================================================
# First validate frozen functions against Beauty
# ============================================================

print()
print("=" * 80)
print("BEAUTY FROZEN-PROTOCOL REPRODUCTION CHECK")
print("=" * 80)

beauty_full = (
    ROOT
    / "processed"
    / "diagnostic_a"
    / "Amazon_Beauty"
    / "file_order.csv"
)

beauty_split_root = (
    ROOT
    / "processed"
    / "diagnostic_a_splits"
    / "Amazon_Beauty"
    / "file_order"
)

assert beauty_full.is_file()
assert beauty_split_root.is_dir()

beauty = pd.read_csv(
    beauty_full
)

beauty_train_end, beauty_val_end = (
    step25.get_global_time_cutoffs(
        beauty
    )
)

(
    beauty_train,
    beauty_val,
    beauty_test,
) = step25.split_by_global_time(
    beauty,
    beauty_train_end,
    beauty_val_end,
)

expected_beauty = {}

for name, generated in [
    ("train", beauty_train),
    ("val", beauty_val),
    ("test", beauty_test),
]:

    existing_path = (
        beauty_split_root
        / f"{name}.csv"
    )

    assert existing_path.is_file()

    existing = pd.read_csv(
        existing_path,
        usecols=["raw_line"],
    )

    generated_lines = set(
        generated["raw_line"]
        .astype(int)
        .tolist()
    )

    existing_lines = set(
        existing["raw_line"]
        .astype(int)
        .tolist()
    )

    if len(generated) != len(existing):
        raise RuntimeError(
            f"Beauty {name} count mismatch: "
            f"{len(generated)} vs {len(existing)}"
        )

    if generated_lines != existing_lines:
        raise RuntimeError(
            f"Beauty {name} raw_line set mismatch"
        )

    expected_beauty[name] = int(
        len(existing)
    )

    print(
        f"Beauty {name:5s}: "
        f"{len(existing):,} "
        f"| exact raw_line set PASS"
    )

print(
    "Beauty train_end:",
    beauty_train_end,
)

print(
    "Beauty val_end:",
    beauty_val_end,
)

print(
    "BEAUTY FROZEN SPLIT REPRODUCTION: PASS"
)


# ============================================================
# Construct exact file_order schema
# ============================================================

EXPECTED_COLUMNS = [
    "user_id",
    "item_id",
    "rating",
    "timestamp",
    "l",
    "r",
    "g",
    "time_bucket",
    "raw_line",
]


def make_file_order(
    interaction_path,
):

    df = pd.read_csv(
        interaction_path,
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
        "source_row",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    df["timestamp"] = (
        df["timestamp"]
        .astype(np.int64)
    )

    df["source_row"] = (
        df["source_row"]
        .astype(np.int64)
    )

    if not df[
        "source_row"
    ].is_unique:
        raise RuntimeError(
            "source_row is not unique"
        )

    # Preserve original Amazon review file order.
    df = (
        df
        .sort_values(
            "source_row",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # Daily timestamp bucket:
    # [l, r] = UTC Unix day interval.
    day_start = (
        df["timestamp"]
        // 86400
        * 86400
    ).astype(np.int64)

    df["l"] = day_start

    df["r"] = (
        day_start
        + 86399
    ).astype(np.int64)

    df["g"] = "day"

    df["time_bucket"] = (
        day_start
        // 86400
    ).astype(np.int64)

    # raw_line must remain the original source-row identity.
    df["raw_line"] = (
        df["source_row"]
        .astype(np.int64)
    )

    out = df[
        EXPECTED_COLUMNS
    ].copy()

    if list(out.columns) != EXPECTED_COLUMNS:
        raise RuntimeError(
            "Output schema mismatch"
        )

    if out["raw_line"].duplicated().any():
        raise RuntimeError(
            "Duplicate raw_line detected"
        )

    if (
        out["timestamp"]
        < out["l"]
    ).any():
        raise RuntimeError(
            "timestamp < bucket left boundary"
        )

    if (
        out["timestamp"]
        > out["r"]
    ).any():
        raise RuntimeError(
            "timestamp > bucket right boundary"
        )

    return out


# ============================================================
# Safe write
# ============================================================

def safe_write_csv(
    df,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():

        old = pd.read_csv(
            path
        )

        if (
            list(old.columns)
            == list(df.columns)
            and
            len(old) == len(df)
            and
            old.equals(df)
        ):
            print(
                "Existing identical file: SKIP",
                path.relative_to(ROOT),
            )
            return

        raise RuntimeError(
            f"Refusing to overwrite "
            f"non-identical file: {path}"
        )

    df.to_csv(
        path,
        index=False,
    )


# ============================================================
# Formal processing
# ============================================================

results = []

for dataset, interaction_path in DATASETS.items():

    print()
    print("=" * 80)
    print("DATASET:", dataset)
    print("=" * 80)

    assert interaction_path.is_file(), (
        f"Missing interaction file: "
        f"{interaction_path}"
    )

    file_order = make_file_order(
        interaction_path
    )

    print(
        "Interactions:",
        f"{len(file_order):,}",
    )

    print(
        "Users:",
        f"{file_order['user_id'].nunique():,}",
    )

    print(
        "Items:",
        f"{file_order['item_id'].nunique():,}",
    )

    print(
        "Schema:",
        file_order.columns.tolist(),
    )

    # --------------------------------------------------------
    # Save full formal file_order
    # --------------------------------------------------------

    full_path = (
        ROOT
        / "processed"
        / "diagnostic_a"
        / dataset
        / "file_order.csv"
    )

    safe_write_csv(
        file_order,
        full_path,
    )

    # --------------------------------------------------------
    # Exact frozen global-time cutoffs
    # --------------------------------------------------------

    train_end, val_end = (
        step25.get_global_time_cutoffs(
            file_order
        )
    )

    print(
        "Train 80% boundary:",
        train_end,
    )

    print(
        "Validation 90% boundary:",
        val_end,
    )

    (
        train_df,
        val_df,
        test_df,
    ) = step25.split_by_global_time(
        file_order,
        train_end,
        val_end,
    )

    # --------------------------------------------------------
    # Strict split integrity
    # --------------------------------------------------------

    if (
        len(train_df)
        + len(val_df)
        + len(test_df)
        != len(file_order)
    ):
        raise RuntimeError(
            f"{dataset}: "
            "Train+Val+Test != full data"
        )

    train_lines = set(
        train_df["raw_line"].astype(int)
    )

    val_lines = set(
        val_df["raw_line"].astype(int)
    )

    test_lines = set(
        test_df["raw_line"].astype(int)
    )

    if train_lines & val_lines:
        raise RuntimeError(
            "Train/Val overlap"
        )

    if train_lines & test_lines:
        raise RuntimeError(
            "Train/Test overlap"
        )

    if val_lines & test_lines:
        raise RuntimeError(
            "Val/Test overlap"
        )

    if (
        train_lines
        | val_lines
        | test_lines
    ) != set(
        file_order["raw_line"]
        .astype(int)
    ):
        raise RuntimeError(
            "Split raw_line union mismatch"
        )

    if not (
        train_df["timestamp"]
        <= train_end
    ).all():
        raise RuntimeError(
            "Invalid train timestamp"
        )

    if not (
        (
            val_df["timestamp"]
            > train_end
        )
        &
        (
            val_df["timestamp"]
            <= val_end
        )
    ).all():
        raise RuntimeError(
            "Invalid validation timestamp"
        )

    if not (
        test_df["timestamp"]
        > val_end
    ).all():
        raise RuntimeError(
            "Invalid test timestamp"
        )

    # --------------------------------------------------------
    # Save formal split
    # --------------------------------------------------------

    split_root = (
        ROOT
        / "processed"
        / "diagnostic_a_splits"
        / dataset
        / "file_order"
    )

    safe_write_csv(
        train_df,
        split_root / "train.csv",
    )

    safe_write_csv(
        val_df,
        split_root / "val.csv",
    )

    safe_write_csv(
        test_df,
        split_root / "test.csv",
    )

    record = {
        "dataset":
            dataset,

        "source_interactions":
            str(
                interaction_path
                .relative_to(ROOT)
            ),

        "source_interactions_sha256":
            sha256(interaction_path),

        "file_order_path":
            str(
                full_path
                .relative_to(ROOT)
            ),

        "file_order_sha256":
            sha256(full_path),

        "rows":
            int(len(file_order)),

        "users":
            int(
                file_order[
                    "user_id"
                ].nunique()
            ),

        "items":
            int(
                file_order[
                    "item_id"
                ].nunique()
            ),

        "train_end_timestamp":
            int(train_end),

        "val_end_timestamp":
            int(val_end),

        "train_count":
            int(len(train_df)),

        "val_count":
            int(len(val_df)),

        "test_count":
            int(len(test_df)),

        "train_ratio":
            float(
                len(train_df)
                / len(file_order)
            ),

        "val_ratio":
            float(
                len(val_df)
                / len(file_order)
            ),

        "test_ratio":
            float(
                len(test_df)
                / len(file_order)
            ),

        "train_sha256":
            sha256(
                split_root
                / "train.csv"
            ),

        "val_sha256":
            sha256(
                split_root
                / "val.csv"
            ),

        "test_sha256":
            sha256(
                split_root
                / "test.csv"
            ),
    }

    results.append(record)

    print()
    print(
        "Train:",
        f"{len(train_df):,}",
        f"({len(train_df)/len(file_order):.6f})",
    )

    print(
        "Validation:",
        f"{len(val_df):,}",
        f"({len(val_df)/len(file_order):.6f})",
    )

    print(
        "Test:",
        f"{len(test_df):,}",
        f"({len(test_df)/len(file_order):.6f})",
    )

    print(
        "SPLIT INTEGRITY: PASS"
    )


# ============================================================
# Manifest
# ============================================================

manifest = {
    "step":
        "36.6",

    "purpose":
        (
            "Extend frozen Beauty global timestamp "
            "split protocol to Amazon Sports and Toys"
        ),

    "frozen_split_source":
        str(
            OLD_SPLIT_SCRIPT
            .relative_to(ROOT)
        ),

    "frozen_split_source_sha256":
        sha256(
            OLD_SPLIT_SCRIPT
        ),

    "split_protocol":
        {
            "train":
                "timestamp <= train_end",

            "validation":
                (
                    "train_end < timestamp "
                    "<= val_end"
                ),

            "test":
                "timestamp > val_end",

            "cutoff_functions":
                [
                    "get_global_time_cutoffs",
                    "split_by_global_time",
                ],

            "nominal_cutoffs":
                {
                    "train": 0.80,
                    "validation": 0.90,
                },
        },

    "beauty_reproduction_check":
        {
            "status": "PASS",
            "counts":
                expected_beauty,
            "train_end_timestamp":
                int(beauty_train_end),
            "val_end_timestamp":
                int(beauty_val_end),
        },

    "datasets":
        results,
}

manifest_path = (
    REP
    / "step36_6_sports_toys_split_manifest.json"
)

if manifest_path.exists():
    raise RuntimeError(
        f"Refusing to overwrite manifest: "
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
print("STEP36.6 SUMMARY")
print("=" * 80)

for r in results:

    print(
        f"{r['dataset']:16s} "
        f"N={r['rows']:,} "
        f"train={r['train_count']:,} "
        f"val={r['val_count']:,} "
        f"test={r['test_count']:,}"
    )

print()
print(
    "MANIFEST:",
    manifest_path,
)

print(
    "STEP36.6 STATUS: PASS"
)
