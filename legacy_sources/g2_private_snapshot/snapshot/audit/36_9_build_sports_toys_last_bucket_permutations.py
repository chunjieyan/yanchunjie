from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

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
    / "step36"
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
    "Amazon_Sports",
    "Amazon_Toys",
]

PERMUTATION_SEEDS = [
    101,
    202,
    303,
    404,
    505,
]

EXPECTED_AMBIGUOUS = {
    "Amazon_Beauty": 949,
    "Amazon_Sports": 1676,
    "Amazon_Toys": 868,
}

TARGET_COLUMNS = [
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


def load_source(dataset):

    path = (
        INPUT_ROOT
        / dataset
        / "file_order"
        / "test.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    source = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_items": str,
            "sequence_raw_lines": str,
        },
    )

    required = set(
        TARGET_COLUMNS
        +
        [
            "sequence_items",
            "sequence_raw_lines",
        ]
    )

    missing = (
        required
        - set(source.columns)
    )

    if missing:
        raise RuntimeError(
            f"{dataset}: missing "
            f"{missing}"
        )

    if source[
        "sample_id"
    ].duplicated().any():

        raise RuntimeError(
            f"{dataset}: "
            "duplicate sample_id"
        )

    ambiguous = int(
        source[
            "ambiguous_last_bucket"
        ].sum()
    )

    if (
        dataset
        in EXPECTED_AMBIGUOUS
        and
        ambiguous
        != EXPECTED_AMBIGUOUS[
            dataset
        ]
    ):
        raise RuntimeError(
            f"{dataset}: ambiguous count "
            f"{ambiguous} != "
            f"{EXPECTED_AMBIGUOUS[dataset]}"
        )

    return (
        source,
        path,
    )


def build_one_permutation(
    dataset,
    source,
    permutation_seed,
):

    output = source.copy()

    changed_samples = 0
    changed_ambiguous = 0

    prefix_changed = 0
    content_changed = 0
    nonambiguous_changed = 0

    target_changed = 0
    target_leakage = 0

    identity_permutations = 0

    new_items_col = []
    new_raw_col = []

    changed_ids = []

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

        if (
            len(seq_raw)
            != fixed_history_len
        ):
            raise RuntimeError(
                f"{dataset}/"
                f"{row.sample_id}: "
                "history length mismatch"
            )

        ambiguous = (
            int(
                row.ambiguous_last_bucket
            )
            == 1
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
                for i
                in permutation
            ]

            shuffled_raw = [
                tail_raw[i]
                for i
                in permutation
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
                identity_permutations += 1

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

            changed_ids.append(
                str(
                    row.sample_id
                )
            )

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

    output[
        "sequence_items"
    ] = new_items_col

    output[
        "sequence_raw_lines"
    ] = new_raw_col

    # Target metadata must be unchanged.
    for col in TARGET_COLUMNS:

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

    if not pass_flag:
        raise RuntimeError(
            f"{dataset}/seed"
            f"{permutation_seed}: "
            "permutation integrity failed"
        )

    stats = {
        "dataset":
            dataset,

        "permutation_seed":
            int(
                permutation_seed
            ),

        "samples":
            int(
                len(output)
            ),

        "ambiguous_samples":
            int(
                source[
                    "ambiguous_last_bucket"
                ].sum()
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
                identity_permutations
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
    }

    return (
        output,
        stats,
        changed_ids,
    )


def compare_frozen_beauty():

    print()
    print("=" * 88)
    print(
        "BEAUTY STEP35.5 "
        "FROZEN PERMUTATION REPRODUCTION"
    )
    print("=" * 88)

    source, source_path = (
        load_source(
            "Amazon_Beauty"
        )
    )

    # Frozen Step35.5 definition:
    # actually_permuted means at least two of the
    # five tested model-input sequences differ.
    variant_sequences = {
        str(sample_id): []
        for sample_id in
        source["sample_id"]
        .astype(str)
        .tolist()
    }

    results = []

    for permutation_seed in (
        PERMUTATION_SEEDS
    ):

        generated, stats, changed = (
            build_one_permutation(
                "Amazon_Beauty",
                source,
                permutation_seed,
            )
        )

        frozen_path = (
            OUTPUT_ROOT
            / "Amazon_Beauty"
            / (
                f"perm_seed"
                f"{permutation_seed}"
            )
            / "test.csv"
        )

        if not frozen_path.is_file():
            raise FileNotFoundError(
                frozen_path
            )

        frozen = pd.read_csv(
            frozen_path,
            dtype={
                "sample_id": str,
                "user_id": str,
                "target_item": str,
                "sequence_items": str,
                "sequence_raw_lines": str,
            },
        )

        if (
            generated.columns.tolist()
            !=
            frozen.columns.tolist()
        ):
            raise RuntimeError(
                "Beauty frozen "
                "column mismatch"
            )

        if (
            len(generated)
            !=
            len(frozen)
        ):
            raise RuntimeError(
                "Beauty frozen "
                "row-count mismatch"
            )

        for col in (
            generated.columns
        ):

            a = (
                generated[col]
                .fillna("__NA__")
                .astype(str)
                .tolist()
            )

            b = (
                frozen[col]
                .fillna("__NA__")
                .astype(str)
                .tolist()
            )

            if a != b:

                idx = next(
                    i
                    for i, (x, y)
                    in enumerate(
                        zip(a, b)
                    )
                    if x != y
                )

                raise RuntimeError(
                    f"Beauty seed"
                    f"{permutation_seed}: "
                    f"{col} mismatch "
                    f"row {idx}"
                )

        for sample_id, sequence_items in zip(
            generated["sample_id"],
            generated["sequence_items"],
        ):
            variant_sequences[
                str(sample_id)
            ].append(
                tuple(
                    str(x)
                    for x in
                    parse_list(
                        sequence_items
                    )
                )
            )

        results.append(
            stats
        )

        print(
            f"Beauty seed"
            f"{permutation_seed}: "
            f"changed="
            f"{stats['changed_samples']:,} "
            "| EXACT FRAME PASS"
        )

    # Frozen Step35.5:
    # At least two of the five tested inputs differ.
    primary_ids = sorted(
        sample_id
        for sample_id, sequences
        in variant_sequences.items()
        if len(
            set(
                sequences
            )
        ) > 1
    )

    if len(
        primary_ids
    ) != 922:
        raise RuntimeError(
            "Beauty primary population "
            f"mismatch: "
            f"{len(primary_ids)} != 922"
        )

    print(
        "Beauty actually_permuted:",
        f"{len(primary_ids):,}",
        "| expected=922 | PASS"
    )

    print(
        "BEAUTY STEP35.5 "
        "PERMUTATION REPRODUCTION: PASS"
    )

    return {
        "source":
            str(
                source_path
                .relative_to(ROOT)
            ),

        "actually_permuted":
            int(
                len(primary_ids)
            ),

        "permutations":
            results,
    }


print("=" * 88)
print(
    "STEP36.9-B SPORTS/TOYS "
    "LAST-BUCKET-ONLY PERMUTATIONS"
)
print("=" * 88)


# ============================================================
# Critical frozen Beauty reproduction
# ============================================================

beauty_reproduction = (
    compare_frozen_beauty()
)


# ============================================================
# Sports / Toys generation
# ============================================================

all_seed_stats = []
dataset_results = []

for dataset in DATASETS:

    print()
    print("=" * 88)
    print("DATASET:", dataset)
    print("=" * 88)

    source, source_path = (
        load_source(
            dataset
        )
    )

    ambiguous_ids = set(
        source.loc[
            source[
                "ambiguous_last_bucket"
            ]
            == 1,
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    # Collect the five actual model-input
    # sequence variants for every sample.
    variant_sequences = {
        str(sample_id): []
        for sample_id in
        source["sample_id"]
        .astype(str)
        .tolist()
    }

    per_seed = []

    for permutation_seed in (
        PERMUTATION_SEEDS
    ):

        print()
        print(
            "Permutation seed:",
            permutation_seed
        )

        output, stats, changed = (
            build_one_permutation(
                dataset,
                source,
                permutation_seed,
            )
        )

        for sample_id, sequence_items in zip(
            output["sample_id"],
            output["sequence_items"],
        ):
            variant_sequences[
                str(sample_id)
            ].append(
                tuple(
                    str(x)
                    for x in
                    parse_list(
                        sequence_items
                    )
                )
            )

        out_dir = (
            OUTPUT_ROOT
            / dataset
            / (
                f"perm_seed"
                f"{permutation_seed}"
            )
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_path = (
            out_dir
            / "test.csv"
        )

        if out_path.exists():

            existing = pd.read_csv(
                out_path,
                dtype=str,
            )

            candidate = (
                output
                .astype(str)
            )

            if (
                existing.columns.tolist()
                ==
                candidate.columns.tolist()
                and
                existing.equals(
                    candidate
                )
            ):
                print(
                    "Existing identical "
                    "file: SKIP"
                )

            else:
                raise RuntimeError(
                    f"Refusing to overwrite "
                    f"non-identical file: "
                    f"{out_path}"
                )

        else:

            output.to_csv(
                out_path,
                index=False,
                encoding="utf-8-sig",
            )

        stats[
            "output"
        ] = str(
            out_path
            .relative_to(ROOT)
        )

        stats[
            "output_sha256"
        ] = sha256(
            out_path
        )

        all_seed_stats.append(
            stats
        )

        per_seed.append(
            stats
        )

        print(
            "Samples:",
            f"{stats['samples']:,}"
        )

        print(
            "Changed:",
            f"{stats['changed_samples']:,}"
        )

        print(
            "Identity permutations:",
            f"{stats['identity_permutations']:,}"
        )

        print(
            "Integrity: PASS"
        )

    # --------------------------------------------------------
    # Frozen Step35.5 primary population:
    # at least two of the five tested model-input
    # sequence variants differ.
    # --------------------------------------------------------

    primary_ids = sorted(
        sample_id
        for sample_id, sequences
        in variant_sequences.items()
        if len(
            set(
                sequences
            )
        ) > 1
    )

    primary_id_set = set(
        primary_ids
    )

    if not primary_id_set.issubset(
        ambiguous_ids
    ):
        raise RuntimeError(
            f"{dataset}: "
            "nonambiguous sample in "
            "actually_permuted population"
        )

    primary_path = (
        REPRO_DIR
        / (
            f"step36_9_"
            f"{dataset}_"
            f"primary_sample_ids.csv"
        )
    )

    if primary_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite: "
            f"{primary_path}"
        )

    pd.DataFrame(
        {
            "sample_id":
                primary_ids
        }
    ).to_csv(
        primary_path,
        index=False,
    )

    ambiguous_n = int(
        len(
            ambiguous_ids
        )
    )

    primary_n = int(
        len(
            primary_ids
        )
    )

    never_changed = (
        ambiguous_n
        -
        primary_n
    )

    dataset_record = {
        "dataset":
            dataset,

        "test_samples":
            int(
                len(source)
            ),

        "ambiguous_last_bucket_samples":
            ambiguous_n,

        "actually_permuted_samples":
            primary_n,

        "ambiguous_never_changed":
            int(
                never_changed
            ),

        "actually_permuted_ratio_of_ambiguous":
            float(
                primary_n
                / ambiguous_n
            ),

        "primary_sample_ids":
            str(
                primary_path
                .relative_to(ROOT)
            ),

        "primary_sample_ids_sha256":
            sha256(
                primary_path
            ),

        "permutations":
            per_seed,
    }

    dataset_results.append(
        dataset_record
    )

    print()
    print(
        "Ambiguous last bucket:",
        f"{ambiguous_n:,}"
    )

    print(
        "Actually permuted:",
        f"{primary_n:,}"
    )

    print(
        "Ambiguous never changed:",
        f"{never_changed:,}"
    )

    print(
        "PRIMARY POPULATION: PASS"
    )


# ============================================================
# Save summaries
# ============================================================

seed_summary_path = (
    REPRO_DIR
    / "step36_9_permutation_seed_summary.csv"
)

if seed_summary_path.exists():
    raise RuntimeError(
        f"Refusing to overwrite: "
        f"{seed_summary_path}"
    )

pd.DataFrame(
    all_seed_stats
).to_csv(
    seed_summary_path,
    index=False,
)


primary_summary = pd.DataFrame(
    [
        {
            "dataset":
                r["dataset"],

            "test_samples":
                r[
                    "test_samples"
                ],

            "ambiguous_last_bucket_samples":
                r[
                    "ambiguous_last_bucket_samples"
                ],

            "actually_permuted_samples":
                r[
                    "actually_permuted_samples"
                ],

            "ambiguous_never_changed":
                r[
                    "ambiguous_never_changed"
                ],

            "actually_permuted_ratio_of_ambiguous":
                r[
                    "actually_permuted_ratio_of_ambiguous"
                ],
        }
        for r in dataset_results
    ]
)

primary_summary_path = (
    REPRO_DIR
    / "step36_9_primary_population_summary.csv"
)

if primary_summary_path.exists():
    raise RuntimeError(
        f"Refusing to overwrite: "
        f"{primary_summary_path}"
    )

primary_summary.to_csv(
    primary_summary_path,
    index=False,
)


manifest = {
    "step":
        "36.9",

    "protocol":
        "frozen Step35.5 last-bucket-only permutation",

    "permutation_seeds":
        PERMUTATION_SEEDS,

    "deterministic_seed":
        (
            "SHA256(dataset|sample_id|"
            "permutation_seed), "
            "first 8 bytes little-endian"
        ),

    "primary_population":
        (
            "ambiguous_last_bucket samples "
            "for which at least two of the five "
            "tested fixed model-input sequence "
            "variants differ"
        ),

    "beauty_frozen_reproduction":
        beauty_reproduction,

    "datasets":
        dataset_results,
}

manifest_path = (
    REPRO_DIR
    / "step36_9_last_bucket_permutation_manifest.json"
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
    )
    + "\n",
    encoding="utf-8",
)


print()
print("=" * 88)
print("STEP36.9 SUMMARY")
print("=" * 88)

for r in dataset_results:

    print(
        f"{r['dataset']:16s} "
        f"test={r['test_samples']:,} "
        f"ambiguous="
        f"{r['ambiguous_last_bucket_samples']:,} "
        f"actually_permuted="
        f"{r['actually_permuted_samples']:,} "
        f"never_changed="
        f"{r['ambiguous_never_changed']:,}"
    )

print()
print(
    "SEED SUMMARY:",
    seed_summary_path
)

print(
    "PRIMARY SUMMARY:",
    primary_summary_path
)

print(
    "MANIFEST:",
    manifest_path
)

print(
    "STEP36.9 STATUS: PASS"
)
