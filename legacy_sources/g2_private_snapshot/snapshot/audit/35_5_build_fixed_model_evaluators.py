from pathlib import Path
import re


ROOT = Path(
    r"D:\time_uncertainty_rec"
)


JOBS = [
    {
        "dataset":
            "Amazon_Beauty",

        "source":
            ROOT
            / "audit"
            / "27_2_evaluate_diagnostic_b.py",

        "output":
            ROOT
            / "audit"
            / "35_5_evaluate_last_bucket_only_beauty.py",

        "source_model_protocol":
            "diagnostic_a",
    },

    {
        "dataset":
            "MovieLens-1M",

        "source":
            ROOT
            / "audit"
            / "29_2_evaluate_movielens_diagnostic_b.py",

        "output":
            ROOT
            / "audit"
            / "35_5_evaluate_last_bucket_only_movielens.py",

        "source_model_protocol":
            "diagnostic_a_strict",
    },
]


OLD_INPUT_ROOT = '''INPUT_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_b"
    / DATASET
)'''


NEW_INPUT_ROOT = '''INPUT_ROOT = (
    ROOT
    / "processed"
    / "strict_last_bucket_only"
    / DATASET
)'''


OLD_OUTPUT_ROOT = '''OUTPUT_ROOT = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "diagnostic_b"
    / "control_eval"
    / DATASET
    / f"seed{MODEL_SEED}"
)'''


NEW_OUTPUT_ROOT = '''OUTPUT_ROOT = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "step35_5_last_bucket_only"
    / "fixed_model_eval"
    / DATASET
    / f"seed{MODEL_SEED}"
)'''


OLD_INPUT_PATH = '''        input_path = (
            split_input_dir
            / (
                f"permutation_seed"
                f"{permutation_seed}.csv"
            )
        )'''


NEW_INPUT_PATH = '''        input_path = (
            INPUT_ROOT
            / f"perm_seed{permutation_seed}"
            / f"{split_name}.csv"
        )'''


def replace_once(
    text,
    old,
    new,
    label,
):

    count = text.count(
        old
    )

    if count != 1:

        raise RuntimeError(
            f"{label}: expected exactly "
            f"1 occurrence, got {count}"
        )

    return text.replace(
        old,
        new,
        1,
    )


def patch_model_dir(
    text,
    source_protocol,
):

    old = f'''MODEL_DIR = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "{source_protocol}"
    / "sasrec"
    / DATASET
    / "file_order"
    / f"seed{{MODEL_SEED}}"
)'''


    new = '''MODEL_DIR = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "diagnostic_a_strict"
    / "sasrec"
    / DATASET
    / "file_order"
    / f"seed{MODEL_SEED}"
)'''


    return replace_once(
        text,
        old,
        new,
        "MODEL_DIR",
    )


def patch_test_only_loop(
    text,
):

    # Supports:
    #
    # for split_name in [
    #     "val",
    #     "test",
    # ]:
    #
    # or tuple equivalent.

    pattern = re.compile(
        r'^for split_name in\s*'
        r'(?P<container>'
        r'\[[\s\S]*?\]'
        r'|'
        r'\([\s\S]*?\)'
        r')\s*:',
        flags=re.MULTILINE,
    )


    matches = list(
        pattern.finditer(
            text
        )
    )


    valid = []

    for m in matches:

        block = m.group(0)

        if (
            '"val"' in block
            and
            '"test"' in block
        ):

            valid.append(
                m
            )


    if len(valid) != 1:

        raise RuntimeError(
            "Could not uniquely locate "
            "val/test split loop. "
            f"Candidates={len(valid)}"
        )


    m = valid[0]


    return (
        text[
            :m.start()
        ]
        +
        'for split_name in ["test"]:'
        +
        text[
            m.end():
        ]
    )


def validate_generated(
    dataset,
    text,
):

    required_strings = [
        '"diagnostic_a_strict"',
        '"strict_last_bucket_only"',
        '"step35_5_last_bucket_only"',
        'f"perm_seed{permutation_seed}"',
        'f"{split_name}.csv"',
    ]


    for value in required_strings:

        if value not in text:

            raise RuntimeError(
                f"{dataset}: generated evaluator "
                f"missing {value}"
            )


    forbidden_strings = [
        '/ "diagnostic_b"\n'
        '    / DATASET',

        'f"permutation_seed"'
        '\n'
        '                f"{permutation_seed}.csv"',
    ]


    for value in forbidden_strings:

        if value in text:

            raise RuntimeError(
                f"{dataset}: stale input "
                f"protocol remains"
            )


print(
    "=" * 100
)

print(
    "35.5-B Build Fixed-Model "
    "Last-Bucket-Only Evaluators"
)

print(
    "=" * 100
)


for job in JOBS:

    dataset = job[
        "dataset"
    ]

    source = job[
        "source"
    ]

    output = job[
        "output"
    ]


    print()
    print(
        "=" * 100
    )
    print(
        dataset
    )
    print(
        "=" * 100
    )

    print(
        "SOURCE:",
        source
    )


    if not source.exists():

        raise FileNotFoundError(
            source
        )


    text = source.read_text(
        encoding="utf-8"
    )


    # --------------------------------------------
    # 1. Both datasets use strict-trained model
    # --------------------------------------------

    text = patch_model_dir(
        text,
        job[
            "source_model_protocol"
        ],
    )


    # --------------------------------------------
    # 2. Use 35.5-A inputs
    # --------------------------------------------

    text = replace_once(
        text,
        OLD_INPUT_ROOT,
        NEW_INPUT_ROOT,
        "INPUT_ROOT",
    )


    # --------------------------------------------
    # 3. Dedicated output; never overwrite old B
    # --------------------------------------------

    text = replace_once(
        text,
        OLD_OUTPUT_ROOT,
        NEW_OUTPUT_ROOT,
        "OUTPUT_ROOT",
    )


    # --------------------------------------------
    # 4. New per-seed file layout
    # --------------------------------------------

    text = replace_once(
        text,
        OLD_INPUT_PATH,
        NEW_INPUT_PATH,
        "input_path",
    )


    # --------------------------------------------
    # 6. Metadata wording only
    # --------------------------------------------

    text = text.replace(
        '"27.2"',
        '"35.5-B"',
    )


    text = text.replace(
        '"fixed_unpadded_inference"',
        (
            '"last_bucket_only_'
            'fixed_unpadded_inference"'
        ),
    )


    text = text.replace(
        "第27.2步：Diagnostic B Control 推理（修正版）",
        (
            "35.5-B：Last-Bucket-Only "
            "Fixed-Model Inference"
        ),
    )


    validate_generated(
        dataset,
        text,
    )


    output.write_text(
        text,
        encoding="utf-8",
    )


    print(
        "OUTPUT:",
        output
    )

    print(
        "MODEL PROTOCOL:",
        "diagnostic_a_strict"
    )

    print(
        "INPUT PROTOCOL:",
        "strict_last_bucket_only"
    )

    print(
        "SPLIT:",
        "test only"
    )

    print(
        "STATUS: PASS"
    )


print()
print(
    "=" * 100
)

print(
    "35.5-B evaluator build completed"
)

print(
    "=" * 100
)