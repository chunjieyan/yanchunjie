from pathlib import Path
import json
import pandas as pd


# ============================================================
# Step 31.1
# Prepare Recent Bucket-Set R=1 protocol
#
# 目的：
# 1. 不重新划分数据
# 2. 不改变 history 成员
# 3. 不改变 target
# 4. 只把最后一个历史时间桶显式拆出来
#
# 后续 V1：
# prefix -> 保持普通顺序
# last bucket -> mean pooling
# ============================================================


ROOT = Path(r"D:\time_uncertainty_rec")

DATASETS = [
    "Amazon_Beauty",
    "MovieLens-1M",
]

SPLITS = [
    "val",
    "test",
]


INPUT_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_b"
)


OUTPUT_ROOT = (
    ROOT
    / "processed"
    / "recent_bucketset_r1"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 工具函数
# ============================================================

def parse_json_list(value):

    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    value = str(value).strip()

    if value == "":
        return []

    x = json.loads(value)

    if not isinstance(x, list):
        raise ValueError(
            f"Expected list but got: {type(x)}"
        )

    return x


# ============================================================
# 单个 split 处理
# ============================================================

def process_split(
    dataset,
    split
):

    input_path = (
        INPUT_ROOT
        / dataset
        / split
        / "canonical.csv"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Missing input:\n{input_path}"
        )


    print()
    print("=" * 80)
    print(
        f"Dataset: {dataset}"
    )
    print(
        f"Split: {split}"
    )
    print("=" * 80)


    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig"
    )


    required_columns = [
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


    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]


    if missing:

        raise RuntimeError(
            "Missing columns: "
            + ", ".join(
                missing
            )
        )


    output_rows = []


    for row in df.itertuples(
        index=False
    ):

        sequence_items = (
            parse_json_list(
                row.sequence_items
            )
        )

        sequence_raw_lines = (
            parse_json_list(
                row.sequence_raw_lines
            )
        )


        # ----------------------------------------------------
        # 基础一致性检查
        # ----------------------------------------------------

        if (
            len(sequence_items)
            !=
            len(sequence_raw_lines)
        ):

            raise RuntimeError(
                f"sample_id={row.sample_id}: "
                f"items/raw_lines length mismatch"
            )


        if (
            len(sequence_items)
            !=
            int(row.fixed_history_len)
        ):

            raise RuntimeError(
                f"sample_id={row.sample_id}: "
                f"sequence length="
                f"{len(sequence_items)}, "
                f"fixed_history_len="
                f"{row.fixed_history_len}"
            )


        if len(sequence_items) == 0:

            raise RuntimeError(
                f"sample_id={row.sample_id}: "
                "empty history"
            )


        # ----------------------------------------------------
        # 最后历史桶大小
        #
        # 注意：
        # sequence 已经是固定后的最后 MAX_LEN 个历史行为。
        # 如果原始最后桶非常大，大于 fixed_history_len，
        # 最多只能在当前模型可见历史中取 fixed_history_len。
        # ----------------------------------------------------

        last_bucket_size = min(
            int(
                row.last_history_bucket_size
            ),
            len(sequence_items)
        )


        if last_bucket_size < 1:

            raise RuntimeError(
                f"sample_id={row.sample_id}: "
                f"invalid last bucket size "
                f"{last_bucket_size}"
            )


        # ----------------------------------------------------
        # R = 1 拆分
        # ----------------------------------------------------

        prefix_items = (
            sequence_items[
                :-last_bucket_size
            ]
        )

        last_bucket_items = (
            sequence_items[
                -last_bucket_size:
            ]
        )


        prefix_raw_lines = (
            sequence_raw_lines[
                :-last_bucket_size
            ]
        )

        last_bucket_raw_lines = (
            sequence_raw_lines[
                -last_bucket_size:
            ]
        )


        # ----------------------------------------------------
        # direct_pooling
        #
        # 最后桶只有1个item时，mean pooling退化为原表示，
        # 不会实际产生集合化效果。
        # ----------------------------------------------------

        direct_pooling = int(
            last_bucket_size >= 2
        )


        # ----------------------------------------------------
        # 与前面歧义定义核对
        # ----------------------------------------------------

        expected_ambiguity = int(
            bool(
                row.ambiguous_last_bucket
            )
        )


        if (
            direct_pooling
            !=
            expected_ambiguity
        ):

            raise RuntimeError(
                f"sample_id={row.sample_id}: "
                f"direct_pooling="
                f"{direct_pooling}, "
                f"ambiguous_last_bucket="
                f"{expected_ambiguity}"
            )


        # ----------------------------------------------------
        # history 成员必须完全不变
        # ----------------------------------------------------

        reconstructed_items = (
            prefix_items
            +
            last_bucket_items
        )


        if (
            reconstructed_items
            !=
            sequence_items
        ):

            raise RuntimeError(
                f"sample_id={row.sample_id}: "
                "history reconstruction failed"
            )


        output_rows.append(
            {
                "sample_id":
                    row.sample_id,

                "dataset":
                    dataset,

                "split":
                    split,

                "user_id":
                    row.user_id,

                "target_item":
                    row.target_item,

                "target_raw_line":
                    row.target_raw_line,

                "target_timestamp":
                    row.target_timestamp,

                "target_bucket":
                    row.target_bucket,

                "fixed_history_len":
                    len(
                        sequence_items
                    ),

                "prefix_len":
                    len(
                        prefix_items
                    ),

                "last_bucket_size":
                    last_bucket_size,

                "ambiguous_last_bucket":
                    expected_ambiguity,

                "direct_pooling":
                    direct_pooling,

                "sequence_items":
                    json.dumps(
                        sequence_items,
                        ensure_ascii=False
                    ),

                "prefix_items":
                    json.dumps(
                        prefix_items,
                        ensure_ascii=False
                    ),

                "last_bucket_items":
                    json.dumps(
                        last_bucket_items,
                        ensure_ascii=False
                    ),

                "sequence_raw_lines":
                    json.dumps(
                        sequence_raw_lines,
                        ensure_ascii=False
                    ),

                "prefix_raw_lines":
                    json.dumps(
                        prefix_raw_lines,
                        ensure_ascii=False
                    ),

                "last_bucket_raw_lines":
                    json.dumps(
                        last_bucket_raw_lines,
                        ensure_ascii=False
                    ),
            }
        )


    out = pd.DataFrame(
        output_rows
    )


    # ========================================================
    # 统计
    # ========================================================

    samples = len(
        out
    )

    direct_samples = int(
        out[
            "direct_pooling"
        ].sum()
    )

    direct_ratio = (
        direct_samples
        /
        samples
        if samples > 0
        else 0.0
    )


    mean_last_bucket_size = (
        out[
            "last_bucket_size"
        ].mean()
    )


    max_last_bucket_size = int(
        out[
            "last_bucket_size"
        ].max()
    )


    pooled = out[
        out[
            "direct_pooling"
        ]
        ==
        1
    ]


    if len(pooled) > 0:

        pooled_mean_bucket_size = (
            pooled[
                "last_bucket_size"
            ].mean()
        )

    else:

        pooled_mean_bucket_size = 0.0


    # ========================================================
    # 保存
    # ========================================================

    output_dir = (
        OUTPUT_ROOT
        / dataset
        / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_csv = (
        output_dir
        / "r1_samples.csv"
    )


    out.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )


    summary = pd.DataFrame(
        [
            {
                "dataset":
                    dataset,

                "split":
                    split,

                "samples":
                    samples,

                "direct_pooling_samples":
                    direct_samples,

                "direct_pooling_ratio":
                    direct_ratio,

                "mean_last_bucket_size":
                    mean_last_bucket_size,

                "pooled_mean_last_bucket_size":
                    pooled_mean_bucket_size,

                "max_last_bucket_size":
                    max_last_bucket_size,
            }
        ]
    )


    summary_csv = (
        output_dir
        / "summary.csv"
    )


    summary.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "Samples:",
        samples
    )

    print(
        "Direct pooling samples:",
        direct_samples
    )

    print(
        "Direct pooling ratio:",
        f"{direct_ratio:.4%}"
    )

    print(
        "Mean last bucket size:",
        f"{mean_last_bucket_size:.4f}"
    )

    print(
        "Pooled mean last bucket size:",
        f"{pooled_mean_bucket_size:.4f}"
    )

    print(
        "Max last bucket size:",
        max_last_bucket_size
    )

    print(
        "Output:",
        output_csv
    )


    return {
        "dataset":
            dataset,

        "split":
            split,

        "samples":
            samples,

        "direct_pooling_samples":
            direct_samples,

        "direct_pooling_ratio":
            direct_ratio,

        "mean_last_bucket_size":
            mean_last_bucket_size,

        "pooled_mean_last_bucket_size":
            pooled_mean_bucket_size,

        "max_last_bucket_size":
            max_last_bucket_size,
    }


# ============================================================
# Main
# ============================================================

all_summaries = []


for dataset in DATASETS:

    for split in SPLITS:

        result = process_split(
            dataset,
            split
        )

        all_summaries.append(
            result
        )


summary_df = pd.DataFrame(
    all_summaries
)


overall_summary_path = (
    OUTPUT_ROOT
    / "r1_protocol_summary.csv"
)


summary_df.to_csv(
    overall_summary_path,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 80)
print("Step 31.1 completed")
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print(
    "Summary:",
    overall_summary_path
)