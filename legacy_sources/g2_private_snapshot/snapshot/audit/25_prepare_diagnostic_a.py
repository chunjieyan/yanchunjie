from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


# ============================================================
# 第25步 / 诊断A：
# 文件顺序是否携带真实顺序信息
#
# 本步只生成两套配对数据：
#
# 1. file_order
#    同一时间桶内部按 raw_line 排列
#
# 2. bucket_shuffled
#    同一时间桶内部固定随机打乱
#
# 暂时不训练模型
# ============================================================


ROOT = Path(__file__).resolve().parent.parent

INPUT_ROOT = (
    ROOT
    / "processed"
    / "natural_timestamp"
)

OUTPUT_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_a"
)

RUN_DIR = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "diagnostic_a"
)

SHUFFLE_SEED = 101


DATASETS = {

    "Amazon_Beauty": {
        "input":
            INPUT_ROOT
            / "Amazon_Beauty"
            / "interactions_5core.csv",

        "bucket_column":
            "time_bucket",
    },

    "MovieLens-1M": {
        "input":
            INPUT_ROOT
            / "MovieLens-1M"
            / "interactions_5core.csv",

        "bucket_column":
            "time_bucket",
    },
}


# ============================================================
# MD5
# ============================================================

def md5_file(path):

    md5 = hashlib.md5()

    with open(path, "rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            md5.update(block)

    return md5.hexdigest().upper()


# ============================================================
# file-order版本
# ============================================================

def build_file_order(
    df,
    bucket_col
):

    # 首先按用户和时间排序。
    #
    # 时间相同的时候，
    # 使用原始文件行号 raw_line
    # 作为诊断顺序。
    #
    # mergesort保证稳定排序。
    ordered = df.sort_values(
        [
            "user_id",
            bucket_col,
            "raw_line",
        ],
        kind="mergesort"
    ).copy()

    return ordered


# ============================================================
# bucket-shuffled版本
# ============================================================

def build_bucket_shuffled(
    df,
    bucket_col,
    seed
):

    work = df.copy()

    rng = np.random.default_rng(
        seed
    )

    # 给每一条记录生成随机键
    #
    # 不改变：
    # user
    # timestamp bucket
    # item集合
    #
    # 只改变桶内排列。
    work["_shuffle_key"] = (
        rng.random(
            len(work)
        )
    )

    shuffled = work.sort_values(
        [
            "user_id",
            bucket_col,
            "_shuffle_key",
            "raw_line",
        ],
        kind="mergesort"
    ).copy()

    shuffled = shuffled.drop(
        columns=[
            "_shuffle_key"
        ]
    )

    return shuffled


# ============================================================
# 严格检查
# ============================================================

def validate_pair(
    original,
    file_order,
    shuffled,
    bucket_col,
    dataset_name,
):

    print(
        f"\n检查 {dataset_name}"
    )

    # 1. 行数必须完全一致
    assert (
        len(original)
        ==
        len(file_order)
        ==
        len(shuffled)
    )

    # 2. 所有 raw_line 必须完全一致
    original_lines = set(
        original["raw_line"]
    )

    assert set(
        file_order["raw_line"]
    ) == original_lines

    assert set(
        shuffled["raw_line"]
    ) == original_lines

    # 3. user-item-timestamp集合完全相同
    key_cols = [
        "user_id",
        "item_id",
        "timestamp",
        "raw_line",
    ]

    a = (
        original[
            key_cols
        ]
        .sort_values(
            "raw_line"
        )
        .reset_index(
            drop=True
        )
    )

    b = (
        file_order[
            key_cols
        ]
        .sort_values(
            "raw_line"
        )
        .reset_index(
            drop=True
        )
    )

    c = (
        shuffled[
            key_cols
        ]
        .sort_values(
            "raw_line"
        )
        .reset_index(
            drop=True
        )
    )

    assert a.equals(b)
    assert a.equals(c)

    # 4. 检查真正发生顺序变化的歧义桶
    file_groups = (
        file_order
        .groupby(
            [
                "user_id",
                bucket_col
            ],
            sort=False
        )["item_id"]
        .apply(list)
    )

    shuffle_groups = (
        shuffled
        .groupby(
            [
                "user_id",
                bucket_col
            ],
            sort=False
        )["item_id"]
        .apply(list)
    )

    bucket_size = (
        original
        .groupby(
            [
                "user_id",
                bucket_col
            ]
        )["item_id"]
        .nunique()
    )

    ambiguous_keys = (
        bucket_size[
            bucket_size >= 2
        ].index
    )

    changed = 0

    for key in ambiguous_keys:

        if (
            file_groups.loc[key]
            !=
            shuffle_groups.loc[key]
        ):
            changed += 1

    total_ambiguous = len(
        ambiguous_keys
    )

    changed_ratio = (
        changed
        / total_ambiguous
        if total_ambiguous > 0
        else 0.0
    )

    print(
        f"总交互：{len(original):,}"
    )

    print(
        f"歧义桶："
        f"{total_ambiguous:,}"
    )

    print(
        f"实际发生排列变化的歧义桶："
        f"{changed:,}"
    )

    print(
        "变化比例："
        f"{changed_ratio * 100:.4f}%"
    )

    return {
        "dataset":
            dataset_name,

        "interactions":
            len(original),

        "ambiguous_buckets":
            total_ambiguous,

        "changed_ambiguous_buckets":
            changed,

        "changed_ratio":
            changed_ratio,

        "shuffle_seed":
            SHUFFLE_SEED,
    }


# ============================================================
# 单数据集处理
# ============================================================

def process_dataset(
    dataset_name,
    config
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"处理 {dataset_name}"
    )

    print(
        "=" * 80
    )

    input_path = config[
        "input"
    ]

    bucket_col = config[
        "bucket_column"
    ]

    if not input_path.exists():

        raise FileNotFoundError(
            input_path
        )

    df = pd.read_csv(
        input_path
    )

    print(
        f"输入：{input_path}"
    )

    print(
        f"交互数：{len(df):,}"
    )

    # --------------------------------------------------------
    # A. 原始文件顺序
    # --------------------------------------------------------

    file_order = (
        build_file_order(
            df,
            bucket_col
        )
    )

    # --------------------------------------------------------
    # B. 桶内随机顺序
    # --------------------------------------------------------

    shuffled = (
        build_bucket_shuffled(
            df,
            bucket_col,
            SHUFFLE_SEED
        )
    )

    # --------------------------------------------------------
    # 检查
    # --------------------------------------------------------

    stats = validate_pair(
        df,
        file_order,
        shuffled,
        bucket_col,
        dataset_name,
    )

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    dataset_dir = (
        OUTPUT_ROOT
        / dataset_name
    )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_order_path = (
        dataset_dir
        / "file_order.csv"
    )

    shuffled_path = (
        dataset_dir
        / f"bucket_shuffled_seed{SHUFFLE_SEED}.csv"
    )

    file_order.to_csv(
        file_order_path,
        index=False,
        encoding="utf-8-sig"
    )

    shuffled.to_csv(
        shuffled_path,
        index=False,
        encoding="utf-8-sig"
    )

    stats[
        "file_order_md5"
    ] = md5_file(
        file_order_path
    )

    stats[
        "shuffled_md5"
    ] = md5_file(
        shuffled_path
    )

    print(
        f"\nfile-order："
        f"{file_order_path}"
    )

    print(
        f"MD5："
        f"{stats['file_order_md5']}"
    )

    print(
        f"\nshuffled："
        f"{shuffled_path}"
    )

    print(
        f"MD5："
        f"{stats['shuffled_md5']}"
    )

    return stats


# ============================================================
# 主程序
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "第25步：诊断A数据准备"
    )

    print(
        "文件顺序 vs 桶内随机排列"
    )

    print(
        "=" * 80
    )

    results = []

    for (
        dataset_name,
        config
    ) in DATASETS.items():

        results.append(
            process_dataset(
                dataset_name,
                config
            )
        )

    result_df = (
        pd.DataFrame(
            results
        )
    )

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path = (
        RUN_DIR
        / "diagnostic_a_data_summary.csv"
    )

    result_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "第25步数据准备结果"
    )

    print(
        "=" * 80
    )

    show = result_df[
        [
            "dataset",
            "interactions",
            "ambiguous_buckets",
            "changed_ambiguous_buckets",
            "changed_ratio",
            "shuffle_seed",
        ]
    ].copy()

    show[
        "changed_ratio"
    ] = (
        show[
            "changed_ratio"
        ]
        * 100
    ).round(4)

    print(
        show.to_string(
            index=False
        )
    )

    print(
        "\n结果："
    )

    print(
        summary_path
    )

    print(
        "\n第25.1步完成。"
    )


if __name__ == "__main__":
    main()