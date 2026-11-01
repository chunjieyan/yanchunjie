from pathlib import Path
import hashlib
import json
import gc

import numpy as np
import pandas as pd


# ============================================================
# 第24步：固定天然数据预处理版本
#
# 输入：
#   Amazon Beauty 原始 ratings_Beauty.csv
#   MovieLens-1M 原始 ratings.dat
#
# 输出：
#   processed/natural_timestamp/...
#
# 原则：
#   1. 不修改原始数据
#   2. 删除 (user,item,timestamp) 完全重复
#   3. 迭代 5-core
#   4. 保留 raw_line
#   5. 生成 l / r / g
#   6. 保存 MD5 与处理摘要
# ============================================================


ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 1. 输入路径
# ============================================================

AMAZON_SOURCE = (
    ROOT
    / "data"
    / "Amazon_Beauty"
    / "ratings_Beauty.csv"
)

MOVIELENS_SOURCE = (
    ROOT
    / "data"
    / "MovieLens-1M"
    / "ml-1m"
    / "ratings.dat"
)


# ============================================================
# 2. 输出路径
# ============================================================

PROCESSED_ROOT = (
    ROOT
    / "processed"
    / "natural_timestamp"
)

AMAZON_OUTPUT_DIR = (
    PROCESSED_ROOT
    / "Amazon_Beauty"
)

MOVIELENS_OUTPUT_DIR = (
    PROCESSED_ROOT
    / "MovieLens-1M"
)

RUN_OUTPUT_DIR = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "preprocessing"
)

for path in [
    AMAZON_OUTPUT_DIR,
    MOVIELENS_OUTPUT_DIR,
    RUN_OUTPUT_DIR,
]:
    path.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 3. 第23步已经确认的数据规模
#
# 用于防止第24步预处理口径偷偷发生变化
# ============================================================

EXPECTED = {

    "Amazon Beauty": {
        "raw_interactions": 2023070,
        "fivecore_interactions": 198502,
        "fivecore_users": 22363,
        "fivecore_items": 12101,
    },

    "MovieLens-1M": {
        "raw_interactions": 1000209,
        "fivecore_interactions": 999611,
        "fivecore_users": 6040,
        "fivecore_items": 3416,
    },
}


# ============================================================
# 4. MD5
# ============================================================

def md5_file(path, chunk_size=1024 * 1024):

    md5 = hashlib.md5()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest().upper()


# ============================================================
# 5. 迭代 5-core
# ============================================================

def iterative_5core(df, name):

    iteration = 0

    while True:

        iteration += 1

        before = len(df)

        user_counts = (
            df["user_id"]
            .value_counts()
        )

        valid_users = (
            user_counts[
                user_counts >= 5
            ].index
        )

        df = df[
            df["user_id"].isin(
                valid_users
            )
        ]

        item_counts = (
            df["item_id"]
            .value_counts()
        )

        valid_items = (
            item_counts[
                item_counts >= 5
            ].index
        )

        df = df[
            df["item_id"].isin(
                valid_items
            )
        ].copy()

        after = len(df)

        print(
            f"{name} 5-core 第{iteration}轮："
            f"{before:,} -> {after:,}"
        )

        if before == after:
            break

    return df


# ============================================================
# 6. 完整性检查
# ============================================================

def validate_5core(df, dataset_name):

    min_user = (
        df["user_id"]
        .value_counts()
        .min()
    )

    min_item = (
        df["item_id"]
        .value_counts()
        .min()
    )

    duplicate_count = int(
        df.duplicated(
            subset=[
                "user_id",
                "item_id",
                "timestamp"
            ]
        ).sum()
    )

    print(
        f"{dataset_name} 最小用户交互数："
        f"{min_user}"
    )

    print(
        f"{dataset_name} 最小物品交互数："
        f"{min_item}"
    )

    print(
        f"{dataset_name} 剩余完全重复记录："
        f"{duplicate_count}"
    )

    if min_user < 5:
        raise RuntimeError(
            f"{dataset_name} 用户5-core检查失败"
        )

    if min_item < 5:
        raise RuntimeError(
            f"{dataset_name} 物品5-core检查失败"
        )

    if duplicate_count != 0:
        raise RuntimeError(
            f"{dataset_name} 仍存在完全重复记录"
        )


# ============================================================
# 7. 与第23步结果核对
# ============================================================

def validate_expected(
    df,
    dataset_name,
    raw_interactions
):

    expected = EXPECTED[
        dataset_name
    ]

    actual = {

        "raw_interactions":
            raw_interactions,

        "fivecore_interactions":
            len(df),

        "fivecore_users":
            df["user_id"].nunique(),

        "fivecore_items":
            df["item_id"].nunique(),
    }

    for key, expected_value in expected.items():

        actual_value = actual[key]

        if actual_value != expected_value:

            raise RuntimeError(
                f"{dataset_name} 与第23步结果不一致："
                f"{key} "
                f"expected={expected_value}, "
                f"actual={actual_value}"
            )

    print(
        f"{dataset_name} 与第23步统计完全一致。"
    )


# ============================================================
# 8. Amazon Beauty
# ============================================================

def process_amazon():

    print("\n" + "=" * 80)
    print("处理 Amazon Beauty")
    print("=" * 80)

    if not AMAZON_SOURCE.exists():
        raise FileNotFoundError(
            AMAZON_SOURCE
        )

    source_md5 = md5_file(
        AMAZON_SOURCE
    )

    print(
        "原始文件 MD5：",
        source_md5
    )

    df = pd.read_csv(
        AMAZON_SOURCE,
        header=None,
        names=[
            "user_id",
            "item_id",
            "rating",
            "timestamp"
        ],
        dtype={
            "user_id": str,
            "item_id": str,
            "rating": "float32",
            "timestamp": "int64"
        }
    )

    raw_interactions = len(df)

    # 原始文件行号
    df["raw_line"] = np.arange(
        1,
        len(df) + 1,
        dtype=np.int64
    )

    # 完全重复去除
    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "user_id",
            "item_id",
            "timestamp"
        ],
        keep="first"
    ).copy()

    duplicate_removed = (
        before - len(df)
    )

    print(
        "完全重复删除：",
        duplicate_removed
    )

    # 5-core
    df = iterative_5core(
        df,
        "Amazon Beauty"
    )

    # --------------------------------------------------------
    # 时间观测区间
    #
    # Amazon 2014 这里按“天”解释粗时间
    #
    # bucket = Unix day
    # l = 当天起点
    # r = 当天最后一秒
    # g = day
    # --------------------------------------------------------

    df["time_bucket"] = (
        df["timestamp"] // 86400
    ).astype("int64")

    df["l"] = (
        df["time_bucket"] * 86400
    ).astype("int64")

    df["r"] = (
        df["l"] + 86400 - 1
    ).astype("int64")

    df["g"] = "day"

    validate_5core(
        df,
        "Amazon Beauty"
    )

    validate_expected(
        df,
        "Amazon Beauty",
        raw_interactions
    )

    # 保持原始文件中的相对顺序
    df = df.sort_values(
        "raw_line",
        kind="mergesort"
    )

    output_path = (
        AMAZON_OUTPUT_DIR
        / "interactions_5core.csv"
    )

    output_columns = [
        "user_id",
        "item_id",
        "rating",
        "timestamp",
        "l",
        "r",
        "g",
        "time_bucket",
        "raw_line"
    ]

    df[
        output_columns
    ].to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    output_md5 = md5_file(
        output_path
    )

    summary = {

        "dataset":
            "Amazon Beauty",

        "source_file":
            str(AMAZON_SOURCE),

        "source_md5":
            source_md5,

        "raw_interactions":
            raw_interactions,

        "exact_duplicates_removed":
            duplicate_removed,

        "fivecore_interactions":
            len(df),

        "fivecore_users":
            df["user_id"].nunique(),

        "fivecore_items":
            df["item_id"].nunique(),

        "granularity":
            "day",

        "processed_file":
            str(output_path),

        "processed_md5":
            output_md5,
    }

    print(
        f"输出：{output_path}"
    )

    del df
    gc.collect()

    return summary


# ============================================================
# 9. MovieLens-1M
# ============================================================

def process_movielens():

    print("\n" + "=" * 80)
    print("处理 MovieLens-1M")
    print("=" * 80)

    if not MOVIELENS_SOURCE.exists():
        raise FileNotFoundError(
            MOVIELENS_SOURCE
        )

    source_md5 = md5_file(
        MOVIELENS_SOURCE
    )

    print(
        "原始文件 MD5：",
        source_md5
    )

    df = pd.read_csv(
        MOVIELENS_SOURCE,
        sep="::",
        engine="python",
        header=None,
        names=[
            "user_id",
            "item_id",
            "rating",
            "timestamp"
        ],
        dtype={
            "user_id": "int32",
            "item_id": "int32",
            "rating": "float32",
            "timestamp": "int64"
        }
    )

    raw_interactions = len(df)

    df["raw_line"] = np.arange(
        1,
        len(df) + 1,
        dtype=np.int64
    )

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "user_id",
            "item_id",
            "timestamp"
        ],
        keep="first"
    ).copy()

    duplicate_removed = (
        before - len(df)
    )

    print(
        "完全重复删除：",
        duplicate_removed
    )

    df = iterative_5core(
        df,
        "MovieLens-1M"
    )

    # --------------------------------------------------------
    # MovieLens 主协议：
    # 原始时间戳精度为秒
    #
    # 同秒记录无法由 timestamp 区分顺序
    # --------------------------------------------------------

    df["time_bucket"] = (
        df["timestamp"]
    ).astype("int64")

    df["l"] = (
        df["timestamp"]
    ).astype("int64")

    df["r"] = (
        df["timestamp"]
    ).astype("int64")

    df["g"] = "second"

    # 后续一分钟诊断直接由 timestamp // 60 派生
    df["minute_bucket"] = (
        df["timestamp"] // 60
    ).astype("int64")

    validate_5core(
        df,
        "MovieLens-1M"
    )

    validate_expected(
        df,
        "MovieLens-1M",
        raw_interactions
    )

    # 保留原始文件相对顺序
    df = df.sort_values(
        "raw_line",
        kind="mergesort"
    )

    output_path = (
        MOVIELENS_OUTPUT_DIR
        / "interactions_5core.csv"
    )

    output_columns = [
        "user_id",
        "item_id",
        "rating",
        "timestamp",
        "l",
        "r",
        "g",
        "time_bucket",
        "minute_bucket",
        "raw_line"
    ]

    df[
        output_columns
    ].to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    output_md5 = md5_file(
        output_path
    )

    summary = {

        "dataset":
            "MovieLens-1M",

        "source_file":
            str(MOVIELENS_SOURCE),

        "source_md5":
            source_md5,

        "raw_interactions":
            raw_interactions,

        "exact_duplicates_removed":
            duplicate_removed,

        "fivecore_interactions":
            len(df),

        "fivecore_users":
            df["user_id"].nunique(),

        "fivecore_items":
            df["item_id"].nunique(),

        "granularity":
            "second",

        "processed_file":
            str(output_path),

        "processed_md5":
            output_md5,
    }

    print(
        f"输出：{output_path}"
    )

    del df
    gc.collect()

    return summary


# ============================================================
# 10. 主程序
# ============================================================

def main():

    print("=" * 80)
    print("第24步：固定天然数据预处理版本")
    print("=" * 80)

    amazon_summary = (
        process_amazon()
    )

    ml_summary = (
        process_movielens()
    )

    summaries = [
        amazon_summary,
        ml_summary
    ]

    # --------------------------------------------------------
    # CSV 摘要
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        RUN_OUTPUT_DIR
        / "preprocessing_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # JSON manifest
    # --------------------------------------------------------

    manifest = {

        "step":
            24,

        "description":
            "Frozen natural timestamp preprocessing",

        "rules": {
            "duplicate_rule":
                "(user_id,item_id,timestamp)",

            "duplicate_keep":
                "first",

            "core":
                "iterative 5-core",

            "raw_line":
                "preserved; diagnostic only",

            "amazon_granularity":
                "day",

            "movielens_granularity":
                "second",
        },

        "datasets":
            summaries,
    }

    manifest_path = (
        RUN_OUTPUT_DIR
        / "preprocessing_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # 单独保存 processed MD5 表
    # --------------------------------------------------------

    md5_df = summary_df[
        [
            "dataset",
            "source_md5",
            "processed_md5"
        ]
    ].copy()

    md5_path = (
        RUN_OUTPUT_DIR
        / "processed_md5.csv"
    )

    md5_df.to_csv(
        md5_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 打印最终结果
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("第24步最终结果")
    print("=" * 80)

    show_cols = [
        "dataset",
        "raw_interactions",
        "fivecore_interactions",
        "fivecore_users",
        "fivecore_items",
        "granularity",
        "processed_md5",
    ]

    print(
        summary_df[
            show_cols
        ].to_string(
            index=False
        )
    )

    print("\n保存位置：")

    print(
        AMAZON_OUTPUT_DIR
        / "interactions_5core.csv"
    )

    print(
        MOVIELENS_OUTPUT_DIR
        / "interactions_5core.csv"
    )

    print(summary_path)
    print(manifest_path)
    print(md5_path)

    print(
        "\n第24步完成："
        "天然数据预处理版本已冻结。"
    )


if __name__ == "__main__":
    main()