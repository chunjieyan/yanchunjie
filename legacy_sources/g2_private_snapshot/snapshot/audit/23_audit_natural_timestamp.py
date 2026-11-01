from pathlib import Path
import gc

import numpy as np
import pandas as pd


# ============================================================
# 第23步：天然粗时间戳数据接入与审计
#
# 数据集：
# 1. Amazon Beauty
# 2. MovieLens-1M
#
# 输出：
# runs/phase2_natural_timestamp/audit/
# ============================================================


# ============================================================
# 1. 路径
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

AMAZON_PATH = (
    ROOT
    / "data"
    / "Amazon_Beauty"
    / "ratings_Beauty.csv"
)

MOVIELENS_PATH = (
    ROOT
    / "data"
    / "MovieLens-1M"
    / "ml-1m"
    / "ratings.dat"
)

OUTPUT_DIR = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 数据读取
# ============================================================

def load_amazon():
    print("\n" + "=" * 80)
    print("读取 Amazon Beauty")
    print("=" * 80)

    if not AMAZON_PATH.exists():
        raise FileNotFoundError(
            f"找不到文件：{AMAZON_PATH}"
        )

    df = pd.read_csv(
        AMAZON_PATH,
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

    # 保存原始文件行号
    # 只用于诊断，不作为模型输入
    df["raw_line"] = np.arange(
        1,
        len(df) + 1,
        dtype=np.int64
    )

    return df


def load_movielens():
    print("\n" + "=" * 80)
    print("读取 MovieLens-1M")
    print("=" * 80)

    if not MOVIELENS_PATH.exists():
        raise FileNotFoundError(
            f"找不到文件：{MOVIELENS_PATH}"
        )

    df = pd.read_csv(
        MOVIELENS_PATH,
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

    df["raw_line"] = np.arange(
        1,
        len(df) + 1,
        dtype=np.int64
    )

    return df


# ============================================================
# 3. 数据基本信息
# ============================================================

def print_basic_info(df, name):
    print(f"\n{name} 原始数据：")
    print(f"交互数：{len(df):,}")
    print(
        f"用户数："
        f"{df['user_id'].nunique():,}"
    )
    print(
        f"物品数："
        f"{df['item_id'].nunique():,}"
    )


# ============================================================
# 4. 去除完全重复
#
# 只删除：
# (user_id, item_id, timestamp)
# 三者完全一致的记录
#
# 不删除不同时间发生的重复交互
# ============================================================

def remove_exact_duplicates(df, name):

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "user_id",
            "item_id",
            "timestamp"
        ],
        keep="first"
    ).copy()

    removed = before - len(df)

    print(
        f"\n{name} 完全重复记录："
        f"{removed:,}"
    )

    return df, removed


# ============================================================
# 5. 迭代 5-core
#
# 用户交互数 >= 5
# 物品交互数 >= 5
#
# 必须迭代，直到稳定
# ============================================================

def iterative_5core(df, name):

    iteration = 0

    while True:

        iteration += 1

        before = len(df)

        user_count = (
            df["user_id"]
            .value_counts()
        )

        valid_users = (
            user_count[
                user_count >= 5
            ].index
        )

        df = df[
            df["user_id"].isin(
                valid_users
            )
        ]

        item_count = (
            df["item_id"]
            .value_counts()
        )

        valid_items = (
            item_count[
                item_count >= 5
            ].index
        )

        df = df[
            df["item_id"].isin(
                valid_items
            )
        ].copy()

        after = len(df)

        print(
            f"{name} 5-core 第"
            f"{iteration}轮："
            f"{before:,} -> {after:,}"
        )

        if after == before:
            break

    print(
        f"{name} 5-core完成："
        f"交互={len(df):,}，"
        f"用户={df['user_id'].nunique():,}，"
        f"物品={df['item_id'].nunique():,}"
    )

    return df


# ============================================================
# 6. 统计不同时间重复交互
# ============================================================

def count_repeat_user_item(df):

    pair_count = (
        df.groupby(
            ["user_id", "item_id"],
            sort=False
        )
        .size()
    )

    repeated_pairs = int(
        (pair_count >= 2).sum()
    )

    return repeated_pairs


# ============================================================
# 7. 时间桶审计
#
# 重点统计：
# 1. 同桶多个物品记录占比
# 2. 含歧义桶用户占比
# 3. 桶大小 2 / 3 / >=4
# 4. 最后历史桶 >=2 个物品比例
# 5. target 与最后历史项同桶比例
# ============================================================

def audit_timestamp(
    df,
    dataset_name,
    bucket_rule,
    bucket_values
):

    print(
        f"\n开始审计："
        f"{dataset_name} / {bucket_rule}"
    )

    work = df[
        [
            "user_id",
            "item_id",
            "timestamp",
            "raw_line"
        ]
    ].copy()

    work["bucket"] = np.asarray(
        bucket_values
    )

    # --------------------------------------------------------
    # A. 每个 user + bucket 的统计
    # --------------------------------------------------------

    bucket_stats = (
        work
        .groupby(
            ["user_id", "bucket"],
            sort=False
        )
        .agg(
            bucket_records=(
                "item_id",
                "size"
            ),
            bucket_items=(
                "item_id",
                "nunique"
            )
        )
    )

    # 真正的“多个物品”
    ambiguous = bucket_stats[
        bucket_stats["bucket_items"] >= 2
    ]

    total_buckets = len(
        bucket_stats
    )

    ambiguous_buckets = len(
        ambiguous
    )

    ambiguous_records = int(
        ambiguous[
            "bucket_records"
        ].sum()
    )

    ambiguous_record_ratio = (
        ambiguous_records
        / len(work)
    )

    ambiguous_users = (
        ambiguous
        .reset_index()[
            "user_id"
        ]
        .nunique()
    )

    total_users = (
        work["user_id"]
        .nunique()
    )

    ambiguous_user_ratio = (
        ambiguous_users
        / total_users
    )

    # --------------------------------------------------------
    # B. 桶大小分布
    # 按不同物品数统计
    # --------------------------------------------------------

    bucket_size_2 = int(
        (
            bucket_stats[
                "bucket_items"
            ] == 2
        ).sum()
    )

    bucket_size_3 = int(
        (
            bucket_stats[
                "bucket_items"
            ] == 3
        ).sum()
    )

    bucket_size_ge4 = int(
        (
            bucket_stats[
                "bucket_items"
            ] >= 4
        ).sum()
    )

    # --------------------------------------------------------
    # C. Leave-One-Out 检查
    #
    # 时间相同时：
    # 使用 raw_line 作为诊断 tie-breaking
    #
    # raw_line 不进入模型
    # --------------------------------------------------------

    ordered = work.sort_values(
        by=[
            "user_id",
            "timestamp",
            "raw_line"
        ],
        kind="mergesort"
    )

    # 每个用户最后一条作为 target
    target_idx = (
        ordered
        .groupby(
            "user_id",
            sort=False
        )
        .tail(1)
        .index
    )

    targets = ordered.loc[
        target_idx,
        [
            "user_id",
            "bucket"
        ]
    ].copy()

    targets = targets.rename(
        columns={
            "bucket":
                "target_bucket"
        }
    )

    # history = 除 target 外全部历史
    history = ordered.drop(
        index=target_idx
    )

    # 每个用户最后一条历史记录
    last_history = (
        history
        .groupby(
            "user_id",
            sort=False
        )
        .tail(1)[
            [
                "user_id",
                "bucket"
            ]
        ]
        .copy()
    )

    last_history = (
        last_history.rename(
            columns={
                "bucket":
                    "last_history_bucket"
            }
        )
    )

    # history 中每个时间桶包含多少个不同物品
    history_bucket_size = (
        history
        .groupby(
            ["user_id", "bucket"],
            sort=False
        )["item_id"]
        .nunique()
        .rename(
            "history_bucket_items"
        )
        .reset_index()
    )

    last_history = (
        last_history.merge(
            history_bucket_size,
            left_on=[
                "user_id",
                "last_history_bucket"
            ],
            right_on=[
                "user_id",
                "bucket"
            ],
            how="left"
        )
    )

    loo = targets.merge(
        last_history[
            [
                "user_id",
                "last_history_bucket",
                "history_bucket_items"
            ]
        ],
        on="user_id",
        how="inner"
    )

    # 最后历史桶 >=2 个物品
    last_history_ambiguous = (
        loo[
            "history_bucket_items"
        ] >= 2
    )

    last_history_ambiguous_ratio = (
        float(
            last_history_ambiguous.mean()
        )
    )

    # target 与最后历史项同桶
    target_same_bucket = (
        loo["target_bucket"]
        ==
        loo["last_history_bucket"]
    )

    target_same_bucket_ratio = (
        float(
            target_same_bucket.mean()
        )
    )

    # --------------------------------------------------------
    # 输出结果
    # --------------------------------------------------------

    result = {
        "dataset":
            dataset_name,

        "bucket_rule":
            bucket_rule,

        "interactions":
            len(work),

        "users":
            total_users,

        "items":
            work["item_id"].nunique(),

        "total_buckets":
            total_buckets,

        "ambiguous_buckets":
            ambiguous_buckets,

        "ambiguous_records":
            ambiguous_records,

        "ambiguous_record_ratio":
            ambiguous_record_ratio,

        "ambiguous_users":
            ambiguous_users,

        "ambiguous_user_ratio":
            ambiguous_user_ratio,

        "bucket_size_2":
            bucket_size_2,

        "bucket_size_3":
            bucket_size_3,

        "bucket_size_ge4":
            bucket_size_ge4,

        "loo_test_users":
            len(loo),

        "last_history_bucket_ge2_ratio":
            last_history_ambiguous_ratio,

        "target_same_last_history_bucket_ratio":
            target_same_bucket_ratio
    }

    return result


# ============================================================
# 8. 单数据集处理
# ============================================================

def process_amazon():

    raw = load_amazon()

    print_basic_info(
        raw,
        "Amazon Beauty"
    )

    raw_interactions = len(raw)
    raw_users = raw[
        "user_id"
    ].nunique()
    raw_items = raw[
        "item_id"
    ].nunique()

    df, duplicates = (
        remove_exact_duplicates(
            raw,
            "Amazon Beauty"
        )
    )

    df = iterative_5core(
        df,
        "Amazon Beauty"
    )

    repeated_pairs = (
        count_repeat_user_item(df)
    )

    overview = {
        "dataset":
            "Amazon Beauty",

        "raw_interactions":
            raw_interactions,

        "raw_users":
            raw_users,

        "raw_items":
            raw_items,

        "exact_duplicates_removed":
            duplicates,

        "fivecore_interactions":
            len(df),

        "fivecore_users":
            df["user_id"].nunique(),

        "fivecore_items":
            df["item_id"].nunique(),

        "repeated_user_item_pairs":
            repeated_pairs
    }

    # Amazon 2014 时间戳按天审计
    day_bucket = (
        df["timestamp"]
        // 86400
    )

    audit = audit_timestamp(
        df,
        "Amazon Beauty",
        "day",
        day_bucket
    )

    del raw
    del df

    gc.collect()

    return overview, audit


def process_movielens():

    raw = load_movielens()

    print_basic_info(
        raw,
        "MovieLens-1M"
    )

    raw_interactions = len(raw)
    raw_users = raw[
        "user_id"
    ].nunique()
    raw_items = raw[
        "item_id"
    ].nunique()

    df, duplicates = (
        remove_exact_duplicates(
            raw,
            "MovieLens-1M"
        )
    )

    df = iterative_5core(
        df,
        "MovieLens-1M"
    )

    repeated_pairs = (
        count_repeat_user_item(df)
    )

    overview = {
        "dataset":
            "MovieLens-1M",

        "raw_interactions":
            raw_interactions,

        "raw_users":
            raw_users,

        "raw_items":
            raw_items,

        "exact_duplicates_removed":
            duplicates,

        "fivecore_interactions":
            len(df),

        "fivecore_users":
            df["user_id"].nunique(),

        "fivecore_items":
            df["item_id"].nunique(),

        "repeated_user_item_pairs":
            repeated_pairs
    }

    # 原始秒级时间戳
    second_audit = audit_timestamp(
        df,
        "MovieLens-1M",
        "second",
        df["timestamp"]
    )

    # 额外统计分钟粒度
    minute_audit = audit_timestamp(
        df,
        "MovieLens-1M",
        "minute",
        df["timestamp"] // 60
    )

    del raw
    del df

    gc.collect()

    return (
        overview,
        second_audit,
        minute_audit
    )


# ============================================================
# 9. 主程序
# ============================================================

def main():

    print("=" * 80)
    print("第23步：天然粗时间戳数据审计")
    print("=" * 80)

    print("\n项目目录：")
    print(ROOT)

    print("\nAmazon路径：")
    print(AMAZON_PATH)

    print("\nMovieLens路径：")
    print(MOVIELENS_PATH)

    # Amazon
    amazon_overview, amazon_audit = (
        process_amazon()
    )

    # MovieLens
    (
        ml_overview,
        ml_second,
        ml_minute
    ) = process_movielens()

    # --------------------------------------------------------
    # 数据集概况
    # --------------------------------------------------------

    overview_df = pd.DataFrame(
        [
            amazon_overview,
            ml_overview
        ]
    )

    overview_file = (
        OUTPUT_DIR
        / "dataset_overview.csv"
    )

    overview_df.to_csv(
        overview_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 时间戳审计表
    # --------------------------------------------------------

    audit_df = pd.DataFrame(
        [
            amazon_audit,
            ml_second,
            ml_minute
        ]
    )

    audit_file = (
        OUTPUT_DIR
        / "timestamp_audit.csv"
    )

    audit_df.to_csv(
        audit_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 终端打印百分比版
    # --------------------------------------------------------

    show = audit_df[
        [
            "dataset",
            "bucket_rule",
            "ambiguous_record_ratio",
            "ambiguous_user_ratio",
            "last_history_bucket_ge2_ratio",
            "target_same_last_history_bucket_ratio",
            "bucket_size_2",
            "bucket_size_3",
            "bucket_size_ge4"
        ]
    ].copy()

    ratio_cols = [
        "ambiguous_record_ratio",
        "ambiguous_user_ratio",
        "last_history_bucket_ge2_ratio",
        "target_same_last_history_bucket_ratio"
    ]

    for col in ratio_cols:
        show[col] = (
            show[col]
            * 100
        ).round(4)

    print("\n" + "=" * 80)
    print("数据集概况")
    print("=" * 80)

    print(
        overview_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("天然时间戳审计结果（比例列单位：%）")
    print("=" * 80)

    print(
        show.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("结果保存位置")
    print("=" * 80)

    print(overview_file)
    print(audit_file)

    print("\n第23步完成。")


if __name__ == "__main__":
    main()