from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# 第26步：冻结严格评价协议
#
# 核心要求：
#
# target_bucket > last_history_bucket
#
# 即：
# 测试目标必须严格晚于最后一个历史时间桶。
#
# 不改变 target，不通过删掉同桶历史来“修复”样本。
# 如果标准 target 与最后历史桶同桶，直接排除该样本。
#
# 目的：
# 1. 将“目标不可辨识”与“历史顺序歧义”分开
# 2. 后续 file_order / shuffled / pooling
#    全部评价完全相同的 target
# ============================================================


ROOT = Path(__file__).resolve().parent.parent

INPUT_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_a"
)

SPLIT_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_a_splits"
)

OUTPUT_ROOT = (
    ROOT
    / "processed"
    / "strict_protocol"
)

RUN_DIR = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "protocol"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MIN_HISTORY = 5
SHUFFLE_SEED = 101


DATASETS = [
    "Amazon_Beauty",
    "MovieLens-1M",
]


# ============================================================
# 读取 split 中的 raw_line
# ============================================================

def load_split_raw_lines(
    dataset_name,
    split_name
):

    path = (
        SPLIT_ROOT
        / dataset_name
        / "file_order"
        / f"{split_name}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path,
        usecols=["raw_line"]
    )

    return set(
        df["raw_line"]
        .astype(np.int64)
        .tolist()
    )


# ============================================================
# 给完整数据标记 train / val / test
# ============================================================

def attach_split_labels(
    df,
    dataset_name
):

    train_lines = load_split_raw_lines(
        dataset_name,
        "train"
    )

    val_lines = load_split_raw_lines(
        dataset_name,
        "val"
    )

    test_lines = load_split_raw_lines(
        dataset_name,
        "test"
    )

    raw_line = (
        df["raw_line"]
        .astype(np.int64)
    )

    df = df.copy()

    df["_split"] = np.select(
        [
            raw_line.isin(
                train_lines
            ),

            raw_line.isin(
                val_lines
            ),

            raw_line.isin(
                test_lines
            ),
        ],
        [
            "train",
            "val",
            "test",
        ],
        default="unknown"
    )

    unknown = int(
        (
            df["_split"]
            ==
            "unknown"
        ).sum()
    )

    if unknown != 0:

        raise RuntimeError(
            f"{dataset_name} "
            f"发现 {unknown} 条记录没有split"
        )

    return df


# ============================================================
# 验证 shuffled 与 file_order 包含相同记录
# ============================================================

def validate_paired_data(
    file_order,
    shuffled,
    dataset_name
):

    a = np.sort(
        file_order[
            "raw_line"
        ].to_numpy(
            dtype=np.int64
        )
    )

    b = np.sort(
        shuffled[
            "raw_line"
        ].to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        a,
        b
    ):

        raise RuntimeError(
            f"{dataset_name} "
            "file_order 与 shuffled "
            "不是相同记录集合"
        )

    print(
        "file_order / shuffled "
        "记录集合检查：通过"
    )


# ============================================================
# 为某一个 split 构建严格样本
# ============================================================

def build_strict_samples(
    df,
    split_name,
    dataset_name
):

    samples = []

    candidate_users = 0

    excluded_same_bucket = 0
    excluded_short_history = 0
    excluded_no_history = 0

    # --------------------------------------------------------
    # canonical order：
    # timestamp/time_bucket 相同的时候使用 raw_line
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "user_id",
            "time_bucket",
            "raw_line",
        ],
        kind="mergesort"
    )

    for (
        user_id,
        group
    ) in df.groupby(
        "user_id",
        sort=False
    ):

        group = (
            group
            .reset_index(
                drop=True
            )
        )

        candidate_idx = np.where(
            group["_split"]
            .to_numpy()
            ==
            split_name
        )[0]

        if len(candidate_idx) == 0:
            continue

        candidate_users += 1

        # ----------------------------------------------------
        # 固定 target：
        #
        # 当前 split 中最后一个交互，
        # 即 canonical leave-one-out target。
        #
        # 不因为 strict protocol 改 target。
        # ----------------------------------------------------

        target_idx = int(
            candidate_idx[-1]
        )

        target = group.iloc[
            target_idx
        ]

        # 标准历史：
        # target 之前的全部交互
        history = group.iloc[
            :target_idx
        ]

        if len(history) == 0:

            excluded_no_history += 1

            continue

        last_history_bucket = int(
            history[
                "time_bucket"
            ].iloc[-1]
        )

        target_bucket = int(
            target[
                "time_bucket"
            ]
        )

        # ----------------------------------------------------
        # 核心协议
        # ----------------------------------------------------

        if (
            target_bucket
            <=
            last_history_bucket
        ):

            excluded_same_bucket += 1

            continue

        # 到这里已经保证：
        #
        # target_bucket > last_history_bucket
        #
        # 因而 target 不属于历史最后桶。

        if len(history) < MIN_HISTORY:

            excluded_short_history += 1

            continue

        # ----------------------------------------------------
        # 最后历史桶统计
        # ----------------------------------------------------

        last_bucket_history = history[
            history[
                "time_bucket"
            ]
            ==
            last_history_bucket
        ]

        last_bucket_size = int(
            last_bucket_history[
                "item_id"
            ].nunique()
        )

        ambiguous_last_bucket = int(
            last_bucket_size >= 2
        )

        sample_id = (
            f"{dataset_name}"
            f"__{split_name}"
            f"__{target['raw_line']}"
        )

        samples.append(
            {
                "sample_id":
                    sample_id,

                "dataset":
                    dataset_name,

                "split":
                    split_name,

                "user_id":
                    str(
                        user_id
                    ),

                "target_item":
                    str(
                        target[
                            "item_id"
                        ]
                    ),

                "target_raw_line":
                    int(
                        target[
                            "raw_line"
                        ]
                    ),

                "target_timestamp":
                    int(
                        target[
                            "timestamp"
                        ]
                    ),

                "target_bucket":
                    target_bucket,

                "history_len":
                    int(
                        len(history)
                    ),

                "last_history_timestamp":
                    int(
                        history[
                            "timestamp"
                        ].iloc[-1]
                    ),

                "last_history_bucket":
                    last_history_bucket,

                "last_history_bucket_size":
                    last_bucket_size,

                "ambiguous_last_bucket":
                    ambiguous_last_bucket,

                "bucket_gap":
                    int(
                        target_bucket
                        -
                        last_history_bucket
                    ),
            }
        )

    sample_df = pd.DataFrame(
        samples
    )

    summary = {

        "dataset":
            dataset_name,

        "split":
            split_name,

        "candidate_users":
            candidate_users,

        "kept_samples":
            len(sample_df),

        "excluded_same_bucket":
            excluded_same_bucket,

        "excluded_short_history":
            excluded_short_history,

        "excluded_no_history":
            excluded_no_history,
    }

    if candidate_users > 0:

        summary[
            "keep_ratio"
        ] = (
            len(sample_df)
            /
            candidate_users
        )

    else:

        summary[
            "keep_ratio"
        ] = 0.0

    if len(sample_df) > 0:

        summary[
            "ambiguous_last_bucket_samples"
        ] = int(
            sample_df[
                "ambiguous_last_bucket"
            ].sum()
        )

        summary[
            "ambiguous_last_bucket_ratio"
        ] = float(
            sample_df[
                "ambiguous_last_bucket"
            ].mean()
        )

    else:

        summary[
            "ambiguous_last_bucket_samples"
        ] = 0

        summary[
            "ambiguous_last_bucket_ratio"
        ] = 0.0

    return (
        sample_df,
        summary
    )


# ============================================================
# 检查严格样本在 shuffled 下仍是同一 target / history set
# ============================================================

def validate_samples_against_shuffled(
    samples,
    shuffled,
    dataset_name,
    split_name
):

    if len(samples) == 0:
        return

    shuffled = shuffled.copy()

    shuffled["user_id"] = (
        shuffled["user_id"]
        .astype(str)
    )

    grouped = {
        str(user_id): group

        for (
            user_id,
            group
        ) in shuffled.groupby(
            "user_id",
            sort=False
        )
    }

    checked = 0

    for row in samples.itertuples(
        index=False
    ):

        user_id = str(
            row.user_id
        )

        if user_id not in grouped:

            raise RuntimeError(
                f"{dataset_name} "
                f"{split_name} "
                f"user不存在：{user_id}"
            )

        group = grouped[
            user_id
        ]

        target_match = group[
            group[
                "raw_line"
            ]
            ==
            row.target_raw_line
        ]

        if len(target_match) != 1:

            raise RuntimeError(
                f"{dataset_name} "
                f"{split_name} "
                "target无法唯一匹配"
            )

        # 严格协议下历史只能来自更早时间桶
        shuffled_history = group[
            group[
                "time_bucket"
            ]
            <
            row.target_bucket
        ]

        if len(
            shuffled_history
        ) != row.history_len:

            raise RuntimeError(
                f"{dataset_name} "
                f"{split_name} "
                f"sample={row.sample_id} "
                "history集合不一致"
            )

        checked += 1

    print(
        f"{split_name} shuffled配对检查："
        f"{checked:,} 个样本通过"
    )


# ============================================================
# 单数据集
# ============================================================

def process_dataset(
    dataset_name
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

    file_order_path = (
        INPUT_ROOT
        / dataset_name
        / "file_order.csv"
    )

    shuffled_path = (
        INPUT_ROOT
        / dataset_name
        / (
            f"bucket_shuffled_"
            f"seed{SHUFFLE_SEED}.csv"
        )
    )

    if not file_order_path.exists():
        raise FileNotFoundError(
            file_order_path
        )

    if not shuffled_path.exists():
        raise FileNotFoundError(
            shuffled_path
        )

    file_order = pd.read_csv(
        file_order_path
    )

    shuffled = pd.read_csv(
        shuffled_path
    )

    file_order["user_id"] = (
        file_order[
            "user_id"
        ].astype(str)
    )

    file_order["item_id"] = (
        file_order[
            "item_id"
        ].astype(str)
    )

    shuffled["user_id"] = (
        shuffled[
            "user_id"
        ].astype(str)
    )

    shuffled["item_id"] = (
        shuffled[
            "item_id"
        ].astype(str)
    )

    validate_paired_data(
        file_order,
        shuffled,
        dataset_name
    )

    file_order = (
        attach_split_labels(
            file_order,
            dataset_name
        )
    )

    out_dir = (
        OUTPUT_ROOT
        / dataset_name
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summaries = []

    for split_name in [
        "val",
        "test",
    ]:

        (
            samples,
            summary
        ) = build_strict_samples(
            file_order,
            split_name,
            dataset_name
        )

        output_name = (
            "validation_samples.csv"
            if split_name == "val"
            else "test_samples.csv"
        )

        output_path = (
            out_dir
            / output_name
        )

        samples.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        validate_samples_against_shuffled(
            samples,
            shuffled,
            dataset_name,
            split_name
        )

        print(
            f"\n{split_name.upper()}"
        )

        print(
            "候选用户：",
            summary[
                "candidate_users"
            ]
        )

        print(
            "严格协议保留：",
            summary[
                "kept_samples"
            ]
        )

        print(
            "target与最后历史桶同桶而排除：",
            summary[
                "excluded_same_bucket"
            ]
        )

        print(
            "历史不足而排除：",
            summary[
                "excluded_short_history"
            ]
        )

        print(
            "保留比例：",
            f"{summary['keep_ratio'] * 100:.4f}%"
        )

        print(
            "保留样本中最后历史桶歧义比例：",
            f"{summary['ambiguous_last_bucket_ratio'] * 100:.4f}%"
        )

        print(
            "输出：",
            output_path
        )

        summaries.append(
            summary
        )

    return summaries


# ============================================================
# 主程序
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "第26步：冻结严格评价协议"
    )

    print(
        "要求：target_bucket "
        "> last_history_bucket"
    )

    print(
        "=" * 80
    )

    all_summaries = []

    for dataset_name in DATASETS:

        summaries = process_dataset(
            dataset_name
        )

        all_summaries.extend(
            summaries
        )

    summary_df = pd.DataFrame(
        all_summaries
    )

    summary_path = (
        RUN_DIR
        / "strict_protocol_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    manifest = {

        "step":
            26,

        "protocol":
            (
                "target_bucket > "
                "last_history_bucket"
            ),

        "target_rule":
            (
                "last canonical interaction "
                "inside each validation/test split"
            ),

        "same_bucket_rule":
            (
                "exclude sample; "
                "do not modify target"
            ),

        "min_history":
            MIN_HISTORY,

        "canonical_order":
            (
                "time_bucket + raw_line"
            ),

        "paired_evaluation":
            True,

        "shuffle_seed":
            SHUFFLE_SEED,

        "datasets":
            all_summaries,
    }

    manifest_path = (
        RUN_DIR
        / "strict_protocol_manifest.json"
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

    print(
        "\n"
        + "=" * 80
    )

    print(
        "第26步汇总"
    )

    print(
        "=" * 80
    )

    show = summary_df.copy()

    show[
        "keep_ratio"
    ] = (
        show[
            "keep_ratio"
        ]
        * 100
    ).round(4)

    show[
        "ambiguous_last_bucket_ratio"
    ] = (
        show[
            "ambiguous_last_bucket_ratio"
        ]
        * 100
    ).round(4)

    print(
        show.to_string(
            index=False
        )
    )

    print(
        "\n保存："
    )

    print(
        summary_path
    )

    print(
        manifest_path
    )

    print(
        "\n第26步完成。"
    )


if __name__ == "__main__":
    main()