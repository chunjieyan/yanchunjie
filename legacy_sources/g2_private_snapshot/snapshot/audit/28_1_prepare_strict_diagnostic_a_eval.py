from pathlib import Path
import json

import pandas as pd
import numpy as np

# ============================================================
# 第28.1步
# Strict Diagnostic A evaluation preparation
#
# 目的：
# 为 file_order 与 bucket_shuffled_seed101
# 准备完全配对的严格 Val/Test 输入。
#
# 固定：
#   sample
#   target
#   history raw_line集合
#
# 只改变：
#   同一history集合中的桶内顺序
# ============================================================


ROOT = Path(r"D:\time_uncertainty_rec")

DATASETS = [
    "Amazon_Beauty",
    "MovieLens-1M",
]

ORDERS = [
    "file_order",
    "bucket_shuffled_seed101",
]

SPLITS = [
    "val",
    "test",
]


SOURCE_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_a"
)

CANONICAL_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_b"
)

OUTPUT_ROOT = (
    ROOT
    / "processed"
    / "strict_diagnostic_a"
)

RUN_DIR = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "diagnostic_a_strict"
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


EXPECTED = {
    "Amazon_Beauty": {
        "val": 2856,
        "test": 2966,
    },
    "MovieLens-1M": {
        "val": 752,
        "test": 945,
    },
}


# ============================================================
# 找到顺序列
# ============================================================

def detect_order_column(df):

    if "_source_order" not in df.columns:

        raise RuntimeError(
            "缺少_source_order"
        )

    return "_source_order"

# ============================================================
# 单数据集
# ============================================================

def process_dataset(dataset):

    print("\n" + "=" * 80)
    print("Dataset:", dataset)
    print("=" * 80)

    order_frames = {}

    for order_name in ORDERS:

        path = (
            SOURCE_ROOT
            / dataset
            / f"{order_name}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(
            path,
            dtype={
                "user_id": str,
                "item_id": str,
            }
        )
        # --------------------------------------------------------
        # 保存当前CSV中的真实物理行顺序
        #
        # file_order.csv 和 bucket_shuffled_seed101.csv
        # 的区别就保存在这里。
        #
        # raw_line只是原始记录ID，不能作为当前顺序。
        # --------------------------------------------------------

        df["_source_order"] = np.arange(
            len(df),
            dtype=np.int64
        )

        if "raw_line" not in df.columns:
            raise RuntimeError(
                f"{path} 缺少 raw_line"
            )

        df["raw_line"] = (
            df["raw_line"]
            .astype(int)
        )

        order_frames[
            order_name
        ] = df

        print(
            order_name,
            "records=",
            len(df),
            "order_col=",
            detect_order_column(df)
        )

    # --------------------------------------------------------
    # 两份数据必须是完全相同的记录集合
    # --------------------------------------------------------

    a = set(
        order_frames[
            "file_order"
        ][
            "raw_line"
        ].tolist()
    )

    b = set(
        order_frames[
            "bucket_shuffled_seed101"
        ][
            "raw_line"
        ].tolist()
    )

    if a != b:

        raise RuntimeError(
            f"{dataset}: 两种order记录集合不同"
        )

    dataset_summaries = []

    # ========================================================
    # val / test
    # ========================================================

    for split in SPLITS:

        print("\n---", split.upper(), "---")

        canonical_path = (
            CANONICAL_ROOT
            / dataset
            / split
            / "canonical.csv"
        )

        if not canonical_path.exists():
            raise FileNotFoundError(
                canonical_path
            )

        canonical = pd.read_csv(
            canonical_path,
            dtype={
                "sample_id": str,
                "user_id": str,
                "target_item": str,
                "sequence_raw_lines": str,
            }
        )

        expected = EXPECTED[
            dataset
        ][
            split
        ]

        if len(canonical) != expected:

            raise RuntimeError(
                f"{dataset}/{split}: "
                f"expected={expected}, "
                f"actual={len(canonical)}"
            )

        outputs = {}

        # ====================================================
        # 分别构造两种order
        # ====================================================

        for order_name in ORDERS:

            full_df = (
                order_frames[
                    order_name
                ]
            )

            order_col = detect_order_column(
                full_df
            )

            raw_lookup = (
                full_df
                .set_index(
                    "raw_line",
                    drop=False
                )
                .rename_axis(
                    "_raw_line_index"
                )
            )

            rows = []

            for row in canonical.itertuples(
                index=False
            ):

                raw_lines = [
                    int(x)
                    for x in json.loads(
                        row.sequence_raw_lines
                    )
                ]

                # --------------------------------------------
                # 使用完全固定的raw_line集合
                # --------------------------------------------

                try:

                    hist = raw_lookup.loc[
                        raw_lines
                    ].copy()

                except KeyError as e:

                    raise RuntimeError(
                        f"{dataset}/{split}/"
                        f"{order_name}/"
                        f"{row.sample_id}: "
                        "history raw_line缺失"
                    ) from e

                # --------------------------------------------
                # 防止索引重复导致异常
                # --------------------------------------------

                if len(hist) != len(
                    raw_lines
                ):

                    raise RuntimeError(
                        f"{row.sample_id}: "
                        "history记录数量变化"
                    )

                # --------------------------------------------
                # 强制检查user一致
                # --------------------------------------------

                hist_users = set(
                    hist[
                        "user_id"
                    ].astype(str)
                )

                if hist_users != {
                    str(row.user_id)
                }:

                    raise RuntimeError(
                        f"{row.sample_id}: "
                        "history user不一致"
                    )

                # --------------------------------------------
                # 根据当前order排序
                #
                # 优先：
                # time_bucket + sequence_order
                #
                # 若没有sequence_order，
                # detect_order_column会退回raw_line。
                # --------------------------------------------

                sort_cols = []

                if "time_bucket" in hist.columns:
                    sort_cols.append(
                        "time_bucket"
                    )

                if (
                    order_col
                    not in sort_cols
                ):
                    sort_cols.append(
                        order_col
                    )

                hist = (
                    hist
                    .sort_values(
                        sort_cols,
                        kind="mergesort"
                    )
                    .reset_index(
                        drop=True
                    )
                )

                seq_items = (
                    hist[
                        "item_id"
                    ]
                    .astype(str)
                    .tolist()
                )

                seq_raw_lines = (
                    hist[
                        "raw_line"
                    ]
                    .astype(int)
                    .tolist()
                )

                if len(
                    seq_items
                ) > 50:

                    raise RuntimeError(
                        f"{row.sample_id}: "
                        "history > 50"
                    )

                rows.append(
                    {
                        "sample_id":
                            str(
                                row.sample_id
                            ),

                        "user_id":
                            str(
                                row.user_id
                            ),

                        "target_item":
                            str(
                                row.target_item
                            ),

                        "target_raw_line":
                            int(
                                row.target_raw_line
                            ),

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
                            int(
                                row.last_history_bucket_size
                            ),

                        "fixed_history_len":
                            len(
                                seq_items
                            ),

                        "sequence_items":
                            json.dumps(
                                seq_items,
                                ensure_ascii=False
                            ),

                        "sequence_raw_lines":
                            json.dumps(
                                seq_raw_lines
                            ),
                    }
                )

            out_df = pd.DataFrame(
                rows
            )

            out_dir = (
                OUTPUT_ROOT
                / dataset
                / order_name
            )

            out_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            out_path = (
                out_dir
                / f"{split}.csv"
            )

            out_df.to_csv(
                out_path,
                index=False,
                encoding="utf-8-sig"
            )

            outputs[
                order_name
            ] = out_df

            print(
                order_name,
                "samples=",
                len(out_df),
                "->",
                out_path
            )

        # ====================================================
        # 严格配对检查
        # ====================================================

        file_df = outputs[
            "file_order"
        ]

        shuffle_df = outputs[
            "bucket_shuffled_seed101"
        ]

        fixed_columns = [
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

        for col in fixed_columns:

            if not file_df[
                col
            ].equals(
                shuffle_df[
                    col
                ]
            ):

                raise RuntimeError(
                    f"{dataset}/{split}: "
                    f"{col}不一致"
                )

        changed_order = 0
        changed_item_set = 0

        for i in range(
            len(file_df)
        ):

            f_lines = [
                int(x)
                for x in json.loads(
                    file_df.iloc[
                        i
                    ][
                        "sequence_raw_lines"
                    ]
                )
            ]

            s_lines = [
                int(x)
                for x in json.loads(
                    shuffle_df.iloc[
                        i
                    ][
                        "sequence_raw_lines"
                    ]
                )
            ]

            # --------------------------------------------
            # 集合绝对不能变化
            # --------------------------------------------

            if sorted(
                f_lines
            ) != sorted(
                s_lines
            ):

                changed_item_set += 1

            # --------------------------------------------
            # 顺序是否变化
            # --------------------------------------------

            if f_lines != s_lines:

                changed_order += 1

        if changed_item_set != 0:

            raise RuntimeError(
                f"{dataset}/{split}: "
                f"发现{changed_item_set}个history集合变化"
            )

        changed_ratio = (
            changed_order
            /
            len(file_df)
        )

        print(
            "history集合变化：",
            changed_item_set
        )

        print(
            "history顺序发生变化：",
            f"{changed_order}/{len(file_df)} "
            f"({changed_ratio * 100:.2f}%)"
        )

        dataset_summaries.append(
            {
                "dataset":
                    dataset,

                "split":
                    split,

                "samples":
                    len(file_df),

                "history_set_changed":
                    changed_item_set,

                "history_order_changed":
                    changed_order,

                "history_order_changed_ratio":
                    changed_ratio,
            }
        )

    return dataset_summaries


# ============================================================
# main
# ============================================================

def main():

    print("=" * 80)
    print("第28.1步：Strict Diagnostic A Protocol")
    print("=" * 80)

    all_results = []

    for dataset in DATASETS:

        all_results.extend(
            process_dataset(
                dataset
            )
        )

    summary = pd.DataFrame(
        all_results
    )

    summary_path = (
        RUN_DIR
        / "strict_diagnostic_a_protocol_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("第28.1步完成")
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSummary：",
        summary_path
    )


if __name__ == "__main__":
    main()