from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# 第27.1步
# Diagnostic B：严格协议下生成5套配对桶内排列
#
# 固定：
#   target
#   history item集合
#   最大历史长度 MAX_LEN=50
#
# 变化：
#   同一 time_bucket 内部的排列
#
# 排列 seeds：
#   101, 202, 303, 404, 505
# ============================================================


ROOT = Path(__file__).resolve().parent.parent

SOURCE_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_a"
)

STRICT_ROOT = (
    ROOT
    / "processed"
    / "strict_protocol"
)

OUTPUT_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_b"
)

RUN_DIR = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "diagnostic_b"
    / "permutations"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MAX_LEN = 50

PERMUTATION_SEEDS = [
    101,
    202,
    303,
    404,
    505,
]


EXPECTED_COUNTS = {

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
# 稳定、可复现的样本级随机种子
# ============================================================

def sample_rng_seed(
    sample_id,
    permutation_seed
):

    text = (
        f"{sample_id}"
        f"__{permutation_seed}"
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False
    )


# ============================================================
# 在固定历史集合内部按桶随机排列
# ============================================================

def permute_fixed_history(
    history,
    sample_id,
    permutation_seed
):

    pieces = []

    # history已经按：
    # time_bucket + raw_line
    # 排好 canonical 顺序
    for (
        bucket,
        bucket_df
    ) in history.groupby(
        "time_bucket",
        sort=True
    ):

        bucket_df = (
            bucket_df.copy()
        )

        if len(bucket_df) >= 2:

            rng = np.random.default_rng(
                sample_rng_seed(
                    f"{sample_id}__{bucket}",
                    permutation_seed
                )
            )

            order = rng.permutation(
                len(bucket_df)
            )

            bucket_df = (
                bucket_df
                .iloc[order]
            )

        pieces.append(
            bucket_df
        )

    return pd.concat(
        pieces,
        ignore_index=True
    )


# ============================================================
# 单个split
# ============================================================

def process_split(
    dataset_name,
    split_name,
    full_df,
    samples
):

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"{dataset_name} / {split_name}"
    )

    print(
        "-" * 80
    )

    expected = (
        EXPECTED_COUNTS[
            dataset_name
        ][
            split_name
        ]
    )

    if len(samples) != expected:

        raise RuntimeError(
            f"{dataset_name} {split_name} "
            f"样本数变化："
            f"expected={expected}, "
            f"actual={len(samples)}"
        )

    print(
        "严格协议样本数：",
        len(samples)
    )

    print(
        "最后历史桶歧义样本：",
        int(
            samples[
                "ambiguous_last_bucket"
            ].sum()
        )
    )

    # --------------------------------------------------------
    # 按用户缓存，减少重复筛选
    # --------------------------------------------------------

    user_groups = {}

    for (
        user_id,
        group
    ) in full_df.groupby(
        "user_id",
        sort=False
    ):

        user_groups[
            str(user_id)
        ] = (
            group.sort_values(
                [
                    "time_bucket",
                    "raw_line",
                ],
                kind="mergesort"
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # 先构造 canonical fixed-history
    #
    # 关键：
    # 先固定最后 MAX_LEN 条记录，
    # 后续5个排列全部使用完全相同的记录集合。
    # --------------------------------------------------------

    canonical_cache = {}

    for row in samples.itertuples(
        index=False
    ):

        sample_id = str(
            row.sample_id
        )

        user_id = str(
            row.user_id
        )

        target_bucket = int(
            row.target_bucket
        )

        if user_id not in user_groups:

            raise RuntimeError(
                f"找不到user：{user_id}"
            )

        group = user_groups[
            user_id
        ]

        history = group[
            group[
                "time_bucket"
            ]
            <
            target_bucket
        ].copy()

        # 与第26步的完整历史长度核对
        if len(history) != int(
            row.history_len
        ):

            raise RuntimeError(
                f"{sample_id} history_len不一致："
                f"step26={row.history_len}, "
                f"step27={len(history)}"
            )

        # canonical顺序
        history = history.sort_values(
            [
                "time_bucket",
                "raw_line",
            ],
            kind="mergesort"
        )

        # ----------------------------------------------------
        # 先固定最后50条
        # ----------------------------------------------------

        fixed_history = (
            history
            .tail(
                MAX_LEN
            )
            .copy()
            .reset_index(
                drop=True
            )
        )

        if len(
            fixed_history
        ) == 0:

            raise RuntimeError(
                f"{sample_id} history为空"
            )

        if not (
            int(
                fixed_history[
                    "time_bucket"
                ].max()
            )
            <
            target_bucket
        ):

            raise RuntimeError(
                f"{sample_id} "
                "违反严格协议"
            )

        canonical_cache[
            sample_id
        ] = fixed_history

    # --------------------------------------------------------
    # 保存 canonical 版本
    # --------------------------------------------------------

    dataset_out = (
        OUTPUT_ROOT
        / dataset_name
        / split_name
    )

    dataset_out.mkdir(
        parents=True,
        exist_ok=True
    )

    canonical_rows = []

    for row in samples.itertuples(
        index=False
    ):

        sample_id = str(
            row.sample_id
        )

        history = (
            canonical_cache[
                sample_id
            ]
        )

        canonical_rows.append(
            {
                "sample_id":
                    sample_id,

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
                    len(history),

                "sequence_items":
                    json.dumps(
                        history[
                            "item_id"
                        ]
                        .astype(str)
                        .tolist(),
                        ensure_ascii=False
                    ),

                "sequence_raw_lines":
                    json.dumps(
                        history[
                            "raw_line"
                        ]
                        .astype(int)
                        .tolist()
                    ),
            }
        )

    canonical_df = pd.DataFrame(
        canonical_rows
    )

    canonical_path = (
        dataset_out
        / "canonical.csv"
    )

    canonical_df.to_csv(
        canonical_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 五个排列
    # --------------------------------------------------------

    summaries = []

    for permutation_seed in (
        PERMUTATION_SEEDS
    ):

        output_rows = []

        changed_any = 0
        changed_last_bucket = 0

        ambiguous_total = int(
            samples[
                "ambiguous_last_bucket"
            ].sum()
        )

        ambiguous_last_changed = 0

        for row in samples.itertuples(
            index=False
        ):

            sample_id = str(
                row.sample_id
            )

            canonical_history = (
                canonical_cache[
                    sample_id
                ]
            )

            permuted = (
                permute_fixed_history(
                    canonical_history,
                    sample_id,
                    permutation_seed
                )
            )

            canonical_lines = (
                canonical_history[
                    "raw_line"
                ]
                .astype(int)
                .tolist()
            )

            permuted_lines = (
                permuted[
                    "raw_line"
                ]
                .astype(int)
                .tolist()
            )

            # ----------------------------------------------
            # item集合必须100%相同
            # ----------------------------------------------

            if sorted(
                canonical_lines
            ) != sorted(
                permuted_lines
            ):

                raise RuntimeError(
                    f"{sample_id} "
                    "排列后history集合改变"
                )

            if (
                canonical_lines
                !=
                permuted_lines
            ):

                changed_any += 1

            # ----------------------------------------------
            # 单独看最后历史桶是否发生变化
            # ----------------------------------------------

            last_bucket = int(
                canonical_history[
                    "time_bucket"
                ].max()
            )

            canonical_last = (
                canonical_history[
                    canonical_history[
                        "time_bucket"
                    ]
                    ==
                    last_bucket
                ][
                    "raw_line"
                ]
                .astype(int)
                .tolist()
            )

            permuted_last = (
                permuted[
                    permuted[
                        "time_bucket"
                    ]
                    ==
                    last_bucket
                ][
                    "raw_line"
                ]
                .astype(int)
                .tolist()
            )

            last_changed = int(
                canonical_last
                !=
                permuted_last
            )

            changed_last_bucket += (
                last_changed
            )

            if int(
                row.ambiguous_last_bucket
            ) == 1:

                ambiguous_last_changed += (
                    last_changed
                )

            output_rows.append(
                {
                    "sample_id":
                        sample_id,

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
                            permuted
                        ),

                    "permutation_seed":
                        permutation_seed,

                    "sequence_items":
                        json.dumps(
                            permuted[
                                "item_id"
                            ]
                            .astype(str)
                            .tolist(),
                            ensure_ascii=False
                        ),

                    "sequence_raw_lines":
                        json.dumps(
                            permuted[
                                "raw_line"
                            ]
                            .astype(int)
                            .tolist()
                        ),
                }
            )

        out_df = pd.DataFrame(
            output_rows
        )

        out_path = (
            dataset_out
            / (
                f"permutation_seed"
                f"{permutation_seed}.csv"
            )
        )

        out_df.to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig"
        )

        # ----------------------------------------------
        # 与canonical target严格核对
        # ----------------------------------------------

        if not np.array_equal(
            canonical_df[
                "target_raw_line"
            ].to_numpy(),
            out_df[
                "target_raw_line"
            ].to_numpy()
        ):

            raise RuntimeError(
                "target发生变化"
            )

        if not np.array_equal(
            canonical_df[
                "sample_id"
            ].to_numpy(),
            out_df[
                "sample_id"
            ].to_numpy()
        ):

            raise RuntimeError(
                "sample_id发生变化"
            )

        changed_ratio = (
            changed_any
            /
            len(samples)
        )

        if ambiguous_total > 0:

            ambiguous_last_changed_ratio = (
                ambiguous_last_changed
                /
                ambiguous_total
            )

        else:

            ambiguous_last_changed_ratio = (
                0.0
            )

        summary = {
            "dataset":
                dataset_name,

            "split":
                split_name,

            "permutation_seed":
                permutation_seed,

            "samples":
                len(samples),

            "ambiguous_last_bucket_samples":
                ambiguous_total,

            "any_history_order_changed":
                changed_any,

            "any_history_order_changed_ratio":
                changed_ratio,

            "last_bucket_order_changed":
                changed_last_bucket,

            "ambiguous_last_bucket_changed":
                ambiguous_last_changed,

            "ambiguous_last_bucket_changed_ratio":
                ambiguous_last_changed_ratio,

            "output_file":
                str(
                    out_path
                ),
        }

        summaries.append(
            summary
        )

        print(
            f"seed={permutation_seed} "
            f"| 任意历史顺序变化 "
            f"{changed_any:,}/"
            f"{len(samples):,} "
            f"({changed_ratio * 100:.2f}%) "
            f"| 歧义最后桶发生变化 "
            f"{ambiguous_last_changed:,}/"
            f"{ambiguous_total:,} "
            f"({ambiguous_last_changed_ratio * 100:.2f}%)"
        )

    print(
        "canonical：",
        canonical_path
    )

    return summaries


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

    source_path = (
        SOURCE_ROOT
        / dataset_name
        / "file_order.csv"
    )

    if not source_path.exists():

        raise FileNotFoundError(
            source_path
        )

    full_df = pd.read_csv(
        source_path
    )

    full_df[
        "user_id"
    ] = (
        full_df[
            "user_id"
        ].astype(str)
    )

    full_df[
        "item_id"
    ] = (
        full_df[
            "item_id"
        ].astype(str)
    )

    results = []

    for split_name in [
        "val",
        "test",
    ]:

        sample_filename = (
            "validation_samples.csv"
            if split_name == "val"
            else "test_samples.csv"
        )

        sample_path = (
            STRICT_ROOT
            / dataset_name
            / sample_filename
        )

        if not sample_path.exists():

            raise FileNotFoundError(
                sample_path
            )

        samples = pd.read_csv(
            sample_path
        )

        samples[
            "user_id"
        ] = (
            samples[
                "user_id"
            ].astype(str)
        )

        samples[
            "target_item"
        ] = (
            samples[
                "target_item"
            ].astype(str)
        )

        split_results = (
            process_split(
                dataset_name,
                split_name,
                full_df,
                samples
            )
        )

        results.extend(
            split_results
        )

    return results


# ============================================================
# 主程序
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "第27.1步：Diagnostic B"
    )

    print(
        "严格协议 + 5种桶内排列"
    )

    print(
        "=" * 80
    )

    all_results = []

    for dataset_name in [
        "Amazon_Beauty",
        "MovieLens-1M",
    ]:

        results = process_dataset(
            dataset_name
        )

        all_results.extend(
            results
        )

    summary_df = pd.DataFrame(
        all_results
    )

    summary_path = (
        RUN_DIR
        / "permutation_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    manifest = {

        "step":
            "27.1",

        "purpose":
            (
                "Diagnostic B paired "
                "tie-breaking permutations"
            ),

        "strict_protocol":
            (
                "target_bucket > "
                "last_history_bucket"
            ),

        "max_len":
            MAX_LEN,

        "permutation_seeds":
            PERMUTATION_SEEDS,

        "critical_rule":
            (
                "freeze history item set "
                "before within-bucket permutation"
            ),

        "results":
            all_results,
    }

    manifest_path = (
        RUN_DIR
        / "permutation_manifest.json"
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
        "第27.1步完成"
    )

    print(
        "=" * 80
    )

    print(
        summary_df[
            [
                "dataset",
                "split",
                "permutation_seed",
                "samples",
                "ambiguous_last_bucket_samples",
                "any_history_order_changed_ratio",
                "ambiguous_last_bucket_changed_ratio",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n摘要：",
        summary_path
    )

    print(
        "Manifest：",
        manifest_path
    )


if __name__ == "__main__":
    main()