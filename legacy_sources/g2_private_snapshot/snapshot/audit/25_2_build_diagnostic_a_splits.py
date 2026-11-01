from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# 第25.2步
# 诊断A：建立严格配对的全局时间 Train / Val / Test 划分
#
# 协议沿用第1—22步：
#
# 全体交互 timestamp：
# 0% ---- 80% -------- 90% -------- 100%
#         Train         Val          Test
#
# 两个版本：
# 1. file_order
# 2. bucket_shuffled_seed101
#
# 必须保证：
# - 时间边界完全相同
# - Train 样本完全相同
# - Val 样本完全相同
# - Test 样本完全相同
# - 唯一差别 = 桶内排列
# ============================================================


ROOT = Path(__file__).resolve().parent.parent

INPUT_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_a"
)

OUTPUT_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_a_splits"
)

RUN_DIR = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "diagnostic_a"
    / "splits"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SHUFFLE_SEED = 101


DATASETS = {

    "Amazon_Beauty": {
        "file_order":
            INPUT_ROOT
            / "Amazon_Beauty"
            / "file_order.csv",

        "shuffled":
            INPUT_ROOT
            / "Amazon_Beauty"
            / f"bucket_shuffled_seed{SHUFFLE_SEED}.csv",
    },

    "MovieLens-1M": {
        "file_order":
            INPUT_ROOT
            / "MovieLens-1M"
            / "file_order.csv",

        "shuffled":
            INPUT_ROOT
            / "MovieLens-1M"
            / f"bucket_shuffled_seed{SHUFFLE_SEED}.csv",
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
# 计算全局时间边界
# ============================================================

def get_global_time_cutoffs(df):

    timestamps = (
        df["timestamp"]
        .to_numpy(
            dtype=np.float64
        )
    )

    train_end = float(
        np.quantile(
            timestamps,
            0.80
        )
    )

    val_end = float(
        np.quantile(
            timestamps,
            0.90
        )
    )

    return train_end, val_end


# ============================================================
# 划分
# ============================================================

def split_by_global_time(
    df,
    train_end,
    val_end
):

    train = df[
        df["timestamp"]
        <= train_end
    ].copy()

    val = df[
        (
            df["timestamp"]
            > train_end
        )
        &
        (
            df["timestamp"]
            <= val_end
        )
    ].copy()

    test = df[
        df["timestamp"]
        > val_end
    ].copy()

    return train, val, test


# ============================================================
# 检查两个版本是否包含相同记录
# ============================================================

def assert_same_records(
    a,
    b,
    dataset_name,
    split_name
):

    if len(a) != len(b):

        raise RuntimeError(
            f"{dataset_name} {split_name} "
            f"行数不一致："
            f"{len(a)} vs {len(b)}"
        )

    # raw_line 是原始记录的唯一追踪标识
    a_ids = np.sort(
        a["raw_line"]
        .to_numpy()
    )

    b_ids = np.sort(
        b["raw_line"]
        .to_numpy()
    )

    if not np.array_equal(
        a_ids,
        b_ids
    ):

        raise RuntimeError(
            f"{dataset_name} {split_name} "
            "两个版本包含的原始记录不一致"
        )

    print(
        f"{dataset_name} {split_name}："
        f"配对检查通过，"
        f"{len(a):,} 条"
    )


# ============================================================
# 检查时间泄漏
# ============================================================

def validate_time_split(
    train,
    val,
    test,
    train_end,
    val_end,
    dataset_name
):

    if len(train) > 0:

        if not (
            train["timestamp"].max()
            <= train_end
        ):
            raise RuntimeError(
                f"{dataset_name} Train时间泄漏"
            )

    if len(val) > 0:

        if not (
            val["timestamp"].min()
            > train_end
        ):
            raise RuntimeError(
                f"{dataset_name} Val左边界错误"
            )

        if not (
            val["timestamp"].max()
            <= val_end
        ):
            raise RuntimeError(
                f"{dataset_name} Val右边界错误"
            )

    if len(test) > 0:

        if not (
            test["timestamp"].min()
            > val_end
        ):
            raise RuntimeError(
                f"{dataset_name} Test时间泄漏"
            )


# ============================================================
# 保存一个版本
# ============================================================

def save_version(
    dataset_name,
    version_name,
    train,
    val,
    test
):

    out_dir = (
        OUTPUT_ROOT
        / dataset_name
        / version_name
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    paths = {

        "train":
            out_dir / "train.csv",

        "val":
            out_dir / "val.csv",

        "test":
            out_dir / "test.csv",
    }

    train.to_csv(
        paths["train"],
        index=False,
        encoding="utf-8-sig"
    )

    val.to_csv(
        paths["val"],
        index=False,
        encoding="utf-8-sig"
    )

    test.to_csv(
        paths["test"],
        index=False,
        encoding="utf-8-sig"
    )

    return paths


# ============================================================
# 单数据集
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

    file_order_path = (
        config["file_order"]
    )

    shuffled_path = (
        config["shuffled"]
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

    print(
        f"file_order："
        f"{len(file_order):,}"
    )

    print(
        f"shuffled："
        f"{len(shuffled):,}"
    )

    # --------------------------------------------------------
    # 重要：
    # 时间边界只根据 file_order 计算一次
    #
    # shuffled 使用完全相同的边界
    # --------------------------------------------------------

    train_end, val_end = (
        get_global_time_cutoffs(
            file_order
        )
    )

    print(
        f"\nTrain 80% 时间边界："
        f"{train_end}"
    )

    print(
        f"Validation 90% 时间边界："
        f"{val_end}"
    )

    # --------------------------------------------------------
    # file order
    # --------------------------------------------------------

    (
        fo_train,
        fo_val,
        fo_test
    ) = split_by_global_time(
        file_order,
        train_end,
        val_end
    )

    # --------------------------------------------------------
    # shuffled
    # --------------------------------------------------------

    (
        sh_train,
        sh_val,
        sh_test
    ) = split_by_global_time(
        shuffled,
        train_end,
        val_end
    )

    # --------------------------------------------------------
    # 检查时间划分
    # --------------------------------------------------------

    validate_time_split(
        fo_train,
        fo_val,
        fo_test,
        train_end,
        val_end,
        dataset_name
    )

    validate_time_split(
        sh_train,
        sh_val,
        sh_test,
        train_end,
        val_end,
        dataset_name
    )

    # --------------------------------------------------------
    # 严格配对检查
    # --------------------------------------------------------

    print(
        "\n开始严格配对检查："
    )

    assert_same_records(
        fo_train,
        sh_train,
        dataset_name,
        "Train"
    )

    assert_same_records(
        fo_val,
        sh_val,
        dataset_name,
        "Validation"
    )

    assert_same_records(
        fo_test,
        sh_test,
        dataset_name,
        "Test"
    )

    # --------------------------------------------------------
    # 总数检查
    # --------------------------------------------------------

    total = (
        len(fo_train)
        +
        len(fo_val)
        +
        len(fo_test)
    )

    if total != len(file_order):

        raise RuntimeError(
            f"{dataset_name} "
            "Train+Val+Test数量不等于总数据"
        )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    fo_paths = save_version(
        dataset_name,
        "file_order",
        fo_train,
        fo_val,
        fo_test
    )

    sh_paths = save_version(
        dataset_name,
        f"bucket_shuffled_seed{SHUFFLE_SEED}",
        sh_train,
        sh_val,
        sh_test
    )

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    result = {

        "dataset":
            dataset_name,

        "total_interactions":
            len(file_order),

        "train_end_timestamp":
            train_end,

        "val_end_timestamp":
            val_end,

        "train_count":
            len(fo_train),

        "val_count":
            len(fo_val),

        "test_count":
            len(fo_test),

        "train_ratio":
            len(fo_train)
            / len(file_order),

        "val_ratio":
            len(fo_val)
            / len(file_order),

        "test_ratio":
            len(fo_test)
            / len(file_order),

        "shuffle_seed":
            SHUFFLE_SEED,

        "file_order_train_md5":
            md5_file(
                fo_paths["train"]
            ),

        "file_order_val_md5":
            md5_file(
                fo_paths["val"]
            ),

        "file_order_test_md5":
            md5_file(
                fo_paths["test"]
            ),

        "shuffled_train_md5":
            md5_file(
                sh_paths["train"]
            ),

        "shuffled_val_md5":
            md5_file(
                sh_paths["val"]
            ),

        "shuffled_test_md5":
            md5_file(
                sh_paths["test"]
            ),
    }

    print(
        "\n划分结果："
    )

    print(
        f"Train："
        f"{len(fo_train):,} "
        f"({len(fo_train) / len(file_order) * 100:.4f}%)"
    )

    print(
        f"Validation："
        f"{len(fo_val):,} "
        f"({len(fo_val) / len(file_order) * 100:.4f}%)"
    )

    print(
        f"Test："
        f"{len(fo_test):,} "
        f"({len(fo_test) / len(file_order) * 100:.4f}%)"
    )

    return result


# ============================================================
# 主程序
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "第25.2步：诊断A全局时间划分"
    )

    print(
        "协议：全局 timestamp 80% / 10% / 10%"
    )

    print(
        "=" * 80
    )

    results = []

    for (
        dataset_name,
        config
    ) in DATASETS.items():

        result = process_dataset(
            dataset_name,
            config
        )

        results.append(
            result
        )

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    summary_path = (
        RUN_DIR
        / "diagnostic_a_split_summary.csv"
    )

    result_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # JSON manifest
    # --------------------------------------------------------

    manifest = {

        "step":
            "25.2",

        "protocol":
            "global timestamp split",

        "train_quantile":
            0.80,

        "validation_quantile":
            0.90,

        "rules": {

            "train":
                "timestamp <= train_end",

            "validation":
                "train_end < timestamp <= val_end",

            "test":
                "timestamp > val_end",

            "paired_versions":
                True,

            "shuffle_seed":
                SHUFFLE_SEED,
        },

        "datasets":
            results,
    }

    manifest_path = (
        RUN_DIR
        / "diagnostic_a_split_manifest.json"
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
    # 最终打印
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "第25.2步最终结果"
    )

    print(
        "=" * 80
    )

    show = result_df[
        [
            "dataset",
            "total_interactions",
            "train_count",
            "val_count",
            "test_count",
            "train_ratio",
            "val_ratio",
            "test_ratio",
        ]
    ].copy()

    for col in [
        "train_ratio",
        "val_ratio",
        "test_ratio",
    ]:

        show[col] = (
            show[col]
            * 100
        ).round(4)

    print(
        show.to_string(
            index=False
        )
    )

    print(
        "\n摘要："
    )

    print(
        summary_path
    )

    print(
        manifest_path
    )

    print(
        "\n第25.2步完成。"
    )


if __name__ == "__main__":
    main()