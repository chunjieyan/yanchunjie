from pathlib import Path
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 第27.2步
# Diagnostic B：
#
# 固定一个 SASRec Control checkpoint
# 对5种桶内排列分别进行全量物品排序
#
# 修正版：
# 1. 不重新训练模型
# 2. target固定
# 3. history item集合固定
# 4. 对左侧padding序列采用无padding推理
# 5. 保留原MAX_LEN右对齐的位置编码
# 6. 强制检查NaN / Inf
#
# 输出：
#   target_rank
#   top20
#   hit10
#   hit20
#   ndcg10
#
# 下一步27.3再计算：
#   Rank Std
#   Pairwise Top-10 overlap
#   NDCG@10 range
# ============================================================


# ============================================================
# 0. 命令行参数
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    required=True,
    choices=[
        "Amazon_Beauty",
        "MovieLens-1M",
    ]
)

parser.add_argument(
    "--model_seed",
    type=int,
    default=42
)

parser.add_argument(
    "--split",
    default="both",
    choices=[
        "val",
        "test",
        "both",
    ]
)

args = parser.parse_args()


# ============================================================
# 1. 基础路径
# ============================================================

ROOT = Path(
    r"D:\time_uncertainty_rec"
)

DATASET = args.dataset
MODEL_SEED = args.model_seed

PERMUTATION_SEEDS = [
    101,
    202,
    303,
    404,
    505,
]

BATCH_SIZE = 128


# ============================================================
# 2. 模型输入路径
# ============================================================

MODEL_DIR = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "diagnostic_a_strict"
    / "sasrec"
    / DATASET
    / "file_order"
    / f"seed{MODEL_SEED}"
)

CHECKPOINT_PATH = (
    MODEL_DIR
    / "sasrec_best.pt"
)

ITEM_MAP_PATH = (
    MODEL_DIR
    / "item_id_map.csv"
)


# ============================================================
# 3. Diagnostic B输入
# ============================================================

INPUT_ROOT = (
    ROOT
    / "processed"
    / "diagnostic_b"
    / DATASET
)


# ============================================================
# 4. 输出目录
# ============================================================

OUTPUT_ROOT = (
    ROOT
    / "runs"
    / "phase2_natural_timestamp"
    / "diagnostic_b"
    / "control_eval"
    / DATASET
    / f"seed{MODEL_SEED}"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. 第26步固定样本数
# ============================================================

EXPECTED_SAMPLES = {

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
# 6. 设备
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 80)
print("第27.2步：Diagnostic B Control 推理（修正版）")
print("=" * 80)

print(
    "数据集：",
    DATASET
)

print(
    "模型训练 seed：",
    MODEL_SEED
)

print(
    "设备：",
    device
)

if torch.cuda.is_available():

    print(
        "GPU：",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 7. SASRec模型
#
# 模块结构与25.3训练版本保持一致
# ============================================================

class SASRec(nn.Module):

    def __init__(
        self,
        num_items,
        max_len,
        hidden_size,
        num_heads,
        num_blocks,
        dropout
    ):

        super().__init__()

        self.num_items = (
            num_items
        )

        self.max_len = (
            max_len
        )

        self.hidden_size = (
            hidden_size
        )


        # item 0 = padding
        self.item_embedding = nn.Embedding(
            num_items + 1,
            hidden_size,
            padding_idx=0
        )


        self.position_embedding = nn.Embedding(
            max_len,
            hidden_size
        )


        self.dropout = nn.Dropout(
            dropout
        )


        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=
                    hidden_size,

                nhead=
                    num_heads,

                dim_feedforward=
                    hidden_size * 4,

                dropout=
                    dropout,

                activation=
                    "gelu",

                batch_first=
                    True,

                norm_first=
                    True
            )
        )


        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=
                    num_blocks
            )
        )


        self.final_norm = (
            nn.LayerNorm(
                hidden_size
            )
        )


    # --------------------------------------------------------
    # 原始有padding版本
    # 只用于必要的对照，不用于本次主要推理
    # --------------------------------------------------------

    def encode(
        self,
        seq
    ):

        batch_size, seq_len = (
            seq.shape
        )

        padding_mask = (
            seq == 0
        )

        positions = (
            torch.arange(
                seq_len,
                device=seq.device
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                seq_len
            )
        )

        x = (
            self.item_embedding(
                seq
            )
            *
            math.sqrt(
                self.hidden_size
            )
        )

        x = (
            x
            +
            self.position_embedding(
                positions
            )
        )

        x = self.dropout(
            x
        )

        x = x.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0
        )

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=seq.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        x = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=
                padding_mask
        )

        x = self.final_norm(
            x
        )

        return x


    def predict_all(
        self,
        seq
    ):

        features = self.encode(
            seq
        )

        final_feature = (
            features[:, -1, :]
        )

        scores = torch.matmul(
            final_feature,
            self.item_embedding.weight.T
        )

        return scores


# ============================================================
# 8. 无padding推理函数
#
# 关键修复：
#
# 对有效长度 < MAX_LEN 的序列：
# 去掉左侧padding，
# 但是位置编码仍使用原右对齐位置。
#
# 例如：
#
# MAX_LEN = 50
# history length = 20
#
# 原位置：
# 30,31,...,49
#
# 不能重新变成：
# 0,1,...,19
# ============================================================

def predict_all_unpadded(
    model,
    seq,
    position_offset
):

    batch_size, seq_len = (
        seq.shape
    )


    if seq_len <= 0:

        raise RuntimeError(
            "无padding推理收到空序列"
        )


    # --------------------------------------------------------
    # 保留原来的绝对位置
    # --------------------------------------------------------

    positions = (
        torch.arange(
            position_offset,
            position_offset + seq_len,
            device=seq.device
        )
        .unsqueeze(0)
        .expand(
            batch_size,
            seq_len
        )
    )


    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    x = (
        model.item_embedding(
            seq
        )
        *
        math.sqrt(
            model.hidden_size
        )
    )


    x = (
        x
        +
        model.position_embedding(
            positions
        )
    )


    x = model.dropout(
        x
    )


    # --------------------------------------------------------
    # 因果mask
    #
    # 已经没有padding，
    # 所以不再传src_key_padding_mask
    # --------------------------------------------------------

    causal_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            device=seq.device,
            dtype=torch.bool
        ),
        diagonal=1
    )


    x = model.encoder(
        x,
        mask=causal_mask
    )


    x = model.final_norm(
        x
    )


    # --------------------------------------------------------
    # 最后一个真实历史位置
    # --------------------------------------------------------

    final_feature = (
        x[:, -1, :]
    )


    scores = torch.matmul(
        final_feature,
        model.item_embedding.weight.T
    )


    return scores


# ============================================================
# 9. 加载checkpoint
# ============================================================

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"找不到checkpoint："
        f"{CHECKPOINT_PATH}"
    )


if not ITEM_MAP_PATH.exists():

    raise FileNotFoundError(
        f"找不到item map："
        f"{ITEM_MAP_PATH}"
    )


print(
    "\nCheckpoint：",
    CHECKPOINT_PATH
)


try:

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

except TypeError:

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )


required_keys = [
    "model_state_dict",
    "num_items",
    "max_len",
    "hidden_size",
    "num_heads",
    "num_blocks",
    "dropout",
    "best_val_ndcg10",
    "epoch",
    "seed",
]


for key in required_keys:

    if key not in checkpoint:

        raise RuntimeError(
            f"checkpoint缺少：{key}"
        )


print(
    "Checkpoint keys：",
    list(
        checkpoint.keys()
    )
)


NUM_ITEMS = int(
    checkpoint[
        "num_items"
    ]
)

MAX_LEN = int(
    checkpoint[
        "max_len"
    ]
)

HIDDEN_SIZE = int(
    checkpoint[
        "hidden_size"
    ]
)

NUM_HEADS = int(
    checkpoint[
        "num_heads"
    ]
)

NUM_BLOCKS = int(
    checkpoint[
        "num_blocks"
    ]
)

DROPOUT = float(
    checkpoint[
        "dropout"
    ]
)


print(
    "NUM_ITEMS：",
    NUM_ITEMS
)

print(
    "MAX_LEN：",
    MAX_LEN
)

print(
    "Hidden：",
    HIDDEN_SIZE
)

print(
    "Blocks：",
    NUM_BLOCKS
)

print(
    "Heads：",
    NUM_HEADS
)

print(
    "Dropout：",
    DROPOUT
)

print(
    "Checkpoint训练seed：",
    checkpoint[
        "seed"
    ]
)

print(
    "Checkpoint最佳Val NDCG@10：",
    checkpoint[
        "best_val_ndcg10"
    ]
)

print(
    "Checkpoint最佳epoch：",
    checkpoint[
        "epoch"
    ]
)


if int(
    checkpoint[
        "seed"
    ]
) != MODEL_SEED:

    raise RuntimeError(
        "命令行model_seed与checkpoint seed不一致"
    )


# ============================================================
# 10. 创建模型
# ============================================================

model = SASRec(
    num_items=
        NUM_ITEMS,

    max_len=
        MAX_LEN,

    hidden_size=
        HIDDEN_SIZE,

    num_heads=
        NUM_HEADS,

    num_blocks=
        NUM_BLOCKS,

    dropout=
        DROPOUT
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ],
    strict=True
)


model = model.to(
    device
)


model.eval()


print(
    "模型加载成功。"
)


# ============================================================
# 11. 加载item映射
# ============================================================

item_map_df = pd.read_csv(
    ITEM_MAP_PATH,
    dtype={
        "raw_item_id": str
    }
)


item_map_df[
    "raw_item_id"
] = (
    item_map_df[
        "raw_item_id"
    ]
    .astype(str)
)


item_map = dict(
    zip(
        item_map_df[
            "raw_item_id"
        ],
        item_map_df[
            "item_idx"
        ].astype(int)
    )
)


if len(
    item_map
) != NUM_ITEMS:

    raise RuntimeError(
        "item map数量与checkpoint NUM_ITEMS不一致："
        f"{len(item_map)} vs {NUM_ITEMS}"
    )


if max(
    item_map.values()
) != NUM_ITEMS:

    raise RuntimeError(
        "item map最大idx与NUM_ITEMS不一致"
    )


print(
    "物品映射数：",
    len(
        item_map
    )
)


# ============================================================
# 12. JSON历史序列 -> item_idx
# ============================================================

def parse_sequence(
    raw_sequence
):

    raw_items = json.loads(
        raw_sequence
    )

    mapped_items = []

    for raw_item in raw_items:

        raw_item = str(
            raw_item
        )

        if raw_item not in item_map:

            raise RuntimeError(
                f"历史物品不在item map中："
                f"{raw_item}"
            )

        mapped_items.append(
            int(
                item_map[
                    raw_item
                ]
            )
        )


    if len(
        mapped_items
    ) > MAX_LEN:

        raise RuntimeError(
            "第27.1步历史超过MAX_LEN，"
            "说明协议发生变化"
        )


    if len(
        mapped_items
    ) == 0:

        raise RuntimeError(
            "发现空历史"
        )


    # --------------------------------------------------------
    # 左侧padding，只作为统一存储格式
    # 实际模型推理时会去掉padding
    # --------------------------------------------------------

    seq = np.zeros(
        MAX_LEN,
        dtype=np.int64
    )


    seq[
        -len(
            mapped_items
        ):
    ] = mapped_items


    return seq


# ============================================================
# 13. 按有效长度做无padding推理
# ============================================================

@torch.no_grad()
def predict_batch_without_padding(
    seq_np
):

    current_batch_size = (
        len(
            seq_np
        )
    )


    valid_lengths = (
        seq_np != 0
    ).sum(
        axis=1
    )


    if (
        valid_lengths <= 0
    ).any():

        raise RuntimeError(
            "batch中发现有效历史长度为0"
        )


    if (
        valid_lengths > MAX_LEN
    ).any():

        raise RuntimeError(
            "batch中历史长度超过MAX_LEN"
        )


    scores = torch.empty(
        (
            current_batch_size,
            NUM_ITEMS + 1
        ),
        device=device,
        dtype=torch.float32
    )


    unique_lengths = np.unique(
        valid_lengths
    )


    for valid_len in (
        unique_lengths
    ):

        valid_len = int(
            valid_len
        )


        group_indices_np = np.where(
            valid_lengths
            ==
            valid_len
        )[0]


        group_indices = torch.as_tensor(
            group_indices_np,
            device=device,
            dtype=torch.long
        )


        # ----------------------------------------------------
        # 取最后valid_len个真实item
        # ----------------------------------------------------

        seq_group_np = (
            seq_np[
                group_indices_np,
                -valid_len:
            ]
        )


        # 理论上这里不应再有0
        if (
            seq_group_np == 0
        ).any():

            raise RuntimeError(
                "去padding后仍然出现item 0"
            )


        seq_group = torch.from_numpy(
            seq_group_np
        ).long().to(
            device
        )


        # ----------------------------------------------------
        # 保留原来的右对齐位置
        # ----------------------------------------------------

        position_offset = (
            MAX_LEN
            -
            valid_len
        )


        group_scores = (
            predict_all_unpadded(
                model,
                seq_group,
                position_offset
            )
        )


        # ----------------------------------------------------
        # 强制finite检查
        # ----------------------------------------------------

        if not torch.isfinite(
            group_scores
        ).all():

            nan_count = int(
                torch.isnan(
                    group_scores
                )
                .sum()
                .item()
            )

            inf_count = int(
                torch.isinf(
                    group_scores
                )
                .sum()
                .item()
            )

            raise RuntimeError(
                "无padding推理仍出现异常数值："
                f"valid_len={valid_len}, "
                f"NaN={nan_count}, "
                f"Inf={inf_count}"
            )


        scores[
            group_indices
        ] = group_scores


    # --------------------------------------------------------
    # 整个batch再次检查
    # --------------------------------------------------------

    if not torch.isfinite(
        scores
    ).all():

        raise RuntimeError(
            "最终batch scores存在NaN或Inf"
        )


    return scores


# ============================================================
# 14. 单个排列评价
# ============================================================

@torch.no_grad()
def evaluate_permutation(
    input_path,
    permutation_seed,
    split_name
):

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"{split_name.upper()} "
        f"/ permutation seed="
        f"{permutation_seed}"
    )

    print(
        "-" * 80
    )


    df = pd.read_csv(
        input_path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
        }
    )


    # --------------------------------------------------------
    # 样本数必须与第26步一致
    # --------------------------------------------------------

    expected_count = (
        EXPECTED_SAMPLES[
            DATASET
        ][
            split_name
        ]
    )


    if len(
        df
    ) != expected_count:

        raise RuntimeError(
            f"{DATASET}/{split_name} "
            f"样本数错误："
            f"expected={expected_count}, "
            f"actual={len(df)}"
        )


    print(
        "样本数：",
        len(
            df
        )
    )


    # ========================================================
    # target映射
    # ========================================================

    target_indices = []


    for raw_target in (
        df[
            "target_item"
        ].astype(str)
    ):

        if raw_target not in item_map:

            raise RuntimeError(
                f"target不在item map中："
                f"{raw_target}"
            )


        target_indices.append(
            int(
                item_map[
                    raw_target
                ]
            )
        )


    target_indices = np.asarray(
        target_indices,
        dtype=np.int64
    )


    # ========================================================
    # sequence
    # ========================================================

    sequences = np.stack(
        [
            parse_sequence(
                x
            )

            for x in (
                df[
                    "sequence_items"
                ].tolist()
            )
        ]
    )


    # ========================================================
    # 检查target不能在history
    # ========================================================

    target_in_history_count = 0


    for i in range(
        len(
            df
        )
    ):

        if target_indices[
            i
        ] in set(
            sequences[
                i
            ].tolist()
        ):

            target_in_history_count += 1


    print(
        "target_in_history：",
        target_in_history_count
    )


    if target_in_history_count != 0:

        raise RuntimeError(
            "发现target泄漏进history"
        )


    # ========================================================
    # 统计history长度
    # ========================================================

    valid_lengths = (
        sequences != 0
    ).sum(
        axis=1
    )


    print(
        "history长度范围：",
        int(
            valid_lengths.min()
        ),
        "~",
        int(
            valid_lengths.max()
        )
    )


    print(
        "history<50：",
        int(
            (
                valid_lengths
                <
                MAX_LEN
            ).sum()
        )
    )


    print(
        "history=50：",
        int(
            (
                valid_lengths
                ==
                MAX_LEN
            ).sum()
        )
    )


    # ========================================================
    # 输出容器
    # ========================================================

    target_ranks = []
    hit10_list = []
    hit20_list = []
    ndcg10_list = []
    top20_list = []


    start_time = time.time()


    # ========================================================
    # batch推理
    # ========================================================

    for start in range(
        0,
        len(
            df
        ),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(
                df
            )
        )


        seq_np = sequences[
            start:end
        ]


        target_np = target_indices[
            start:end
        ]


        # ----------------------------------------------------
        # 修正版无padding推理
        # ----------------------------------------------------

        scores = (
            predict_batch_without_padding(
                seq_np
            )
        )


        # padding item不能推荐
        scores[
            :,
            0
        ] = -1e9


        # ----------------------------------------------------
        # 屏蔽history中已经出现过的item
        #
        # target本身理论上不在history，
        # 但仍保留原评价逻辑
        # ----------------------------------------------------

        for b in range(
            len(
                seq_np
            )
        ):

            seen_items = np.unique(
                seq_np[
                    b
                ]
            )


            seen_items = (
                seen_items[
                    (
                        seen_items
                        !=
                        0
                    )
                    &
                    (
                        seen_items
                        !=
                        target_np[
                            b
                        ]
                    )
                ]
            )


            if len(
                seen_items
            ) > 0:

                seen_tensor = (
                    torch.as_tensor(
                        seen_items,
                        device=device,
                        dtype=torch.long
                    )
                )


                scores[
                    b,
                    seen_tensor
                ] = -1e9


        # ----------------------------------------------------
        # mask后仍然必须finite
        #
        # -1e9是finite
        # ----------------------------------------------------

        if not torch.isfinite(
            scores
        ).all():

            raise RuntimeError(
                "seen-item masking后scores存在"
                "NaN或Inf"
            )


        # ----------------------------------------------------
        # Top-20
        # ----------------------------------------------------

        top20 = torch.topk(
            scores,
            k=20,
            dim=1
        ).indices


        # ----------------------------------------------------
        # target score
        # ----------------------------------------------------

        target_tensor = (
            torch.from_numpy(
                target_np
            )
            .long()
            .to(
                device
            )
        )


        row_indices = torch.arange(
            len(
                target_tensor
            ),
            device=device
        )


        target_scores = scores[
            row_indices,
            target_tensor
        ]


        if not torch.isfinite(
            target_scores
        ).all():

            raise RuntimeError(
                "target score出现NaN或Inf"
            )


        # ----------------------------------------------------
        # 全量排名
        #
        # rank=1 + 比target分数严格更高的物品数量
        # ----------------------------------------------------

        ranks = (
            (
                scores
                >
                target_scores.unsqueeze(
                    1
                )
            )
            .sum(
                dim=1
            )
            +
            1
        )


        ranks_np = (
            ranks
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int64
            )
        )


        top20_np = (
            top20
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int64
            )
        )


        # ----------------------------------------------------
        # 基本rank检查
        # ----------------------------------------------------

        if (
            ranks_np < 1
        ).any():

            raise RuntimeError(
                "出现rank<1"
            )


        if (
            ranks_np
            >
            NUM_ITEMS
        ).any():

            raise RuntimeError(
                "出现rank>NUM_ITEMS"
            )


        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        for local_idx in range(
            len(
                ranks_np
            )
        ):

            rank = int(
                ranks_np[
                    local_idx
                ]
            )


            target_ranks.append(
                rank
            )


            hit10 = int(
                rank <= 10
            )


            hit20 = int(
                rank <= 20
            )


            if rank <= 10:

                ndcg10 = (
                    1.0
                    /
                    math.log2(
                        rank + 1
                    )
                )

            else:

                ndcg10 = 0.0


            hit10_list.append(
                hit10
            )


            hit20_list.append(
                hit20
            )


            ndcg10_list.append(
                ndcg10
            )


            top20_list.append(
                json.dumps(
                    top20_np[
                        local_idx
                    ].tolist()
                )
            )


    elapsed = (
        time.time()
        -
        start_time
    )


    # ========================================================
    # 逐样本结果
    # ========================================================

    pred_df = pd.DataFrame(
        {
            "sample_id":
                df[
                    "sample_id"
                ].astype(
                    str
                ),

            "user_id":
                df[
                    "user_id"
                ].astype(
                    str
                ),

            "target_item":
                df[
                    "target_item"
                ].astype(
                    str
                ),

            "target_item_idx":
                target_indices,

            "ambiguous_last_bucket":
                df[
                    "ambiguous_last_bucket"
                ].astype(
                    int
                ),

            "last_history_bucket_size":
                df[
                    "last_history_bucket_size"
                ].astype(
                    int
                ),

            "fixed_history_len":
                df[
                    "fixed_history_len"
                ].astype(
                    int
                ),

            "permutation_seed":
                permutation_seed,

            "target_rank":
                target_ranks,

            "hit10":
                hit10_list,

            "hit20":
                hit20_list,

            "ndcg10":
                ndcg10_list,

            "top20":
                top20_list,
        }
    )


    # ========================================================
    # 核心调试输出
    # ========================================================

    rank1_ratio = float(
        (
            pred_df[
                "target_rank"
            ]
            ==
            1
        ).mean()
    )


    print(
        "Rank=1比例：",
        f"{rank1_ratio * 100:.4f}%"
    )


    print(
        "Rank中位数：",
        f"{pred_df['target_rank'].median():.2f}"
    )


    print(
        "Rank最小/最大：",
        int(
            pred_df[
                "target_rank"
            ].min()
        ),
        "/",
        int(
            pred_df[
                "target_rank"
            ].max()
        )
    )


    # ========================================================
    # 输出目录
    # ========================================================

    split_output_dir = (
        OUTPUT_ROOT
        / split_name
    )


    split_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 保存CSV
    # ========================================================

    csv_path = (
        split_output_dir
        / (
            f"per_sample_preds_"
            f"seed{permutation_seed}.csv"
        )
    )


    pred_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # 尝试保存Parquet
    # ========================================================

    parquet_path = (
        split_output_dir
        / (
            f"per_sample_preds_"
            f"seed{permutation_seed}.parquet"
        )
    )


    try:

        pred_df.to_parquet(
            parquet_path,
            index=False
        )

        parquet_saved = True

    except Exception:

        parquet_saved = False


    # ========================================================
    # 总体指标
    # ========================================================

    ambiguous_mask = (
        pred_df[
            "ambiguous_last_bucket"
        ]
        ==
        1
    )


    metrics = {

        "dataset":
            DATASET,

        "split":
            split_name,

        "model_seed":
            MODEL_SEED,

        "permutation_seed":
            permutation_seed,

        "samples":
            len(
                pred_df
            ),

        "ambiguous_last_bucket_samples":
            int(
                ambiguous_mask.sum()
            ),

        "rank1_ratio":
            rank1_ratio,

        "NDCG@10":
            float(
                pred_df[
                    "ndcg10"
                ].mean()
            ),

        "HR@10":
            float(
                pred_df[
                    "hit10"
                ].mean()
            ),

        "HR@20":
            float(
                pred_df[
                    "hit20"
                ].mean()
            ),

        "MeanRank":
            float(
                pred_df[
                    "target_rank"
                ].mean()
            ),

        "MedianRank":
            float(
                pred_df[
                    "target_rank"
                ].median()
            ),

        "Ambiguous_NDCG@10":
            float(
                pred_df.loc[
                    ambiguous_mask,
                    "ndcg10"
                ].mean()
            ),

        "Ambiguous_HR@10":
            float(
                pred_df.loc[
                    ambiguous_mask,
                    "hit10"
                ].mean()
            ),

        "Ambiguous_HR@20":
            float(
                pred_df.loc[
                    ambiguous_mask,
                    "hit20"
                ].mean()
            ),

        "Ambiguous_MeanRank":
            float(
                pred_df.loc[
                    ambiguous_mask,
                    "target_rank"
                ].mean()
            ),

        "elapsed_seconds":
            float(
                elapsed
            ),

        "csv":
            str(
                csv_path
            ),

        "parquet_saved":
            parquet_saved,
    }


    # ========================================================
    # 保存metrics
    # ========================================================

    metrics_path = (
        split_output_dir
        / (
            f"metrics_"
            f"seed{permutation_seed}.json"
        )
    )


    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2
        )


    # ========================================================
    # 打印
    # ========================================================

    print(
        f"NDCG@10："
        f"{metrics['NDCG@10']:.6f}"
    )


    print(
        f"HR@10："
        f"{metrics['HR@10']:.6f}"
    )


    print(
        f"HR@20："
        f"{metrics['HR@20']:.6f}"
    )


    print(
        f"MeanRank："
        f"{metrics['MeanRank']:.2f}"
    )


    print(
        f"歧义子群 NDCG@10："
        f"{metrics['Ambiguous_NDCG@10']:.6f}"
    )


    print(
        f"歧义子群 MeanRank："
        f"{metrics['Ambiguous_MeanRank']:.2f}"
    )


    print(
        f"耗时："
        f"{elapsed:.2f} 秒"
    )


    return metrics


# ============================================================
# 15. 一个split跑5种排列
# ============================================================

def process_split(
    split_name
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"开始处理 {split_name.upper()}"
    )

    print(
        "=" * 80
    )


    split_input_dir = (
        INPUT_ROOT
        / split_name
    )


    results = []


    expected_sample_ids = None
    expected_target_items = None


    for permutation_seed in (
        PERMUTATION_SEEDS
    ):

        input_path = (
            split_input_dir
            / (
                f"permutation_seed"
                f"{permutation_seed}.csv"
            )
        )


        if not input_path.exists():

            raise FileNotFoundError(
                input_path
            )


        # ----------------------------------------------------
        # 5个排列必须完全相同sample/target
        # ----------------------------------------------------

        check_df = pd.read_csv(
            input_path,
            usecols=[
                "sample_id",
                "target_item",
            ],
            dtype={
                "sample_id": str,
                "target_item": str,
            }
        )


        sample_ids = (
            check_df[
                "sample_id"
            ].tolist()
        )


        target_items = (
            check_df[
                "target_item"
            ]
            .astype(str)
            .tolist()
        )


        if expected_sample_ids is None:

            expected_sample_ids = (
                sample_ids
            )

            expected_target_items = (
                target_items
            )


        else:

            if (
                sample_ids
                !=
                expected_sample_ids
            ):

                raise RuntimeError(
                    f"{split_name} "
                    "5种排列sample_id不一致"
                )


            if (
                target_items
                !=
                expected_target_items
            ):

                raise RuntimeError(
                    f"{split_name} "
                    "5种排列target不一致"
                )


        result = evaluate_permutation(
            input_path=
                input_path,

            permutation_seed=
                permutation_seed,

            split_name=
                split_name
        )


        results.append(
            result
        )


    # ========================================================
    # 汇总
    # ========================================================

    result_df = pd.DataFrame(
        results
    )


    split_output_dir = (
        OUTPUT_ROOT
        / split_name
    )


    summary_path = (
        split_output_dir
        / "metrics_summary.csv"
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
        f"{split_name.upper()} "
        "5种排列总体结果"
    )

    print(
        "=" * 80
    )


    print(
        result_df[
            [
                "permutation_seed",
                "samples",
                "rank1_ratio",
                "NDCG@10",
                "HR@10",
                "HR@20",
                "MeanRank",
                "Ambiguous_NDCG@10",
                "Ambiguous_MeanRank",
            ]
        ].to_string(
            index=False
        )
    )


    ndcg_min = float(
        result_df[
            "NDCG@10"
        ].min()
    )


    ndcg_max = float(
        result_df[
            "NDCG@10"
        ].max()
    )


    ndcg_range = (
        ndcg_max
        -
        ndcg_min
    )


    print(
        "\nNDCG@10范围：",
        f"{ndcg_min:.6f}",
        "~",
        f"{ndcg_max:.6f}"
    )


    print(
        "NDCG@10绝对range：",
        f"{ndcg_range:.6f}"
    )


    return results


# ============================================================
# 16. 主程序
# ============================================================

def main():

    all_results = []


    if args.split in [
        "val",
        "both",
    ]:

        all_results.extend(
            process_split(
                "val"
            )
        )


    if args.split in [
        "test",
        "both",
    ]:

        all_results.extend(
            process_split(
                "test"
            )
        )


    # ========================================================
    # 保存配置
    # ========================================================

    config = {

        "step":
            "27.2",

        "version":
            "fixed_unpadded_inference",

        "dataset":
            DATASET,

        "model_seed":
            MODEL_SEED,

        "checkpoint":
            str(
                CHECKPOINT_PATH
            ),

        "permutation_seeds":
            PERMUTATION_SEEDS,

        "evaluation":
            "full item ranking",

        "strict_protocol":
            (
                "target_bucket > "
                "last_history_bucket"
            ),

        "history_protocol":
            (
                "fixed history set "
                "before permutation"
            ),

        "padding_fix":
            (
                "remove left padding during inference "
                "while preserving right-aligned "
                "absolute positional indices"
            ),

        "seen_item_masking":
            True,

        "target_not_masked":
            True,

        "max_len":
            MAX_LEN,

        "batch_size":
            BATCH_SIZE,
    }


    config_path = (
        OUTPUT_ROOT
        / "config.json"
    )


    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2
        )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "第27.2步完成"
    )

    print(
        "=" * 80
    )


    print(
        "输出目录：",
        OUTPUT_ROOT
    )


if __name__ == "__main__":

    main()