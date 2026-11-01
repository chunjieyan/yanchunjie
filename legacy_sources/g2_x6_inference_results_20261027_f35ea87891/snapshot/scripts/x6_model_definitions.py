# Frozen definitions only. No archived training/evaluation top-level execution.
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
MAX_LEN = 50

class SASRecControl(nn.Module):
    def __init__(
        self,
        num_items,
        max_len,
        hidden_size,
        num_heads,
        num_blocks,
        dropout,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len
        self.hidden_size = hidden_size

        self.item_embedding = nn.Embedding(
            num_items + 1,
            hidden_size,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            max_len,
            hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_blocks,
        )

        self.final_norm = nn.LayerNorm(hidden_size)

    def build_token_embeddings(
        self,
        seq,
        group_positions,
    ):
        """原始事件 embedding -> group/token 的 mean embedding。"""

        batch_size = seq.size(0)

        item_embeddings = (
            self.item_embedding(seq)
            * math.sqrt(self.hidden_size)
        )

        valid_event_mask = (
            group_positions >= 0
        )

        safe_positions = (
            group_positions
            .clamp(min=0)
            .long()
        )

        # ====================================================
        # Deterministic group mean pooling
        #
        # 不使用 CUDA scatter_add_。
        # 每个 event 通过 one-hot assignment 映射到 group，
        # 再使用固定矩阵乘法完成 mean pooling。
        # ====================================================

        assignment = F.one_hot(
            safe_positions,
            num_classes=self.max_len,
        ).to(
            dtype=item_embeddings.dtype,
            device=item_embeddings.device,
        )

        assignment = (
            assignment
            *
            valid_event_mask
            .unsqueeze(-1)
            .to(dtype=item_embeddings.dtype)
        )

        # [B, group]
        token_count = (
            assignment
            .sum(dim=1)
            .clamp(min=1.0)
        )

        # [B, event, group]
        weights = (
            assignment
            /
            token_count.unsqueeze(1)
        )

        # [B, group, event] @ [B, event, hidden]
        # -> [B, group, hidden]
        token_embeddings = torch.bmm(
            weights.transpose(1, 2),
            item_embeddings,
        )

        return token_embeddings

    def predict_group_unpadded(
        self,
        seq,
        group_positions,
        num_tokens,
    ):
        """
        输入这一组样本的 num_tokens 必须完全相同。
        Transformer 不接收 padding；位置编码右对齐。
        """

        if seq.size(0) == 0:
            raise ValueError("empty batch")

        num_tokens = int(num_tokens)

        if num_tokens <= 0 or num_tokens > self.max_len:
            raise ValueError(
                f"invalid num_tokens={num_tokens}"
            )

        # Step38 Control fast path:
        # 每个历史事件对应唯一 token，不需要 One-hot + BMM。
        # 仅适用于 Control，不适用于 LBSP / R1 pooling。
        x = (
            self.item_embedding(seq[:, -num_tokens:])
            * math.sqrt(self.hidden_size)
        )

        position_offset = self.max_len - num_tokens

        positions = (
            torch.arange(
                position_offset,
                self.max_len,
                device=seq.device,
            )
            .unsqueeze(0)
            .expand(seq.size(0), num_tokens)
        )

        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(
                num_tokens,
                num_tokens,
                device=seq.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        x = self.encoder(
            x,
            mask=causal_mask,
        )

        x = self.final_norm(x)
        final_feature = x[:, -1, :]

        scores = torch.matmul(
            final_feature,
            self.item_embedding.weight.T,
        )

        return scores


    def predict_batch_masked(self, seq, num_tokens):
        """Control: full-batch masked forward, BS unchanged."""
        B, L = seq.shape

        lengths = num_tokens.long()
        pos = torch.arange(L, device=seq.device)
        offsets = L - lengths

        index = (
            pos.unsqueeze(0) + offsets.unsqueeze(1)
        ) % L

        left_seq = seq.gather(1, index)

        padding_mask = (
            pos.unsqueeze(0) >= lengths.unsqueeze(1)
        )

        absolute_pos = (
            offsets.unsqueeze(1) + pos.unsqueeze(0)
        ).clamp(max=L - 1)

        x = (
            self.item_embedding(left_seq)
            * math.sqrt(self.hidden_size)
        )

        x = x + self.position_embedding(absolute_pos)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(
                L, L,
                device=seq.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        x = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        final_feature = self.final_norm(
            x[
                torch.arange(B, device=seq.device),
                lengths - 1,
            ]
        )

        return (
            final_feature
            @ self.item_embedding.weight.T
        )

def build_r1_group_positions_from_size(
    history_len,
    last_bucket_size,
    max_len=MAX_LEN,
):
    if history_len <= 0:
        raise ValueError("history_len must be > 0")

    if last_bucket_size < 1 or last_bucket_size > history_len:
        raise ValueError(
            f"invalid last_bucket_size={last_bucket_size}, "
            f"history_len={history_len}"
        )

    group_positions = np.full(
        max_len,
        -1,
        dtype=np.int64,
    )

    direct_pooling = int(last_bucket_size >= 2)

    if direct_pooling:
        prefix_len = history_len - last_bucket_size
        event_groups = np.empty(history_len, dtype=np.int64)

        for j in range(prefix_len):
            event_groups[j] = j

        event_groups[prefix_len:] = prefix_len
        num_tokens = prefix_len + 1
    else:
        event_groups = np.arange(history_len, dtype=np.int64)
        num_tokens = history_len

    group_positions[-history_len:] = event_groups

    return (
        group_positions,
        int(num_tokens),
        int(direct_pooling),
    )

def canonicalize_last_bucket_items(
    history_items,
    last_bucket_size,
):

    items = np.asarray(
        history_items,
        dtype=np.int64,
    ).copy()

    if len(items) == 0:
        raise ValueError(
            "canonicalize出现空history"
        )

    if (
        last_bucket_size < 1
        or
        last_bucket_size > len(items)
    ):
        raise ValueError(
            f"invalid last_bucket_size="
            f"{last_bucket_size}, "
            f"history_len={len(items)}"
        )

    if last_bucket_size >= 2:

        start = (
            len(items)
            -
            last_bucket_size
        )

        items[
            start:
        ] = np.sort(
            items[
                start:
            ],
            kind="mergesort",
        )

    return items

class RecentBucketSetR1SASRec(nn.Module):
    def __init__(
        self,
        num_items,
        max_len,
        hidden_size,
        num_heads,
        num_blocks,
        dropout,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len
        self.hidden_size = hidden_size

        self.item_embedding = nn.Embedding(
            num_items + 1,
            hidden_size,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            max_len,
            hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_blocks,
        )

        self.final_norm = nn.LayerNorm(hidden_size)

    def build_token_embeddings(
        self,
        seq,
        group_positions,
    ):
        """原始事件 embedding -> group/token 的 mean embedding。"""

        batch_size = seq.size(0)

        item_embeddings = (
            self.item_embedding(seq)
            * math.sqrt(self.hidden_size)
        )

        valid_event_mask = (
            group_positions >= 0
        )

        safe_positions = (
            group_positions
            .clamp(min=0)
            .long()
        )

        # ====================================================
        # Deterministic group mean pooling
        #
        # 不使用 CUDA scatter_add_。
        # 每个 event 通过 one-hot assignment 映射到 group，
        # 再使用固定矩阵乘法完成 mean pooling。
        # ====================================================

        assignment = F.one_hot(
            safe_positions,
            num_classes=self.max_len,
        ).to(
            dtype=item_embeddings.dtype,
            device=item_embeddings.device,
        )

        assignment = (
            assignment
            *
            valid_event_mask
            .unsqueeze(-1)
            .to(dtype=item_embeddings.dtype)
        )

        # [B, group]
        token_count = (
            assignment
            .sum(dim=1)
            .clamp(min=1.0)
        )

        # [B, event, group]
        weights = (
            assignment
            /
            token_count.unsqueeze(1)
        )

        # [B, group, event] @ [B, event, hidden]
        # -> [B, group, hidden]
        token_embeddings = torch.bmm(
            weights.transpose(1, 2),
            item_embeddings,
        )

        return token_embeddings

    def predict_group_unpadded(
        self,
        seq,
        group_positions,
        num_tokens,
    ):
        """
        输入这一组样本的 num_tokens 必须完全相同。
        Transformer 不接收 padding；位置编码右对齐。
        """

        if seq.size(0) == 0:
            raise ValueError("empty batch")

        num_tokens = int(num_tokens)

        if num_tokens <= 0 or num_tokens > self.max_len:
            raise ValueError(
                f"invalid num_tokens={num_tokens}"
            )

        token_embeddings = self.build_token_embeddings(
            seq,
            group_positions,
        )

        x = token_embeddings[:, :num_tokens, :]

        position_offset = self.max_len - num_tokens

        positions = (
            torch.arange(
                position_offset,
                self.max_len,
                device=seq.device,
            )
            .unsqueeze(0)
            .expand(seq.size(0), num_tokens)
        )

        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(
                num_tokens,
                num_tokens,
                device=seq.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        x = self.encoder(
            x,
            mask=causal_mask,
        )

        x = self.final_norm(x)
        final_feature = x[:, -1, :]

        scores = torch.matmul(
            final_feature,
            self.item_embedding.weight.T,
        )

        return scores

def require(condition, msg):
    if not condition:
        raise RuntimeError(msg)

def predict_batch_masked(
    self,
    seq,
    group_positions,
    num_tokens,
):
    batch_size, width = seq.shape

    require(
        width == self.max_len,
        "输入序列长度异常",
    )

    pooled = self.build_token_embeddings(
        seq,
        group_positions,
    )

    positions = torch.arange(
        width,
        device=seq.device,
    ).unsqueeze(0).expand(batch_size, width)

    source_index = (
        positions
        - (width - num_tokens).unsqueeze(1)
    )

    valid = source_index >= 0

    gather_index = (
        source_index
        .clamp(min=0)
        .unsqueeze(-1)
        .expand(-1, -1, self.hidden_size)
    )

    x = torch.gather(
        pooled,
        dim=1,
        index=gather_index,
    )

    x = x * valid.unsqueeze(-1)

    x = x + self.position_embedding(positions)
    x = self.dropout(x)

    # ----------------------------------------------------
    # Safe masked causal attention for left-padded tokens.
    #
    # 有效 Query：
    #   只能关注真实 Token，且保持因果约束。
    #
    # Padding Query：
    #   只允许关注自身，避免整行全部被屏蔽产生 NaN。
    #
    # Padding Query 不能影响任何有效 Query。
    # ----------------------------------------------------

    causal_mask = torch.triu(
        torch.ones(
            width,
            width,
            dtype=torch.bool,
            device=seq.device,
        ),
        diagonal=1,
    )

    # [B, L, L]
    attention_mask = (
        causal_mask.unsqueeze(0)
        | (~valid).unsqueeze(1)
    )

    # 仅为 Padding Query 放开自身这一位置。
    pad_self = (
        (~valid).unsqueeze(-1)
        & torch.eye(
            width,
            dtype=torch.bool,
            device=seq.device,
        ).unsqueeze(0)
    )

    attention_mask = (
        attention_mask & (~pad_self)
    )

    # nn.MultiheadAttention 使用 [B * Heads, L, L]
    num_heads = (
        self.encoder.layers[0].self_attn.num_heads
    )

    attention_mask = (
        attention_mask.repeat_interleave(
            num_heads,
            dim=0,
        )
    )

    x = self.encoder(
        x,
        mask=attention_mask,
    )

    x = self.final_norm(x)

    final_feature = x[:, -1, :]

    return torch.matmul(
        final_feature,
        self.item_embedding.weight.T,
    )

RecentBucketSetR1SASRec.predict_batch_masked = predict_batch_masked
