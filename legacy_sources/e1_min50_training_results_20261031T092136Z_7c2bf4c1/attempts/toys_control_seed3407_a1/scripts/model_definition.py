# Frozen class only; no original top-level training statements.
import math
import numpy as np
import torch
import torch.nn as nn
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
