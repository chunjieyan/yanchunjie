"""Unified sequential recommendation model zoo for KAIS Revision 2.0.

All models share the exact matched protocol:
- hidden_size = 64, num_heads = 2, num_layers = 2, dropout = 0.2, max_len = 50
- Full-catalog dot-product readout against item_embedding.weight (index 0 = pad)
- Seamless plug-in BucketSetPooler supporting:
  none, lbsp_mean, lbsp_max, lbsp_attn, lbsp_deepsets, absp_mean

Included architectures (8 distinct backbones across 7 paradigms):
1. SASRec      (Kang & McAuley, ICDM 2018)        — Causal Self-Attention (exact original match)
2. TiSASRec    (Li et al., WSDM 2020)             — Time-Interval Self-Attention (RecBole-TRM + LBSP/ABSP)
3. GRU4Rec     (Hidasi et al., ICLR 2016/CIKM 18) — 2-Layer Gated Recurrent Unit (RNN)
4. FMLP-Rec    (Zhou et al., WWW 2022)            — Filter-Enhanced All-MLP (FFT frequency denoising)
5. BSARec      (Shin et al., AAAI 2024)           — Attentive Frequency Inductive Bias + Self-Attention
6. LRURec      (Yue et al., WSDM 2024)            — Diagonal Complex Linear Recurrent Unit (parallel exact scan)
7. TiCoSeRec   (Dang et al., IEEE TKDE 2024)      — Time-Interval-Aware Contrastive Learning (SASRec + InfoNCE)
8. LSAB        (Han et al., ACM TKDD 2024)        — Learnable Self-Attention Bias Transformer
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .pooling import (
    BucketSetPooler,
    build_safe_causal_mask,
    right_align_tokens,
)


def _scaled_item_embeddings(
    item_embedding: nn.Embedding, seq: torch.Tensor, scale: bool = True
) -> torch.Tensor:
    emb = item_embedding(seq)
    if scale:
        emb = emb * math.sqrt(item_embedding.embedding_dim)
    return emb


# =============================================================================
# 1. SASRec (ICDM 2018) — exact architectural match to original SASRecControl / LBSP
# =============================================================================
class UnifiedSASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_size: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        pooling_mode: str = "none",
        extra_tokens: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pooling_mode = pooling_mode

        # Index 0 = pad, 1..num_items = catalog, optional extra_tokens (e.g. [MASK] for TiCoSeRec)
        self.item_embedding = nn.Embedding(
            num_items + 1 + extra_tokens, hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.pooler = BucketSetPooler(hidden_size, max_len, pooling_mode)

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
            layer, num_layers=num_blocks, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def encode(
        self,
        seq: torch.Tensor,
        group_positions: torch.Tensor,
        num_tokens: torch.Tensor,
        time_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = seq.shape
        raw_emb = _scaled_item_embeddings(self.item_embedding, seq, scale=True)
        if self.pooling_mode == "none":
            # Fast right-aligned path mathematically identical to pooled + right_align_tokens
            positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
            valid = positions >= (L - num_tokens).unsqueeze(1)
            x = raw_emb * valid.unsqueeze(-1).to(dtype=raw_emb.dtype)
        else:
            pooled = self.pooler(raw_emb, group_positions)
            x, valid = right_align_tokens(pooled, num_tokens)
            positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)

        x = x + self.position_embedding(positions)
        x = self.dropout(x)
        attn_mask = build_safe_causal_mask(valid, self.num_heads)
        x = self.encoder(x, mask=attn_mask)
        x = self.final_norm(x)
        return x[:, -1, :]

    def score_all(
        self,
        seq: torch.Tensor,
        group_positions: torch.Tensor,
        num_tokens: torch.Tensor,
        time_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.encode(seq, group_positions, num_tokens, time_matrix)
        return torch.matmul(feat, self.item_embedding.weight[: self.num_items + 1].T)


# =============================================================================
# 2. TiSASRec (WSDM 2020) — faithful RecBole-TRM adaptation + plug-in pooling
# =============================================================================
class _TiFeedForward(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int, dropout: float, eps: float = 1e-12):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dense_1(x)
        h = h * 0.5 * (1.0 + torch.erf(h / math.sqrt(2.0)))
        h = self.dropout(self.dense_2(h))
        return self.LayerNorm(h + x)


class _TimeAwareMultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, hidden_size: int, dropout: float, eps: float = 1e-12):
        super().__init__()
        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.all_head_size = hidden_size
        self.sqrt_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)
        self.out_dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.size()[:-1], self.num_attention_heads, self.attention_head_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        abs_pos_K: torch.Tensor,
        abs_pos_V: torch.Tensor,
        time_K: torch.Tensor,
        time_V: torch.Tensor,
    ) -> torch.Tensor:
        q = self._split(self.query(x)).permute(0, 2, 1, 3)
        k = self._split(self.key(x)).permute(0, 2, 3, 1)
        v = self._split(self.value(x)).permute(0, 2, 1, 3)

        pos_k = self._split(abs_pos_K).permute(0, 2, 3, 1)
        pos_v = self._split(abs_pos_V).permute(0, 2, 1, 3)
        tm_k = self._split(time_K).permute(0, 3, 1, 2, 4)  # [B, H, L, L, D_h]
        tm_v = self._split(time_V).permute(0, 3, 1, 2, 4)

        scores = torch.matmul(q, k) + torch.matmul(q, pos_k)
        scores = scores + torch.matmul(tm_k, q.unsqueeze(-1)).squeeze(-1)
        scores = scores / self.sqrt_head_size + attention_mask
        probs = self.attn_dropout(torch.softmax(scores, dim=-1))

        ctx = torch.matmul(probs, v) + torch.matmul(probs, pos_v)
        ctx = ctx + torch.matmul(probs.unsqueeze(-2), tm_v).squeeze(-2)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(*x.size())
        out = self.out_dropout(self.dense(ctx))
        return self.LayerNorm(out + x)


class _TiTransformerLayer(nn.Module):
    def __init__(self, n_heads: int, hidden_size: int, inner_size: int, dropout: float, eps: float = 1e-12):
        super().__init__()
        self.multi_head_attention = _TimeAwareMultiHeadAttention(n_heads, hidden_size, dropout, eps)
        self.feed_forward = _TiFeedForward(hidden_size, inner_size, dropout, eps)

    def forward(self, x, mask, abs_pos_K, abs_pos_V, time_K, time_V):
        return self.feed_forward(
            self.multi_head_attention(x, mask, abs_pos_K, abs_pos_V, time_K, time_V)
        )


class UnifiedTiSASRec(nn.Module):
    """RecBole-TRM TiSASRec extended with plug-in BucketSetPooler.

    Uses valid-first left-aligned token embeddings (matching pinned X4 TiSASRec)
    and compressed visible-prefix time_matrix [B, L, L].
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_size: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        pooling_mode: str = "none",
        time_span: int = 256,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.time_span = time_span
        self.pooling_mode = pooling_mode
        self.requires_time_matrix = True

        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.absolute_pos_K_embedding = nn.Embedding(max_len, hidden_size, padding_idx=0)
        self.absolute_pos_V_embedding = nn.Embedding(max_len, hidden_size, padding_idx=0)
        self.time_matrix_emb_K_embedding = nn.Embedding(time_span + 1, hidden_size, padding_idx=0)
        self.time_matrix_emb_V_embedding = nn.Embedding(time_span + 1, hidden_size, padding_idx=0)

        self.pooler = BucketSetPooler(hidden_size, max_len, pooling_mode)
        layer = _TiTransformerLayer(num_heads, hidden_size, hidden_size * 4, dropout, eps=1e-12)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_blocks)])
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(
        self,
        seq: torch.Tensor,
        group_positions: torch.Tensor,
        num_tokens: torch.Tensor,
        time_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if time_matrix is None:
            raise ValueError("UnifiedTiSASRec requires time_matrix [B, L, L]")
        B, L = seq.shape
        device = seq.device
        raw_emb = self.item_embedding(seq)  # RecBole-TRM does not scale by sqrt(d)
        # Pooler outputs left-aligned token embeddings [B, L, D], exactly matching TiSASRec valid-first!
        token_emb = self.pooler(raw_emb, group_positions)
        valid = torch.arange(L, device=device).unsqueeze(0) < num_tokens.unsqueeze(1)
        token_emb = token_emb * valid.unsqueeze(-1).to(dtype=token_emb.dtype)

        pos_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, L)
        abs_pos_K = self.dropout(self.absolute_pos_K_embedding(pos_ids))
        abs_pos_V = self.dropout(self.absolute_pos_V_embedding(pos_ids))
        time_K = self.dropout(self.time_matrix_emb_K_embedding(time_matrix))
        time_V = self.dropout(self.time_matrix_emb_V_embedding(time_matrix))

        x = self.dropout(self.LayerNorm(token_emb))

        # Causal + valid-key attention mask matching pinned RecBole-TRM:
        # [B, 1, L, L] with 0.0 for allowed, -10000.0 for masked.
        ext_mask = valid.unsqueeze(1).unsqueeze(2).expand(B, 1, L, L)
        ext_mask = torch.tril(ext_mask)
        # Prevent NaN on padded query rows (though never gathered, safe for backward)
        eye = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)
        pad_q = (~valid).unsqueeze(1).unsqueeze(-1)
        ext_mask = ext_mask | (pad_q & eye)
        attn_mask = torch.where(ext_mask, 0.0, -10000.0)

        for layer in self.layers:
            x = layer(x, attn_mask, abs_pos_K, abs_pos_V, time_K, time_V)

        gather_idx = (num_tokens - 1).view(B, 1, 1).expand(B, 1, self.hidden_size)
        return x.gather(dim=1, index=gather_idx).squeeze(1)

    def score_all(self, seq, group_positions, num_tokens, time_matrix=None) -> torch.Tensor:
        feat = self.encode(seq, group_positions, num_tokens, time_matrix)
        return torch.matmul(feat, self.item_embedding.weight.T)


# =============================================================================
# 3. GRU4Rec (ICLR 2016 / CIKM 2018) — Recurrent paradigm + plug-in pooling
# =============================================================================
class UnifiedGRU4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_size: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        pooling_mode: str = "none",
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.pooling_mode = pooling_mode

        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.pooler = BucketSetPooler(hidden_size, max_len, pooling_mode)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_blocks,
            batch_first=True,
            dropout=dropout if num_blocks > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)

    def encode(self, seq, group_positions, num_tokens, time_matrix=None) -> torch.Tensor:
        B, L = seq.shape
        raw_emb = self.item_embedding(seq)
        # Left-aligned pooled tokens [B, L, D]. Since nn.GRU is strictly forward-causal,
        # gathering output at (num_tokens - 1) is mathematically identical to pack_padded_sequence
        # while avoiding any per-batch num_tokens.cpu() GPU->CPU sync!
        pooled = self.pooler(raw_emb, group_positions)
        valid = torch.arange(L, device=seq.device).unsqueeze(0) < num_tokens.unsqueeze(1)
        pooled = self.emb_dropout(pooled) * valid.unsqueeze(-1).to(dtype=pooled.dtype)
        out, _ = self.gru(pooled)  # [B, L, D]
        last_h = out[torch.arange(B, device=seq.device), num_tokens - 1]
        feat = self.final_norm(self.out_proj(last_h))
        return feat

    def score_all(self, seq, group_positions, num_tokens, time_matrix=None) -> torch.Tensor:
        feat = self.encode(seq, group_positions, num_tokens, time_matrix)
        return torch.matmul(feat, self.item_embedding.weight.T)


# =============================================================================
# 4. FMLP-Rec (Zhou et al., WWW 2022) — Filter-Enhanced All-MLP (FFT Denoising)
# =============================================================================
class _FilterLayer(nn.Module):
    def __init__(self, max_len: int, hidden_size: int, dropout: float):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(1, max_len // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02
        )
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFT along sequence length dimension
        B, L, D = x.shape
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_filtered = x_fft * weight
        seq_out = torch.fft.irfft(x_filtered, n=L, dim=1, norm="ortho")
        seq_out = self.out_dropout(seq_out)
        return self.LayerNorm(seq_out + x)


class _FMLPLayer(nn.Module):
    def __init__(self, max_len: int, hidden_size: int, dropout: float):
        super().__init__()
        self.filter_layer = _FilterLayer(max_len, hidden_size, dropout)
        self.ffn = _TiFeedForward(hidden_size, hidden_size * 4, dropout, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.filter_layer(x))


class UnifiedFMLPRec(nn.Module):
    """FMLP-Rec (WWW 2022) with right-aligned sequence input and plug-in pooling."""

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_size: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        pooling_mode: str = "none",
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.pooling_mode = pooling_mode

        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.emb_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.pooler = BucketSetPooler(hidden_size, max_len, pooling_mode)
        self.layers = nn.ModuleList(
            [_FMLPLayer(max_len, hidden_size, dropout) for _ in range(num_blocks)]
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, seq, group_positions, num_tokens, time_matrix=None) -> torch.Tensor:
        B, L = seq.shape
        raw_emb = self.item_embedding(seq)
        pooled = self.pooler(raw_emb, group_positions)
        x, valid = right_align_tokens(pooled, num_tokens)
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = (x + self.position_embedding(positions)) * valid.unsqueeze(-1).to(dtype=x.dtype)
        x = self.dropout(self.emb_norm(x))
        for layer in self.layers:
            x = layer(x)
        return x[:, -1, :]

    def score_all(self, seq, group_positions, num_tokens, time_matrix=None) -> torch.Tensor:
        feat = self.encode(seq, group_positions, num_tokens, time_matrix)
        return torch.matmul(feat, self.item_embedding.weight.T)


# =============================================================================
# 5. BSARec (Shin et al., AAAI 2024) — Attentive Frequency Inductive Bias
# =============================================================================
class _FrequencyRescaleLayer(nn.Module):
    """Low-pass + learnable high-pass rescaler from BSARec (AAAI 2024):
    Rescale_c(x) = LFC_c(x) + sqrt_beta^2 * HFC_c(x)
    """

    def __init__(self, hidden_size: int, cutoff_c: int = 5, dropout: float = 0.2):
        super().__init__()
        self.cutoff_c = cutoff_c
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        fft = torch.fft.rfft(x, dim=1, norm="ortho")
        low_fft = fft.clone()
        low_fft[:, self.cutoff_c :, :] = 0
        low_pass = torch.fft.irfft(low_fft, n=L, dim=1, norm="ortho")
        high_pass = x - low_pass
        rescaled = low_pass + (self.sqrt_beta ** 2) * high_pass
        rescaled = self.out_dropout(rescaled)
        return self.LayerNorm(rescaled + x)


class _BSARecBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.2,
        alpha: float = 0.5,
        cutoff_c: int = 5,
    ):
        super().__init__()
        self.alpha = alpha
        self.freq_layer = _FrequencyRescaleLayer(hidden_size, cutoff_c=cutoff_c, dropout=dropout)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = _TiFeedForward(hidden_size, hidden_size * 4, dropout, eps=1e-12)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        dsp = self.freq_layer(x)
        attn_out, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        gsp = self.attn_norm(self.attn_dropout(attn_out) + x)
        mixed = self.alpha * dsp + (1.0 - self.alpha) * gsp
        return self.ffn(mixed)


class UnifiedBSARec(nn.Module):
    """BSARec (AAAI 2024) with safe causal masking and plug-in BucketSetPooler."""

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_size: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        pooling_mode: str = "none",
        alpha: float = 0.5,
        cutoff_c: int = 5,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pooling_mode = pooling_mode

        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.emb_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.pooler = BucketSetPooler(hidden_size, max_len, pooling_mode)
        self.layers = nn.ModuleList(
            [
                _BSARecBlock(hidden_size, num_heads, dropout=dropout, alpha=alpha, cutoff_c=cutoff_c)
                for _ in range(num_blocks)
            ]
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, seq, group_positions, num_tokens, time_matrix=None) -> torch.Tensor:
        B, L = seq.shape
        raw_emb = self.item_embedding(seq)
        pooled = self.pooler(raw_emb, group_positions)
        x, valid = right_align_tokens(pooled, num_tokens)
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = (x + self.position_embedding(positions)) * valid.unsqueeze(-1).to(dtype=x.dtype)
        x = self.dropout(self.emb_norm(x))
        attn_mask = build_safe_causal_mask(valid, self.num_heads)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x[:, -1, :]

    def score_all(self, seq, group_positions, num_tokens, time_matrix=None) -> torch.Tensor:
        feat = self.encode(seq, group_positions, num_tokens, time_matrix)
        return torch.matmul(feat, self.item_embedding.weight.T)


# =============================================================================
# 6. LRURec (Yue et al., WSDM 2024) — Parallel Exact Diagonal Complex LRU
# =============================================================================
class _LRULayer(nn.Module):
    """Exact diagonal complex Linear Recurrent Unit:
        h_t = Lambda * h_{t-1} + gamma * (B x_t)
        y_t = Re(C h_t) + D * x_t
    Evaluated in parallel via exact lower-triangular Vandermonde kernel M[t, s] = Lambda^(t-s).
    """

    def __init__(
        self,
        hidden_size: int,
        max_len: int = 50,
        dropout: float = 0.2,
        r_min: float = 0.8,
        r_max: float = 0.99,
        max_phase: float = math.pi / 10.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len

        # Initialize polar eigenvalues lambda = exp(-exp(nu_log) + i * exp(theta_log))
        u1 = torch.rand(hidden_size)
        u2 = torch.rand(hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(max_phase * u2 + 1e-6)
        diag_lambda = torch.exp(-torch.exp(nu_log) + 1j * torch.exp(theta_log))
        gamma_log = torch.log(torch.sqrt(1.0 - torch.abs(diag_lambda) ** 2 + 1e-8))

        self.params_log = nn.Parameter(torch.stack([nu_log, theta_log, gamma_log], dim=0))
        scale = 1.0 / math.sqrt(2.0 * hidden_size)
        self.in_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.out_proj_re = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj_im = nn.Linear(hidden_size, hidden_size, bias=False)
        self.skip_D = nn.Parameter(torch.randn(hidden_size) * 0.02 + 1.0)
        nn.init.normal_(self.in_proj.weight, std=scale)
        nn.init.normal_(self.out_proj_re.weight, std=scale)
        nn.init.normal_(self.out_proj_im.weight, std=scale)

        self.in_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Precompute lower-triangular lag matrix [L, L]: lag[t, s] = t - s if t >= s else 0
        idx = torch.arange(max_len)
        lag = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0)
        tril = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("_lag", lag, persistent=False)
        self.register_buffer("_tril", tril, persistent=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], valid: [B, L] (right-aligned, left-padded with zeros)
        B, L, D = x.shape
        x_norm = self.in_norm(x) * valid.unsqueeze(-1).to(dtype=x.dtype)

        nu = torch.exp(self.params_log[0])       # [D]
        theta = torch.exp(self.params_log[1])    # [D]
        gamma = torch.exp(self.params_log[2])    # [D]

        # Complex decay per step in polar form (pure real arithmetic -> 100% deterministic on CUDA!)
        # lambda^k = exp(-nu * k) * (cos(theta * k) + i * sin(theta * k))
        lag = self._lag[:L, :L].to(dtype=x.dtype)          # [L, L]
        tril = self._tril[:L, :L].to(dtype=x.dtype)        # [L, L]
        mag = torch.exp(-lag.unsqueeze(0) * nu.view(D, 1, 1)) * tril.unsqueeze(0)  # [D, L, L]
        ang = lag.unsqueeze(0) * theta.view(D, 1, 1)
        K_re = mag * torch.cos(ang)                         # [D, L, L]
        K_im = mag * torch.sin(ang)                         # [D, L, L]

        # Project input to complex space and scale by gamma:
        bx = self.in_proj(x_norm)                           # [B, L, 2D]
        u_re = bx[:, :, :D] * gamma.view(1, 1, D)
        u_im = bx[:, :, D:] * gamma.view(1, 1, D)

        # Parallel exact causal scan via batched matmul: [D, L, L] @ [B, L, D] -> [B, L, D]
        # Transpose u to [D, L, B], matmul with K [D, L, L], transpose back!
        u_re_t = u_re.permute(2, 1, 0)                      # [D, L, B]
        u_im_t = u_im.permute(2, 1, 0)                      # [D, L, B]
        h_re = (torch.bmm(K_re, u_re_t) - torch.bmm(K_im, u_im_t)).permute(2, 1, 0)
        h_im = (torch.bmm(K_re, u_im_t) + torch.bmm(K_im, u_re_t)).permute(2, 1, 0)

        # Real output y_t = Re(C h_t) + D * x_t
        y = self.out_proj_re(h_re) - self.out_proj_im(h_im) + x_norm * self.skip_D.view(1, 1, D)
        y = self.dropout(y) * valid.unsqueeze(-1).to(dtype=x.dtype)
        return x + y


class _LRUBlock(nn.Module):
    def __init__(self, hidden_size: int, max_len: int, dropout: float):
        super().__init__()
        self.lru = _LRULayer(hidden_size, max_len=max_len, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = self.lru(x, valid)
        ffn_out = self.ffn(self.ffn_norm(x)) * valid.unsqueeze(-1).to(dtype=x.dtype)
        return x + ffn_out


class UnifiedLRURec(nn.Module):
    """LRURec (WSDM 2024) with plug-in BucketSetPooler and exact zero-pad recurrence."""

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_size: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        pooling_mode: str = "none",
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.pooling_mode = pooling_mode

        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.emb_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.pooler = BucketSetPooler(hidden_size, max_len, pooling_mode)
        self.layers = nn.ModuleList(
            [_LRUBlock(hidden_size, max_len=max_len, dropout=dropout) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def encode(self, seq, group_positions, num_tokens, time_matrix=None) -> torch.Tensor:
        raw_emb = self.item_embedding(seq)
        pooled = self.pooler(raw_emb, group_positions)
        x, valid = right_align_tokens(pooled, num_tokens)
        x = self.dropout(self.emb_norm(x)) * valid.unsqueeze(-1).to(dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, valid)
        x = self.final_norm(x)
        return x[:, -1, :]

    def score_all(self, seq, group_positions, num_tokens, time_matrix=None) -> torch.Tensor:
        feat = self.encode(seq, group_positions, num_tokens, time_matrix)
        return torch.matmul(feat, self.item_embedding.weight.T)


# =============================================================================
# 7. LSAB (Han et al., ACM TKDD 2024) — Learnable Self-Attention Bias
# =============================================================================
class _LSABAttentionBlock(nn.Module):
    """Causal multi-head self-attention augmented with learnable behavioral bias
    matrix B_{h, i, j} (relative-distance + query-recency bias) as in LSAB (TKDD 2024).
    """

    def __init__(self, hidden_size: int, num_heads: int, max_len: int, dropout: float = 0.2):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.max_len = max_len

        self.norm1 = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Learnable relative-distance bias [H, max_len] + position-pair residual bias [H, max_len, max_len]
        self.rel_dist_bias = nn.Parameter(torch.zeros(num_heads, max_len))
        self.pair_bias = nn.Parameter(torch.zeros(num_heads, max_len, max_len))

        idx = torch.arange(max_len)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0)  # [L, L]
        self.register_buffer("_dist", dist, persistent=False)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask_bll: torch.Tensor) -> torch.Tensor:
        # attn_mask_bll: [B, L, L] boolean where True = masked
        B, L, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, L, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # [B, H, L, Dh]
        k = qkv[:, :, 1].permute(0, 2, 3, 1)  # [B, H, Dh, L]
        v = qkv[:, :, 2].permute(0, 2, 1, 3)  # [B, H, L, Dh]

        scores = torch.matmul(q, k) / self.scale  # [B, H, L, L]
        # Add learned behavioral bias matrix B[H, L, L]
        dist_idx = self._dist[:L, :L]
        bias = self.rel_dist_bias[:, dist_idx] + self.pair_bias[:, :L, :L]  # [H, L, L]
        scores = scores + bias.unsqueeze(0)
        scores = scores.masked_fill(attn_mask_bll.unsqueeze(1), -1e9)
        probs = self.attn_dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        x = x + self.out_dropout(self.out_proj(ctx))
        x = x + self.ffn(self.norm2(x))
        return x


class UnifiedLSAB(nn.Module):
    """LSAB (ACM TKDD 2024) with plug-in BucketSetPooler."""

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_size: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        pooling_mode: str = "none",
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pooling_mode = pooling_mode

        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.pooler = BucketSetPooler(hidden_size, max_len, pooling_mode)
        self.layers = nn.ModuleList(
            [_LSABAttentionBlock(hidden_size, num_heads, max_len, dropout) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def encode(self, seq, group_positions, num_tokens, time_matrix=None) -> torch.Tensor:
        B, L = seq.shape
        raw_emb = _scaled_item_embeddings(self.item_embedding, seq, scale=True)
        pooled = self.pooler(raw_emb, group_positions)
        x, valid = right_align_tokens(pooled, num_tokens)
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1)
        attn_mask = causal_mask.unsqueeze(0) | (~valid).unsqueeze(1)
        pad_self = (~valid).unsqueeze(-1) & torch.eye(L, dtype=torch.bool, device=seq.device).unsqueeze(0)
        attn_mask = attn_mask & (~pad_self)

        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.final_norm(x)
        return x[:, -1, :]

    def score_all(self, seq, group_positions, num_tokens, time_matrix=None) -> torch.Tensor:
        feat = self.encode(seq, group_positions, num_tokens, time_matrix)
        return torch.matmul(feat, self.item_embedding.weight.T)


# =============================================================================
# 8. TiCoSeRec (Dang et al., IEEE TKDE 2024) — Time-Interval Contrastive Learning
# =============================================================================
class UnifiedTiCoSeRec(UnifiedSASRec):
    """TiCoSeRec (IEEE TKDE 2024): SASRec backbone + [MASK] token + InfoNCE projection
    trained jointly with time-interval-aware augmentations (Ti-Reorder, Ti-Crop, Ti-Mask).
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_size: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        pooling_mode: str = "none",
        cl_weight: float = 0.1,
        cl_temp: float = 1.0,
    ):
        super().__init__(
            num_items=num_items,
            max_len=max_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout,
            pooling_mode=pooling_mode,
            extra_tokens=1,  # mask token index = num_items + 1
        )
        self.mask_token_id = num_items + 1
        self.cl_weight = cl_weight
        self.cl_temp = cl_temp
        self.is_contrastive = True

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Symmetric in-batch InfoNCE (NT-Xent) between two augmented views."""
        B = z1.size(0)
        if B < 2:
            return z1.new_tensor(0.0)
        h1 = F.normalize(z1, dim=-1)
        h2 = F.normalize(z2, dim=-1)
        z = torch.cat([h1, h2], dim=0)  # [2B, D]
        sim = torch.matmul(z, z.T) / self.cl_temp  # [2B, 2B]
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e9)
        targets = torch.cat(
            [torch.arange(B, 2 * B, device=z.device), torch.arange(0, B, device=z.device)], dim=0
        )
        return F.cross_entropy(sim, targets)


BACKBONE_REGISTRY = {
    "sasrec": UnifiedSASRec,
    "tisasrec": UnifiedTiSASRec,
    "gru4rec": UnifiedGRU4Rec,
    "fmlprec": UnifiedFMLPRec,
    "bsarec": UnifiedBSARec,
    "lrurec": UnifiedLRURec,
    "ticoserec": UnifiedTiCoSeRec,
    "lsab": UnifiedLSAB,
}


def build_model(
    backbone: str,
    num_items: int,
    pooling_mode: str = "none",
    max_len: int = 50,
    hidden_size: int = 64,
    num_heads: int = 2,
    num_blocks: int = 2,
    dropout: float = 0.2,
    **kwargs,
) -> nn.Module:
    key = backbone.lower()
    if key not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{backbone}'. Available: {sorted(BACKBONE_REGISTRY)}")
    cls = BACKBONE_REGISTRY[key]
    return cls(
        num_items=num_items,
        max_len=max_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
        pooling_mode=pooling_mode,
        **kwargs,
    )
