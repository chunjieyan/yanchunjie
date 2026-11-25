"""Deterministic set-pooling front-end modules for timestamp-tie invariance.

Supported pooling modes:
- ``none``:          No pooling; every event maps to its own token.
- ``lbsp_mean``:     Last-Bucket Set Pooling with arithmetic mean (exact match to original LBSP).
- ``lbsp_max``:      Last-Bucket Set Pooling with element-wise maximum.
- ``lbsp_attn``:     Last-Bucket Set Pooling with permutation-invariant attention (PMA-style).
- ``lbsp_deepsets``: Last-Bucket Set Pooling with DeepSets phi/rho + log-cardinality encoding.
- ``absp_mean``:     All-Bucket Set Pooling (mean pooling applied to every tied historical bucket).

All pooling modes operate strictly BEFORE positional encoding or sequence encoders,
making them architecture-agnostic across SASRec, TiSASRec, GRU4Rec, FMLP-Rec,
BSARec, LRURec, TiCoSeRec, and LSAB.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

POOLING_MODES = (
    "none",
    "lbsp_mean",
    "lbsp_max",
    "lbsp_attn",
    "lbsp_deepsets",
    "absp_mean",
)


def is_lbsp_mode(pooling_mode: str) -> bool:
    return pooling_mode.startswith("lbsp_")


def is_absp_mode(pooling_mode: str) -> bool:
    return pooling_mode.startswith("absp_")


def canonicalize_and_group(
    items: np.ndarray,
    buckets: np.ndarray,
    pooling_mode: str,
    max_len: int = 50,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Build right-aligned padded ``seq``, ``group_positions``, ``num_tokens``, and
    left-aligned compressed ``token_buckets`` for a single visible history.

    Parameters
    ----------
    items : 1-D int64 array of length n (1 <= n <= max_len)
        Mapped item indices in chronological/reference order.
    buckets : 1-D int64 array of length n
        Native timestamp bucket indices (non-decreasing).
    pooling_mode : str
        One of :data:`POOLING_MODES`.
    max_len : int
        Maximum history window width (default 50).

    Returns
    -------
    seq : (max_len,) int64
        Right-aligned padded item indices (0 = pad). Canonicalized within pooled
        buckets to eliminate floating-point summation non-associativity.
    group_positions : (max_len,) int64
        Right-aligned event-to-group map in ``0 .. num_tokens - 1`` (-1 = pad).
    num_tokens : int
        Number of resulting compressed tokens (1 <= num_tokens <= n).
    token_buckets : (max_len,) int64
        Left-aligned native bucket timestamps for the compressed tokens
        (indices ``0 .. num_tokens - 1`` valid; trailing entries 0). Used by
        time-aware encoders such as TiSASRec + LBSP / ABSP.
    """
    n = int(len(items))
    if not 1 <= n <= max_len:
        raise ValueError(f"Visible history length must be in [1, {max_len}], got {n}")
    if len(buckets) != n:
        raise ValueError("items and buckets must have equal length")

    items_out = np.asarray(items, dtype=np.int64).copy()
    buckets_arr = np.asarray(buckets, dtype=np.int64)

    seq = np.zeros(max_len, dtype=np.int64)
    group_positions = np.full(max_len, -1, dtype=np.int64)
    token_buckets = np.zeros(max_len, dtype=np.int64)

    if pooling_mode == "none":
        event_groups = np.arange(n, dtype=np.int64)
        num_tokens = n
        token_buckets[:n] = buckets_arr

    elif is_lbsp_mode(pooling_mode):
        last_b = buckets_arr[-1]
        start = n - 1
        while start > 0 and buckets_arr[start - 1] == last_b:
            start -= 1
        last_k = n - start
        if last_k >= 2:
            # Canonicalize item index order inside the last bucket for exact FP determinism.
            items_out[start:n] = np.sort(items_out[start:n], kind="mergesort")
            event_groups = np.empty(n, dtype=np.int64)
            event_groups[:start] = np.arange(start, dtype=np.int64)
            event_groups[start:n] = start
            num_tokens = start + 1
            token_buckets[:start] = buckets_arr[:start]
            token_buckets[start] = last_b
        else:
            event_groups = np.arange(n, dtype=np.int64)
            num_tokens = n
            token_buckets[:n] = buckets_arr

    elif is_absp_mode(pooling_mode):
        event_groups = np.empty(n, dtype=np.int64)
        g_idx = 0
        pos = 0
        while pos < n:
            end = pos + 1
            while end < n and buckets_arr[end] == buckets_arr[pos]:
                end += 1
            if end - pos >= 2:
                items_out[pos:end] = np.sort(items_out[pos:end], kind="mergesort")
            event_groups[pos:end] = g_idx
            token_buckets[g_idx] = buckets_arr[pos]
            g_idx += 1
            pos = end
        num_tokens = g_idx

    else:
        raise ValueError(f"Unknown pooling_mode: {pooling_mode}")

    seq[-n:] = items_out
    group_positions[-n:] = event_groups
    return seq, group_positions, int(num_tokens), token_buckets


class BucketSetPooler(nn.Module):
    """Differentiable, deterministic set-pooling layer mapping raw event embeddings
    ``[B, L, D]`` to compressed token embeddings ``[B, L, D]`` (left-aligned by
    group index ``0 .. num_tokens - 1``).
    """

    def __init__(self, hidden_size: int, max_len: int, pooling_mode: str = "lbsp_mean"):
        super().__init__()
        if pooling_mode not in POOLING_MODES:
            raise ValueError(f"Unsupported pooling_mode={pooling_mode}")
        self.hidden_size = int(hidden_size)
        self.max_len = int(max_len)
        self.pooling_mode = pooling_mode

        if pooling_mode == "lbsp_attn":
            # Permutation-invariant additive attention pooling (PMA-style).
            self.attn_proj = nn.Linear(hidden_size, hidden_size)
            self.attn_query = nn.Parameter(torch.empty(hidden_size))
            nn.init.normal_(self.attn_query, mean=0.0, std=0.02)
            nn.init.normal_(self.attn_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.attn_proj.bias)

        elif pooling_mode == "lbsp_deepsets":
            # Residual DeepSets with explicit log-cardinality encoding:
            # z_g = mean(E) + rho(mean(phi(E)) + w_cnt * log(|B_g|))
            self.phi = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.rho = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.count_proj = nn.Linear(1, hidden_size)
            for mod in (self.phi, self.rho):
                for layer in mod:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                        nn.init.zeros_(layer.bias)
            nn.init.normal_(self.count_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.count_proj.bias)

    def _build_assignment(
        self, group_positions: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(assignment [B, E, G], valid_group_mask [B, G])``."""
        valid_event = group_positions >= 0
        safe_pos = group_positions.clamp(min=0).long()
        assignment = F.one_hot(safe_pos, num_classes=self.max_len).to(dtype=dtype)
        assignment = assignment * valid_event.unsqueeze(-1).to(dtype=dtype)
        counts = assignment.sum(dim=1)  # [B, G]
        return assignment, counts

    def forward(
        self, item_embeddings: torch.Tensor, group_positions: torch.Tensor
    ) -> torch.Tensor:
        """Pool ``item_embeddings [B, L, D]`` into left-aligned group tokens ``[B, L, D]``."""
        dtype = item_embeddings.dtype
        assignment, counts = self._build_assignment(group_positions, dtype)  # [B, E, G], [B, G]
        safe_counts = counts.clamp(min=1.0)
        mean_weights = assignment / safe_counts.unsqueeze(1)  # [B, E, G]

        if self.pooling_mode in ("none", "lbsp_mean", "absp_mean"):
            return torch.bmm(mean_weights.transpose(1, 2), item_embeddings)

        if self.pooling_mode == "lbsp_max":
            # Masked max pooling over events belonging to each group g:
            # membership: [B, G, E]
            membership = assignment.transpose(1, 2) > 0.5
            # [B, 1, E, D] -> [B, G, E, D]
            expanded = item_embeddings.unsqueeze(1).expand(-1, self.max_len, -1, -1)
            neg_inf = torch.tensor(-1e9, dtype=dtype, device=item_embeddings.device)
            masked = torch.where(membership.unsqueeze(-1), expanded, neg_inf)
            pooled = masked.max(dim=2).values  # [B, G, D]
            valid_group = (counts > 0.5).unsqueeze(-1).to(dtype=dtype)
            return pooled * valid_group

        if self.pooling_mode == "lbsp_attn":
            # Compute scalar logit per event: [B, E]
            logits = torch.matmul(
                torch.tanh(self.attn_proj(item_embeddings)),
                self.attn_query,
            ) / math.sqrt(self.hidden_size)
            # Softmax strictly within each group g:
            # membership [B, G, E]
            membership = assignment.transpose(1, 2) > 0.5
            group_logits = logits.unsqueeze(1).expand(-1, self.max_len, -1)
            group_logits = group_logits.masked_fill(~membership, -1e9)
            # For empty groups, mask row cleanly after softmax.
            attn_weights = torch.softmax(group_logits, dim=-1)
            valid_group = (counts > 0.5).unsqueeze(-1).to(dtype=dtype)
            attn_weights = attn_weights * valid_group
            return torch.bmm(attn_weights, item_embeddings)

        if self.pooling_mode == "lbsp_deepsets":
            base_mean = torch.bmm(mean_weights.transpose(1, 2), item_embeddings)
            phi_emb = self.phi(item_embeddings)
            phi_mean = torch.bmm(mean_weights.transpose(1, 2), phi_emb)
            # Explicit cardinality feature: log(count) is 0 for singletons, >0 for multi-event buckets.
            log_cnt = torch.log(safe_counts).unsqueeze(-1)  # [B, G, 1]
            delta = self.rho(phi_mean + self.count_proj(log_cnt))
            valid_group = (counts > 0.5).unsqueeze(-1).to(dtype=dtype)
            return (base_mean + delta) * valid_group

        raise ValueError(f"Unhandled pooling_mode={self.pooling_mode}")


def right_align_tokens(
    pooled_tokens: torch.Tensor, num_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert left-aligned pooled tokens ``[B, L, D]`` to right-aligned tokens
    ``[B, L, D]`` + boolean valid mask ``[B, L]`` (matching SASRecControl/LBSP).
    """
    B, L, D = pooled_tokens.shape
    positions = torch.arange(L, device=pooled_tokens.device).unsqueeze(0).expand(B, L)
    source_index = positions - (L - num_tokens).unsqueeze(1)
    valid = source_index >= 0
    gather_index = source_index.clamp(min=0).unsqueeze(-1).expand(-1, -1, D)
    aligned = torch.gather(pooled_tokens, dim=1, index=gather_index)
    aligned = aligned * valid.unsqueeze(-1).to(dtype=aligned.dtype)
    return aligned, valid


def build_safe_causal_mask(
    valid: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Safe causal attention mask for left-padded (right-aligned) sequences.

    Valid queries attend only to earlier/current valid keys; padding queries
    attend only to themselves so softmax never encounters an all-masked row.
    Returns boolean mask of shape ``[B * num_heads, L, L]`` where True = masked.
    """
    B, L = valid.shape
    device = valid.device
    causal_mask = torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
    )
    attn_mask = causal_mask.unsqueeze(0) | (~valid).unsqueeze(1)
    pad_self = (~valid).unsqueeze(-1) & torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
    attn_mask = attn_mask & (~pad_self)
    return attn_mask.repeat_interleave(num_heads, dim=0)
