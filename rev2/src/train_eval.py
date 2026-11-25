"""Unified GPU-resident trainer and multi-protocol evaluator for KAIS Revision 2.0.

Preserves the exact frozen KAIS protocol:
- Batch size 128, Adam(lr=0.001), grad clip 5.0, full-item CrossEntropyLoss
- Hidden size 64, 2 heads, 2 blocks, dropout 0.2, max_len 50
- Min epochs 50, patience 15 on validation NDCG@10, max epochs 200
- Restores best validation checkpoint (reproducing exact validation NDCG@10)
- Evaluates:
    1. Canonical test panel (complete, affected, warm, cold, warm-affected)
    2. Sealed Last-Bucket 5-Permutation panels (perm_seed101..505)
    3. Deterministic All-Bucket 5-Permutation panels (allbucket_seed101..505)
    4. Bucket-size stratified stability (k = 1, 2, 3, 4, 5+)
    5. Automatic exact-invariance assertion for LBSP / ABSP models!
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .models import build_model
from .pooling import (
    POOLING_MODES,
    canonicalize_and_group,
    is_absp_mode,
    is_lbsp_mode,
)
from .protocol import (
    MAX_LEN,
    PERMUTATION_SEEDS,
    FrozenDatasetBundle,
    bucket_stratified_stability,
    build_time_matrix_gpu,
    metrics_from_ranks,
    stability_summary,
)


def set_deterministic_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# GPU-Vectorized TiCoSeRec Time-Interval-Aware Augmentations (IEEE TKDE 2024)
# =============================================================================
def _ticoserec_augment_batch(
    seq: torch.Tensor,
    raw_buckets: torch.Tensor,
    raw_lengths: torch.Tensor,
    mask_token_id: int,
    pooling_mode: str,
    rng_gen: torch.Generator,
    crop_ratio: float = 0.6,
    mask_ratio: float = 0.2,
    reorder_ratio: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-GPU vectorized time-interval-aware augmentation (TiCoSeRec, IEEE TKDE 2024).

    Zero Python loops and zero GPU->CPU syncs. Randomly selects among:
      0 = Ti-Reorder (permute minimum time-span contiguous segment)
      1 = Ti-Crop    (keep minimum time-span contiguous crop)
      2 = Ti-Mask    (mask items adjacent to largest time-interval jumps)
    """
    B, L = seq.shape
    device = seq.device
    pos_idx = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
    w_grid = pos_idx.expand(B, L)                          # [B, L]
    starts = L - raw_lengths                               # [B]
    valid = w_grid >= starts.unsqueeze(1)                  # [B, L]

    ops = torch.randint(0, 3, (B,), generator=rng_gen, device=device)

    aug_seq = seq.clone()
    aug_buckets = raw_buckets.clone()
    aug_len = raw_lengths.clone()

    # --- Op 2: Ti-Mask (mask items adjacent to largest time gaps) ---
    mask_rows = (ops == 2) & (raw_lengths >= 2)
    dt = (raw_buckets[:, 1:] - raw_buckets[:, :-1]).abs().float()
    left_gap = F.pad(dt, (1, 0), value=0.0)
    right_gap = F.pad(dt, (0, 1), value=0.0)
    valid_left = F.pad(valid[:, :-1], (1, 0), value=False)
    valid_right = F.pad(valid[:, 1:], (0, 1), value=False)
    gap_score = torch.maximum(
        torch.where(valid_left, left_gap, torch.zeros_like(left_gap)),
        torch.where(valid_right, right_gap, torch.zeros_like(right_gap)),
    )
    noise_m = torch.rand((B, L), generator=rng_gen, device=device) * 1e-3
    priority = torch.where(valid, gap_score + noise_m, torch.full_like(gap_score, -1.0))
    order = priority.argsort(dim=1, descending=True)
    rank_in_row = torch.empty_like(order)
    rank_in_row.scatter_(1, order, w_grid)
    n_mask = (raw_lengths.float() * mask_ratio).floor().long().clamp(min=1)
    to_mask = mask_rows.unsqueeze(1) & (rank_in_row < n_mask.unsqueeze(1)) & valid
    aug_seq = torch.where(to_mask, torch.full_like(aug_seq, mask_token_id), aug_seq)

    # --- Op 1: Ti-Crop (minimum time-span contiguous crop, fully vectorized) ---
    crop_rows = (ops == 1) & (raw_lengths >= 3)
    c_len = (raw_lengths.float() * crop_ratio).floor().long().clamp(min=1)
    c_end = (w_grid + c_len.unsqueeze(1) - 1).clamp(max=L - 1)
    c_valid = (w_grid >= starts.unsqueeze(1)) & (w_grid + c_len.unsqueeze(1) <= L)
    c_spans = (raw_buckets.gather(1, c_end) - raw_buckets.gather(1, w_grid)).abs().float()
    c_jitter = torch.rand((B, L), generator=rng_gen, device=device) * 1e-4
    c_score = torch.where(c_valid, c_spans + c_jitter, torch.full_like(c_spans, 1e30))
    c_best_w = c_score.argmin(dim=1)  # [B]

    crop_start_dst = L - c_len  # [B]
    in_crop = w_grid >= crop_start_dst.unsqueeze(1)
    src_idx = (c_best_w.unsqueeze(1) + (w_grid - crop_start_dst.unsqueeze(1))).clamp(0, L - 1)
    cropped_seq = torch.where(in_crop, seq.gather(1, src_idx), torch.zeros_like(seq))
    cropped_bs = torch.where(in_crop, raw_buckets.gather(1, src_idx), torch.zeros_like(raw_buckets))

    crop_mask_2d = crop_rows.unsqueeze(1)
    aug_seq = torch.where(crop_mask_2d, cropped_seq, aug_seq)
    aug_buckets = torch.where(crop_mask_2d, cropped_bs, aug_buckets)
    aug_len = torch.where(crop_rows, c_len, aug_len)

    # --- Op 0: Ti-Reorder (uniform random permutation of min time-span segment, vectorized) ---
    reord_rows = (ops == 0) & (raw_lengths >= 2)
    r_len = torch.minimum(
        (raw_lengths.float() * reorder_ratio).floor().long().clamp(min=2),
        raw_lengths,
    )
    r_end = (w_grid + r_len.unsqueeze(1) - 1).clamp(max=L - 1)
    r_valid = (w_grid >= starts.unsqueeze(1)) & (w_grid + r_len.unsqueeze(1) <= L)
    r_spans = (raw_buckets.gather(1, r_end) - raw_buckets.gather(1, w_grid)).abs().float()
    r_jitter = torch.rand((B, L), generator=rng_gen, device=device) * 1e-4
    r_score = torch.where(r_valid, r_spans + r_jitter, torch.full_like(r_spans, 1e30))
    r_best_w = r_score.argmin(dim=1)  # [B]

    in_seg = (w_grid >= r_best_w.unsqueeze(1)) & (w_grid < (r_best_w + r_len).unsqueeze(1))
    # Keys outside [best_w, best_w + r) keep exact index p; keys inside take random values in [best_w, best_w + r - 1e-3)
    seg_keys = r_best_w.unsqueeze(1).float() + torch.rand(
        (B, L), generator=rng_gen, device=device
    ) * (r_len.unsqueeze(1).float() - 1e-3).clamp(min=1e-4)
    sort_keys = torch.where(in_seg, seg_keys, w_grid.float())
    perm_idx = sort_keys.argsort(dim=1)
    reord_seq = seq.gather(1, perm_idx)
    reord_bs = raw_buckets.gather(1, perm_idx)

    reord_mask_2d = reord_rows.unsqueeze(1)
    aug_seq = torch.where(reord_mask_2d, reord_seq, aug_seq)
    aug_buckets = torch.where(reord_mask_2d, reord_bs, aug_buckets)

    # --- Build matching group_positions & num_tokens purely on GPU ---
    aug_starts = L - aug_len
    aug_valid = w_grid >= aug_starts.unsqueeze(1)
    rel = w_grid - aug_starts.unsqueeze(1)

    if pooling_mode == "none":
        aug_groups = torch.where(aug_valid, rel, torch.full_like(rel, -1))
        return aug_seq, aug_groups, aug_len

    if is_lbsp_mode(pooling_mode):
        last_b = aug_buckets[:, -1:]
        same_last = aug_valid & (aug_buckets == last_b)
        # Contiguous trailing last-bucket mask via reversed cumprod (pure GPU)
        in_last = torch.flip(
            torch.cumprod(torch.flip(same_last.long(), dims=[1]), dim=1), dims=[1]
        ).bool()
        last_k = in_last.sum(dim=1).clamp(min=1)
        prefix_len = aug_len - last_k
        aug_nt = prefix_len + 1
        aug_groups = torch.where(in_last, prefix_len.unsqueeze(1), rel)
        aug_groups = torch.where(aug_valid, aug_groups, torch.full_like(rel, -1))
        return aug_seq, aug_groups, aug_nt

    if is_absp_mode(pooling_mode):
        # Contiguous bucket changes along valid positions (pure GPU)
        prev_b = F.pad(aug_buckets[:, :-1], (1, 0), value=0)
        is_new_bucket = aug_valid & ((w_grid == aug_starts.unsqueeze(1)) | (aug_buckets != prev_b))
        grp_1based = is_new_bucket.long().cumsum(dim=1)
        aug_nt = grp_1based[:, -1].clamp(min=1)
        aug_groups = torch.where(aug_valid, grp_1based - 1, torch.full_like(rel, -1))
        return aug_seq, aug_groups, aug_nt

    raise ValueError(f"Unsupported pooling_mode={pooling_mode}")


# =============================================================================
# Strict Full-Catalog Evaluation (exact match to original KAIS protocol)
# =============================================================================
@torch.no_grad()
def evaluate_panel(
    model: torch.nn.Module,
    panel_rows: List[dict],
    panel_tensors: Dict[str, np.ndarray],
    idx_to_raw: Dict[int, str],
    device: torch.device,
    eval_batch_size: int = 256,
    need_time_features: bool = False,
) -> Tuple[dict, List[dict]]:
    model.eval()
    total = len(panel_rows)
    seq_all = panel_tensors["seq"]
    grp_all = panel_tensors["group_positions"]
    nt_all = panel_tensors["num_tokens"]
    tgt_all = panel_tensors["targets"]
    seen_all = panel_tensors.get("seen_mask")
    off_all = panel_tensors.get("token_offsets")
    scl_all = panel_tensors.get("token_scales")

    predictions: List[dict] = []
    ranks_list: List[int] = []

    for start in range(0, total, eval_batch_size):
        end = min(start + eval_batch_size, total)
        seq_t = torch.from_numpy(seq_all[start:end]).long().to(device)
        grp_t = torch.from_numpy(grp_all[start:end]).long().to(device)
        nt_t = torch.from_numpy(nt_all[start:end]).long().to(device)
        tgt_t = torch.from_numpy(tgt_all[start:end]).long().to(device)

        tm_t = None
        if need_time_features:
            off_t = torch.from_numpy(off_all[start:end]).long().to(device)
            scl_t = torch.from_numpy(scl_all[start:end]).long().to(device)
            tm_t = build_time_matrix_gpu(off_t, scl_t, nt_t)

        scores = model.score_all(seq_t, grp_t, nt_t, tm_t)

        # Mask padding index 0 and all unique uncompressed visible history items (except target)
        # using a single vectorized GPU scatter (zero per-user Python loop!).
        scores = scores.clone()
        scores[:, 0] = -torch.inf
        batch_rows = panel_rows[start:end]
        if seen_all is not None:
            seen_t = torch.from_numpy(seen_all[start:end]).long().to(device)
            scores.scatter_(1, seen_t, float("-inf"))
        else:
            for i, row in enumerate(batch_rows):
                seen = set(row["items"]) - {row["target_idx"]}
                if seen:
                    scores[i, list(seen)] = -torch.inf

        target_scores = scores.gather(1, tgt_t.unsqueeze(1))
        ranks = ((scores > target_scores).sum(dim=1) + 1).cpu().tolist()
        top10 = scores.topk(10, dim=1).indices.cpu().tolist()

        for row, rank, top in zip(batch_rows, ranks, top10):
            if 0 in top or len(set(top)) != 10:
                raise RuntimeError(f"Malformed Top-10 list for {row['sample_id']}")
            seen_excl = set(row["items"]) - {row["target_idx"]}
            if set(top) & seen_excl:
                raise RuntimeError(f"Seen-item mask violated in Top-10 for {row['sample_id']}")
            ranks_list.append(int(rank))
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "user_id": row["user_id"],
                    "target_item": row["target_item"],
                    "target_idx": row["target_idx"],
                    "rank": int(rank),
                    "top10_list": top,
                    "top10_item_idx": json.dumps(top),
                    "top10_raw_item_id": json.dumps([idx_to_raw[x] for x in top]),
                }
            )

    return metrics_from_ranks(ranks_list), predictions


def _write_predictions_csv(path: Path, preds: List[dict]) -> None:
    fields = [
        "sample_id",
        "user_id",
        "target_item",
        "target_idx",
        "rank",
        "top10_item_idx",
        "top10_raw_item_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(preds)


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# =============================================================================
# Core Single-Job Runner
# =============================================================================
def run_job(
    dataset_tag: str,
    backbone: str,
    pooling_mode: str,
    seed: int,
    data_root: Path,
    project_root: Path,
    output_dir: Path,
    bundle: Optional[FrozenDatasetBundle] = None,
    batch_size: int = 128,
    eval_batch_size: int = 256,
    lr: float = 0.001,
    min_epochs: int = 50,
    patience: int = 15,
    max_epochs: int = 200,
    hidden_size: int = 64,
    num_heads: int = 2,
    num_blocks: int = 2,
    dropout: float = 0.2,
    verify_hashes: bool = True,
    allow_cpu: bool = False,
    smoke_mode: bool = False,
    save_all_perm_csvs: bool = True,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError("CUDA not available; pass --allow-cpu only for local smoke testing")

    set_deterministic_seed(seed, deterministic=True)
    if bundle is None:
        bundle = FrozenDatasetBundle(dataset_tag, data_root, project_root, verify_hashes=verify_hashes)

    need_time = backbone.lower() == "tisasrec"
    is_cl = backbone.lower() == "ticoserec"

    t0 = time.perf_counter()
    max_pos = 2048 if smoke_mode else None
    train_data = bundle.materialize_train_tensors(
        pooling_mode=pooling_mode,
        need_time_features=need_time,
        need_raw_buckets=is_cl,
        max_positions=max_pos,
    )

    val_tensors = bundle.materialize_panel_tensors("validation", pooling_mode, need_time_features=need_time)
    eval_panels = (
        ["original"]
        + [f"perm_seed{ps}" for ps in PERMUTATION_SEEDS]
        + [f"allbucket_seed{ps}" for ps in PERMUTATION_SEEDS]
    )
    panel_tensors = {
        name: bundle.materialize_panel_tensors(name, pooling_mode, need_time_features=need_time)
        for name in eval_panels
    }
    prep_seconds = time.perf_counter() - t0

    # Move training tensors straight to GPU (tiny footprint: ~160-320 MB on RTX 4090!)
    seq_gpu = torch.from_numpy(train_data["seq"]).to(device=device, dtype=torch.long)
    grp_gpu = torch.from_numpy(train_data["group_positions"]).to(device=device, dtype=torch.long)
    nt_gpu = torch.from_numpy(train_data["num_tokens"]).to(device=device, dtype=torch.long)
    tgt_gpu = torch.from_numpy(train_data["targets"]).to(device=device, dtype=torch.long)
    off_gpu = torch.from_numpy(train_data["token_offsets"]).to(device=device, dtype=torch.long) if need_time else None
    scl_gpu = torch.from_numpy(train_data["token_scales"]).to(device=device, dtype=torch.long) if need_time else None
    rb_gpu = torch.from_numpy(train_data["raw_buckets"]).to(device=device, dtype=torch.long) if is_cl else None
    rl_gpu = torch.from_numpy(train_data["raw_lengths"]).to(device=device, dtype=torch.long) if is_cl else None

    model = build_model(
        backbone=backbone,
        num_items=bundle.num_items,
        pooling_mode=pooling_mode,
        max_len=MAX_LEN,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    perm_gen = torch.Generator(device=device).manual_seed(seed)
    aug_gen = torch.Generator(device=device).manual_seed(seed + 100003)

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": dataset_tag,
        "dataset_name": bundle.dataset_name,
        "backbone": backbone,
        "pooling_mode": pooling_mode,
        "seed": seed,
        "num_items": bundle.num_items,
        "num_params": num_params,
        "train_positions": int(seq_gpu.size(0)),
        "hyperparameters": {
            "max_len": MAX_LEN,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_blocks": num_blocks,
            "dropout": dropout,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "lr": lr,
            "min_epochs": min_epochs,
            "patience": patience,
            "max_epochs": max_epochs,
            "grad_clip": 5.0,
        },
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "prep_seconds": prep_seconds,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _write_json_atomic(output_dir / "config.json", config)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    n_train = int(seq_gpu.size(0))
    actual_max_epochs = 2 if smoke_mode else max_epochs
    actual_min_epochs = 1 if smoke_mode else min_epochs

    best_val_ndcg = -math.inf
    best_epoch = 0
    bad_epochs = 0
    ckpt_path = output_dir / "best_model.pt"

    log_fields = [
        "epoch",
        "train_loss",
        "cl_loss",
        "val_ndcg10",
        "val_hr10",
        "val_hr20",
        "val_recall50",
        "val_mrr",
        "val_mrr10",
        "val_mean_rank",
        "epoch_seconds",
        "best_epoch",
        "bad_epochs",
    ]
    train_start = time.perf_counter()

    with (output_dir / "training_log.csv").open("w", encoding="utf-8", newline="") as log_f:
        writer = csv.DictWriter(log_f, fieldnames=log_fields)
        writer.writeheader()

        for epoch in range(1, actual_max_epochs + 1):
            ep_t0 = time.perf_counter()
            model.train()
            order = torch.randperm(n_train, generator=perm_gen, device=device)

            # Accumulate losses on GPU to avoid per-batch GPU->CPU pipeline stalls!
            loss_sum_gpu = torch.zeros((), dtype=torch.float64, device=device)
            cl_sum_gpu = torch.zeros((), dtype=torch.float64, device=device)
            seen_samples = 0

            for start in range(0, n_train, batch_size):
                idx = order[start : start + batch_size]
                b_seq = seq_gpu[idx]
                b_grp = grp_gpu[idx]
                b_nt = nt_gpu[idx]
                b_tgt = tgt_gpu[idx]

                b_tm = None
                if need_time:
                    b_tm = build_time_matrix_gpu(off_gpu[idx], scl_gpu[idx], b_nt)

                optimizer.zero_grad(set_to_none=True)
                scores = model.score_all(b_seq, b_grp, b_nt, b_tm)
                ce_loss = F.cross_entropy(scores[:, 1:], b_tgt - 1)

                if is_cl:
                    b_rb = rb_gpu[idx]
                    b_rl = rl_gpu[idx]
                    s1, g1, nt1 = _ticoserec_augment_batch(
                        b_seq, b_rb, b_rl, model.mask_token_id, pooling_mode, aug_gen
                    )
                    s2, g2, nt2 = _ticoserec_augment_batch(
                        b_seq, b_rb, b_rl, model.mask_token_id, pooling_mode, aug_gen
                    )
                    # Batch both views [2B, L] into a single GPU forward pass (halves kernel launches)
                    z12 = model.encode(
                        torch.cat([s1, s2], dim=0),
                        torch.cat([g1, g2], dim=0),
                        torch.cat([nt1, nt2], dim=0),
                    )
                    z1, z2 = z12.chunk(2, dim=0)
                    c_loss = model.contrastive_loss(z1, z2)
                    loss = ce_loss + model.cl_weight * c_loss
                    cl_sum_gpu = cl_sum_gpu + c_loss.detach().double() * idx.numel()
                else:
                    loss = ce_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                bs = int(idx.numel())
                loss_sum_gpu = loss_sum_gpu + ce_loss.detach().double() * bs
                seen_samples += bs

            loss_sum = float(loss_sum_gpu.item())
            cl_sum = float(cl_sum_gpu.item())
            if not math.isfinite(loss_sum) or not math.isfinite(cl_sum):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")

            val_metrics, _ = evaluate_panel(
                model,
                bundle.panels["validation"],
                val_tensors,
                bundle.idx_to_raw,
                device,
                eval_batch_size=eval_batch_size,
                need_time_features=need_time,
            )

            if val_metrics["ndcg10"] > best_val_ndcg:
                best_val_ndcg = val_metrics["ndcg10"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "seed": seed,
                        "backbone": backbone,
                        "pooling_mode": pooling_mode,
                        "dataset": dataset_tag,
                        "val_ndcg10": best_val_ndcg,
                    },
                    ckpt_path,
                )
            else:
                bad_epochs += 1

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ep_sec = time.perf_counter() - ep_t0

            row = {
                "epoch": epoch,
                "train_loss": loss_sum / seen_samples,
                "cl_loss": cl_sum / seen_samples,
                "val_ndcg10": val_metrics["ndcg10"],
                "val_hr10": val_metrics["hr10"],
                "val_hr20": val_metrics["hr20"],
                "val_recall50": val_metrics["recall50"],
                "val_mrr": val_metrics["mrr"],
                "val_mrr10": val_metrics["mrr10"],
                "val_mean_rank": val_metrics["mean_rank"],
                "epoch_seconds": ep_sec,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
            }
            writer.writerow(row)
            log_f.flush()

            if epoch == 1 or epoch % 10 == 0 or (epoch >= actual_min_epochs and bad_epochs >= patience):
                print(
                    f"[{dataset_tag}|{backbone}|{pooling_mode}|s{seed}] "
                    f"ep={epoch:03d} loss={row['train_loss']:.4f} "
                    f"val_ndcg10={row['val_ndcg10']:.5f} best={best_val_ndcg:.5f}@{best_epoch} "
                    f"bad={bad_epochs}/{patience} ({ep_sec:.1f}s)",
                    flush=True,
                )

            if epoch >= actual_min_epochs and bad_epochs >= patience:
                break

    total_train_seconds = time.perf_counter() - train_start
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )

    # Restore best validation checkpoint & verify exact reproduction
    saved = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    repro_val, _ = evaluate_panel(
        model,
        bundle.panels["validation"],
        val_tensors,
        bundle.idx_to_raw,
        device,
        eval_batch_size=eval_batch_size,
        need_time_features=need_time,
    )
    if abs(repro_val["ndcg10"] - best_val_ndcg) > 1e-10:
        raise RuntimeError(
            f"Best validation checkpoint reproduction drift: {repro_val['ndcg10']} vs {best_val_ndcg}"
        )

    # Evaluate canonical test + Last-Bucket 5-perm + All-Bucket 5-perm
    panel_metrics: Dict[str, dict] = {}
    panel_preds: Dict[str, List[dict]] = {}
    for name in eval_panels:
        m, preds = evaluate_panel(
            model,
            bundle.panels[name],
            panel_tensors[name],
            bundle.idx_to_raw,
            device,
            eval_batch_size=eval_batch_size,
            need_time_features=need_time,
        )
        panel_metrics[name] = m
        panel_preds[name] = preds
        if name == "original" or save_all_perm_csvs:
            _write_predictions_csv(output_dir / f"predictions_{name}.csv", preds)

    # Compute structured population summaries
    n_test = len(bundle.panels["original"])
    all_idx = list(range(n_test))
    lb_aff_idx = [i for i, f in enumerate(bundle.last_bucket_affected) if f]
    ab_aff_idx = [i for i, f in enumerate(bundle.all_bucket_affected) if f]
    warm_idx = [i for i, f in enumerate(bundle.warm_mask) if f]
    cold_idx = [i for i, f in enumerate(bundle.warm_mask) if not f]
    warm_lb_aff_idx = [
        i for i in range(n_test) if bundle.warm_mask[i] and bundle.last_bucket_affected[i]
    ]
    pooled_idx = [i for i, r in enumerate(bundle.panels["original"]) if r["last_bucket_size"] >= 2]
    singleton_idx = [i for i, r in enumerate(bundle.panels["original"]) if r["last_bucket_size"] == 1]

    orig_ranks = [p["rank"] for p in panel_preds["original"]]
    populations_accuracy = {
        "complete": metrics_from_ranks(orig_ranks),
        "last_bucket_affected": metrics_from_ranks([orig_ranks[i] for i in lb_aff_idx]),
        "all_bucket_affected": metrics_from_ranks([orig_ranks[i] for i in ab_aff_idx]),
        "warm": metrics_from_ranks([orig_ranks[i] for i in warm_idx]),
        "cold": metrics_from_ranks([orig_ranks[i] for i in cold_idx]),
        "warm_last_bucket_affected": metrics_from_ranks([orig_ranks[i] for i in warm_lb_aff_idx]),
        "pooled_last_bucket": metrics_from_ranks([orig_ranks[i] for i in pooled_idx]),
        "singleton_last_bucket": metrics_from_ranks([orig_ranks[i] for i in singleton_idx]),
    }

    lb_five = [panel_preds[f"perm_seed{ps}"] for ps in PERMUTATION_SEEDS]
    ab_five = [panel_preds[f"allbucket_seed{ps}"] for ps in PERMUTATION_SEEDS]

    stability_report = {
        "last_bucket_5perm": {
            "complete": stability_summary(lb_five, all_idx),
            "affected": stability_summary(lb_five, lb_aff_idx),
            "warm_affected": stability_summary(lb_five, warm_lb_aff_idx),
            "by_last_bucket_size": bucket_stratified_stability(lb_five, bundle.panels["original"]),
        },
        "all_bucket_5perm": {
            "complete": stability_summary(ab_five, all_idx),
            "last_bucket_affected": stability_summary(ab_five, lb_aff_idx),
            "all_bucket_affected": stability_summary(ab_five, ab_aff_idx),
        },
    }

    # Exact invariance mathematical assertions!
    if is_lbsp_mode(pooling_mode):
        ov = stability_report["last_bucket_5perm"]["affected"]["overlap10"]
        hf = stability_report["last_bucket_5perm"]["affected"]["hitflip"]
        if abs(ov - 1.0) > 1e-12 or abs(hf - 0.0) > 1e-12:
            raise RuntimeError(
                f"LBSP exact local invariance violated! overlap={ov}, hitflip={hf}"
            )
    if is_absp_mode(pooling_mode):
        for scope in ("last_bucket_5perm", "all_bucket_5perm"):
            ov = stability_report[scope]["complete"]["overlap10"]
            hf = stability_report[scope]["complete"]["hitflip"]
            if abs(ov - 1.0) > 1e-12 or abs(hf - 0.0) > 1e-12:
                raise RuntimeError(
                    f"ABSP exact global invariance violated in {scope}! overlap={ov}, hitflip={hf}"
                )

    summary = {
        "status": "COMPLETED",
        "dataset": dataset_tag,
        "backbone": backbone,
        "pooling_mode": pooling_mode,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation": repro_val,
        "accuracy": populations_accuracy,
        "stability": stability_report,
        "runtime": {
            "prep_seconds": prep_seconds,
            "train_seconds": total_train_seconds,
            "mean_epoch_seconds": total_train_seconds / max(1, epoch),
            "peak_vram_mb": peak_vram_mb,
        },
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _write_json_atomic(output_dir / "metrics.json", summary)
    _write_json_atomic(output_dir / "COMPLETED.json", {"status": "COMPLETED", "best_epoch": best_epoch})

    print(
        f"DONE [{dataset_tag}|{backbone}|{pooling_mode}|s{seed}] "
        f"test_ndcg10={populations_accuracy['complete']['ndcg10']*1000:.2f}e-3 "
        f"aff_ndcg10={populations_accuracy['last_bucket_affected']['ndcg10']*1000:.2f}e-3 "
        f"LB_ov={stability_report['last_bucket_5perm']['affected']['overlap10']*100:.1f}% "
        f"AB_ov={stability_report['all_bucket_5perm']['all_bucket_affected']['overlap10']*100:.1f}% "
        f"({total_train_seconds:.1f}s)",
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["ml1m", "beauty", "sports", "toys"])
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--pooling", default="none", choices=list(POOLING_MODES))
    parser.add_argument("--seed", type=int, default=42, choices=[42, 2026, 3407])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    run_job(
        dataset_tag=args.dataset,
        backbone=args.backbone,
        pooling_mode=args.pooling,
        seed=args.seed,
        data_root=args.data_root,
        project_root=args.project_root,
        output_dir=args.output_dir,
        verify_hashes=not args.skip_hash_check,
        allow_cpu=args.allow_cpu,
        smoke_mode=args.smoke,
    )


if __name__ == "__main__":
    main()
