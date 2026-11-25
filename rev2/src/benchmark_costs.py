"""Unified single-GPU inference & training-epoch cost benchmark across all seed-42 models.

Eliminates cross-device hardware heterogeneity by timing every backbone and
pooling variant on the exact same GPU under identical batching, warm-up,
and CUDA synchronization boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import time
from typing import List

import torch
import torch.nn.functional as F

from .eval_legacy_checkpoints import load_legacy_state_dict
from .models import build_model
from .protocol import (
    MAX_LEN,
    FrozenDatasetBundle,
    build_time_matrix_gpu,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_inference(
    model: torch.nn.Module,
    bundle: FrozenDatasetBundle,
    pooling_mode: str,
    device: torch.device,
    need_time: bool,
    eval_batch_size: int = 256,
    repeats: int = 5,
) -> dict:
    model.eval()
    rows = bundle.panels["original"]
    tensors = bundle.materialize_panel_tensors("original", pooling_mode, need_time_features=need_time)
    total = len(rows)

    def run_pass(limit_batches: int = 0) -> None:
        n_batches = 0
        for start in range(0, total, eval_batch_size):
            end = min(start + eval_batch_size, total)
            seq_t = torch.from_numpy(tensors["seq"][start:end]).long().to(device)
            grp_t = torch.from_numpy(tensors["group_positions"][start:end]).long().to(device)
            nt_t = torch.from_numpy(tensors["num_tokens"][start:end]).long().to(device)
            tgt_t = torch.from_numpy(tensors["targets"][start:end]).long().to(device)
            tm_t = None
            if need_time:
                off_t = torch.from_numpy(tensors["token_offsets"][start:end]).long().to(device)
                scl_t = torch.from_numpy(tensors["token_scales"][start:end]).long().to(device)
                tm_t = build_time_matrix_gpu(off_t, scl_t, nt_t)

            scores = model.score_all(seq_t, grp_t, nt_t, tm_t)
            scores[:, 0] = -torch.inf
            for i, row in enumerate(rows[start:end]):
                seen = set(row["items"]) - {row["target_idx"]}
                if seen:
                    scores[i, list(seen)] = -torch.inf
            target_scores = scores.gather(1, tgt_t.unsqueeze(1))
            _ = ((scores > target_scores).sum(dim=1) + 1).cpu()
            _ = scores.topk(10, dim=1).indices.cpu()
            n_batches += 1
            if limit_batches and n_batches >= limit_batches:
                break

    # Warm-up 2 batches outside timing
    run_pass(limit_batches=2)
    _sync(device)

    times = []
    peaks_mb = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        t0 = time.perf_counter()
        run_pass()
        _sync(device)
        times.append(time.perf_counter() - t0)
        if device.type == "cuda":
            peaks_mb.append(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    return {
        "n_users": total,
        "repeats": repeats,
        "mean_seconds": statistics.mean(times),
        "sd_seconds": statistics.stdev(times) if len(times) > 1 else 0.0,
        "ms_per_user": statistics.mean(times) * 1000.0 / total,
        "peak_vram_mb": max(peaks_mb) if peaks_mb else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA required for GPU cost profiling (pass --allow-cpu for testing)")

    runs_root = args.runs_root or (args.project_root / "runs")
    out_csv = args.out_csv or (runs_root / "unified_cost_profile.csv")

    # Find all completed seed-42 jobs (both newly trained best_model.pt and stage0_legacy_42 references)
    job_dirs: List[Path] = sorted(p.parent for p in runs_root.glob("*/*__seed42/COMPLETED.json"))
    if not job_dirs:
        print(f"No completed seed-42 jobs found under {runs_root}")
        return

    bundles = {}
    records = []
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

    for job_dir in job_dirs:
        cfg = json.loads((job_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((job_dir / "metrics.json").read_text(encoding="utf-8"))
        ds = cfg["dataset"]
        bb = cfg["backbone"]
        arch = cfg.get("arch", "sasrec" if bb == "sasrec_shuffle" else bb)
        pool = cfg["pooling_mode"]

        ckpt_path = job_dir / "best_model.pt"
        if not ckpt_path.is_file() and cfg.get("legacy_checkpoint_path"):
            ckpt_path = Path(cfg["legacy_checkpoint_path"])
        if not ckpt_path.is_file():
            continue

        if ds not in bundles:
            bundles.clear()
            bundles[ds] = FrozenDatasetBundle(ds, args.data_root, args.project_root, verify_hashes=False)
        bundle = bundles[ds]

        model = build_model(arch, bundle.num_items, pooling_mode=pool, max_len=MAX_LEN).to(device)
        sd = load_legacy_state_dict(ckpt_path, arch, device)
        model.load_state_dict(sd, strict=True)

        inf = benchmark_inference(
            model, bundle, pool, device, need_time=(arch == "tisasrec"), repeats=args.repeats
        )
        mean_ep = metrics["runtime"].get("mean_epoch_seconds")
        rec = {
            "dataset": ds,
            "backbone": bb,
            "pooling_mode": pool,
            "num_params": cfg["num_params"],
            "device": gpu_name,
            "mean_train_epoch_s": mean_ep if mean_ep is not None else "",
            "train_peak_vram_mb": metrics["runtime"].get("peak_vram_mb") or "",
            "inf_panel_mean_s": inf["mean_seconds"],
            "inf_panel_sd_s": inf["sd_seconds"],
            "inf_ms_per_user": inf["ms_per_user"],
            "inf_peak_vram_mb": inf["peak_vram_mb"],
        }
        records.append(rec)
        ep_str = f"{mean_ep:.2f}s" if isinstance(mean_ep, (int, float)) else "legacy"
        print(
            f"[COST] {ds:6s} | {bb:14s} | {pool:13s} | "
            f"train_ep={ep_str:>7s} | inf={rec['inf_panel_mean_s']*1000:.1f}ms",
            flush=True,
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"Saved unified cost profile to {out_csv}")


if __name__ == "__main__":
    main()
