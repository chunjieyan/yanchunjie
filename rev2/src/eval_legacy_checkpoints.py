"""Read-only zero-training evaluator for the 42 existing legacy .pt checkpoints
already stored on the AutoDL server (/root/autodl-tmp/time_uncertainty_rec).

Does NOT retrain or modify any existing checkpoint. For every legacy model:
1. Resolves its .pt file (exact server path first, SHA256 scan fallback).
2. Verifies SHA256 (if recorded) and loads weights into UnifiedSASRec / UnifiedTiSASRec.
3. Evaluates canonical test, sealed Last-Bucket 5-Perm, and NEW All-Bucket 5-Perm + Warm/Cold.
4. Verifies canonical test NDCG@10 reproduces the published manuscript number.
5. Writes standardized outputs under runs/stage0_legacy_42/ so benchmark_costs.py and
   aggregate_results.py seamlessly merge all 42 legacy + all 174 new models.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, Optional

import torch

from .models import build_model
from .protocol import (
    MAX_LEN,
    PERMUTATION_SEEDS,
    FrozenDatasetBundle,
    bucket_stratified_stability,
    metrics_from_ranks,
    stability_summary,
)
from .train_eval import _write_json_atomic, _write_predictions_csv, evaluate_panel


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_legacy_state_dict(ckpt_path: Path, backbone: str, device: torch.device) -> dict:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            sd = raw["model_state_dict"]
        elif "state_dict" in raw:
            sd = raw["state_dict"]
        else:
            sd = raw
    else:
        raise RuntimeError(f"Unexpected checkpoint object type: {type(raw)}")

    if backbone == "tisasrec":
        remapped = {}
        for k, v in sd.items():
            if k.startswith("ti_trm_encoder.layer."):
                remapped["layers." + k[len("ti_trm_encoder.layer.") :]] = v
            else:
                remapped[k] = v
        return remapped
    return sd


def _index_pt_by_sha(server_root: Path) -> Dict[str, Path]:
    index = {}
    for pt in server_root.rglob("*.pt"):
        if pt.is_file():
            index[sha256_file(pt)] = pt
    return index


def evaluate_all_legacy(
    server_root: Path,
    project_root: Path,
    runs_root: Path,
    verify_data_hashes: bool = True,
    allow_cpu: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError("CUDA not available; pass --allow-cpu only for local testing")

    manifest = json.loads(
        (project_root / "manifests" / "legacy_42_checkpoints.json").read_text(encoding="utf-8")
    )
    models = manifest["models"]

    # Group by dataset so each FrozenDatasetBundle is loaded once
    by_ds: Dict[str, list] = {}
    for entry in models:
        by_ds.setdefault(entry["dataset"], []).append(entry)

    sha_index: Optional[Dict[str, Path]] = None
    out_stage = runs_root / "stage0_legacy_42"
    out_stage.mkdir(parents=True, exist_ok=True)

    done = 0
    missing = 0

    for ds, entries in by_ds.items():
        gc.collect()
        print(f"\n>>> [Stage 0 Legacy] Loading frozen dataset: {ds} ...", flush=True)
        bundle = FrozenDatasetBundle(ds, server_root, project_root, verify_hashes=verify_data_hashes)

        # Pre-materialize panel tensors per (pooling_mode, need_time)
        tensor_cache = {}
        eval_panels = (
            ["original"]
            + [f"perm_seed{ps}" for ps in PERMUTATION_SEEDS]
            + [f"allbucket_seed{ps}" for ps in PERMUTATION_SEEDS]
        )

        for entry in entries:
            bb = entry["backbone"]  # sasrec, sasrec_shuffle, tisasrec
            pool = entry["pooling_mode"]  # none, lbsp_mean
            seed = int(entry["seed"])
            out_dir = out_stage / f"{ds}__{bb}__{pool}__seed{seed}"
            if (out_dir / "COMPLETED.json").is_file():
                done += 1
                print(f"  [SKIP] {out_dir.name} already evaluated", flush=True)
                continue

            # Resolve checkpoint path on server
            raw_path = entry["server_checkpoint_path"]
            rel = raw_path.replace("/root/autodl-tmp/time_uncertainty_rec/", "")
            ckpt_path = server_root / rel
            if not ckpt_path.is_file() and entry.get("checkpoint_sha256"):
                if sha_index is None:
                    print("  Scanning server_root for .pt files by SHA256 fallback...", flush=True)
                    sha_index = _index_pt_by_sha(server_root)
                ckpt_path = sha_index.get(entry["checkpoint_sha256"])

            if ckpt_path is None or not ckpt_path.is_file():
                missing += 1
                print(
                    f"  [WARN] Legacy checkpoint not found for {entry['training_job_id']}: {raw_path}",
                    flush=True,
                )
                continue

            actual_sha = sha256_file(ckpt_path)
            if entry.get("checkpoint_sha256") and actual_sha != entry["checkpoint_sha256"]:
                raise RuntimeError(
                    f"SHA256 mismatch for legacy checkpoint {ckpt_path}: "
                    f"{actual_sha} != {entry['checkpoint_sha256']}"
                )

            arch = "sasrec" if bb == "sasrec_shuffle" else bb
            need_time = arch == "tisasrec"
            cache_key = (pool, need_time)
            if cache_key not in tensor_cache:
                tensor_cache[cache_key] = {
                    name: bundle.materialize_panel_tensors(name, pool, need_time_features=need_time)
                    for name in eval_panels
                }

            model = build_model(arch, bundle.num_items, pooling_mode=pool, max_len=MAX_LEN).to(device)
            sd = load_legacy_state_dict(ckpt_path, arch, device)
            model.load_state_dict(sd, strict=True)
            model.eval()

            t0 = time.perf_counter()
            out_dir.mkdir(parents=True, exist_ok=True)
            panel_preds = {}
            for name in eval_panels:
                _, preds = evaluate_panel(
                    model,
                    bundle.panels[name],
                    tensor_cache[cache_key][name],
                    bundle.idx_to_raw,
                    device,
                    eval_batch_size=256,
                    need_time_features=need_time,
                )
                panel_preds[name] = preds
                _write_predictions_csv(out_dir / f"predictions_{name}.csv", preds)

            n_test = len(bundle.panels["original"])
            all_idx = list(range(n_test))
            lb_aff_idx = [i for i, f in enumerate(bundle.last_bucket_affected) if f]
            ab_aff_idx = [i for i, f in enumerate(bundle.all_bucket_affected) if f]
            warm_idx = [i for i, f in enumerate(bundle.warm_mask) if f]
            cold_idx = [i for i, f in enumerate(bundle.warm_mask) if not f]
            warm_lb_aff_idx = [
                i for i in range(n_test) if bundle.warm_mask[i] and bundle.last_bucket_affected[i]
            ]
            pooled_idx = [
                i for i, r in enumerate(bundle.panels["original"]) if r["last_bucket_size"] >= 2
            ]
            singleton_idx = [
                i for i, r in enumerate(bundle.panels["original"]) if r["last_bucket_size"] == 1
            ]

            orig_ranks = [p["rank"] for p in panel_preds["original"]]
            acc = {
                "complete": metrics_from_ranks(orig_ranks),
                "last_bucket_affected": metrics_from_ranks([orig_ranks[i] for i in lb_aff_idx]),
                "all_bucket_affected": metrics_from_ranks([orig_ranks[i] for i in ab_aff_idx]),
                "warm": metrics_from_ranks([orig_ranks[i] for i in warm_idx]),
                "cold": metrics_from_ranks([orig_ranks[i] for i in cold_idx]),
                "warm_last_bucket_affected": metrics_from_ranks([orig_ranks[i] for i in warm_lb_aff_idx]),
                "pooled_last_bucket": metrics_from_ranks([orig_ranks[i] for i in pooled_idx]),
                "singleton_last_bucket": metrics_from_ranks([orig_ranks[i] for i in singleton_idx]),
            }

            pub_ndcg = entry.get("published_canonical_ndcg10")
            repro_diff = abs(acc["complete"]["ndcg10"] - pub_ndcg) if pub_ndcg is not None else 0.0
            if pub_ndcg is not None and repro_diff > 1e-5:
                raise RuntimeError(
                    f"Legacy reproduction drift for {entry['training_job_id']}: "
                    f"actual={acc['complete']['ndcg10']:.8f} vs published={pub_ndcg:.8f}"
                )

            lb_five = [panel_preds[f"perm_seed{ps}"] for ps in PERMUTATION_SEEDS]
            ab_five = [panel_preds[f"allbucket_seed{ps}"] for ps in PERMUTATION_SEEDS]
            stab = {
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

            cfg = {
                "dataset": ds,
                "dataset_name": bundle.dataset_name,
                "backbone": bb,
                "arch": arch,
                "pooling_mode": pool,
                "seed": seed,
                "legacy_method": entry["legacy_method"],
                "legacy_checkpoint_path": str(ckpt_path),
                "legacy_checkpoint_sha256": actual_sha,
                "num_items": bundle.num_items,
                "num_params": sum(p.numel() for p in model.parameters()),
                "reused_legacy_checkpoint": True,
                "published_canonical_ndcg10": pub_ndcg,
                "canonical_ndcg10_abs_diff": repro_diff,
            }
            _write_json_atomic(out_dir / "config.json", cfg)

            summary = {
                "status": "COMPLETED",
                "dataset": ds,
                "backbone": bb,
                "pooling_mode": pool,
                "seed": seed,
                "reused_legacy_checkpoint": True,
                "accuracy": acc,
                "stability": stab,
                "runtime": {
                    "eval_seconds": time.perf_counter() - t0,
                    "mean_epoch_seconds": None,
                    "peak_vram_mb": None,
                },
                "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            _write_json_atomic(out_dir / "metrics.json", summary)
            _write_json_atomic(out_dir / "COMPLETED.json", {"status": "COMPLETED", "legacy": True})
            done += 1
            print(
                f"  [PASS] {ds:6s} | {bb:14s} | {pool:9s} | s{seed} | "
                f"ndcg10={acc['complete']['ndcg10']*1000:.2f}e-3 (diff={repro_diff:.1e}) | "
                f"LB_ov={stab['last_bucket_5perm']['affected']['overlap10']*100:.1f}% | "
                f"AB_ov={stab['all_bucket_5perm']['all_bucket_affected']['overlap10']*100:.1f}%",
                flush=True,
            )

    print(f"\n=== Stage 0 Legacy Evaluation Done: {done}/42 evaluated, {missing} missing ===", flush=True)
    if missing > 0:
        raise RuntimeError(
            f"{missing} legacy checkpoints were not found under {server_root}. "
            "Check paths or run without --reuse-legacy-server to retrain missing models."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/time_uncertainty_rec"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    runs_root = args.runs_root or (args.project_root / "runs")
    evaluate_all_legacy(
        args.data_root,
        args.project_root,
        runs_root,
        verify_data_hashes=not args.skip_hash_check,
        allow_cpu=args.allow_cpu,
    )


if __name__ == "__main__":
    main()
