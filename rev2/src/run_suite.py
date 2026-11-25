"""Resume-safe AutoDL RTX 4090 experiment suite orchestrator for KAIS Revision 2.0.

Supports multi-worker single-GPU concurrency (``--workers 3``) so small d=64
models (~0.5-1.5 GB VRAM each) saturate a 24 GB RTX 4090 simultaneously with
ZERO mathematical or protocol change!

Stages (ordered by acceptance-probability ROI):
- stage1: Complete 8-backbone comparative benchmark across all 4 datasets x 3 seeds:
          TiSASRec (WSDM 20), GRU4Rec (ICLR 16), FMLP-Rec (WWW 22), BSARec (AAAI 24),
          LRURec (WSDM 24), LSAB (ACM TKDD 24), TiCoSeRec (IEEE TKDE 24).
- stage2: Cross-backbone plug-in LBSP generality across 4 non-SASRec families
          (TiSASRec+LBSP, GRU4Rec+LBSP, BSARec+LBSP, TiCoSeRec+LBSP) x 4 datasets x 3 seeds.
- stage3: Pooling-operator ablations (LBSP-Max, LBSP-Attn, LBSP-DeepSets) + Global All-Bucket
          Set Pooling (ABSP-Mean).
- all:    stage1 -> stage2 -> stage3 sequentially.
"""

from __future__ import annotations

import argparse
import csv
import gc
import multiprocessing as mp
from pathlib import Path
import time
import traceback
from typing import List, Tuple

import torch

from .protocol import TRAINING_SEEDS, FrozenDatasetBundle
from .train_eval import run_job

ALL_DATASETS = ("toys", "beauty", "sports", "ml1m")
LEDGER_FIELDS = [
    "stage", "dataset", "backbone", "pooling", "seed", "status",
    "best_epoch", "epochs_run", "complete_ndcg10", "affected_ndcg10",
    "warm_ndcg10", "lb_affected_overlap10", "lb_affected_hitflip",
    "ab_affected_overlap10", "ab_affected_hitflip", "train_seconds", "error",
]


def build_job_list(
    stage: str,
    datasets: Tuple[str, ...],
    seeds: Tuple[int, ...],
    reuse_legacy: bool = False,
) -> List[Tuple[str, str, str, str, int]]:
    jobs: List[Tuple[str, str, str, str, int]] = []

    if stage in ("stage1", "all"):
        for ds in datasets:
            for backbone in ("tisasrec", "gru4rec", "fmlprec", "bsarec", "lrurec", "lsab", "ticoserec"):
                if reuse_legacy and backbone == "tisasrec" and ds in ("ml1m", "beauty"):
                    continue
                for s in seeds:
                    jobs.append(("stage1_baselines", ds, backbone, "none", s))

    if stage in ("stage2", "all"):
        for ds in datasets:
            for backbone in ("tisasrec", "gru4rec", "bsarec", "ticoserec"):
                for s in seeds:
                    jobs.append(("stage2_cross_backbone_lbsp", ds, backbone, "lbsp_mean", s))

    if stage in ("stage3", "all"):
        poolings = (
            ("lbsp_max", "lbsp_attn", "lbsp_deepsets", "absp_mean")
            if reuse_legacy
            else ("none", "lbsp_mean", "lbsp_max", "lbsp_attn", "lbsp_deepsets", "absp_mean")
        )
        for ds in datasets:
            for pooling in poolings:
                for s in seeds:
                    jobs.append(("stage3_ablations_absp", ds, "sasrec", pooling, s))

    return jobs


def _execute_single(job, data_root, project_root, bundle_cache, verify_hashes, allow_cpu, smoke, worker_tag=""):
    stg, ds, bb, pool, seed, out_dir = job
    if ds not in bundle_cache:
        bundle_cache.clear()
        gc.collect()
        print(f"{worker_tag}>>> Loading & auditing frozen dataset: {ds} ...", flush=True)
        bundle_cache[ds] = FrozenDatasetBundle(ds, data_root, project_root, verify_hashes=verify_hashes)

    print(f"{worker_tag}--- START: {stg} | {ds} | {bb} | {pool} | seed={seed} ---", flush=True)
    try:
        res = run_job(
            dataset_tag=ds,
            backbone=bb,
            pooling_mode=pool,
            seed=seed,
            data_root=data_root,
            project_root=project_root,
            output_dir=out_dir,
            bundle=bundle_cache[ds],
            verify_hashes=verify_hashes,
            allow_cpu=allow_cpu,
            smoke_mode=smoke,
        )
        row = {
            "stage": stg,
            "dataset": ds,
            "backbone": bb,
            "pooling": pool,
            "seed": seed,
            "status": "COMPLETED",
            "best_epoch": res["best_epoch"],
            "epochs_run": res["epochs_run"],
            "complete_ndcg10": res["accuracy"]["complete"]["ndcg10"],
            "affected_ndcg10": res["accuracy"]["last_bucket_affected"]["ndcg10"],
            "warm_ndcg10": res["accuracy"]["warm"]["ndcg10"],
            "lb_affected_overlap10": res["stability"]["last_bucket_5perm"]["affected"]["overlap10"],
            "lb_affected_hitflip": res["stability"]["last_bucket_5perm"]["affected"]["hitflip"],
            "ab_affected_overlap10": res["stability"]["all_bucket_5perm"]["all_bucket_affected"]["overlap10"],
            "ab_affected_hitflip": res["stability"]["all_bucket_5perm"]["all_bucket_affected"]["hitflip"],
            "train_seconds": res["runtime"]["train_seconds"],
            "error": "",
        }
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"{worker_tag}ERROR in {ds}|{bb}|{pool}|s{seed}: {exc}\n{tb}", flush=True)
        row = {
            "stage": stg, "dataset": ds, "backbone": bb, "pooling": pool, "seed": seed,
            "status": "FAILED", "best_epoch": "", "epochs_run": "", "complete_ndcg10": "",
            "affected_ndcg10": "", "warm_ndcg10": "", "lb_affected_overlap10": "",
            "lb_affected_hitflip": "", "ab_affected_overlap10": "", "ab_affected_hitflip": "",
            "train_seconds": "", "error": f"{type(exc).__name__}: {exc}",
        }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def _worker_main(worker_id, job_queue, result_queue, data_root, project_root, verify_hashes, allow_cpu, smoke):
    # Limit CPU threads per worker so N workers don't contend on CPU
    torch.set_num_threads(2)
    bundle_cache = {}
    tag = f"[W{worker_id}] "
    while True:
        job = job_queue.get()
        if job is None:
            break
        row = _execute_single(
            job, data_root, project_root, bundle_cache, verify_hashes, allow_cpu, smoke, worker_tag=tag
        )
        result_queue.put(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=["stage1", "stage2", "stage3", "all"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS), choices=["ml1m", "beauty", "sports", "toys"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAINING_SEEDS), choices=[42, 2026, 3407])
    parser.add_argument("--workers", type=int, default=1, help="Concurrent GPU worker processes (recommended: 3 on RTX 4090 24GB)")
    parser.add_argument("--reuse-legacy-server", action="store_true",
                        help="Skip retraining the 30 models whose .pt weights already exist on /root/autodl-tmp/time_uncertainty_rec")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs_root = args.runs_root or (args.project_root / "runs")
    jobs = build_job_list(
        args.stage, tuple(args.datasets), tuple(args.seeds), reuse_legacy=args.reuse_legacy_server
    )

    pending = []
    completed = 0
    for stg, ds, bb, pool, seed in jobs:
        out_dir = runs_root / stg / f"{ds}__{bb}__{pool}__seed{seed}"
        if (out_dir / "COMPLETED.json").is_file():
            completed += 1
        else:
            pending.append((stg, ds, bb, pool, seed, out_dir))

    n_workers = max(1, min(int(args.workers), max(1, len(pending))))
    print(
        f"=== KAIS Revision 2.0 Suite [{args.stage}] (workers={n_workers}) ===\n"
        f"Total jobs: {len(jobs)} | Already completed: {completed} | Pending: {len(pending)}",
        flush=True,
    )
    if args.dry_run or not pending:
        for i, (stg, ds, bb, pool, seed, _) in enumerate(pending, 1):
            print(f"  {i:03d}. {stg} | {ds:6s} | {bb:10s} | {pool:13s} | seed={seed}")
        return

    runs_root.mkdir(parents=True, exist_ok=True)
    ledger_path = runs_root / "suite_ledger.csv"
    write_header = not ledger_path.exists()

    def record_row(row, done_count, elapsed):
        nonlocal write_header
        with ledger_path.open("a", encoding="utf-8", newline="") as lf:
            w = csv.DictWriter(lf, fieldnames=LEDGER_FIELDS)
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)
        avg = elapsed / done_count
        eta_min = avg * (len(pending) - done_count) / 60.0
        print(
            f"Progress: {done_count}/{len(pending)} (overall {completed + done_count}/{len(jobs)}) "
            f"| Elapsed: {elapsed/60:.1f}m | ETA: {eta_min:.1f}m",
            flush=True,
        )

    suite_t0 = time.perf_counter()
    done_now = 0
    failed = 0

    if n_workers == 1:
        bundle_cache = {}
        for idx, job in enumerate(pending, 1):
            row = _execute_single(
                job, args.data_root, args.project_root, bundle_cache,
                not args.skip_hash_check, args.allow_cpu, args.smoke
            )
            if row["status"] == "COMPLETED":
                done_now += 1
            else:
                failed += 1
            record_row(row, idx, time.perf_counter() - suite_t0)
    else:
        ctx = mp.get_context("spawn")
        job_q = ctx.Queue()
        res_q = ctx.Queue()
        for job in pending:
            job_q.put(job)
        for _ in range(n_workers):
            job_q.put(None)

        procs = []
        for wid in range(1, n_workers + 1):
            p = ctx.Process(
                target=_worker_main,
                args=(
                    wid, job_q, res_q, args.data_root, args.project_root,
                    not args.skip_hash_check, args.allow_cpu, args.smoke,
                ),
            )
            p.start()
            procs.append(p)

        for idx in range(1, len(pending) + 1):
            row = res_q.get()
            if row["status"] == "COMPLETED":
                done_now += 1
            else:
                failed += 1
            record_row(row, idx, time.perf_counter() - suite_t0)

        for p in procs:
            p.join()

    print(
        f"\n=== Suite Finished: completed={done_now}, failed={failed}, "
        f"total_time={(time.perf_counter()-suite_t0)/60:.1f} min ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
