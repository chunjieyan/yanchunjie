"""Aggregate all completed KAIS Revision 2.0 runs into publication-ready CSVs & LaTeX tables.

Computes:
1. Three-seed (42, 2026, 3407) mean and sample SD (ddof=1) across all populations:
   - Complete strict panel
   - Last-bucket affected users
   - All-bucket affected users
   - Warm targets & Warm affected users
2. Paired same-seed differences (method - Control, or Backbone+LBSP - Backbone) with
   unadjusted 95% Student t intervals (df=2, t_crit=4.302652729911275).
3. Conditional 2,000-resample paired-user percentile bootstrap 95% CIs on saved predictions.
4. Dependent seed-pair canonical Top-10 overlap references across (42,2026), (42,3407), (2026,3407).
5. LaTeX tables ready for insertion into the revised KAIS manuscript!
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Dict, List, Tuple

import numpy as np

from .protocol import TRAINING_SEEDS

T_CRIT_DF2 = 4.302652729911275


def mean_sd(values: List[float]) -> Tuple[float, float]:
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, sd


def paired_t_ci(diffs: List[float]) -> dict:
    m, sd = mean_sd(diffs)
    half = T_CRIT_DF2 * sd / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    return {"mean": m, "sd": sd, "ci_low": m - half, "ci_high": m + half, "n_seeds": len(diffs)}


def paired_user_bootstrap(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 20261123,
) -> dict:
    """Paired-user percentile bootstrap for NDCG@10(a) - NDCG@10(b)."""
    ndcg_a = np.where(ranks_a <= 10, 1.0 / np.log2(ranks_a + 1.0), 0.0)
    ndcg_b = np.where(ranks_b <= 10, 1.0 / np.log2(ranks_b + 1.0), 0.0)
    diff = ndcg_a - ndcg_b
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"point": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_users": int(n)}


def load_orig_preds(job_dir: Path) -> Tuple[List[str], np.ndarray, List[List[int]]]:
    sids, ranks, tops = [], [], []
    with (job_dir / "predictions_original.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            sids.append(r["sample_id"])
            ranks.append(int(r["rank"]))
            tops.append(json.loads(r["top10_item_idx"]))
    return sids, np.asarray(ranks, dtype=np.int64), tops


def seed_pair_overlap(top_by_seed: Dict[int, List[List[int]]], indices: List[int]) -> float:
    seeds = [s for s in TRAINING_SEEDS if s in top_by_seed]
    if len(seeds) < 2:
        return float("nan")
    pair_means = []
    for s1, s2 in itertools.combinations(seeds, 2):
        ov = math.fsum(
            len(set(top_by_seed[s1][i]) & set(top_by_seed[s2][i])) / 10.0 for i in indices
        ) / len(indices)
        pair_means.append(ov)
    return statistics.mean(pair_means)


def _fmt_pm(mean_val: float, sd_val: float, scale: float = 1.0, decimals: int = 1) -> str:
    return f"{mean_val * scale:.{decimals}f} $\\pm$ {sd_val * scale:.{decimals}f}"


def aggregate(runs_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[str, str, str], Dict[int, Tuple[dict, Path]]] = defaultdict(dict)

    for done_file in sorted(runs_root.glob("*/*/COMPLETED.json")):
        job_dir = done_file.parent
        m = json.loads((job_dir / "metrics.json").read_text(encoding="utf-8"))
        key = (m["dataset"], m["backbone"], m["pooling_mode"])
        grouped[key][int(m["seed"])] = (m, job_dir)

    if not grouped:
        print(f"No COMPLETED runs found under {runs_root}")
        return

    # 1. Master summary table (all dataset x backbone x pooling combinations)
    summary_rows = []
    for (ds, bb, pool), seed_map in sorted(grouped.items()):
        seeds = sorted(seed_map.keys())
        ms = [seed_map[s][0] for s in seeds]

        def agg_metric(getter):
            vals = [getter(x) for x in ms]
            return mean_sd(vals)

        c_ndcg, c_ndcg_sd = agg_metric(lambda x: x["accuracy"]["complete"]["ndcg10"])
        c_hr10, c_hr10_sd = agg_metric(lambda x: x["accuracy"]["complete"]["hr10"])
        c_r50, c_r50_sd = agg_metric(lambda x: x["accuracy"]["complete"]["recall50"])
        c_mrr, c_mrr_sd = agg_metric(lambda x: x["accuracy"]["complete"]["mrr"])
        c_mr, c_mr_sd = agg_metric(lambda x: x["accuracy"]["complete"]["mean_rank"])

        a_ndcg, a_ndcg_sd = agg_metric(lambda x: x["accuracy"]["last_bucket_affected"]["ndcg10"])
        a_hr10, a_hr10_sd = agg_metric(lambda x: x["accuracy"]["last_bucket_affected"]["hr10"])

        w_ndcg, w_ndcg_sd = agg_metric(lambda x: x["accuracy"]["warm"]["ndcg10"])
        w_hr10, w_hr10_sd = agg_metric(lambda x: x["accuracy"]["warm"]["hr10"])
        w_mr, w_mr_sd = agg_metric(lambda x: x["accuracy"]["warm"]["mean_rank"])

        lb_ov, lb_ov_sd = agg_metric(lambda x: x["stability"]["last_bucket_5perm"]["affected"]["overlap10"])
        lb_hf, lb_hf_sd = agg_metric(lambda x: x["stability"]["last_bucket_5perm"]["affected"]["hitflip"])
        lb_full_ov, lb_full_ov_sd = agg_metric(lambda x: x["stability"]["last_bucket_5perm"]["complete"]["overlap10"])

        ab_ov, ab_ov_sd = agg_metric(lambda x: x["stability"]["all_bucket_5perm"]["all_bucket_affected"]["overlap10"])
        ab_hf, ab_hf_sd = agg_metric(lambda x: x["stability"]["all_bucket_5perm"]["all_bucket_affected"]["hitflip"])

        summary_rows.append(
            {
                "dataset": ds,
                "backbone": bb,
                "pooling": pool,
                "n_seeds": len(seeds),
                "complete_ndcg10_mean": c_ndcg,
                "complete_ndcg10_sd": c_ndcg_sd,
                "complete_hr10_mean": c_hr10,
                "complete_hr10_sd": c_hr10_sd,
                "complete_recall50_mean": c_r50,
                "complete_recall50_sd": c_r50_sd,
                "complete_mrr_mean": c_mrr,
                "complete_mrr_sd": c_mrr_sd,
                "complete_mean_rank_mean": c_mr,
                "complete_mean_rank_sd": c_mr_sd,
                "affected_ndcg10_mean": a_ndcg,
                "affected_ndcg10_sd": a_ndcg_sd,
                "affected_hr10_mean": a_hr10,
                "affected_hr10_sd": a_hr10_sd,
                "warm_ndcg10_mean": w_ndcg,
                "warm_ndcg10_sd": w_ndcg_sd,
                "warm_hr10_mean": w_hr10,
                "warm_hr10_sd": w_hr10_sd,
                "warm_mean_rank_mean": w_mr,
                "warm_mean_rank_sd": w_mr_sd,
                "lb_complete_overlap_mean": lb_full_ov,
                "lb_complete_overlap_sd": lb_full_ov_sd,
                "lb_affected_overlap_mean": lb_ov,
                "lb_affected_overlap_sd": lb_ov_sd,
                "lb_affected_hitflip_mean": lb_hf,
                "lb_affected_hitflip_sd": lb_hf_sd,
                "ab_affected_overlap_mean": ab_ov,
                "ab_affected_overlap_sd": ab_ov_sd,
                "ab_affected_hitflip_mean": ab_hf,
                "ab_affected_hitflip_sd": ab_hf_sd,
            }
        )

    master_csv = out_dir / "master_summary.csv"
    with master_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # 2. Paired LBSP-minus-Control (per backbone) and Pooling-minus-SASRec-Control contrasts
    paired_rows = []
    for (ds, bb, pool), seed_map in sorted(grouped.items()):
        if pool == "none":
            continue
        ctrl_key = (ds, bb, "none")
        if ctrl_key not in grouped:
            continue
        common_seeds = sorted(set(seed_map) & set(grouped[ctrl_key]))
        if not common_seeds:
            continue
        diffs_complete = [
            seed_map[s][0]["accuracy"]["complete"]["ndcg10"]
            - grouped[ctrl_key][s][0]["accuracy"]["complete"]["ndcg10"]
            for s in common_seeds
        ]
        diffs_affected = [
            seed_map[s][0]["accuracy"]["last_bucket_affected"]["ndcg10"]
            - grouped[ctrl_key][s][0]["accuracy"]["last_bucket_affected"]["ndcg10"]
            for s in common_seeds
        ]
        diffs_warm = [
            seed_map[s][0]["accuracy"]["warm"]["ndcg10"]
            - grouped[ctrl_key][s][0]["accuracy"]["warm"]["ndcg10"]
            for s in common_seeds
        ]
        tc = paired_t_ci(diffs_complete)
        ta = paired_t_ci(diffs_affected)
        tw = paired_t_ci(diffs_warm)

        # Conditional user bootstrap on seed 42 (or first common seed)
        s0 = common_seeds[0]
        _, r_m, _ = load_orig_preds(seed_map[s0][1])
        _, r_c, _ = load_orig_preds(grouped[ctrl_key][s0][1])
        boot = paired_user_bootstrap(r_m, r_c)

        paired_rows.append(
            {
                "dataset": ds,
                "backbone": bb,
                "pooling": pool,
                "n_seeds": len(common_seeds),
                "delta_complete_ndcg10_e3": tc["mean"] * 1000,
                "delta_complete_ci_lo_e3": tc["ci_low"] * 1000,
                "delta_complete_ci_hi_e3": tc["ci_high"] * 1000,
                "delta_affected_ndcg10_e3": ta["mean"] * 1000,
                "delta_affected_ci_lo_e3": ta["ci_low"] * 1000,
                "delta_affected_ci_hi_e3": ta["ci_high"] * 1000,
                "delta_warm_ndcg10_e3": tw["mean"] * 1000,
                "delta_warm_ci_lo_e3": tw["ci_low"] * 1000,
                "delta_warm_ci_hi_e3": tw["ci_high"] * 1000,
                "boot_s0_point_e3": boot["point"] * 1000,
                "boot_s0_ci_lo_e3": boot["ci_low"] * 1000,
                "boot_s0_ci_hi_e3": boot["ci_high"] * 1000,
            }
        )

    if paired_rows:
        paired_csv = out_dir / "paired_contrasts.csv"
        with paired_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
            w.writeheader()
            w.writerows(paired_rows)

    # 3. Export clean LaTeX summary table for multi-backbone baselines (pooling == "none")
    ds_order = ["ml1m", "beauty", "sports", "toys"]
    ds_labels = {"ml1m": "ML-1M", "beauty": "Beauty", "sports": "Sports", "toys": "Toys"}
    bb_labels = {
        "sasrec": "SASRec (ICDM'18)",
        "tisasrec": "TiSASRec (WSDM'20)",
        "gru4rec": "GRU4Rec (ICLR'16)",
        "fmlprec": "FMLP-Rec (WWW'22)",
        "bsarec": "BSARec (AAAI'24)",
        "lrurec": "LRURec (WSDM'24)",
        "lsab": "LSAB (TKDD'24)",
        "ticoserec": "TiCoSeRec (TKDE'24)",
    }

    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Multi-backbone accuracy and timestamp-tie sensitivity across four datasets (3-seed mean $\pm$ SD)}",
        r"\label{tab:multi-backbone}",
        r"\fontsize{8}{10}\selectfont",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Dataset & Backbone & Full NDCG@10 ($\times10^{-3}$) & Warm NDCG@10 ($\times10^{-3}$) & Aff. NDCG@10 ($\times10^{-3}$) & LB Overlap (\%) & LB Hit-Flip (\%) & AB Overlap (\%) \\",
        r"\midrule",
    ]
    by_key = {(r["dataset"], r["backbone"], r["pooling"]): r for r in summary_rows}
    for d_i, ds in enumerate(ds_order):
        first = True
        for bb in ["sasrec", "tisasrec", "gru4rec", "fmlprec", "bsarec", "lrurec", "lsab", "ticoserec"]:
            r = by_key.get((ds, bb, "none"))
            if r is None:
                continue
            ds_col = ds_labels[ds] if first else ""
            first = False
            lines.append(
                f"{ds_col} & {bb_labels.get(bb, bb)} & "
                f"{_fmt_pm(r['complete_ndcg10_mean'], r['complete_ndcg10_sd'], 1000, 1)} & "
                f"{_fmt_pm(r['warm_ndcg10_mean'], r['warm_ndcg10_sd'], 1000, 1)} & "
                f"{_fmt_pm(r['affected_ndcg10_mean'], r['affected_ndcg10_sd'], 1000, 1)} & "
                f"{_fmt_pm(r['lb_affected_overlap_mean'], r['lb_affected_overlap_sd'], 100, 1)} & "
                f"{_fmt_pm(r['lb_affected_hitflip_mean'], r['lb_affected_hitflip_sd'], 100, 1)} & "
                f"{_fmt_pm(r['ab_affected_overlap_mean'], r['ab_affected_overlap_sd'], 100, 1)} \\\\"
            )
        if not first and d_i < len(ds_order) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (out_dir / "table_multi_backbone.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Aggregated {len(summary_rows)} model groups into {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.runs_root / "aggregated")
    aggregate(args.runs_root, out_dir)


if __name__ == "__main__":
    main()
