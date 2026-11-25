"""Extract the exact server checkpoint paths, SHA256 hashes, and published canonical
NDCG@10 references for all 42 original models into manifests/legacy_42_checkpoints.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_legacy_manifest(data_root: Path, project_root: Path) -> dict:
    gh_root = data_root / "kais_revision_20261102" / "GitHub_v101_check_20261103"
    mm = _read_json(gh_root / "config" / "model_matrix.json")

    # Published canonical full-panel NDCG@10 for exact reproduction verification
    published_ndcg10 = {}
    with (gh_root / "aggregates" / "statistics" / "per_seed_metrics.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        for r in csv.DictReader(f):
            if (
                r["population"] == "full"
                and r["accuracy_view"] == "canonical"
                and r["metric"] == "ndcg10"
            ):
                published_ndcg10[(r["dataset"], r["method"], int(r["training_seed"]))] = float(
                    r["value"]
                )

    entries = []
    for row in mm:
        ds = row["dataset"]
        meth = row["method"]
        seed = int(row["training_seed"])
        lin = row["lineage"]
        eval_dir = data_root / row["evaluation_output_directory"]

        ckpt_path = None
        ckpt_sha = None

        meta_p = eval_dir / "metadata.json"
        cfg_p = eval_dir / "config.json"

        if meta_p.is_file():
            m = _read_json(meta_p)
            ckpt_path = m.get("checkpoint_path")
            ckpt_sha = m.get("checkpoint_sha256")

        if cfg_p.is_file():
            c = _read_json(cfg_p)
            job = c.get("job", {})
            if not ckpt_sha:
                ckpt_sha = job.get("checkpoint_sha256") or c.get("checkpoint_sha256")
            if not ckpt_path and job.get("model_dir") and job.get("checkpoint"):
                ckpt_path = f"/root/autodl-tmp/time_uncertainty_rec/{job['model_dir']}/{job['checkpoint']}"
            if not ckpt_path and c.get("model_dir") and c.get("checkpoint"):
                ckpt_path = f"/root/autodl-tmp/time_uncertainty_rec/{c['model_dir']}/{c['checkpoint']}"

        # Sibling verification / manifest files (for TiSASRec and X3 Shuffle SHA256)
        stage_snap = eval_dir.parent.parent
        ver_dir = stage_snap / "verification"
        if ver_dir.is_dir():
            for vf in sorted(ver_dir.glob("completed_*.json")):
                name = vf.name.lower()
                if ds in name and str(seed) in name:
                    if meth == "TiSASRec" or meth.lower() in name:
                        vo = _read_json(vf)
                        if not ckpt_sha:
                            ckpt_sha = vo.get("checkpoint_sha256")

        if not ckpt_path and meth == "TiSASRec":
            man_p = stage_snap / "manifest.json"
            if man_p.is_file():
                man = _read_json(man_p)
                if man.get("stage"):
                    ckpt_path = f"{man['stage']}/outputs/{ds}_seed{seed}/tisasrec_best.pt"

        # Map legacy method name -> unified (backbone, pooling_mode, tag)
        if meth == "Control":
            backbone, pooling_mode = "sasrec", "none"
        elif meth == "LBSP":
            backbone, pooling_mode = "sasrec", "lbsp_mean"
        elif meth == "Shuffle":
            backbone, pooling_mode = "sasrec_shuffle", "none"
        elif meth == "TiSASRec":
            backbone, pooling_mode = "tisasrec", "none"
        else:
            raise ValueError(f"Unknown legacy method: {meth}")

        entries.append(
            {
                "dataset": ds,
                "legacy_method": meth,
                "backbone": backbone,
                "pooling_mode": pooling_mode,
                "seed": seed,
                "lineage": lin,
                "training_job_id": row["training_job_id"],
                "server_checkpoint_path": ckpt_path,
                "checkpoint_sha256": ckpt_sha,
                "published_canonical_ndcg10": published_ndcg10.get((ds, meth, seed)),
            }
        )

    missing_path = [e["training_job_id"] for e in entries if not e["server_checkpoint_path"]]
    if missing_path:
        raise RuntimeError(f"Missing server_checkpoint_path for: {missing_path}")

    out_obj = {
        "version": "legacy_42_checkpoints_v1",
        "count": len(entries),
        "default_server_root": "/root/autodl-tmp/time_uncertainty_rec",
        "models": entries,
    }
    out_path = project_root / "manifests" / "legacy_42_checkpoints.json"
    out_path.write_text(json.dumps(out_obj, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {out_path} ({len(entries)}/42 paths resolved, "
        f"{sum(1 for e in entries if e['checkpoint_sha256'])}/42 with SHA256)",
        flush=True,
    )
    return out_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\time_uncertainty_rec"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    build_legacy_manifest(args.data_root, args.project_root)


if __name__ == "__main__":
    main()
