"""Audit frozen datasets in D:\\time_uncertainty_rec and build manifests/dataset_contracts.json."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


DATASET_SPECS = {
    "ml1m": {
        "dataset": "MovieLens-1M",
        "native_time_unit": "Unix second",
        "sort_mode": "source_order",
        "event_source": "processed/diagnostic_a/MovieLens-1M/file_order.csv",
        "train": "processed/diagnostic_a_splits/MovieLens-1M/file_order/train.csv",
        "val": "processed/recent_bucketset_r1/MovieLens-1M/val/r1_samples.csv",
        "test": "processed/recent_bucketset_r1/MovieLens-1M/test/r1_samples.csv",
        "perm_dir": "processed/strict_last_bucket_only/MovieLens-1M",
        "source_item_map": "g2_lbsp_ml1m_20261023/snapshot/references/ml1m_seed42/item_id_map.csv",
    },
    "beauty": {
        "dataset": "Amazon_Beauty",
        "native_time_unit": "Unix day (timestamp // 86400)",
        "sort_mode": "source_order",
        "event_source": "processed/diagnostic_a/Amazon_Beauty/file_order.csv",
        "train": "processed/diagnostic_a_splits/Amazon_Beauty/file_order/train.csv",
        "val": "processed/recent_bucketset_r1/Amazon_Beauty/val/r1_samples.csv",
        "test": "processed/recent_bucketset_r1/Amazon_Beauty/test/r1_samples.csv",
        "perm_dir": "processed/strict_last_bucket_only/Amazon_Beauty",
        "source_item_map": "g2_lbsp_amazon_stability_20261025/snapshot/references/beauty_seed42/item_id_map.csv",
    },
    "sports": {
        "dataset": "Amazon_Sports",
        "native_time_unit": "Unix day (timestamp // 86400)",
        "sort_mode": "raw_line",
        "event_source": "processed/diagnostic_a/Amazon_Sports/file_order.csv",
        "train": "processed/diagnostic_a_splits/Amazon_Sports/file_order/train.csv",
        "val": "processed/strict_diagnostic_a/Amazon_Sports/file_order/val.csv",
        "test": "processed/strict_diagnostic_a/Amazon_Sports/file_order/test.csv",
        "perm_dir": "processed/strict_last_bucket_only/Amazon_Sports",
        "source_item_map": "g2_lbsp_amazon_stability_20261025/snapshot/references/sports_seed42/item_id_map.csv",
    },
    "toys": {
        "dataset": "Amazon_Toys",
        "native_time_unit": "Unix day (timestamp // 86400)",
        "sort_mode": "raw_line",
        "event_source": "processed/diagnostic_a/Amazon_Toys/file_order.csv",
        "train": "processed/diagnostic_a_splits/Amazon_Toys/file_order/train.csv",
        "val": "processed/strict_diagnostic_a/Amazon_Toys/file_order/val.csv",
        "test": "processed/strict_diagnostic_a/Amazon_Toys/file_order/test.csv",
        "perm_dir": "processed/strict_last_bucket_only/Amazon_Toys",
        "source_item_map": "g2_lbsp_amazon_stability_20261025/snapshot/references/toys_seed42/item_id_map.csv",
    },
}


def build_contracts(data_root: Path, project_root: Path) -> dict:
    map_dir = project_root / "manifests" / "item_maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    contracts = {
        "version": "kais_revision_v2_contracts",
        "training_seeds": [42, 2026, 3407],
        "permutation_seeds": [101, 202, 303, 404, 505],
        "datasets": {},
    }

    for tag, spec in DATASET_SPECS.items():
        src_map = data_root / spec["source_item_map"]
        dst_map = map_dir / f"{tag}_item_id_map.csv"
        shutil.copyfile(src_map, dst_map)

        with dst_map.open(encoding="utf-8-sig", newline="") as f:
            imap = {r["raw_item_id"]: int(r["item_idx"]) for r in csv.DictReader(f)}
        assert set(imap.values()) == set(range(1, len(imap) + 1)), f"Invalid map for {tag}"

        users = defaultdict(int)
        train_rows = 0
        train_pos_items = set()
        with (data_root / spec["train"]).open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                users[r["user_id"]] += 1
                train_pos_items.add(r["item_id"])
                train_rows += 1
        trainable_users = sum(1 for c in users.values() if c >= 2)
        train_positions = sum(c - 1 for c in users.values() if c >= 2)

        with (data_root / spec["val"]).open(encoding="utf-8-sig", newline="") as f:
            val_rows = list(csv.DictReader(f))
        with (data_root / spec["test"]).open(encoding="utf-8-sig", newline="") as f:
            test_rows = list(csv.DictReader(f))

        perm_files = {}
        perm_by_seed = {}
        for ps in (101, 202, 303, 404, 505):
            rel = f"{spec['perm_dir']}/perm_seed{ps}/test.csv"
            perm_files[str(ps)] = rel
            with (data_root / rel).open(encoding="utf-8-sig", newline="") as f:
                perm_by_seed[ps] = {r["sample_id"]: r["sequence_items"] for r in csv.DictReader(f)}

        affected = 0
        warm_test = 0
        warm_affected = 0
        for r in test_rows:
            sid = r["sample_id"]
            distinct = len({perm_by_seed[ps][sid] for ps in (101, 202, 303, 404, 505)})
            is_aff = distinct >= 2
            is_warm = r["target_item"] in train_pos_items
            affected += int(is_aff)
            warm_test += int(is_warm)
            warm_affected += int(is_aff and is_warm)

        # Join visible test events to file_order.csv to count histories with ANY multi-event bucket.
        wanted = set()
        for r in test_rows:
            u = r["user_id"]
            for rl in json.loads(r["sequence_raw_lines"]):
                wanted.add((u, int(rl)))
        ev_bucket = {}
        with (data_root / spec["event_source"]).open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                k = (r["user_id"], int(r["raw_line"]))
                if k in wanted:
                    ev_bucket[k] = int(r["time_bucket"])
        assert len(ev_bucket) == len(wanted), f"Incomplete event join for {tag}"

        any_multi_bucket = 0
        any_distinct_multi_bucket = 0
        for r in test_rows:
            u = r["user_id"]
            items = json.loads(r["sequence_items"])
            rls = json.loads(r["sequence_raw_lines"])
            bs = [ev_bucket[(u, int(rl))] for rl in rls]
            by_b = defaultdict(list)
            for it, b in zip(items, bs):
                by_b[b].append(it)
            if any(len(v) >= 2 for v in by_b.values()):
                any_multi_bucket += 1
            if any(len(set(v)) >= 2 for v in by_b.values()):
                any_distinct_multi_bucket += 1

        hashes = {
            "item_map": sha256_file(dst_map),
            "event_source": sha256_file(data_root / spec["event_source"]),
            "train": sha256_file(data_root / spec["train"]),
            "val": sha256_file(data_root / spec["val"]),
            "test": sha256_file(data_root / spec["test"]),
        }
        for ps, rel in perm_files.items():
            hashes[f"perm_seed{ps}"] = sha256_file(data_root / rel)

        entry = {
            "dataset": spec["dataset"],
            "native_time_unit": spec["native_time_unit"],
            "sort_mode": spec["sort_mode"],
            "item_map": f"manifests/item_maps/{tag}_item_id_map.csv",
            "event_source": spec["event_source"],
            "train": spec["train"],
            "val": spec["val"],
            "test": spec["test"],
            "perm_files": perm_files,
            "expected": {
                "num_items": len(imap),
                "train_rows": train_rows,
                "trainable_users": trainable_users,
                "train_positions": train_positions,
                "val_samples": len(val_rows),
                "test_samples": len(test_rows),
                "last_bucket_affected": affected,
                "warm_test_samples": warm_test,
                "warm_last_bucket_affected": warm_affected,
                "any_multi_bucket_test": any_multi_bucket,
                "any_distinct_multi_bucket_test": any_distinct_multi_bucket,
            },
            "sha256": hashes,
        }
        contracts["datasets"][tag] = entry
        print(f"[{tag}] {json.dumps(entry['expected'])}", flush=True)

    out_path = project_root / "manifests" / "dataset_contracts.json"
    out_path.write_text(json.dumps(contracts, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return contracts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\time_uncertainty_rec"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    build_contracts(args.data_root, args.project_root)


if __name__ == "__main__":
    main()
