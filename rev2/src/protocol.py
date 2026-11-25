"""Frozen dataset loading, event-time joins, All-Bucket permutation generator,
fast GPU-ready tensor materialization, and exact metric/stability contracts.

Uses only Python stdlib + NumPy + PyTorch (no pandas dependency) so it runs
identically on local Windows Python 3.13 and AutoDL Linux Python 3.10+.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .pooling import canonicalize_and_group

MAX_LEN = 50
TIME_SPAN = 256
TRAINING_SEEDS = (42, 2026, 3407)
PERMUTATION_SEEDS = (101, 202, 303, 404, 505)
METRICS = ("ndcg10", "hr10", "hr20", "recall50", "mrr", "mrr10", "mean_rank")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_contracts(project_root: Path) -> dict:
    path = project_root / "manifests" / "dataset_contracts.json"
    return json.loads(path.read_text(encoding="utf-8"))


def verify_dataset_hashes(
    tag: str, data_root: Path, project_root: Path, contracts: dict
) -> None:
    spec = contracts["datasets"][tag]
    expected = spec["sha256"]
    checks = {
        "item_map": project_root / spec["item_map"],
        "event_source": data_root / spec["event_source"],
        "train": data_root / spec["train"],
        "val": data_root / spec["val"],
        "test": data_root / spec["test"],
    }
    for ps, rel in spec["perm_files"].items():
        checks[f"perm_seed{ps}"] = data_root / rel
    for key, path in checks.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen file ({key}): {path}")
        actual = sha256_file(path)
        if actual != expected[key]:
            raise RuntimeError(f"SHA256 mismatch for {tag}/{key}: {path}")


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def time_features_from_buckets(buckets: Sequence[int]) -> Tuple[List[int], int]:
    """Visible-history-only exact integer normalization (matches X4 TiSASRec protocol)."""
    vals = [int(v) for v in buckets]
    origin = min(vals)
    offsets = [v - origin for v in vals]
    distinct = sorted(set(vals))
    scale = min((b - a for a, b in zip(distinct, distinct[1:])), default=1)
    return offsets, int(scale)


def build_time_matrix_gpu(
    token_offsets: torch.Tensor,
    token_scales: torch.Tensor,
    num_tokens: torch.Tensor,
    time_span: int = TIME_SPAN,
) -> torch.Tensor:
    """Compute exact clipped integer time matrix [B, L, L] on device."""
    B, L = token_offsets.shape
    diff = (token_offsets[:, :, None] - token_offsets[:, None, :]).abs()
    matrix = torch.div(diff, token_scales[:, None, None], rounding_mode="floor").clamp(max=time_span)
    valid = torch.arange(L, device=token_offsets.device).unsqueeze(0) < num_tokens.unsqueeze(1)
    matrix = matrix.masked_fill(~(valid[:, :, None] & valid[:, None, :]), 0)
    return matrix.long()


def deterministic_allbucket_seed(dataset: str, sample_id: str, perm_seed: int) -> int:
    payload = f"{dataset}|allbucket|{sample_id}|{int(perm_seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def apply_allbucket_permutation(
    items: List[str],
    raw_lines: List[int],
    buckets: List[int],
    dataset: str,
    sample_id: str,
    perm_seed: int,
) -> Tuple[List[str], List[int], List[int]]:
    """Permute EVERY contiguous multi-event timestamp bucket in visible history."""
    n = len(items)
    rng = np.random.default_rng(deterministic_allbucket_seed(dataset, sample_id, perm_seed))
    new_items = list(items)
    new_rls = list(raw_lines)
    new_bs = list(buckets)
    pos = 0
    while pos < n:
        end = pos + 1
        while end < n and buckets[end] == buckets[pos]:
            end += 1
        k = end - pos
        if k >= 2:
            order = rng.permutation(k).tolist()
            new_items[pos:end] = [items[pos + i] for i in order]
            new_rls[pos:end] = [raw_lines[pos + i] for i in order]
            new_bs[pos:end] = [buckets[pos + i] for i in order]
        pos = end
    # Strict invariant check: bucket chronology and joint multiset preserved.
    if new_bs != buckets or Counter(zip(items, raw_lines, buckets)) != Counter(
        zip(new_items, new_rls, new_bs)
    ):
        raise RuntimeError(f"All-bucket permutation invariant violated for {sample_id}")
    return new_items, new_rls, new_bs


class FrozenDatasetBundle:
    """In-memory audited representation of one frozen dataset (all splits + event times)."""

    def __init__(
        self,
        tag: str,
        data_root: Path,
        project_root: Path,
        verify_hashes: bool = True,
    ):
        self.tag = tag
        self.contracts = load_contracts(project_root)
        if tag not in self.contracts["datasets"]:
            raise ValueError(f"Unknown dataset tag '{tag}'")
        self.spec = self.contracts["datasets"][tag]
        self.dataset_name = self.spec["dataset"]
        self.expected = self.spec["expected"]

        if verify_hashes:
            verify_dataset_hashes(tag, data_root, project_root, self.contracts)

        # 1. Exact frozen item mapping (1..num_items, 0=pad)
        map_rows = _read_csv(project_root / self.spec["item_map"])
        self.item_map: Dict[str, int] = {r["raw_item_id"]: int(r["item_idx"]) for r in map_rows}
        self.idx_to_raw: Dict[int, str] = {v: k for k, v in self.item_map.items()}
        self.num_items = len(self.item_map)
        if self.num_items != self.expected["num_items"]:
            raise RuntimeError(f"{tag}: item map count mismatch")

        # 2. Training sequences sorted by exact dataset sort convention
        sort_mode = self.spec["sort_mode"]
        groups: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
        train_rows = 0
        self.train_positive_items = set()
        with (data_root / self.spec["train"]).open(encoding="utf-8-sig", newline="") as f:
            for order, r in enumerate(csv.DictReader(f)):
                raw_item = r["item_id"]
                if raw_item not in self.item_map:
                    raise RuntimeError(f"{tag}: unknown train item {raw_item}")
                tie_key = order if sort_mode == "source_order" else int(r["raw_line"])
                groups[r["user_id"]].append((int(r["time_bucket"]), tie_key, self.item_map[raw_item]))
                self.train_positive_items.add(raw_item)
                train_rows += 1

        self.train_sequences: List[Tuple[np.ndarray, np.ndarray]] = []
        for user in sorted(groups):
            recs = sorted(groups[user], key=lambda x: (x[0], x[1]))
            if len(recs) >= 2:
                items_arr = np.asarray([x[2] for x in recs], dtype=np.int64)
                buckets_arr = np.asarray([x[0] for x in recs], dtype=np.int64)
                self.train_sequences.append((items_arr, buckets_arr))

        self.train_positions = sum(len(it) - 1 for it, _ in self.train_sequences)
        if (
            train_rows != self.expected["train_rows"]
            or len(self.train_sequences) != self.expected["trainable_users"]
            or self.train_positions != self.expected["train_positions"]
        ):
            raise RuntimeError(
                f"{tag}: training coverage mismatch rows={train_rows}, "
                f"users={len(self.train_sequences)}, pos={self.train_positions}"
            )

        # 3. Validation, canonical test, and 5 sealed last-bucket permutation panels
        raw_panels: Dict[str, List[dict]] = {
            "validation": _read_csv(data_root / self.spec["val"]),
            "original": _read_csv(data_root / self.spec["test"]),
        }
        ref_ids = [r["sample_id"] for r in raw_panels["original"]]
        for ps in PERMUTATION_SEEDS:
            rows = _read_csv(data_root / self.spec["perm_files"][str(ps)])
            by_id = {r["sample_id"]: r for r in rows}
            if len(by_id) != len(ref_ids) or set(by_id) != set(ref_ids):
                raise RuntimeError(f"{tag}: perm_seed{ps} sample_id population mismatch")
            raw_panels[f"perm_seed{ps}"] = [by_id[sid] for sid in ref_ids]

        # 4. Event-time join against sealed file_order.csv (no target time ever fed to model)
        wanted = set()
        for rows in raw_panels.values():
            for r in rows:
                u = r["user_id"]
                for rl in json.loads(r["sequence_raw_lines"]):
                    wanted.add((u, int(rl)))
                wanted.add((u, int(r["target_raw_line"])))

        event_index: Dict[Tuple[str, int], Tuple[str, int]] = {}
        with (data_root / self.spec["event_source"]).open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                k = (r["user_id"], int(r["raw_line"]))
                if k in wanted:
                    event_index[k] = (r["item_id"], int(r["time_bucket"]))
        if len(event_index) != len(wanted):
            raise RuntimeError(f"{tag}: incomplete event-time join")

        self.panels: Dict[str, List[dict]] = {}
        for name, rows in raw_panels.items():
            parsed = []
            for r in rows:
                u = r["user_id"]
                raw_items = [str(x) for x in json.loads(r["sequence_items"])]
                raw_lines = [int(x) for x in json.loads(r["sequence_raw_lines"])]
                target_raw = str(r["target_item"])
                target_rl = int(r["target_raw_line"])
                if target_raw in raw_items or target_rl in raw_lines:
                    raise RuntimeError(f"{tag}/{name}: target leakage in {r['sample_id']}")
                buckets = []
                for it, rl in zip(raw_items, raw_lines):
                    ev_item, ev_b = event_index[(u, rl)]
                    if ev_item != it:
                        raise RuntimeError(f"{tag}/{name}: event item mismatch in {r['sample_id']}")
                    buckets.append(ev_b)
                t_item, t_bucket = event_index[(u, target_rl)]
                if t_item != target_raw or max(buckets) >= t_bucket:
                    raise RuntimeError(f"{tag}/{name}: strict target time violation in {r['sample_id']}")
                if any(a > b for a, b in zip(buckets, buckets[1:])):
                    raise RuntimeError(f"{tag}/{name}: non-monotone history buckets in {r['sample_id']}")
                # Actual trailing last-bucket size
                k = 1
                while k < len(buckets) and buckets[-1 - k] == buckets[-1]:
                    k += 1
                parsed.append(
                    {
                        "sample_id": r["sample_id"],
                        "user_id": u,
                        "target_item": target_raw,
                        "target_idx": self.item_map[target_raw],
                        "raw_items": raw_items,
                        "raw_lines": raw_lines,
                        "items": [self.item_map[it] for it in raw_items],
                        "buckets": buckets,
                        "last_bucket_size": k,
                        "is_warm": int(target_raw in self.train_positive_items),
                    }
                )
            self.panels[name] = parsed

        # 5. Last-bucket affected mask (exact match to original paper)
        orig = self.panels["original"]
        self.last_bucket_affected: List[int] = []
        for idx in range(len(orig)):
            seqs = {tuple(self.panels[f"perm_seed{ps}"][idx]["raw_items"]) for ps in PERMUTATION_SEEDS}
            self.last_bucket_affected.append(int(len(seqs) >= 2))
        if sum(self.last_bucket_affected) != self.expected["last_bucket_affected"]:
            raise RuntimeError(
                f"{tag}: last_bucket_affected mismatch {sum(self.last_bucket_affected)} vs {self.expected['last_bucket_affected']}"
            )

        # 6. Deterministic All-Bucket 5-Permutation panels + all_bucket_affected mask
        for ps in PERMUTATION_SEEDS:
            ab_rows = []
            for row in orig:
                n_items, n_rls, n_bs = apply_allbucket_permutation(
                    row["raw_items"],
                    row["raw_lines"],
                    row["buckets"],
                    self.dataset_name,
                    row["sample_id"],
                    ps,
                )
                ab_rows.append(
                    {
                        **row,
                        "raw_items": n_items,
                        "raw_lines": n_rls,
                        "items": [self.item_map[it] for it in n_items],
                        "buckets": n_bs,
                    }
                )
            self.panels[f"allbucket_seed{ps}"] = ab_rows

        self.all_bucket_affected: List[int] = []
        for idx in range(len(orig)):
            seqs = {
                tuple(self.panels[f"allbucket_seed{ps}"][idx]["raw_items"])
                for ps in PERMUTATION_SEEDS
            }
            self.all_bucket_affected.append(int(len(seqs) >= 2))

        self.warm_mask: List[int] = [row["is_warm"] for row in orig]
        if sum(self.warm_mask) != self.expected["warm_test_samples"]:
            raise RuntimeError(f"{tag}: warm_test_samples mismatch")

    def materialize_train_tensors(
        self,
        pooling_mode: str,
        need_time_features: bool = False,
        need_raw_buckets: bool = False,
        max_positions: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Pre-materialize all legal training positions into compact NumPy arrays
        suitable for zero-overhead GPU-resident batching.
        """
        total = self.train_positions if max_positions is None else min(self.train_positions, max_positions)
        seq_arr = np.zeros((total, MAX_LEN), dtype=np.int32)
        group_arr = np.full((total, MAX_LEN), -1, dtype=np.int8)
        num_tokens_arr = np.empty(total, dtype=np.int16)
        targets_arr = np.empty(total, dtype=np.int32)
        offsets_arr = np.zeros((total, MAX_LEN), dtype=np.int64) if need_time_features else None
        scales_arr = np.ones(total, dtype=np.int64) if need_time_features else None
        raw_buckets_arr = np.zeros((total, MAX_LEN), dtype=np.int64) if need_raw_buckets else None
        raw_len_arr = np.empty(total, dtype=np.int16) if need_raw_buckets else None

        cursor = 0
        for items, buckets in self.train_sequences:
            seq_len = len(items)
            for p in range(1, seq_len):
                if cursor >= total:
                    break
                start = max(0, p - MAX_LEN)
                h_items = items[start:p]
                h_buckets = buckets[start:p]
                n = p - start

                if pooling_mode == "none":
                    seq_arr[cursor, MAX_LEN - n :] = h_items
                    group_arr[cursor, MAX_LEN - n :] = np.arange(n, dtype=np.int8)
                    nt = n
                    tb = h_buckets
                else:
                    s, g, nt, tb_full = canonicalize_and_group(h_items, h_buckets, pooling_mode, MAX_LEN)
                    seq_arr[cursor] = s
                    group_arr[cursor] = g
                    tb = tb_full[:nt]

                num_tokens_arr[cursor] = nt
                targets_arr[cursor] = int(items[p])

                if need_time_features:
                    origin = int(tb[0])  # buckets are non-decreasing so tb[0] == min(tb)
                    off = tb - origin
                    offsets_arr[cursor, :nt] = off
                    if nt > 1:
                        diffs = np.diff(tb)
                        pos_diffs = diffs[diffs > 0]
                        scales_arr[cursor] = int(pos_diffs.min()) if len(pos_diffs) > 0 else 1
                    else:
                        scales_arr[cursor] = 1

                if need_raw_buckets:
                    raw_buckets_arr[cursor, MAX_LEN - n :] = h_buckets
                    raw_len_arr[cursor] = n

                cursor += 1
            if cursor >= total:
                break

        result: Dict[str, np.ndarray] = {
            "seq": seq_arr,
            "group_positions": group_arr,
            "num_tokens": num_tokens_arr,
            "targets": targets_arr,
        }
        if need_time_features:
            result["token_offsets"] = offsets_arr
            result["token_scales"] = scales_arr
        if need_raw_buckets:
            result["raw_buckets"] = raw_buckets_arr
            result["raw_lengths"] = raw_len_arr
        return result

    def materialize_panel_tensors(
        self,
        panel_name: str,
        pooling_mode: str,
        need_time_features: bool = False,
    ) -> Dict[str, np.ndarray]:
        rows = self.panels[panel_name]
        n_rows = len(rows)
        seq_arr = np.zeros((n_rows, MAX_LEN), dtype=np.int64)
        group_arr = np.full((n_rows, MAX_LEN), -1, dtype=np.int64)
        num_tokens_arr = np.empty(n_rows, dtype=np.int64)
        targets_arr = np.empty(n_rows, dtype=np.int64)
        seen_mask_arr = np.zeros((n_rows, MAX_LEN), dtype=np.int64)
        offsets_arr = np.zeros((n_rows, MAX_LEN), dtype=np.int64) if need_time_features else None
        scales_arr = np.ones(n_rows, dtype=np.int64) if need_time_features else None

        for i, row in enumerate(rows):
            h_items = np.asarray(row["items"], dtype=np.int64)
            h_buckets = np.asarray(row["buckets"], dtype=np.int64)
            s, g, nt, tb_full = canonicalize_and_group(h_items, h_buckets, pooling_mode, MAX_LEN)
            seq_arr[i] = s
            group_arr[i] = g
            num_tokens_arr[i] = nt
            targets_arr[i] = row["target_idx"]
            seen = list(set(row["items"]) - {row["target_idx"]})
            if seen:
                seen_mask_arr[i, : len(seen)] = seen
            if need_time_features:
                tb = tb_full[:nt]
                off, scale = time_features_from_buckets(tb.tolist())
                offsets_arr[i, :nt] = off
                scales_arr[i] = scale

        out = {
            "seq": seq_arr,
            "group_positions": group_arr,
            "num_tokens": num_tokens_arr,
            "targets": targets_arr,
            "seen_mask": seen_mask_arr,
        }
        if need_time_features:
            out["token_offsets"] = offsets_arr
            out["token_scales"] = scales_arr
        return out


# =============================================================================
# Metric & Stability Contracts (exact match to original X3/X4 protocol)
# =============================================================================
def metrics_from_ranks(ranks: Sequence[int]) -> dict:
    vals = [int(r) for r in ranks]
    if not vals:
        return {"samples": 0, **{k: None for k in METRICS}, "median_rank": None}
    n = len(vals)
    return {
        "samples": n,
        "ndcg10": math.fsum(1.0 / math.log2(r + 1) if r <= 10 else 0.0 for r in vals) / n,
        "hr10": sum(r <= 10 for r in vals) / n,
        "hr20": sum(r <= 20 for r in vals) / n,
        "recall50": sum(r <= 50 for r in vals) / n,
        "mrr": math.fsum(1.0 / r for r in vals) / n,
        "mrr10": math.fsum(1.0 / r if r <= 10 else 0.0 for r in vals) / n,
        "mean_rank": math.fsum(vals) / n,
        "median_rank": float(statistics.median(vals)),
    }


def stability_summary(
    five_predictions: List[List[dict]],
    subset_indices: Sequence[int],
) -> dict:
    if len(five_predictions) != 5:
        raise ValueError("Stability requires exactly 5 permutation prediction lists")
    if not subset_indices:
        return {"n": 0, "overlap10": None, "hitflip": None, "all_identical_frac": None}
    overlaps = []
    flips = []
    identicals = []
    for idx in subset_indices:
        rows = [preds[idx] for preds in five_predictions]
        tops = [row["top10_list"] for row in rows]
        ranks = [int(row["rank"]) for row in rows]
        pair_ov = (
            math.fsum(len(set(a) & set(b)) / 10.0 for a, b in itertools.combinations(tops, 2))
            / 10.0
        )
        hits = [r <= 10 for r in ranks]
        overlaps.append(pair_ov)
        flips.append(int(any(hits) and not all(hits)))
        identicals.append(int(all(t == tops[0] for t in tops[1:])))
    n = len(subset_indices)
    return {
        "n": n,
        "overlap10": math.fsum(overlaps) / n,
        "hitflip": sum(flips) / n,
        "all_identical_frac": sum(identicals) / n,
    }


def bucket_stratified_stability(
    five_predictions: List[List[dict]],
    original_rows: List[dict],
) -> Dict[str, dict]:
    """Stratify complete-panel Top-10 overlap by last_bucket_size: 1, 2, 3, 4, 5+."""
    strata: Dict[str, List[int]] = {"1": [], "2": [], "3": [], "4": [], "5+": []}
    for i, row in enumerate(original_rows):
        k = int(row["last_bucket_size"])
        key = str(k) if k <= 4 else "5+"
        strata[key].append(i)
    return {k: stability_summary(five_predictions, idxs) for k, idxs in strata.items()}
