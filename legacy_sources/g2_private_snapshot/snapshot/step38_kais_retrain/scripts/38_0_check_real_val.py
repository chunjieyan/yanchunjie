import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/root/autodl-tmp/time_uncertainty_rec")
DEVICE = torch.device("cuda")
MAX_LEN = 50
BATCH = 128

script = ROOT / "step38_kais_retrain/scripts/38_0_ml1m_control_masked.py"

tree = ast.parse(script.read_text(encoding="utf-8"))
cls = next(
    n for n in tree.body
    if isinstance(n, ast.ClassDef)
    and n.name == "RecentBucketSetR1SASRec"
)

ns = {"torch": torch, "nn": nn, "F": F, "math": math}
exec(
    compile(ast.Module(body=[cls], type_ignores=[]), str(script), "exec"),
    ns,
)
Model = ns["RecentBucketSetR1SASRec"]

ckpt_path = ROOT / (
    "runs/phase2_natural_timestamp/matched_control/"
    "sasrec/MovieLens-1M/file_order/seed42/"
    "matched_control_sasrec_best.pt"
)

try:
    ckpt = torch.load(
        ckpt_path, map_location="cpu", weights_only=True
    )
except TypeError:
    ckpt = torch.load(ckpt_path, map_location="cpu")

model = Model(
    num_items=ckpt["num_items"],
    max_len=ckpt["max_len"],
    hidden_size=ckpt["hidden_size"],
    num_heads=ckpt["num_heads"],
    num_blocks=ckpt["num_blocks"],
    dropout=ckpt["dropout"],
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

mapping_path = ROOT / (
    "runs/phase2_natural_timestamp/matched_control/"
    "sasrec/MovieLens-1M/file_order/seed42/item_id_map.csv"
)

mapping = pd.read_csv(mapping_path, dtype={"raw_item_id": str})
item_map = dict(zip(mapping.raw_item_id, mapping.item_idx))

val_path = ROOT / (
    "processed/recent_bucketset_r1/"
    "MovieLens-1M/val/r1_samples.csv"
)
df = pd.read_csv(val_path, dtype={"target_item": str})

sequences = []
groups_all = []
lengths = []
targets = []

for row in df.itertuples(index=False):
    history = [
        item_map[str(v)]
        for v in json.loads(row.sequence_items)
    ]

    n = len(history)
    if not (1 <= n <= MAX_LEN):
        raise RuntimeError(f"Invalid history length: {n}")

    target = item_map[str(row.target_item)]

    if target in history:
        raise RuntimeError("Target leaked into history")

    seq = np.zeros(MAX_LEN, dtype=np.int64)
    seq[-n:] = history

    groups = np.full(MAX_LEN, -1, dtype=np.int64)
    groups[-n:] = np.arange(n, dtype=np.int64)

    sequences.append(seq)
    groups_all.append(groups)
    lengths.append(n)
    targets.append(target)

sequences = np.stack(sequences)
groups_all = np.stack(groups_all)
lengths = np.asarray(lengths)
targets = np.asarray(targets)

def predict(seq, groups, token_count, mode):
    if mode == "masked":
        return model.predict_batch_masked(seq, token_count)

    result = torch.empty(
        (seq.size(0), ckpt["num_items"] + 1),
        device=DEVICE,
    )

    for n in torch.unique(token_count).tolist():
        ids = torch.where(token_count == n)[0]
        part = model.predict_group_unpadded(
            seq[ids], groups[ids], n
        )
        result[ids] = part

    return result

def evaluate_mode(mode):
    all_ranks = []
    all_top10 = []
    max_score_diff = 0.0

    with torch.inference_mode():
        for start in range(0, len(sequences), BATCH):
            end = min(start + BATCH, len(sequences))

            seq = torch.as_tensor(
                sequences[start:end], device=DEVICE
            )
            groups = torch.as_tensor(
                groups_all[start:end], device=DEVICE
            )
            token_count = torch.as_tensor(
                lengths[start:end], device=DEVICE
            )
            target = torch.as_tensor(
                targets[start:end], device=DEVICE
            )

            scores = predict(seq, groups, token_count, mode)
            scores = scores.clone()
            scores[:, 0] = -1e9

            # 与原 strict evaluate 保持一致的 seen-item mask
            for i in range(seq.size(0)):
                seen = torch.unique(seq[i])
                seen = seen[
                    (seen != 0) & (seen != target[i])
                ]
                scores[i, seen] = -1e9

            rows = torch.arange(seq.size(0), device=DEVICE)
            target_scores = scores[rows, target]

            ranks = (
                (scores > target_scores[:, None]).sum(dim=1)
                + 1
            )

            top10 = torch.topk(scores, 10, dim=1).indices

            all_ranks.extend(ranks.cpu().tolist())
            all_top10.extend(top10.cpu().tolist())

    return np.asarray(all_ranks), np.asarray(all_top10)

print("===== REAL VALIDATION CONSISTENCY =====")
print("Samples:", len(sequences))
print("Unique history lengths:", len(np.unique(lengths)))
print("Checkpoint epoch:", ckpt["epoch"])

ref_rank, ref_top10 = evaluate_mode("grouped")
new_rank, new_top10 = evaluate_mode("masked")

def summarize(rank):
    return {
        "NDCG@10": np.where(
            rank <= 10, 1 / np.log2(rank + 1), 0
        ).mean(),
        "HR@10": (rank <= 10).mean(),
        "MeanRank": rank.mean(),
    }

ref_metrics = summarize(ref_rank)
new_metrics = summarize(new_rank)

print()
for key in ref_metrics:
    print(
        f"{key}: "
        f"Grouped={ref_metrics[key]:.9f} "
        f"Masked={new_metrics[key]:.9f}"
    )

rank_mismatch = int(np.count_nonzero(ref_rank != new_rank))

top10_set_mismatch = sum(
    set(a.tolist()) != set(b.tolist())
    for a, b in zip(ref_top10, new_top10)
)

print()
print("Rank mismatches:", rank_mismatch)
print("Top10 set mismatches:", top10_set_mismatch)
print("Max rank difference:", np.abs(ref_rank - new_rank).max())

passed = rank_mismatch == 0 and top10_set_mismatch == 0

print(
    "REAL VALIDATION CONSISTENCY:",
    "PASS" if passed else "NEEDS INVESTIGATION"
)

if not passed:
    raise SystemExit(1)
