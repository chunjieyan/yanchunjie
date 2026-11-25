"""Verify exact weight-loading & forward-pass equivalence between original legacy
model classes (SASRecControl, RecentBucketSetR1SASRec, TiSASRecModel) and
UnifiedSASRec / UnifiedTiSASRec via load_legacy_state_dict.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_legacy_checkpoints import load_legacy_state_dict
from src.models import build_model
from src.pooling import canonicalize_and_group
from src.protocol import MAX_LEN, build_time_matrix_gpu, time_features_from_buckets

LEGACY_DIR = Path(
    r"D:\time_uncertainty_rec\kais_revision_20261102\GitHub_v101_check_20261103"
    r"\legacy_sources\kais_revision_outline_20261022\server_g2"
)


def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    x6 = _load_mod(LEGACY_DIR / "x6_model_definitions.py", "legacy_x6")
    x4 = _load_mod(LEGACY_DIR / "x4_tisasrec_model.py", "legacy_x4")

    num_items = 80
    histories = [[3, 7, 11, 2, 9, 5], [4, 8, 12, 15], [10]]
    buckets_list = [[1, 1, 2, 5, 5, 5], [10, 20, 20, 30], [42]]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)

        # 1. SASRecControl -> UnifiedSASRec(pooling_mode="none")
        torch.manual_seed(42)
        orig_ctrl = x6.SASRecControl(num_items, MAX_LEN, 64, 2, 2, 0.2).eval()
        pt = tmp_p / "ctrl.pt"
        torch.save({"model_state_dict": orig_ctrl.state_dict()}, pt)

        uni_ctrl = build_model("sasrec", num_items, pooling_mode="none", max_len=MAX_LEN).eval()
        uni_ctrl.load_state_dict(load_legacy_state_dict(pt, "sasrec", torch.device("cpu")), strict=True)

        seqs, grps, nts = [], [], []
        for h, b in zip(histories, buckets_list):
            s, g, n, _ = canonicalize_and_group(np.array(h), np.array(b), "none", MAX_LEN)
            seqs.append(s); grps.append(g); nts.append(n)
        seq_t = torch.from_numpy(np.stack(seqs))
        grp_t = torch.from_numpy(np.stack(grps))
        nt_t = torch.tensor(nts, dtype=torch.long)

        with torch.no_grad():
            s_orig = orig_ctrl.predict_batch_masked(seq_t, nt_t)
            s_uni = uni_ctrl.score_all(seq_t, grp_t, nt_t)
        err_ctrl = (s_orig - s_uni).abs().max().item()
        assert err_ctrl < 1e-5, f"SASRecControl mismatch: {err_ctrl}"
        print(f"[PASS] Legacy SASRecControl -> UnifiedSASRec exact match (max_err={err_ctrl:.2e})")

        # 2. RecentBucketSetR1SASRec -> UnifiedSASRec(pooling_mode="lbsp_mean")
        torch.manual_seed(2026)
        orig_lbsp = x6.RecentBucketSetR1SASRec(num_items, MAX_LEN, 64, 2, 2, 0.2).eval()
        pt = tmp_p / "lbsp.pt"
        torch.save({"model_state_dict": orig_lbsp.state_dict()}, pt)

        uni_lbsp = build_model("sasrec", num_items, pooling_mode="lbsp_mean", max_len=MAX_LEN).eval()
        uni_lbsp.load_state_dict(load_legacy_state_dict(pt, "sasrec", torch.device("cpu")), strict=True)

        seqs, grps, nts = [], [], []
        for h, b in zip(histories, buckets_list):
            s, g, n, _ = canonicalize_and_group(np.array(h), np.array(b), "lbsp_mean", MAX_LEN)
            seqs.append(s); grps.append(g); nts.append(n)
        seq_t = torch.from_numpy(np.stack(seqs))
        grp_t = torch.from_numpy(np.stack(grps))
        nt_t = torch.tensor(nts, dtype=torch.long)

        with torch.no_grad():
            s_orig = orig_lbsp.predict_batch_masked(seq_t, grp_t, nt_t)
            s_uni = uni_lbsp.score_all(seq_t, grp_t, nt_t)
        err_lbsp = (s_orig - s_uni).abs().max().item()
        assert err_lbsp < 1e-5, f"RecentBucketSetR1SASRec mismatch: {err_lbsp}"
        print(f"[PASS] Legacy RecentBucketSetR1SASRec -> UnifiedSASRec(lbsp_mean) exact match (max_err={err_lbsp:.2e})")

        # 3. TiSASRecModel -> UnifiedTiSASRec(pooling_mode="none")
        torch.manual_seed(3407)
        orig_ti = x4.TiSASRecModel(num_items, hidden_size=64, max_len=MAX_LEN, num_heads=2, num_layers=2, dropout=0.2).eval()
        pt = tmp_p / "tisas.pt"
        torch.save({"state_dict": orig_ti.state_dict()}, pt)

        uni_ti = build_model("tisasrec", num_items, pooling_mode="none", max_len=MAX_LEN).eval()
        uni_ti.load_state_dict(load_legacy_state_dict(pt, "tisasrec", torch.device("cpu")), strict=True)

        # Build left-aligned item_seq for orig_ti vs right-aligned seq + group_positions for uni_ti
        left_seq = torch.zeros((len(histories), MAX_LEN), dtype=torch.long)
        seqs, grps, nts, offs, scls = [], [], [], [], []
        for i, (h, b) in enumerate(zip(histories, buckets_list)):
            left_seq[i, : len(h)] = torch.tensor(h, dtype=torch.long)
            s, g, n, tb = canonicalize_and_group(np.array(h), np.array(b), "none", MAX_LEN)
            o, sc = time_features_from_buckets(tb[:n].tolist())
            off_pad = np.zeros(MAX_LEN, dtype=np.int64)
            off_pad[:n] = o
            seqs.append(s); grps.append(g); nts.append(n); offs.append(off_pad); scls.append(sc)
        seq_t = torch.from_numpy(np.stack(seqs))
        grp_t = torch.from_numpy(np.stack(grps))
        nt_t = torch.tensor(nts, dtype=torch.long)
        tm_t = build_time_matrix_gpu(torch.from_numpy(np.stack(offs)), torch.tensor(scls, dtype=torch.long), nt_t)

        with torch.no_grad():
            s_orig = orig_ti.score_all(left_seq, nt_t, tm_t)
            s_uni = uni_ti.score_all(seq_t, grp_t, nt_t, tm_t)
        err_ti = (s_orig - s_uni).abs().max().item()
        assert err_ti < 1e-5, f"TiSASRecModel mismatch: {err_ti}"
        print(f"[PASS] Legacy TiSASRecModel -> UnifiedTiSASRec exact match (max_err={err_ti:.2e})")

    print("ALL LEGACY CHECKPOINT COMPATIBILITY CHECKS PASSED!")


if __name__ == "__main__":
    main()
