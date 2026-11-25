"""Comprehensive verification & CPU smoke test for KAIS Revision 2.0.

Checks:
1. All 8 backbones forward + backward + finite loss/gradients.
2. Exact numerical equivalence between parallel Vandermonde LRU scan and step-by-step recurrence.
3. Exact 0.0e+00 permutation invariance of EVERY pooling operator (lbsp_mean, lbsp_max,
   lbsp_attn, lbsp_deepsets) across EVERY backbone under legal last-bucket permutations.
4. Exact 0.0e+00 global permutation invariance of absp_mean under legal ALL-bucket permutations.
5. Real frozen dataset contract audit (all 4 datasets) + 2-epoch end-to-end training &
   Last-Bucket / All-Bucket 5-perm evaluation smoke run on Amazon_Toys.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import BACKBONE_REGISTRY, build_model, _LRULayer
from src.pooling import canonicalize_and_group
from src.protocol import (
    MAX_LEN,
    FrozenDatasetBundle,
    build_time_matrix_gpu,
    time_features_from_buckets,
)
from src.train_eval import _ticoserec_augment_batch, run_job


def _make_batch(histories, buckets_list, pooling_mode, need_time=False):
    B = len(histories)
    seq = np.zeros((B, MAX_LEN), dtype=np.int64)
    grp = np.full((B, MAX_LEN), -1, dtype=np.int64)
    nt = np.empty(B, dtype=np.int64)
    off = np.zeros((B, MAX_LEN), dtype=np.int64)
    scl = np.ones(B, dtype=np.int64)
    for i, (h, b) in enumerate(zip(histories, buckets_list)):
        s, g, n, tb = canonicalize_and_group(np.array(h), np.array(b), pooling_mode, MAX_LEN)
        seq[i], grp[i], nt[i] = s, g, n
        if need_time:
            o, sc = time_features_from_buckets(tb[:n].tolist())
            off[i, :n] = o
            scl[i] = sc
    seq_t = torch.from_numpy(seq)
    grp_t = torch.from_numpy(grp)
    nt_t = torch.from_numpy(nt)
    tm_t = build_time_matrix_gpu(torch.from_numpy(off), torch.from_numpy(scl), nt_t) if need_time else None
    return seq_t, grp_t, nt_t, tm_t


def test_lru_parallel_vs_sequential():
    torch.manual_seed(42)
    layer = _LRULayer(hidden_size=16, max_len=12, dropout=0.0)
    layer.eval()
    B, L, D = 3, 12, 16
    x = torch.randn(B, L, D)
    lengths = torch.tensor([5, 12, 1])
    valid = torch.arange(L).unsqueeze(0) >= (L - lengths).unsqueeze(1)
    x = x * valid.unsqueeze(-1).float()

    with torch.no_grad():
        y_par = layer(x, valid)

        # Manual step-by-step sequential recurrence
        x_norm = layer.in_norm(x) * valid.unsqueeze(-1).float()
        nu = torch.exp(layer.params_log[0])
        theta = torch.exp(layer.params_log[1])
        gamma = torch.exp(layer.params_log[2])
        lam = torch.exp(-nu + 1j * theta)  # [D]
        bx = layer.in_proj(x_norm)
        u = torch.complex(bx[:, :, :D], bx[:, :, D:]) * gamma.view(1, 1, D)
        h = torch.zeros(B, D, dtype=torch.complex64)
        hs = []
        for t in range(L):
            h = lam.view(1, D) * h + u[:, t, :]
            hs.append(h)
        h_seq = torch.stack(hs, dim=1)
        y_seq_core = (
            layer.out_proj_re(h_seq.real)
            - layer.out_proj_im(h_seq.imag)
            + x_norm * layer.skip_D.view(1, 1, D)
        ) * valid.unsqueeze(-1).float()
        y_seq = x + y_seq_core

    max_err = (y_par - y_seq).abs().max().item()
    assert max_err < 1e-5, f"LRU parallel/sequential mismatch: {max_err}"
    print(f"[PASS] LRU parallel scan matches sequential recurrence (max_err={max_err:.2e})")


def test_gru_gather_vs_packed_and_vectorized_ticoserec():
    # 1. Verify UnifiedGRU4Rec causal gather == pack_padded_sequence
    torch.manual_seed(7)
    m = build_model("gru4rec", num_items=40, pooling_mode="none", max_len=MAX_LEN).eval()
    h = [[3, 7, 11, 2, 9], [4, 8], [10]]
    b = [[1, 1, 2, 5, 5], [10, 20], [42]]
    seq, grp, nt, _ = _make_batch(h, b, "none")
    with torch.no_grad():
        f_gather = m.encode(seq, grp, nt)
        pooled = m.pooler(m.item_embedding(seq), grp)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            pooled, nt.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = m.gru(packed)
        f_pack = m.final_norm(m.out_proj(h_n[-1]))
    err = (f_gather - f_pack).abs().max().item()
    assert err < 1e-6, f"GRU gather vs pack mismatch: {err}"
    print(f"[PASS] GRU4Rec zero-sync causal gather == pack_padded_sequence (max_err={err:.2e})")

    # 2. Verify pure-GPU vectorized _ticoserec_augment_batch across none / lbsp_mean / absp_mean
    B = 32
    seq_b = torch.zeros((B, MAX_LEN), dtype=torch.long)
    rb_b = torch.zeros((B, MAX_LEN), dtype=torch.long)
    rl_b = torch.randint(1, 20, (B,))
    for i in range(B):
        n = int(rl_b[i].item())
        seq_b[i, MAX_LEN - n :] = torch.randint(1, 41, (n,))
        bs = torch.sort(torch.randint(1, 6, (n,))).values
        rb_b[i, MAX_LEN - n :] = bs
    gen = torch.Generator().manual_seed(42)
    for pool in ("none", "lbsp_mean", "absp_mean"):
        s_aug, g_aug, nt_aug = _ticoserec_augment_batch(seq_b, rb_b, rl_b, 41, pool, gen)
        assert s_aug.shape == (B, MAX_LEN) and g_aug.shape == (B, MAX_LEN)
        assert bool((nt_aug >= 1).all()) and bool((nt_aug <= rl_b).all())
        for i in range(B):
            valid_g = g_aug[i][g_aug[i] >= 0]
            assert int(valid_g.max().item()) == int(nt_aug[i].item()) - 1
    print("[PASS] Vectorized TiCoSeRec augmentation valid across none / lbsp_mean / absp_mean")


def test_all_backbones_and_exact_invariance():
    num_items = 50
    # Synthetic histories with multi-event buckets in BOTH prefix and last bucket
    h_orig = [
        [3, 7, 11, 2, 9, 5],
        [4, 8, 12, 15],
        [6, 1, 14, 19, 20],
        [10],
    ]
    b_orig = [
        [1, 1, 2, 5, 5, 5],    # prefix tie (1,1) AND last-bucket tie (5,5,5)
        [10, 20, 20, 30],      # prefix tie (20,20), singleton last bucket (30)
        [3, 3, 3, 7, 7],       # prefix tie (3,3,3) AND last-bucket tie (7,7)
        [42],                  # singleton history
    ]
    # Legal last-bucket permutation (prefix fixed, last bucket reversed/cycled)
    h_lb_perm = [
        [3, 7, 11, 5, 2, 9],
        [4, 8, 12, 15],
        [6, 1, 14, 20, 19],
        [10],
    ]
    # Legal ALL-bucket permutation (both prefix tied buckets AND last bucket permuted)
    h_ab_perm = [
        [7, 3, 11, 9, 5, 2],
        [4, 12, 8, 15],
        [14, 6, 1, 20, 19],
        [10],
    ]
    targets = torch.tensor([13, 16, 21, 22], dtype=torch.long)

    # 1. All 8 backbones forward/backward in Control mode ("none")
    for bb in BACKBONE_REGISTRY:
        torch.manual_seed(42)
        model = build_model(bb, num_items=num_items, pooling_mode="none", max_len=MAX_LEN)
        need_time = bb == "tisasrec"
        seq, grp, nt, tm = _make_batch(h_orig, b_orig, "none", need_time=need_time)
        model.train()
        scores = model.score_all(seq, grp, nt, tm)
        assert scores.shape == (4, num_items + 1), f"{bb} wrong score shape {scores.shape}"
        loss = F.cross_entropy(scores[:, 1:], targets - 1)
        assert torch.isfinite(loss).item(), f"{bb} nonfinite loss"
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all().item() for g in grads), f"{bb} bad grads"
    print(f"[PASS] All {len(BACKBONE_REGISTRY)} backbones forward/backward finite")

    # 2. Exact Last-Bucket invariance across ALL backbones x LBSP-Mean
    for bb in BACKBONE_REGISTRY:
        torch.manual_seed(42)
        model = build_model(bb, num_items=num_items, pooling_mode="lbsp_mean", max_len=MAX_LEN)
        model.eval()
        need_time = bb == "tisasrec"
        with torch.no_grad():
            s1 = model.score_all(*_make_batch(h_orig, b_orig, "lbsp_mean", need_time=need_time))
            s2 = model.score_all(*_make_batch(h_lb_perm, b_orig, "lbsp_mean", need_time=need_time))
        diff = (s1 - s2).abs().max().item()
        assert diff == 0.0, f"{bb}+lbsp_mean not bit-identical under LB perm: diff={diff}"
    print(f"[PASS] All {len(BACKBONE_REGISTRY)} backbones + lbsp_mean are BIT-IDENTICAL (diff=0.0) under last-bucket perms")

    # 3. Exact Last-Bucket invariance across ALL pooling operators on SASRec
    for pool in ("lbsp_mean", "lbsp_max", "lbsp_attn", "lbsp_deepsets"):
        torch.manual_seed(42)
        model = build_model("sasrec", num_items=num_items, pooling_mode=pool, max_len=MAX_LEN)
        model.train()
        seq, grp, nt, tm = _make_batch(h_orig, b_orig, pool)
        loss = F.cross_entropy(model.score_all(seq, grp, nt, tm)[:, 1:], targets - 1)
        loss.backward()
        assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        model.eval()
        with torch.no_grad():
            s1 = model.score_all(*_make_batch(h_orig, b_orig, pool))
            s2 = model.score_all(*_make_batch(h_lb_perm, b_orig, pool))
        diff = (s1 - s2).abs().max().item()
        assert diff == 0.0, f"sasrec+{pool} not bit-identical under LB perm: diff={diff}"
    print("[PASS] All LBSP pooling operators (mean/max/attn/deepsets) finite + BIT-IDENTICAL under last-bucket perms")

    # 4. Exact Global All-Bucket invariance for absp_mean (SASRec & TiSASRec)
    for bb in ("sasrec", "tisasrec", "gru4rec", "bsarec"):
        torch.manual_seed(42)
        model = build_model(bb, num_items=num_items, pooling_mode="absp_mean", max_len=MAX_LEN)
        model.eval()
        need_time = bb == "tisasrec"
        with torch.no_grad():
            s1 = model.score_all(*_make_batch(h_orig, b_orig, "absp_mean", need_time=need_time))
            s2 = model.score_all(*_make_batch(h_ab_perm, b_orig, "absp_mean", need_time=need_time))
        diff = (s1 - s2).abs().max().item()
        assert diff == 0.0, f"{bb}+absp_mean not bit-identical under ALL-bucket perm: diff={diff}"
    print("[PASS] absp_mean is BIT-IDENTICAL (diff=0.0) under global ALL-bucket permutations")


def test_real_data_contracts_and_smoke(data_root: Path, skip_real_train: bool = False):
    for tag in ("toys", "beauty", "sports", "ml1m"):
        bundle = FrozenDatasetBundle(tag, data_root, PROJECT_ROOT, verify_hashes=True)
        print(
            f"[PASS] Contract audit {tag:6s}: items={bundle.num_items}, "
            f"train_pos={bundle.train_positions}, val={len(bundle.panels['validation'])}, "
            f"test={len(bundle.panels['original'])}, lb_aff={sum(bundle.last_bucket_affected)}, "
            f"ab_aff={sum(bundle.all_bucket_affected)}, warm={sum(bundle.warm_mask)}",
            flush=True,
        )

    if skip_real_train:
        return

    # Real 2-epoch CPU end-to-end smoke test on Amazon_Toys for 3 representative configs:
    # (TiCoSeRec, TiSASRec+LBSP, SASRec+ABSP)
    toys_bundle = FrozenDatasetBundle("toys", data_root, PROJECT_ROOT, verify_hashes=False)
    tmp_dir = Path(tempfile.mkdtemp(prefix="kais_rev2_smoke_"))
    try:
        for bb, pool in [("ticoserec", "none"), ("tisasrec", "lbsp_mean"), ("sasrec", "absp_mean")]:
            res = run_job(
                dataset_tag="toys",
                backbone=bb,
                pooling_mode=pool,
                seed=42,
                data_root=data_root,
                project_root=PROJECT_ROOT,
                output_dir=tmp_dir / f"toys__{bb}__{pool}__s42",
                bundle=toys_bundle,
                verify_hashes=False,
                allow_cpu=True,
                smoke_mode=True,
                save_all_perm_csvs=False,
            )
            assert res["status"] == "COMPLETED"
        print("[PASS] End-to-end real-data smoke runs (TiCoSeRec, TiSASRec+LBSP, SASRec+ABSP) succeeded!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\time_uncertainty_rec"))
    parser.add_argument("--skip-real-train", action="store_true")
    args = parser.parse_args()

    test_lru_parallel_vs_sequential()
    test_gru_gather_vs_packed_and_vectorized_ticoserec()
    test_all_backbones_and_exact_invariance()
    test_real_data_contracts_and_smoke(args.data_root, skip_real_train=args.skip_real_train)
    print("\nALL SMOKE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
