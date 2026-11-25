# Prediction sensitivity to timestamp ties in sequential recommendation (v2.0.0)

[![Zenodo Concept DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.23879414-blue)](https://doi.org/10.5281/zenodo.23879414)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official source code, frozen dataset/split manifests, bit-exact invariance tests, complete 216-model aggregate records, and manuscript figure/table reproduction scripts for the substantially revised *Knowledge and Information Systems* (KAIS) manuscript:

> **Prediction sensitivity to timestamp ties in sequential recommendation**  
> Chunjie Yan, Jingjing Fan*, Yunan Zhang, Yanxiao Liu, and Wenna Zhang  
> School of Information Engineering & Hebei Key Laboratory of Smart City Perception and Intelligent Computing, Hebei University of Architecture, China

- **Public GitHub repository:** https://github.com/chunjieyan/yanchunjie
- **Zenodo all-version concept DOI:** https://doi.org/10.5281/zenodo.23879414 (resolves to the latest published version while retaining v1.0.x permanently)
- **License:** MIT License (`LICENSE`); the pinned RecBole-TRM time-aware adaptation retains its upstream MIT notice in `third_party/`.

---

## What is New in Version 2.0.0 (KAIS Major Revision)

Version 2.0.0 preserves **100% of the v1.0.1 legacy archive** (`legacy_sources/`, `aggregates/`, `provenance/`, `config/`, `data_manifest/`, `third_party/`) untouched and adds the complete Revision 2.0 experimental framework under **`rev2/`**, expanding the study from 42 to **216 independently fitted models** (72 dataset × backbone × pooling configurations × 3 training seeds, 0 failures, 0 invariance violations):

1. **Eight-Backbone / Seven-Paradigm Sensitivity Audit (All 4 Datasets × 3 Seeds):**
   - `SASRec` (ICDM'18), `TiSASRec` (WSDM'20, completed on Sports & Toys), `GRU4Rec` (ICLR'16 / CIKM'18), `FMLP-Rec` (WWW'22), `BSARec` (AAAI'24), `LRURec` (WSDM'24), `LSAB` (ACM TKDD'24), and `TiCoSeRec` (IEEE TKDE'24), all under matched capacity (`d=64`, `L=50`, 2 blocks/heads, dropout `0.2`, Adam `lr=1e-3`, full-item cross-entropy, seeds `42 / 2026 / 3407`).
   - Prominent **Warm-target NDCG@10** reporting alongside complete strict full-catalogue evaluation.
2. **Cross-Backbone Plug-In Generality of LBSP (20 Backbone–Dataset Pairs):**
   - Parameter-free `LBSP-Mean` attached to `SASRec`, `TiSASRec`, `GRU4Rec`, `BSARec`, and `TiCoSeRec` across MovieLens-1M, Beauty, Sports, and Toys: **100.0% last-bucket Top-10 overlap and 0.0% hit flips across all 20 pairs**, **20/20 affected-user NDCG@10 gains**, **18/20 complete-panel gains** (0 significant losses), **#1 overall model on 4/4 datasets**, and **85–93% spillover reduction in all-bucket jitter**.
3. **Pooling-Operator Ablations & Global All-Bucket Invariance (ABSP):**
   - Direct comparison of `Mean`, `Max`, attention-based `PMA` (Set Transformer), and cardinality-aware `DeepSets` (`+log(1+|B|)`) on SASRec, plus global `ABSP-Mean` achieving **100.0% overlap / 0.00% hit flips under both last-bucket and all-bucket permutations** (which affect 81.7–97.8% of strict test users) with 0 extra parameters.
4. **Unified Single-RTX-4090 Cost Profiling:**
   - All 72 seed-42 configurations benchmarked sequentially on one NVIDIA GeForce RTX 4090 GPU (`rev2/results/unified_cost_profile.csv`).

---

## Repository Directory Map

| Path | Scope | Contents |
| :--- | :--- | :--- |
| `rev2/src/` | **v2.0.0** | Unified 8-backbone PyTorch framework (`models.py`), deterministic set-pooling front-ends (`pooling.py`), frozen protocol & all-bucket evaluator (`protocol.py`), GPU-vectorized trainer (`train_eval.py`), legacy 42-checkpoint read-only evaluator (`eval_legacy_checkpoints.py`), multi-worker suite runner (`run_suite.py`), unified cost profiler (`benchmark_costs.py`), statistical aggregator (`aggregate_results.py`) |
| `rev2/tests/` | **v2.0.0** | Bit-exact permutation invariance (`diff=0.0`), parallel LRU scan vs. sequential recurrence, causal GRU gather vs. `pack_padded_sequence`, legacy checkpoint compatibility (`test_smoke_all_models.py`, `test_legacy_compat.py`) |
| `rev2/manifests/` | **v2.0.0** | Frozen SHA-256 dataset & split contracts (`dataset_contracts.json`), exact frozen item-ID bijections (`item_maps/*.csv`), and 42 legacy checkpoint index (`legacy_42_checkpoints.json`) |
| `rev2/results/` | **v2.0.0** | Complete unrounded 216-model CSV ledgers (`master_summary.csv`, `paired_contrasts.csv`, `suite_ledger.csv`, `unified_cost_profile.csv`) |
| `rev2/manuscript_reproduction/` | **v2.0.0** | Self-contained scripts (`build_rev2_tables.py`, `build_rev2_figures.py`, `audit_manuscript.py`, `paper_style.py`), generated LaTeX tables (`tables/*.tex`), vector PDF figures (`figures/Fig1.pdf`–`Fig11.pdf`), and verified 63-entry `references.bib` |
| `rev2/scripts/` | **v2.0.0** | Reproducible Linux/AutoDL launchers (`run_existing_server.sh`, `run_autodl.sh`) and `requirements.txt` |
| `legacy_sources/` | v1.0.1 | Preserved unmodified v1.0.x training, split, perturbation, and evaluation scripts (403 indexed files + 46 JSON records) |
| `aggregates/`, `config/`, `data_manifest/`, `provenance/`, `third_party/`, `tools/` | v1.0.1 | Original v1.0.1 42-model provenance, data acquisition proofs, and RecBole-TRM MIT license notice |

> **Data & Weight Policy:** Raw MovieLens-1M and Amazon 2014 interaction files and fitted `.pt` model checkpoints are not redistributed due to dataset provider terms and archive size limits. Acquire MovieLens-1M from [GroupLens](https://grouplens.org/datasets/movielens/1m/) and Amazon 2014 5-core subsets from [McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) (see `data_manifest/README.md`).

---

## Quick Verification Commands

### 1. Bit-Exact Invariance & Architectural Smoke Tests (CPU or GPU, no raw data required)
```bash
pip install -r rev2/scripts/requirements.txt
python rev2/tests/test_smoke_all_models.py
```

### 2. Rebuild All Revision 2.0 LaTeX Tables & Vector PDF Figures (CPU only, no raw data required)
```bash
python rev2/manuscript_reproduction/scripts/build_rev2_tables.py
python rev2/manuscript_reproduction/scripts/build_rev2_figures.py
```

### 3. Run the Full 216-Model Suite on a Single RTX 4090 GPU (with frozen splits in place)
```bash
# Reusing the 42 frozen legacy checkpoints (174 new models + 42 read-only evaluations + cost profiling):
WORKERS=3 bash rev2/scripts/run_existing_server.sh all

# Or full from-scratch 216-model execution:
bash rev2/scripts/run_autodl.sh all
```

### 4. Legacy v1.0.1 Offline Integrity Checks (preserved)
```bash
python tools/validate_release.py
python tests/test_result_contracts.py
```

---

## Environment
- **Tested Hardware:** Single NVIDIA GeForce RTX 4090 (24 GB VRAM), Linux x86-64
- **Software Stack:** Python 3.10, PyTorch 2.13.0+cu132, NumPy 2.2+, Matplotlib 3.10+ (zero `pandas` or `RecBole` runtime dependency required for `rev2/src/`)
