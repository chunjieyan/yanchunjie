# Changelog

## [2.0.0] - 2026-11-25 (KAIS Major Revision)
### Added
- **`rev2/src/`**: Unified pure-PyTorch/NumPy 8-backbone framework (`SASRec`, `TiSASRec`, `GRU4Rec`, `FMLP-Rec`, `BSARec`, `LRURec`, `LSAB`, `TiCoSeRec`) with deterministic set-pooling front-ends (`lbsp_mean`, `lbsp_max`, `lbsp_attn`, `lbsp_deepsets`, `absp_mean`), GPU-vectorized TiCoSeRec augmentations, all-bucket 5-permutation evaluation, warm-target breakdowns, multi-worker runner (`--workers`), and unified single-RTX-4090 cost profiler.
- **`rev2/tests/`**: Automated smoke and bit-exact invariance suite verifying `diff=0.0` across all 8 backbones and all pooling operators, plus bit-exact compatibility with legacy v1.0.1 checkpoints.
- **`rev2/manifests/`**: SHA-256 contracts (`dataset_contracts.json`), frozen item-ID bijections (`item_maps/*.csv`), and legacy 42-checkpoint manifest.
- **`rev2/results/`**: Complete unrounded 216-model ledgers (`master_summary.csv`, `paired_contrasts.csv`, `suite_ledger.csv`, `unified_cost_profile.csv`).
- **`rev2/manuscript_reproduction/`**: Self-contained builders for new Revision 2.0 tables (`Table 2, 6, 7, 11`) and vector PDF figures (`Fig 5, 7, 8, 10`), plus the verified 63-entry `references.bib` and automated manuscript integrity auditor.
- **`CITATION.cff`**: Machine-readable citation metadata for GitHub and Zenodo.

### Preserved
- All v1.0.1 historical source files (`legacy_sources/`), frozen provenance indices, data acquisition proofs, and third-party MIT notices remain byte-identical and intact.

## [1.0.1] - 2026-11-03
- Updated figure styling and table layouts for the initial KAIS manuscript; restored four historical `step38_kais_retrain` scripts (`commit 0ffe6cad`).

## [1.0.0] - 2026-11-02
- Initial 42-model (SASRec Control/Shuffle/LBSP + Ensemble + partial TiSASRec) release archived on Zenodo (`10.5281/zenodo.23879414`).
