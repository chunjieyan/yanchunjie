# Prediction sensitivity to timestamp ties in sequential recommendation

This repository contains source and reproduction materials for the KAIS manuscript. The public entry is https://github.com/chunjieyan/yanchunjie. The associated reproduction README records the inspected commit, methodological settings and scope. The original project is distributed under its owner-provided MIT LICENSE; the pinned time-aware adaptation retains its separate third-party notice. The project archive is https://doi.org/10.5281/zenodo.23879414. This GitHub source snapshot is version 1.0.1; the archive link does not by itself establish that the archived files contain this revision.

Version 1.0.1 (3 November 2026) updates the figure styling and table layout used in the revised manuscript. Source values, experiment code and numerical results are unchanged from version 1.0.0.

The GitHub repository already includes the expanded `legacy_sources/` directory: 403 indexed scientific-source files and 46 supporting JSON records, plus three directory placeholders. Run `python tools/validate_release.py` for dependency-free checks; no source-unpacking step is required for this checkout. See [the command/source guide](docs/REPRODUCING.md), and [checks actually run](VALIDATION.md). This is an independent source snapshot; the rest of the research workspace is not required for these checks.

The package contains the original preprocessing, split construction, model, training, perturbation, evaluation and statistical sources; original stage/configuration records; a 42-model matrix and 12 Control-reusing Ensemble evaluations; frozen aggregate inputs; common figure styling; and offline validation/replay entrypoints. Every copied original source is byte-identical to its recorded SHA256. No complete interaction/prediction panel or model checkpoint is included. The single published Fig. 1 example is retained with its product metadata and recorded ranks/lists so that the diagram can be redrawn.

## Directory map

| Path | Contents |
| --- | --- |
| `legacy_sources/` | Unmodified scientific and orchestration sources, with original relative stage structure |
| `provenance/source_index.json` | Source category, original origin, SHA256, imports, path guards and packaged path |
| `config/model_matrix.json` | 42 independent models: Control 12, Shuffle 12, LBSP 12, TiSASRec 6 |
| `config/ensemble_evaluations.json` | 12 Ensemble configurations reusing Control weights |
| `data_manifest/` | Four provider links, 76 raw/processed fingerprints and exact official raw acquisition proofs; no datasets |
| `environment/` | Recorded Linux/server package inventory, core versions and plotting dependencies |
| `aggregates/` | Unrounded model/seed/population summaries, bootstrap intervals, training curves and measured costs |
| `figures/` | Two portable builders, one shared style and minimal unrounded inputs for seven figures |
| `tables/` | Editable article tables and the frozen numeric provenance trace |
| `tools/` | Data acquisition, source-integrity/preflight, replay and statistics entrypoints |
| `third_party/` | Pinned RecBole-TRM attribution and its complete MIT notice |

`MANIFEST.sha256.json` checks the released file bytes. It excludes itself and generated `outputs/` files. The original server stage scripts are retained as evidence of the computation actually used, including their original path and manifest guards. Do not remove those guards or replace frozen hashes silently.

## Environment

The recorded research environment is Linux x86-64, Python 3.10.22, PyTorch 2.13.0+cu132, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, Matplotlib 3.10.9 and tqdm 4.70.1, with an RTX 4090 and CUDA 13.2. Full sampled package versions are in `environment/server_snapshot_packages.txt`; newer core runtime measurements are separate in `recorded_environment.json`. These records are not a newly solved and GPU-tested environment lock.

In a Python 3.10 environment on a compatible Linux/CUDA host, the recorded PyTorch wheel is available from the [official cu132 index](https://download.pytorch.org/whl/cu132/torch/):

```sh
python -m pip install torch==2.13.0+cu132 --index-url https://download.pytorch.org/whl/cu132
python -m pip install numpy==2.2.6 pandas==2.3.3 scipy==1.15.3 matplotlib==3.10.9 tqdm==4.70.1
python -m pip install -r environment/plotting_requirements.txt
```

No installer or GPU job is run by the integrity/preflight commands. Plotting is CPU-only. Arial or an appropriately licensed installed font is needed to reproduce exact lettering; proprietary font files are not redistributed.

## Offline validation and recorded plan

From the repository root:

```sh
python tools/reproduce.py source-integrity
python tools/reproduce.py preflight
python tools/reproduce.py model-plan
python tools/reproduce.py relocation-plan --target-root /root/autodl-tmp/time_uncertainty_rec
python tests/test_result_contracts.py
```

The first two commands verify file bytes and parse Python source without importing trainers. Optional package metadata is reported without claiming the local interpreter is the original server. `--strict` additionally fails for missing execution dependencies or unprovided data. The recorded experiment comprises 42 independently fitted models and 54 evaluated configurations; permutation views, Ensemble members and users are not extra training seeds.

## Data acquisition and preprocessing

Use the [GroupLens MovieLens-1M release](https://grouplens.org/datasets/movielens/1m/) and [Amazon 2014 dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html), following their terms. Exact file links, the original input layout and acquisition commands are in `data_manifest/README.md`. Use the older Amazon files, not the linked 2023 dataset.

Sports and Toys official gzip files were independently reacquired and exactly matched the recorded compressed SHA256, byte counts, gzip EOF/CRC and complete row counts; only the proof metadata is included. The old local incomplete copies were not changed and are not distributed. Raw data acquisition is separate from code licensing.

After raw data and preprocessing reconstruction:

```sh
python tools/reproduce.py preflight --data-root /root/autodl-tmp/time_uncertainty_rec --verify-data
```

The original preprocessing dependency order is recorded in `config/pipeline_stages.json` and the full source map. Stable source-row order, catalogue maps, timestamp buckets, 80/90-percentile global split boundaries, strict target exclusions, history truncation and event/time joins are part of the protocol. They must not be replaced by a generic split or item sort.

## Reproduce the frozen figures and tables

The figure builders use their compact unrounded inputs, so they need no raw interaction data, fitted checkpoints or GPU:

```sh
python tools/reproduce.py redraw --output-dir outputs/figures
```

To register licensed font files from a local directory, add `--font-dir /path/to/fonts`. All seven figures use `figures/paper_style.py`. Displayed NDCG@10 differences use units of 10^-3; the underlying summaries and interval endpoints retain their original precision. Editable table sources and the frozen numeric trace are included under `tables/`.

## Replaying scientific experiments

The supplied replay entrypoints assemble/inspect source-based jobs rather than executing the historical E1-only follow-up as if it were the entire experiment. E1's 19 training jobs and 18 follow-up evaluations are historical subsets; the complete design is 42 training jobs and 54 evaluations. The wrappers keep original scientific sources separate from generated runtime stages and output registration.

The original scripts target `/root/autodl-tmp/time_uncertainty_rec`. Some enforce both POSIX and this exact root, and generated child scripts can check their exact stage and `__file__`. Using a `--root` parameter alone does not prove arbitrary-path portability. Use a dedicated Linux workspace/container with the documented layout, and preserve a separate output directory for a fresh run. Existing frozen manifests must remain unchanged.

Fresh output registration and summary recomputation use separate records from the paper's frozen-result verification. A new training run can produce different bytes because of training stochasticity or hardware/runtime differences; a matching scientific protocol is not a guarantee of bit-identical checkpoints or published point estimates. Record checkpoint, input, user/target identity and model-source fingerprints before method/seed pairing.

See [the tested planning, registration and statistics command sequence](docs/REPRODUCING.md). All six release-tool CLI help entrypoints and the complete source/matrix preflight were run successfully. The replay interfaces inspect/prepare historical scientific sources; they are not an automatic one-command fresh training/inference service. Synthetic tests also check the 54-configuration result contract and reference comparison failures.

## Validation scope

Checks run for this snapshot include original-source fingerprints, Python syntax, model-design counts, six CLI help entrypoints, 11 synthetic result/reference contracts, and rebuilding/visually inspecting all seven figures. Data fingerprints and historical acquisition proofs are included as records; this packaging task did not freshly reconstruct all data or rerun all 42 GPU trainings. The separate manuscript is checked in its submission workflow. Historical evidence, CPU figure checks and new GPU execution have distinct scopes; see `VALIDATION.md`. No account action, public push, DOI registration or publication is performed by these scripts.

## Complete supplementary snapshot

The GitHub checkout supplies the complete expanded historical sources in `legacy_sources/`, including the four files restored by the earlier repository repair. A separately supplied `ESM_1_Reproducibility.zip` may use a compact layout: only if its extracted root contains both `legacy_sources.zip` and `unpack_sources.py`, first run `python unpack_sources.py` inside that supplementary package before its offline checks. Do not substitute a partial extraction or silently replace expected source hashes. See `README_REPRODUCIBILITY.md` for the article identity, implementation settings and limits of reproduction.
