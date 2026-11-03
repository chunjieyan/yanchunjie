# Online Resource 1 Reproduction materials

Article: Prediction sensitivity to timestamp ties in sequential recommendation

Journal: Knowledge and Information Systems (KAIS)

Authors: Chunjie Yan, Jingjing Fan, Yunan Zhang, Yanxiao Liu and Wenna Zhang

Affiliations: School of Information Engineering, Hebei University of Architecture; Hebei Key Laboratory of Smart City Perception and Intelligent Computing. Both: Zhangjiakou, 075000, China.

Corresponding author: Jingjing Fan, fjj1960@hebiace.edu.cn

Prepared: 3 November 2026 (version 1.0.1). Public code: https://github.com/chunjieyan/yanchunjie. Archive: https://doi.org/10.5281/zenodo.23879414.

# Reproduction details

The article reports summaries of saved model outputs. Display changes do not alter the unrounded scientific records. This companion records implementation details kept outside the article for readability.

## Public code and fixed snapshot

The public repository is [chunjieyan/yanchunjie](https://github.com/chunjieyan/yanchunjie). The repository contains model, preprocessing, training, perturbation, evaluation and statistical sources, experiment configurations, unrounded aggregate records and artwork sources. The project archive is https://doi.org/10.5281/zenodo.23879414. Version 1.0.1 identifies this GitHub source revision; the archive link is not confirmation that these revised files have been deposited there.

The GitHub checkout contains the expanded `legacy_sources/` directory, with 403 indexed scientific-source files and 46 supporting JSON records (449 preserved files), plus three directory placeholders. From the repository root, run:

```sh
python tools/validate_release.py
python tests/test_result_contracts.py
```

No `unpack_sources.py` step is required for this GitHub layout. A separately supplied `ESM_1_Reproducibility.zip` (Online Resource 1) can use a compact layout to reduce the upload file count. If its extracted root contains `legacy_sources.zip` and `unpack_sources.py`, first run `python unpack_sources.py` inside that supplementary package to restore and verify the original source paths, then run the two checks above.

The checks inspect source identity, Python syntax, the 42-model/54-evaluation matrix, CLI interfaces and synthetic output contracts. They do not train models, download benchmark interactions or recreate all per-user predictions. Each distribution retains the original scientific-source fingerprints and a release manifest appropriate to its own documentation, display files and expanded or compact layout.

An earlier public-repository inspection found four missing historical `step38_kais_retrain/scripts/` files relative to its source manifest: `38_6_beauty_sasrec_sampled100.py`, `38_7_eval_ml1m_sasrec_stability.py`, `38_8_analyze_ml1m_stability.py` and `38_9_eval_beauty_sasrec_stability.py`. The repository repair restored all four byte-identically to their recorded originals. They are present in this GitHub checkout and the complete supplementary source snapshot; use the manifest belonging to the distribution being checked.

The public repository contains an MIT `LICENSE` added by its owner. The supplementary snapshot preserves that license and the separate upstream MIT notice for the pinned RecBole-TRM adaptation. Provider terms govern the benchmark datasets, which are obtained separately.

## Material map and obtaining the data

| Material | Manuscript source project | Complete supplementary source snapshot |
| --- | --- | --- |
| Reproduction instructions | This `README_REPRODUCIBILITY.md` | `README_REPRODUCIBILITY.md`, `README.md`, `docs/REPRODUCING.md` |
| Figures 1–7 | `figures/`, `reproduction/figures/` and its `inputs/` | `figures/` and `figures/inputs/` |
| Editable article tables | `tables/*.tex`, `reproduction/tables/` | `tables/*.tex`, the self-contained `reproduction/tables/` builder and its original-precision inputs; historical trace in `tables/frozen_numeric_cells.json` |
| Models, splits, training and evaluation | Public repository and complete supplementary snapshot | GitHub: use the expanded `legacy_sources/`; compact ESM: expand `legacy_sources.zip` if supplied. Both use `provenance/source_index.json` and `config/pipeline_stages.json` |
| Experiment design and runtime records | Described in the article and this README | `config/model_matrix.json`, `config/ensemble_evaluations.json`, `environment/` |
| Data acquisition and fingerprints | Provider links below | `data_manifest/README.md`, `expected_sha256.csv`, `tools/download_data.py` |

Use the [GroupLens MovieLens-1M release](https://grouplens.org/datasets/movielens/1m/) and the older [Amazon 2014 review-data release](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html), following the providers' terms. Data acquisition, exact filenames and checksums are documented in the supplementary snapshot. For example, `python tools/download_data.py --dataset ml1m --workspace /root/autodl-tmp/time_uncertainty_rec` acquires the MovieLens input into the documented Linux layout; use `beauty`, `sports` and `toys` for the other datasets. Data are not downloaded by the offline checks above.

The supplied original server drivers contain path, stage and fingerprint guards. They target the documented Linux root and are not arbitrary-path, one-command training launchers. `docs/REPRODUCING.md` provides the dependency order, planning commands, fresh-result registration and statistical recomputation steps. A matching protocol does not guarantee byte-identical weights from a new stochastic GPU run. Raw interactions, full individual prediction panels and fitted checkpoints are not distributed with these source packages.

## Compile the manuscript

Use `main.tex` as the main file and pdfLaTeX as the compiler. The source archive includes the Springer class, bibliography style, bibliography, editable tables and seven vector figures. A normal pdfLaTeX/BibTeX build requires one initial pdfLaTeX pass, BibTeX, and two further pdfLaTeX passes. On Overleaf, upload the ZIP as a project and select pdfLaTeX.

## Redraw figures

The two drawing scripts share `reproduction/figures/paper_style.py`. Their minimal unrounded inputs are in `reproduction/figures/inputs/`. Install the Python dependencies used by the scripts in a suitable research environment (NumPy, pandas, Matplotlib and pypdf). Run the scripts with `--help` for data and output-directory options. Figures use a 174 mm canvas, Arial text and embedded vector fonts; font files are not redistributed. Supply an appropriately licensed installed font when reproducing the exact typography.

From the manuscript project root, generate new output files without overwriting the accepted figures:

```sh
python reproduction/figures/build_structural_figures.py --output-dir outputs/figures
python reproduction/figures/build_quantitative_figures.py --data-root reproduction/figures/inputs --output-dir outputs/figures
```

In the GitHub checkout or complete supplementary source snapshot, install `requirements.txt` and run `python tools/reproduce.py redraw --output-dir outputs/figures`. The GitHub sources are already expanded; only a compact ESM distribution containing `legacy_sources.zip` and `unpack_sources.py` needs the source-unpacking step described above. Rebuild all 13 current editable table files with `python reproduction/tables/build_tables.py`, which uses only the Python standard library. It checks 1,124 original-precision source cells against the packaged CSV selectors before applying the documented display-unit and formatting rules. The editable LaTeX tables compile directly with the article. The unrounded aggregates and frozen numerical trace support independent inspection of their displayed cells; rebuilding figures or compiling tables does not rerun statistical estimation or training. The table builder writes a fresh path-specific validation report; a deliberate local rebuild can therefore change the report bytes even when all 13 generated table files are identical.

Method color/marker mappings are constant: Control charcoal/circle, Shuffle blue/square, Ensemble orange/triangle, and LBSP green/diamond. Time-aware backbone comparisons use purple/filled-plus for TiSASRec. Training seeds use line styles in the learning-curve figure. Best checkpoints and stopping epochs have separate markers. Timing is reported in tables; marker area never encodes cost.

## Realized views and score ensembling

The five single-view permutation seeds are 101, 202, 303, 404 and 505. Truncation precedes permutation. Affected users have at least two distinct raw input histories among these five views; the reference input is excluded from this definition. This mask is fixed before pooling and shared by methods and training seeds.

Each final Ensemble output averages five raw score vectors from the corresponding Control checkpoint. The five groups are:

| Group | Member seeds |
| --- | --- |
| 1 | 101, 202, 303, 404, 505 |
| 2 | 1101, 1202, 1303, 1404, 1505 |
| 3 | 2101, 2202, 2303, 2404, 2505 |
| 4 | 3101, 3202, 3303, 3404, 3505 |
| 5 | 4101, 4202, 4303, 4404, 4505 |

Vectors are cast to float64 and accumulated in ascending member-seed order before division by five. Masking and ranking follow score averaging. Repeated realized histories retain their multiplicity. Each recommendation requires five views; all five stability outputs together require 25. Neither view counts nor user counts are independent training replications.

## Bootstrap reproducibility

Each paired-user percentile bootstrap cell uses 2,000 resamples, retaining the method/Control user pairing and original population size. All seven metrics share a cell's sampled indices. The master seed is 20261027. A NumPy Generator with explicit PCG64 uses the integer represented by the first eight big-endian bytes of the SHA256 digest of `20261027|dataset|method|trainingseed|population` as its cell seed. Quantiles use linear interpolation. Zero differences retain zero endpoints. Training-seed intervals are a separate analysis with three paired differences and the full-precision Student t quantile for two degrees of freedom; the article displays this quantile as 4.303.

## Event identity, time features and upstream implementation

Reference event order is stable by user, native timestamp bucket and source-row index. The time-aware training loader retains the training-CSV row ordinal within a tie. Event identity/time features join by user and original raw-row identifier, with an item-identity check; evaluation does not replace the stored order by a new sort. Time normalization is computed only from the truncated visible history, using its minimum positive native-time gap or 1 if all times are equal. Target timestamps check eligibility and do not enter model features. Same-bucket and padding relations are zero.

The time-aware implementation adapts RecBole-TRM commit `18b47f8ff98980bed2f2113b398e55b484e748b9` and retains its MIT notice. Prefix normalization, padding/readout alignment, objective and feed-forward differences are disclosed in the article; this is an adapted comparison with the matched training budget.

## Computational measurement settings

Both measurement phases use representative training seed 42. Inference and training profiling use different physical RTX 4090 devices, with device/runtime identities fixed within each phase. Inference batch size is 256. Saved predictions are reproduced before timing. Two batches warm up the device; five synchronized complete-panel repetitions supply means and sample SDs. Timing includes CPU feature/permutation construction, transfers, forward computation, score averaging where applicable, masking, ranking and small result transfers to CPU. Model loading, disk input, output serialization and reference checks are excluded. There is no score cache.

Training profiling uses one warm-up epoch and five measured epochs, including Shuffle epoch setup and excluding validation/checkpointing. Ensemble shares Control training. Both phases disable TF32. Inference disables deterministic-algorithm enforcement; training enables it with cuBLAS workspace configuration `:4096:8`. Reported memory is peak allocated memory; training peaks are averaged over measured epochs. Allocated, reserved and whole-device memory are different quantities. One MiB is 2^20 bytes. Per-user inference time divides by complete-panel N, not affected-user N. Calibration runtime is excluded.

## Motivating example and metadata

The Beauty illustration keeps the seed-42 Control checkpoint, prefix, target and candidates fixed. It uses the first saved hit-flip user, preferring a final bucket with two or three events. The two final-bucket records have UTC date 17 April 2014; the target record has date 25 June 2014. The full Top-10 lists share no items and the saved target ranks are 1 and 335. Only the first three recommendations are drawn. Product names are abbreviated from verified original Amazon 2014 Beauty metadata joined by ASIN; title records and fingerprints remain in the provenance materials. Metadata lookup does not change interaction times, scores, ranking candidates or predictions.

## Numerical records and scope

The article states units in every table and axis. NDCG@10, MRR and MRR@10 entries in the main accuracy and four-method tables and the warm/cold target accuracy table are displayed in units of 10^-3 with one decimal; their unrounded values remain in the aggregate records. NDCG@10 differences also use units of 10^-3 with one decimal. Narrative accuracy values and the calibration table retain their stated original units. Percentages use one decimal and mean target ranks use integers throughout. Training-seed identifiers are categorical labels (`42`, `2026`, `3407`) without thousands separators. Negative differences that round to zero are displayed as 0.0. All calculations and interval endpoints use unrounded saved inputs. Complete-panel, affected-user, pooled, singleton, warm and cold populations remain distinct.

The manuscript source archive contains compact plot inputs and editable numerical tables. The research workspace retains the complete aggregate inputs and their source fingerprints. Original interaction files and model weights are not part of this manuscript source archive. Reproducing training additionally requires the benchmark data, data construction and scientific training/evaluation implementations; rebuilding a plot does not rerun those experiments.
