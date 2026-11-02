# Reproduction commands and source map

Run commands from the repository root with Python 3.10 or newer. All output paths below are new generated files and are excluded from the source manifest. The matrix has 42 independently fitted models and 54 evaluation configurations; 12 Ensemble evaluations share the matching Control checkpoints.

## Inspect the entire design

```sh
python tools/validate_release.py --output outputs/offline_validation.json
python tools/replay_pipeline.py --plan --output outputs/complete_pipeline_plan.json
python tools/replay_training.py --plan --output outputs/training_plan.json
python tools/replay_evaluation.py --plan --output outputs/evaluation_plan.json
python tools/replay_pipeline.py --write-registry outputs/run_registry_template.json
```

Plans resolve the actual archived source references and inspect their path/early-stopping contracts. They do not start models. `legacy_sources/` preserves the original scientific computation; `tools/` supplies release and replay inspection interfaces. The preflight checks all 403 source records, rather than interpreting the 19-run E1 follow-up as the entire study.

## Download and rebuild input data

Follow `data_manifest/README.md` and the provider terms. An example download invocation is:

```sh
python tools/download_data.py --dataset ml1m --workspace /root/autodl-tmp/time_uncertainty_rec
```

Use `beauty`, `sports` and `toys` for the remaining inputs. Downloads verify recorded fingerprints and do not silently overwrite different files. Preprocessing is provided in the preserved scientific scripts. `config/pipeline_stages.json` lists their dependency order and output layout; `provenance/source_index.json` resolves archive/local references to bundled files. Keep stable source-row order, the original item maps, split boundaries and timestamp resolution.

The historical stage record mentions old truncated local Sports/Toys gzip copies. `data_manifest/official_source_verification.json` documents separately reacquired matching provider archives. None of these raw archives is included. The acquisition proof does not establish that a fresh end-to-end preprocessing rebuild was run in this package.

## Training and evaluation

Use the Linux layout `/root/autodl-tmp/time_uncertainty_rec` to satisfy original guards. Several stage scripts bind their exact root, stage, data, model-source hash and expected manifests. They are executable scientific sources, not arbitrary-path portable launchers. Inspect the detailed plan before invoking them.

`--prepare-job` can produce an isolated path-adjusted trainer copy outside this repository while checking that all function/class ASTs are identical. For example, after creating the required Linux data workspace:

```sh
python tools/replay_training.py --prepare-job ml1m_control_seed42 --data-root /root/autodl-tmp/time_uncertainty_rec --runtime-root /root/autodl-tmp/kais_fresh_runtime --output outputs/prepared_job.json
```

This prepares source and stage records. It does not prove that every data/manifest guard has been satisfied and does not launch training. Follow the returned `direct_python_argv_after_contract_validation` only after reviewing its contracts. The evaluation plan identifies archived perturbation/inference drivers for all methods. `replay_evaluation.py` exposes the same planning and result-contract interface; it does not automatically launch all historical inference jobs.

## Register fresh predictions and recompute statistics

The generated registry is a template of 54 expected configurations. Populate each output directory with actual fresh checkpoint/prediction/label metadata; preserve shared user/target/panel identity across methods and fixed seeds. Add `label_files` and `target_metadata` mappings as documented in `tools/recompute_statistics.py --help` and its source. The latter needs the per-sample analytic-baseline metadata, not just ranks. Full user prediction panels are intentionally not shipped.

```sh
python tools/replay_pipeline.py --validate-results /path/to/fresh-results --run-registry outputs/run_registry_template.json --output outputs/result_contract.json
python tools/recompute_statistics.py --root /path/to/fresh-results --run-registry outputs/run_registry_template.json --output-dir outputs/fresh_statistics
```

The result-contract JSON includes the run directories and hashes and can itself be used as a registry. Its `files_sha256_scope: output_directory` makes each hash relative to that run; a historical registry without this field uses root-relative keys. Preserve the optional label/target-metadata mappings when creating registries. Use `--verify-frozen` only when the supplied hashes are the intended baseline for those saved files, never as a claim that new stochastic training must match historical checkpoints. Omit `--skip-bootstrap` to compute user bootstrap intervals. Aggregates in `aggregates/statistics/` are historical reference outputs; they alone cannot recreate the individual prediction panels or retrain models.

## Figure and table mapping

| Article material | Packaged sources and inputs |
| --- | --- |
| Figures 1–3 | `figures/build_structural_figures.py`, `fig1_example.json`, `fig3_counts.csv` |
| Figures 4–7 | `figures/build_quantitative_figures.py`, method/bucket/seed/paired summaries and training curves in `figures/inputs/` |
| Dataset metadata and accuracy | `tables/table1_dataset_metadata.tex`, `tables/table2_accuracy.tex`, `aggregates/statistics/` |
| Sensitivity, four-method comparison and subgroups | `tables/table3_sensitivity.tex` through `table5_lbsp_subgroups.tex` |
| Supplementary costs, seeds and diagnostics | `tables/appendix_*.tex`, `aggregates/X7/`, `aggregates/statistics/` |
| Exact table number provenance | `tables/frozen_numeric_cells.json` |

The manuscript and its author declarations are supplied separately from this code-only package. The public repository is https://github.com/chunjieyan/yanchunjie; its inspected commit is 40bf70113553489d51581225a4953c76a18b8332. No archive DOI is claimed. Full article reproduction details accompany this package in `README_REPRODUCIBILITY.md`.
