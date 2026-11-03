# Checks performed for this source snapshot

Prepared on 1 November 2026; figure and table builders updated on 3 November 2026 (version 1.0.1). The following checks were executed locally; the GitHub Actions workflow is supplied for remote runs.

| Check | Observed result |
| --- | --- |
| Archived scientific source hashes | All 403 source records matched their original recorded SHA256 |
| Python source syntax | Parsed with Python 3.10 syntax compatibility; no syntax failures |
| Complete training/evaluation source plan | 42 independent training models, 54 evaluation configurations, 12 shared-Control Ensembles; source references resolved |
| Release CLI entrypoints | Six `--help` calls returned 0 |
| Synthetic result/reference contracts | 11 checks passed using 54 synthetic configurations and 378 generated files |
| Figure rebuild | Figures 1–7 rebuilt from the bundled minimum saved inputs; no raw data, predictions or checkpoints read |
| Figure output inspection | All seven previews visually inspected; no clipping observed; structural figure widths verified as 174 mm |
| Scoped release scan | No user-home paths, recognizable private-key/token formats or literal account passwords detected in the inspected text; no bundled runtime, data archive or model weights |

Synthetic checks cover output directory and metadata retention, explicit per-run/root-relative hash scopes, missing Ensemble original predictions, path escape rejection, equal numeric references, changed numeric references and missing reference files. The executable test is `python tests/test_result_contracts.py`; generated fixtures/reports go under ignored `outputs/`.

The local offline checker used Python 3.12.14 on Windows. Its base environment lacks PyTorch, SciPy, tqdm and Matplotlib; preflight reports this rather than treating offline validation as training readiness. The separate CPU figure check made an existing local plotting installation available to that interpreter (NumPy 2.5.3, pandas 3.0.1, Matplotlib 3.11.2, pypdf 6.10.0) and used the installed Arial font. That support installation and font are not shipped. This check establishes that the source builders produced the figures in that environment; it is not a clean-install/GPU environment regression.

The original Linux/server version records remain under `environment/`. `requirements.txt` supplies CPU dependency ranges and has not been installed into a new clean virtual environment during this task. Exact scientific training requires the original protocol, eligible data, path-bound stage configuration, compatible hardware/runtime and checkpoint fingerprints. Fresh full-data preprocessing, 42-model GPU training, 54-model-output inference, complete per-user statistical recomputation, and bit-identical reproduction of historical point estimates were not performed in this packaging task.

No scientific function/class implementation in `legacy_sources/` was changed. Release wrappers were corrected to normalize Ensemble metadata, retain fresh-run registry paths/metadata, define hash scope explicitly, require canonical predictions for every method and fail when an expected numerical reference file is absent. Do not interpret these packaging fixes as changed experimental results.

After intentional edits, run `python tools/update_manifest.py` to inspect changes, then `python tools/update_manifest.py --write` and `python tools/validate_release.py`. The updater changes only the release manifest; archived scientific-source hashes remain independently checked against provenance.

## Compact upload edition

The 449 original source files are distributed in one ZIP and restored by `python unpack_sources.py`. The original source hashes and experiment matrix are unchanged. Run this command before the offline checks listed above. Four additional extraction contract tests cover exact reconstruction, repeat execution, refusal to overwrite edited sources, archive tampering and parent-path escape. See the external compact delivery validation for the checks executed on the final ZIP.
