# Benchmark acquisition and input fingerprints

No raw interactions or per-user derived records are redistributed. Acquire MovieLens-1M from the [GroupLens release](https://grouplens.org/datasets/movielens/1m/) and the Amazon 2014 category files from the [original dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html). `providers.json` gives the specific file links and expected layout. Read and follow each provider's terms; a code license does not grant data redistribution rights.

Beauty uses `ratings_Beauty.csv`, followed by the disclosed deduplication/5-core processing. Sports and Toys use the 2014 five-core review files. Keep MovieLens timestamps in seconds and Amazon native day buckets. Use the source-chain entrypoints in `provenance/source_index.json`; do not substitute a modern dataset version or reorder tied records.

`expected_sha256.csv` preserves 76 recorded raw/processed input fingerprints. `local_input_verification.csv` records the local audit: 74 exact byte matches and two compressed-archive byte mismatches. Those two old local gzip copies were incomplete; both official archives were separately reacquired and verified against the recorded compressed SHA256, byte counts, complete gzip EOF/CRC and review-row counts. See `official_source_verification.json`. Neither raw cache is redistributed, and the old local files were left unchanged. Acquire the provider archives using the documented links rather than using the incomplete local copies. Do not silently replace the historical expected hash. Downloaded archives should be checked before preprocessing, and the processed CSVs should then be checked against their own frozen fingerprints.

After data acquisition/reconstruction, run:

```sh
python tools/reproduce.py preflight --data-root /path/to/research-workspace --verify-data
```

The command is read-only. It checks paths and hashes without importing a trainer or executing a GPU job. Unrounded aggregate figure inputs are separately supplied under `figures/inputs/`; those small inputs permit figure reproduction without benchmark downloads.

Acquire the four inputs in the original Linux workspace layout with:

```sh
python tools/download_data.py --dataset ml1m --workspace /root/autodl-tmp/time_uncertainty_rec
python tools/download_data.py --dataset beauty --workspace /root/autodl-tmp/time_uncertainty_rec
python tools/download_data.py --dataset sports --workspace /root/autodl-tmp/time_uncertainty_rec
python tools/download_data.py --dataset toys --workspace /root/autodl-tmp/time_uncertainty_rec
```

These commands perform downloads only when invoked. The packaging workflow did not invoke this helper. It refuses to overwrite a different existing input and keeps a mismatched download for inspection.
