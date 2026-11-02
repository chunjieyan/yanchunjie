# Rebuild the manuscript tables

Run with Python 3.10 or later from any working directory:

```text
python reproduction/tables/build_tables.py
```

The builder uses only the Python standard library. It reads the 13 original
editable templates and the packaged CSV inputs, verifies every original-precision
numeric record against its selector and recorded values, then writes the 13
current table files under `tables/`. No datasets, checkpoints, model training,
statistical estimators or confidence intervals are recalculated.

NDCG@10, MRR and MRR@10 in Tables 2, 4 and 12 are displayed in units of 0.001.
Their means and standard deviations are rendered directly from original values,
not multiplied from previously rounded display strings. Mean ranks and their SDs
are displayed as integers. Table 2 boldface compares learned methods only, using
unrounded means and the smaller-is-better direction for mean rank. Numeric seed
identifiers disable thousands grouping. A small positive SD can still round to
0.0 at the displayed precision; the Table 12 note explicitly explains this.

`inputs/formal_numeric_cells.json` maps each original numeric cell to a packaged
CSV, selector, exact values and original display. `input_provenance.json` records
the original CSV byte hashes. The generated `table_rebuild_validation.json`
records each display change and its original values. Figure and prose values may
use their explicitly stated original units; the table headers state the scale.
