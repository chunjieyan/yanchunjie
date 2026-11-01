#!/usr/bin/env python3
"""Prepare a private, isolated Sports/Toys order correction; never run research code.

Python 3.10+, standard library only. All expected hashes are pinned to the
2026-10-22 private server snapshot. Default and only action: prepare files.
"""

import argparse
import ast
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import sys
import uuid


SOURCE_SHA256 = {
    "step38_kais_retrain/scripts/38_12_train_sports_sasrec_bs128.py": "3d60649944f8932ff16d5c5af22e7281d8f531da5135723be4db251dd7b93e0f",
    "step38_kais_retrain/scripts/38_13_train_toys_sasrec_bs128.py": "f519ade3dc3e4d7096272bd413514dc1d20b07e842ace29cf0c3a4db797971d1",
    "step38_kais_retrain/scripts/38_1_train_ml1m_sasrec_bs128.py": "56fdd85061943778708a2de7ec18bbe79bd3873a4d9ffbc12b9e693fadb4b27c",
}
DATA_SHA256 = {
    "processed/diagnostic_a/Amazon_Sports/file_order.csv": "e4619ab4f9e74f98b817a6a485b8453c1519bf380bc248fd1d2238fb8bed3a9c",
    "processed/diagnostic_a/Amazon_Toys/file_order.csv": "8129322bddc716bfca6877a8c0e7cbf2b7ad3baf236fccc58ee95af1d08cf541",
    "processed/diagnostic_a_splits/Amazon_Sports/file_order/train.csv": "8b78398cf350881125d04a15cd0b6181b17d1772d35f428ed61d3e9220a16539",
    "processed/diagnostic_a_splits/Amazon_Toys/file_order/train.csv": "26e4e892b1e520fdc213bddad4af3f314e865f53a1c8cbcf1a4e0f60a83ff8c0",
    "processed/strict_diagnostic_a/Amazon_Sports/file_order/test.csv": "5f2f31713ee7f6c92bb6fde915500d51f1a49328f3f1fc94f146530ad54e0e89",
    "processed/strict_diagnostic_a/Amazon_Sports/file_order/val.csv": "8ffe18f11957ba4229aa42d4e1b5d2ffdfbf3ef2adb4ecd0ff3b349469cbe7fb",
    "processed/strict_diagnostic_a/Amazon_Toys/file_order/test.csv": "3e287ca6be6c556d5abfd691666abdc4fde1eca10327d6b8329a8aa55d0df445",
    "processed/strict_diagnostic_a/Amazon_Toys/file_order/val.csv": "5b2187611068dbf7f384e55489d3704710387ef42bf86fcf028836ae26e8f14d",
}
MODEL_SOURCE = "step38_kais_retrain/scripts/38_1_train_ml1m_sasrec_bs128.py"
TRAINERS = {
    "sports": "step38_kais_retrain/scripts/38_12_train_sports_sasrec_bs128.py",
    "toys": "step38_kais_retrain/scripts/38_13_train_toys_sasrec_bs128.py",
}


def plain_path(path):
    """Reject symbolic links and Windows reparse points in every component."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected an absolute path without '..': " + str(path))
    for part in reversed((path,) + tuple(path.parents)):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError("Symbolic link/reparse point rejected: " + str(part))
    return path


def checked_file(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or any(p in ("", ".", "..") for p in relative.parts):
        raise ValueError("Unsafe relative input path: " + str(relative))
    if "\\" in str(relative) or ":" in str(relative):
        raise ValueError("Unsafe relative input path: " + str(relative))
    root = plain_path(root)
    path = plain_path(root.joinpath(*relative.parts))
    path.resolve(strict=False).relative_to(root.resolve(strict=False))
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Expected a regular file: " + str(path))
    return path


def checked_sha256(root, relative, expected):
    path = checked_file(root, relative)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError("Input changed while opening: " + relative)
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = checked_file(root, relative).stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError("Input changed during hashing: " + relative)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError("SHA256 mismatch; no training is allowed: " + relative)
    return actual


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Source structure changed; refusing uncertain patch: " + old[:65])
    return text.replace(old, new, 1)


ORDER_OLD = '''train_df["raw_line"] = pd.to_numeric(
    train_df["raw_line"],
    errors="raise",
)

train_df = train_df.sort_values(
    "raw_line",
    kind="stable",
)'''

ORDER_NEW = '''# The sole scientific correction: canonical per-user temporal order.
for _key in ("time_bucket", "raw_line"):
    _values = train_df[_key].astype(str)
    if not _values.str.fullmatch(r"[+-]?[0-9]+").all():
        raise RuntimeError("Non-integer or missing training field: " + _key)
    _integers = [int(value) for value in _values]
    if any(value < -(2**63) or value >= 2**63 for value in _integers):
        raise RuntimeError("Training integer outside int64: " + _key)
    train_df[_key] = np.asarray(_integers, dtype=np.int64)

if train_df.duplicated(["user_id", "raw_line"]).any():
    raise RuntimeError("Duplicate (user_id, raw_line) event key")
_event_count = len(train_df)
_event_keys = set(zip(train_df["user_id"], train_df["raw_line"]))
train_df = train_df.sort_values(
    ["user_id", "time_bucket", "raw_line"], kind="stable",
)
if (len(train_df) != _event_count
        or set(zip(train_df["user_id"], train_df["raw_line"])) != _event_keys
        or sorted(train_df.index.tolist()) != list(range(_event_count))):
    raise RuntimeError("Ordering correction added or removed training events")
_same_user = train_df["user_id"].eq(train_df["user_id"].shift())
_bucket = train_df["time_bucket"]
_raw = train_df["raw_line"]
_bad_order = (_bucket.lt(_bucket.shift())
              | (_bucket.eq(_bucket.shift()) & _raw.le(_raw.shift())))
if (_same_user & _bad_order).any():
    raise RuntimeError("Canonical temporal/tie order assertion failed")
print("ORDERFIX PASS: unchanged event keys; user,time_bucket,raw_line stable order")'''


def frozen_model(source):
    tree = ast.parse(source)
    found = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SASRecControl"]
    if len(found) != 1:
        raise RuntimeError("Expected exactly one SASRecControl definition")
    header = ("# Frozen class only; no original top-level training statements.\n"
              "import math\nimport numpy as np\nimport torch\n"
              "import torch.nn as nn\nimport torch.nn.functional as F\nMAX_LEN = 50\n\n")
    model = header + ast.get_source_segment(source, found[0]) + "\n"
    new_tree = ast.parse(model)
    new_class = next(n for n in new_tree.body if isinstance(n, ast.ClassDef))
    if ast.dump(found[0]) != ast.dump(new_class):
        raise RuntimeError("Frozen model class differs from original")
    return model


def fixed_trainer(source, root, stage, model_hash):
    parser_old = '''parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["smoke", "train"],
    default="smoke",
)
args = parser.parse_args()'''
    source = replace_once(source, parser_old, "# Arguments parsed before numerical imports and integrity checks.")
    source = replace_once(source, "ROOT = Path(__file__).resolve().parents[2]", "ROOT = _ORDERFIX_ROOT")
    source = replace_once(source, 'BASE = ROOT / "step38_kais_retrain"', "BASE = _ORDERFIX_STAGE")
    source = replace_once(source, "SEED = 42", "SEED = args.seed")
    source = replace_once(source, '''MODEL_SCRIPT = (
    BASE / "scripts/38_1_train_ml1m_sasrec_bs128.py"
)''', 'MODEL_SCRIPT = _ORDERFIX_STAGE / "scripts/model_definition.py"')
    tree = ast.parse(source)
    outputs = [n for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "OUTPUT_DIR" for t in n.targets)]
    if len(outputs) != 1:
        raise RuntimeError("Expected exactly one OUTPUT_DIR assignment")
    source = replace_once(source, ast.get_source_segment(source, outputs[0]),
                          'OUTPUT_DIR = (BASE / "outputs" / f"{DATASET.lower()}_control_orderfix_seed{SEED}")')
    source = replace_once(source, '''required_cols = {
    "user_id",
    "item_id",
    "raw_line",
}''', '''required_cols = {"user_id", "item_id", "raw_line", "time_bucket"}''')
    source = replace_once(source, ORDER_OLD, ORDER_NEW)
    source = replace_once(source, '# 4. 读取训练交互，按冻结 file_order 排序',
                          '# 4. 读取原始冻结交互，按用户、时间桶、原始行号构造序列')
    source = replace_once(source, 'config = {\n    "dataset": DATASET,', '''config = {
    "orderfix": {
        "version": "kais_orderfix_v1",
        "sequence_sort_keys": ["user_id", "time_bucket", "raw_line"],
        "sort_kind": "stable",
        "original_input_files_unchanged": True,
        "input_sha256": _INPUT_SHA256,
        "frozen_model_sha256": _FROZEN_MODEL_SHA256,
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preparation_manifest_sha256": hashlib.sha256(
            (_ORDERFIX_STAGE / "manifests/orderfix_manifest.json").read_bytes()
        ).hexdigest(),
    },
    "dataset": DATASET,''')
    helper_source = "\n\n".join(inspect.getsource(f) for f in (plain_path, checked_file, checked_sha256))
    prelude = ("#!/usr/bin/env python3\n"
               "# Generated private order correction. Preparation never runs this script.\n"
               "import argparse\nimport hashlib\nimport os\nfrom pathlib import Path, PurePosixPath\nimport stat\n\n"
               + helper_source + "\n\n"
               + "parser = argparse.ArgumentParser(description='Isolated KAIS order correction')\n"
               + "parser.add_argument('--mode', choices=['smoke', 'train'], default='smoke')\n"
               + "parser.add_argument('--seed', type=int, choices=[42, 2026, 3407], default=42)\n"
               + "args = parser.parse_args()\n"
               + "_ORDERFIX_ROOT = plain_path(Path(" + repr(str(root)) + "))\n"
               + "_ORDERFIX_STAGE = plain_path(Path(" + repr(str(stage)) + "))\n"
               + "if Path(__file__).resolve().parent != _ORDERFIX_STAGE / 'scripts':\n"
               + "    raise RuntimeError('Run the original generated trainer in its original stage')\n"
               + "_INPUT_SHA256 = " + repr(DATA_SHA256) + "\n"
               + "for _relative, _expected in _INPUT_SHA256.items():\n"
               + "    checked_sha256(_ORDERFIX_ROOT, _relative, _expected)\n"
               + "_FROZEN_MODEL_SHA256 = " + repr(model_hash) + "\n"
               + "checked_sha256(_ORDERFIX_STAGE, 'scripts/model_definition.py', _FROZEN_MODEL_SHA256)\n"
               + "print('INTEGRITY PASS: eight fixed CSV inputs and frozen model definition')\n\n")
    result = prelude + source
    ast.parse(result)
    return result


def write_new(path, content):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = plain_path(args.root)
    if not root.is_dir():
        raise ValueError("Project root is not a directory")
    root = root.resolve(strict=False)
    # Finish all source and input hash checks before creating any directory.
    verified = {}
    for relative, expected in dict(SOURCE_SHA256, **DATA_SHA256).items():
        verified[relative] = checked_sha256(root, relative, expected)
    sources = {}
    for relative, expected in SOURCE_SHA256.items():
        content = checked_file(root, relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError("Source changed after initial checks: " + relative)
        sources[relative] = content.decode("utf-8-sig")
        ast.parse(sources[relative])
    now = datetime.datetime.now(datetime.timezone.utc)
    stage = root / ("kais_orderfix_v1_" + now.strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12])
    model = frozen_model(sources[MODEL_SOURCE])
    model_hash = hashlib.sha256(model.encode("utf-8")).hexdigest()
    generated = {"scripts/model_definition.py": model}
    for dataset, relative in TRAINERS.items():
        generated["scripts/train_" + dataset + "_orderfix.py"] = fixed_trainer(sources[relative], root, stage, model_hash)
    # Static parse only: neither original nor generated research code is executed.
    generated_hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                        for name, text in generated.items()}
    changes = {
        "status": "PREPARED_NOT_RUN_NOT_FULL_REPRODUCTION",
        "created_utc": now.isoformat(), "root": str(root), "stage": str(stage),
        "sources_sha256": SOURCE_SHA256, "data_sha256": DATA_SHA256,
        "generated_sha256": generated_hashes,
        "scientific_change": "Sports/Toys training loader uses stable user_id,time_bucket,raw_line order, validates integer keys, unique event keys, order and event preservation. Global user ordering also changes training sample indices and seeded minibatch trajectories; identical minibatches are not claimed.",
        "operational_changes": ["Seed CLI: 42, 2026, 3407; default 42", "Default smoke preserved; no automatic execution", "Outputs isolated in this unique stage; original overwrite guard preserved", "SASRecControl class frozen by AST; original class AST unchanged", "Each trainer rehashes all eight CSVs and frozen class before numpy/pandas/torch imports"],
        "preserved": ["Hyperparameters, loss, architecture and remaining training/evaluation logic", "Original catalog source and mapping order", "Original fixed strict validation/test CSVs", "Original data, scripts, checkpoints, outputs and logs"],
        "pending": ["Run and inspect both smoke checks before full training", "Retrain Sports/Toys and evaluate new models; this preparation produces no new experimental result", "Physical ordering of 36_6 outputs remains unchanged; only corrected loaders normalize order", "Downstream 38_14/16 evaluation, 38_15/17 analysis and 38_18 aggregation still need new explicit paths", "Three seeds, further methods, timing, publication cleanup and full source/data closure remain unfinished", "No conclusion about invalidity of Beauty/MovieLens follows from this Sports/Toys bug"],
    }
    python = shlex.quote(sys.executable)
    smoke_commands = []
    train_commands = []
    for dataset in TRAINERS:
        script = shlex.quote(str(stage / "scripts" / ("train_" + dataset + "_orderfix.py")))
        smoke_commands.append(python + " -u " + script + " --mode smoke --seed 42")
        for seed in (42, 2026, 3407):
            logfile = shlex.quote(str(stage / "verification" / (dataset + "_seed" + str(seed) + "_train.log")))
            train_commands.append(python + " -u " + script + " --mode train --seed " + str(seed) + " >> " + logfile + " 2>&1")
    smoke_runner = ("#!/usr/bin/env bash\nset -uo pipefail\n"
                    "# User-invoked GPU smoke checks only; no full training.\n"
                    "sports_status=0\ntoys_status=0\n")
    for dataset, command in zip(TRAINERS, smoke_commands):
        logfile = shlex.quote(str(stage / "verification" / (dataset + "_seed42_smoke.log")))
        smoke_runner += command + " 2>&1 | tee " + logfile + "\n"
        smoke_runner += dataset + "_status=$?\n"
    smoke_runner += (python + " - " + shlex.quote(str(stage))
                     + ' "$sports_status" "$toys_status" <<\'ORDERFIX_REPORT_PY\'\n'
                     "import json, sys, zipfile\nfrom pathlib import Path\n"
                     "stage = Path(sys.argv[1])\n"
                     "status = dict(zip(('sports', 'toys'), map(int, sys.argv[2:])))\n"
                     "(stage / 'verification/smoke_status.json').write_text(json.dumps(status, indent=2) + '\\n', encoding='utf-8')\n"
                     "bundle = stage / 'smoke_reports.zip'\n"
                     "with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
                     "    for folder in ('scripts', 'manifests', 'verification'):\n"
                     "        for path in sorted((stage / folder).rglob('*')):\n"
                     "            if path.is_file():\n"
                     "                archive.write(path, path.relative_to(stage).as_posix())\n"
                     "print('SMOKE_STATUS=' + json.dumps(status))\n"
                     "print('SMOKE_REPORT_ZIP=' + str(bundle))\n"
                     "ORDERFIX_REPORT_PY\n"
                     'report_status=$?\n'
                     'if (( sports_status != 0 || toys_status != 0 || report_status != 0 )); then exit 1; fi\n')
    readme = ("# Private Sports/Toys order correction\n\nPrepared only. No training/inference/checkpoint loading has occurred.\n"
              "Original files are read only; model and trainer copies are isolated here.\n"
              "Use the original yancj environment; preparation itself uses only Python's standard library.\n\n"
              "## First: manually run smoke checks\n\nA smoke command executes research code and may use the GPU; review it first.\n"
              "It preserves the original smoke behavior and does not save a training run.\n\n```bash\n"
              + "\n".join(smoke_commands) + "\n```\n\n"
              "Or use the generated run_smoke_checks.sh to run both checks sequentially, save logs, and collect smoke_reports.zip.\n"
              "Running that shell script is an explicit smoke execution; preparation never invokes it.\n\n"
              "## Later: full training after smoke verification\n\nThese commands are documentation, never auto-executed.\n"
              "Run sequentially after smoke verification. Logs append only inside this new stage.\n"
              "Existing per-seed output directories cause a refusal rather than an overwrite.\n\n```bash\n"
              + "\n".join(train_commands) + "\n```\n\n"
              "Both trainers verify all eight pinned inputs and the frozen model before numerical imports.\n"
              "Do not edit old inputs to resolve a hash refusal; investigate and prepare a separately reviewed version.\n"
              "See manifests/orderfix_manifest.json for hashes, preserved behavior and outstanding work.\n"
              "Old Sports/Toys results remain historical and must not be mixed with corrected runs.\n"
              "Physical data sorting, downstream evaluation paths and full reproduction are still pending.\n")
    stage.mkdir(exist_ok=False)
    for child in ("scripts", "outputs", "verification", "manifests"):
        (stage / child).mkdir(exist_ok=False)
    for relative, content in generated.items():
        write_new(stage / relative, content)
    write_new(stage / "manifests/orderfix_manifest.json", json.dumps(changes, indent=2, ensure_ascii=False) + "\n")
    write_new(stage / "verification/static_checks.json", json.dumps({"syntax": "AST_PARSE_PASS", "model_class_ast": "UNCHANGED", "verified_inputs": verified, "research_execution": False}, indent=2) + "\n")
    write_new(stage / "README.md", readme)
    write_new(stage / "run_smoke_checks.sh", smoke_runner)
    write_new(stage / "verification/PREPARED.json", json.dumps({"status": "PREPARED_ONLY", "generated_sha256": generated_hashes}, indent=2) + "\n")
    print("WORKDIR=" + str(stage))
    print("STATUS=PREPARED_ONLY_NO_TRAINING")
    for dataset, command in zip(TRAINERS, smoke_commands):
        print("SMOKE_" + dataset.upper() + "=" + command)
    print("NEXT_SMOKE_COMMAND=bash " + shlex.quote(str(stage / "run_smoke_checks.sh")))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, SyntaxError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(2)
