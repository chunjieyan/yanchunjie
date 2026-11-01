#!/usr/bin/env python3
"""Prepare isolated Beauty Control seed extensions; standard library, no execution.
Original algorithm, temporal sorting, inputs and seed-42 evidence remain unchanged.
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

TRAINER_SOURCE = 'step38_kais_retrain/scripts/38_4_train_beauty_sasrec_bs128.py'
MODEL_SOURCE = 'step38_kais_retrain/scripts/38_1_train_ml1m_sasrec_bs128.py'
SOURCE_SHA256 = {
    TRAINER_SOURCE: '29c528543757d18b54b18a8c09d4dd4e573e125cc5128e9a13ae702b4be5d48e',
    MODEL_SOURCE: '56fdd85061943778708a2de7ec18bbe79bd3873a4d9ffbc12b9e693fadb4b27c',
}
DATA_SHA256 = {
    'processed/diagnostic_a/Amazon_Beauty/file_order.csv': '2d2bc662b76400c93073b77673c7c2a1396f6b1604ea4b216461e925d7d54f47',
    'processed/diagnostic_a_splits/Amazon_Beauty/file_order/train.csv': 'c81fd1c1dbcbbe88ed4596e0d952397d1ff74ba70a5ba9d8f9e4038e72e05971',
    'processed/recent_bucketset_r1/Amazon_Beauty/val/r1_samples.csv': '0f19f717b9c8a3230c8ce52735960fa48f4fb5e2c08c1787e0a4c52dbfbb5eb9',
    'processed/recent_bucketset_r1/Amazon_Beauty/test/r1_samples.csv': 'f9362385825ed79b07e4d262cf9ac7df2591c54cd7d8e95e54ba116bdaad5009',
}
REFERENCE_PATH = 'step38_kais_retrain/outputs/beauty_sasrec_control_ce_bs128_lr1em3_max200_seed42'
REFERENCE_SHA256 = {
    'config.json': 'a453f393b09d374f25868d8d2a205953b2d0da064c41aad443568bc693ad8425',
    'metrics.json': '5e10b55ec8c550344e66c092ef6511ed524527f00d02d006627b8ec9206f9b2b',
    'training_log.csv': '2b22f6ad9d8be45e775c053d65ff2f34b4392c14ebaa571d8e0e353341a9d887',
    'test_per_sample_predictions.csv': 'c9f0efd243532633594be0ca5b82f81aa1091facd8ad37ea06ffdb478330a40e',
    'item_id_map.csv': '259b8725bf180ff02c3f9a7351c43fdf1de22aa51829c40437c2040f95737289',
}


def plain_path(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected an absolute path without ..')
    for part in (path,) + tuple(path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Symbolic link/reparse point rejected: ' + str(part))
    return path.resolve(strict=False)


def checked_file(root, relative):
    if ('\\' in relative or ':' in relative or PurePosixPath(relative).is_absolute()
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise ValueError('Unsafe relative input path')
    root = plain_path(root)
    path = plain_path(root.joinpath(*relative.split('/')))
    path.relative_to(root)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Expected a regular input file')
    return path


def checked_sha256(root, relative, expected):
    path = checked_file(root, relative)
    before = path.stat()
    digest = hashlib.sha256()
    signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    with os.fdopen(os.open(str(path), flags), 'rb') as handle:
        if signature(before) != signature(os.fstat(handle.fileno())):
            raise RuntimeError('Input changed before hashing: ' + relative)
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
        after_open = os.fstat(handle.fileno())
    if signature(before) != signature(after_open) or signature(before) != signature(checked_file(root, relative).stat()):
        raise RuntimeError('Input changed while hashing: ' + relative)
    if digest.hexdigest() != expected:
        raise RuntimeError('SHA256 mismatch: ' + relative)
    return digest.hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Unexpected source structure: ' + old[:70])
    return source.replace(old, new, 1)


def class_node(source):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'SASRecControl']
    if len(nodes) != 1:
        raise RuntimeError('Expected one SASRecControl class')
    return nodes[0]


def frozen_model(source, trainer_source):
    model, original = class_node(source), class_node(trainer_source)
    if ast.dump(model) != ast.dump(original):
        raise RuntimeError('Beauty and ML1M model definitions differ')
    text = ('# Frozen SASRecControl class only; original top-level code is excluded.\n'
            'import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
            + ast.get_source_segment(source, model) + '\n')
    if ast.dump(class_node(text)) != ast.dump(model):
        raise RuntimeError('Frozen model AST changed')
    return text


SMOKE = '''if args.mode == "smoke":
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _batch = next(iter(train_loader))
    _seq, _groups, _tokens, _direct, _last, _target = [value.long().to(device) for value in _batch]
    _loss = train_one_batch(model, _seq, _groups, _tokens, _target)
    _loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Smoke produced no gradients")
    if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
        raise RuntimeError("Smoke produced nonfinite gradients")
    _n = min(32, len(val_targets))
    if _n < 1:
        raise RuntimeError("Smoke validation panel is empty")
    _overall, _subgroup = evaluate(model, val_sequences[:_n], val_groups[:_n],
        val_num_tokens[:_n], val_targets[:_n], val_direct_pooling[:_n])
    if not all(math.isfinite(float(value)) for value in _overall.values()):
        raise RuntimeError("Smoke validation metrics are nonfinite")
    print("BEAUTY_SMOKE_PASS=" + json.dumps({"seed": SEED, "loss": float(_loss.detach().cpu()),
        "validation_samples": _n, "validation": _overall, "saved_training_output": False}))
    raise SystemExit(0)

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
'''


def generated_trainer(source, root, stage, model_hash):
    output = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'OUTPUT_DIR' for t in n.targets))
    source = replace_once(source, ast.get_source_segment(source, output),
        'OUTPUT_DIR = _BEAUTY_STAGE / "outputs" / f"amazon_beauty_control_seed{SEED}"\n'
        'if OUTPUT_DIR.exists():\n    raise RuntimeError("Existing output directory; refusing overwrite: " + str(OUTPUT_DIR))')
    source = replace_once(source, 'OUTPUT_DIR.mkdir(parents=True, exist_ok=True)', '# Output creation is deferred until train mode.')
    mapping_nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                     and n.value.func.attr == 'to_csv' and '"item_id_map.csv"' in ast.get_source_segment(source, n)]
    if len(mapping_nodes) != 1:
        raise RuntimeError('Expected one top-level mapping write')
    mapping_write = ast.get_source_segment(source, mapping_nodes[0])
    source = replace_once(source, mapping_write, '# Mapping write is deferred until train mode.')
    source = replace_once(source, 'ROOT = Path(__file__).resolve().parents[2]', 'ROOT = _BEAUTY_ROOT')
    source = replace_once(source, 'SEED = 42', 'SEED = args.seed')
    source = replace_once(source, ast.get_source_segment(source, class_node(source)), 'from model_definition import SASRecControl')
    source = replace_once(source, 'best_ndcg = -1.0', SMOKE + '\n' + mapping_write + '\n\nbest_ndcg = -1.0')
    source = replace_once(source, 'config = {\n    "dataset": DATASET,', '''config = {
    "seed_extension": {
        "version": "beauty_control_seeds_v1",
        "original_source_sha256": _SOURCE_SHA256,
        "input_sha256": _INPUT_SHA256,
        "frozen_model_sha256": _FROZEN_MODEL_SHA256,
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preparation_manifest_sha256": hashlib.sha256(
            (_BEAUTY_STAGE / "manifests/beauty_seed_manifest.json").read_bytes()).hexdigest(),
        "algorithm_and_sorting_unchanged": True,
    },
    "dataset": DATASET,''')
    helpers = '\n\n'.join(inspect.getsource(f) for f in (plain_path, checked_file, checked_sha256))
    prelude = ('#!/usr/bin/env python3\n# Isolated Beauty Control seed extension.\n'
        'import argparse\nimport hashlib\nimport os\nimport stat\nfrom pathlib import Path, PurePosixPath\n\n'
        + helpers + '\n\n'
        + "parser = argparse.ArgumentParser(description='Beauty Control isolated seed extension')\n"
        + "parser.add_argument('--mode', choices=['smoke', 'train'], default='smoke')\n"
        + "parser.add_argument('--seed', type=int, choices=[42, 2026, 3407], default=42)\nargs = parser.parse_args()\n"
        + '_BEAUTY_ROOT = plain_path(Path(' + repr(str(root)) + '))\n'
        + '_BEAUTY_STAGE = plain_path(Path(' + repr(str(stage)) + '))\n'
        + "if Path(__file__).resolve(strict=False).parent != _BEAUTY_STAGE / 'scripts':\n"
        + "    raise RuntimeError('Run the original generated trainer from its original stage')\n"
        + '_SOURCE_SHA256 = ' + repr(SOURCE_SHA256) + '\n_INPUT_SHA256 = ' + repr(DATA_SHA256) + '\n'
        + 'for _relative, _expected in dict(_SOURCE_SHA256, **_INPUT_SHA256).items():\n'
        + '    checked_sha256(_BEAUTY_ROOT, _relative, _expected)\n'
        + '_FROZEN_MODEL_SHA256 = ' + repr(model_hash) + '\n'
        + "checked_sha256(_BEAUTY_STAGE, 'scripts/model_definition.py', _FROZEN_MODEL_SHA256)\n"
        + "print('INTEGRITY PASS: two original sources, four input CSVs and frozen model')\n\n")
    result = prelude + source
    ast.parse(result)
    return result


def write_new(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)


def prepare(root):
    root = plain_path(root)
    if not root.is_dir():
        raise ValueError('Project root must be a directory')
    verified = {}
    expected_files = dict(SOURCE_SHA256, **DATA_SHA256)
    expected_files.update({REFERENCE_PATH + '/' + name: value for name, value in REFERENCE_SHA256.items()})
    for relative, expected in expected_files.items():
        verified[relative] = checked_sha256(root, relative, expected)
    sources = {}
    for relative, expected in SOURCE_SHA256.items():
        blob = checked_file(root, relative).read_bytes()
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError('Source changed after hashing')
        sources[relative] = blob.decode('utf-8-sig')
        ast.parse(sources[relative])
    now = datetime.datetime.now(datetime.timezone.utc)
    stage = root / ('kais_beauty_control_seeds_v1_' + now.strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:12])
    model = frozen_model(sources[MODEL_SOURCE], sources[TRAINER_SOURCE])
    model_hash = hashlib.sha256(model.encode('utf-8')).hexdigest()
    trainer = generated_trainer(sources[TRAINER_SOURCE], root, stage, model_hash)
    generated = {'scripts/model_definition.py': model, 'scripts/train_beauty_control.py': trainer}
    hashes = {name: hashlib.sha256(text.encode('utf-8')).hexdigest() for name, text in generated.items()}
    manifest = {'version': 'beauty_control_seeds_v1', 'status': 'PREPARED_ONLY_NO_RESEARCH_EXECUTION',
        'created_utc': now.isoformat(), 'root': str(root), 'stage': str(stage), 'sources_sha256': SOURCE_SHA256,
        'data_sha256': DATA_SHA256, 'generated_sha256': hashes,
        'reference_run': {'path': REFERENCE_PATH, 'seed': 42, 'sha256': REFERENCE_SHA256},
        'scientific_change': 'None. Existing Beauty algorithm, model class AST, catalogue mapping, temporal sorting, panels and hyperparameters are preserved.',
        'operational_changes': ['Seed CLI 42/2026/3407, default42; mode CLI smoke/train, defaultsmoke added to a source with no CLI',
            'Smoke performs one backward pass and up to32 validation samples, then exits without creating an output run',
            'Unique stage, existing-output refusal, input/source/model hash checks before numerical imports',
            'Model class AST frozen from matching verified ML1M/Beauty definitions; no top-level original code imported'],
        'limits': ['No training, inference or checkpoint reading performed during preparation',
            'Old seed42 is preserved as historical reference; cross-seed panel/map/config checks remain necessary',
            'This is a seed extension, not the Sports/Toys ordering correction; no significance or full reproduction claim']}
    smoke = shlex.quote(sys.executable) + ' -u ' + shlex.quote(str(stage / 'scripts/train_beauty_control.py')) + ' --mode smoke --seed 42'
    train = [shlex.quote(sys.executable) + ' -u ' + shlex.quote(str(stage / 'scripts/train_beauty_control.py'))
             + ' --mode train --seed ' + str(seed) for seed in (2026, 3407)]
    readme = ('# Private Beauty Control seed extension\n\nPrepared only. Original files are read only.\n'
        'The original trainer had no smoke CLI; this copy defaults to a one-batch backward/32-sample validation smoke check.\n'
        'Smoke does not create a run directory. User-invoked smoke/training may use the GPU.\n'
        'Activate the original yancj environment and run the smoke command first:\n\n```bash\n' + smoke
        + '\n```\n\nAfter checking smoke success, run these two trainings sequentially:\n\n```bash\n'
        + '\n'.join(train) + '\n```\n\nNo commands are executed by this preparer. Existing output directories are refused.\n'
        'The original seed42 stays in ' + REFERENCE_PATH + '.\n'
        'Input/source/model hashes, exact source paths and scope are in manifests/beauty_seed_manifest.json.\n')
    stage.mkdir(exist_ok=False)
    for child in ('scripts', 'outputs', 'verification', 'manifests'):
        (stage / child).mkdir(exist_ok=False)
    for relative, text in generated.items():
        write_new(stage / relative, text)
    write_new(stage / 'manifests/beauty_seed_manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    write_new(stage / 'verification/PREPARED.json', json.dumps({'status': 'PREPARED_ONLY', 'ast_syntax': 'PASS',
        'model_class_ast': 'UNCHANGED', 'verified_inputs': verified, 'research_execution': False}, indent=2) + '\n')
    write_new(stage / 'README.md', readme)
    print('WORKDIR=' + str(stage))
    print('STATUS=PREPARED_ONLY_NO_TRAINING')
    print('SMOKE_COMMAND=' + smoke)
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.root.expanduser())


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, SyntaxError) as exc:
        print('PREPARATION_FAILED=' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
