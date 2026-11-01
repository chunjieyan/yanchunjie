#!/usr/bin/env python3
"""Static KAIS release-readiness inventory. Does not run research scripts.

Requires Python >=3.8 and only its standard library. New reports are written to
one unique directory under ROOT/reproducibility; existing inputs are read only.
"""
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

DATASETS = ('ml1m', 'movielens', 'beauty', 'sports', 'toys')
EXCLUDED = ('.git', '.ssh', '.aws', '__pycache__', 'step39_mts_cql')
PREPROCESS = re.compile(r'^(25_2_|26_|28_1_|31_1_|33_2_|35_5_|36_(?:2|4|6|7|8|9)_)')
TEXT_LIMIT = 4 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def contained(path, root):
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def safe_files(directory, root):
    """Walk a named subtree without following directory/file symlinks."""
    if not directory.is_dir() or directory.is_symlink() or not contained(directory, root):
        return
    for base, dirs, names in os.walk(str(directory), followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not d.startswith('.')
                         and not (Path(base) / d).is_symlink())
        for name in sorted(names):
            path = Path(base) / name
            if name.startswith('.') or path.is_symlink() or not contained(path, root):
                continue
            if path.is_file():
                yield path


def metadata(path, root, hash_small=False):
    stat = path.stat()
    do_hash = hash_small and stat.st_size <= TEXT_LIMIT
    return {'path': path.relative_to(root).as_posix(), 'bytes': stat.st_size,
            'mtime_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            'sha256': sha256(path) if do_hash else '',
            'hash_status': 'computed' if do_hash else 'not_computed'}


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def source_flags(path, root):
    source = path.read_text(encoding='utf-8-sig', errors='replace')
    row = {'path': path.relative_to(root).as_posix(), 'syntax': 'ok',
           'main_guard': False, 'argparse_detected': False, 'exec_detected': False,
           'write_calls_detected': False, 'dataset_literal': '', 'seed_literal': '',
           'imports': '', 'static_path_literals': ''}
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        row['syntax'] = 'parse_error_line_%s' % exc.lineno
        return row
    imports = set()
    literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split('.')[0])
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            if '__name__' in ast.dump(node.test) and '__main__' in ast.dump(node.test):
                row['main_guard'] = True
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ''
            if name == 'exec': row['exec_detected'] = True
            if name in ('mkdir', 'to_csv', 'write_text', 'write_bytes', 'save', 'savefig'):
                row['write_calls_detected'] = True
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('DATASET', 'SEED', 'MODEL_SEED'):
                    if isinstance(node.value, ast.Constant):
                        field = 'dataset_literal' if target.id == 'DATASET' else 'seed_literal'
                        row[field] = str(node.value.value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if (len(value) < 220 and not re.search(r'(?i)(token|password|secret|https?://|ssh://|@)', value)
                    and (value.endswith('.py') or value.startswith(('processed/', 'audit/', '/root/autodl-tmp/')))):
                literals.append(value)
    row['argparse_detected'] = 'argparse' in imports
    row['imports'] = ';'.join(sorted(imports))
    row['static_path_literals'] = ';'.join(sorted(set(literals))[:30])
    return row


def collect_environment():
    packages = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get('Name')
        if name:
            packages.append({'name': name, 'version': dist.version})
    result = {'python_version': sys.version, 'python_executable': sys.executable,
              'platform': platform.platform(), 'machine': platform.machine(),
              'packages': sorted(packages, key=lambda x: x['name'].lower()),
              'note': 'Package names/versions only; no pip direct URLs or environment-variable dump.'}
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        result['gpu_query_status'] = gpu.returncode
        result['gpu_name_memory_driver'] = gpu.stdout.strip() if gpu.returncode == 0 else 'query_failed'
    except (OSError, subprocess.TimeoutExpired) as exc:
        result['gpu_query_status'] = type(exc).__name__
        result['gpu_name_memory_driver'] = 'unavailable'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Existing research project root')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not (root / 'step38_kais_retrain/scripts').is_dir():
        parser.error('Cannot find ROOT/step38_kais_retrain/scripts. Check --root; no audit directory was created.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    parent = root / 'reproducibility'
    if parent.is_symlink() or not contained(parent, root):
        parser.error('Refusing a report parent outside the project root.')
    parent.mkdir(exist_ok=True)
    report = parent / ('g2_server_audit_' + stamp + '_' + uuid4().hex[:8])
    report.mkdir(exist_ok=False)
    print('Report directory:', report, flush=True)

    environment = collect_environment()
    (report / 'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding='utf-8')
    sources = []
    for path in safe_files(root / 'step38_kais_retrain/scripts', root):
        if path.suffix == '.py' and 'kuairec' not in path.name.lower(): sources.append(path)
    for path in safe_files(root / 'audit', root):
        if path.suffix == '.py' and PREPROCESS.match(path.name): sources.append(path)
    paper_dirs = [root / 'paper_20261022', root / 'paper_20261022/kais_english_revised']
    for directory in paper_dirs:
        for relative in ('build_manuscript.py', 'analyze_saved_evidence.py', 'support/analyze_saved_evidence.py'):
            path = directory / relative
            if path.is_file() and not path.is_symlink() and contained(path, root): sources.append(path)
    sources = sorted(set(sources))
    source_rows = [metadata(path, root, True) for path in sources]
    flags = [source_flags(path, root) for path in sources]
    write_csv(report / 'source_inventory.csv', source_rows,
              ['path', 'bytes', 'mtime_utc', 'sha256', 'hash_status'])
    write_csv(report / 'script_flags.csv', flags,
              ['path', 'syntax', 'main_guard', 'argparse_detected', 'exec_detected',
               'write_calls_detected', 'dataset_literal', 'seed_literal', 'imports', 'static_path_literals'])

    runs = []
    outputs = root / 'step38_kais_retrain/outputs'
    if outputs.is_dir() and not outputs.is_symlink():
        for directory in sorted(outputs.iterdir()):
            name = directory.name.lower()
            if not directory.is_dir() or directory.is_symlink() or 'kuairec' in name: continue
            if not any(key in name for key in DATASETS) and 'four_dataset' not in name: continue
            for path in safe_files(directory, root):
                if path.suffix.lower() not in ('.json', '.csv', '.pt', '.pth', '.ckpt'): continue
                row = metadata(path, root, path.suffix == '.json')
                row['run'] = directory.name
                row['kind'] = 'checkpoint' if path.suffix in ('.pt', '.pth', '.ckpt') else 'metadata' if path.suffix == '.json' else 'csv'
                runs.append(row)
    write_csv(report / 'run_inventory.csv', runs,
              ['run', 'kind', 'path', 'bytes', 'mtime_utc', 'sha256', 'hash_status'])

    data_rows = []
    for subdirectory in ('diagnostic_a', 'diagnostic_a_splits', 'recent_bucketset_r1',
                         'strict_diagnostic_a', 'strict_last_bucket_only'):
        directory = root / 'processed' / subdirectory
        for path in safe_files(directory, root):
            rel = path.relative_to(directory).as_posix().lower()
            if 'kuairec' in rel or not any(key in rel for key in DATASETS): continue
            if path.suffix.lower() in ('.csv', '.json', '.jsonl', '.npz', '.npy'):
                data_rows.append(metadata(path, root, path.suffix == '.json'))
    # Original data are enumerated by filename and size only, never read or bundled.
    for path in safe_files(root / 'data', root):
        rel = path.relative_to(root / 'data').as_posix().lower()
        if 'kuairec' in rel or not any(key in rel for key in DATASETS): continue
        if path.suffix.lower() in ('.gz', '.zip', '.csv', '.dat', '.json', '.jsonl'):
            data_rows.append(metadata(path, root, False))
    write_csv(report / 'data_inventory.csv', data_rows,
              ['path', 'bytes', 'mtime_utc', 'sha256', 'hash_status'])

    evidence = []
    evidence_names = ('support/evidence_index.json', 'evidence_index.json',
                      'support/analysis_results.json', 'support/main_evidence.json',
                      'support/dataset_sizes.json', 'support/accuracy_values.json')
    for directory in paper_dirs:
        for relative in evidence_names:
            path = directory / relative
            if path.is_file() and not path.is_symlink() and contained(path, root): evidence.append(path)
    step36 = root / 'reproducibility/step36'
    for path in safe_files(step36, root):
        if re.match(r'^step36_(?:2|4|6|7|8|9)_.*manifest\.json$', path.name): evidence.append(path)
    evidence_rows = [metadata(path, root, True) for path in sorted(set(evidence))]
    write_csv(report / 'evidence_inventory.csv', evidence_rows,
              ['path', 'bytes', 'mtime_utc', 'sha256', 'hash_status'])
    dependencies = ['step38_kais_retrain/scripts/38_1_train_ml1m_sasrec_bs128.py',
                    'audit/33_2_train_movielens_r1_seed42.py']
    missing_dependencies = [path for path in dependencies if not (root / path).is_file()]
    missing_packages = sorted({'torch', 'numpy', 'pandas'} - {item['name'].lower() for item in environment['packages']})
    result = {'audit_type': 'static_inventory_only', 'root': str(root), 'report_directory': str(report),
              'source_files': len(sources), 'run_files': len(runs), 'data_files': len(data_rows),
              'evidence_files': len(evidence_rows), 'missing_known_source_dependencies': missing_dependencies,
              'missing_expected_packages': missing_packages,
              'syntax_parse_errors': [row['path'] for row in flags if row['syntax'] != 'ok'],
              'hash_policy': 'Source and small JSON only. Large data, predictions and checkpoints are NOT verified.',
              'training_executed': False, 'research_scripts_executed': False,
              'original_files_modified': False, 'network_uploads': False,
              'reproducibility_verified': False, 'release_ready': False}
    (report / 'audit_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    has_torch = next((x['version'] for x in environment['packages'] if x['name'].lower() == 'torch'), 'NOT FOUND')
    summary = f'''# G2 server inventory / G2 服务器盘点

Project root: `{root}`
Python executable: `{sys.executable}`
Python version: {platform.python_version()}
PyTorch package version: {has_torch}
GPU name, memory, driver: {environment['gpu_name_memory_driver']}

## Counts

- Candidate source files: {len(sources)}
- Existing result/checkpoint/metadata files: {len(runs)}
- Original/processed data files enumerated: {len(data_rows)}
- Existing evidence/manifest files: {len(evidence_rows)}
- Missing known source dependencies: {json.dumps(missing_dependencies, ensure_ascii=False)}
- Expected packages not found: {json.dumps(missing_packages, ensure_ascii=False)}
- Python syntax parse errors: {len(result['syntax_parse_errors'])}

## What this run did

Statically read source, listed files, recorded installed package names/versions,
and queried GPU hardware. It did NOT import or execute research scripts,
train models, load checkpoints, regenerate predictions, edit the manuscript,
or upload files. Reports are written only in this newly created directory.

## Limits / 本次尚未完成

This is an inventory, not a reproduction test or release certification.
Data, checkpoints and large predictions have NOT been hashed or verified.
Existing manifest files were indexed and hashed, but their referenced inputs
were NOT recomputed or compared. A missing source marker in static analysis
is not proof a script is safe to execute.

## Next steps / 下一步

1. Confirm that this is the Python environment used for the original training.
2. Review missing dependencies and dataset/method/seed coverage in run_inventory.csv.
3. Build a separate release candidate with the full preprocessing/model dependency chain.
4. Add configuration, actual reproduction commands, data hashes and documentation.
5. Validate in a clean directory before creating a public release or filling code/DOI declarations.

Read SUMMARY.md first, then return the small g2_audit_reports.zip if further
assistance is needed. The ZIP contains reports only, no source code or data.
'''
    (report / 'SUMMARY.md').write_text(summary, encoding='utf-8')
    archive = report / 'g2_audit_reports.zip'
    with ZipFile(archive, 'x', ZIP_DEFLATED) as bundle:
        for path in sorted(report.iterdir()):
            if path.name != archive.name and path.is_file(): bundle.write(path, path.name)
    print('\n' + summary)
    print('REPORT_ZIP=' + str(archive))
    return 0


if __name__ == '__main__':
    sys.exit(main())
