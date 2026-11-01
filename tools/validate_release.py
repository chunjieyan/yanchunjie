#!/usr/bin/env python3
"""Dependency-free release checks. No datasets, network, model imports or GPU jobs."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {'.git', '__pycache__', 'outputs', '.venv', '.pytest_cache'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def release_files():
    return {
        path.relative_to(ROOT).as_posix(): path
        for path in ROOT.rglob('*')
        if path.is_file()
        and not any(part in IGNORED_PARTS for part in path.relative_to(ROOT).parts)
        and path.name != 'MANIFEST.sha256.json'
    }


def check():
    failures = []
    manifest = json.loads((ROOT / 'MANIFEST.sha256.json').read_text(encoding='utf-8-sig'))
    files = release_files()
    expected = manifest['files']
    for name, entry in expected.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or name not in files:
            failures.append({'path': name, 'reason': 'missing or outside repository'})
        elif digest(path) != entry['sha256']:
            failures.append({'path': name, 'reason': 'SHA256 differs'})
    for name in sorted(files.keys() - expected.keys()):
        failures.append({'path': name, 'reason': 'not covered by manifest'})
    syntax_count = 0
    for name, path in files.items():
        if path.suffix == '.py':
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=name, feature_version=(3, 10))
                syntax_count += 1
            except (SyntaxError, UnicodeError) as error:
                failures.append({'path': name, 'reason': str(error)})
        if path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.pkl', '.zip', '.gz', '.exe', '.dll'}:
            failures.append({'path': name, 'reason': 'unexpected data/model/archive/runtime file'})
        if path.name in {'.env', 'credentials', 'id_rsa', 'id_ed25519'}:
            failures.append({'path': name, 'reason': 'unexpected private configuration file'})
    sources = json.loads((ROOT / 'provenance/source_index.json').read_text(encoding='utf-8-sig'))['files']
    for record in sources:
        name = record.get('repository_path') or record['package_relative_path']
        if name not in files or digest(files[name]) != record['sha256']:
            failures.append({'path': name, 'reason': 'archived scientific source differs'})
    cli_results = []
    for script in ['reproduce.py', 'replay_pipeline.py', 'replay_training.py', 'replay_evaluation.py', 'recompute_statistics.py', 'download_data.py']:
        result = subprocess.run([sys.executable, str(ROOT / 'tools' / script), '--help'], capture_output=True, text=True, timeout=30)
        cli_results.append({'script': script, 'exit_code': result.returncode})
        if result.returncode:
            failures.append({'path': 'tools/' + script, 'reason': 'CLI --help failed'})
    result = subprocess.run([sys.executable, str(ROOT / 'tools/replay_pipeline.py'), '--preflight'], capture_output=True, text=True, timeout=60)
    try:
        pipeline = json.loads(result.stdout)
    except json.JSONDecodeError:
        pipeline = {'status': 'FAIL_PIPELINE_OUTPUT', 'exit_code': result.returncode}
    if result.returncode:
        failures.append({'path': 'tools/replay_pipeline.py', 'reason': 'full matrix/source preflight failed'})
    return {
        'status': 'PASS_RELEASE_OFFLINE_CHECKS' if not failures else 'FAIL_RELEASE_CHECKS',
        'files_checked': len(files),
        'python_files_parsed': syntax_count,
        'original_source_records_checked': len(sources),
        'cli_help_checks': cli_results,
        'pipeline': pipeline,
        'failures': failures,
        'network_access': False,
        'datasets_or_checkpoints_required': False,
        'training_or_inference_executed': False,
        'fresh_GPU_reproduction_verified': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON report under ignored outputs/.')
    args = parser.parse_args()
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to((ROOT / 'outputs').resolve()):
            parser.error('--output must be under repository outputs/')
    report = check()
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding='utf-8')
    print(text, end='')
    return bool(report['failures'])


if __name__ == '__main__':
    sys.exit(main())
