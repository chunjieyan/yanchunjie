#!/usr/bin/env python3
"""Run three Amazon LBSP mechanism smoke checks; no formal training or weight access.

Without --run, print the plan without creating files or starting inference.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve(strict=False).parent
PREPARER_PATH = HERE / 'prepare_lbsp_amazon_smoke.py'
COLLECTOR_PATH = HERE / 'collect_lbsp_amazon_smoke.py'
INPUTS_PATH = HERE / 'lbsp_amazon_smoke_inputs_20261023.json'
INPUTS_SHA256 = 'c65912d443b01d2f76ed164e57638a99ad1b1972394d542b5f22fe1eacd92265'
STAGE_PREFIX = 'kais_lbsp_amazon_smoke_v1_'
PLAN = (('beauty', 42), ('sports', 42), ('toys', 42))
SCRIPTS = {'scripts/model_definition.py'} | {'scripts/smoke_' + tag + '_lbsp.py' for tag in ('beauty', 'sports', 'toys')}


def plain(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected an absolute path without ..: ' + str(path))
    for component in (path,) + tuple(path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Symbolic link/reparse point rejected: ' + str(component))
    return path.resolve(strict=False)


def sha(path):
    return hashlib.sha256(plain(path).read_bytes()).hexdigest()


def load_preparer():
    spec = importlib.util.spec_from_file_location('reviewed_lbsp_amazon_smoke_preparer', plain(PREPARER_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def execute(root):
    root = plain(root)
    if not root.is_dir():
        raise ValueError('Project root must be a directory')
    if not plain(COLLECTOR_PATH).is_file():
        raise ValueError('Upload collect_lbsp_amazon_smoke.py next to this driver')
    if sha(INPUTS_PATH) != INPUTS_SHA256:
        raise RuntimeError('The reviewed LBSP smoke input manifest changed')
    inputs = json.loads(INPUTS_PATH.read_text(encoding='utf-8'))
    previous = [p for p in root.glob(STAGE_PREFIX + '*') if p.is_dir() or p.is_symlink()]
    if previous:
        raise RuntimeError('An earlier LBSP smoke stage needs inspection/collection before another start: ' + str(previous[0]))
    stage = plain(load_preparer().prepare(root, INPUTS_PATH))
    if stage.parent != root or not stage.name.startswith(STAGE_PREFIX):
        raise RuntimeError('Prepared stage is outside the expected project location')
    manifest = json.loads((stage / 'manifests/lbsp_amazon_smoke_manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('version') != 'lbsp_amazon_smoke_v1' or manifest.get('root') != str(root)
            or manifest.get('stage') != str(stage) or manifest.get('inputs') != inputs
            or manifest.get('inputs_sha256') != INPUTS_SHA256):
        raise RuntimeError('Unexpected prepared stage identity')
    if set(manifest['generated_sha256']) != SCRIPTS:
        raise RuntimeError('Unexpected generated evaluation script set')
    for relative, expected in manifest['generated_sha256'].items():
        if sha(stage / relative) != expected:
            raise RuntimeError('Generated file hash mismatch: ' + relative)
    verification = plain(stage / 'verification')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
    status_file = verification / ('lbsp_amazon_smoke_driver_' + stamp + '.json')
    state = {'stage': str(stage), 'python': sys.executable, 'status': 'RUNNING', 'runs': {}, 'training_performed': False}
    with status_file.open('x', encoding='utf-8') as handle:
        json.dump(state, handle, indent=2)
    result, process = 0, None
    try:
        for tag, seed in PLAN:
            label = tag + '_seed' + str(seed)
            script = stage / 'scripts' / ('smoke_' + tag + '_lbsp.py')
            logfile = verification / (label + '_smoke_' + stamp + '.log')
            command = [sys.executable, '-u', str(script), '--mode', 'smoke']
            record = {'command': command, 'log': str(logfile),
                      'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'exit_code': None}
            state['runs'][label] = record
            status_file.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
            print('SMOKE=' + shlex.join(command), flush=True)
            with logfile.open('x', encoding='utf-8', newline='\n') as handle:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace', bufsize=1)
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(line, end='', flush=True)
                result = process.wait()
            record['exit_code'] = result
            record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            process = None
            if result:
                break
    except BaseException as exc:
        result = 1
        state['error'] = type(exc).__name__ + ': ' + str(exc)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print('DRIVER_ERROR=' + state['error'], file=sys.stderr, flush=True)
    finally:
        state['status'] = 'ALL_THREE_SMOKES_EXITED_ZERO' if result == 0 and len(state['runs']) == 3 else 'STOPPED_INCOMPLETE'
        state['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status_file.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        print('LBSP_SMOKE_DRIVER_STATUS=' + state['status'], flush=True)
        collected = subprocess.run([sys.executable, str(COLLECTOR_PATH), '--stage', str(stage)], check=False)
        if collected.returncode:
            result = result or collected.returncode
            print('COLLECTION_EXIT_CODE=' + str(collected.returncode), file=sys.stderr)
    return 0 if result == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--run', action='store_true', help='Prepare isolated copies and run Beauty, Sports and Toys smoke checks')
    args = parser.parse_args()
    if not args.run:
        plain(args.root)
        print('PLAN_ONLY_NO_FILES_OR_GPU: Beauty, Sports, Toys LBSP smoke checks; random initialized models, gradient/parity/invariance only; no optimizer steps, weights or formal training')
        return 0
    return execute(args.root)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError, SyntaxError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(2)
