#!/usr/bin/env python3
"""Prepare and run nine Amazon LBSP trainings, then collect saved evidence.

Without --run this prints a plan only. Training runs sequentially in a new
stage; previous research outputs and the audited smoke stage are read only.
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
PREPARER_PATH = HERE / 'prepare_lbsp_amazon_train.py'
COLLECTOR_PATH = HERE / 'collect_lbsp_amazon_train.py'
INPUTS_PATH = HERE / 'lbsp_amazon_train_inputs_20261023.json'
INPUTS_SHA256 = 'aad9c9a404f04df9b355d6aa6b4ca0b26905d9763ee9fb3915c9f3e102682ff2'
STAGE_PREFIX = 'kais_lbsp_amazon_train_v1_'
SEEDS = (42, 2026, 3407)
TAGS = ('beauty', 'sports', 'toys')
PLAN = tuple((tag, seed) for tag in TAGS for seed in SEEDS)
SCRIPTS = {'scripts/model_definition.py'} | {'scripts/train_' + tag + '_lbsp.py' for tag in TAGS}


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
    path = plain(PREPARER_PATH)
    spec = importlib.util.spec_from_file_location('reviewed_lbsp_amazon_train_preparer', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_state(path, state):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(state, handle, indent=2)
        handle.write('\n')
    temporary.replace(path)


def execute(root):
    root = plain(root)
    if not root.is_dir():
        raise ValueError('Project root must be a directory')
    if not plain(COLLECTOR_PATH).is_file():
        raise ValueError('Upload collect_lbsp_amazon_train.py next to this driver')
    if sha(INPUTS_PATH) != INPUTS_SHA256:
        raise RuntimeError('The reviewed formal-training input manifest changed')
    inputs = json.loads(INPUTS_PATH.read_text(encoding='utf-8'))
    previous = list(root.glob(STAGE_PREFIX + '*'))
    if previous:
        raise RuntimeError('An earlier Amazon LBSP training stage needs inspection/collection before another start: ' + str(previous[0]))
    stage = plain(load_preparer().prepare(root, INPUTS_PATH))
    print('LBSP_TRAIN_STAGE=' + str(stage), flush=True)
    if stage.parent != root or not stage.name.startswith(STAGE_PREFIX):
        raise RuntimeError('Prepared stage is outside the expected project location')
    manifest = json.loads((stage / 'manifests/lbsp_amazon_train_manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('version') != 'lbsp_amazon_train_v1' or manifest.get('root') != str(root)
            or manifest.get('stage') != str(stage) or manifest.get('inputs') != inputs
            or manifest.get('inputs_sha256') != INPUTS_SHA256):
        raise RuntimeError('Unexpected prepared stage identity')
    if set(manifest['generated_sha256']) != SCRIPTS:
        raise RuntimeError('Unexpected prepared script set')
    for relative, expected in manifest['generated_sha256'].items():
        if sha(stage / relative) != expected:
            raise RuntimeError('Generated file hash mismatch: ' + relative)
    verification = plain(stage / 'verification')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
    status_file = verification / ('lbsp_amazon_train_driver_' + stamp + '.json')
    state = {'version': 'lbsp_amazon_train_v1', 'stage': str(stage), 'python': sys.executable,
             'inputs_sha256': INPUTS_SHA256, 'formal_seeds': list(SEEDS),
             'status': 'RUNNING', 'runs': {}}
    save_state(status_file, state)
    result, process = 0, None
    try:
        for tag, seed in PLAN:
            label = tag + '_seed' + str(seed)
            script = stage / 'scripts' / ('train_' + tag + '_lbsp.py')
            logfile = verification / (label + '_train_' + stamp + '.log')
            command = [sys.executable, '-u', str(script), '--mode', 'train', '--seed', str(seed)]
            record = {'command': command, 'log': str(logfile),
                      'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'exit_code': None}
            state['runs'][label] = record
            save_state(status_file, state)
            print('TRAIN=' + shlex.join(command), flush=True)
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
            completion = json.loads(plain(verification / ('completed_' + label + '.json')).read_text(encoding='utf-8'))
            if (completion.get('status') != 'COMPLETED' or completion.get('tag') != tag
                    or completion.get('seed') != seed):
                raise RuntimeError('Training exited zero without its expected completion record: ' + label)
            record['completion_verified'] = True
            save_state(status_file, state)
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
        complete = (result == 0 and len(state['runs']) == 9
                    and all(r.get('completion_verified') is True for r in state['runs'].values()))
        state['status'] = 'ALL_NINE_TRAININGS_EXITED_ZERO' if complete else 'STOPPED_INCOMPLETE'
        state['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_state(status_file, state)
        print('LBSP_TRAIN_DRIVER_STATUS=' + state['status'], flush=True)
        collected = subprocess.run([sys.executable, str(COLLECTOR_PATH), '--stage', str(stage)], check=False)
        if collected.returncode:
            result = result or collected.returncode
            print('COLLECTION_EXIT_CODE=' + str(collected.returncode), file=sys.stderr, flush=True)
    return 0 if result == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--run', action='store_true', help='Prepare an isolated stage and train all nine dataset/seed combinations')
    args = parser.parse_args()
    if not args.run:
        plain(args.root)
        print('PLAN_ONLY_NO_FILES_OR_GPU: Beauty 42/2026/3407; Sports 42/2026/3407; Toys 42/2026/3407; sequential formal training; collect saved metadata without model weights')
        return 0
    return execute(args.root)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError, SyntaxError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(2)
