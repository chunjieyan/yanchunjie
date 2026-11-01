#!/usr/bin/env python3
"""Run four guarded X3 smokes, then twelve new shuffle-training jobs sequentially.

Without --run, only print a plan. Research execution is confined to generated
server subprocesses in a new stage. Existing 24 model outputs remain read only.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve(strict=False).parent
PREPARER_PATH = HERE / 'prepare_x3_shuffle.py'
COLLECTOR_PATH = HERE / 'collect_x3_shuffle.py'
INPUTS_PATH = HERE / 'x3_shuffle_inputs_20261025.json'
INPUTS_SHA256 = '3788d6ce525966c74663a7e2e61b4bc20c9eaac5d2e6c7a5a8e33fc46f4dfce3'
STAGE_PREFIX = 'kais_x3_shuffle_v1_'
MANIFEST = 'manifests/x3_shuffle_manifest.json'
TAGS = ('ml1m', 'beauty', 'sports', 'toys')
SEEDS = (42, 2026, 3407)
PLAN = tuple((tag, seed) for tag in TAGS for seed in SEEDS)
SCRIPTS = {'scripts/model_definition.py'} | {'scripts/train_' + tag + '_shuffle.py' for tag in TAGS}
PROOF_FLAGS = ('persistent_worker_outputs_match_main', 'global_rng_unchanged_by_augmentation_and_probe',
               'all_visible_bucket_event_multisets_preserved', 'target_and_cutoff_fixed',
               'target_event_future_excluded', 'original_visible_window_and_length_preserved',
               'repeated_target_item_counts_preserved', 'cross_window_event_priorities_shared')


def plain(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected an absolute path without ..')
    for component in (path,) + tuple(path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Symbolic link/reparse point rejected: ' + str(component))
    return path.resolve(strict=False)


def sha(path):
    return hashlib.sha256(plain(path).read_bytes()).hexdigest()


def load_preparer():
    spec = importlib.util.spec_from_file_location('reviewed_x3_shuffle_preparer', plain(PREPARER_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_state(path, state):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(state, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_smoke(stage, manifest, inputs, tag):
    path = plain(stage / 'verification' / ('smoke_' + tag + '_seed42.json'))
    report = json.loads(path.read_text(encoding='utf-8'))
    dataset = inputs['datasets'][tag]
    expected = {'version': 'x3_shuffle_v1', 'tag': tag, 'dataset': dataset['dataset'], 'seed': 42,
                'mode': 'smoke', 'status': 'SMOKE_PASS', 'input_manifest_sha256': INPUTS_SHA256,
                'preparation_manifest_sha256': sha(stage / MANIFEST),
                'trainer_sha256': manifest['generated_sha256']['scripts/train_' + tag + '_shuffle.py'],
                'frozen_model_sha256': manifest['generated_sha256']['scripts/model_definition.py'],
                'optimizer_steps': 0, 'formal_training_started': False, 'weights_read': False,
                'weights_saved': False, 'test_inference_performed': False,
                'formal_output_created': False, 'original_hashes_verified_after': True}
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError('Smoke identity/scope failed: ' + tag + '/' + key)
    loss, gradients = report.get('smoke_loss'), report.get('finite_gradient_count')
    if (isinstance(loss, bool) or not isinstance(loss, (float, int)) or not math.isfinite(loss)
            or loss <= 0 or isinstance(gradients, bool) or not isinstance(gradients, int) or gradients <= 0
            or report.get('all_gradients_finite') is not True):
        raise RuntimeError('Smoke backward/loss proof failed: ' + tag)
    counts = report.get('counts', {})
    for field, planned in (('num_items', 'expected_items'), ('training_rows', 'expected_training_rows'),
                           ('training_positions', 'expected_training_positions'),
                           ('validation_samples', 'expected_validation'), ('test_samples', 'expected_test')):
        if counts.get(field) != dataset[planned]:
            raise RuntimeError('Smoke frozen count differs: ' + tag + '/' + field)
    proof = report.get('invariants', {})
    if (any(proof.get(key) is not True for key in PROOF_FLAGS)
            or proof.get('epoch_checks') != [1, 2] or proof.get('worker_epoch_checks') != [1, 2]
            or proof.get('probe_workers') != (4 if tag in ('ml1m', 'beauty') else 0)
            or proof.get('checked_visible_windows') != 2 * min(512, dataset['expected_training_positions'])
            or not isinstance(proof.get('probe_samples_per_epoch'), int)
            or not 0 < proof['probe_samples_per_epoch'] <= 32):
        raise RuntimeError('Smoke visible-history/epoch/worker/global-RNG proof failed: ' + tag)
    if any((stage / 'outputs').iterdir()):
        raise RuntimeError('Smoke created a formal output before all four checks completed')
    return path


def execute(root):
    root = plain(root)
    if not root.is_dir() or not plain(COLLECTOR_PATH).is_file():
        raise ValueError('Project root or collect_x3_shuffle.py is missing')
    if sha(INPUTS_PATH) != INPUTS_SHA256:
        raise RuntimeError('The reviewed X3 input seal changed')
    previous = list(root.glob(STAGE_PREFIX + '*'))
    if previous:
        raise RuntimeError('An existing X3 stage needs inspection/collection before another start: ' + str(previous[0]))
    inputs = json.loads(INPUTS_PATH.read_text(encoding='utf-8'))
    stage = plain(load_preparer().prepare(root, INPUTS_PATH))
    print('X3_SHUFFLE_STAGE=' + str(stage), flush=True)
    manifest = json.loads(plain(stage / MANIFEST).read_text(encoding='utf-8'))
    if (stage.parent != root or not stage.name.startswith(STAGE_PREFIX)
            or manifest.get('version') != 'x3_shuffle_v1' or manifest.get('root') != str(root)
            or manifest.get('stage') != str(stage) or manifest.get('inputs') != inputs
            or manifest.get('inputs_sha256') != INPUTS_SHA256 or set(manifest['generated_sha256']) != SCRIPTS):
        raise RuntimeError('Prepared X3 stage identity or script inventory differs')
    for relative, digest in manifest['generated_sha256'].items():
        if sha(stage / relative) != digest:
            raise RuntimeError('Generated file hash differs: ' + relative)
    verification = plain(stage / 'verification')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
    status_path = verification / ('x3_shuffle_driver_' + stamp + '.json')
    state = {'version': 'x3_shuffle_v1', 'root': str(root), 'stage': str(stage), 'python': sys.executable,
             'inputs_sha256': INPUTS_SHA256, 'training_seeds': list(SEEDS), 'status': 'RUNNING',
             'smokes': {}, 'runs': {}}
    save_state(status_path, state)
    result, process = 0, None
    try:
        tasks = [('smoke', tag, 42) for tag in TAGS] + [('train', tag, seed) for tag, seed in PLAN]
        for mode, tag, seed in tasks:
            if mode == 'train' and (len(state['smokes']) != 4 or not all(r.get('summary_verified') is True for r in state['smokes'].values())):
                raise RuntimeError('All four smokes must be verified before any formal training')
            label = tag + '_seed' + str(seed)
            logfile = verification / (label + '_' + mode + '_' + stamp + '.log')
            command = [sys.executable, '-u', str(stage / 'scripts' / ('train_' + tag + '_shuffle.py')),
                       '--mode', mode, '--seed', str(seed)]
            record = {'command': command, 'log': str(logfile), 'exit_code': None,
                      'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            (state['smokes'] if mode == 'smoke' else state['runs'])[tag if mode == 'smoke' else label] = record
            save_state(status_path, state)
            print(mode.upper() + '=' + shlex.join(command), flush=True)
            with logfile.open('x', encoding='utf-8', newline='\n') as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace', bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                result = process.wait()
            record['exit_code'] = result
            record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            process = None
            if result:
                break
            if mode == 'smoke':
                summary = verify_smoke(stage, manifest, inputs, tag)
                record.update(summary_verified=True, summary=str(summary), summary_sha256=sha(summary))
            else:
                completed_path = plain(verification / ('completed_' + label + '.json'))
                completed = json.loads(completed_path.read_text(encoding='utf-8'))
                if (completed.get('status') != 'COMPLETED' or completed.get('mode') != 'train'
                        or completed.get('tag') != tag or completed.get('seed') != seed
                        or completed.get('input_manifest_sha256') != INPUTS_SHA256):
                    raise RuntimeError('Training exited zero without its bound completion record: ' + label)
                record.update(completion_verified=True, completion=str(completed_path), completion_sha256=sha(completed_path))
            save_state(status_path, state)
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
        complete = (result == 0 and len(state['smokes']) == 4 and len(state['runs']) == 12
                    and all(r.get('summary_verified') is True for r in state['smokes'].values())
                    and all(r.get('completion_verified') is True for r in state['runs'].values()))
        state['status'] = 'ALL_FOUR_SMOKES_AND_TWELVE_TRAININGS_EXITED_ZERO' if complete else 'STOPPED_INCOMPLETE'
        state['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_state(status_path, state)
        print('X3_SHUFFLE_DRIVER_STATUS=' + state['status'], flush=True)
        collected = subprocess.run([sys.executable, str(COLLECTOR_PATH), '--stage', str(stage)], check=False)
        if collected.returncode:
            result = result or collected.returncode
            print('COLLECTION_EXIT_CODE=' + str(collected.returncode), file=sys.stderr, flush=True)
    return 0 if result == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--run', action='store_true', help='Prepare, verify four smokes, then train twelve new shuffle models')
    args = parser.parse_args()
    if not args.run:
        print('PLAN_ONLY_NO_FILES_OR_GPU: four dataset seed42 smokes; only after all pass, four datasets x seeds42/2026/3407 formal training; sequential; collect saved evidence')
        return 0
    return execute(args.root)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, SyntaxError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(2)
