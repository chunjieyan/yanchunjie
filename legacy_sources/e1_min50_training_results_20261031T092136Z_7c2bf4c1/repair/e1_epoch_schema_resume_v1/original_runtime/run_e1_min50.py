#!/usr/bin/env python3
"""E1: plan by default; explicit server-only preflight and19 isolated fresh trainings."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from e1_common import HERE, PREFIX, lock, make_attempt, output_path, preflight, require, sha, read_json, validate_output, write_json, checked_native


def run_child(command, logfile):
    process = None
    try:
        with logfile.open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
        require(code == 0, 'Child failed with exit ' + str(code) + '; see ' + str(logfile))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/time_uncertainty_rec'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true', help='Verify frozen inputs only; no model imports/GPU')
    mode.add_argument('--run', action='store_true', help='Run smokes and19 trainings sequentially on CUDA')
    parser.add_argument('--stage', type=Path, help='Resume DRIVER by keeping completed jobs and creating fresh attempts for failed jobs')
    args = parser.parse_args()
    policy = lock()
    if not args.run and not args.preflight:
        print('PLAN_ONLY: no project inputs, checkpoints or GPU accessed.')
        print('Policy: monitor all epochs; stop iff epoch>=50 and bad_epochs>=15; cap200; select validation-best across ALL epochs.')
        for job in policy['jobs']:
            print(job['id'] + ' old_best/stop=' + str(job['old_best_epoch']) + '/' + str(job['old_stop_epoch']))
        print('19 fresh trainings; original23 remain accepted under the same stop gate. Non-X4 five-permutation/updated Control Ensemble inference remains a later step.')
        return 0
    root = args.root.resolve()
    require(os.name == 'posix' and str(root) == '/root/autodl-tmp/time_uncertainty_rec', 'Execution requires original Linux project root /root/autodl-tmp/time_uncertainty_rec')
    print(json.dumps(preflight(root, policy)), flush=True)
    if args.preflight:
        return 0
    # A separate subprocess performs the dependency/device check before creating outputs.
    checked = subprocess.run([sys.executable, '-c', 'import sys,json,torch; assert torch.cuda.is_available(); print(json.dumps(dict(python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0))))'], capture_output=True, text=True)
    require(checked.returncode == 0, 'CUDA/environment check failed')
    environment = json.loads(checked.stdout)
    print(json.dumps(environment), flush=True)
    import fcntl
    global_guard = (root / '.kais_e1_min50.lock').open('a')
    fcntl.flock(global_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.stage:
        stage = args.stage.resolve()
        require(stage.parent == root and stage.name.startswith(PREFIX) and not stage.is_symlink(), 'Invalid E1 stage')
        state = read_json(stage / 'driver.json')
        require(state['root'] == str(root) and state['stage'] == str(stage) and state['policy_sha256'] == sha(HERE / 'e1_policy.json'), 'Resume root/stage/policy changed')
    else:
        require(not list(root.glob(PREFIX + '*')), 'Existing E1 stage found. Resume it with --stage; do not start a duplicate19-run job.')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:10]
        stage = root / (PREFIX + stamp)
        stage.mkdir()
        (stage / 'attempts').mkdir()
        (stage / 'reports').mkdir()
        state = {'root': str(root), 'stage': str(stage), 'policy_sha256': sha(HERE / 'e1_policy.json'),
                 'bundle_root': str(HERE), 'status': 'PREPARED', 'jobs': {j['id']: {'attempts': []} for j in policy['jobs']}}
        write_json(stage / 'policy.json', policy)
        write_json(stage / 'driver.json', state)
    print('E1_STAGE=' + str(stage), flush=True)
    with (stage / 'driver.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(signum, frame):
            raise KeyboardInterrupt('Received signal ' + str(signum))
        signal.signal(signal.SIGTERM, stop)
        result = 1
        try:
            state['status'] = 'RUNNING'
            state.setdefault('execution_environments', []).append(environment)
            write_json(stage / 'driver.json', state)
            for job in policy['jobs']:
                record = state['jobs'][job['id']]
                if record.get('status') == 'COMPLETED':
                    previous = checked_native(root, stage, record['native_stage'])
                    expected_proof = previous / 'verification/E1_COMPLETED.json'
                    require(Path(record['completion_proof']) == expected_proof, 'Unexpected completion proof path')
                    proof = read_json(expected_proof)
                    fresh = validate_output(previous, job)
                    require(fresh == proof, 'Completed job metadata changed; refusing resume')
                    print('E1_KEEP_COMPLETED=' + job['id'], flush=True)
                    continue
                number = len(record['attempts']) + 1
                native = make_attempt(root, stage, job, number, policy)
                attempt = {'number': number, 'native_stage': str(native), 'status': 'RUNNING'}
                record['attempts'].append(attempt)
                record.update(status='RUNNING', native_stage=str(native))
                write_json(stage / 'driver.json', state)
                script = native / 'scripts' / job['script_name']
                if job['family'] == 'tisasrec':
                    # The original time-aware trainer requires both contract smokes.
                    for tag in ('ml1m', 'beauty'):
                        command = [sys.executable, '-u', str(script), '--root', str(root), '--run', '--stage', str(native), '--mode', 'smoke', '--dataset', tag, '--seed', '42']
                        run_child(command, native / 'verification' / ('smoke_' + tag + '.log'))
                    command = [sys.executable, '-u', str(script), '--root', str(root), '--run', '--stage', str(native), '--mode', 'train', '--dataset', job['dataset'], '--seed', str(job['seed'])]
                else:
                    seed_args = [] if job['family'] == 'ml_lbsp' else ['--seed', str(job['seed'])]
                    run_child([sys.executable, '-u', str(script), '--mode', 'smoke'] + seed_args, native / 'verification/smoke.log')
                    require(not output_path(native, job).exists(), 'Smoke wrote a formal output')
                    command = [sys.executable, '-u', str(script), '--mode', 'train'] + seed_args
                print('E1_TRAIN_START=' + job['id'], flush=True)
                run_child(command, native / 'verification/train.log')
                proof = validate_output(native, job)
                path = native / 'verification/E1_COMPLETED.json'
                write_json(path, proof)
                record.update(status='COMPLETED', completion_proof=str(path))
                attempt['status'] = 'COMPLETED'
                write_json(stage / 'driver.json', state)
                print('E1_TRAIN_COMPLETED=' + job['id'], flush=True)
            state['status'] = 'ALL19_TRAININGS_COMPLETE_INFERENCE_PENDING'
            result = 0
        except BaseException as exc:
            state['status'] = 'STOPPED_INCOMPLETE'
            state['error'] = type(exc).__name__ + ': ' + str(exc)
            print('E1_ERROR=' + state['error'], flush=True)
        finally:
            write_json(stage / 'driver.json', state)
            print('E1_DRIVER_STATUS=' + state['status'], flush=True)
            collected = subprocess.run([sys.executable, str(HERE / 'collect_e1_min50.py'), '--stage', str(stage)])
            if collected.returncode:
                result = collected.returncode
        return result


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('ERROR: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(1)
