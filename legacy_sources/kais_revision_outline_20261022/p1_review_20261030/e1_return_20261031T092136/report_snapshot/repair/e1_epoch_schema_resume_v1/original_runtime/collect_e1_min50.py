#!/usr/bin/env python3
"""Collect only new saved E1 outputs; never put checkpoints or source interaction data into ZIP."""
import argparse
import datetime
from pathlib import Path
import sys
import uuid
import zipfile
from e1_common import HERE, PREFIX, lock, read_json, require, safe, sha, validate_output, write_json, checked_native, output_path


def allowed_attempt_files(native, job, policy):
    """Explicit report closure; never recursively export arbitrary debug/data CSVs."""
    family = policy['families'][job['family']]
    relative = {'scripts/' + name for name in family['scripts']}
    relative.update('verification/transformation_' + name + '.json' for name in family['patch_scripts'])
    relative.update(('verification/E1_COMPLETED.json', 'verification/smoke.log', 'verification/train.log'))
    if job['family'] in ('shuffle', 'amazon_lbsp'):
        relative.add(family['manifest_name'])
        relative.add('verification/completed_' + job['dataset'] + '_seed' + str(job['seed']) + '.json')
    if job['family'] == 'shuffle':
        relative.add('verification/smoke_' + job['dataset'] + '_seed' + str(job['seed']) + '.json')
    if job['family'] == 'amazon_lbsp':
        relative.update('references/' + job['dataset'] + '_seed42/' + name for name in ('config.json', 'metrics.json', 'training_log.csv', 'test_per_sample_predictions.csv', 'item_id_map.csv'))
    names = {'config.json', 'metrics.json', 'training_log.csv', 'item_id_map.csv', 'test_per_sample_predictions.csv', 'validation_per_sample_predictions.csv'}
    if job['family'] == 'tisasrec':
        relative.update(('manifest.json', 'x4_tisasrec_inputs_20261026.json'))
        relative.update('verification/smoke_' + tag + suffix for tag in ('ml1m', 'beauty') for suffix in ('.json', '.log'))
        relative.add('verification/completed_' + job['dataset'] + '_seed' + str(job['seed']) + '.json')
        names.update(['predictions_original.csv', 'validation_best_predictions.csv', 'stability.json', 'panel_labels.csv'])
        names.update('predictions_perm_seed' + str(seed) + '.csv' for seed in (101, 202, 303, 404, 505))
    output_relative = output_path(native, job).relative_to(native).as_posix()
    relative.update(output_relative + '/' + name for name in names)
    found = []
    for name in sorted(relative):
        path = safe(native, name)
        if path.is_file():
            found.append((name, path))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    args = parser.parse_args()
    policy = lock()
    stage = args.stage.resolve()
    state = read_json(stage / 'driver.json')
    root = Path(state['root']).resolve()
    require(stage.parent == root and stage.name.startswith(PREFIX) and state['stage'] == str(stage), 'Invalid E1 stage identity')
    require(state['policy_sha256'] == sha(HERE / 'e1_policy.json'), 'Policy changed')
    checks = []
    incomplete = []
    for job in policy['jobs']:
        record = state['jobs'][job['id']]
        if record.get('status') != 'COMPLETED':
            incomplete.append(job['id'])
            continue
        native = checked_native(root, stage, record['native_stage'])
        proof = validate_output(native, job)
        expected_proof = native / 'verification/E1_COMPLETED.json'
        require(Path(record['completion_proof']) == expected_proof, 'Unexpected completion proof path')
        require(proof == read_json(expected_proof), 'Completed proof differs')
        checks.append(proof)
    report = {'status': 'ALL19_TRAININGS_VALIDATED_INFERENCE_PENDING' if not incomplete else 'INCOMPLETE',
              'training_complete_count': len(checks), 'incomplete_jobs': incomplete, 'minimum_epochs': 50,
              'new_results_promoted_to_manuscript': False, 'nonX4_five_permutation_and_updated_ensemble_inference_pending': True,
              'jobs': checks}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
    reports = stage / 'reports'
    reports.mkdir(exist_ok=True)
    summary = reports / ('E1_SUMMARY_' + stamp + '.json')
    write_json(summary, report)
    name = 'e1_min50_training_results_' + stamp + '.zip'
    archive = reports / name
    selected = {'driver.json': stage / 'driver.json', 'policy.json': stage / 'policy.json', 'SUMMARY.json': summary}
    for job in policy['jobs']:
        for attempt in state['jobs'][job['id']]['attempts']:
            native = checked_native(root, stage, attempt['native_stage'])
            for relative, p in allowed_attempt_files(native, job, policy):
                selected['attempts/' + job['id'] + '_a' + str(attempt['number']) + '/' + relative] = p
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(selected.items()):
            z.write(path, name)
    checksum = archive.with_suffix('.zip.sha256.txt')
    checksum.write_text(sha(archive) + '  ' + archive.name + '\n', encoding='utf-8')
    print('E1_REPORT_STATUS=' + report['status'])
    print('E1_RESULTS_ZIP=' + str(archive))
    print('E1_RESULTS_SHA256=' + str(checksum))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('ERROR: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(1)
