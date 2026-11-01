#!/usr/bin/env python3
"""Collect PRIVATE saved-file evidence for Sports/Toys orderfix seeds; stdlib only.
No research execution, inference, data access, or checkpoint content access.
"""
import argparse
import ast
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import statistics
import sys
import uuid
import zipfile

SCRIPTS = ('scripts/model_definition.py', 'scripts/train_sports_orderfix.py', 'scripts/train_toys_orderfix.py')
FILES = ('config.json', 'metrics.json', 'training_log.csv', 'test_per_sample_predictions.csv', 'item_id_map.csv')
SEEDS = (42, 2026, 3407)
LIMIT, TOTAL = 64 * 1024 ** 2, 256 * 1024 ** 2
METRICS = {'N': ('N', 'samples'), 'NDCG@10': ('NDCG@10', 'ndcg10'), 'HR@10': ('HR@10', 'hr10'),
           'HR@20': ('HR@20', 'hr20'), 'MeanRank': ('MeanRank', 'mean_rank')}


def safe(root, relative):
    if (not relative or '\\' in relative or PurePosixPath(relative).is_absolute()
            or PureWindowsPath(relative).drive or any(x in ('', '.', '..') for x in relative.split('/'))):
        raise ValueError('Unsafe relative path')
    path = root
    for part in relative.split('/'):
        path = path / part
        if path.is_symlink() or (path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Symbolic link/reparse point rejected')
    path.resolve(strict=False).relative_to(root)
    return path


def signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def read_stable(root, relative):
    path = safe(root, relative)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > LIMIT:
        raise ValueError('Nonregular or oversized selected file')
    chunks, count = [], 0
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    with os.fdopen(os.open(str(path), flags), 'rb') as handle:
        if signature(before) != signature(os.fstat(handle.fileno())):
            raise ValueError('File changed before reading')
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            count += len(block)
            if count > LIMIT:
                raise ValueError('Selected file exceeds size limit')
            chunks.append(block)
        after = os.fstat(handle.fileno())
    if signature(before) != signature(after) or signature(before) != signature(safe(root, relative).lstat()) or count != before.st_size:
        raise ValueError('File changed during reading')
    blob = b''.join(chunks)
    return blob, hashlib.sha256(blob).hexdigest()


def decode_json(blob):
    value = json.loads(blob.decode('utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value


def rows(blob):
    return list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig'), newline='')))


def close(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def validate_stage(stage):
    stage = Path(stage)
    if not stage.is_absolute() or '..' in stage.parts:
        raise ValueError('Stage must be absolute without ..')
    for path in (stage,) + tuple(stage.parents):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Stage symbolic link/reparse point rejected')
    stage = stage.resolve(strict=False)
    manifest_blob, manifest_hash = read_stable(stage, 'manifests/orderfix_manifest.json')
    manifest, content = decode_json(manifest_blob), {}
    for relative in SCRIPTS:
        blob, actual = read_stable(stage, relative)
        expected = manifest.get('generated_sha256', {}).get(relative, '')
        if not isinstance(expected, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', expected) or actual != expected.lower():
            raise ValueError('Generated script hash mismatch: ' + relative)
        ast.parse(blob.decode('utf-8-sig'), filename=relative)
        content[relative] = (blob, actual)
    content['manifests/orderfix_manifest.json'] = (manifest_blob, manifest_hash)
    return stage, manifest, manifest_hash, content


def check_run(payload, dataset, seed, manifest, manifest_hash):
    result = {'dataset': dataset, 'seed': seed, 'passed': False, 'checkpoint_content_checked': False}
    result['missing_required_files'] = [name for name in FILES if name not in payload]
    if result['missing_required_files']:
        result['status'] = 'incomplete_required_files'
        return result, None
    try:
        config, metrics = decode_json(payload['config.json']), decode_json(payload['metrics.json'])
        prediction, mapping, log = (rows(payload[name]) for name in (
            'test_per_sample_predictions.csv', 'item_id_map.csv', 'training_log.csv'))
        checks = {'dataset_identity': config.get('dataset') == metrics.get('dataset') == dataset,
                  'seed_identity': config.get('seed') == metrics.get('seed') == seed}
        provenance = config.get('orderfix', {})
        trainer = SCRIPTS[1 if dataset == 'Amazon_Sports' else 2]
        checks['orderfix_provenance'] = (isinstance(provenance, dict)
            and provenance.get('version') == 'kais_orderfix_v1'
            and provenance.get('sequence_sort_keys') == ['user_id', 'time_bucket', 'raw_line']
            and provenance.get('original_input_files_unchanged') is True
            and bool(manifest.get('data_sha256')) and provenance.get('input_sha256') == manifest['data_sha256']
            and provenance.get('frozen_model_sha256') == manifest['generated_sha256'][SCRIPTS[0]]
            and provenance.get('trainer_sha256') == manifest['generated_sha256'][trainer]
            and provenance.get('preparation_manifest_sha256') == manifest_hash)
        panel = sorted((r['sample_id'], r['user_id'], r['target_item']) for r in prediction)
        itemmap = {r['raw_item_id']: int(r['item_idx']) for r in mapping}
        checks['panel_identity_valid'] = (bool(panel) and all(all(x for x in row) for row in panel)
            and len({r[0] for r in panel}) == len(panel) and len({r[1] for r in panel}) == len(panel)
            and all(r[2] in itemmap for r in panel))
        checks['itemmap_valid'] = (len(itemmap) == len(mapping) == config.get('num_items')
            and set(itemmap.values()) == set(range(1, len(mapping) + 1)))
        rank = [float(r['target_rank']) for r in prediction]
        if not rank or any(not math.isfinite(r) or r < 1 or r > len(itemmap) or not r.is_integer() for r in rank):
            raise ValueError('Invalid target ranks')
        top_valid = True
        for row, target_rank in zip(prediction, rank):
            top = json.loads(row['top10_item_idx'])
            raw = json.loads(row['top10_raw_item_id'])
            target = itemmap.get(row['target_item'])
            top_valid = top_valid and len(top) == len(raw) == min(10, len(itemmap)) and len(set(top)) == len(top)
            top_valid = top_valid and all(itemmap.get(raw_id) == index for raw_id, index in zip(raw, top))
            top_valid = top_valid and ((target in top) == (target_rank <= 10))
            top_valid = top_valid and (target_rank > 10 or top.index(target) + 1 == target_rank)
        checks['top10_matches_itemmap_and_rank'] = bool(top_valid)
        n = len(rank)
        derived = {'N': n, 'NDCG@10': math.fsum(1 / math.log2(r + 1) if r <= 10 else 0 for r in rank) / n,
                   'HR@10': sum(r <= 10 for r in rank) / n, 'HR@20': sum(r <= 20 for r in rank) / n,
                   'MeanRank': math.fsum(rank) / n}
        test = metrics.get('test', {})
        test = test.get('overall', test)
        checks['all_required_test_metrics_match_ranks'] = all(
            any(name in test for name in names) and all(close(derived[key], test[name]) for name in names if name in test)
            for key, names in METRICS.items())
        epochs = {int(r['epoch']): float(r['val_ndcg10']) for r in log}
        best = metrics.get('best_epoch')
        checks['epoch_history_valid'] = (bool(log) and len(epochs) == len(log)
            and sorted(epochs) == list(range(1, len(log) + 1)) and all(math.isfinite(x) for x in epochs.values()))
        checks['best_epoch_matches_validation'] = (best in epochs and close(epochs[best], max(epochs.values()))
            and close(epochs[best], metrics.get('validation', {}).get('ndcg10')))
        result.update(checks=checks, recomputed_from_rank=derived, N=n, passed=all(checks.values()),
                      status='saved_file_checks_passed' if all(checks.values()) else 'saved_file_checks_failed')
        comparable = {key: value for key, value in config.items() if key != 'seed'}
        return result, {'panel': panel, 'itemmap': itemmap, 'config': comparable}
    except (KeyError, ValueError, TypeError, AttributeError, csv.Error, UnicodeError, OverflowError) as exc:
        result.update(status='invalid_or_unknown_saved_schema', error=type(exc).__name__ + ': ' + str(exc)[:160])
        return result, None


def collect(stage, seeds):
    stage, manifest, manifest_hash, content = validate_stage(stage)
    verification = safe(stage, 'verification')
    if not verification.is_dir():
        raise ValueError('Missing verification directory')
    out = verification / ('multiseed_review_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:10])
    out.mkdir(exist_ok=False)
    selected, payloads, records, warnings, total = list(content), {}, [], [], 0
    selected += ['verification/' + p.name for p in sorted(verification.iterdir())
                 if p.suffix.lower() in ('.log', '.json') and not p.name.startswith('.')]
    for label in ('sports', 'toys'):
        for seed in seeds:
            key = label + '_seed' + str(seed)
            base = 'outputs/amazon_' + label + '_control_orderfix_seed' + str(seed) + '/'
            payloads[key] = {'base': base, 'data': {}, 'dataset': 'Amazon_' + label.title(), 'seed': seed}
            selected += [base + name for name in FILES]
    for relative in dict.fromkeys(selected):
        record = {'path': relative, 'status': 'unknown'}
        try:
            blob, digest = content.get(relative) or read_stable(stage, relative)
            if total + len(blob) > TOTAL:
                raise ValueError('Total copy limit exceeded')
            destination = safe(out, 'snapshot/' + relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(blob)
            total += len(blob)
            record.update(status='copied', bytes=len(blob), sha256=digest)
            for run in payloads.values():
                if relative.startswith(run['base']):
                    run['data'][PurePosixPath(relative).name] = blob
        except (OSError, ValueError) as exc:
            record['status'] = 'not_observed' if isinstance(exc, FileNotFoundError) else 'rejected_' + type(exc).__name__
            warnings.append(relative + ': ' + record['status'])
        records.append(record)
    checks, identities, aggregate = {}, {}, {}
    for key, run in payloads.items():
        checks[key], identities[key] = check_run(run['data'], run['dataset'], run['seed'], manifest, manifest_hash)
        try:
            checks[key]['checkpoint_exists_only'] = safe(stage, run['base'] + 'matched_control_sasrec_best.pt').is_file()
        except (OSError, ValueError):
            checks[key]['checkpoint_exists_only'] = 'unknown_or_rejected_path'
    for label in ('sports', 'toys'):
        keys = [label + '_seed' + str(seed) for seed in seeds]
        ready = all(checks[key]['passed'] and identities[key] for key in keys)
        alignment = {field: ready and all(identities[key][field] == identities[keys[0]][field] for key in keys)
                     for field in ('panel', 'itemmap', 'config')}
        aggregate[label] = {'seeds': list(seeds), 'all_requested_runs_passed': bool(ready), 'alignment': alignment,
                            'summary_available': False, 'std_ddof': 1, 'significance_tested': False}
    globally_eligible = (len(seeds) >= 2 and all(check['passed'] for check in checks.values())
                         and all(all(summary['alignment'].values()) for summary in aggregate.values()))
    if globally_eligible:
        for label in ('sports', 'toys'):
            keys = [label + '_seed' + str(seed) for seed in seeds]
            aggregate[label]['summary_available'] = True
            aggregate[label]['test_descriptive_statistics'] = {
                metric: {'mean': statistics.mean(checks[key]['recomputed_from_rank'][metric] for key in keys),
                         'std': statistics.stdev(checks[key]['recomputed_from_rank'][metric] for key in keys)}
                for metric in ('NDCG@10', 'HR@10', 'HR@20', 'MeanRank')}
    report = {'stage': str(stage), 'requested_seeds': list(seeds), 'run_checks': checks, 'descriptive_summary': aggregate,
              'source_sha_and_ast_verified': True, 'files': records, 'warnings': warnings,
              'scope': 'PRIVATE saved-file consistency only; no research execution, inference, checkpoint contents or data accessed. No significance claims or full training reproduction proof.'}
    (out / 'result_checks.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (out / 'COLLECTED_SHA256.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'status', 'bytes', 'sha256'])
        writer.writeheader()
        writer.writerows(records)
    (out / 'SUMMARY.md').write_text('# PRIVATE orderfix multiseed review\n\n' + report['scope']
        + '\n\nRequested seeds: ' + ', '.join(map(str, seeds))
        + '\n\nSee result_checks.json for per-run failures, missing files, alignment and descriptive statistics.\n', encoding='utf-8')
    bundle = out / 'orderfix_multiseed_results.zip'
    with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path != bundle:
                if path.is_symlink():
                    raise ValueError('Output symbolic link rejected')
                archive.write(path, path.relative_to(out).as_posix())
    digest = hashlib.sha256()
    with bundle.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    (out / (bundle.name + '.sha256.txt')).write_text(digest.hexdigest() + '  ' + bundle.name + '\n', encoding='ascii')
    print('SUMMARY=' + str(out / 'SUMMARY.md'))
    print('BUNDLE_ZIP=' + str(bundle))
    return out, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args()
    try:
        collect(args.stage.expanduser(), sorted(set(args.seeds)))
        return 0
    except (OSError, ValueError, UnicodeError, SyntaxError) as exc:
        print('COLLECTION_FAILED=' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
