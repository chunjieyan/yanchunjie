#!/usr/bin/env python3
"""Collect a PRIVATE Sports/Toys seed-42 review bundle. Standard library only.

No training, inference, research-code execution, network, raw-data copying,
or checkpoint reading/hashing/loading. Existing stage files are never modified.
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
import sys
import uuid
import zipfile

SCRIPTS = ('scripts/model_definition.py', 'scripts/train_sports_orderfix.py', 'scripts/train_toys_orderfix.py')
FILES = ('config.json', 'metrics.json', 'training_log.csv', 'test_per_sample_predictions.csv', 'item_id_map.csv')
LIMIT, TOTAL = 64 * 1024 ** 2, 256 * 1024 ** 2


def safe(root, relative):
    if (not relative or '\\' in relative or PurePosixPath(relative).is_absolute()
            or PureWindowsPath(relative).drive or any(x in ('', '.', '..') for x in relative.split('/'))):
        raise ValueError('Unsafe relative path')
    path = root
    for part in relative.split('/'):
        path = path / part
        if path.is_symlink():
            raise ValueError('Symbolic link rejected')
        if path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError('Reparse point rejected')
    path.resolve(strict=False).relative_to(root)
    return path


def signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def read_stable(root, relative):
    path = safe(root, relative)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('Not a regular file')
    if before.st_size > LIMIT:
        raise ValueError('File exceeds 64 MiB limit')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    chunks, digest, count = [], hashlib.sha256(), 0
    with os.fdopen(os.open(str(path), flags), 'rb') as handle:
        if signature(before) != signature(os.fstat(handle.fileno())):
            raise ValueError('File changed before reading')
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            if count > LIMIT:
                raise ValueError('File exceeds 64 MiB limit')
            chunks.append(block)
            digest.update(block)
        after_fd = os.fstat(handle.fileno())
    if (signature(before) != signature(after_fd)
            or signature(before) != signature(safe(root, relative).lstat()) or count != before.st_size):
        raise ValueError('File changed during reading')
    return b''.join(chunks), digest.hexdigest()


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
    """Read-only integrity API for a separately reviewed driver; returns manifest."""
    stage = Path(stage)
    if not stage.is_absolute() or '..' in stage.parts:
        raise ValueError('Stage must be absolute without ..')
    for path in (stage,) + tuple(stage.parents):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Stage symbolic link/reparse point rejected')
    manifest = decode_json(read_stable(stage, 'manifests/orderfix_manifest.json')[0])
    for relative in SCRIPTS:
        blob, actual = read_stable(stage, relative)
        expected = manifest.get('generated_sha256', {}).get(relative, '')
        if not isinstance(expected, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', expected) or actual != expected.lower():
            raise ValueError('Generated script hash mismatch: ' + relative)
        ast.parse(blob.decode('utf-8-sig'), filename=relative)
    return manifest


def check_run(payload, dataset, manifest=None):
    result = {'dataset': dataset, 'checkpoint_content_checked': False, 'scope': 'saved_file_consistency_only'}
    try:
        config = decode_json(payload['config.json'])
        metrics = decode_json(payload['metrics.json'])
        prediction = rows(payload['test_per_sample_predictions.csv'])
    except KeyError:
        result['status'] = 'incomplete_required_files'
        return result
    except (ValueError, UnicodeError, csv.Error):
        result['status'] = 'unreadable_saved_metadata_or_predictions'
        return result
    result['seed_identity'] = close(config.get('seed'), 42) and close(metrics.get('seed'), 42)
    result['dataset_identity'] = config.get('dataset') == metrics.get('dataset') == dataset
    orderfix = config.get('orderfix', {})
    if isinstance(orderfix, dict) and orderfix and manifest:
        result['sequence_sort_keys_match'] = orderfix.get('sequence_sort_keys') == ['user_id', 'time_bucket', 'raw_line']
        result['input_hashes_match_manifest'] = orderfix.get('input_sha256') == manifest.get('data_sha256')
        result['frozen_model_hash_matches_manifest'] = orderfix.get('frozen_model_sha256') == manifest.get('generated_sha256', {}).get(SCRIPTS[0])
    else:
        result['orderfix_provenance'] = 'unknown_missing_saved_metadata'
    result['N'] = len(prediction)
    result['sample_ids_unique'] = (len({r.get('sample_id') for r in prediction}) == len(prediction)
                                   if prediction and 'sample_id' in prediction[0] else 'unknown')
    result['user_ids_unique'] = (len({r.get('user_id') for r in prediction}) == len(prediction)
                                 if prediction and 'user_id' in prediction[0] else 'unknown')
    try:
        rank = [float(r['target_rank']) for r in prediction]
        if not rank or any(not math.isfinite(r) or r < 1 or not r.is_integer() for r in rank):
            raise ValueError('Invalid target ranks')
        n = len(rank)
        derived = {'N': n, 'NDCG@10': math.fsum(1 / math.log2(r + 1) if r <= 10 else 0 for r in rank) / n,
                   'HR@10': sum(r <= 10 for r in rank) / n, 'HR@20': sum(r <= 20 for r in rank) / n,
                   'MeanRank': math.fsum(rank) / n, 'Recall@50': sum(r <= 50 for r in rank) / n,
                   'MRR': math.fsum(1 / r for r in rank) / n}
        test = metrics.get('test', {})
        test = test.get('overall', test) if isinstance(test, dict) else {}
        aliases = {'N': ('N', 'samples'), 'NDCG@10': ('NDCG@10', 'ndcg10'), 'HR@10': ('HR@10', 'hr10'),
                   'HR@20': ('HR@20', 'hr20'), 'MeanRank': ('MeanRank', 'mean_rank'),
                   'Recall@50': ('Recall@50', 'recall50'), 'MRR': ('MRR', 'mrr')}
        result['recomputed_from_rank'] = derived
        result['recorded_metric_checks'] = {key: close(derived[key], test[name]) for key, names in aliases.items()
                                           for name in names if name in test}
        result['missing_recorded_metrics'] = [key for key, names in aliases.items() if not any(k in test for k in names)]
        if 'training_log.csv' in payload:
            log = rows(payload['training_log.csv'])
            epoch = {float(r['epoch']): float(r['val_ndcg10']) for r in log}
            best = float(metrics.get('best_epoch', 'nan'))
            result['best_epoch_in_log'] = best in epoch
            result['best_epoch_matches_max_validation'] = close(epoch[best], max(epoch.values())) if best in epoch else 'unknown'
        result['status'] = 'checked_saved_files_not_full_reproduction'
    except (KeyError, ValueError, TypeError, csv.Error, UnicodeError, OverflowError):
        result['status'] = 'unknown_rank_or_log_schema_needs_review'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    args = parser.parse_args()
    stage = args.stage.expanduser()
    if not stage.is_absolute() or '..' in stage.parts:
        parser.error('--stage must be an absolute existing directory without ..')
    for path in reversed((stage,) + tuple(stage.parents)):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            parser.error('Stage path contains a symbolic link/reparse point')
    stage = stage.resolve(strict=False)
    verification = safe(stage, 'verification')
    if not verification.is_dir():
        parser.error('Stage verification directory is missing')
    out = verification / ('fulltrain_review_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_UTC_') + uuid.uuid4().hex[:10])
    out.mkdir(exist_ok=False)
    records, warnings, integrity, payloads = [], [], [], {}
    content, total = {}, 0
    try:
        manifest_blob, manifest_hash = read_stable(stage, 'manifests/orderfix_manifest.json')
        manifest = decode_json(manifest_blob)
        generated = manifest.get('generated_sha256', {})
        for relative in SCRIPTS:
            blob, actual = read_stable(stage, relative)
            expected = generated.get(relative, '')
            if not isinstance(expected, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', expected) or actual != expected.lower():
                raise ValueError('Generated script hash mismatch: ' + relative)
            ast.parse(blob.decode('utf-8-sig'), filename=relative)
            content[relative] = (blob, actual)
        content['manifests/orderfix_manifest.json'] = (manifest_blob, manifest_hash)
        integrity.append('All three generated script hashes match stage manifest; AST syntax parsed, never executed.')
    except (OSError, ValueError, UnicodeError, SyntaxError) as exc:
        warnings.append('Source/manifest verification failed (' + type(exc).__name__ + ': ' + str(exc)[:240] + '); no bundle created.')
    source_ok = bool(integrity)
    if source_ok:
        selected = list(content)
        for path in sorted(verification.iterdir()):
            if path.suffix.lower() in ('.log', '.json') and not path.name.startswith('.'):
                selected.append('verification/' + path.name)
        for label in ('sports', 'toys'):
            base = 'outputs/amazon_' + label + '_control_orderfix_seed42/'
            payloads[label] = {}
            for name in FILES:
                selected.append(base + name)
        for relative in dict.fromkeys(selected):
            record = {'path': relative, 'status': 'unknown'}
            try:
                blob, digest = content.get(relative) or read_stable(stage, relative)
                if total + len(blob) > TOTAL:
                    raise ValueError('Total copy limit 256 MiB exceeded')
                destination = safe(out, 'snapshot/' + relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('xb') as handle:
                    handle.write(blob)
                total += len(blob)
                record.update(status='copied', bytes=len(blob), sha256=digest)
                for label in payloads:
                    if relative.startswith('outputs/amazon_' + label + '_control_orderfix_seed42/'):
                        payloads[label][PurePosixPath(relative).name] = blob
            except FileNotFoundError:
                record['status'] = 'not_observed'
                warnings.append('Missing selected file: ' + relative)
            except (OSError, ValueError) as exc:
                record['status'] = 'skipped_' + type(exc).__name__
                warnings.append('Rejected, changing or oversized selected file: ' + relative)
            records.append(record)
    checks = {}
    for label, payload in payloads.items():
        checks[label] = check_run(payload, 'Amazon_' + label.title(), manifest)
        try:
            checkpoint = safe(stage, 'outputs/amazon_' + label + '_control_orderfix_seed42/matched_control_sasrec_best.pt')
            checks[label]['checkpoint_exists_only'] = checkpoint.is_file()
        except (OSError, ValueError):
            checks[label]['checkpoint_exists_only'] = 'unknown_or_rejected_path'
    report = {'stage': str(stage), 'status': 'collected_private_review' if source_ok else 'integrity_failed_no_bundle',
              'source_verification': integrity, 'files': records, 'run_checks': checks, 'warnings': warnings,
              'scope': 'Saved-file consistency only. No research code/model execution, no checkpoint reading/hashing, no original data copied. Not a publishable package or proof of training reproduction.'}
    (out / 'result_checks.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (out / 'COLLECTED_SHA256.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'status', 'bytes', 'sha256'])
        writer.writeheader()
        writer.writerows(records)
    summary = '# PRIVATE orderfix seed-42 result review\n\n' + report['scope'] + '\n\n' + '\n'.join('- ' + x for x in integrity + warnings)
    summary += '\n\nMissing/skipped outputs mean the run is not yet complete in this collection. Check result_checks.json for rank/metric consistency and unknown fields.\n'
    (out / 'SUMMARY.md').write_text(summary, encoding='utf-8')
    bundle = out / 'orderfix_seed42_results.zip'
    if source_ok:
        included = sorted(p for p in out.rglob('*') if p.is_file())
        with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in included:
                if path.is_symlink():
                    raise ValueError('Output symbolic link rejected')
                archive.write(path, path.relative_to(out).as_posix())
        digest = hashlib.sha256()
        with bundle.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        (out / 'orderfix_seed42_results.zip.sha256.txt').write_text(digest.hexdigest() + '  ' + bundle.name + '\n', encoding='ascii')
    print('SUMMARY=' + str(out / 'SUMMARY.md'))
    print('BUNDLE_ZIP=' + (str(bundle) if source_ok else 'NOT_CREATED'))
    return 0 if source_ok else 2


if __name__ == '__main__':
    sys.exit(main())
