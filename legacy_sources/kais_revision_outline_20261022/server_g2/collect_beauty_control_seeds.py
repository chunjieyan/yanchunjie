#!/usr/bin/env python3
"""Collect PRIVATE Beauty Control saved results; standard library only.

No training, inference, dataset reading or checkpoint access. Consistency of
saved files is not a complete reproduction or a significance test.
"""
import argparse
import ast
import csv
import datetime
import hashlib
import io
import itertools
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

FILES = ('config.json', 'metrics.json', 'training_log.csv', 'test_per_sample_predictions.csv', 'item_id_map.csv')
SCRIPTS = ('scripts/model_definition.py', 'scripts/train_beauty_control.py')
MANIFEST = 'manifests/beauty_seed_manifest.json'
REFERENCE = 'step38_kais_retrain/outputs/beauty_sasrec_control_ce_bs128_lr1em3_max200_seed42'
SOURCE_SHA256 = {
    'step38_kais_retrain/scripts/38_4_train_beauty_sasrec_bs128.py': '29c528543757d18b54b18a8c09d4dd4e573e125cc5128e9a13ae702b4be5d48e',
    'step38_kais_retrain/scripts/38_1_train_ml1m_sasrec_bs128.py': '56fdd85061943778708a2de7ec18bbe79bd3873a4d9ffbc12b9e693fadb4b27c',
}
DATA_SHA256 = {
    'processed/diagnostic_a/Amazon_Beauty/file_order.csv': '2d2bc662b76400c93073b77673c7c2a1396f6b1604ea4b216461e925d7d54f47',
    'processed/diagnostic_a_splits/Amazon_Beauty/file_order/train.csv': 'c81fd1c1dbcbbe88ed4596e0d952397d1ff74ba70a5ba9d8f9e4038e72e05971',
    'processed/recent_bucketset_r1/Amazon_Beauty/val/r1_samples.csv': '0f19f717b9c8a3230c8ce52735960fa48f4fb5e2c08c1787e0a4c52dbfbb5eb9',
    'processed/recent_bucketset_r1/Amazon_Beauty/test/r1_samples.csv': 'f9362385825ed79b07e4d262cf9ac7df2591c54cd7d8e95e54ba116bdaad5009',
}
REFERENCE_SHA256 = {
    'config.json': 'a453f393b09d374f25868d8d2a205953b2d0da064c41aad443568bc693ad8425',
    'metrics.json': '5e10b55ec8c550344e66c092ef6511ed524527f00d02d006627b8ec9206f9b2b',
    'training_log.csv': '2b22f6ad9d8be45e775c053d65ff2f34b4392c14ebaa571d8e0e353341a9d887',
    'test_per_sample_predictions.csv': 'c9f0efd243532633594be0ca5b82f81aa1091facd8ad37ea06ffdb478330a40e',
    'item_id_map.csv': '259b8725bf180ff02c3f9a7351c43fdf1de22aa51829c40437c2040f95737289',
}
STANDARD_FIELDS = ('dataset', 'order', 'method', 'max_len', 'hidden_size', 'num_heads', 'num_blocks', 'dropout',
                   'batch_size', 'learning_rate', 'epochs', 'patience', 'training_positions', 'validation_protocol',
                   'test_protocol', 'evaluation', 'padding_evaluation', 'position_encoding', 'training_objective',
                   'training_cutoff_sampling')
LIMIT, TOTAL = 64 * 1024**2, 256 * 1024**2


def plain(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected absolute path without ..')
    for component in (path,) + tuple(path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Symbolic link/reparse point rejected: ' + str(component))
    return path.resolve(strict=False)


def safe(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative
            or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive
            or any(x in ('', '.', '..') for x in relative.split('/'))):
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
    count, blocks = 0, []
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    with os.fdopen(os.open(str(path), flags), 'rb') as handle:
        if signature(before) != signature(os.fstat(handle.fileno())):
            raise ValueError('Selected file changed before reading')
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            count += len(block)
            if count > LIMIT:
                raise ValueError('Selected file exceeds size limit')
            blocks.append(block)
        after = os.fstat(handle.fileno())
    if signature(before) != signature(after) or signature(before) != signature(safe(root, relative).lstat()):
        raise ValueError('Selected file changed during reading')
    blob = b''.join(blocks)
    return blob, hashlib.sha256(blob).hexdigest()


def obj(blob):
    value = json.loads(blob.decode('utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value


def rows(blob):
    return list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig'), newline='')))


def integer(value):
    if isinstance(value, bool) or not re.fullmatch(r'[+-]?[0-9]+', str(value)):
        raise ValueError('Expected integer')
    return int(value)


def close(a, b):
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def rank_metrics(rank):
    n = len(rank)
    if not n:
        raise ValueError('Empty prediction panel')
    return {'N': n, 'NDCG@10': math.fsum(1 / math.log2(r + 1) if r <= 10 else 0 for r in rank) / n,
            'HR@10': sum(r <= 10 for r in rank) / n, 'HR@20': sum(r <= 20 for r in rank) / n,
            'MeanRank': math.fsum(rank) / n, 'MedianRank': statistics.median(rank), 'Rank1Ratio': sum(r == 1 for r in rank) / n}


def check_run(payload, seed, manifest, manifest_hash):
    result = {'seed': seed, 'passed': False, 'status': 'incomplete_saved_files',
              'missing': [name for name in FILES if name not in payload]}
    if result['missing']:
        return result, None
    try:
        config, metrics = obj(payload['config.json']), obj(payload['metrics.json'])
        prediction, mapping, log = (rows(payload[name]) for name in ('test_per_sample_predictions.csv', 'item_id_map.csv', 'training_log.csv'))
        checks = {'dataset_seed': config.get('dataset') == metrics.get('dataset') == 'Amazon_Beauty'
                  and config.get('seed') == metrics.get('seed') == seed}
        if seed != 42:
            provenance = config.get('seed_extension', {})
            checks['seed_extension_provenance'] = (isinstance(provenance, dict)
                and provenance.get('version') == 'beauty_control_seeds_v1'
                and provenance.get('algorithm_and_sorting_unchanged') is True
                and provenance.get('original_source_sha256') == manifest['sources_sha256']
                and provenance.get('input_sha256') == manifest['data_sha256']
                and provenance.get('frozen_model_sha256') == manifest['generated_sha256'][SCRIPTS[0]]
                and provenance.get('trainer_sha256') == manifest['generated_sha256'][SCRIPTS[1]]
                and provenance.get('preparation_manifest_sha256') == manifest_hash)
        itemmap = {r['raw_item_id']: integer(r['item_idx']) for r in mapping}
        checks['itemmap'] = bool(itemmap) and len(itemmap) == len(mapping) and set(itemmap.values()) == set(range(1, len(mapping) + 1))
        panel = sorted((r['sample_id'], r['user_id'], r['target_item']) for r in prediction)
        checks['panel'] = (bool(panel) and all(all(x for x in row) for row in panel)
            and len({r[0] for r in panel}) == len(panel) and len({r[1] for r in panel}) == len(panel)
            and all(r[2] in itemmap for r in panel))
        rank = [integer(r['target_rank']) for r in prediction]
        checks['rank_domain'] = bool(rank) and all(1 <= r <= len(itemmap) for r in rank)
        top_ok, per_row_ok, indexed = True, True, {}
        for row, rank_value in zip(prediction, rank):
            top = json.loads(row['top10_item_idx'])
            raw = json.loads(row['top10_raw_item_id'])
            if not isinstance(top, list) or not isinstance(raw, list):
                raise ValueError('Top10 must be JSON lists')
            top_ok &= len(top) == len(raw) == min(10, len(itemmap)) and len(set(top)) == len(top)
            top_ok &= all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in top)
            top_ok &= all(itemmap.get(raw_id) == index for raw_id, index in zip(raw, top))
            target = itemmap[row['target_item']]
            top_ok &= (target in top) == (rank_value <= 10)
            if rank_value <= 10:
                top_ok &= rank_value <= len(top) and top[rank_value - 1] == target
            expected = {'ndcg10': 1 / math.log2(rank_value + 1) if rank_value <= 10 else 0,
                        'hit10': int(rank_value <= 10), 'hit20': int(rank_value <= 20)}
            per_row_ok &= all(name in row and close(row[name], value) for name, value in expected.items())
            indexed[row['sample_id']] = {'rank': rank_value, 'top': set(raw)}
        checks['top10_mapping_rank'] = bool(top_ok)
        checks['per_row_metrics'] = bool(per_row_ok)
        derived = rank_metrics(rank)
        test = metrics['test']['overall']
        checks['test_metrics'] = all(name in test and close(value, test[name]) for name, value in derived.items())
        epochs = [integer(r['epoch']) for r in log]
        checks['epochs_contiguous'] = epochs == list(range(1, len(log) + 1)) and bool(log)
        checks['log_finite'] = all(math.isfinite(float(v)) for r in log for v in r.values())
        best_value, best_epoch, bad = -1.0, 0, 0
        for row in log:
            value = float(row['val_ndcg10'])
            if value > best_value:
                best_value, best_epoch, bad = value, integer(row['epoch']), 0
            else:
                bad += 1
        checks['best_epoch'] = metrics.get('best_epoch') == best_epoch
        validation = metrics['validation']['overall']
        checks['best_validation_ndcg'] = close(validation.get('NDCG@10'), best_value)
        checks['normal_stop'] = bool(epochs) and (epochs[-1] == config['epochs'] or bad == config['patience'])
        checks['fixed_config_fields_present'] = all(name in config for name in STANDARD_FIELDS)
        result.update(status='checked_saved_files', checks=checks, passed=all(checks.values()),
                      recomputed_test=derived, epochs=len(log), best_epoch=best_epoch, last_bad_epochs=bad)
        return result, {'config': config, 'panel': panel, 'map': itemmap, 'indexed': indexed, 'derived': derived}
    except (KeyError, ValueError, TypeError, OverflowError, ZeroDivisionError, json.JSONDecodeError) as exc:
        result.update(status='invalid_or_unknown_saved_schema', error=type(exc).__name__ + ': ' + str(exc))
        return result, None


def across_seeds(run_checks, details):
    result = {'passed': False, 'descriptive_only': True,
              'population': 'entire fixed test panel; no affected-user subgroup inferred'}
    if set(details) != {42, 2026, 3407} or not all(r['passed'] for r in run_checks.values()):
        result['status'] = 'incomplete_or_failed_runs_no_aggregate'
        return result
    baseline = details[42]
    checks = {'same_panel': all(d['panel'] == baseline['panel'] for d in details.values()),
              'same_itemmap': all(d['map'] == baseline['map'] for d in details.values()),
              'same_fixed_config': all(all(d['config'].get(k) == baseline['config'].get(k) for k in STANDARD_FIELDS) for d in details.values())}
    result['checks'] = checks
    if not all(checks.values()):
        result['status'] = 'alignment_failed_no_aggregate'
        return result
    result['passed'], result['status'] = True, 'three_saved_runs_aligned'
    result['mean_sample_sd'] = {name: {'mean': statistics.mean(d['derived'][name] for d in details.values()),
                                      'sample_sd': statistics.stdev(d['derived'][name] for d in details.values()), 'seeds': 3}
                                for name in ('NDCG@10', 'HR@10', 'HR@20', 'MeanRank')}
    pairs = []
    for a, b in itertools.combinations((42, 2026, 3407), 2):
        left, right = details[a]['indexed'], details[b]['indexed']
        overlap = [len(left[s]['top'] & right[s]['top']) / len(left[s]['top']) for s in left]
        flips = [int((left[s]['rank'] <= 10) != (right[s]['rank'] <= 10)) for s in left]
        pairs.append({'seeds': [a, b], 'N': len(left), 'mean_top10_overlap_fraction': statistics.mean(overlap),
                      'target_hit10_flip_fraction': statistics.mean(flips)})
    result['three_seed_pairs'] = pairs
    result['mean_pair_reference'] = {name: statistics.mean(p[name] for p in pairs)
                                     for name in ('mean_top10_overlap_fraction', 'target_hit10_flip_fraction')}
    return result


def collect(stage):
    stage = plain(stage)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
    verification = plain(stage / 'verification')
    out = verification / ('beauty_review_' + stamp)
    out.mkdir(exist_ok=False)
    content, inventory, payloads, warnings = {}, [], {42: {}, 2026: {}, 3407: {}}, []
    source_ok, manifest, manifest_hash, total = True, {}, '', 0

    def add(root, relative, archive_relative, expected=None, required_hash=False):
        nonlocal total, source_ok
        record = {'origin': str(root), 'path': relative, 'archive_path': archive_relative, 'status': 'unknown'}
        try:
            blob, digest = read_stable(root, relative)
            if expected is not None and digest != expected:
                source_ok = False
                raise ValueError('Pinned SHA256 mismatch')
            if total + len(blob) > TOTAL:
                raise ValueError('Total collection size limit')
            content[archive_relative] = blob
            total += len(blob)
            record.update(status='copied', bytes=len(blob), sha256=digest)
            return blob, digest
        except FileNotFoundError:
            record['status'] = 'missing'
            if required_hash:
                source_ok = False
        except (OSError, ValueError) as exc:
            record.update(status='skipped', error=type(exc).__name__ + ': ' + str(exc))
            if required_hash:
                source_ok = False
        finally:
            inventory.append(record)
        return None, None

    try:
        manifest_blob, manifest_hash = add(stage, MANIFEST, MANIFEST, required_hash=True)
        manifest = obj(manifest_blob)
        root = plain(Path(manifest['root']))
        if str(stage) != manifest['stage'] or stage.parent != root:
            raise ValueError('Stage is not the original stage under the project root')
        if manifest.get('sources_sha256') != SOURCE_SHA256 or manifest.get('data_sha256') != DATA_SHA256:
            raise ValueError('Manifest source/input hashes differ from reviewed originals')
        ref = manifest.get('reference_run', {})
        if ref.get('path') != REFERENCE or ref.get('seed') != 42 or ref.get('sha256') != REFERENCE_SHA256:
            raise ValueError('Reference seed42 identity/hashes differ from reviewed originals')
        if set(manifest.get('generated_sha256', {})) != set(SCRIPTS):
            raise ValueError('Unexpected generated script set')
        for relative, expected in SOURCE_SHA256.items():
            blob, _ = add(root, relative, 'original_sources/' + relative, expected, True)
            if blob is not None:
                ast.parse(blob.decode('utf-8-sig'), filename=relative)
        for relative in SCRIPTS:
            expected = manifest['generated_sha256'][relative]
            if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected):
                raise ValueError('Invalid generated script hash')
            blob, _ = add(stage, relative, relative, expected, True)
            if blob is not None:
                ast.parse(blob.decode('utf-8-sig'), filename=relative)
        for name in FILES:
            blob, _ = add(root, REFERENCE + '/' + name, 'reference_seed42/' + name, REFERENCE_SHA256[name])
            if blob is not None:
                payloads[42][name] = blob
    except (OSError, ValueError, TypeError, KeyError, SyntaxError, AttributeError) as exc:
        source_ok = False
        warnings.append('Source/manifest validation failed: ' + type(exc).__name__ + ': ' + str(exc))
    # These are only saved result files and immediate verification logs/JSONs.
    for seed in (2026, 3407):
        prefix = 'outputs/amazon_beauty_control_seed' + str(seed) + '/'
        for name in FILES:
            blob, _ = add(stage, prefix + name, prefix + name)
            if blob is not None:
                payloads[seed][name] = blob
    for path in sorted(verification.iterdir()):
        if path.suffix.lower() in ('.log', '.json') and not path.name.startswith('.'):
            add(stage, 'verification/' + path.name, 'verification/' + path.name)
    run_checks, details = {}, {}
    if source_ok:
        for seed in (42, 2026, 3407):
            checked, detail = check_run(payloads[seed], seed, manifest, manifest_hash)
            run_checks[str(seed)] = checked
            if detail is not None:
                details[seed] = detail
    aggregate = across_seeds(run_checks, details) if source_ok else {'passed': False, 'status': 'source_integrity_failed_no_aggregate'}
    report = {'status': 'private_saved_file_review' if source_ok else 'source_integrity_failed_no_bundle',
              'stage': str(stage), 'manifest_sha256': manifest_hash, 'source_integrity_passed': source_ok,
              'runs': run_checks, 'aggregate': aggregate, 'warnings': warnings,
              'limits': ['No research code executed or imported; no checkpoint or dataset contents accessed.',
                         'Input hashes in the manifest/provenance were compared, not recomputed from dataset contents.',
                         'Training processes/checkpoint contents/inference reproduction are not certified.',
                         'Per-user prediction CSVs are private review material; do not publish this ZIP directly.',
                         'No affected-user subgroup or significance/equivalence claim is inferred.']}
    (out / 'result_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (out / 'COLLECTED_SHA256.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['origin', 'path', 'archive_path', 'status', 'bytes', 'sha256', 'error'])
        writer.writeheader()
        writer.writerows(inventory)
    (out / 'SUMMARY.md').write_text('# PRIVATE Beauty Control three-seed review\n\nSaved-file consistency only. No dataset/checkpoint contents, training or inference accessed.\n\nSee result_checks.json for missing/failed files and descriptive-only three-seed comparisons.\n', encoding='utf-8')
    if source_ok:
        bundle = out / 'beauty_control_three_seed_results.zip'
        with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ('COLLECTED_SHA256.csv', 'SUMMARY.md', 'result_checks.json'):
                archive.write(out / name, name)
            for relative, blob in sorted(content.items()):
                archive.writestr('snapshot/' + relative, blob)
        digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
        (out / (bundle.name + '.sha256.txt')).write_text(digest + '  ' + bundle.name + '\n', encoding='ascii')
        print('BEAUTY_REPORT_ZIP=' + str(bundle))
    print('BEAUTY_REVIEW_DIR=' + str(out))
    print('SAVED_RESULTS_ALIGNED=' + str(bool(aggregate.get('passed'))))
    return 0 if source_ok and aggregate.get('passed') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    args = parser.parse_args()
    return collect(args.stage)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print('ERROR: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
