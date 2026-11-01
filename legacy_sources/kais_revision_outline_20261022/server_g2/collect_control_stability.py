#!/usr/bin/env python3
"""Collect PRIVATE fixed-model stability evidence; Python 3.8+, standard library.

Never executes research code, performs inference, accesses checkpoints, or reads
the 24 original/permuted interaction inputs. No existing file is modified.
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

INPUTS_FILE = Path(__file__).absolute().with_name('control_stability_inputs_20261023.json')
EXPECTED_INPUTS_SHA256 = '8e8d13ded1ded7729e7f951b97d65ef3bda84614f4f4a9ad881bec68f7c81458'
TAGS = {'ml1m': 'MovieLens-1M', 'beauty': 'Amazon_Beauty', 'sports': 'Amazon_Sports', 'toys': 'Amazon_Toys'}
SEEDS, PERMS = (42, 2026, 3407), (101, 202, 303, 404, 505)
REFERENCE_FILES = ('config.json', 'metrics.json', 'training_log.csv', 'item_id_map.csv', 'test_per_sample_predictions.csv')
PREDICTIONS = ('predictions_original.csv',) + tuple('predictions_perm_seed%d.csv' % s for s in PERMS)
RUN_FILES = PREDICTIONS + ('panel_labels.csv', 'metadata.json')
MANIFEST = 'manifests/control_stability_manifest.json'
FILE_LIMIT, TOTAL_LIMIT = 64 * 1024 ** 2, 512 * 1024 ** 2


def absolute_plain(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected an absolute path without ..')
    for component in (path,) + tuple(path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Symbolic link/reparse point rejected')
    return path.resolve(strict=False)


def relative_path(value):
    if (not isinstance(value, str) or not value or '\\' in value or '\x00' in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(x in ('', '.', '..') or x.startswith('.') or x.lower() == 'notes' for x in value.split('/'))):
        raise ValueError('Unsafe relative path')
    if 'kuairec' in value.lower() or 'step39' in value.lower():
        raise ValueError('Unrelated project path rejected')
    return value


def safe(root, relative):
    path = root
    for part in relative_path(relative).split('/'):
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
    if PurePosixPath(relative).suffix.lower() in ('.pt', '.pth', '.ckpt'):
        raise ValueError('Checkpoint access forbidden')
    path = safe(root, relative)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > FILE_LIMIT:
        raise ValueError('Nonregular or oversized file')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    parts, h, n = [], hashlib.sha256(), 0
    with os.fdopen(os.open(str(path), flags), 'rb') as handle:
        if signature(before) != signature(os.fstat(handle.fileno())):
            raise ValueError('File changed before reading')
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            n += len(block)
            if n > FILE_LIMIT:
                raise ValueError('File size limit exceeded')
            h.update(block)
            parts.append(block)
        after = os.fstat(handle.fileno())
    if signature(before) != signature(after) or signature(before) != signature(safe(root, relative).lstat()) or n != before.st_size:
        raise ValueError('File changed during reading')
    return b''.join(parts), h.hexdigest()


def json_object(blob):
    value = json.loads(blob.decode('utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def csv_rows(blob):
    return list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig'), newline='')))


def mean(values):
    return statistics.mean(values) if values else None


def integer(value):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError('Invalid integer field')
    return int(number)


def parse_predictions(blob, item_map):
    result = {}
    for row in csv_rows(blob):
        sid, user, target = row['sample_id'], row['user_id'], row['target_item']
        if not sid or not user or sid in result or target not in item_map:
            raise ValueError('Prediction sample/target identity invalid')
        rank = integer(row['target_rank'])
        raw = tuple(str(v) for v in json.loads(row['top10_raw_item_id']))
        idx = tuple(integer(v) for v in json.loads(row['top10_item_idx']))
        if rank < 1 or rank > len(item_map) or len(raw) != 10 or len(idx) != 10 or len(set(raw)) != 10 or len(set(idx)) != 10:
            raise ValueError('Prediction rank/Top10 shape invalid')
        if any(item_map.get(a) != b for a, b in zip(raw, idx)):
            raise ValueError('Top10 raw/index mapping mismatch')
        if ((rank <= 10) != (target in raw)) or (rank <= 10 and raw.index(target) + 1 != rank):
            raise ValueError('Target rank and Top10 disagree')
        for key, expected in (('ndcg10', 1 / math.log2(rank + 1) if rank <= 10 else 0), ('hit10', int(rank <= 10))):
            if key in row and not math.isclose(float(row[key]), expected, abs_tol=1e-10, rel_tol=1e-10):
                raise ValueError('Saved per-row metric disagrees with rank')
        result[sid] = (user, target, rank, raw, idx)
    if len({v[0] for v in result.values()}) != len(result):
        raise ValueError('Duplicate user in one-target panel')
    return result


def identities(records):
    return {k: v[:2] for k, v in records.items()}


def parse_labels(blob):
    result = {}
    hashes = ['original_sequence_sha256'] + ['perm_seed%d_sequence_sha256' % p for p in PERMS]
    for row in csv_rows(blob):
        sid = row['sample_id']
        if not sid or sid in result:
            raise ValueError('Duplicate/empty label sample')
        affected = str(row['affected']).lower()
        if affected not in ('0', '1', 'false', 'true'):
            raise ValueError('Invalid affected flag')
        affected = affected in ('1', 'true')
        size = integer(row['last_bucket_size'])
        sequence_hashes = tuple(row[k] for k in hashes)
        if any(not re.fullmatch('[a-fA-F0-9]{64}', h) for h in sequence_hashes):
            raise ValueError('Missing/invalid sequence identity hashes')
        sequence_hashes = tuple(h.lower() for h in sequence_hashes)
        distinct = len(set(sequence_hashes[1:]))
        if size < 1 or (affected and size < 2) or affected != (distinct >= 2) or integer(row['distinct_sequences_5perm']) != distinct:
            raise ValueError('Affected/distinct-sequence label mismatch')
        result[sid] = (row['user_id'], row['target_item'], size, affected, sequence_hashes)
    return result


def compare_population(predictions, labels, selected):
    primary_overlap, primary_flip, pair_flip, baseline_overlap, baseline_pairflip, baseline_anyflip, distinct_overlap, distinct_flat = [], [], [], [], [], [], [], []
    for sid in selected:
        rows = [prediction[sid] for prediction in predictions]
        permutation = rows[1:]
        overlaps = [len(set(a[3]) & set(b[3])) / 10 for a, b in itertools.combinations(permutation, 2)]
        flips = [(a[2] <= 10) != (b[2] <= 10) for a, b in itertools.combinations(permutation, 2)]
        states = [p[2] <= 10 for p in permutation]
        primary_overlap.append(mean(overlaps))
        primary_flip.append(any(states) and not all(states))
        pair_flip.append(mean(flips))
        baseline_overlap.append(mean([len(set(rows[0][3]) & set(p[3])) / 10 for p in permutation]))
        base_flips = [(rows[0][2] <= 10) != (p[2] <= 10) for p in permutation]
        baseline_pairflip.append(mean(base_flips))
        baseline_anyflip.append(any(base_flips))
        sequence_hashes = labels[sid][4][1:]
        distinct = [o for (a, b), o in zip(itertools.combinations(range(5), 2), overlaps) if sequence_hashes[a] != sequence_hashes[b]]
        if distinct:
            distinct_overlap.append(mean(distinct))
            distinct_flat.extend(distinct)
    return {'N': len(selected), 'overlap_5perm_pair_mean': mean(primary_overlap),
            'hit_flip_user_fraction': mean(primary_flip),
            'permutation_pairwise_hit_flip_mean_diagnostic': mean(pair_flip),
            'baseline_vs_permutation_overlap_diagnostic': mean(baseline_overlap),
            'baseline_vs_permutation_pairwise_hit_flip_diagnostic': mean(baseline_pairflip),
            'baseline_vs_any_permutation_user_flip_diagnostic': mean(baseline_anyflip),
            'distinct_input_pair_overlap_user_mean_diagnostic': mean(distinct_overlap),
            'distinct_input_pair_overlap_pooled_diagnostic': mean(distinct_flat),
            'distinct_input_pair_N': len(distinct_flat),
            'distinct_pair_user_N': len(distinct_overlap)}


class Collector:
    def __init__(self, stage):
        self.stage = absolute_plain(stage)
        if not self.stage.name.startswith('kais_control_stability_v1_'):
            raise ValueError('Unexpected stage prefix')
        verification = safe(self.stage, 'verification')
        if not verification.is_dir():
            raise ValueError('Stage verification directory missing')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_UTC_') + uuid.uuid4().hex[:10]
        self.out = verification / ('stability_review_' + stamp)
        self.out.mkdir(exist_ok=False)
        self.files, self.warnings, self.failures, self.runs = [], [], [], {}
        self.copies, self.total = {}, 0
        self.root, self.manifest, self.inputs = None, None, None

    def copy(self, origin, relative, archive_path, expected=None):
        record = {'path': relative, 'archive_path': archive_path, 'origin': str(origin), 'status': 'unknown'}
        try:
            relative_path(archive_path)
            blob, actual = read_stable(origin, relative)
            if expected is not None and actual != expected:
                raise ValueError('Pinned SHA256 mismatch')
            if self.total + len(blob) > TOTAL_LIMIT:
                raise ValueError('Collection exceeds 512 MiB limit')
            destination = safe(self.out, 'snapshot/' + archive_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(blob)
            self.total += len(blob)
            self.copies[archive_path] = destination
            record.update(status='copied', bytes=len(blob), sha256=actual)
            return blob
        except (OSError, ValueError) as exc:
            record.update(status='not_observed' if isinstance(exc, FileNotFoundError) else 'rejected_or_changed', error=str(exc)[:200])
            self.warnings.append(record['status'] + ': ' + relative)
            return None
        finally:
            self.files.append(record)

    def verify_sources(self):
        companion = absolute_plain(INPUTS_FILE)
        blob, sha = read_stable(companion.parent, companion.name)
        if sha != EXPECTED_INPUTS_SHA256:
            raise ValueError('Companion input manifest SHA mismatch')
        self.inputs = json_object(blob)
        manifest_blob, manifest_sha = read_stable(self.stage, MANIFEST)
        self.manifest = json_object(manifest_blob)
        m = self.manifest
        if m.get('version') != 'control_stability_v1' or m.get('inputs_sha256') != sha or m.get('inputs') != self.inputs:
            raise ValueError('Stage/input manifest binding mismatch')
        self.root = absolute_plain(m['root'])
        if self.stage.parent != self.root or str(self.stage) != m.get('stage'):
            raise ValueError('Stage/root identity mismatch')
        if self.inputs.get('training_seeds') != list(SEEDS) or self.inputs.get('permutation_seeds') != list(PERMS) or set(self.inputs['datasets']) != set(TAGS):
            raise ValueError('Unexpected planned datasets/seeds/permutations')
        if m.get('sources_sha256') != self.inputs.get('sources_sha256') or m.get('data_sha256') != self.inputs.get('data_sha256'):
            raise ValueError('Manifest source/data identities disagree')
        verified = []
        for relative, expected in m['sources_sha256'].items():
            if not relative_path(relative).startswith('step38_kais_retrain/scripts/') or not relative.endswith('.py'):
                raise ValueError('Unexpected original source path')
            original, actual = read_stable(self.root, relative)
            if actual != expected:
                raise ValueError('Original source SHA mismatch')
            ast.parse(original.decode('utf-8-sig'), filename=relative)
            verified.append((self.root, relative, 'original_sources/' + relative, expected))
        generated = m.get('generated_sha256', {})
        if not generated or len(generated) > 64:
            raise ValueError('Unexpected generated source count')
        for relative, expected in generated.items():
            if not relative_path(relative).startswith('scripts/') or not relative.endswith('.py'):
                raise ValueError('Unexpected generated source path')
            source, actual = read_stable(self.stage, relative)
            if actual != expected:
                raise ValueError('Generated source SHA mismatch')
            ast.parse(source.decode('utf-8-sig'), filename=relative)
            verified.append((self.stage, relative, relative, expected))
        self.manifest_sha = manifest_sha
        verified.append((self.stage, MANIFEST, MANIFEST, manifest_sha))
        for origin, relative, destination, expected in verified:
            if self.copy(origin, relative, destination, expected) is None:
                raise ValueError('Verified source changed while collecting')
        if self.copy(companion.parent, companion.name, 'manifests/pinned_inputs.json', sha) is None:
            raise ValueError('Companion input manifest changed while collecting')

    def run(self, tag, seed):
        plan = self.inputs['datasets'][tag]
        specification = plan['runs'][str(seed)]
        run_name = '%s_seed%d' % (tag, seed)
        result = {'tag': tag, 'seed': seed, 'status': 'incomplete_no_statistics', 'checks': {}, 'errors': []}
        self.runs[run_name] = result
        try:
            relative_path(specification['model_dir'])
            if not (specification['model_dir'].startswith('step38_kais_retrain/outputs/') or
                    re.match(r'^kais_(?:orderfix_v1_|beauty_control_seeds_v1_)[^/]+/outputs/', specification['model_dir'])):
                raise ValueError('Model reference directory outside permitted stages')
            references, output = {}, {}
            for name in REFERENCE_FILES:
                expected = specification['files_sha256'][name]
                references[name] = self.copy(self.root, specification['model_dir'] + '/' + name,
                                              'references/' + run_name + '/' + name, expected)
            for name in RUN_FILES:
                output[name] = self.copy(self.stage, 'outputs/' + run_name + '/' + name, 'outputs/' + run_name + '/' + name)
            if any(v is None for v in list(references.values()) + list(output.values())):
                result['errors'].append('Missing/rejected required files; statistics not computed.')
                return
            config, saved_metrics = json_object(references['config.json']), json_object(references['metrics.json'])
            mapping_rows = csv_rows(references['item_id_map.csv'])
            item_map = {r['raw_item_id']: integer(r['item_idx']) for r in mapping_rows}
            if len(item_map) != len(mapping_rows) or set(item_map.values()) != set(range(1, len(item_map) + 1)):
                raise ValueError('Non-bijective item map')
            predictions = [parse_predictions(output[name], item_map) for name in PREDICTIONS]
            saved = parse_predictions(references['test_per_sample_predictions.csv'], item_map)
            labels = parse_labels(output['panel_labels.csv'])
            panel = identities(saved)
            if any(identities(p) != panel for p in predictions) or {k: v[:2] for k, v in labels.items()} != panel:
                raise ValueError('Sample/user/target panel mismatch')
            if len(panel) != integer(plan['expected_samples']):
                raise ValueError('Unexpected panel sample count')
            rank_mismatch = sum(saved[k][2] != predictions[0][k][2] for k in panel)
            top_mismatch = sum(saved[k][3:] != predictions[0][k][3:] for k in panel)
            if rank_mismatch or top_mismatch:
                raise ValueError('Canonical predictions do not reproduce saved reference')
            identical_mismatch = 0
            for sid in panel:
                seen = {}
                for value, sequence_sha in zip([p[sid] for p in predictions], labels[sid][4]):
                    if sequence_sha in seen and seen[sequence_sha] != value[2:]:
                        identical_mismatch += 1
                    seen[sequence_sha] = value[2:]
            if identical_mismatch:
                raise ValueError('Identical actual inputs produce different rank/Top10')
            meta = json_object(output['metadata.json'])
            expected_paths = [plan['original_path']] + [plan['permutation_paths'][str(p)] for p in PERMS]
            expected_data = {p: self.inputs['data_sha256'][p] for p in expected_paths}
            expected_checkpoint = specification['model_dir'] + '/' + specification['checkpoint']
            checks = {'dataset': meta.get('dataset') == plan['dataset'] == TAGS[tag] == config.get('dataset') == saved_metrics.get('dataset'),
                'seed': integer(meta.get('model_seed')) == seed == integer(config.get('seed')) == integer(saved_metrics.get('seed')),
                'epoch': integer(meta.get('best_epoch')) == integer(specification['best_epoch']) == integer(saved_metrics.get('best_epoch')),
                'num_items': integer(meta.get('num_items')) == len(item_map),
                'sample_count': integer(meta.get('test_samples')) == len(panel),
                'permutation_seeds': meta.get('permutation_seeds') == list(PERMS),
                'input_manifest_sha256': meta.get('input_manifest_sha256') == EXPECTED_INPUTS_SHA256,
                'preparation_manifest_sha256': meta.get('preparation_manifest_sha256') == self.manifest_sha,
                'reference_files_sha256': meta.get('reference_files_sha256') == specification['files_sha256'],
                'data_sha256': meta.get('data_sha256') == expected_data,
                'model_dir': meta.get('model_dir') in (specification['model_dir'], str(self.root / specification['model_dir'])),
                'checkpoint_path_identity_only_not_opened': meta.get('checkpoint_path') in (expected_checkpoint, str(self.root / expected_checkpoint)),
                'original_rank_mismatches': integer(meta.get('original_rank_mismatches')) == rank_mismatch == 0,
                'original_top10_mismatches': integer(meta.get('original_top10_mismatches')) == top_mismatch == 0,
                'identical_input_prediction_consistency': identical_mismatch == 0}
            result['checks'] = checks
            if not all(checks.values()):
                raise ValueError('Metadata identity mismatch: ' + ','.join(k for k, v in checks.items() if not v))
            result.update(status='valid_saved_predictions', best_epoch=meta['best_epoch'], num_items=len(item_map),
                populations={name: compare_population(predictions, labels, selected) for name, selected in
                    (('complete', list(panel)), ('affected', [k for k, v in labels.items() if v[3]]))})
            result['_baseline'], result['_labels'], result['_map'] = predictions[0], labels, item_map
            result['_config_core'] = {k: v for k, v in config.items() if k not in ('seed', 'orderfix', 'seed_extension')}
        except (OSError, ValueError, KeyError, TypeError, SyntaxError, csv.Error) as exc:
            result['status'] = 'invalid_no_statistics'
            result.pop('populations', None)
            result['errors'].append(type(exc).__name__ + ': ' + str(exc)[:240])

    def summaries(self):
        summaries = {}
        for tag in TAGS:
            runs = [self.runs['%s_seed%d' % (tag, s)] for s in SEEDS]
            data = {'status': 'incomplete_no_three_seed_summary'}
            summaries[tag] = data
            if not all(r['status'] == 'valid_saved_predictions' for r in runs):
                continue
            aligned = {'labels_equal_across_seeds': all(r['_labels'] == runs[0]['_labels'] for r in runs),
                       'maps_equal_across_seeds': all(r['_map'] == runs[0]['_map'] for r in runs),
                       'core_configs_equal_except_seed_and_provenance': all(r['_config_core'] == runs[0]['_config_core'] for r in runs),
                       'baseline_panels_equal_across_seeds': all(identities(r['_baseline']) == identities(runs[0]['_baseline']) for r in runs)}
            data['alignment'] = aligned
            if not all(aligned.values()):
                data['status'] = 'cross_seed_mismatch_no_summary'
                continue
            data.update(status='three_seed_saved_result_summary', populations={})
            for population in ('complete', 'affected'):
                labels = runs[0]['_labels']
                selected = [k for k in labels if population == 'complete' or labels[k][3]]
                metrics = {}
                for metric in runs[0]['populations'][population]:
                    if metric in ('N', 'distinct_pair_user_N', 'distinct_input_pair_N'):
                        continue
                    values = [r['populations'][population][metric] for r in runs]
                    metrics[metric] = {'mean': statistics.mean(values), 'sample_SD_ddof1': statistics.stdev(values)} if all(v is not None for v in values) else {'mean': None, 'sample_SD_ddof1': None}
                pairs = []
                for a, b in itertools.combinations(range(3), 2):
                    left, right = runs[a]['_baseline'], runs[b]['_baseline']
                    overlap = [len(set(left[k][3]) & set(right[k][3])) / 10 for k in selected]
                    flips = [(left[k][2] <= 10) != (right[k][2] <= 10) for k in selected]
                    pairs.append({'training_seeds': [SEEDS[a], SEEDS[b]], 'N': len(selected),
                                  'baseline_top10_overlap': mean(overlap), 'baseline_pair_hit_flip_fraction': mean(flips)})
                any_seed_flip = [len({r['_baseline'][k][2] <= 10 for r in runs}) > 1 for k in selected]
                data['populations'][population] = {'N': len(selected), 'three_seed_mean_sampleSD': metrics,
                    'distinct_input_pair_N': runs[0]['populations'][population]['distinct_input_pair_N'],
                    'distinct_pair_user_N': runs[0]['populations'][population]['distinct_pair_user_N'],
                    'training_seed_baseline_pairs_reference': pairs,
                    'training_seed_baseline_three_pair_overlap_mean_reference': mean([p['baseline_top10_overlap'] for p in pairs]) if selected else None,
                    'training_seed_baseline_three_pair_hit_flip_mean_reference': mean([p['baseline_pair_hit_flip_fraction'] for p in pairs]) if selected else None,
                    'training_seed_baseline_any_user_flip_across_three_seeds_reference': mean(any_seed_flip)}
        return summaries

    def finish(self, summaries):
        public_runs = {name: {k: v for k, v in run.items() if not k.startswith('_')} for name, run in self.runs.items()}
        complete = len(public_runs) == 12 and all(r['status'] == 'valid_saved_predictions' for r in public_runs.values()) and all(s['status'] == 'three_seed_saved_result_summary' for s in summaries.values())
        status = 'INTEGRITY_FAILED_NO_BUNDLE' if self.failures else ('ALL_TWELVE_SAVED_RUNS_CHECKED' if complete else 'PARTIAL_COLLECTION_REQUIRES_REVIEW')
        result = {'status': status, 'stage': str(self.stage), 'input_manifest_sha256': EXPECTED_INPUTS_SHA256,
            'run_checks': public_runs, 'dataset_summaries': summaries, 'files': self.files, 'warnings': self.warnings,
            'failures': self.failures, 'formulas': {
                'overlap_5perm_pair_mean': 'For each sample average all 10 unordered pairs of five permuted Top10 sets (intersection/10), then average samples; repeated input pairs retained.',
                'hit_flip_user_fraction': 'Fraction of samples with at least one hit@10 and one nonhit among FIVE PERMUTATIONS; original prediction excluded.',
                'affected': 'Five actual permuted input sequence hashes contain at least two distinct values; original is excluded from this definition.',
                'baseline_diagnostics': 'Canonical-versus-five-permutation metrics are diagnostics, separate from primary five-permutation metrics.',
                'training_seed_reference': 'Three unordered pairs of canonical-input model predictions, using the same complete/affected sample population; never pooled with input-permutation results.',
                'empty_population': 'N=0 yields null means/SD; no division by zero.',
                'three_seed_SD': 'Sample standard deviation across the three fixed model seeds, ddof=1.'},
            'scope': 'PRIVATE saved-file verification; no training, inference, checkpoint access, or interaction-data reads. Sequence hashes are generated provenance labels, not independently recomputed here from raw inputs. No full reproduction, significance, equivalence or best-seed claim.'}
        (self.out / 'stability_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        fields = ['origin', 'path', 'archive_path', 'status', 'bytes', 'sha256', 'error']
        with (self.out / 'COLLECTED_SHA256.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.files)
        lines = ['# PRIVATE Control stability evidence review', '', 'Status: **' + status + '**', '', result['scope'], '',
            'Primary overlap averages ten pairs per sample, then samples. Primary hit_flip_user_fraction is ANY hit-status variation across the FIVE permutations, not a pair-flip mean.',
            'Affected masks use distinct FIVE permutation inputs and are checked identical across model seeds. Empty affected populations produce null metrics.',
            'Missing/invalid runs are collected where possible; they receive no statistics and prevent their dataset three-seed summary.', '', '## Runs', '']
        lines += ['- ' + k + ': ' + v['status'] for k, v in public_runs.items()]
        lines += ['', '## Findings', ''] + ['- ' + x for x in self.failures + self.warnings]
        lines += ['', 'See stability_checks.json for all populations, three-seed SD and separately named training-seed references.',
                  'Prediction CSVs/labels/references contain user-level material. Do not publish this private ZIP directly.', '']
        (self.out / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
        bundle = None
        if not self.failures:
            bundle = self.out / 'control_stability_results.zip'
            paths = sorted(p for p in self.out.rglob('*') if p.is_file())
            with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for path in paths:
                    if path.is_symlink():
                        raise ValueError('Output symbolic link rejected')
                    archive.write(path, path.relative_to(self.out).as_posix())
            h = hashlib.sha256()
            with bundle.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    h.update(block)
            (self.out / 'control_stability_results.zip.sha256.txt').write_text(h.hexdigest() + '  ' + bundle.name + '\n', encoding='ascii')
        return {'report_dir': str(self.out), 'zip_path': str(bundle) if bundle else None, 'status': status, 'exit_code': 2 if self.failures else 0}


def collect(stage):
    collector = Collector(stage)
    summaries = {}
    try:
        collector.verify_sources()
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        collector.failures.append(type(exc).__name__ + ': ' + str(exc)[:240])
    if not collector.failures:
        for path in sorted(safe(collector.stage, 'verification').iterdir()):
            if path.is_file() and not path.name.startswith('.') and path.suffix.lower() in ('.log', '.json'):
                collector.copy(collector.stage, 'verification/' + path.name, 'verification/' + path.name)
        for tag in TAGS:
            for seed in SEEDS:
                collector.run(tag, seed)
        summaries = collector.summaries()
    return collector.finish(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.stage.expanduser())
    print('SUMMARY=' + str(Path(result['report_dir']) / 'SUMMARY.md'))
    print('STABILITY_REPORT_ZIP=' + (result['zip_path'] or 'NOT_CREATED'))
    print('STATUS=' + result['status'])
    return result['exit_code']


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        print('COLLECTION_FAILED=' + type(exc).__name__ + ': ' + str(exc)[:240], file=sys.stderr)
        sys.exit(2)
