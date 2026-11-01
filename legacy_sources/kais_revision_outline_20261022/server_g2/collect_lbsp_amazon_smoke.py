#!/usr/bin/env python3
"""Collect private Amazon LBSP smoke evidence without running research code.

Standard library only. Never reads interaction CSV contents or model weights.
Smoke summaries describe random initialized model checks, not model quality.
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

INPUTS_FILE = Path(__file__).absolute().with_name('lbsp_amazon_smoke_inputs_20261023.json')
EXPECTED_INPUTS_SHA256 = 'c65912d443b01d2f76ed164e57638a99ad1b1972394d542b5f22fe1eacd92265'
TAGS = {'beauty': 'Amazon_Beauty', 'sports': 'Amazon_Sports', 'toys': 'Amazon_Toys'}
REFERENCE_FILES = ('config.json', 'metrics.json', 'training_log.csv', 'item_id_map.csv', 'test_per_sample_predictions.csv')
MANIFEST = 'manifests/lbsp_amazon_smoke_manifest.json'
GENERATED = ('scripts/model_definition.py',) + tuple('scripts/smoke_%s_lbsp.py' % t for t in TAGS)
FILE_LIMIT, TOTAL_LIMIT = 64 * 1024 ** 2, 256 * 1024 ** 2
NORMALIZER_AST_SHA256 = {
    'normalize_eval_records': '902563b2402ef9dbdbe56fa834bf47ba89f99408c0239ca6a71ce3c253ad0915',
    'normalize_eval_frame': '2c02b83c8ce77a06316fcb1c010cc69bad46a48b828d22e0aeeb374d496819c1',
}

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
    if PurePosixPath(relative).suffix.lower() in ('.pt', '.pth', '.ckpt') or relative.startswith(('data/', 'processed/')):
        raise ValueError('Checkpoint/interaction-input access forbidden')
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


def integer(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not an integer count')
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError('Invalid integer count')
    return int(number)


def finite(value, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a numeric measurement')
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        raise ValueError('Nonfinite or out-of-range smoke measurement')
    return number


def ast_definitions(blob):
    tree = ast.parse(blob.decode('utf-8-sig'))
    return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted_name(node.value) + '.' + node.attr
    return ''


class Collector:
    def __init__(self, stage):
        self.stage = absolute_plain(stage)
        if not self.stage.name.startswith('kais_lbsp_amazon_smoke_v1_'):
            raise ValueError('Unexpected smoke stage prefix')
        verification = safe(self.stage, 'verification')
        if not verification.is_dir():
            raise ValueError('Smoke verification directory missing')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
        self.out = verification / ('lbsp_amazon_smoke_review_' + stamp)
        self.out.mkdir(exist_ok=False)
        self.files, self.warnings, self.integrity_failures = [], [], []
        self.total, self.copies, self.sources, self.generated = 0, {}, {}, {}
        self.root, self.inputs, self.manifest, self.manifest_sha = None, None, None, None
        self.run_checks, self.scope_check, self.driver_check = {}, {}, {}

    def copy(self, origin, relative, destination, expected=None):
        record = {'origin': str(origin), 'path': relative, 'archive_path': destination, 'status': 'unknown'}
        try:
            relative_path(destination)
            blob, actual = read_stable(origin, relative)
            if expected is not None and actual != expected:
                raise ValueError('Pinned SHA256 mismatch')
            if self.total + len(blob) > TOTAL_LIMIT:
                raise ValueError('Collection exceeds 256 MiB limit')
            target = safe(self.out, 'snapshot/' + destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(blob)
            self.total += len(blob)
            self.copies[destination] = blob
            record.update(status='copied', sha256=actual, bytes=len(blob))
            return blob
        except (OSError, ValueError) as exc:
            record.update(status='missing' if isinstance(exc, FileNotFoundError) else 'rejected_or_changed', error=str(exc)[:240])
            self.warnings.append(record['status'] + ': ' + relative)
            if expected is not None and isinstance(exc, ValueError):
                self.integrity_failures.append('Pinned file rejected or changed: ' + relative)
            return None
        finally:
            self.files.append(record)

    def verify_sources(self):
        companion = absolute_plain(INPUTS_FILE)
        blob, digest = read_stable(companion.parent, companion.name)
        if digest != EXPECTED_INPUTS_SHA256:
            raise ValueError('Companion input manifest SHA256 mismatch')
        self.inputs = json_object(blob)
        manifest_blob, manifest_sha = read_stable(self.stage, MANIFEST)
        self.manifest = json_object(manifest_blob)
        m = self.manifest
        if (m.get('version') != 'lbsp_amazon_smoke_v1' or m.get('inputs') != self.inputs
                or m.get('inputs_sha256') != digest):
            raise ValueError('Preparation/input manifest binding mismatch')
        self.root = absolute_plain(m['root'])
        if self.stage.parent != self.root or m.get('stage') != str(self.stage):
            raise ValueError('Stage/root identity mismatch')
        if (self.inputs.get('version') != 'lbsp_amazon_smoke_v1' or self.inputs.get('seed') != 42
                or set(self.inputs['datasets']) != set(TAGS)
                or m.get('sources_sha256') != self.inputs['sources_sha256']
                or m.get('data_sha256') != self.inputs['data_sha256']
                or len(m['sources_sha256']) != 5 or len(m['data_sha256']) != 9):
            raise ValueError('Unexpected smoke plan source/data/dataset identities')
        if set(m.get('generated_sha256', {})) != set(GENERATED):
            raise ValueError('Unexpected generated script inventory')
        for relative, expected in m['sources_sha256'].items():
            if not relative_path(relative).endswith('.py'):
                raise ValueError('Unexpected source extension')
            source, actual = read_stable(self.root, relative)
            if actual != expected:
                raise ValueError('Original source SHA256 mismatch')
            ast.parse(source.decode('utf-8-sig'))
            self.sources[relative] = source
        for relative, expected in m['generated_sha256'].items():
            source, actual = read_stable(self.stage, relative)
            if actual != expected:
                raise ValueError('Generated source SHA256 mismatch')
            ast.parse(source.decode('utf-8-sig'))
            self.generated[relative] = source
        self.verify_algorithm_ast()
        self.manifest_sha = manifest_sha
        for relative, source in self.sources.items():
            if self.copy(self.root, relative, 'original_sources/' + relative, self.inputs['sources_sha256'][relative]) is None:
                raise ValueError('Original source changed while copying')
        for relative in self.generated:
            if self.copy(self.stage, relative, relative, m['generated_sha256'][relative]) is None:
                raise ValueError('Generated source changed while copying')
        if self.copy(self.stage, MANIFEST, MANIFEST, manifest_sha) is None:
            raise ValueError('Manifest changed while copying')
        if self.copy(companion.parent, companion.name, 'manifests/pinned_inputs.json', digest) is None:
            raise ValueError('Input manifest changed while copying')

    def verify_algorithm_ast(self):
        template = ast_definitions(self.sources[self.inputs['template_source']])
        original_model = ast_definitions(self.sources[self.inputs['model_source']])
        frozen = ast_definitions(self.generated['scripts/model_definition.py'])
        model_names = ('build_r1_group_positions_from_buckets', 'build_r1_group_positions_from_size',
                       'canonicalize_last_bucket_items', 'RecentBucketSetR1SASRec')
        for name in model_names:
            if name not in original_model or frozen.get(name) != original_model[name]:
                raise ValueError('Frozen model algorithm AST differs: ' + name)
        if frozen.get('predict_batch_masked') != template.get('predict_batch_masked'):
            raise ValueError('Frozen masked-forward AST differs')
        for tag in TAGS:
            generated = self.generated['scripts/smoke_%s_lbsp.py' % tag]
            emitted = ast_definitions(generated)
            for name, expected in NORMALIZER_AST_SHA256.items():
                if hashlib.sha256(emitted.get(name, '').encode('utf-8')).hexdigest() != expected:
                    raise ValueError('Evaluation normalization adapter AST differs: %s/%s' % (tag, name))
            for name in ('AllPositionR1Dataset', 'training_loss', 'evaluate', 'predict_batch_masked'):
                if name not in template or emitted.get(name) != template[name]:
                    raise ValueError('Frozen smoke algorithm AST differs: %s/%s' % (tag, name))
            tree = ast.parse(generated.decode('utf-8-sig'))
            backward = [i for i, n in enumerate(tree.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and dotted_name(n.value.func) == 'loss.backward']
            gradient_statement = ast.parse('_ALL_GRADIENTS_FINITE = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())').body[0]
            if (len(backward) != 1 or backward[0] + 2 >= len(tree.body)
                    or ast.dump(tree.body[backward[0] + 1], include_attributes=False) != ast.dump(gradient_statement, include_attributes=False)):
                raise ValueError('All-gradient finite capture is not immediately after backward')
            require_statement = tree.body[backward[0] + 2]
            if (not isinstance(require_statement, ast.Expr) or not isinstance(require_statement.value, ast.Call)
                    or dotted_name(require_statement.value.func) != 'require' or not require_statement.value.args
                    or not isinstance(require_statement.value.args[0], ast.Name) or require_statement.value.args[0].id != '_ALL_GRADIENTS_FINITE'):
                raise ValueError('All-gradient finite capture is not asserted before zero_grad')
            forbidden = {'torch.load', 'torch.save', 'torch.optim.Adam', 'torch.optim.AdamW', 'optimizer.step', 'optimizer.zero_grad'}
            if any(isinstance(n, ast.Call) and dotted_name(n.func) in forbidden for n in ast.walk(tree)):
                raise ValueError('Formal training or model-weight call present in smoke source')
            loaders = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_eval']
            if len(loaders) != 1:
                raise ValueError('Expected one evaluation loader')
            loader = loaders[0]
            expected_insert = ast.dump(ast.parse('df = normalize_eval_frame(df)').body[0], include_attributes=False)
            insertions = [i for i, statement in enumerate(loader.body) if ast.dump(statement, include_attributes=False) == expected_insert]
            if len(insertions) != 1 or insertions[0] != 1:
                raise ValueError('Evaluation loader normalization adapter is not uniquely in the approved location')
            del loader.body[insertions[0]]
            if ast.dump(loader, include_attributes=False) != template['load_eval']:
                raise ValueError('Evaluation loader changes exceed the normalization adapter')
        self.algorithm_ast_check = {'status': 'PASS', 'scope': 'Frozen model/helpers, masked forward, dataset, loss and evaluator AST match pinned sources. The evaluation loader permits exactly one df=normalize_eval_frame(df) adapter after read_csv; all remaining AST matches. No source executed.'}

    def verify_scope(self):
        problems = []
        outputs = safe(self.stage, 'outputs')
        observed = []
        if outputs.exists():
            if not outputs.is_dir():
                problems.append('outputs is not an ordinary directory')
            else:
                observed = [p.name for p in outputs.iterdir()]
                if observed:
                    problems.append('Smoke-only stage outputs is not empty')
        visited = 0
        for base, dirs, filenames in os.walk(str(self.stage), followlinks=False):
            current = Path(base)
            if current == self.stage / 'verification':
                dirs[:] = []  # Collector report subfolders are independent outputs.
            for name in list(dirs) + filenames:
                visited += 1
                if visited > 4096:
                    problems.append('Unexpected stage entry count')
                    break
                candidate = current / name
                info = candidate.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                    problems.append('Stage contains symbolic link/reparse point: ' + candidate.relative_to(self.stage).as_posix())
                if candidate.suffix.lower() in ('.pt', '.pth', '.ckpt', '.safetensors'):
                    problems.append('Unexpected model-weight artifact: ' + candidate.relative_to(self.stage).as_posix())
            if visited > 4096:
                break
        self.scope_check = {'status': 'PASS' if not problems else 'FAIL', 'outputs_exists': outputs.exists(),
                            'outputs_entries': observed[:64], 'weights_or_formal_outputs_read': False, 'errors': problems}

    def references(self, tag):
        plan = self.inputs['datasets'][tag]
        reference = plan['control_reference']
        model_dir = relative_path(reference['model_dir'])
        if set(reference['files_sha256']) != set(REFERENCE_FILES):
            raise ValueError('Unexpected Control reference inventory')
        blobs = {name: self.copy(self.root, model_dir + '/' + name, 'references/' + tag + '_seed42/' + name,
                                reference['files_sha256'][name]) for name in REFERENCE_FILES}
        if any(v is None for v in blobs.values()):
            raise ValueError('Missing/rejected pinned Control references')
        config, metrics = json_object(blobs['config.json']), json_object(blobs['metrics.json'])
        if config.get('dataset') != plan['dataset'] or metrics.get('dataset') != plan['dataset'] or integer(config.get('seed')) != 42 or integer(metrics.get('seed')) != 42:
            raise ValueError('Pinned Control identity mismatch')
        mapping_rows = csv_rows(blobs['item_id_map.csv'])
        mapping = {r['raw_item_id']: integer(r['item_idx']) for r in mapping_rows}
        if (len(mapping_rows) != len(mapping) or len(mapping) != integer(plan['expected_items'])
                or set(mapping.values()) != set(range(1, len(mapping) + 1))):
            raise ValueError('Pinned Control map size/bijection mismatch')
        for phase, expected in (('validation', plan['expected_val']), ('test', plan['expected_test'])):
            value = metrics[phase]
            count = value['overall']['N'] if 'overall' in value else value['samples']
            if integer(count) != integer(expected):
                raise ValueError('Pinned Control %s count mismatch' % phase)
        predictions = csv_rows(blobs['test_per_sample_predictions.csv'])
        if (len(predictions) != integer(plan['expected_test']) or len({r['sample_id'] for r in predictions}) != len(predictions)
                or any(r['target_item'] not in mapping for r in predictions) or not csv_rows(blobs['training_log.csv'])):
            raise ValueError('Pinned Control prediction/log count or identity mismatch')
        return {'num_items': len(mapping), 'control_reference_sha256': reference['files_sha256'],
                'control_map_sha256': reference['files_sha256']['item_id_map.csv']}

    def run(self, tag):
        plan = self.inputs['datasets'][tag]
        result = {'dataset': plan['dataset'], 'tag': tag, 'seed': 42, 'status': 'INCOMPLETE', 'checks': {}, 'errors': []}
        self.run_checks[tag] = result
        try:
            reference = self.references(tag)
            blob = self.copies.get('verification/smoke_' + tag + '.json')
            if blob is None:
                result['errors'].append('Required smoke JSON summary not observed.')
                return
            summary = json_object(blob)
            data_paths = [plan[k] for k in ('train_path', 'validation_path', 'test_path')]
            data_hashes = {p: self.inputs['data_sha256'][p] for p in data_paths}
            checks = {
                'version': summary.get('version') == 'lbsp_amazon_smoke_v1',
                'dataset_tag_seed': summary.get('dataset') == TAGS[tag] == plan['dataset'] and summary.get('tag') == tag and integer(summary.get('seed')) == 42,
                'smoke_pass_record': summary.get('status') == 'SMOKE_PASS',
                'input_manifest_sha256': summary.get('input_manifest_sha256') == EXPECTED_INPUTS_SHA256,
                'preparation_manifest_sha256': summary.get('preparation_manifest_sha256') == self.manifest_sha,
                'generated_script_sha256': summary.get('generated_script_sha256') == self.manifest['generated_sha256']['scripts/smoke_%s_lbsp.py' % tag],
                'model_definition_sha256': summary.get('model_definition_sha256') == self.manifest['generated_sha256']['scripts/model_definition.py'],
                'source_hashes': summary.get('sources_sha256') == self.inputs['sources_sha256'],
                'data_hashes_identity_only_not_opened': summary.get('data_sha256') == data_hashes,
                'control_reference_hashes': summary.get('reference_files_sha256') == plan['control_reference']['files_sha256'],
                'control_reference_identity': summary.get('control_reference') == plan['control_reference'],
                'sequence_sort_keys': summary.get('sequence_sort_keys') == plan['sequence_sort_keys'],
                'fixed_hyperparameters': summary.get('fixed_hyperparameters') == self.inputs['hyperparameters'],
            }
            counts = summary['counts']
            if set(counts) != {'num_items', 'training_rows', 'trainable_users', 'training_positions', 'validation_samples', 'test_samples'}:
                raise ValueError('Unexpected counts schema')
            nitems = integer(counts['num_items'])
            users = integer(counts['trainable_users'])
            nrows = integer(counts['training_rows'])
            positions = integer(counts['training_positions'])
            checks.update({
                'control_item_map_count': nitems == reference['num_items'] == integer(plan['expected_items']),
                'train_position_count': positions == integer(plan['expected_positions']),
                'positive_training_row_and_user_counts': 0 < users <= nrows and users <= positions,
                'validation_count': integer(counts['validation_samples']) == integer(plan['expected_val']),
                'test_count': integer(counts['test_samples']) == integer(plan['expected_test']),
            })
            loss = finite(summary['smoke_loss'], positive=True)
            gradients = integer(summary['finite_gradient_count'])
            checks['finite_positive_loss_and_gradients'] = gradients > 0
            checks['all_gradients_finite_server_record'] = summary.get('all_gradients_finite') is True
            parity = summary['parity']
            if set(parity) != {'max_abs_diff', 'atol', 'rtol', 'allclose', 'samples'}:
                raise ValueError('Unexpected parity schema')
            maxdiff = finite(parity['max_abs_diff'], nonnegative=True)
            checks['masked_unpadded_parity_original_allclose'] = (parity.get('allclose') is True
                and finite(parity['atol']) == 1e-4 and finite(parity['rtol']) == 1e-5 and integer(parity['samples']) == 32)
            invariance = summary['invariance']
            if set(invariance) != {'tested', 'arrangements_per_sample', 'exact_logits_invariance'}:
                raise ValueError('Unexpected invariance schema')
            tested = integer(invariance['tested'])
            checks['last_bucket_exact_invariance_server_record'] = (1 <= tested <= 16
                and integer(invariance['arrangements_per_sample']) == 6 and invariance.get('exact_logits_invariance') is True)
            validation = summary['validation_smoke']
            if set(validation) != {'samples', 'ndcg10', 'hr10', 'hr20', 'mean_rank'}:
                raise ValueError('Unexpected validation smoke schema')
            val_values = {name: finite(validation[name], nonnegative=True) for name in ('ndcg10', 'hr10', 'hr20', 'mean_rank')}
            checks['32_sample_validation_smoke_finite'] = (integer(validation['samples']) == 32
                and 0 <= val_values['ndcg10'] <= 1 and 0 <= val_values['hr10'] <= val_values['hr20'] <= 1
                and 1 <= val_values['mean_rank'] <= nitems)
            for field in ('formal_training_started', 'weights_read', 'weights_statted', 'test_inference_performed'):
                checks[field + '_false'] = summary.get(field) is False
            checks['optimizer_steps_zero'] = integer(summary.get('optimizer_steps')) == 0
            for field in ('random_initialization', 'outputs_empty', 'original_hashes_unchanged'):
                checks[field + '_true'] = summary.get(field) is True
            checks['stage_has_no_formal_outputs'] = self.scope_check.get('status') == 'PASS'
            result['checks'] = checks
            if not all(checks.values()):
                raise ValueError('Smoke summary checks failed: ' + ','.join(k for k, good in checks.items() if not good))
            result.update(status='PASS', counts=counts, smoke_loss=loss, finite_gradient_count=gradients,
                parity={'max_abs_diff': maxdiff, 'allclose_server_record': True, 'atol': 1e-4, 'rtol': 1e-5, 'samples': 32},
                invariance=invariance, validation_smoke=validation,
                interpretation='Random initialized smoke diagnostics only; not formal training, test evaluation, or method effectiveness.')
        except (OSError, ValueError, KeyError, TypeError, SyntaxError, csv.Error) as exc:
            result['status'] = 'FAIL'
            result['errors'].append(type(exc).__name__ + ': ' + str(exc)[:320])

    def driver(self):
        names = [name for name in self.copies if re.fullmatch(r'verification/lbsp_amazon_smoke_driver_[^/]+\.json', name)]
        self.driver_check = {'status': 'MISSING' if not names else 'FAIL', 'errors': []}
        if len(names) != 1:
            self.driver_check['errors'].append('Exactly one driver JSON is required for the all-three result.')
            return
        try:
            driver = json_object(self.copies[names[0]])
            planned = {tag + '_seed42' for tag in TAGS}
            checks = {'stage': driver.get('stage') == str(self.stage),
                'driver_status': driver.get('status') == 'ALL_THREE_SMOKES_EXITED_ZERO',
                'training_performed_false': driver.get('training_performed') is False,
                'three_runs_only': set(driver.get('runs', {})) == planned}
            for tag in TAGS:
                run = driver['runs'][tag + '_seed42']
                expected = [driver['python'], '-u', str(self.stage / ('scripts/smoke_%s_lbsp.py' % tag)), '--mode', 'smoke']
                log = Path(run['log'])
                checks[tag + '_command_smoke_only'] = run.get('command') == expected
                checks[tag + '_exit_zero'] = integer(run.get('exit_code')) == 0
                checks[tag + '_log_within_verification'] = log.is_absolute() and log.parent == self.stage / 'verification' and log.name.endswith('.log')
                checks[tag + '_log_collected'] = 'verification/' + log.name in self.copies
                checks[tag + '_summary_pass'] = self.run_checks.get(tag, {}).get('status') == 'PASS'
            self.driver_check.update(status='PASS' if all(checks.values()) else 'FAIL', path=names[0], checks=checks)
            if not all(checks.values()):
                self.driver_check['errors'].append('Driver and smoke summary do not consistently record all three passing smoke-only runs.')
        except (ValueError, KeyError, TypeError) as exc:
            self.driver_check['errors'].append(type(exc).__name__ + ': ' + str(exc)[:240])

    def finish(self):
        all_pass = (len(self.run_checks) == 3 and all(r['status'] == 'PASS' for r in self.run_checks.values())
                    and self.scope_check.get('status') == 'PASS' and self.driver_check.get('status') == 'PASS')
        status = ('INTEGRITY_FAILED_NO_BUNDLE' if self.integrity_failures else
                  'ALL_THREE_SMOKES_CHECKED' if all_pass else 'SMOKE_CHECKS_FAILED_OR_INCOMPLETE')
        result = {'status': status, 'stage': str(self.stage), 'input_manifest_sha256': EXPECTED_INPUTS_SHA256,
            'run_checks': self.run_checks, 'driver_check': self.driver_check, 'scope_check': self.scope_check,
            'algorithm_ast_check': getattr(self, 'algorithm_ast_check', {'status': 'NOT_COMPLETED'}),
            'files': self.files, 'warnings': self.warnings, 'integrity_failures': self.integrity_failures,
            'scope': 'Private smoke evidence only: random initialized models, loss/gradient checks, masked/unpadded parity, limited last-bucket invariance checks and 32 validation samples. This collector does not train, run inference, execute submitted source, or open interaction CSV contents/weights. Source/data/ref hashes bind server records; tensor/logit results are not independently rerun here. No formal model quality, test accuracy, significance or equivalence conclusion.'}
        (self.out / 'smoke_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
        with (self.out / 'COLLECTED_SHA256.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['origin', 'path', 'archive_path', 'status', 'bytes', 'sha256', 'error'])
            writer.writeheader()
            writer.writerows(self.files)
        lines = ['# PRIVATE Amazon LBSP smoke evidence', '', 'Status: **' + status + '**', '', result['scope'], '',
                 'All three PASS results additionally require one matching driver record with three exit codes zero and smoke-only commands.',
                 'Parity relies on the original allclose result with atol=1e-4 and rtol=1e-5; max_abs_diff alone cannot determine relative-tolerance allclose.',
                 'No trained model is produced. Smoke validation values are diagnostics, never formal effectiveness results.', '',
                 '| Dataset | Check | Items | Positions | Loss | Finite gradients |', '|---|---|---:|---:|---:|---:|']
        for tag in TAGS:
            run = self.run_checks.get(tag, {'status': 'NOT_CHECKED'})
            counts = run.get('counts', {})
            lines.append('| %s | %s | %s | %s | %s | %s |' % (TAGS[tag], run['status'], counts.get('num_items', ''),
                counts.get('training_positions', ''), run.get('smoke_loss', ''), run.get('finite_gradient_count', '')))
        lines += ['', '## Review messages', '']
        messages = self.integrity_failures + self.warnings + self.scope_check.get('errors', []) + self.driver_check.get('errors', [])
        messages += [tag + ': ' + error for tag, run in self.run_checks.items() for error in run['errors']]
        lines += ['- ' + message for message in messages] or ['- No discrepancy found in the supplied smoke records.']
        lines += ['', 'This private ZIP contains source and prior Control reference predictions. Do not publish it directly.', '']
        (self.out / 'SUMMARY.md').write_text('\n'.join(lines), 'utf-8')
        bundle = None
        if not self.integrity_failures:
            bundle = self.out / 'lbsp_amazon_smoke_reports.zip'
            paths = sorted(p for p in self.out.rglob('*') if p.is_file())
            with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for path in paths:
                    if path.is_symlink():
                        raise ValueError('Unexpected output symbolic link')
                    archive.write(path, path.relative_to(self.out).as_posix())
            digest = hashlib.sha256()
            with bundle.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            bundle.with_name(bundle.name + '.sha256.txt').write_text(digest.hexdigest() + '  ' + bundle.name + '\n', 'ascii')
        return {'report_dir': str(self.out), 'zip_path': str(bundle) if bundle else None,
                'status': status, 'exit_code': 2 if self.integrity_failures else 0 if all_pass else 1}


def collect(stage):
    collector = Collector(stage)
    try:
        collector.verify_sources()
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        collector.integrity_failures.append(type(exc).__name__ + ': ' + str(exc)[:320])
    if not collector.integrity_failures:
        collector.verify_scope()
        observed = sorted(safe(collector.stage, 'verification').iterdir())
        eligible = [p for p in observed if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in ('.json', '.log')]
        if len(eligible) > 64:
            collector.integrity_failures.append('Unexpected number of verification logs/JSON files')
        else:
            for path in eligible:
                collector.copy(collector.stage, 'verification/' + path.name, 'verification/' + path.name)
            for tag in TAGS:
                collector.run(tag)
            collector.driver()
    return collector.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.stage.expanduser())
    print('SUMMARY=' + str(Path(result['report_dir']) / 'SUMMARY.md'))
    print('LBSP_SMOKE_REPORT_ZIP=' + (result['zip_path'] or 'NOT_CREATED'))
    print('STATUS=' + result['status'])
    return result['exit_code']


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        print('COLLECTION_FAILED=' + type(exc).__name__ + ': ' + str(exc)[:320], file=sys.stderr)
        sys.exit(2)


