#!/usr/bin/env python3
"""Prepare controlled within-visible-bucket training augmentation; stdlib only.
No research execution during preparation. Frozen inputs and existing results
are read only; generated smoke/train programs run only on explicit invocation.
"""
import argparse
import ast
from collections import Counter
import ctypes
import datetime
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import uuid

INPUTS_SHA256='3788d6ce525966c74663a7e2e61b4bc20c9eaac5d2e6c7a5a8e33fc46f4dfce3'
VERSION='x3_shuffle_v1'
MANIFEST_PATH='manifests/x3_shuffle_manifest.json'
SEEDS=(42,2026,3407)
DATASETS={'ml1m':'MovieLens-1M','beauty':'Amazon_Beauty','sports':'Amazon_Sports','toys':'Amazon_Toys'}
SAVED_FILES={'config.json','metrics.json','training_log.csv','test_per_sample_predictions.csv','item_id_map.csv'}


def plain(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected absolute path without ..')
    for part in (path,) + tuple(path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Symbolic link/reparse point rejected: ' + str(part))
    return path.resolve(strict=False)


def checked_file(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or PurePosixPath(relative).is_absolute() or any(p in ('', '.', '..') for p in relative.split('/'))):
        raise ValueError('Unsafe relative path')
    root = plain(root)
    path = plain(root.joinpath(*relative.split('/')))
    path.relative_to(root)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Expected regular file: ' + relative)
    return path


def file_signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def checked_sha256(root, relative, expected):
    path = checked_file(root, relative)
    before = path.stat()
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    with os.fdopen(os.open(str(path), flags), 'rb') as handle:
        if file_signature(before) != file_signature(os.fstat(handle.fileno())):
            raise RuntimeError('Input changed before hashing: ' + relative)
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if (file_signature(before) != file_signature(after)
            or file_signature(before) != file_signature(checked_file(root, relative).stat())):
        raise RuntimeError('Input changed during hashing: ' + relative)
    if expected is not None and digest.hexdigest() != expected:
        raise RuntimeError('SHA256 mismatch: ' + relative)
    return digest.hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Unexpected template structure: ' + old[:70])
    return source.replace(old, new, 1)


def replace_assignment(source, name, expression):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(nodes) != 1:
        raise RuntimeError('Expected one assignment: ' + name)
    return replace_once(source, ast.get_source_segment(source, nodes[0]), name + ' = ' + expression)


def definition(source, name):
    matches = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    if len(matches) != 1:
        raise RuntimeError('Expected one definition: ' + name)
    return matches[0]


def write_new(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)


def event_priority(seed, dataset, user, event_key, epoch):
    payload = json.dumps([int(seed), str(dataset), str(user), int(event_key), int(epoch)],
                         separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    value = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(value[:8], 'big'), int.from_bytes(value[8:], 'big')


def visible_order(items, buckets, event_keys, priorities):
    n = len(items)
    if not 1 <= n <= 50 or len(buckets) != n or len(event_keys) != n or len(priorities) != n:
        raise ValueError('Visible history arrays must have the same length in 1..50')
    order, start = [], 0
    while start < n:
        end = start + 1
        while end < n and buckets[end] == buckets[start]:
            end += 1
        if end == start + 1:
            order.append(start)
        else:
            order.extend(sorted(range(start, end), key=lambda i: (priorities[i], int(event_keys[i]))))
        start = end
    return order


def verify_visible_invariants(items, buckets, event_keys, order, future_keys=()):
    n = len(items)
    if sorted(order) != list(range(n)) or len(set(event_keys)) != n or set(event_keys) & set(future_keys):
        raise ValueError('Visible events changed or target/future event leaked')
    if [buckets[i] for i in order] != list(buckets):
        raise ValueError('Visible time-bucket positions changed')
    start, eligible, segments = 0, 0, 0
    while start < n:
        end = start + 1
        while end < n and buckets[end] == buckets[start]:
            end += 1
        before = Counter((int(event_keys[i]), int(items[i])) for i in range(start, end))
        after = Counter((int(event_keys[order[i]]), int(items[order[i]])) for i in range(start, end))
        if before != after:
            raise ValueError('Per-bucket event/item multiset changed')
        eligible += int(end - start >= 2 and len({int(items[i]) for i in range(start, end)}) >= 2)
        segments += 1; start = end
    return {'visible_events': n, 'bucket_segments': segments, 'eligible_bucket_segments': eligible,
            'changed_visible_items': int([int(items[i]) for i in order] != [int(i) for i in items]),
            'all_visible_bucket_event_multisets_preserved': True, 'target_event_future_excluded': True}


class SharedPriorityState:
    """Epoch priorities shared with the original persistent Linux workers.

    Allocating/filling these standard-library buffers never uses Python's
    global random state, NumPy RNG, Torch RNG, or the sampler generator.
    """
    def __init__(self, dataset, seed, metadata, context=None):
        self.dataset, self.seed = str(dataset), int(seed)
        self.metadata, self.offsets = {}, {}
        total = 0
        for user in sorted(metadata):
            entry = metadata[user]
            items, buckets, keys = tuple(entry['items']), tuple(entry['buckets']), tuple(int(k) for k in entry['event_keys'])
            if len(items) < 2 or len(items) != len(buckets) or len(items) != len(keys) or len(set(keys)) != len(keys):
                raise ValueError('Each retained training user needs unique original event keys and aligned arrays')
            self.metadata[str(user)] = {'items': items, 'buckets': buckets, 'event_keys': keys}
            self.offsets[str(user)] = total; total += len(items)
        if not total:
            raise ValueError('No training events available for augmentation')
        ctx = context if context is not None else mp.get_context()
        self.high = ctx.RawArray(ctypes.c_uint64, total)
        self.low = ctx.RawArray(ctypes.c_uint64, total)
        self.epoch = ctx.Value('q', -1)
        self.total_events = total

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError('Augmentation epoch must be a nonnegative integer')
        with self.epoch.get_lock():
            for user, entry in self.metadata.items():
                offset = self.offsets[user]
                for i, event_key in enumerate(entry['event_keys']):
                    high, low = event_priority(self.seed, self.dataset, user, event_key, epoch)
                    self.high[offset + i] = high; self.low[offset + i] = low
            self.epoch.value = epoch

    def window(self, user, start, stop):
        user = str(user)
        if user not in self.metadata:
            raise ValueError('Training user is absent from augmentation metadata')
        entry = self.metadata[user]
        if not 0 <= start < stop < len(entry['items']) or stop - start > 50:
            raise ValueError('Original cutoff/visible truncation window is invalid')
        with self.epoch.get_lock():
            if self.epoch.value < 0:
                raise ValueError('Set the shared augmentation epoch before iterating the loader')
            offset = self.offsets[user]
            priorities = [(int(self.high[offset + i]), int(self.low[offset + i])) for i in range(start, stop)]
        items, buckets, keys = (entry[k][start:stop] for k in ('items', 'buckets', 'event_keys'))
        return items, buckets, keys, visible_order(items, buckets, keys, priorities)

    def visible_items(self, user, start, stop, reference_items):
        items, buckets, keys, order = self.window(user, start, stop)
        if tuple(reference_items) != items:
            raise ValueError('Dataset visible prefix differs from the original frozen cutoff')
        return [items[i] for i in order]

    def proof(self, user, start, stop):
        items, buckets, keys, order = self.window(user, start, stop)
        entry = self.metadata[str(user)]
        result = verify_visible_invariants(items, buckets, keys, order, entry['event_keys'][stop:])
        target = entry['items'][stop]
        if sum(item == target for item in items) != sum(items[i] == target for i in order):
            raise ValueError('A legitimate repeated target-item occurrence was changed')
        result.update(target_and_cutoff_fixed=True, original_visible_window_and_length_preserved=True,
                      repeated_target_item_counts_preserved=True)
        return result


def frozen_model(source):
    node = definition(source, 'SASRecControl')
    result = '# Frozen SASRecControl definition only. No original top-level execution.\n' + ast.get_source_segment(source, node) + '\n'
    if ast.dump(definition(result, 'SASRecControl')) != ast.dump(node):
        raise RuntimeError('Frozen Control model AST changed')
    return result


def replace_span(source, start, end, text):
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError('Ambiguous protected source boundary')
    before, rest = source.split(start, 1)
    _, after = rest.split(end, 1)
    return before + text + end + after


def mapping_code():
    return '''mapping_df = pd.read_csv(_MAP_PATH, dtype={"raw_item_id": str})
if not {"raw_item_id", "item_idx"}.issubset(mapping_df.columns):
    raise RuntimeError("Frozen Control mapping columns missing")
if (not mapping_df["raw_item_id"].is_unique or not mapping_df["item_idx"].is_unique
        or mapping_df["item_idx"].astype(int).tolist() != list(range(1, len(mapping_df) + 1))):
    raise RuntimeError("Frozen Control item mapping is not an ordered bijection")
raw_items = mapping_df["raw_item_id"].tolist()
item_map = dict(zip(raw_items, mapping_df["item_idx"].astype(int)))
idx_to_raw_item = {int(index): str(raw) for raw, index in item_map.items()}
idx_to_raw = idx_to_raw_item
NUM_ITEMS = len(item_map)
if NUM_ITEMS != _PLAN["expected_items"]:
    raise RuntimeError("Frozen catalog count differs")

'''


def metadata_code(tag):
    bucket = 'BUCKET_COLUMN' if tag in ('ml1m', 'beauty') else '"time_bucket"'
    return '''_AUGMENT_METADATA = {}
_AUGMENT_USERS = []
for _user, _group in train_df.groupby("user_id", sort=False):
    if len(_group) >= 2:
        _user = str(_user)
        _AUGMENT_USERS.append(_user)
        _AUGMENT_METADATA[_user] = {
            "items": tuple(item_map[str(item)] for item in _group["item_id"]),
            "buckets": tuple(_group[''' + bucket + ''']),
            "event_keys": tuple(int(key) for key in _group[_PLAN["event_key_column"]]),
        }
if len(train_df) != _PLAN["expected_training_rows"]:
    raise RuntimeError("Original training row count differs")
_AUGMENT = SharedPriorityState(DATASET, SEED, _AUGMENT_METADATA)

'''


def model_extraction():
    return '''_model_tree = ast.parse((_X3_STAGE / "scripts/model_definition.py").read_text(encoding="utf-8"))
_model_nodes = [node for node in _model_tree.body if isinstance(node, ast.ClassDef) and node.name == "SASRecControl"]
if len(_model_nodes) != 1:
    raise RuntimeError("Frozen SASRecControl class missing")
_model_namespace = dict(globals())
exec(compile(ast.Module(body=_model_nodes, type_ignores=[]), "model_definition.py", "exec"), _model_namespace)
SASRecControl = _model_namespace["SASRecControl"]'''


def _augmentation_smoke_proof():
    """Server-only small loader probe; separate generator protects global RNG."""
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    cuda_before = [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    n = len(train_dataset)
    indices = sorted({i * (n - 1) // max(1, min(512, n) - 1) for i in range(min(512, n))})
    proofs = []
    for epoch in (1, 2):
        _AUGMENT.set_epoch(epoch)
        for index in indices:
            if _TAG in ('ml1m', 'beauty'):
                user, end = train_dataset.samples[index]; stop = end - 1
            else:
                user_index, stop = train_dataset.positions[index]; user = _AUGMENT_USERS[user_index]
            proofs.append(_AUGMENT.proof(user, max(0, stop - MAX_LEN), stop))
    workers = 4 if _TAG in ('ml1m', 'beauty') else 0
    probe_indices = indices[:min(32, len(indices))]
    generator = torch.Generator()
    generator.manual_seed(SEED + 735781)
    probe = DataLoader(torch.utils.data.Subset(train_dataset, probe_indices), batch_size=8, shuffle=False,
        num_workers=workers, persistent_workers=bool(workers), generator=generator, pin_memory=False)
    for epoch in (1, 2):
        _AUGMENT.set_epoch(epoch)
        expected = [train_dataset[index] for index in probe_indices]
        offset = 0
        for batch in probe:
            size = len(batch[0])
            for component, tensor in enumerate(batch):
                wanted = torch.stack([row[component] for row in expected[offset:offset + size]])
                if not torch.equal(tensor, wanted):
                    raise RuntimeError('Persistent-worker epoch/window output differs from main process')
            offset += size
        if offset != len(probe_indices):
            raise RuntimeError('Persistent-worker probe sample coverage differs')
    del probe
    _AUGMENT.set_epoch(0)
    numpy_after = np.random.get_state()
    if (random.getstate() != python_before or numpy_before[0] != numpy_after[0]
            or not np.array_equal(numpy_before[1], numpy_after[1]) or numpy_before[2:] != numpy_after[2:]
            or not torch.equal(torch_before, torch.get_rng_state())
            or any(not torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all() if cuda_before else []))):
        raise RuntimeError('Augmentation/probe consumed a global RNG stream')
    return {'checked_visible_windows': len(proofs), 'epoch_checks': [1, 2], 'worker_epoch_checks': [1, 2],
        'probe_samples_per_epoch': len(probe_indices), 'probe_workers': workers,
        'persistent_worker_outputs_match_main': True, 'global_rng_unchanged_by_augmentation_and_probe': True,
        'eligible_bucket_segments': sum(row['eligible_bucket_segments'] for row in proofs),
        'changed_visible_windows': sum(row['changed_visible_items'] for row in proofs),
        'all_visible_bucket_event_multisets_preserved': True, 'target_and_cutoff_fixed': True,
        'target_event_future_excluded': True, 'original_visible_window_and_length_preserved': True,
        'repeated_target_item_counts_preserved': True, 'cross_window_event_priorities_shared': True}


def _save_smoke_report(loss, gradients, all_finite, proof):
    if not math.isfinite(loss) or loss <= 0 or gradients <= 0 or not all_finite:
        raise RuntimeError('Smoke loss/backward gradients failed')
    _verify_originals()
    if OUTPUT_DIR.exists():
        raise RuntimeError('Smoke must not create a formal output directory')
    report = dict(_X3_PROVENANCE)
    report.update(tag=_TAG, dataset=DATASET, seed=SEED, status='SMOKE_PASS', mode='smoke',
        counts={'num_items': NUM_ITEMS, 'training_rows': len(train_df), 'training_positions': len(train_dataset),
                'trainable_users': len(_AUGMENT.metadata), 'validation_samples': len(val_df), 'test_samples': len(test_df)},
        smoke_loss=loss, finite_gradient_count=gradients, all_gradients_finite=bool(all_finite), invariants=proof,
        optimizer_steps=0, formal_training_started=False, weights_read=False, weights_saved=False,
        test_inference_performed=False, formal_output_created=False, original_hashes_verified_after=True)
    path = _X3_STAGE / 'verification' / ('smoke_' + _TAG + '_seed' + str(SEED) + '.json')
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print('X3_SHUFFLE_SMOKE_PASS=' + str(path), flush=True)


def _canonical_test_mask_proof(reference_rows, prediction_rows, item_map):
    """Post-save audit only; preserve the protected ranking implementation."""
    if len(reference_rows) != len(prediction_rows) or not reference_rows:
        raise RuntimeError('Saved test predictions do not cover the canonical test panel')
    seen_samples = set()
    violations = unknown = identity = raw_index_mismatches = 0
    for reference, prediction in zip(reference_rows, prediction_rows):
        sample = str(reference['sample_id'])
        if sample in seen_samples:
            raise RuntimeError('Duplicate canonical test sample identity')
        seen_samples.add(sample)
        identity += int(any(str(reference[key]) != str(prediction[key])
                            for key in ('sample_id', 'user_id', 'target_item')))
        target = str(reference['target_item'])
        unknown += int(target not in item_map)
        history = {str(item) for item in json.loads(reference['sequence_items'])}
        masked = history - {target}
        raw_top = [str(item) for item in json.loads(prediction['top10_raw_item_id'])]
        indices = json.loads(prediction['top10_item_idx'])
        if len(raw_top) != 10 or len(indices) != 10 or len(set(raw_top)) != 10:
            raise RuntimeError('Saved Top10 list is malformed')
        for raw, index in zip(raw_top, indices):
            if isinstance(index, bool) or not isinstance(index, int):
                raise RuntimeError('Saved Top10 index must be an integer')
            raw_index_mismatches += int(raw not in item_map or item_map.get(raw) != index)
            violations += int(raw in masked)
    if violations or unknown or identity or raw_index_mismatches:
        raise RuntimeError('Canonical test mask/identity/map audit failed: ' + json.dumps({
            'seen_item_violations': violations, 'target_unknown_count': unknown,
            'panel_identity_mismatches': identity, 'top10_raw_index_mismatches': raw_index_mismatches}))
    return {'canonical_evaluation_mask_verified': True, 'test_samples_verified': len(reference_rows),
        'test_top10_seen_item_violations': violations, 'test_target_unknown_count': unknown,
        'test_panel_identity_mismatches': identity, 'test_top10_raw_index_mismatches': raw_index_mismatches,
        'test_mask_rule': 'unique canonical history items except target; full frozen Control catalog'}


def ml_smoke_code(optimizer_text):
    return '''if args.mode == "smoke":
    _SMOKE_PROOF = _augmentation_smoke_proof()
    _batch = torch.utils.data.default_collate([train_dataset[i] for i in range(min(8, len(train_dataset)))])
    _seq, _groups, _tokens, _pool, _last, _targets = [value.to(device) for value in _batch]
    model.train(); model.zero_grad(set_to_none=True)
    _loss = train_one_batch(model, _seq, _groups, _tokens, _targets)
    if not torch.isfinite(_loss).item() or not _loss.requires_grad:
        raise RuntimeError("Smoke loss is nonfinite or has no gradient")
    _loss.backward()
    _all_finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    _nonzero = sum(p.grad is not None and torch.isfinite(p.grad).all().item() and p.grad.abs().sum().item() > 0 for p in model.parameters())
    model.zero_grad(set_to_none=True)
    _save_smoke_report(float(_loss.item()), _nonzero, _all_finite, _SMOKE_PROOF)
    raise SystemExit(0)

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
(OUTPUT_DIR / "item_id_map.csv").write_bytes(_MAP_PATH.read_bytes())
''' + optimizer_text + '\n\n'


def runtime_prelude(tag, inputs, root, stage, model_hash):
    helpers = (plain, checked_file, file_signature, checked_sha256, event_priority, visible_order,
        verify_visible_invariants, SharedPriorityState, _augmentation_smoke_proof, _save_smoke_report,
        _canonical_test_mask_proof)
    text = '''#!/usr/bin/env python3
# Controlled visible-history augmentation; no old training top-level imported.
import argparse
import ast
from collections import Counter
import ctypes
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import stat
parser=argparse.ArgumentParser(description='Explicit X3 visible-bucket smoke or training')
parser.add_argument('--mode',choices=['smoke','train'],required=True)
parser.add_argument('--seed',type=int,choices=[42,2026,3407],required=True)
args=parser.parse_args()
'''
    text += '\n\n'.join(inspect.getsource(helper) for helper in helpers) + '\n'
    text += '_X3_ROOT=plain(Path(' + repr(str(root)) + '))\n_X3_STAGE=plain(Path(' + repr(str(stage)) + '))\n'
    text += '_TAG=' + repr(tag) + '\n_INPUTS=' + repr(inputs) + '\n_INPUTS_SHA256=' + repr(INPUTS_SHA256) + '\n'
    text += '_MODEL_SHA256=' + repr(model_hash) + '\n'
    text += '''_PLAN=_INPUTS['datasets'][_TAG]
_MANIFEST_PATH='manifests/x3_shuffle_manifest.json'
_MANIFEST_BYTES=checked_file(_X3_STAGE,_MANIFEST_PATH).read_bytes()
_MANIFEST_SHA256=hashlib.sha256(_MANIFEST_BYTES).hexdigest()
_MANIFEST=json.loads(_MANIFEST_BYTES.decode('utf-8-sig'))
if (_MANIFEST['inputs']!=_INPUTS or _MANIFEST['inputs_sha256']!=_INPUTS_SHA256
        or _MANIFEST['root']!=str(_X3_ROOT) or _MANIFEST['stage']!=str(_X3_STAGE)):
    raise RuntimeError('Prepared X3 identity differs')
_TRAINER_RELATIVE='scripts/train_'+_TAG+'_shuffle.py'
_TRAINER_SHA256=_MANIFEST['generated_sha256'][_TRAINER_RELATIVE]
_DATA_SHA256={path:_INPUTS['data_sha256'][path] for path in
    (_PLAN['train_path'],_PLAN['validation_path'],_PLAN['test_path'])}
_REFERENCE_SHA256=_PLAN['control_reference']['files_sha256']
_MAP_PATH=checked_file(_X3_ROOT,_PLAN['control_reference']['model_dir']+'/item_id_map.csv')
def _verify_originals():
    for relative,expected in dict(_INPUTS['sources_sha256'],**_DATA_SHA256).items():
        checked_sha256(_X3_ROOT,relative,expected)
    for name,expected in _REFERENCE_SHA256.items():
        checked_sha256(_X3_ROOT,_PLAN['control_reference']['model_dir']+'/'+name,expected)
    checked_sha256(_X3_STAGE,'scripts/model_definition.py',_MODEL_SHA256)
    checked_sha256(_X3_STAGE,_TRAINER_RELATIVE,_TRAINER_SHA256)
    checked_sha256(_X3_STAGE,_MANIFEST_PATH,_MANIFEST_SHA256)
_verify_originals()
if mp.get_context().get_start_method()!='fork':
    raise RuntimeError('These protected top-level Control trainers require the original Linux fork worker context')
_X3_PROVENANCE={'version':'x3_shuffle_v1','input_manifest_sha256':_INPUTS_SHA256,
    'preparation_manifest_sha256':_MANIFEST_SHA256,'trainer_sha256':_TRAINER_SHA256,
    'frozen_model_sha256':_MODEL_SHA256,'sources_sha256':_INPUTS['sources_sha256'],
    'data_sha256':_DATA_SHA256,'control_reference':_PLAN['control_reference'],
    'augmentation':{'visible_history_first':True,'target_and_cutoff_fixed':True,
        'all_visible_time_bucket_segments':True,'shared_event_epoch_priorities':True,
        'global_rng_consumed':False,'event_key_column':_PLAN['event_key_column'],
        'priority_algorithm':'BLAKE2b-128(json([seed,dataset,user,eventkey,epoch]))',
        'worker_epoch_shared':True,'epoch_setup_included_in_epoch_seconds':True,
        'legitimate_repeated_target_item_ids_retained':True},
    'original_sampler_loader_flow_preserved':True,'canonical_validation_test':True}
print('X3 INTEGRITY PASS: frozen sources/data/Control metadata/model; no old checkpoint accessed',flush=True)
'''
    return text


def generated_trainer(original, tag, inputs, root, stage, model_hash, model_source):
    source = original
    if tag in ('sports', 'toys'):
        marker = '\nimport argparse\nimport ast\n'
        if source.count(marker) != 1:
            raise RuntimeError('Corrected Control body boundary changed')
        source = 'import argparse\nimport ast\n' + source.split(marker, 1)[1]
    original_body = source
    protected_names = ('load_r1_eval', 'train_one_batch', 'calculate_metrics', 'evaluate', 'build_control_group_positions') if tag in ('ml1m', 'beauty') else ('load_strict_samples', 'loss_for_batch', 'evaluate')
    protected = {name: ast.dump(definition(source, name)) for name in protected_names}
    dataset_name = 'ControlTrainDataset' if tag in ('ml1m', 'beauty') else 'AllPositionsDataset'
    dataset_before = definition(source, dataset_name)
    loop_before = next(n for n in ast.parse(source).body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    plan = inputs['datasets'][tag]
    replacements = {'ROOT': '_X3_ROOT', 'SEED': 'args.seed', 'TRAIN_PATH': 'ROOT / _PLAN["train_path"]',
        'VAL_PATH': 'ROOT / _PLAN["validation_path"]', 'TEST_PATH': 'ROOT / _PLAN["test_path"]',
        'ITEM_SOURCE_PATH': '_MAP_PATH', 'OUTPUT_DIR': '_X3_STAGE / "outputs" / f"' + tag + '_shuffle_seed{SEED}"'}
    if tag in ('sports', 'toys'):
        replacements.update(BASE='_X3_STAGE', MODEL_SCRIPT='_X3_STAGE / "scripts/model_definition.py"')
    for name, expression in replacements.items():
        source = replace_assignment(source, name, expression)
    # Every original output mkdir/map write is removed. The sole replacement
    # occurs AFTER the smoke exit and retains original map bytes exactly.
    for node in reversed(ast.parse(source).body):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if isinstance(call.func, ast.Attribute) and ((call.func.attr == 'mkdir' and ast.unparse(call.func.value) == 'OUTPUT_DIR')
                or (call.func.attr == 'to_csv' and 'item_id_map.csv' in ast.get_source_segment(source, node))):
            source = replace_once(source, ast.get_source_segment(source, node), '')
    source = replace_span(source, 'item_source = pd.read_csv(', 'train_df = pd.read_csv(', mapping_code())
    if tag in ('ml1m', 'beauty'):
        if ast.dump(definition(source, 'SASRecControl')) != ast.dump(definition(model_source, 'SASRecControl')):
            raise RuntimeError('Dataset Control class differs from frozen shared model')
        source = replace_once(source, ast.get_source_segment(source, definition(source, 'SASRecControl')), model_extraction())
    # Preserve the original cutoff and truncation statements, then permute only
    # the already visible events. No full user sequence is shuffled.
    dataset = definition(source, dataset_name)
    getitem = next(n for n in dataset.body if isinstance(n, ast.FunctionDef) and n.name == '__getitem__')
    history_name = 'history_items' if tag in ('ml1m', 'beauty') else 'history'
    anchor_name = 'history_buckets' if tag in ('ml1m', 'beauty') else 'history'
    anchor = next(n for n in getitem.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == anchor_name for t in n.targets))
    old = ast.get_source_segment(source, anchor)
    args = 'user_id, history_start, target_idx, history_items' if tag in ('ml1m', 'beauty') else '_AUGMENT_USERS[u], max(0, p - MAX_LEN), p, history'
    insertion = history_name + ' = np.asarray(_AUGMENT.visible_items(' + args + '), dtype=np.int64)'
    source = replace_once(source, old, old + '\n        ' + insertion)
    dataset_after = definition(source, dataset_name)
    getitem_after = next(n for n in dataset_after.body if isinstance(n, ast.FunctionDef) and n.name == '__getitem__')
    matching = [n for n in getitem_after.body if isinstance(n, ast.Assign) and '_AUGMENT.visible_items' in ast.unparse(n)]
    if len(matching) != 1:
        raise RuntimeError('Expected exactly one visible-history augmentation insertion')
    getitem_after.body.remove(matching[0])
    if ast.dump(dataset_after) != ast.dump(dataset_before):
        raise RuntimeError('Original Dataset cutoff/target/padding AST changed beyond augmentation')
    source = replace_once(source, 'train_dataset = ', metadata_code(tag) + 'train_dataset = ')
    train_assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'train_dataset' for t in n.targets))
    source = replace_once(source, ast.get_source_segment(source, train_assignment), ast.get_source_segment(source, train_assignment) + '''
if len(train_dataset) != _PLAN["expected_training_positions"]:
    raise RuntimeError("Original legal-position coverage differs from the sealed Control protocol")
if _TAG in ('sports', 'toys'):
    if len(user_sequences) != len(_AUGMENT_USERS) or any(tuple(seq) != _AUGMENT.metadata[user]["items"] for seq,user in zip(user_sequences,_AUGMENT_USERS)):
        raise RuntimeError("Parallel Sports/Toys user/event metadata lost original sequence alignment")
''')
    source = replace_once(source, 'model = SASRecControl(', '''if len(val_df) != _PLAN["expected_validation"] or len(test_df) != _PLAN["expected_test"]:
    raise RuntimeError("Frozen canonical validation/test panel count differs")
model = SASRecControl(''')
    # Strong no-overwrite guard applies before model construction in both modes.
    output_assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'OUTPUT_DIR' for t in n.targets))
    source = replace_once(source, ast.get_source_segment(source, output_assignment), ast.get_source_segment(source, output_assignment) + '''
if OUTPUT_DIR.exists():
    raise RuntimeError("X3 formal output already exists; refusing to overwrite")
''')
    if tag in ('ml1m', 'beauty'):
        optimizer = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'optimizer' for t in n.targets))
        optimizer_text = ast.get_source_segment(source, optimizer)
        source = replace_once(source, optimizer_text, '')
        evaluate = definition(source, 'evaluate')
        old = ast.get_source_segment(source, evaluate)
        # Preserve the decorator, which is outside get_source_segment's first line.
        source = replace_once(source, old, old + '\n\n' + ml_smoke_code(optimizer_text))
        tries = [n for n in ast.parse(source).body if isinstance(n, ast.Try) and 'best_checkpoint = torch.load' in ast.get_source_segment(source, n)]
        if len(tries) != 1:
            raise RuntimeError('Original best-checkpoint load boundary differs')
        source = replace_once(source, ast.get_source_segment(source, tries[0]),
            'best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)')
    else:
        # Keep the original pre-training gradient and val32 smoke flow for ST:
        # removing it would shift the original dropout RNG stream before epoch1.
        source = replace_once(source, 'model.train()\n\nsmoke_loader', '''if args.mode == "smoke":
    _SMOKE_PROOF = _augmentation_smoke_proof()
else:
    _AUGMENT.set_epoch(0)
model.train()

smoke_loader''')
        source = replace_once(source, 'smoke_loss.backward()', '''smoke_loss.backward()
_SMOKE_ALL_GRADIENTS_FINITE = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
if not _SMOKE_ALL_GRADIENTS_FINITE:
    raise RuntimeError("Smoke backward contains a nonfinite gradient")''')
        smoke_exit = next(n for n in ast.parse(source).body if isinstance(n, ast.If) and ast.unparse(n.test) == "args.mode == 'smoke'" and any(isinstance(x, ast.Raise) for x in n.body))
        source = replace_once(source, ast.get_source_segment(source, smoke_exit), '''if args.mode == "smoke":
    _save_smoke_report(float(smoke_loss.item()), gradient_count, _SMOKE_ALL_GRADIENTS_FINITE, _SMOKE_PROOF)
    raise SystemExit(0)
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
(OUTPUT_DIR / "item_id_map.csv").write_bytes(_MAP_PATH.read_bytes())''')
        config = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'config' for t in n.targets))
        keep = [(k,v) for k,v in zip(config.value.keys,config.value.values) if not (isinstance(k,ast.Constant) and k.value=='orderfix')]
        cleaned = ast.Dict(keys=[k for k,v in keep], values=[v for k,v in keep])
        source = replace_assignment(source, 'config', ast.unparse(cleaned))
    config = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'config' for t in n.targets))
    source = replace_once(source, ast.get_source_segment(source, config), ast.get_source_segment(source, config) + '\nconfig["x3_shuffle"] = dict(_X3_PROVENANCE)')
    # Count augmentation setup inside the original epoch timer; all other loop
    # AST statements (loss, optimizer, sampler, validation, stopping) are exact.
    loop = next(n for n in ast.parse(source).body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    timer = next(n for n in loop.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id == ('epoch_begin' if tag in ('ml1m','beauty') else 'start_time') for t in n.targets))
    timer_text = ast.get_source_segment(source, timer)
    source = replace_once(source, timer_text, timer_text + '\n    _AUGMENT.set_epoch(epoch)')
    loop_after = next(n for n in ast.parse(source).body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    refresh = [n for n in loop_after.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='_AUGMENT.set_epoch']
    if len(refresh)!=1:
        raise RuntimeError('Exactly one epoch priority refresh is required')
    loop_after.body.remove(refresh[0])
    if ast.dump(loop_after)!=ast.dump(loop_before):
        raise RuntimeError('Protected original epoch loop changed beyond timed priority refresh')
    for name, expected in protected.items():
        if ast.dump(definition(source,name))!=expected:
            raise RuntimeError('Protected Control loss/eval function changed: '+name)
    result = runtime_prelude(tag,inputs,root,stage,model_hash) + '\n' + source + '\n' + completion_code(tag)
    ast.parse(result,feature_version=(3,10))
    if any(name in result for name in ('_ORDERFIX_ROOT','_ORDERFIX_STAGE','_INPUT_SHA256','_FROZEN_MODEL_SHA256')):
        raise RuntimeError('Legacy unsealed orderfix runtime dependencies survived')
    return result


def completion_code(tag):
    epoch = 'int(best_checkpoint["epoch"])' if tag in ('ml1m','beauty') else 'int(best_epoch)'
    return '''_verify_originals()
with (OUTPUT_DIR/'test_per_sample_predictions.csv').open('r',encoding='utf-8-sig',newline='') as _handle:
    _SAVED_TEST_ROWS=list(csv.DictReader(_handle))
_MASK_PROOF=_canonical_test_mask_proof(test_df.to_dict('records'),_SAVED_TEST_ROWS,item_map)
_OUTPUT_METADATA_SHA256={name:checked_sha256(OUTPUT_DIR,name,None) for name in
    ('config.json','metrics.json','training_log.csv','test_per_sample_predictions.csv','item_id_map.csv')}
_COMPLETED=dict(_X3_PROVENANCE)
_COMPLETED.update(tag=_TAG,dataset=DATASET,seed=SEED,mode='train',status='COMPLETED',output_dir=str(OUTPUT_DIR),
    best_epoch=''' + epoch + ''',checkpoint='matched_control_sasrec_best.pt',
    output_metadata_sha256=_OUTPUT_METADATA_SHA256,all_original_hashes_verified=True,originals_unchanged=True,
    counts={'num_items':NUM_ITEMS,'training_rows':len(train_df),'training_positions':len(train_dataset),
        'trainable_users':len(_AUGMENT.metadata),'validation_samples':len(val_df),'test_samples':len(test_df)})
_COMPLETED.update(_MASK_PROOF)
_COMPLETED_PATH=_X3_STAGE/'verification'/('completed_'+_TAG+'_seed'+str(SEED)+'.json')
with _COMPLETED_PATH.open('x',encoding='utf-8') as _handle:
    _handle.write(json.dumps(_COMPLETED,indent=2,ensure_ascii=False,allow_nan=False)+'\\n')
print('X3_SHUFFLE_TRAIN_COMPLETED='+str(_COMPLETED_PATH),flush=True)
'''


def prepare(root, inputs_path):
    root,inputs_path=plain(root),plain(inputs_path)
    raw=inputs_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=INPUTS_SHA256:
        raise RuntimeError('X3 input JSON differs from reviewed seal')
    inputs=json.loads(raw.decode('utf-8-sig'))
    if inputs['version']!=VERSION or inputs['training_seeds']!=list(SEEDS) or set(inputs['datasets'])!=set(DATASETS):
        raise ValueError('Unexpected X3 input schema')
    data,source_paths,refs=set(),set(),{}
    for tag,plan in inputs['datasets'].items():
        if plan['dataset']!=DATASETS[tag] or set(plan['control_reference']['files_sha256'])!=SAVED_FILES:
            raise ValueError('Unexpected frozen Control reference')
        data.update(plan[k] for k in ('train_path','validation_path','test_path'));source_paths.add(plan['trainer_source'])
        refs.update({plan['control_reference']['model_dir']+'/'+n:h for n,h in plan['control_reference']['files_sha256'].items()})
    if set(inputs['data_sha256'])!=data or len(data)!=12 or set(inputs['sources_sha256'])!=source_paths or len(source_paths)!=4 or inputs['model_source'] not in source_paths or len(refs)!=20:
        raise ValueError('Expected four sources, twelve inputs and twenty Control metadata files')
    verified={}
    for path,digest in {**inputs['sources_sha256'],**inputs['data_sha256'],**refs}.items():
        verified[path]=checked_sha256(root,path,digest)
    sources={}
    for path,digest in inputs['sources_sha256'].items():
        blob=checked_file(root,path).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=digest:raise RuntimeError('Source changed after hashing')
        sources[path]=blob.decode('utf-8-sig');ast.parse(sources[path])
    now=datetime.datetime.now(datetime.timezone.utc)
    stage=root/('kais_x3_shuffle_v1_'+now.strftime('%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:12])
    model=frozen_model(sources[inputs['model_source']]);model_hash=hashlib.sha256(model.encode()).hexdigest()
    generated={'scripts/model_definition.py':model}
    for tag,plan in inputs['datasets'].items():
        generated['scripts/train_'+tag+'_shuffle.py']=generated_trainer(sources[plan['trainer_source']],tag,inputs,root,stage,model_hash,sources[inputs['model_source']])
    reference_copies={}
    for tag,plan in inputs['datasets'].items():
        for name,digest in plan['control_reference']['files_sha256'].items():
            blob=checked_file(root,plan['control_reference']['model_dir']+'/'+name).read_bytes()
            if hashlib.sha256(blob).hexdigest()!=digest:raise RuntimeError('Control reference changed during preparation')
            reference_copies['references/'+tag+'_control_seed42/'+name]=blob
    hashes={name:hashlib.sha256(text.encode()).hexdigest() for name,text in generated.items()}
    manifest={'version':VERSION,'root':str(root),'stage':str(stage),'created_utc':now.isoformat(),
        'status':'PREPARED_ONLY_NO_RESEARCH_EXECUTED','inputs':inputs,'inputs_sha256':INPUTS_SHA256,
        'generated_sha256':hashes,'sources_sha256':inputs['sources_sha256'],'data_sha256':inputs['data_sha256'],
        'reference_files_sha256':{name:hashlib.sha256(blob).hexdigest() for name,blob in reference_copies.items()},
        'augmentation_protocol':inputs['augmentation_protocol'],'training_seeds':list(SEEDS),
        'preserved':'Frozen model/loss/eval/load functions; Dataset cutoff/target/truncation; original epoch loop except timed priority refresh; original sampler/loader flow.',
        'worker_transport':'Original Linux fork context; shared RawArray high/low words and synchronized epoch value for persistent workers.',
        'scope':inputs['scope'],'checkpoint_accessed':False}
    stage.mkdir(exist_ok=False)
    for name in ('scripts','outputs','verification','manifests','references'):(stage/name).mkdir(exist_ok=False)
    for name,text in generated.items():write_new(stage/name,text)
    for name,blob in reference_copies.items():
        path=stage/name;path.parent.mkdir(exist_ok=True)
        with path.open('xb') as handle:handle.write(blob)
    write_new(stage/MANIFEST_PATH,json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    write_new(stage/'verification/PREPARED.json',json.dumps({'status':'PREPARED_ONLY','hash_verified':verified,
        'research_code_executed':False,'checkpoint_read_or_statted':False},indent=2)+'\n')
    print('WORKDIR='+str(stage));print('STATUS=PREPARED_ONLY_NO_RESEARCH_EXECUTED')
    return stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,default=Path(__file__).with_name('x3_shuffle_inputs_20261025.json'))
    args=parser.parse_args();prepare(args.root,args.inputs)


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,KeyError,TypeError,SyntaxError) as exc:
        print('PREPARATION_FAILED='+type(exc).__name__+': '+str(exc),file=sys.stderr);sys.exit(2)
