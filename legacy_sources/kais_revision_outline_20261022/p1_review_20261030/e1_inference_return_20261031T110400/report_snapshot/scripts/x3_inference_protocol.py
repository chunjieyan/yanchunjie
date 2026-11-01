"""Pure X3 inference/ensemble definitions, with no file, NumPy or Torch I/O.

PCG64 permutations are made by the server's local NumPy generator. This module
only specifies its legacy seed and applies caller-supplied permutation indices.
Time buckets are optional metadata, never inferred when absent.
"""
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from numbers import Integral
import re
import statistics

VERSION = 'x3_inference_protocol_v1'
TRAINING_SEEDS = (42, 2026, 3407)
ENSEMBLE_GROUPS = {
    'g1': (101, 202, 303, 404, 505),
    'g2': (1101, 1202, 1303, 1404, 1505),
    'g3': (2101, 2202, 2303, 2404, 2505),
    'g4': (3101, 3202, 3303, 3404, 3505),
    'g5': (4101, 4202, 4303, 4404, 4505),
}
GROUP_NAMES = ('g1', 'g2', 'g3', 'g4', 'g5')
PERMUTATION_SEEDS = (101, 202, 303, 404, 505)
METRICS = ('ndcg10', 'hr10', 'hr20', 'recall50', 'mrr', 'mrr10', 'mean_rank')
MAX_LEN = 50
ENSEMBLE_RULES = {
    'members': 'Five legal last-bucket permutations per group; g1 reuses and verifies the five sealed panels.',
    'seed': 'int.from_bytes(SHA256(f"{dataset}|{sample_id}|{seed}").digest()[:8], little), then NumPy default_rng(seed).permutation(k).',
    'reduction': 'Sort the five numeric seeds ascending; cast each model score tensor to torch.float64 before ordered summation; divide by5.',
    'candidate_mask': 'One shared frozen catalog; exclude padding and unique original-history items except target; apply the shared mask to the mean scores.',
    'ranking': 'Strictly greater target score count +1; ordered Top10 from torch.topk; tie diagnostics never rewrite ranks.',
    'repeated_members': 'Retain all five contributions, including repeated histories. Cache reuse does not change their multiplicities.',
    'affected': 'Frozen g1 raw-item histories contain >=2 distinct sequences; canonical excluded; same population for Control/Shuffle/Ensemble/LBSP.',
    'stability': 'Five final group recommendations, all10 unordered pairs per user retained, then users; any rank<=10 flip across groups.',
    'accuracy': 'Report each group and the unweighted mean of its five final-group metrics; do not select the best group.',
    'seed_summary': 'Exactly three training seeds; mean and sample SD(ddof1) of per-model group means, not15 training replicates.',
    'clock': 'Optional time buckets are permuted jointly if actually supplied. No clock metadata is invented or verified when absent.',
    'cache': 'Cache only unmasked deterministic model scores. Key must bind dataset/checkpoint/model identity and every actual model-input component.',
}
X6_PROPOSED_POLICY = {
    'status': 'INFERENCE_SUBSTAGE_PROTOCOL_ONLY_NOT_A_BENCHMARK_RESULT',
    'methods': ('Control', 'Shuffle', 'Ensemble', 'LBSP'),
    'evaluation_batch_size': 256,
    'representative_training_seed': 42,
    'model_dtype': 'float32; ensemble score accumulation float64',
    'warmup': 'First two batches for each method, outside measured repeats.',
    'full_panel_repeats': 5,
    'timer': 'Synchronize device before perf_counter start and after fixed output D2H; wall-clock interval includes CPU preprocessing and GPU work.',
    'included': ('batch feature/pooling construction', 'legal permutation generation', 'H2D', 'forward',
                 'ensemble ordered score mean', 'candidate mask', 'GPU strict rank and Top10', 'fixed small result D2H'),
    'excluded': ('disk reads', 'checkpoint loading', 'CSV/JSON serialization', 'disk writes', 'CPU re-sorting'),
    'ensemble_unit': 'One g1 five-member recommendation; not the25 views of outer-group stability.',
    'cache': 'Clear score cache before every full-panel repeat; at most reuse identical members inside one user recommendation, with multiplicity retained.',
    'memory': 'After warmup/synchronization record resident baseline; reset peak stats per repeat and report absolute/incremental peak allocated plus reserved bytes.',
    'hardware': 'Same declared GPU, software, batch and background load; serialize benchmark against training/other GPU jobs.',
    'training_epoch_cost': 'Pending separate controlled temporary-model measurement; not measured by this no-training substage and not filled from old heterogeneous logs.',
}


def integer(value, label='integer', minimum=None):
    if isinstance(value, bool):
        raise ValueError(label + ': Boolean is not an integer')
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, str) and re.fullmatch(r'[+-]?[0-9]+', value.strip()):
        number = int(value.strip())
    else:
        raise ValueError(label + ': Expected an exact integer')
    if minimum is not None and number < minimum:
        raise ValueError(label + ': Integer below minimum')
    return number


def identifier(value):
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(int(value))
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Expected a nonempty raw identifier')
    return value


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric metric/score')
    return float(value)


def legacy_seed(dataset, sample_id, seed):
    payload = f'{identifier(dataset)}|{identifier(sample_id)}|{integer(seed, "seed", 0)}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little', signed=False)


def bucket_size(row, history_len):
    n = integer(history_len, 'history length', 1)
    sizes = [min(integer(row[k], k, 1), n) for k in ('last_bucket_size', 'last_history_bucket_size')
             if k in row and row[k] is not None and row[k] != '']
    if not sizes or len(set(sizes)) != 1:
        raise ValueError('Missing/conflicting clipped last-bucket size aliases')
    return sizes[0]


def _history(items, events, buckets):
    raw = [identifier(v) for v in items]
    keys = [integer(v, 'event key', 0) for v in events]
    if not 1 <= len(raw) <= MAX_LEN or len(raw) != len(keys) or len(set(keys)) != len(keys):
        raise ValueError('History length/event identities invalid')
    times = None if buckets is None else [integer(v, 'time bucket') for v in buckets]
    if times is not None and (len(times) != len(raw) or any(a > b for a, b in zip(times, times[1:]))):
        raise ValueError('Optional clock length/chronology invalid; do not reorder the input')
    return raw, keys, times


def _check_k(times, size, n):
    k = integer(size, 'last-bucket size', 1)
    if k > n:
        raise ValueError('Clip declared last-bucket size to retained history first')
    if times is not None:
        actual = 0
        for value in reversed(times):
            if value != times[-1]:
                break
            actual += 1
        if k != actual:
            raise ValueError('Last-bucket size disagrees with supplied native clock')
    return k


def apply_last_bucket(items, raw_lines, time_buckets, k, indices):
    raw, events, times = _history(items, raw_lines, time_buckets)
    size = _check_k(times, k, len(raw))
    order = [integer(v, 'permutation index', 0) for v in indices]
    if sorted(order) != list(range(size)):
        raise ValueError('Supplied indices are not exactly one last-bucket permutation')
    def apply(values):
        return values[:-size] + [values[-size:][i] for i in order]
    return apply(raw), apply(events), None if times is None else apply(times)


def validate_joint(reference_items, reference_events, reference_buckets,
                   variant_items, variant_events, variant_buckets, k):
    old_items, old_events, old_times = _history(reference_items, reference_events, reference_buckets)
    new_items, new_events, new_times = _history(variant_items, variant_events, variant_buckets)
    if len(old_items) != len(new_items) or (old_times is None) != (new_times is None):
        raise ValueError('Permutation length/clock-presence changed')
    size = _check_k(old_times, k, len(old_items))
    _check_k(new_times, size, len(new_items))
    old = list(zip(old_items, old_events)) if old_times is None else list(zip(old_items, old_events, old_times))
    new = list(zip(new_items, new_events)) if new_times is None else list(zip(new_items, new_events, new_times))
    if old[:-size] != new[:-size] or Counter(old[-size:]) != Counter(new[-size:]):
        raise ValueError('Permutation changed prefix or joint last-bucket event/item/optional-clock multiset')
    return {'prefix_fixed': True, 'joint_multiset_fixed': True, 'native_clock_verified': old_times is not None,
            'changed_raw_items': old_items != new_items, 'changed_event_order': old_events != new_events}


def validate_identity(reference, variant):
    for key in ('sample_id', 'user_id', 'target_item'):
        if identifier(reference[key]) != identifier(variant[key]):
            raise ValueError('Permutation changed sample/user/target identity: ' + key)
    for key in ('target_raw_line', 'target_bucket', 'target_timestamp'):
        if key in reference or key in variant:
            if key not in reference or key not in variant or integer(reference[key], key) != integer(variant[key], key):
                raise ValueError('Permutation changed target event/time identity: ' + key)
    return True


def control_groups(history_len, max_len=MAX_LEN):
    n, width = integer(history_len, 'history length', 1), integer(max_len, 'max_len', 1)
    if n > width:
        raise ValueError('History exceeds model input width')
    return [-1] * (width - n) + list(range(n)), n


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def sequence_hash(items):
    # Identical to existing Control/LBSP affected-label hashing.
    return hashlib.sha256(json.dumps(tuple(identifier(v) for v in items), separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def joint_hash(items, raw_lines, time_buckets=None):
    raw, events, times = _history(items, raw_lines, time_buckets)
    return identity_hash({'items': raw, 'raw_lines': events, 'time_buckets': times})


def model_input_hash(sequence, group_positions, num_tokens):
    seq = [integer(v, 'model item index', 0) for v in sequence]
    groups = [integer(v, 'group index', -1) for v in group_positions]
    tokens = integer(num_tokens, 'num_tokens', 1)
    if not seq or len(seq) != len(groups) or tokens > len(seq):
        raise ValueError('Model input/hash component shapes differ')
    return identity_hash({'sequence': seq, 'group_positions': groups, 'num_tokens': tokens})


def cache_key(model_identity, input_sha256):
    if not model_identity:
        raise ValueError('A score cache must be namespaced by a nonempty model/checkpoint identity')
    if not re.fullmatch(r'[0-9a-f]{64}', str(input_sha256)):
        raise ValueError('Expected a canonical model-input SHA256')
    return identity_hash({'model_identity': model_identity, 'model_input_sha256': input_sha256})


def cache_ledger(logical_view_rows, actual_forward_rows, forward_batch_calls):
    logical = integer(logical_view_rows, 'logical view rows', 0)
    actual = integer(actual_forward_rows, 'actual forward rows', 0)
    calls = integer(forward_batch_calls, 'forward batch calls', 0)
    if actual > logical or (actual == 0) != (calls == 0) or calls > actual:
        raise ValueError('Impossible cache/actual-forward accounting')
    return {'logical_view_rows': logical, 'actual_forward_rows': actual, 'forward_batch_calls': calls,
            'cache_hits': logical - actual}


def affected_label(first_five_item_sequences):
    values = [tuple(identifier(v) for v in row) for row in first_five_item_sequences]
    if len(values) != 5:
        raise ValueError('Affected population requires five g1 inputs; canonical excluded')
    distinct = len(set(values))
    return {'affected': int(distinct >= 2), 'distinct_sequences_5perm': distinct,
            'sequence_sha256': [sequence_hash(row) for row in values]}


def candidate_indices(num_items, history_item_indices, target_index):
    count = integer(num_items, 'num_items', 1)
    target = integer(target_index, 'target index', 1)
    history = {integer(v, 'history item index', 0) for v in history_item_indices}
    if target > count or any(v > count for v in history):
        raise ValueError('History/target is outside frozen mapped catalog')
    masked = history - {0, target}
    return [i for i in range(1, count + 1) if i not in masked]


def rank_from_scores(scores, target_index, history_item_indices=()):
    """Small CPU oracle only. No deterministic replacement for torch.topk ties."""
    values = list(scores)
    eligible = candidate_indices(len(values) - 1, history_item_indices, target_index)
    target = integer(target_index, 'target index', 1)
    checked = {i: _finite(values[i]) for i in eligible}
    return 1 + sum(value > checked[target] for value in checked.values())


def ordered_score_mean(seed_scores):
    """Small binary64 oracle for the prescribed ordered torch.float64 fold."""
    normalized = {integer(seed, 'member seed', 0): list(values) for seed, values in seed_scores.items()}
    if len(normalized) != len(seed_scores) or len(normalized) != 5:
        raise ValueError('Require five distinct ensemble member seeds')
    if set(normalized) not in [set(values) for values in ENSEMBLE_GROUPS.values()]:
        raise ValueError('Members differ from the predeclared ensemble groups')
    widths = {len(values) for values in normalized.values()}
    if len(widths) != 1 or not next(iter(widths)):
        raise ValueError('All five score vectors must share a nonempty full-catalog width')
    total = [0.0] * next(iter(widths))
    for seed in sorted(normalized):
        total = [left + _finite(right) for left, right in zip(total, normalized[seed])]
    result = [value / 5 for value in total]
    if any(not math.isfinite(value) for value in result):
        raise ValueError('Ordered binary64 ensemble mean is nonfinite')
    return result


def metrics_from_ranks(ranks):
    values = [integer(v, 'target rank', 1) for v in ranks]
    if not values:
        return {'samples': 0, **{k: None for k in METRICS}}
    n = len(values)
    return {'samples': n, 'ndcg10': math.fsum(1 / math.log2(r + 1) if r <= 10 else 0 for r in values) / n,
            'hr10': sum(r <= 10 for r in values) / n, 'hr20': sum(r <= 20 for r in values) / n,
            'recall50': sum(r <= 50 for r in values) / n, 'mrr': math.fsum(1 / r for r in values) / n,
            'mrr10': math.fsum(1 / r if r <= 10 else 0 for r in values) / n,
            'mean_rank': math.fsum(values) / n}


def overlap10(left, right):
    a, b = [identifier(v) for v in left], [identifier(v) for v in right]
    if len(a) != 10 or len(b) != 10 or len(set(a)) != 10 or len(set(b)) != 10:
        raise ValueError('Top10 lists must contain ten unique items')
    return len(set(a) & set(b)) / 10


def stability(five_top10, five_ranks):
    outputs, ranks = list(five_top10), [integer(v, 'target rank', 1) for v in five_ranks]
    if len(outputs) != 5 or len(ranks) != 5:
        raise ValueError('Primary stability requires exactly five outputs, canonical excluded')
    values = [overlap10(a, b) for a, b in combinations(outputs, 2)]
    hits = [rank <= 10 for rank in ranks]
    return {'overlap_5perm_pair_mean': math.fsum(values) / 10,
            'hit_flip_user_fraction': int(any(hits) and not all(hits)), 'unordered_pairs_retained': 10}


def _metric_record(row):
    count = integer(row['samples'], 'metric sample count', 0)
    if not count:
        if any(row[key] is not None for key in METRICS):
            raise ValueError('Empty-population metrics must be null')
        return count
    values = {key: _finite(row[key]) for key in METRICS}
    if (any(not 0 <= values[key] <= 1 for key in METRICS if key != 'mean_rank')
            or not values['hr10'] <= values['hr20'] <= values['recall50']
            or values['mrr10'] > values['mrr'] or values['mean_rank'] < 1):
        raise ValueError('Metric record violates ranking-metric ranges')
    return count


def group_accuracy_summary(five_group_metrics):
    """Input metrics are FIVE FINAL ENSEMBLES, not the five members of one."""
    if isinstance(five_group_metrics, dict):
        if set(five_group_metrics) != set(GROUP_NAMES):
            raise ValueError('Require all five predeclared ensemble groups')
        rows = [five_group_metrics[name] for name in GROUP_NAMES]
    else:
        rows = list(five_group_metrics)
    if len(rows) != 5:
        raise ValueError('Require five final-group metric records')
    counts = [_metric_record(row) for row in rows]
    if len(set(counts)) != 1:
        raise ValueError('Ensemble groups must use the same user population')
    n = counts[0]
    means = {key: math.fsum(_finite(row[key]) for row in rows) / 5 if n else None for key in METRICS}
    if not n and any(row[key] is not None for row in rows for key in METRICS):
        raise ValueError('Empty group metrics must be null')
    return {'samples': n, 'per_group': dict(zip(GROUP_NAMES, rows)), 'mean': means,
            'ensemble_groups': 5, 'training_repetitions': 1, 'group_aggregation': 'unweighted mean; no best-group selection'}


def training_seed_summary(seed_metrics):
    normalized = {integer(seed, 'training seed'): row for seed, row in seed_metrics.items()}
    if len(normalized) != len(seed_metrics) or set(normalized) != set(TRAINING_SEEDS):
        raise ValueError('Require exactly three distinct fixed training seeds')
    rows = [normalized[seed] for seed in TRAINING_SEEDS]
    sizes = [_metric_record(row) for row in rows]
    if len(set(sizes)) != 1:
        raise ValueError('Training seeds do not use the same population')
    n = sizes[0]
    if not n and any(row[key] is not None for row in rows for key in METRICS):
        raise ValueError('Empty seed metrics must be null')
    return {'samples_per_seed': n, 'training_seeds': list(TRAINING_SEEDS), 'training_repetitions': 3,
            'mean': {k: statistics.mean(_finite(row[k]) for row in rows) if n else None for k in METRICS},
            'sample_SD_ddof1': {k: statistics.stdev(_finite(row[k]) for row in rows) if n else None for k in METRICS}}
