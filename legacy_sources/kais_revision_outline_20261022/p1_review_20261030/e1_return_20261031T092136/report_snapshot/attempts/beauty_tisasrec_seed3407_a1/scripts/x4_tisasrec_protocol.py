"""Pure standard-library X4 data/time/statistic contract; no file or model I/O.

The five PRIMARY permutations are the existing sealed CSV panels. Helpers for
their legacy seed specification do not implement or replace NumPy's PCG64.
The dense time_matrix function is a small exact-integer oracle; the runner can
batch time_features on its GPU without allocating Python matrices per sample.
"""
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from numbers import Integral
import re
import statistics

VERSION = 'x4_tisasrec_protocol_v1'
TRAINING_SEEDS = (42, 2026, 3407)
PERMUTATION_SEEDS = (101, 202, 303, 404, 505)
MAX_LEN = 50
TIME_SPAN = 256
MAX_TIME_OFFSET_BOUND = 2 ** 53 - 1  # Conservative safety bound, not a unit conversion.
DATASET_TIME_UNITS = {'MovieLens-1M': 'Unix second', 'Amazon_Beauty': 'Unix day (timestamp // 86400)'}
TRAIN_REQUIRED_COLUMNS = ('user_id', 'item_id', 'time_bucket', 'raw_line')
EVENT_REQUIRED_COLUMNS = ('user_id', 'item_id', 'raw_line', 'timestamp', 'time_bucket')
PANEL_REQUIRED_COLUMNS = ('sample_id', 'user_id', 'target_item', 'target_raw_line',
                          'target_timestamp', 'target_bucket', 'sequence_items', 'sequence_raw_lines')
MAP_COLUMNS = ('raw_item_id', 'item_idx')
PREDICTION_REQUIRED_COLUMNS = ('sample_id', 'user_id', 'target_item', 'rank',
                               'top10_item_idx', 'top10_raw_item_id')
METRICS = ('ndcg10', 'hr10', 'hr20', 'recall50', 'mrr', 'mrr10', 'mean_rank')
METADATA_RULES = {
    'time_source': 'Join only visible sequence_raw_lines (and target for validation) to sealed full file_order by (user_id,raw_line); require item identity.',
    'optional_sequence_buckets': 'Not emitted by the R1 builder; if present, require exact equality with joined native buckets.',
    'time_normalization': 'Visible history only, subtract minimum native bucket, minimum positive adjacent unique gap or1; floor(abs difference/scale), clip256.',
    'target_time': 'Strict-order validation only; never a model feature or normalization statistic.',
    'padding': 'Right padding for the pinned RecBole true-length-minus-one readout; padded relation rows/columns are0.',
    'mapping': 'Exact frozen Control map, raw IDs preserved as strings, item indices contiguous1..N and0 reserved.',
    'training_order': 'Stable user,time_bucket,_source_order; _source_order is original train-CSV row ordinal before sorting. raw_line is event identity.',
    'target_cutoff': 'All original next-item targets and preceding max50 event windows; do not shuffle X4 training histories.',
    'candidate_mask': 'Full frozen catalog; exclude padding0 and unique history item IDs except the target.',
    'rank': 'One plus number of eligible scores strictly greater than target; ordered torch.topk is separate and ties are diagnostic.',
    'permutations': 'Use all five sealed CSV panels; preserve item/event/time pairing and input order, prefix, target and per-bucket multiset.',
    'affected': 'At least two distinct raw item sequences among FIVE permutations; canonical excluded; no tensor canonicalization.',
    'stability': 'Ten unordered pairs per user retained including repeated inputs, then users; user-any rank<=10 flip across five permutations.',
    'seed_sd': 'Three fixed training seeds, sample SD ddof1; input permutations are not independent model repetitions.',
}


def integer(value, label='integer', minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValueError(label + ': Boolean is not an integer')
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, str) and re.fullmatch(r'[+-]?[0-9]+', value.strip()):
        result = int(value.strip())
    else:
        raise ValueError(label + ': Expected an exact integer or integer string')
    if minimum is not None and result < minimum or maximum is not None and result > maximum:
        raise ValueError(label + ': Integer outside declared bounds')
    return result


def identifier(value, label='identifier'):
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(int(value))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + ': Expected a nonempty raw identifier')
    return value


def parse_list(value, label='list'):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(label + ': Invalid JSON list') from exc
    elif isinstance(value, (list, tuple)):
        parsed = list(value)
    else:
        raise ValueError(label + ': Expected a JSON list or in-memory list')
    if not isinstance(parsed, list):
        raise ValueError(label + ': Expected a list')
    if any(isinstance(x, bool) or not isinstance(x, (str, Integral)) for x in parsed):
        raise ValueError(label + ': Entries must be integer/string scalars without null/Boolean/nesting')
    return list(parsed)


def native_time(value, label='time_bucket'):
    return integer(value, label, -(2 ** 63), 2 ** 63 - 1)


def time_features(buckets):
    """Return original-order visible offsets and scale, never target/future time."""
    values = [native_time(v) for v in buckets]
    if not 1 <= len(values) <= MAX_LEN:
        raise ValueError('Visible time history must have1..50 events before padding')
    origin = min(values)
    offsets = [v - origin for v in values]
    if max(offsets) > MAX_TIME_OFFSET_BOUND:
        raise ValueError('Visible offsets exceed the conservative exact-clock bound')
    distinct = sorted(set(values))
    scale = min((b - a for a, b in zip(distinct, distinct[1:])), default=1)
    return offsets, scale


def time_matrix(buckets, max_len=MAX_LEN, time_span=TIME_SPAN):
    """Exact integer oracle, with valid events first and padding last."""
    width = integer(max_len, 'max_len', 1)
    clip = integer(time_span, 'time_span', 0)
    offsets, scale = time_features(buckets)
    if len(offsets) > width:
        raise ValueError('History exceeds requested matrix length; truncate before time features')
    matrix = [[0] * width for _ in range(width)]
    for i, left in enumerate(offsets):
        for j, right in enumerate(offsets):
            matrix[i][j] = min(clip, abs(left - right) // scale)
    return matrix


def normalize_event(row, dataset=None):
    missing = set(EVENT_REQUIRED_COLUMNS) - set(row)
    if missing:
        raise ValueError('Event lacks required fields: ' + ','.join(sorted(missing)))
    event = {'user_id': identifier(row['user_id'], 'event user'), 'item_id': identifier(row['item_id'], 'event item'),
             'raw_line': integer(row['raw_line'], 'event raw_line', 0, 2 ** 63 - 1),
             'timestamp': native_time(row['timestamp'], 'event timestamp'),
             'time_bucket': native_time(row['time_bucket'], 'event time_bucket')}
    if dataset is not None:
        if dataset not in DATASET_TIME_UNITS:
            raise ValueError('Unknown native-clock dataset')
        expected = event['timestamp'] if dataset == 'MovieLens-1M' else event['timestamp'] // 86400
        if event['time_bucket'] != expected:
            raise ValueError('Native bucket disagrees with dataset timestamp unit')
    return event


def build_event_index(records, dataset, needed_keys=None):
    """Caller supplies rows; optional filter avoids retaining unrelated events."""
    needed = None if needed_keys is None else {(identifier(u), integer(k, 'needed raw_line', 0)) for u, k in needed_keys}
    result = {}
    for row in records:
        key = (identifier(row['user_id']), integer(row['raw_line'], 'raw_line', 0))
        if needed is not None and key not in needed:
            continue
        if key in result:
            raise ValueError('Duplicate (user,raw_line) event identity')
        result[key] = normalize_event(row, dataset)
    if needed is not None and set(result) != needed:
        raise ValueError('Visible/target event keys missing from sealed event source')
    return result


def validate_mapping(mapping):
    converted = {identifier(k, 'map raw item'): integer(v, 'map item index', 1) for k, v in mapping.items()}
    if len(converted) != len(mapping) or not converted or set(converted.values()) != set(range(1, len(converted) + 1)):
        raise ValueError('Frozen map must be a nonempty contiguous bijection1..N')
    return converted


def last_bucket_size(row):
    aliases = [integer(row[key], key, 1) for key in ('last_bucket_size', 'last_history_bucket_size') if key in row]
    if not aliases or len(set(aliases)) != 1:
        raise ValueError('Missing or conflicting last-bucket size aliases')
    return aliases[0]


def _flag(value, label):
    if isinstance(value, str) and value.strip().lower() in ('true', 'false'):
        return int(value.strip().lower() == 'true')
    if isinstance(value, bool):
        return int(value)
    return integer(value, label, 0, 1)


def join_event_times(row, event_index, item_map=None, dataset=None):
    """Pure identity/time join in the submitted order, without sorting a panel."""
    if set(PANEL_REQUIRED_COLUMNS) - set(row):
        raise ValueError('Strict panel lacks raw event/time identity fields')
    sample, user, target = (identifier(row[key], key) for key in ('sample_id', 'user_id', 'target_item'))
    items = [identifier(v, 'history item') for v in parse_list(row['sequence_items'], 'sequence_items')]
    keys = [integer(v, 'history raw_line', 0) for v in parse_list(row['sequence_raw_lines'], 'sequence_raw_lines')]
    if not 1 <= len(items) <= MAX_LEN or len(items) != len(keys) or len(set(keys)) != len(keys):
        raise ValueError('Strict panel history length/event identities invalid')
    target_key = integer(row['target_raw_line'], 'target_raw_line', 0)
    if target_key in keys or target in items:
        raise ValueError('Strict ML/Beauty panel leaks target event/item into retained history')
    buckets = []
    for item, key in zip(items, keys):
        if (user, key) not in event_index:
            raise ValueError('Visible event key is missing from the sealed event index')
        event = normalize_event(event_index[(user, key)], dataset)
        if event['user_id'] != user or event['raw_line'] != key or event['item_id'] != item:
            raise ValueError('Panel item/user/event-key identity mismatch')
        buckets.append(event['time_bucket'])
    if any(a > b for a, b in zip(buckets, buckets[1:])):
        raise ValueError('History violates native bucket chronology; do not reorder panel inputs')
    if (user, target_key) not in event_index:
        raise ValueError('Target event key is missing from the sealed event index')
    target_event = normalize_event(event_index[(user, target_key)], dataset)
    target_bucket = native_time(row['target_bucket'], 'target_bucket')
    target_timestamp = native_time(row['target_timestamp'], 'target_timestamp')
    if (target_event['user_id'] != user or target_event['raw_line'] != target_key or target_event['item_id'] != target
            or target_event['time_bucket'] != target_bucket or target_event['timestamp'] != target_timestamp
            or target_bucket <= max(buckets)):
        raise ValueError('Strict target identity/time does not follow all visible history buckets')
    if 'sequence_buckets' in row:
        declared = [native_time(v) for v in parse_list(row['sequence_buckets'], 'sequence_buckets')]
        if declared != buckets:
            raise ValueError('Optional sequence_buckets disagrees with visible event-time join')
    k = last_bucket_size(row)
    actual_k = 0
    for bucket in reversed(buckets):
        if bucket != buckets[-1]:
            break
        actual_k += 1
    if k != actual_k:
        raise ValueError('Declared last-bucket size disagrees with joined visible clock')
    for key, expected in (('fixed_history_len', len(items)), ('prefix_len', len(items) - k)):
        if key in row and integer(row[key], key, 0) != expected:
            raise ValueError(key + ' differs from frozen retained history')
    pooled = int(k > 1)
    for key in ('direct_pooling', 'ambiguous_last_bucket'):
        if key in row and _flag(row[key], key) != pooled:
            raise ValueError(key + ' label disagrees with event bucket size')
    for key, expected in (('prefix_items', items[:-k]), ('last_bucket_items', items[-k:])):
        if key in row and [identifier(v) for v in parse_list(row[key], key)] != expected:
            raise ValueError(key + ' does not reconstruct the supplied item order')
    for key, expected in (('prefix_raw_lines', keys[:-k]), ('last_bucket_raw_lines', keys[-k:])):
        if key in row and [integer(v, key, 0) for v in parse_list(row[key], key)] != expected:
            raise ValueError(key + ' does not reconstruct the supplied event order')
    result = {'sample_id': sample, 'user_id': user, 'target_item': target, 'target_raw_line': target_key,
              'target_bucket': target_bucket, 'target_timestamp': target_timestamp,
              'items': items, 'raw_lines': keys, 'buckets': buckets, 'last_bucket_size': k,
              'direct_pooling': pooled, 'ambiguous_last_bucket': pooled}
    if item_map is not None:
        mapping = validate_mapping(item_map)
        if any(v not in mapping for v in items + [target]):
            raise ValueError('History/target item outside frozen mapping')
        result.update(mapped_items=[mapping[v] for v in items], target_idx=mapping[target])
    return result


def legacy_permutation_seed(dataset, sample_id, permutation_seed):
    seed = integer(permutation_seed, 'permutation_seed', 0)
    payload = f'{identifier(dataset)}|{identifier(sample_id)}|{seed}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little', signed=False)


def apply_last_bucket_permutation(items, raw_lines, bucket_size, indices):
    """Apply caller-provided legacy NumPy indices jointly, with no RNG calls."""
    items, keys = list(items), list(raw_lines)
    k = integer(bucket_size, 'bucket_size', 1)
    order = [integer(i, 'permutation index', 0) for i in indices]
    if len(items) != len(keys) or not k <= len(items) or sorted(order) != list(range(k)):
        raise ValueError('Invalid joint last-bucket permutation')
    return items[:-k] + [items[-k:][i] for i in order], keys[:-k] + [keys[-k:][i] for i in order]


def validate_permutation(reference, variant):
    """Validate already joined records; return raw pre-model input identity."""
    for key in ('sample_id', 'user_id', 'target_item', 'target_raw_line', 'target_bucket', 'target_timestamp', 'last_bucket_size'):
        if reference[key] != variant[key]:
            raise ValueError('Permutation changed frozen identity: ' + key)
    k = reference['last_bucket_size']
    if len(reference['items']) != len(variant['items']) or len(reference['raw_lines']) != len(variant['raw_lines']):
        raise ValueError('Permutation changed visible history length')
    before = list(zip(reference['items'], reference['raw_lines'], reference['buckets']))
    after = list(zip(variant['items'], variant['raw_lines'], variant['buckets']))
    if before[:-k] != after[:-k] or Counter(before[-k:]) != Counter(after[-k:]):
        raise ValueError('Permutation changed prefix or event/item/time joint multiset')
    return {'changed_raw_items': tuple(reference['items']) != tuple(variant['items']),
            'changed_event_order': tuple(reference['raw_lines']) != tuple(variant['raw_lines']),
            'prefix_fixed': True, 'joint_last_bucket_multiset_fixed': True}


def sequence_sha256(sequence):
    raw = tuple(identifier(v, 'raw item') for v in sequence)
    return hashlib.sha256(json.dumps(raw, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


def affected_label(five_sequences):
    values = [tuple(identifier(v, 'raw item') for v in seq) for seq in five_sequences]
    if len(values) != 5:
        raise ValueError('Affected label requires FIVE permutations, canonical excluded')
    distinct = len(set(values))
    return {'affected': int(distinct >= 2), 'distinct_sequences_5perm': distinct,
            'sequence_sha256': [sequence_sha256(v) for v in values]}


def metrics_from_ranks(ranks):
    values = [integer(v, 'target rank', 1) for v in ranks]
    if not values:
        return {'samples': 0, **{k: None for k in METRICS}, 'median_rank': None, 'rank1_ratio': None}
    n = len(values)
    return {'samples': n, 'ndcg10': math.fsum(1 / math.log2(v + 1) if v <= 10 else 0 for v in values) / n,
            'hr10': sum(v <= 10 for v in values) / n, 'hr20': sum(v <= 20 for v in values) / n,
            'recall50': sum(v <= 50 for v in values) / n, 'mrr': math.fsum(1 / v for v in values) / n,
            'mrr10': math.fsum(1 / v if v <= 10 else 0 for v in values) / n,
            'mean_rank': math.fsum(values) / n, 'median_rank': statistics.median(values),
            'rank1_ratio': sum(v == 1 for v in values) / n}


def overlap10(left, right):
    a, b = [identifier(v) for v in left], [identifier(v) for v in right]
    if len(a) != 10 or len(b) != 10 or len(set(a)) != 10 or len(set(b)) != 10:
        raise ValueError('Overlap@10 requires two unique ten-item recommendation lists')
    return len(set(a) & set(b)) / 10


def any_hitflip(five_ranks):
    values = [integer(v, 'target rank', 1) for v in five_ranks]
    if len(values) != 5:
        raise ValueError('Primary user-any flip requires FIVE permutation ranks')
    flags = [v <= 10 for v in values]
    return int(any(flags) and not all(flags))


def stability_from_predictions(five_top10, five_ranks):
    lists = list(five_top10)
    if len(lists) != 5:
        raise ValueError('Primary stability requires FIVE permutation recommendations')
    pairs = [overlap10(a, b) for a, b in combinations(lists, 2)]
    return {'overlap_5perm_pair_mean': math.fsum(pairs) / 10, 'hit_flip_user_fraction': any_hitflip(five_ranks),
            'unordered_pairs_retained': 10}
