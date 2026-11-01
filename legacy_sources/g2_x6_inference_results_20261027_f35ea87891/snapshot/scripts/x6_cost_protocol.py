"""Pure validation/statistics for the X6 inference and training cost protocol.

No file access, interaction/weight access, Torch import, GPU work or training.
The caller measures on the server and binds model/data/source hashes separately.
"""
import math
import re
import statistics

VERSION = 'x6_cost_protocol_v1'
TAGS = ('ml1m', 'beauty', 'sports', 'toys')
METHODS = ('control', 'shuffle', 'ensemble', 'lbsp')
SAMPLES = {'ml1m': 945, 'beauty': 2966, 'sports': 4659, 'toys': 2547}
REPRESENTATIVE_SEED = 42
INFERENCE_BATCH_SIZE = 256
TRAINING_BATCH_SIZE = 128
INFERENCE_WARMUP_BATCHES = 2
INFERENCE_REPEATS = 5
ENSEMBLE_SEEDS = (101, 202, 303, 404, 505)
INFERENCE_SCOPE = {
    'included': ('cpu_feature_and_pooling_construction', 'ensemble_permutation_generation',
                 'host_to_device_transfer', 'model_forward', 'ensemble_float64_score_mean',
                 'candidate_mask', 'gpu_strict_rank_and_top10', 'rank_and_top10_device_to_host'),
    'excluded': ('file_reading', 'checkpoint_loading', 'csv_json_serialization',
                 'post_timing_cpu_result_comparison'),
    'source_panels_preloaded': True,
    'features_built_inside_timing': True,
    'ensemble_member_generation_inside_timing': True,
    'cache_across_repeats': False,
    'cache_within_recommendation': False,
    'gpu_sync_before_and_after_wall_timer': True,
    'precision': 'fp32_models_fp64_fixed_order_ensemble',
    'tf32': False,
    'timing_unit': 'seconds_per_full_test_panel',
    'peak_memory_unit': 'bytes',
}
TRAINING_SCOPE = {
    'measured_methods': ('control', 'shuffle', 'lbsp'),
    'ensemble_training_cost': 'shares_control_model_training_cost',
    'batch_size': TRAINING_BATCH_SIZE,
    'temporary_isolated_models_and_optimizers': True,
    'includes': ('source_method_sampler_and_loader', 'training_forward_and_full_item_loss',
                 'backward', 'source_method_gradient_clipping', 'optimizer_step',
                 'shuffle_epoch_priority_refresh'),
    'excludes': ('validation', 'checkpoint_read_write', 'test_evaluation'),
    'source_method_workers_and_optimization_preserved': True,
    'gpu_sync_before_and_after_wall_timer': True,
    'legacy_training_logs_are_not_current_gpu_measurements': True,
}


def _positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')
    return float(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return value


def _sha256(value, name):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(name + ' must be a lowercase SHA256')
    return value


def work_counts(method, samples, batch_size=INFERENCE_BATCH_SIZE):
    """Counts for uncached forward scoring; one ensemble is five views."""
    if method not in METHODS:
        raise ValueError('Unknown benchmark method')
    n = _integer(samples, 'samples', 1)
    batch = _integer(batch_size, 'batch_size', 1)
    views = 5 if method == 'ensemble' else 1
    return {'views_per_recommendation': views, 'logical_view_rows': n * views,
            'actual_forward_rows': n * views, 'forward_batch_calls': ((n + batch - 1) // batch) * views,
            'cache_hits': 0}


def expected_accounting(method, samples, batch_size=INFERENCE_BATCH_SIZE):
    """Exact three-counter schema written by run_x6_inference.py."""
    counts = work_counts(method, samples, batch_size)
    return {key: counts[key] for key in ('actual_forward_rows', 'forward_batch_calls', 'cache_hits')}


def mean_sample_sd(values, expected_count=None):
    """Sample SD describes repeated timings, not training-seed uncertainty."""
    times = [_positive_finite(value, 'time') for value in values]
    if expected_count is not None and len(times) != expected_count:
        raise ValueError('Unexpected number of timing repeats')
    if len(times) < 2:
        raise ValueError('At least two repeats required for sample SD')
    return {'n': len(times), 'mean': statistics.fmean(times), 'sample_sd_ddof1': statistics.stdev(times)}


def memory_record(baseline_allocated, peak_allocated, baseline_reserved, peak_reserved):
    baseline_allocated = _integer(baseline_allocated, 'baseline_allocated')
    peak_allocated = _integer(peak_allocated, 'peak_allocated')
    baseline_reserved = _integer(baseline_reserved, 'baseline_reserved')
    peak_reserved = _integer(peak_reserved, 'peak_reserved')
    if (peak_allocated < baseline_allocated or baseline_reserved < baseline_allocated
            or peak_reserved < baseline_reserved or peak_reserved < peak_allocated):
        raise ValueError('Inconsistent CUDA allocator baseline/peak values')
    return {'baseline_allocated_bytes': baseline_allocated, 'peak_allocated_bytes': peak_allocated,
            'incremental_peak_allocated_bytes': peak_allocated - baseline_allocated,
            'baseline_reserved_bytes': baseline_reserved, 'peak_reserved_bytes': peak_reserved}


def validate_repeat(tag, method, row, repeat_number=None, reference_path=None, reference_sha256=None):
    """Validate one actual runner row; no GPU or historical data access.

    reference_path/SHA may be bound by the collector to a staged reference.
    Current runner stores allocator allocated baseline and max reserved, but
    does not store baseline reserved. That limitation is not filled in here.
    """
    if tag not in TAGS or method not in METHODS:
        raise ValueError('Unknown dataset or method')
    n = SAMPLES[tag]
    if _integer(row['samples'], 'samples', 1) != n:
        raise ValueError('Not a complete reference panel')
    number = _integer(row['repeat'], 'repeat', 1)
    if number > INFERENCE_REPEATS or (repeat_number is not None and number != repeat_number):
        raise ValueError('Repeat number/order differs')
    seconds = _positive_finite(row['seconds'], 'seconds')
    if not math.isclose(_positive_finite(row['seconds_per_user'], 'seconds_per_user'), seconds / n, rel_tol=1e-12):
        raise ValueError('Per-user time differs from full-panel time')
    if row['accounting'] != expected_accounting(method, n):
        raise ValueError('Actual forward row/batch or cache accounting differs')
    baseline = _integer(row['baseline_allocated_bytes'], 'baseline_allocated_bytes')
    peak = _integer(row['peak_allocated_bytes'], 'peak_allocated_bytes')
    increment = _integer(row['peak_increment_bytes'], 'peak_increment_bytes')
    reserved = _integer(row['peak_reserved_bytes'], 'peak_reserved_bytes')
    if peak < baseline or increment != peak - baseline or reserved < peak:
        raise ValueError('CUDA allocator peak/baseline inconsistent')
    if 'baseline_reserved_bytes' in row:
        baseline_reserved = _integer(row['baseline_reserved_bytes'], 'baseline_reserved_bytes')
        if baseline_reserved < baseline or reserved < baseline_reserved:
            raise ValueError('CUDA reserved baseline/peak inconsistent')
    proof = row['matched_reference']
    if (proof.get('status') != 'PASS' or _integer(proof.get('samples'), 'proof samples', 1) != n
            or any(proof.get(k) != 0 for k in ('rank_mismatches', 'ordered_top10_mismatches', 'raw_top10_mismatches'))):
        raise ValueError('Timed result does not reproduce the accepted reference')
    if reference_path is not None and proof.get('reference_path') != reference_path:
        raise ValueError('Reference path differs')
    if reference_sha256 is not None and proof.get('reference_sha256') != _sha256(reference_sha256, 'reference_sha256'):
        raise ValueError('Reference SHA differs')
    if not isinstance(row.get('telemetry_before'), dict):
        raise ValueError('Missing device telemetry availability record')
    return {'status': 'PASS', 'repeat': number, 'samples': n, 'seconds': seconds,
            'seconds_per_user': seconds / n, 'peak_allocated_bytes': peak,
            'peak_increment_bytes': increment, 'peak_reserved_bytes': reserved,
            'accounting': expected_accounting(method, n)}


def summarize5(tag, method, repeats, reference_path=None, reference_sha256=None):
    """Independently derive the exact four summary fields used by the runner."""
    if len(repeats) != INFERENCE_REPEATS:
        raise ValueError('Exactly five full-panel repeats are required')
    verified = [validate_repeat(tag, method, row, i, reference_path, reference_sha256)
                for i, row in enumerate(repeats, 1)]
    times = [row['seconds'] for row in verified]
    return {'seconds_mean': statistics.mean(times), 'seconds_sample_sd': statistics.stdev(times),
            'seconds_per_user_mean': statistics.mean(times) / SAMPLES[tag],
            'peak_allocated_bytes_max': max(row['peak_allocated_bytes'] for row in verified)}


def summarize_inference(method, tag, repeats):
    """Validate five full-panel repeat records, then summarize wall time/memory.

    Each record contains seconds, samples, work counters, output_verified and
    one memory dict produced by memory_record. The initial correctness check
    and repeated output comparisons are caller responsibilities; the latter
    occur after each timer stops. No claimed GPU timing is generated here.
    """
    if method not in METHODS or tag not in TAGS:
        raise ValueError('Unknown dataset or method')
    if len(repeats) != INFERENCE_REPEATS:
        raise ValueError('Exactly five full-panel inference repeats required')
    expected_n = SAMPLES[tag]
    expected = work_counts(method, expected_n)
    seconds, peaks, incremental, reserved = [], [], [], []
    for record in repeats:
        if _integer(record['samples'], 'samples', 1) != expected_n:
            raise ValueError('Full-panel sample count differs')
        for key, value in expected.items():
            if _integer(record[key], key) != value:
                raise ValueError('Logical/actual view, batch or cache count differs: ' + key)
        if record.get('output_verified') is not True:
            raise ValueError('Timed result was not checked against the accepted reference')
        memory = record['memory']
        normalized = memory_record(memory['baseline_allocated_bytes'], memory['peak_allocated_bytes'],
                                   memory['baseline_reserved_bytes'], memory['peak_reserved_bytes'])
        if memory != normalized:
            raise ValueError('Memory fields or incremental allocation are inconsistent')
        seconds.append(_positive_finite(record['seconds'], 'seconds'))
        peaks.append(memory['peak_allocated_bytes'])
        incremental.append(memory['incremental_peak_allocated_bytes'])
        reserved.append(memory['peak_reserved_bytes'])
    stats = mean_sample_sd(seconds, INFERENCE_REPEATS)
    return {'method': method, 'dataset_tag': tag, 'training_seed': REPRESENTATIVE_SEED,
            'samples': expected_n, 'batch_size': INFERENCE_BATCH_SIZE, 'warmup_batches': INFERENCE_WARMUP_BATCHES,
            'repeats': INFERENCE_REPEATS, 'seconds_total_mean': stats['mean'],
            'seconds_total_sample_sd_ddof1': stats['sample_sd_ddof1'],
            'seconds_per_user_mean': stats['mean'] / expected_n,
            'seconds_per_user_sample_sd_ddof1': stats['sample_sd_ddof1'] / expected_n,
            'peak_allocated_bytes_max': max(peaks),
            'incremental_peak_allocated_bytes_max': max(incremental),
            'peak_reserved_bytes_max': max(reserved), 'work_counts_each_repeat': expected}


def validate_comparison_metadata(records):
    """Verify the 4x4 measured grid has a common device and dataset panels.

    Model weights are identified by SHA recorded by the server, not opened by
    this module. Caller supplies a hash of the normalized raw-ID-to-index map,
    so harmless CSV byte formatting differences do not masquerade as different
    catalogs. Same seed42 Control checkpoint must back the ensemble.
    """
    expected_pairs = {(tag, method) for tag in TAGS for method in METHODS}
    by_pair = {}
    for record in records:
        pair = (record.get('dataset_tag'), record.get('method'))
        if pair not in expected_pairs or pair in by_pair:
            raise ValueError('Unknown or duplicate dataset/method pair')
        if record.get('training_seed') != REPRESENTATIVE_SEED:
            raise ValueError('Unreviewed representative training seed')
        for field in ('panel_sha256', 'normalized_mapping_sha256', 'checkpoint_sha256'):
            _sha256(record.get(field), field)
        environment = record.get('environment')
        if not isinstance(environment, dict) or any(not isinstance(environment.get(k), str) or not environment[k]
                for k in ('device', 'device_uuid', 'torch', 'cuda', 'python', 'numpy')):
            raise ValueError('Incomplete GPU/runtime identity')
        if record.get('reference_output_pass') is not True:
            raise ValueError('Missing saved-prediction reproduction')
        if record.get('reference_kind') != ('ensemble_group1' if pair[1] == 'ensemble' else 'canonical'):
            raise ValueError('Reference is not the measured method output')
        by_pair[pair] = record
    if set(by_pair) != expected_pairs:
        raise ValueError('Incomplete four-dataset/four-method grid')
    identity = None
    for tag in TAGS:
        group = [by_pair[(tag, method)] for method in METHODS]
        if len({record['panel_sha256'] for record in group}) != 1:
            raise ValueError('Methods use different test panels')
        if len({record['normalized_mapping_sha256'] for record in group}) != 1:
            raise ValueError('Methods use different item catalog mappings')
        if by_pair[(tag, 'control')]['checkpoint_sha256'] != by_pair[(tag, 'ensemble')]['checkpoint_sha256']:
            raise ValueError('Ensemble must share the specified Control checkpoint')
        for record in group:
            env = tuple((key, record['environment'][key]) for key in ('device', 'device_uuid', 'torch', 'cuda', 'python', 'numpy'))
            if identity is None:
                identity = env
            elif env != identity:
                raise ValueError('Device or runtime changed across benchmark methods')
    return {'status': 'PASS', 'configurations': len(by_pair), 'datasets': len(TAGS), 'methods': len(METHODS),
            'training_seed': REPRESENTATIVE_SEED, 'same_gpu_and_runtime': True,
            'same_panel_and_normalized_map_per_dataset': True, 'control_checkpoint_shared_by_ensemble': True}
