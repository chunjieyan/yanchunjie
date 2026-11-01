"""Metadata-only build for X6 inference; no local model or input-data execution."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import zipfile

BASE = Path(__file__).resolve().parent
OUTLINE = BASE.parent
PROJECT = OUTLINE.parent
EXECUTION = OUTLINE / 'g2_x3_x4_x6_execution_20261025'
X3_REVIEW = OUTLINE / 'g2_x3_inference_results_review_20261027'
INPUT_NAME = 'x6_inference_inputs_20261027.json'
FILES = ('run_x6_inference.py', 'collect_x6_inference.py', 'x3_inference_adapter.py',
         'x3_inference_protocol.py', 'x3_core_helpers.py', 'x6_model_definitions.py', 'x6_cost_protocol.py', INPUT_NAME)
TAGS = ('ml1m', 'beauty', 'sports', 'toys')
METHODS = ('control', 'shuffle', 'ensemble', 'lbsp')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def owned_adapter():
    spec = importlib.util.spec_from_file_location('owned_x6_adapter', BASE / 'x3_inference_adapter.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seal():
    verification = read(X3_REVIEW / 'archive_verification.json')
    independent = read(X3_REVIEW / 'math_independent_audit.json')
    assert verification['status'] == independent['status'] == 'PASS'
    prior = read(X3_REVIEW / 'next_stage_reference.json')
    extracted = Path(verification['destination']) / 'snapshot'
    x3 = read(BASE / 'x3_inference_inputs_20261026.json')
    maps = read(EXECUTION / 'x6_control_lbsp_seed42_map_equivalence.json')
    assert maps['status'] == 'PASS'
    ml = read(BASE / 'lbsp_ml1m_stability_inputs_20261023.json')
    amazon = read(BASE / 'lbsp_amazon_stability_inputs_20261025.json')
    shuffle_snapshot = PROJECT / 'g2_x3_shuffle_results_20261026/snapshot'
    control_source = shuffle_snapshot / 'scripts/model_definition.py'
    lbsp_source = PROJECT / 'g2_snapshot_20261022/snapshot/audit/33_2_train_movielens_r1_seed42.py'
    lbsp_forward = PROJECT / 'g2_snapshot_20261022/snapshot/step38_kais_retrain/scripts/38_20_train_ml1m_r1_bs128.py'
    source_expected = {'control': 'ac2e47de1cd8cba980be4c049f521a36a08a7d54f3610be6bc687cef7383555e',
                       'lbsp': 'e8ae647f3fa3cc2ef9f3087a0ae44cfbd4da2706b2ce33fc558fff0678a04ffb',
                       'forward': 'c718592e65a3d66f2ec1ce6d22c8b9abc3b8d4a393dde49ae014247414741b38'}
    assert (sha(control_source), sha(lbsp_source), sha(lbsp_forward)) == tuple(source_expected.values())
    # Only the owned AST extraction generator runs locally; source definitions are text.
    generated = owned_adapter().build_frozen_module(control_source.read_text(encoding='utf-8'),
        lbsp_source.read_text(encoding='utf-8-sig'), lbsp_forward.read_text(encoding='utf-8-sig'))
    # The X3 generated class and original source differ in whitespace even when
    # their ASTs are identical. The generator checks every selected AST.
    generated_text_sha256 = hashlib.sha256(generated.encode()).hexdigest()
    ast.parse(generated, feature_version=(3, 10))
    (BASE / 'x6_model_definitions.py').write_text(generated, encoding='utf-8')
    shutil.copyfile(BASE / 'run_x3_inference.py', BASE / 'x3_core_helpers.py')
    assert sha(BASE / 'x3_core_helpers.py') == 'b5550a6f3342176a3f3eca688ed4955e2a65d448cbe6de6a1d24202d2f3ee781'
    root_server = PurePosixPath('/root/autodl-tmp/time_uncertainty_rec')
    stage = str(PurePosixPath(prior['stage']).relative_to(root_server))
    shuffle_manifest = read(shuffle_snapshot / 'manifests/x3_shuffle_manifest.json')
    shuffle_stage = str(PurePosixPath(shuffle_manifest['stage']).relative_to(root_server))
    globals_pinned = {
        stage + '/manifest.json': prior['manifest_sha256'],
        stage + '/scripts/run_x3_inference.py': sha(BASE / 'x3_core_helpers.py'),
        shuffle_stage + '/scripts/model_definition.py': source_expected['control'],
        'audit/33_2_train_movielens_r1_seed42.py': source_expected['lbsp'],
        'step38_kais_retrain/scripts/38_20_train_ml1m_r1_bs128.py': source_expected['forward'],
    }
    result = {'version': 'x6_inference_v1', 'methods': list(METHODS), 'representative_seed': 42,
        'generated_model_source_sha256_lf': generated_text_sha256, 'model_scientific_source_sha256': source_expected,
        'datasets': {}, 'root_files_sha256': dict(globals_pinned), 'references': {}, 'helper_sha256': {},
        'protocol': {'scope': 'inference-cost substage only; 4 datasets x 4 methods seed42; training epoch cost remains separate',
            'batch_size': 256, 'warmup': 'one untimed full-panel reference reproduction plus first2 batches',
            'repeats': 5, 'clock': 'perf_counter wall time, torch.cuda.synchronize before/after',
            'includes': ['CPU history feature/group construction', 'ensemble group1 PCG64 input generation',
                'CPU to GPU transfer', 'model forward', 'float64 score mean for ensemble', 'candidate mask',
                'GPU strict rank and Top10', 'fixed small rank/indices GPU to CPU transfers', 'per-forward finite score guard'],
            'excludes': ['disk input reads', 'checkpoint loading', 'prior-prediction reference verification',
                'CSV or JSON serialization', 'checkpoint writing'],
            'method': 'original independent Control/Shuffle/LBSP models and group1 5-view Control ensemble',
            'model_precision': 'FP32, TF32 disabled; ensemble raw-score sum/divide float64 in fixed seed order',
            'ensemble_views': [101, 202, 303, 404, 505], 'ensemble_is_one_recommendation': 'five full views, not outer25',
            'cache': 'disabled for every repeat', 'model_residency': 'one checkpoint/model per dedicated sequential process',
            'memory': 'allocator baseline allocated, absolute peak allocated and increment plus peak reserved, bytes',
            'target_rank': 'strict-greater+1; ordered torch.topk10 retained',
            'new_training': False, 'training_epoch_cost': 'PENDING_SEPARATE_MEASUREMENT'}}
    models = {(record['method'], record['tag'], record['seed']): record for record in prior['models']}
    for tag in TAGS:
        base = x3['datasets'][tag]
        dataset = {key: base[key] for key in ('dataset', 'expected_samples', 'expected_items', 'expected_affected', 'original_path', 'permutation_paths')}
        dataset.update(normalized_mapping_sha256=maps['datasets'][tag]['normalized_mapping_sha256'], runs={})
        assert maps['datasets'][tag]['items'] == dataset['expected_items']
        panel_paths = [base['original_path']] + list(base['permutation_paths'].values())
        for path in panel_paths:
            result['root_files_sha256'][path] = x3['root_files_sha256'][path]
        for method in METHODS:
            historical_method = 'shuffle' if method == 'shuffle' else 'ensemble'
            old = base['runs'][historical_method]['42']
            if method == 'lbsp':
                old_lbsp = (ml if tag == 'ml1m' else amazon)['datasets'][tag]['runs']['42']
                model_dir, checkpoint, best_epoch = old_lbsp['model_dir'], old_lbsp['checkpoint'], old_lbsp['best_epoch']
                checkpoint_sha = None
                records = {key: {'root_path': model_dir + '/' + filename, 'sha256': old_lbsp['files_sha256'][filename]}
                           for key, filename in (('config', 'config.json'), ('map', 'item_id_map.csv'), ('predictions', 'test_per_sample_predictions.csv'))}
                assert records['map']['sha256'] == maps['datasets'][tag]['lbsp_file_sha256']
                extras = {}
            else:
                known = models[(historical_method, tag, 42)]
                model_dir, checkpoint, best_epoch = old['model_dir'], old['checkpoint'], old['best_epoch']
                assert (known['model_dir'], known['checkpoint'], known['best_epoch']) == (model_dir, checkpoint, best_epoch)
                checkpoint_sha = known['checkpoint_sha256_recorded_on_server']
                records = {key: dict(x3['references'][old[field]]) for key, field in
                           (('config', 'reference_config'), ('map', 'reference_map'), ('predictions', 'reference_predictions'))}
                assert records['map']['sha256'] == maps['datasets'][tag]['control_file_sha256']
                completion_name = 'completed_' + historical_method + '_' + tag + '_seed42.json'
                completion_path = extracted / 'verification' / completion_name
                assert sha(completion_path) == known['completion_sha256']
                extras = {stage + '/verification/' + completion_name: known['completion_sha256']}
                if method == 'ensemble':
                    group_name = 'predictions_group1.csv'
                    proof = read(completion_path)
                    group_hash = proof['metadata_sha256'][group_name]
                    local_group = extracted / 'outputs' / ('ensemble_' + tag + '_seed42') / group_name
                    assert sha(local_group) == group_hash
                    records['predictions'] = {'root_path': stage + '/outputs/ensemble_' + tag + '_seed42/' + group_name,
                                              'sha256': group_hash}
            run = {'model_dir': model_dir, 'checkpoint': checkpoint, 'best_epoch': best_epoch, 'checkpoint_sha256': checkpoint_sha}
            paths = set(panel_paths) | set(globals_pinned) | set(extras)
            result['root_files_sha256'].update(extras)
            for key, filename, field in [('config', 'config.json', 'reference_config'),
                                          ('map', 'item_id_map.csv', 'reference_map'),
                                          ('predictions', 'predictions_saved.csv', 'reference_predictions')]:
                relative = 'references/' + tag + '_' + method + '/' + filename
                result['references'][relative] = records[key]
                path, digest = records[key]['root_path'], records[key]['sha256']
                if path in result['root_files_sha256']:
                    assert result['root_files_sha256'][path] == digest
                result['root_files_sha256'][path] = digest
                paths.add(path)
                run[field] = relative
            run['required_root_files'] = sorted(paths)
            dataset['runs'][method] = run
        result['datasets'][tag] = dataset
    for name in ('x3_inference_adapter.py', 'x3_inference_protocol.py', 'x3_core_helpers.py', 'x6_model_definitions.py', 'x6_cost_protocol.py'):
        result['helper_sha256'][name] = sha(BASE / name)
    write(BASE / INPUT_NAME, result)
    digest = sha(BASE / INPUT_NAME)
    for name in ('run_x6_inference.py', 'collect_x6_inference.py'):
        path = BASE / name
        if path.exists():
            source = path.read_text(encoding='utf-8')
            source, count = re.subn(r"(?m)^INPUT_SHA256 = '[^']+'", "INPUT_SHA256 = '" + digest + "'", source)
            assert count == 1, name
            path.write_text(source, encoding='utf-8')
    print(json.dumps({'status': 'SEALED', 'input_sha256': digest, 'root_files': len(result['root_files_sha256']),
                      'references': len(result['references'])}))


def package():
    inputs = read(BASE / INPUT_NAME)
    for name, expected in inputs['helper_sha256'].items():
        assert sha(BASE / name) == expected, name
    payload = {}
    for name in FILES:
        data = (BASE / name).read_bytes()
        if name.endswith('.py'):
            ast.parse(data.decode('utf-8'), feature_version=(3, 10))
        if name in ('run_x6_inference.py', 'collect_x6_inference.py'):
            assert ("INPUT_SHA256 = '" + sha(BASE / INPUT_NAME) + "'") in data.decode('utf-8')
        payload['server_files/' + name] = data
    payload['X6_推断成本_服务器操作.md'] = (BASE / 'X6_推断成本_服务器操作.md').read_bytes()
    reports = {}
    for name in ('x3_inference_adapter_qa.json', 'x6_cost_protocol_qa.json', 'x6_inference_runner_qa.json', 'x6_inference_collector_qa.json'):
        path = EXECUTION / name
        value = read(path)
        assert 'PASS' in value['status'], name
        if name == 'x3_inference_adapter_qa.json':
            assert value['adapter_sha256'] == sha(BASE / 'x3_inference_adapter.py')
        if name == 'x6_cost_protocol_qa.json':
            assert value['protocol_sha256'] == sha(BASE / 'x6_cost_protocol.py')
        if name == 'x6_inference_runner_qa.json':
            assert value['runner_sha256'] == sha(BASE / 'run_x6_inference.py')
        if name == 'x6_inference_collector_qa.json':
            assert value['collector_sha256'] == sha(BASE / 'collect_x6_inference.py')
        reports[name] = {'status': value['status'], 'sha256': sha(path)}
        payload['qa/' + name] = path.read_bytes()
    summary = {'status': 'LOCAL_SYNTHETIC_QA_PASS_SERVER_MEASUREMENTS_PENDING', 'input_sha256': sha(BASE / INPUT_NAME),
        'production_sha256': {name: sha(BASE / name) for name in FILES}, 'qa_reports': reports,
        'scope': 'X6 inference-cost substage; training epoch costs remain pending',
        'actual_local_model_execution': False, 'local_raw_interaction_or_checkpoint_access': False}
    payload['QA_SUMMARY.json'] = (json.dumps(summary, indent=2) + '\n').encode('utf-8')
    payload['SHA256SUMS.txt'] = ''.join(hashlib.sha256(data).hexdigest() + '  ' + name + '\n' for name, data in sorted(payload.items())).encode('utf-8')
    bundle = BASE / 'X6_Inference_Cost_Toolkit_20261027.zip'
    with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None and len(archive.namelist()) == len(set(archive.namelist())) == len(payload)
        assert all(archive.read(name) == data for name, data in payload.items())
    digest = sha(bundle)
    bundle.with_name(bundle.name + '.sha256.txt').write_text(digest + '  ' + bundle.name + '\n', encoding='ascii')
    report = dict(summary, status='PASS', archive=str(bundle), sha256=digest, members=len(payload), zip_crc='PASS', all_member_bytes='PASS')
    write(BASE / 'X6_INFERENCE_TOOLKIT_DELIVERY_VERIFICATION.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('seal', 'package'))
    args = parser.parse_args()
    seal() if args.action == 'seal' else package()
