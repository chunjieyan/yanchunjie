"""Owned standard-library build utility; source/metadata only, no model/data IO."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
import zipfile

BASE = Path(__file__).resolve().parent
OUTLINE = BASE.parent
PROJECT = OUTLINE.parent
REVIEW = OUTLINE / 'g2_x3_x4_x6_execution_20261025'
INPUT_NAME = 'x3_inference_inputs_20261026.json'
FILES = ('run_x3_inference.py', 'collect_x3_inference.py', 'x3_inference_adapter.py',
         'x3_inference_protocol.py', 'x3_control_model.py', INPUT_NAME)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def module(path):
    spec = importlib.util.spec_from_file_location('owned_x3_adapter_build', path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def seal():
    control = read(BASE / 'control_stability_inputs_20261023.json')
    core_snapshot = PROJECT / 'g2_control_stability_20261023/snapshot'
    control_manifest = read(core_snapshot / 'manifests/control_stability_manifest.json')
    snapshot = PROJECT / 'g2_x3_shuffle_results_20261026/snapshot'
    shuffle_manifest_path = snapshot / 'manifests/x3_shuffle_manifest.json'
    shuffle = read(shuffle_manifest_path)
    server_root = PurePosixPath(shuffle['root'])
    shuffle_stage = str(PurePosixPath(shuffle['stage']).relative_to(server_root))
    control_stage = str(PurePosixPath(control_manifest['stage']).relative_to(server_root))
    source = snapshot / 'scripts/model_definition.py'
    assert sha(source) == shuffle['generated_sha256']['scripts/model_definition.py']
    adapter_qa = read(REVIEW / 'x3_inference_adapter_qa.json')
    # Only our owned AST generator executes. Original research definitions are data.
    generated = module(BASE / 'x3_inference_adapter.py').build_frozen_module(source.read_text(encoding='utf-8'))
    ast.parse(generated, feature_version=(3, 10))
    assert hashlib.sha256(generated.encode()).hexdigest() == '04d049bf6a006bb765b9e93fb3d2bdf771f3a88eacf434d667a260c2e7a4b94a'
    (BASE / 'x3_control_model.py').write_text(generated, encoding='utf-8')
    groups = [[101, 202, 303, 404, 505], [1101, 1202, 1303, 1404, 1505], [2101, 2202, 2303, 2404, 2505],
              [3101, 3202, 3303, 3404, 3505], [4101, 4202, 4303, 4404, 4505]]
    global_pins = {shuffle_stage + '/manifests/x3_shuffle_manifest.json': sha(shuffle_manifest_path),
                   shuffle_stage + '/scripts/model_definition.py': sha(source),
                   control_stage + '/manifests/control_stability_manifest.json': sha(core_snapshot / 'manifests/control_stability_manifest.json')}
    result = {'version': 'x3_inference_v1', 'training_seeds': [42, 2026, 3407], 'groups': groups,
              'datasets': {}, 'references': {}, 'root_files_sha256': dict(global_pins), 'helper_sha256': {},
              'protocol': {'batch_size': 256, 'model': 'frozen Control class; shuffle identical architecture; no model updates',
                  'padding': 'original left padding/right-aligned absolute positions and predict_batch_masked',
                  'candidate_mask': 'full frozen catalog; exclude index0 and unique history items except target',
                  'ranking': 'strict-greater target rank; torch.topk10 independent; preserve tie diagnostics',
                  'group_members': groups, 'first_group': 'reuse sealed PCG64 panels and reproduce all item/event inputs with server NumPy before new groups',
                  'new_groups': 'SHA256(dataset|sample_id|seed) first8bytes little endian -> NumPy default_rng PCG64; jointly permute last-bucket items/raw_lines',
                  'score_mean': 'raw float32 candidate logits -> fixed ascending-seed sequential float64 sum/5 -> common candidate mask -> rank/Top10',
                  'affected': 'five first-group raw item histories distinct>=2, canonical excluded; shared across all methods',
                  'stability': 'five shuffle permutations or five ensemble group results; all10pairs/user average Top10 overlap; user any rank<=10 hitflip',
                  'accuracy': 'shuffle canonical; ensemble arithmeticmean of five group metrics within training seed, then3seed mean/sampleSD',
                  'cache': 'disabled; repeat actual histories retained with full multiplicity; actual and logical forward rows recorded',
                  'score_probe': 'at most first16canonical samples/group, target+Top10 candidates only; bounded arithmetic check, not full-score replay',
                  'training': 'none; original24Control/LBSP and12Shuffle weights remain readonly',
                  'X6': 'not measured here; one ensemble recommendation uses5base views, not the25views of five-group stability'}}
    for tag, count in [('ml1m', 163), ('beauty', 922), ('sports', 1624), ('toys', 842)]:
        old = control['datasets'][tag]
        x3plan = shuffle['inputs']['datasets'][tag]
        paths = [old['original_path']] + list(old['permutation_paths'].values())
        plan = {'dataset': old['dataset'], 'expected_samples': old['expected_samples'],
                'expected_items': x3plan['expected_items'], 'expected_affected': count,
                'original_path': old['original_path'], 'permutation_paths': old['permutation_paths'],
                'runs': {'shuffle': {}, 'ensemble': {}}}
        result['root_files_sha256'].update({path: control['data_sha256'][path] for path in paths})
        for method in ('shuffle', 'ensemble'):
            for seed in (42, 2026, 3407):
                if method == 'shuffle':
                    completion_path = snapshot / ('verification/completed_' + tag + '_seed' + str(seed) + '.json')
                    completed = read(completion_path)
                    model_dir = str(PurePosixPath(completed['output_dir']).relative_to(server_root))
                    hashes = completed['output_metadata_sha256']
                    best_epoch = completed['best_epoch']
                    extra = {shuffle_stage + '/verification/' + completion_path.name: sha(completion_path)}
                else:
                    prior = old['runs'][str(seed)]
                    model_dir, hashes, best_epoch = prior['model_dir'], prior['files_sha256'], prior['best_epoch']
                    extra = {}
                run = {'model_dir': model_dir, 'checkpoint': 'matched_control_sasrec_best.pt', 'best_epoch': best_epoch,
                       'reference_permutations': {}, 'checkpoint_identity': 'SASRecControl/dataset/seed/epoch/architecture + pinned config/provenance + exact canonical reproduction'}
                root_paths = dict(extra)
                for original_name, staged_name, field in [('config.json', 'config.json', 'reference_config'),
                        ('item_id_map.csv', 'item_id_map.csv', 'reference_map'),
                        ('test_per_sample_predictions.csv', 'predictions_saved.csv', 'reference_predictions')]:
                    relative = 'references/' + method + '/' + tag + '_seed' + str(seed) + '/' + staged_name
                    root_path = model_dir + '/' + original_name
                    result['references'][relative] = {'root_path': root_path, 'sha256': hashes[original_name]}
                    root_paths[root_path] = hashes[original_name]
                    run[field] = relative
                if method == 'ensemble':
                    for view in groups[0]:
                        name = 'predictions_perm_seed' + str(view) + '.csv'
                        root_path = control_stage + '/outputs/' + tag + '_seed' + str(seed) + '/' + name
                        relative = 'references/ensemble/' + tag + '_seed' + str(seed) + '/' + name
                        digest = sha(core_snapshot / 'outputs' / (tag + '_seed' + str(seed)) / name)
                        result['references'][relative] = {'root_path': root_path, 'sha256': digest}
                        root_paths[root_path] = digest
                        run['reference_permutations'][str(view)] = relative
                result['root_files_sha256'].update(root_paths)
                run['required_root_files'] = sorted(set(paths) | set(root_paths) | set(global_pins))
                plan['runs'][method][str(seed)] = run
        result['datasets'][tag] = plan
    for name in ('x3_inference_adapter.py', 'x3_inference_protocol.py', 'x3_control_model.py'):
        result['helper_sha256'][name] = sha(BASE / name)
    write(BASE / INPUT_NAME, result)
    value = sha(BASE / INPUT_NAME)
    for name in ('run_x3_inference.py', 'collect_x3_inference.py'):
        path = BASE / name
        if path.exists():
            source = path.read_text(encoding='utf-8')
            source, count = re.subn(r"(?m)^INPUT_SHA256 = '[^']+'", "INPUT_SHA256 = '" + value + "'", source)
            assert count == 1, name
            path.write_text(source, encoding='utf-8')
    print(json.dumps({'status': 'SEALED', 'input_sha256': value, 'root_files': len(result['root_files_sha256']), 'references': len(result['references'])}))


def package():
    inputs = read(BASE / INPUT_NAME)
    for name, expected in inputs['helper_sha256'].items():
        assert sha(BASE / name) == expected
    payload = {}
    for name in FILES:
        data = (BASE / name).read_bytes()
        if name.endswith('.py'):
            ast.parse(data.decode('utf-8'), feature_version=(3, 10))
        if name in ('run_x3_inference.py', 'collect_x3_inference.py'):
            assert ("INPUT_SHA256 = '" + sha(BASE / INPUT_NAME) + "'") in data.decode('utf-8')
        payload['server_files/' + name] = data
    payload['X3_重排评价与分数集成_服务器操作.md'] = (BASE / 'X3_重排评价与分数集成_服务器操作.md').read_bytes()
    reports = {}
    for name in ('x3_inference_adapter_qa.json', 'x3_inference_protocol_qa.json', 'x3_inference_runner_qa.json', 'x3_inference_collector_qa.json'):
        path = REVIEW / name
        value = read(path)
        assert 'PASS' in value['status'], name
        if name == 'x3_inference_adapter_qa.json':
            assert value['adapter_sha256'] == sha(BASE / 'x3_inference_adapter.py')
        if name == 'x3_inference_protocol_qa.json':
            assert value['production_sha256'] == sha(BASE / 'x3_inference_protocol.py')
        if name == 'x3_inference_runner_qa.json':
            assert value['runner_sha256'] == sha(BASE / 'run_x3_inference.py')
            assert value['production_inputs_sha256'] == sha(BASE / INPUT_NAME)
        if name == 'x3_inference_collector_qa.json':
            assert value['collector_sha256'] == sha(BASE / 'collect_x3_inference.py')
        reports[name] = {'status': value['status'], 'sha256': sha(path)}
        payload['qa/' + name] = path.read_bytes()
    for name in ('x3_inference_adapter_qa.md', 'X3_五组集成与X6计时协议.md'):
        payload['qa/' + name] = (REVIEW / name).read_bytes()
    summary = {'status': 'LOCAL_SYNTHETIC_QA_PASS_SERVER_INFERENCE_PENDING', 'input_sha256': sha(BASE / INPUT_NAME),
               'production_sha256': {name: sha(BASE / name) for name in FILES}, 'qa_reports': reports,
               'new_training_runs': 0, 'local_model_execution': False, 'actual_interaction_or_checkpoint_access_locally': False}
    payload['QA_SUMMARY.json'] = (json.dumps(summary, indent=2) + '\n').encode('utf-8')
    payload['SHA256SUMS.txt'] = ''.join(hashlib.sha256(data).hexdigest() + '  ' + name + '\n' for name, data in sorted(payload.items())).encode('utf-8')
    bundle = BASE / 'X3_Inference_Ensemble_Toolkit_20261026.zip'
    with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(payload)
        assert all(archive.read(name) == data for name, data in payload.items())
    value = sha(bundle)
    bundle.with_name(bundle.name + '.sha256.txt').write_text(value + '  ' + bundle.name + '\n', encoding='ascii')
    report = dict(summary, status='PASS', archive=str(bundle), sha256=value, members=len(payload), zip_crc='PASS', all_member_bytes='PASS')
    write(BASE / 'X3_INFERENCE_TOOLKIT_DELIVERY_VERIFICATION.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('seal', 'package'))
    args = parser.parse_args()
    seal() if args.action == 'seal' else package()
