"""Owned metadata-only build utility. Never imports research code or data."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import re
import shutil
import zipfile

BASE = Path(__file__).resolve().parent
OUTLINE = BASE.parent
PROJECT = OUTLINE.parent
EXECUTION = OUTLINE / 'g2_x3_x4_x6_execution_20261025'
NAME = 'x4_tisasrec_inputs_20261026.json'
CORE = ['run_x4_tisasrec.py', 'collect_x4_tisasrec.py', 'x4_tisasrec_model.py', 'x4_tisasrec_protocol.py', NAME,
        'X4_UPSTREAM_MIT_LICENSE.txt', 'X4_MODEL_PROVENANCE.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def seal():
    model_qa = read(EXECUTION / 'x4_tisasrec_model_source_qa.json')
    assert model_qa['status'] == 'SOURCE_AND_AST_QA_PASS'
    assert sha(BASE / 'x4_tisasrec_model.py') == model_qa['model_sha256']
    license_source = EXECUTION / 'x4_pinned_sources_complete_20261025/recbole_trm/LICENSE'
    assert sha(license_source) == model_qa['source_sha256']['LICENSE']
    shutil.copyfile(license_source, BASE / 'X4_UPSTREAM_MIT_LICENSE.txt')
    provenance = {'upstream': model_qa['upstream_repository'], 'commit': model_qa['upstream_commit'],
                  'license': 'MIT', 'upstream_sha256': model_qa['source_sha256'],
                  'model_sha256': model_qa['model_sha256'], 'source_ast_checks': model_qa['assertions'],
                  'scope': 'Retained RecBole time-aware model math; independent strict-panel prefix-time adaptation. Not reproduction of original sampled-candidate paper results.',
                  'adaptation': model_qa.get('adaptation', {})}
    write(BASE / 'X4_MODEL_PROVENANCE.json', provenance)
    x3 = read(BASE / 'x3_shuffle_inputs_20261025.json')
    ml = read(BASE / 'lbsp_ml1m_stability_inputs_20261023.json')
    amazon = read(BASE / 'lbsp_amazon_stability_inputs_20261025.json')
    with (PROJECT / 'g2_snapshot_20261022/DATA_SHA256.csv').open(encoding='utf-8-sig', newline='') as handle:
        ledger_rows = list(csv.DictReader(handle))
    # This ledger contains paths/hashes, never interaction values.
    full_sha = {}
    for row in ledger_rows:
        relative = row.get('path', row.get('relative_path', ''))
        if relative:
            full_sha[relative] = row['sha256']
    inputs = {'version': 'x4_tisasrec_v1', 'training_seeds': [42, 2026, 3407], 'permutation_seeds': [101, 202, 303, 404, 505],
              'datasets': {}, 'data_sha256': {}, 'helper_sha256': {},
              'hyperparameters': {'hidden_size': 64, 'num_heads': 2, 'num_layers': 2, 'dropout': 0.2, 'inner_size': 256,
                  'activation': 'gelu', 'max_len': 50, 'training_batch_size': 128, 'evaluation_batch_size': 256,
                  'optimizer': 'Adam', 'learning_rate': 0.001, 'weight_decay': 0.0, 'clip_grad_norm': 5.0,
                  'max_epochs': 200, 'patience': 15, 'validation_metric': 'ndcg10', 'time_span': 256,
                  'loss': 'full_item_cross_entropy_excluding_padding0', 'loader_workers': 4},
              'protocol': {'model_source': provenance['upstream'], 'commit': provenance['commit'],
                  'time_features': 'visible last50 prefix-only native bucket offsets; scale=min positive unique gap or1; exact integer abs difference//scale clipped256; padding pairs0',
                  'padding': 'valid-first right-padding; final valid length-1 gather; differs from Control right-aligned positions',
                  'target_time': 'validation of strict history identity/chronology only, never model feature',
                  'training': 'all legal next-item positions, chronological native bucket/source-row tie order, no new split/remapping',
                  'evaluation': 'sealed canonical and5existing permutation CSVs; full candidate seen-minus-target mask; padding0 excluded; strict-greater rank, independent torch.topk10',
                  'affected': 'distinct raw item histories across five perm panels>=2; original excluded; fixed across methods',
                  'stability': 'all10unordered pairs/user then user mean overlap; any hit/nonhit change/user; training-seed variation separate',
                  'time_zero_and_position_zero': 'preserve upstream padding_idx0 and normal initialization behavior',
                  'selection': 'validation NDCG@10 only, strict > improvement, patience15, no test/seed selection',
                  'scope': 'controlled external-backbone adaptation; no claim of original-paper metric reproduction or isolated time-matrix causal effect'}}
    for tag, other, affected, trainable, unit in [('ml1m', ml, 163, 5400, 'Unix second'),
                                                 ('beauty', amazon, 922, 20223, 'Unix day (timestamp // 86400)')]:
        source = x3['datasets'][tag]
        perm = other['datasets'][tag]['permutation_paths']
        event = 'processed/diagnostic_a/' + source['dataset'] + '/file_order.csv'
        map_path = source['control_reference']['model_dir'] + '/item_id_map.csv'
        config_path = source['control_reference']['model_dir'] + '/config.json'
        dataset = {key: source[key] for key in ('dataset', 'train_path', 'validation_path', 'test_path', 'expected_items',
                     'expected_validation', 'expected_test', 'expected_training_positions', 'expected_training_rows')}
        dataset.update(event_source_path=event, map_path=map_path, control_config_path=config_path,
                       permutation_paths=perm, expected_affected=affected, expected_trainable_users=trainable, native_time_unit=unit)
        hashes = {source[key]: x3['data_sha256'][source[key]] for key in ('train_path', 'validation_path', 'test_path')}
        hashes.update({path: other['data_sha256'][path] for path in perm.values()})
        hashes[event] = full_sha[event]
        hashes[map_path] = source['control_reference']['files_sha256']['item_id_map.csv']
        hashes[config_path] = source['control_reference']['files_sha256']['config.json']
        dataset['required_files'] = sorted(hashes)
        inputs['datasets'][tag] = dataset
        inputs['data_sha256'].update(hashes)
    for name in ('x4_tisasrec_model.py', 'x4_tisasrec_protocol.py', 'X4_UPSTREAM_MIT_LICENSE.txt', 'X4_MODEL_PROVENANCE.json'):
        inputs['helper_sha256'][name] = sha(BASE / name)
    write(BASE / NAME, inputs)
    seal_sha = sha(BASE / NAME)
    for name in ('run_x4_tisasrec.py', 'collect_x4_tisasrec.py'):
        path = BASE / name
        if path.exists():
            source = path.read_text(encoding='utf-8')
            source, count = re.subn(r"(?m)^INPUT_SHA256 = '[^']+'", "INPUT_SHA256 = '" + seal_sha + "'", source)
            assert count == 1, name
            path.write_text(source, encoding='utf-8')
    print(json.dumps({'status': 'INPUT_SEALED', 'input_sha256': seal_sha, 'root_input_files': len(inputs['data_sha256'])}))


def package():
    inputs = read(BASE / NAME)
    for name, expected in inputs['helper_sha256'].items():
        assert sha(BASE / name) == expected, name
    payload = {}
    for name in CORE:
        source = (BASE / name).read_bytes()
        if name.endswith('.py'):
            ast.parse(source.decode('utf-8'), feature_version=(3, 10))
        if name in ('run_x4_tisasrec.py', 'collect_x4_tisasrec.py'):
            assert ("INPUT_SHA256 = '" + sha(BASE / NAME) + "'") in source.decode('utf-8')
        payload['server_files/' + name] = source
    payload['X4_TiSASRec_服务器操作.md'] = (BASE / 'X4_TiSASRec_服务器操作.md').read_bytes()
    qa_names = ['x4_tisasrec_model_source_qa.json', 'x4_tisasrec_protocol_qa.json',
                'x4_runner_independent_qa.json', 'x4_collector_qa.json', 'x4_preparation_workflow_qa.json']
    qa_summary = {}
    for name in qa_names:
        path = EXECUTION / name
        value = read(path)
        assert 'PASS' in value['status'], (name, value['status'])
        if name == 'x4_tisasrec_model_source_qa.json':
            assert value['model_sha256'] == sha(BASE / 'x4_tisasrec_model.py')
        if name == 'x4_tisasrec_protocol_qa.json':
            assert value['production_sha256'] == sha(BASE / 'x4_tisasrec_protocol.py')
        if name == 'x4_collector_qa.json':
            assert value['collector_sha256'] == sha(BASE / 'collect_x4_tisasrec.py')
        if name in ('x4_runner_independent_qa.json', 'x4_preparation_workflow_qa.json'):
            assert value['runner_sha256'] == sha(BASE / 'run_x4_tisasrec.py')
        if name == 'x4_preparation_workflow_qa.json':
            assert value['production_files_sha256'] == {name: sha(BASE / name) for name in CORE}
        qa_summary[name] = {'sha256': sha(path), 'status': value['status']}
        payload['qa/' + name] = path.read_bytes()
    for name in ('x4_data_time_contract_review.md', 'x4_tisasrec_model_source_qa.md'):
        payload['qa/' + name] = (EXECUTION / name).read_bytes()
    payload['QA_SUMMARY.json'] = (json.dumps({'status': 'LOCAL_QA_PASS_SERVER_SMOKES_REQUIRED',
        'input_sha256': sha(BASE / NAME), 'production_sha256': {name: sha(BASE / name) for name in CORE},
        'qa_reports': qa_summary, 'actual_server_models_trained': 0, 'local_models_executed': False,
        'real_interactions_or_weights_accessed_locally': False}, indent=2) + '\n').encode('utf-8')
    payload['SHA256SUMS.txt'] = ''.join(hashlib.sha256(data).hexdigest() + '  ' + name + '\n'
        for name, data in sorted(payload.items())).encode('utf-8')
    bundle = BASE / 'X4_TiSASRec_Training_Toolkit_20261026.zip'
    if bundle.exists():
        with zipfile.ZipFile(bundle) as archive:
            assert set(archive.namelist()) == set(payload)
            assert all(archive.read(name) == data for name, data in payload.items())
    else:
        with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(payload.items()):
                archive.writestr(name, data)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(payload)
        assert all(archive.read(name) == data for name, data in payload.items())
    value = sha(bundle)
    bundle.with_name(bundle.name + '.sha256.txt').write_text(value + '  ' + bundle.name + '\n', encoding='ascii')
    report = {'status': 'PASS', 'archive': str(bundle), 'sha256': value, 'members': len(payload),
              'zip_crc': 'PASS', 'all_member_bytes_and_ledger': 'PASS',
              'production_sha256': {name: sha(BASE / name) for name in CORE}, 'qa_reports': qa_summary}
    write(BASE / 'X4_TOOLKIT_DELIVERY_VERIFICATION.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('seal', 'package'))
    args = parser.parse_args()
    seal() if args.action == 'seal' else package()
