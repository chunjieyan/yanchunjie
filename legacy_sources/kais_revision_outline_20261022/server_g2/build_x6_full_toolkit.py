"""Merge the two verified, metadata-only X6 toolkits without executing them."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import zipfile

BASE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def report(name):
    return json.loads((BASE / name).read_text(encoding='utf-8'))


def checked_bundle(info):
    path = Path(info['archive'])
    assert info['status'] == 'PASS' and sha(path.read_bytes()) == info['sha256']
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert z.testzip() is None and len(names) == len(set(names))
        assert all(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts for name in names)
        values = {name: z.read(name) for name in names}
    entries = {}
    for line in values['SHA256SUMS.txt'].decode('utf-8').splitlines():
        digest, name = line.split('  ', 1)
        assert name not in entries
        entries[name] = digest
    assert set(entries) == set(values) - {'SHA256SUMS.txt'}
    assert all(sha(values[name]) == digest for name, digest in entries.items())
    for name, digest in info['production_sha256'].items():
        assert sha(values['server_files/' + name]) == digest
        assert sha((BASE / name).read_bytes()) == digest
    return values


def main():
    infer = report('X6_INFERENCE_TOOLKIT_DELIVERY_VERIFICATION.json')
    train = report('X6_TRAIN_TOOLKIT_DELIVERY_VERIFICATION_v2.json')
    # Reject stale training packages whose runner/collector predate the
    # required explicit epoch/driver checks and device telemetry.
    train_runner = (BASE / 'run_x6_train_cost.py').read_text(encoding='utf-8')
    train_collector = (BASE / 'collect_x6_train_cost.py').read_text(encoding='utf-8')
    assert 'nvidia-smi' in train_runner
    assert 'fromisoformat' in train_collector and 'command' in train_collector
    infer_data, train_data = checked_bundle(infer), checked_bundle(train)
    payload = {}
    for label, values in [('inference', infer_data), ('training', train_data)]:
        for name, data in values.items():
            if name.startswith('server_files/'):
                if name in payload:
                    assert payload[name] == data, 'Shared tool differs: ' + name
                payload[name] = data
            elif name.startswith('qa/'):
                payload['qa/' + label + '/' + name[3:]] = data
            elif name == 'QA_SUMMARY.json':
                payload['qa/' + label + '/QA_SUMMARY.json'] = data
            elif name.endswith('服务器操作.md'):
                payload['docs/' + label + '.md'] = data
    expected = {'run_x6_inference.py', 'collect_x6_inference.py', 'x3_inference_adapter.py',
        'x3_inference_protocol.py', 'x3_core_helpers.py', 'x6_model_definitions.py', 'x6_cost_protocol.py',
        'x6_inference_inputs_20261027.json', 'run_x6_train_cost.py', 'collect_x6_train_cost.py',
        'x6_train_cost_prepare.py', 'x6_train_cost_inputs_20261027.json'}
    assert {name.removeprefix('server_files/') for name in payload if name.startswith('server_files/')} == expected
    assert not any(name.lower().endswith(('.pt', '.pth', '.ckpt')) for name in payload)
    payload['X6_完整成本_服务器操作.md'] = (BASE / 'X6_完整成本_服务器操作.md').read_bytes()
    summary = {'status': 'SOURCE_AND_SYNTHETIC_QA_PASS_SERVER_MEASUREMENTS_PENDING',
        'scope': 'Four methods/four datasets inference plus three-method training epoch costs; ensemble training shares control',
        'server_files': sorted(expected), 'unique_server_files': len(expected),
        'source_toolkits_sha256': {'inference': infer['sha256'], 'training': train['sha256']},
        'production_sha256': {name.removeprefix('server_files/'): sha(data) for name, data in payload.items() if name.startswith('server_files/')},
        'actual_gpu_measurements_completed': False, 'local_research_model_execution': False,
        'local_real_interactions_or_checkpoint_access': False}
    payload['QA_SUMMARY.json'] = (json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    payload['SHA256SUMS.txt'] = ''.join(sha(data) + '  ' + name + '\n' for name, data in sorted(payload.items())).encode('utf-8')
    output = BASE / 'X6_Full_Cost_Toolkit_20261027.zip'
    if output.exists():
        with zipfile.ZipFile(output) as z:
            assert set(z.namelist()) == set(payload) and all(z.read(n) == d for n, d in payload.items())
    else:
        with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as z:
            for name, data in sorted(payload.items()):
                z.writestr(name, data)
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(payload)
        assert all(z.read(name) == data for name, data in payload.items())
    digest = sha(output.read_bytes())
    output.with_name(output.name + '.sha256.txt').write_text(digest + '  ' + output.name + '\n', encoding='ascii')
    result = dict(summary, status='PASS', archive=str(output), sha256=digest, members=len(payload), zip_crc='PASS',
                  all_member_bytes='PASS', source_ledgers='PASS')
    (BASE / 'X6_FULL_TOOLKIT_DELIVERY_VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
