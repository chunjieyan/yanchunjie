"""Package frozen train-cost tools and existing synthetic QA evidence."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

BASE = Path(__file__).resolve().parent
OUTLINE = BASE.parent
REVIEW = OUTLINE / 'g2_x3_x4_x6_execution_20261025'
FILES = ('run_x6_train_cost.py', 'collect_x6_train_cost.py', 'x6_train_cost_prepare.py',
         'x3_inference_adapter.py', 'x6_train_cost_inputs_20261027.json')
INPUT_NAME = FILES[-1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    inputs = read(BASE / INPUT_NAME)
    assert inputs['version'] == 'x6_train_cost_v1'
    for name, expected in inputs['helper_sha256'].items():
        assert sha(BASE / name) == expected, name
    data = {}
    for name in FILES:
        contents = (BASE / name).read_bytes()
        if name.endswith('.py'):
            ast.parse(contents.decode('utf-8'), feature_version=(3, 10))
        if name in ('run_x6_train_cost.py', 'collect_x6_train_cost.py'):
            assert ("INPUT_SHA256 = '" + sha(BASE / INPUT_NAME) + "'") in contents.decode('utf-8')
        data['server_files/' + name] = contents
    data['X6_训练每轮成本_服务器操作.md'] = (BASE / 'X6_训练每轮成本_服务器操作.md').read_bytes()
    qa = {}
    for name in ('x6_train_cost_prepare_qa.json', 'x6_train_cost_workflow_qa.json', 'x3_inference_adapter_qa.json'):
        path = REVIEW / name
        value = read(path)
        assert value['status'] == 'PASS'
        if name == 'x6_train_cost_prepare_qa.json':
            assert value['generator_sha256'] == sha(BASE / 'x6_train_cost_prepare.py')
        elif name == 'x6_train_cost_workflow_qa.json':
            assert value['runner_sha256'] == sha(BASE / 'run_x6_train_cost.py')
            assert value['collector_sha256'] == sha(BASE / 'collect_x6_train_cost.py')
            assert value['generator_sha256'] == sha(BASE / 'x6_train_cost_prepare.py')
            assert value['production_input_sha256'] == sha(BASE / INPUT_NAME)
        else:
            assert value['adapter_sha256'] == sha(BASE / 'x3_inference_adapter.py')
        qa[name] = {'status': value['status'], 'sha256': sha(path)}
        data['qa/' + name] = path.read_bytes()
    summary = {'status': 'LOCAL_SYNTHETIC_QA_PASS_SERVER_MEASUREMENTS_PENDING',
               'input_sha256': sha(BASE / INPUT_NAME), 'production_sha256': {name: sha(BASE / name) for name in FILES},
               'qa_reports': qa, 'scope': 'X6 temporary-model training epoch costs, excludes validation/checkpoint IO',
               'actual_local_research_model_execution': False, 'local_real_interaction_or_checkpoint_access': False}
    data['QA_SUMMARY.json'] = (json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    data['SHA256SUMS.txt'] = ''.join(hashlib.sha256(blob).hexdigest() + '  ' + name + '\n' for name, blob in sorted(data.items())).encode('utf-8')
    output = BASE / 'X6_Training_Epoch_Cost_Toolkit_20261027_v2.zip'
    if output.exists():
        with zipfile.ZipFile(output) as z:
            assert set(z.namelist()) == set(data)
            assert all(z.read(name) == blob for name, blob in data.items())
    else:
        with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as z:
            for name, blob in sorted(data.items()):
                z.writestr(name, blob)
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(data)
        assert all(z.read(name) == blob for name, blob in data.items())
    digest = sha(output)
    output.with_name(output.name + '.sha256.txt').write_text(digest + '  ' + output.name + '\n', encoding='ascii')
    result = dict(summary, status='PASS', archive=str(output), sha256=digest, members=len(data),
                  zip_crc='PASS', exact_member_bytes='PASS', ledger_files=len(data)-1)
    (BASE / 'X6_TRAIN_TOOLKIT_DELIVERY_VERIFICATION_v2.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
