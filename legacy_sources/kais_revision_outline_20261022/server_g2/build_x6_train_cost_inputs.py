"""Build an X6 training-cost input seal from audited source/metadata only."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re

BASE = Path(__file__).resolve().parent
OUTLINE = BASE.parent
PROJECT = OUTLINE.parent
NAME = 'x6_train_cost_inputs_20261027.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    x3_input = read(BASE / 'x3_shuffle_inputs_20261025.json')
    shuffle_path = PROJECT / 'g2_x3_shuffle_results_20261026/snapshot/manifests/x3_shuffle_manifest.json'
    shuffle = read(shuffle_path)
    amazon_path = PROJECT / 'g2_lbsp_amazon_train_20261025/snapshot/manifests/lbsp_amazon_train_manifest.json'
    amazon = read(amazon_path)
    ml = read(BASE / 'lbsp_ml1m_stability_inputs_20261023.json')
    root = PurePosixPath(shuffle['root'])
    assert PurePosixPath(amazon['root']) == root
    shuffle_stage = str(PurePosixPath(shuffle['stage']).relative_to(root))
    amazon_stage = str(PurePosixPath(amazon['stage']).relative_to(root))
    source_sha = dict(x3_input['sources_sha256'])
    lbsp_base = 'audit/33_2_train_movielens_r1_seed42.py'
    lbsp_forward = 'step38_kais_retrain/scripts/38_20_train_ml1m_r1_bs128.py'
    source_sha[lbsp_base] = ml['sources_sha256'][lbsp_base]
    source_sha[lbsp_forward] = ml['sources_sha256'][lbsp_forward]
    data_sha, datasets = {}, {}
    for tag in ('ml1m', 'beauty', 'sports', 'toys'):
        source = x3_input['datasets'][tag]
        shuffle_source = shuffle_stage + '/scripts/train_' + tag + '_shuffle.py'
        source_sha[shuffle_source] = shuffle['generated_sha256']['scripts/train_' + tag + '_shuffle.py']
        lbsp_source = lbsp_forward if tag == 'ml1m' else amazon_stage + '/scripts/train_' + tag + '_lbsp.py'
        if tag != 'ml1m':
            source_sha[lbsp_source] = amazon['generated_sha256']['scripts/train_' + tag + '_lbsp.py']
        map_path = source['control_reference']['model_dir'] + '/item_id_map.csv'
        data_sha[source['train_path']] = x3_input['data_sha256'][source['train_path']]
        data_sha[map_path] = source['control_reference']['files_sha256']['item_id_map.csv']
        datasets[tag] = {key: source[key] for key in ('dataset', 'train_path', 'expected_items', 'expected_training_rows', 'expected_training_positions')}
        datasets[tag].update(map_path=map_path, control_source=source['trainer_source'],
                             shuffle_source=shuffle_source, lbsp_training_source=lbsp_source)
    data = {'version': 'x6_train_cost_v1', 'representative_training_seed': 42,
            'sources_sha256': source_sha, 'data_sha256': data_sha, 'datasets': datasets,
            'model_control_source': x3_input['model_source'], 'model_lbsp_source': lbsp_base,
            'model_lbsp_forward_source': lbsp_forward,
            'helper_sha256': {name: sha(BASE / name) for name in ('x6_train_cost_prepare.py', 'x3_inference_adapter.py')},
            'protocol': {'training_batch_size': 128, 'warmup_epochs': [1], 'measured_epochs': [2, 3, 4, 5, 6],
                'methods': ['control', 'shuffle', 'lbsp'], 'ensemble_training': 'shared with control; no independent ensemble training',
                'scope': 'fresh temporary seed42 model and Adam optimizer; full legal-position train-only epoch; no checkpoint load/save, no validation/test',
                'timing': 'GPU synchronize before/after wall perf_counter; includes loader/sampler and shuffle shared event priority setup, H2D, forward, loss, backward, gradient clipping where original and optimizer step; excludes data read/model init/validation/checkpoints',
                'backend': 'one consistent deterministic algorithms policy, TF32 disabled, CUBLAS_WORKSPACE_CONFIG=:4096:8; differs from original nondeterministic defaults where relevant',
                'worker_policy': 'ML/Beauty Control and Shuffle use four persistent workers and clip_norm=5; other methods use original zero-worker seeded-per-epoch sampler and no clip',
                'memory': 'after warmup, per measurement reset peak allocator statistics; report current baseline and absolute/incremental peak allocated and reserved bytes',
                'storage': 'only aggregate metrics, source/provenance and logs; no raw interactions or weights in report'}}
    assert len(source_sha) == 13 and len(data_sha) == 8
    (BASE / NAME).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    digest = sha(BASE / NAME)
    for name in ('run_x6_train_cost.py', 'collect_x6_train_cost.py'):
        path = BASE / name
        if path.exists():
            value, count = re.subn(r"(?m)^INPUT_SHA256 = '[^']+'", "INPUT_SHA256 = '" + digest + "'", path.read_text(encoding='utf-8'))
            assert count == 1
            path.write_text(value, encoding='utf-8')
    print(json.dumps({'status': 'SEALED', 'sha256': digest, 'sources': len(source_sha), 'data': len(data_sha)}))


if __name__ == '__main__':
    build()
