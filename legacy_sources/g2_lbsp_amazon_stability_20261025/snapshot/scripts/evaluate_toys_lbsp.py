#!/usr/bin/env python3
# ML1M LBSP fixed-model inference only; no training.
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath

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
    if file_signature(before) != file_signature(after) or file_signature(before) != file_signature(checked_file(root, relative).stat()):
        raise RuntimeError('Input changed during hashing: ' + relative)
    if digest.hexdigest() != expected:
        raise RuntimeError('SHA256 mismatch: ' + relative)
    return digest.hexdigest()


def row_sequence(row):
    values = json.loads(row['sequence_items'])
    if not isinstance(values, list) or not values or any(isinstance(v, (dict, list, bool)) or v is None for v in values):
        raise ValueError('Invalid input sequence')
    return tuple(str(v) for v in values)


def last_bucket_size(row):
    present = [row[k] for k in ('last_bucket_size', 'last_history_bucket_size') if k in row]
    if not present or any(not re.fullmatch(r'[0-9]+', str(v)) for v in present):
        raise ValueError('Missing/invalid last-bucket size')
    sizes = [int(v) for v in present]
    if len(set(sizes)) != 1 or not 1 <= sizes[0] <= len(row_sequence(row)):
        raise ValueError('Last-bucket size outside history or inconsistent')
    return sizes[0]


def indexed_rows(records):
    result = {}
    for row in records:
        sid = str(row['sample_id'])
        if not sid or sid in result or not str(row['user_id']) or not str(row['target_item']):
            raise ValueError('Duplicate/empty sample identity')
        sequence = row_sequence(row)
        if len(sequence) > 50 or str(row['target_item']) in sequence:
            raise ValueError('Invalid history length or target in history')
        last_bucket_size(row)
        result[sid] = row
    if not result:
        raise ValueError('Empty input panel')
    return result


def verify_permutation(original, records):
    variant = indexed_rows(records)
    if set(variant) != set(original):
        raise ValueError('Permutation sample IDs differ')
    for sid, base in original.items():
        row = variant[sid]
        if any(str(row[k]) != str(base[k]) for k in ('user_id', 'target_item')):
            raise ValueError('Permutation user/target changed: ' + sid)
        for key in ('target_raw_line', 'target_timestamp', 'target_bucket'):
            if key in base and key in row and str(base[key]) != str(row[key]):
                raise ValueError('Permutation target metadata changed: ' + sid)
        a, b = row_sequence(base), row_sequence(row)
        k = last_bucket_size(base)
        if last_bucket_size(row) != k or len(a) != len(b) or a[:-k] != b[:-k] or Counter(a[-k:]) != Counter(b[-k:]):
            raise ValueError('Permutation prefix/length/last-bucket multiset changed: ' + sid)
    return variant


def sequence_sha256(sequence):
    return hashlib.sha256(json.dumps(sequence, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


def panel_labels(original, variants):
    if set(variants) != {str(seed) for seed in PERMUTATIONS}:
        raise ValueError('Exactly five permutation panels are required')
    labels = []
    for sid, row in original.items():
        sequences = [row_sequence(variants[str(seed)][sid]) for seed in PERMUTATIONS]
        distinct = len(set(sequences))
        label = {'sample_id': sid, 'user_id': str(row['user_id']), 'target_item': str(row['target_item']),
                 'last_bucket_size': last_bucket_size(row), 'affected': int(distinct >= 2),
                 'distinct_sequences_5perm': distinct, 'original_sequence_sha256': sequence_sha256(row_sequence(row))}
        label.update({'perm_seed' + str(seed) + '_sequence_sha256': sequence_sha256(seq)
                      for seed, seq in zip(PERMUTATIONS, sequences)})
        labels.append(label)
    return labels


def verify_original_top10(new_records, saved_records):
    saved = {}
    for row in saved_records:
        sid = str(row['sample_id'])
        if sid in saved:
            raise ValueError('Duplicate saved prediction sample')
        saved[sid] = row
    if set(saved) != {str(r['sample_id']) for r in new_records} or len(saved) != len(new_records):
        raise ValueError('Saved/base prediction sample panel differs')
    mismatches = 0
    for row in new_records:
        prior = saved[str(row['sample_id'])]
        if any(str(row[k]) != str(prior[k]) for k in ('user_id', 'target_item')):
            raise ValueError('Saved/base user or target differs')
        a, b = json.loads(row['top10_item_idx']), json.loads(prior['top10_item_idx'])
        ra = [str(x) for x in json.loads(row['top10_raw_item_id'])]
        rb = [str(x) for x in json.loads(prior['top10_raw_item_id'])]
        if len(a) != 10 or len(b) != 10 or len(set(a)) != 10 or len(set(b)) != 10:
            raise ValueError('Invalid base/saved Top10')
        mismatches += int(a != b or ra != rb)
    if mismatches:
        raise RuntimeError('Saved Top10 exact reproduction failed: ' + str(mismatches))
    return mismatches


def control_labels_match(labels, path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        records = list(csv.DictReader(handle))
    expected = {str(r['sample_id']): {k: str(v) for k, v in r.items()} for r in labels}
    actual = {}
    for row in records:
        sid = row['sample_id']
        if sid in actual:
            raise ValueError('Duplicate Control population sample')
        actual[sid] = row
    if actual != expected:
        raise ValueError('Raw-history labels differ from the reviewed Control population')
    return True


def normalize_eval_records(records):
    """Only add missing aliases; reject inconsistent existing fields."""
    result = []
    for record in records:
        row = dict(record)
        present = [row[k] for k in ('last_bucket_size', 'last_history_bucket_size') if k in row]
        if not present or any(not re.fullmatch(r'[0-9]+', str(v)) for v in present):
            raise ValueError('Missing/invalid last-bucket size')
        sizes = [int(v) for v in present]
        history = json.loads(row['sequence_items'])
        if (not isinstance(history, list) or not 1 <= len(history) <= 50
                or len(set(sizes)) != 1 or not 1 <= sizes[0] <= len(history)):
            raise ValueError('Invalid history or inconsistent last-bucket size')
        expected_pooling = int(sizes[0] >= 2)
        if 'direct_pooling' in row and (not re.fullmatch(r'[01]', str(row['direct_pooling']))
                or int(row['direct_pooling']) != expected_pooling):
            raise ValueError('Existing direct_pooling conflicts with last-bucket size')
        if 'last_bucket_size' not in row:
            row['last_bucket_size'] = sizes[0]
        if 'direct_pooling' not in row:
            row['direct_pooling'] = expected_pooling
        result.append(row)
    return result


def normalize_eval_frame(df):
    normalized = normalize_eval_records(df.to_dict(orient='records'))
    df = df.copy()
    for name in ('last_bucket_size', 'direct_pooling'):
        if name not in df.columns:
            df[name] = [row[name] for row in normalized]
    return df


def validate_training_binding(inputs, training_manifest, root):
    reference = inputs['training_reference']
    if (training_manifest['version'] != 'lbsp_amazon_train_v1'
            or training_manifest['root'] != str(root) or training_manifest['stage'] != str(root / reference['stage'])
            or training_manifest['inputs_sha256'] != reference['input_manifest_sha256']):
        raise ValueError('Training stage/manifest binding differs')
    for source in (inputs['model_source'], inputs['forward_source']):
        if training_manifest['sources_sha256'].get(source) != inputs['sources_sha256'][source]:
            raise ValueError('Shared training model/forward source differs')
    for tag, plan in inputs['datasets'].items():
        original = training_manifest['inputs']['datasets'][tag]
        if (original['dataset'] != plan['dataset'] or original['expected_items'] != plan['expected_items']
                or original['expected_test'] != plan['expected_samples'] or original['test_path'] != plan['original_path']
                or training_manifest['data_sha256'].get(plan['original_path']) != inputs['data_sha256'][plan['original_path']]):
            raise ValueError('Original test input differs from audited training: ' + tag)
        for seed, run in plan['runs'].items():
            if run['model_dir'] != reference['stage'] + '/outputs/' + tag + '_lbsp_seed' + seed:
                raise ValueError('Checkpoint run path differs from audited training')
    return True


def validate_run_metadata(inputs, training_manifest, tag, seed, config, metrics, completion, root):
    plan = inputs['datasets'][tag]
    run = plan['runs'][str(seed)]
    original = training_manifest['inputs']['datasets'][tag]
    if (config.get('dataset') != plan['dataset'] or config.get('seed') != seed
            or config.get('num_items') != plan['expected_items']
            or config.get('test_path') != str(root / plan['original_path'])
            or config.get('legal_training_positions') != original['expected_positions']
            or any(config.get(k) != v for k, v in FIXED_CONFIG.items())):
        raise ValueError('Audited training config identity/hyperparameters differ')
    provenance = config['amazon_lbsp_training']
    reference = inputs['training_reference']
    if (provenance.get('input_manifest_sha256') != reference['input_manifest_sha256']
            or provenance.get('preparation_manifest_sha256') != reference['manifest_sha256']
            or provenance.get('generated_script_sha256') != training_manifest['generated_sha256']['scripts/train_' + tag + '_lbsp.py']
            or provenance.get('model_definition_sha256') != training_manifest['generated_sha256']['scripts/model_definition.py']
            or provenance.get('sources_sha256') != training_manifest['sources_sha256']
            or provenance.get('data_sha256', {}).get(plan['original_path']) != inputs['data_sha256'][plan['original_path']]
            or any(provenance.get(k) is not True for k in ('scientific_definitions_unchanged', 'formal_rng_reset_preserved', 'smoke_execution_omitted'))):
        raise ValueError('Audited training config provenance differs')
    if (metrics.get('dataset') != plan['dataset'] or metrics.get('seed') != seed
            or metrics.get('model') != MODEL_NAME or metrics.get('method') != FIXED_CONFIG['method']
            or metrics.get('best_epoch') != run['best_epoch'] or metrics['test']['samples'] != plan['expected_samples']
            or not math.isclose(metrics['test']['ndcg10'], run['test_ndcg10'], rel_tol=0, abs_tol=1e-12)):
        raise ValueError('Saved metrics differ from the audited checkpoint run')
    if (completion.get('status') != 'COMPLETED' or completion.get('tag') != tag or completion.get('dataset') != plan['dataset']
            or completion.get('seed') != seed or completion.get('best_epoch') != run['best_epoch']
            or completion.get('output_dir') != str(root / run['model_dir'])
            or completion.get('checkpoint') != run['checkpoint'] or completion.get('output_metadata_sha256') != run['files_sha256']
            or completion.get('all_original_hashes_verified') is not True or completion.get('originals_unchanged') is not True
            or any(completion.get(k) != v for k, v in provenance.items())):
        raise ValueError('Completed training record differs from saved metadata')
    return True


MODEL_NAME = 'RecentBucketSetR1SASRec'
FIXED_CONFIG = {'model': 'RecentBucketSetR1SASRec', 'method': 'RecentBucketSet_R1_V1_mean_pooling', 'training_batch_size': 128, 'evaluation_batch_size': 256, 'optimizer': 'Adam', 'learning_rate': 0.001, 'max_epochs': 200, 'patience': 15, 'max_len': 50, 'hidden_size': 64, 'num_heads': 2, 'num_blocks': 2, 'dropout': 0.2, 'sampling': 'all_legal_next_item_positions_reshuffled_every_epoch', 'loss': 'full_item_cross_entropy', 'pooling': 'last_history_bucket_only_mean', 'canonicalization': 'sort_last_bucket_by_item_idx'}
PERMUTATIONS = (101, 202, 303, 404, 505)
parser = argparse.ArgumentParser(description='ML1M LBSP fixed-model five-permutation evaluation')
parser.add_argument('--seed', type=int, choices=[42, 2026, 3407], required=True)
args = parser.parse_args()
_LBSP_ROOT = plain(Path('/root/autodl-tmp/time_uncertainty_rec'))
_LBSP_STAGE = plain(Path('/root/autodl-tmp/time_uncertainty_rec/kais_lbsp_amazon_stability_v1_20261025T052741Z_38b2c8952b7d'))
if Path(__file__).resolve(strict=False).parent != _LBSP_STAGE / 'scripts':
    raise RuntimeError('Use the original generated evaluator stage')
_PLAN = {'dataset': 'Amazon_Toys', 'expected_samples': 2547, 'original_path': 'processed/strict_diagnostic_a/Amazon_Toys/file_order/test.csv', 'permutation_paths': {'101': 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed101/test.csv', '202': 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed202/test.csv', '303': 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed303/test.csv', '404': 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed404/test.csv', '505': 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed505/test.csv'}, 'expected_items': 11924, 'eval_source': 'step38_kais_retrain/scripts/38_22_eval_ml1m_r1_stability.py', 'runs': {'42': {'model_dir': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d/outputs/toys_lbsp_seed42', 'files_sha256': {'config.json': '8a7c296f1623752506781279ad0a1357c77162faf31d1b5534e6da2d6650b17f', 'metrics.json': '295d407a4568810173e0db3f81533e15c48761ea29baca7f6eb902c58a44d414', 'training_log.csv': '7163fc161a81387f1d58c1693096b17569c471438f6bb1c418fe1aaab1481623', 'test_per_sample_predictions.csv': '6fa2b33f623626ad8f140f4629b3203e9f349965df6ae567131054b6ae516a81', 'item_id_map.csv': 'd6e1135f71ad9749caa6db9743045e1e65860bd62df6f42e0b76642c08645c60'}, 'best_epoch': 5, 'test_ndcg10': 0.003505244168310362, 'checkpoint': 'r1_meanpool_best.pt'}, '2026': {'model_dir': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d/outputs/toys_lbsp_seed2026', 'files_sha256': {'config.json': 'b5aa136886e17fe2fbf407a9cb0194f31c95270607344bf0e0e4944588c12980', 'metrics.json': 'f64959f92648d31f57d46284101fe665a9bdcb1c0e3ae2b938eee0e36c249e5a', 'training_log.csv': 'f64836deec69fb88727c713ecf28ef0a59619c121c0a74588ca014edc0dfde11', 'test_per_sample_predictions.csv': 'f8c0db82a532bba9a4db43b2ea51b24db542f495c639e2fbe53b4a164d721642', 'item_id_map.csv': 'd6e1135f71ad9749caa6db9743045e1e65860bd62df6f42e0b76642c08645c60'}, 'best_epoch': 67, 'test_ndcg10': 0.010163783998563717, 'checkpoint': 'r1_meanpool_best.pt'}, '3407': {'model_dir': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d/outputs/toys_lbsp_seed3407', 'files_sha256': {'config.json': '5321ea691f6b25e5c038ad8c34ea30ec4cc1b17a687b7ea1e413ef4f025082f4', 'metrics.json': '34aa3cbf961526559b256fd789b9b0d5bd51ae73b930342d413e76e1c371c49e', 'training_log.csv': '161564f0c7699fb150228c6df414f356b501a8b04427533333d1f01f0df66411', 'test_per_sample_predictions.csv': '3d037bbc35ba60ab3227f1b8e78f148eb0a57877f260ac697ffdc8aa145a1de6', 'item_id_map.csv': 'd6e1135f71ad9749caa6db9743045e1e65860bd62df6f42e0b76642c08645c60'}, 'best_epoch': 64, 'test_ndcg10': 0.009442712668951093, 'checkpoint': 'r1_meanpool_best.pt'}}, '_source_sha256': {'step38_kais_retrain/scripts/38_22_eval_ml1m_r1_stability.py': '6695ccae462f786fdda4d14100d676d184fa613487fc62b0c5bd1ebe5e815080', 'step38_kais_retrain/scripts/38_20_train_ml1m_r1_bs128.py': 'c718592e65a3d66f2ec1ce6d22c8b9abc3b8d4a393dde49ae014247414741b38', 'audit/33_2_train_movielens_r1_seed42.py': 'e8ae647f3fa3cc2ef9f3087a0ae44cfbd4da2706b2ce33fc558fff0678a04ffb'}, '_data_sha256': {'processed/strict_diagnostic_a/Amazon_Toys/file_order/test.csv': '3e287ca6be6c556d5abfd691666abdc4fde1eca10327d6b8329a8aa55d0df445', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed101/test.csv': '34810e46f839d91d33dcdc27103986ff877d717ac8d12a94010946210f51e759', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed202/test.csv': '49a5ca2bd1b2e03ad4401ef72680e88fd5631f85af38832f16719ca16278e1f6', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed303/test.csv': 'a8f4073d4152f26a06493294c3177bc57ef16d6c0635834db83aedbdf57c4a9f', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed404/test.csv': 'ecc75e447e38b60abacfbb3b5d6b5e8b055448d0014e32b8b48891548b9e5370', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed505/test.csv': 'a16a0dc314392ba6573011ecbdb3239a83c5b9bc0c394d403580544d9a7c82c4'}}
_RUN = _PLAN["runs"][str(args.seed)]
_INPUTS_SHA256 = 'b98f998d0c169d8c14478795bd119b73aa48f5cdaa1767449e378ce71307667b'
_MODEL_SHA256 = 'b2bb38a5a080589d8d38dfe4485149f2c4c0b3da06922f141cfa483b488165bc'
_SOURCE_SHA256 = {'step38_kais_retrain/scripts/38_22_eval_ml1m_r1_stability.py': '6695ccae462f786fdda4d14100d676d184fa613487fc62b0c5bd1ebe5e815080', 'step38_kais_retrain/scripts/38_20_train_ml1m_r1_bs128.py': 'c718592e65a3d66f2ec1ce6d22c8b9abc3b8d4a393dde49ae014247414741b38', 'audit/33_2_train_movielens_r1_seed42.py': 'e8ae647f3fa3cc2ef9f3087a0ae44cfbd4da2706b2ce33fc558fff0678a04ffb'}
_DATA_SHA256 = {'processed/strict_diagnostic_a/Amazon_Toys/file_order/test.csv': '3e287ca6be6c556d5abfd691666abdc4fde1eca10327d6b8329a8aa55d0df445', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed101/test.csv': '34810e46f839d91d33dcdc27103986ff877d717ac8d12a94010946210f51e759', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed202/test.csv': '49a5ca2bd1b2e03ad4401ef72680e88fd5631f85af38832f16719ca16278e1f6', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed303/test.csv': 'a8f4073d4152f26a06493294c3177bc57ef16d6c0635834db83aedbdf57c4a9f', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed404/test.csv': 'ecc75e447e38b60abacfbb3b5d6b5e8b055448d0014e32b8b48891548b9e5370', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed505/test.csv': 'a16a0dc314392ba6573011ecbdb3239a83c5b9bc0c394d403580544d9a7c82c4'}
_CONTROL_REFERENCE = {'stage': 'kais_control_stability_v1_20261023T032552Z_c9383ff8a16c', 'manifest_path': 'manifests/control_stability_manifest.json', 'manifest_sha256': '7bab86b2638dc73368da5103bf3fcf90e588a6e63d1e6b66749565350efdfd4c', 'input_manifest_sha256': '8e8d13ded1ded7729e7f951b97d65ef3bda84614f4f4a9ad881bec68f7c81458', 'files_sha256': {'outputs/beauty_seed42/predictions_original.csv': '4140a5e57dd9cf3a73212118a39565e78661c7961eeac8bce667592dbabd2c0a', 'outputs/beauty_seed42/panel_labels.csv': 'cec4d8ce8a429dbfb4e953afdd9a4fb1aa10012a573f6b7186ea1fcd0c7d6626', 'outputs/beauty_seed42/metadata.json': 'b78561a12203b7895cc1ff2b7142acc634d6681c0f1844b6605dfed26331eaa4', 'outputs/beauty_seed42/predictions_perm_seed101.csv': 'b5c5fcef8d850aaed16ab4da95ce6d38fd9e312904f0f58a528def2326cdaebf', 'outputs/beauty_seed42/predictions_perm_seed202.csv': 'e980712d0d3464e1550312f025b811d43ccfdfe59808bd54b54d0122d5d7db7a', 'outputs/beauty_seed42/predictions_perm_seed303.csv': '8ec7497e81db1385b4c060e45a51a3c39315c0767a4eed758116dabf7ee9e834', 'outputs/beauty_seed42/predictions_perm_seed404.csv': '3649e3701775795e611b96122f33a9165e6912c9eeac28182dd45977983878c5', 'outputs/beauty_seed42/predictions_perm_seed505.csv': 'cef6fad6b5f58565adc297f0fe4a7b7b60a6ea3bc55753d7f0ac3a440462bcc1', 'outputs/beauty_seed2026/predictions_original.csv': '74bcccaf6729cb280659545e6732b4c5f66ae522839b4f973c80a913e65f62a7', 'outputs/beauty_seed2026/panel_labels.csv': 'cec4d8ce8a429dbfb4e953afdd9a4fb1aa10012a573f6b7186ea1fcd0c7d6626', 'outputs/beauty_seed2026/metadata.json': 'ead846f835a2e84749d05d114969908be9fc606311cae74ed527ebacaf872d51', 'outputs/beauty_seed2026/predictions_perm_seed101.csv': '339825532494aa1b37bb2fea9b9469aba4b6322f818c243975af73e4c72033cb', 'outputs/beauty_seed2026/predictions_perm_seed202.csv': 'ca23538adef5f2e1da0e2b1a5cd5ddb1c442e49046f0d7c83908ec2b1c34b6c3', 'outputs/beauty_seed2026/predictions_perm_seed303.csv': '729ca65b25757a7ffbed66ba6b57f4ba344218421a3a74b4597d93be14ac50fe', 'outputs/beauty_seed2026/predictions_perm_seed404.csv': '30992105b83d0e2ca58e6c7900bedc3c572149a82bacf99add398deec8f17b86', 'outputs/beauty_seed2026/predictions_perm_seed505.csv': 'a5b74aa2bf89820599b3244d8b716d84d244cfe2c77db49aef64ceeda55cb253', 'outputs/beauty_seed3407/predictions_original.csv': 'b9bb1912706e049836c328e3c91afc71f01b7326954f043bb1c848d91fd601d8', 'outputs/beauty_seed3407/panel_labels.csv': 'cec4d8ce8a429dbfb4e953afdd9a4fb1aa10012a573f6b7186ea1fcd0c7d6626', 'outputs/beauty_seed3407/metadata.json': '92655e1d3c6d554b8386e9d7403d1bcd15425c931ec30dfebae606349ab9c1cb', 'outputs/beauty_seed3407/predictions_perm_seed101.csv': '6172dc7a01d9e6b259230a916867ee3bc2f834e488c6e0567152c022c352ba3b', 'outputs/beauty_seed3407/predictions_perm_seed202.csv': '872cb047c594fbd0d62c23f39dedc519447b7c0700dbcc2a8ba3f9815af5662b', 'outputs/beauty_seed3407/predictions_perm_seed303.csv': '586cb78c8871322c39e2908b5ce46e36c706d0d0282afb5fc488b5d76436fbae', 'outputs/beauty_seed3407/predictions_perm_seed404.csv': 'd53fef9dd395b6471a38ecc0ca4e983510dc8a5203bcfb52b86e9479fdea764a', 'outputs/beauty_seed3407/predictions_perm_seed505.csv': 'b323fb61fbfeb7c00436d3fcdf16e5b610fecdad59a06ce63e58db384abccbbb', 'outputs/sports_seed42/predictions_original.csv': '9ef7554252ff843a1b5bf7e3809aba4827c96427aee66744f06de70852119e5e', 'outputs/sports_seed42/panel_labels.csv': '673c8d48470bbe1d5d0b02056ee55a8c9222ef4ef208dbcf5eb2523a676a70b9', 'outputs/sports_seed42/metadata.json': 'c4373b6accf11c0fe07b74a499816b07ab2d6e8135b3376a59c9f829548f8ff8', 'outputs/sports_seed42/predictions_perm_seed101.csv': '4fb86caf2911f467f25193cea878189dd6bbf9b4f648648a07673a2999034eb9', 'outputs/sports_seed42/predictions_perm_seed202.csv': 'd24a242df65cb3d9ffbd6e1414ff1d8857eee02cfe80a34bcb1064fa3a672671', 'outputs/sports_seed42/predictions_perm_seed303.csv': '406d41efa5b40170c5a32cbff9c32d2d99db2386a225ef37b65b471f5f404652', 'outputs/sports_seed42/predictions_perm_seed404.csv': 'd5e8bcbdf69908f3aaa98d6d57a5c4758beaf7d399ff06c4989bed187d9a03bd', 'outputs/sports_seed42/predictions_perm_seed505.csv': 'e844aa6237a39c8bf64caac4cee4f223127facf907b3e474737a0e3ab4484f9f', 'outputs/sports_seed2026/predictions_original.csv': '9c275fced2d98cda43257f1337908991e8c2ea970a494e16d0a0db06d24c9f53', 'outputs/sports_seed2026/panel_labels.csv': '673c8d48470bbe1d5d0b02056ee55a8c9222ef4ef208dbcf5eb2523a676a70b9', 'outputs/sports_seed2026/metadata.json': '39ba2ff2f67184e324d3a46888ec152364814416793d8e30f12089aea0ee6b48', 'outputs/sports_seed2026/predictions_perm_seed101.csv': '3fd32e061c5088b14701a415fd5febb7d784424dddfa10e9ef1e426f84a136ee', 'outputs/sports_seed2026/predictions_perm_seed202.csv': '81e305569eb8d63e6a6c15352cbd3be98460830fa6fa0c686791e91b2be8b5ee', 'outputs/sports_seed2026/predictions_perm_seed303.csv': 'e14aaaf72a74d17798036b0e57160246f47ed35519fd982882edecf383efc9e3', 'outputs/sports_seed2026/predictions_perm_seed404.csv': 'e059204963f5bf232932b056f76fa88f69017182d01961277225f7b26b215746', 'outputs/sports_seed2026/predictions_perm_seed505.csv': '9fb111120dcdd13e9fb8219ff1b779897b6c617cfcbd32ecf2c731099a389cdc', 'outputs/sports_seed3407/predictions_original.csv': '68db693ced7634bbb1fd18faf2152b62114ecaea80f3a5a5474b80ec431f7474', 'outputs/sports_seed3407/panel_labels.csv': '673c8d48470bbe1d5d0b02056ee55a8c9222ef4ef208dbcf5eb2523a676a70b9', 'outputs/sports_seed3407/metadata.json': 'b16592e53d6340b6e87dace53a8a9d3d5655f8828cc7255f84f5beca216cd7e9', 'outputs/sports_seed3407/predictions_perm_seed101.csv': '8a7123f3187711eddb260ee6dd54092a6c546875bc79b9540cf6d837fad41dfc', 'outputs/sports_seed3407/predictions_perm_seed202.csv': 'bd591190c90cdc36b77f4dedf9d19c9e74d88d54cba75c734a5e3c5df2b518b2', 'outputs/sports_seed3407/predictions_perm_seed303.csv': '80713ecd7a1fa5bd1692e227a3e137b307823af5f95b6c650c158ddfe878861f', 'outputs/sports_seed3407/predictions_perm_seed404.csv': '5c51692081ddfec9952e28759f263dc11f684c323051c394335f0d7a0f5fc7e1', 'outputs/sports_seed3407/predictions_perm_seed505.csv': 'cc923653815db276ac5b9dc31808ad23684f1219b79d091b274303107840010d', 'outputs/toys_seed42/predictions_original.csv': '9b4b5854ff4c05a6b563236e2a6b436d83931f640bf435725294ee2e0fb89cd9', 'outputs/toys_seed42/panel_labels.csv': 'dd5d4d50cfe897e344c92e5da7f14830a995d812942865887ffdd4e0d94b1e26', 'outputs/toys_seed42/metadata.json': 'e7252e9e7c6053d5575aee62ee4f406954a24e7151a24e3b9653a0126640bbb3', 'outputs/toys_seed42/predictions_perm_seed101.csv': '25598ff2e7ccc8c18abbca31f6e8f58247c53f9b08391388aa3b69c0b36f767b', 'outputs/toys_seed42/predictions_perm_seed202.csv': 'b710d6f985d7a389fdc0cb5451da706cb2d3b541fa70673d46e72273366d7966', 'outputs/toys_seed42/predictions_perm_seed303.csv': 'a23638e57ca4043cf2f8cedfc69bf578ca98024e9305e8983b38353f1e588eff', 'outputs/toys_seed42/predictions_perm_seed404.csv': '73d253e726fbbf1938c345281ad16602ec5b786d14dfec5319b4e53cc922d95f', 'outputs/toys_seed42/predictions_perm_seed505.csv': 'dd44fd8b26fc7f742501de83eddc75a79f681388c2b1df0b053b24d6995edbfe', 'outputs/toys_seed2026/predictions_original.csv': '4dc328dafddf2d2c2dfe727bbeade1bc88e44e210ac2085e3009952b549f434a', 'outputs/toys_seed2026/panel_labels.csv': 'dd5d4d50cfe897e344c92e5da7f14830a995d812942865887ffdd4e0d94b1e26', 'outputs/toys_seed2026/metadata.json': 'c0f02f05c29ffa41a7423ac33588a7a4da8648d7bbc9af256dd5c4b3c451f969', 'outputs/toys_seed2026/predictions_perm_seed101.csv': 'fb15b218470f58a4611a8ad1caa81ab757ff3f217b741372525df2cd2d350527', 'outputs/toys_seed2026/predictions_perm_seed202.csv': '1fedbbbd88602977aeef69f75b98e74a674907f1471232995b1f8f42146d01ae', 'outputs/toys_seed2026/predictions_perm_seed303.csv': '5648a472692a819faaa90ec58c4ef4c80f17e4cbf582d61e44b44874bc439379', 'outputs/toys_seed2026/predictions_perm_seed404.csv': '66d001c40fb4aae1aa71809ebc30d4842cb3324eec64342fb304a6fa1356db32', 'outputs/toys_seed2026/predictions_perm_seed505.csv': '96e6b1067a5ee52527b29c742456ae0eae529185bb72184a4122db01cfe8e2a6', 'outputs/toys_seed3407/predictions_original.csv': '2f7a18a6772ee1a514db07dd22b5b3770ba7162152d37e7077aa60d02065a03b', 'outputs/toys_seed3407/panel_labels.csv': 'dd5d4d50cfe897e344c92e5da7f14830a995d812942865887ffdd4e0d94b1e26', 'outputs/toys_seed3407/metadata.json': 'e3899886d84198ed09234ba5aae65cd32a2d171d7cf1d0f10e91648b85edccf2', 'outputs/toys_seed3407/predictions_perm_seed101.csv': '0dbe134d072dd5454229dee953128694e8835ff6a02fe7c32ac4df20e15644f1', 'outputs/toys_seed3407/predictions_perm_seed202.csv': 'f014a3ec39f410be875b2843f0f667d1c46389da90e6f61880dd6f8ae2ce8293', 'outputs/toys_seed3407/predictions_perm_seed303.csv': '0e0eaed94dc5255934daae0e7f7973b930ebcb493fa6531e3987be1c01243212', 'outputs/toys_seed3407/predictions_perm_seed404.csv': '983cbd29fcd714ae69099114b57447d56301a26273ad159182546bf58569650a', 'outputs/toys_seed3407/predictions_perm_seed505.csv': '21f18fcf3c03320bd271be26731366ec46dbe5bdf2cf792d1fd4d139f934cc01'}}
for _relative, _expected in dict(_SOURCE_SHA256, **_DATA_SHA256).items():
    checked_sha256(_LBSP_ROOT, _relative, _expected)
for _name, _expected in _RUN["files_sha256"].items():
    checked_sha256(_LBSP_ROOT, _RUN["model_dir"] + "/" + _name, _expected)
checked_sha256(_LBSP_STAGE, 'scripts/model_definition.py', _MODEL_SHA256)
checked_sha256(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + "/" + _CONTROL_REFERENCE["manifest_path"], _CONTROL_REFERENCE["manifest_sha256"])
for _relative, _expected in _CONTROL_REFERENCE["files_sha256"].items():
    checked_sha256(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + "/" + _relative, _expected)
_CONTROL_LABEL_PATH = checked_file(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + f"/outputs/toys_seed{args.seed}/panel_labels.csv")
_CHECKPOINT_PATH = checked_file(_LBSP_ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"])
_CHECKPOINT_SIGNATURE = file_signature(_CHECKPOINT_PATH.stat())
_REFERENCE_CONFIG = json.loads(checked_file(_LBSP_ROOT, _RUN["model_dir"] + "/config.json").read_text(encoding="utf-8-sig"))
print('INTEGRITY PASS: three original sources, six input CSVs, five run files, audited Control reference and frozen model; weights not yet read')


import math
_TAG = 'toys'
_INPUTS_FULL = {'version': 'lbsp_amazon_stability_v1', 'training_seeds': [42, 2026, 3407], 'permutation_seeds': [101, 202, 303, 404, 505], 'datasets': {'beauty': {'dataset': 'Amazon_Beauty', 'expected_samples': 2966, 'original_path': 'processed/recent_bucketset_r1/Amazon_Beauty/test/r1_samples.csv', 'permutation_paths': {'101': 'processed/strict_last_bucket_only/Amazon_Beauty/perm_seed101/test.csv', '202': 'processed/strict_last_bucket_only/Amazon_Beauty/perm_seed202/test.csv', '303': 'processed/strict_last_bucket_only/Amazon_Beauty/perm_seed303/test.csv', '404': 'processed/strict_last_bucket_only/Amazon_Beauty/perm_seed404/test.csv', '505': 'processed/strict_last_bucket_only/Amazon_Beauty/perm_seed505/test.csv'}, 'expected_items': 12101, 'eval_source': 'step38_kais_retrain/scripts/38_22_eval_ml1m_r1_stability.py', 'runs': {'42': {'model_dir': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d/outputs/beauty_lbsp_seed42', 'files_sha256': {'config.json': '31e7900ec88f8d37ba06bac63b6745d8c0ebdbc446024480375b62030dd71e7c', 'metrics.json': '76774b29012bc58c7334e48d6b3ca68393446ae2eb2a0fdc72859bad96eeb3bb', 'training_log.csv': 'bcee697b8b864327e77fbf8fe431e5991569c387a36c05c95e2f929f79a75d33', 'test_per_sample_predictions.csv': '3556f47245c558a6583ca9fc188c76c29ffe3e6ef5f2117e378db5b894d7c589', 'item_id_map.csv': '826da840115432e0d5a1073133f07dace4fe04de0bfe02b6b57adfc84fb5083f'}, 'best_epoch': 72, 'test_ndcg10': 0.007748010479564932, 'checkpoint': 'r1_meanpool_best.pt'}, '2026': {'model_dir': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d/outputs/beauty_lbsp_seed2026', 'files_sha256': {'config.json': 'f9e780913519eba05bcab0b31d9553b9435fd939e6963da6dd4cbac266d166f3', 'metrics.json': '4948f418ee3241eb4821c91f917b48cda9d93dc66d68f970b0e4328d97d6638f', 'training_log.csv': '52ea18fe840548692544e72a58d09432f5b4c80aa60d1a9ab3e0a65fe642cbaf', 'test_per_sample_predictions.csv': 'ce829428265b28827f2503ed7e08480f0c540ceabbb2b4f31db889f2760741b3', 'item_id_map.csv': '826da840115432e0d5a1073133f07dace4fe04de0bfe02b6b57adfc84fb5083f'}, 'best_epoch': 77, 'test_ndcg10': 0.00730090455977494, 'checkpoint': 'r1_meanpool_best.pt'}, '3407': {'model_dir': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d/outputs/beauty_lbsp_seed3407', 'files_sha256': {'config.json': '4463b95f40ec62b15cab9442e883c43dfd05c8c18eb3d98171a85ececdd9609c', 'metrics.json': 'a1897ec9baa3e8a0b925ca05337c1c142e2db64536c9cf7a70eb2d46eb726836', 'training_log.csv': '67e8bc488b09080b7a9c69daacde5f73f177dae45164fa6e2ad703b1cc4339a5', 'test_per_sample_predictions.csv': '99c0b3677c787814c5523a8f4c2848c5310d8298ab4e685065f46c3fddc8e458', 'item_id_map.csv': '826da840115432e0d5a1073133f07dace4fe04de0bfe02b6b57adfc84fb5083f'}, 'best_epoch': 111, 'test_ndcg10': 0.006014711596482316, 'checkpoint': 'r1_meanpool_best.pt'}}}, 'sports': {'dataset': 'Amazon_Sports', 'expected_samples': 4659, 'original_path': 'processed/strict_diagnostic_a/Amazon_Sports/file_order/test.csv', 'permutation_paths': {'101': 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed101/test.csv', '202': 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed202/test.csv', '303': 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed303/test.csv', '404': 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed404/test.csv', '505': 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed505/test.csv'}, 'expected_items': 18357, 'eval_source': 'step38_kais_retrain/scripts/38_22_eval_ml1m_r1_stability.py', 'runs': {'42': {'model_dir': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d/outputs/sports_lbsp_seed42', 'files_sha256': {'config.json': 'c4486bc78fd8c2178da88e3f505146ae224ba20d918d9b2271f05fb0a358c2dd', 'metrics.json': '41bcd0fb6668dd23afb8f6f9efdbe06c1c28bc3fd4018eb48088d0302aed2b09', 'training_log.csv': '863335b4b3f9711f2cf1867d5697c5771346479d339517aa0aa2fe576acc76b5', 'test_per_sample_predictions.csv': '6582fceff354807a9e3d8bf63e0baea7335a9ab138ceddd6bab280b0a606cc6d', 'item_id_map.csv': '8f0e80ba0e996b19409cd72b3ed838242397f548cf38604ffd73d58677e29fee'}, 'best_epoch': 34, 'test_ndcg10': 0.008652291133116283, 'checkpoint': 'r1_meanpool_best.pt'}, '2026': {'model_dir': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d/outputs/sports_lbsp_seed2026', 'files_sha256': {'config.json': '2d975786fed17ed9de479763824c3ab0a5f826b0cdb292e351057947b2d09ea1', 'metrics.json': '170b4981c4d1e867610fe9c7db4dc6840d4b0f1c7632a80f67be93a365cc5316', 'training_log.csv': '6756429a0992cd1d8a0dd82ba9a894441d789a9fe82c9022dea018906d4e9c58', 'test_per_sample_predictions.csv': '64443d2cad346e786cc83c9cc645534de4169d7bb1ac14bf9700467b711a7586', 'item_id_map.csv': '8f0e80ba0e996b19409cd72b3ed838242397f548cf38604ffd73d58677e29fee'}, 'best_epoch': 4, 'test_ndcg10': 0.004854418578824884, 'checkpoint': 'r1_meanpool_best.pt'}, '3407': {'model_dir': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d/outputs/sports_lbsp_seed3407', 'files_sha256': {'config.json': 'c9613f9eca3ab741f3890b9224c7bdcf213b842af211705cec19bff7286e7d84', 'metrics.json': '2879b7d5a574dfbd9389a706da630caf51f16d92f5ebb4f4c1d49fd0d1f2d206', 'training_log.csv': '4df6c5cb16467e5f103a41cbdbcb869169fa66a3e6e41dcb9154ee5f2c22f603', 'test_per_sample_predictions.csv': '490f1ff1ffd970ba16ee855bfe8e5198f3aef7ca32234e6e09f117d50e1ebd95', 'item_id_map.csv': '8f0e80ba0e996b19409cd72b3ed838242397f548cf38604ffd73d58677e29fee'}, 'best_epoch': 41, 'test_ndcg10': 0.00971069920628375, 'checkpoint': 'r1_meanpool_best.pt'}}}, 'toys': {'dataset': 'Amazon_Toys', 'expected_samples': 2547, 'original_path': 'processed/strict_diagnostic_a/Amazon_Toys/file_order/test.csv', 'permutation_paths': {'101': 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed101/test.csv', '202': 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed202/test.csv', '303': 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed303/test.csv', '404': 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed404/test.csv', '505': 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed505/test.csv'}, 'expected_items': 11924, 'eval_source': 'step38_kais_retrain/scripts/38_22_eval_ml1m_r1_stability.py', 'runs': {'42': {'model_dir': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d/outputs/toys_lbsp_seed42', 'files_sha256': {'config.json': '8a7c296f1623752506781279ad0a1357c77162faf31d1b5534e6da2d6650b17f', 'metrics.json': '295d407a4568810173e0db3f81533e15c48761ea29baca7f6eb902c58a44d414', 'training_log.csv': '7163fc161a81387f1d58c1693096b17569c471438f6bb1c418fe1aaab1481623', 'test_per_sample_predictions.csv': '6fa2b33f623626ad8f140f4629b3203e9f349965df6ae567131054b6ae516a81', 'item_id_map.csv': 'd6e1135f71ad9749caa6db9743045e1e65860bd62df6f42e0b76642c08645c60'}, 'best_epoch': 5, 'test_ndcg10': 0.003505244168310362, 'checkpoint': 'r1_meanpool_best.pt'}, '2026': {'model_dir': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d/outputs/toys_lbsp_seed2026', 'files_sha256': {'config.json': 'b5aa136886e17fe2fbf407a9cb0194f31c95270607344bf0e0e4944588c12980', 'metrics.json': 'f64959f92648d31f57d46284101fe665a9bdcb1c0e3ae2b938eee0e36c249e5a', 'training_log.csv': 'f64836deec69fb88727c713ecf28ef0a59619c121c0a74588ca014edc0dfde11', 'test_per_sample_predictions.csv': 'f8c0db82a532bba9a4db43b2ea51b24db542f495c639e2fbe53b4a164d721642', 'item_id_map.csv': 'd6e1135f71ad9749caa6db9743045e1e65860bd62df6f42e0b76642c08645c60'}, 'best_epoch': 67, 'test_ndcg10': 0.010163783998563717, 'checkpoint': 'r1_meanpool_best.pt'}, '3407': {'model_dir': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d/outputs/toys_lbsp_seed3407', 'files_sha256': {'config.json': '5321ea691f6b25e5c038ad8c34ea30ec4cc1b17a687b7ea1e413ef4f025082f4', 'metrics.json': '34aa3cbf961526559b256fd789b9b0d5bd51ae73b930342d413e76e1c371c49e', 'training_log.csv': '161564f0c7699fb150228c6df414f356b501a8b04427533333d1f01f0df66411', 'test_per_sample_predictions.csv': '3d037bbc35ba60ab3227f1b8e78f148eb0a57877f260ac697ffdc8aa145a1de6', 'item_id_map.csv': 'd6e1135f71ad9749caa6db9743045e1e65860bd62df6f42e0b76642c08645c60'}, 'best_epoch': 64, 'test_ndcg10': 0.009442712668951093, 'checkpoint': 'r1_meanpool_best.pt'}}}}, 'sources_sha256': {'step38_kais_retrain/scripts/38_22_eval_ml1m_r1_stability.py': '6695ccae462f786fdda4d14100d676d184fa613487fc62b0c5bd1ebe5e815080', 'step38_kais_retrain/scripts/38_20_train_ml1m_r1_bs128.py': 'c718592e65a3d66f2ec1ce6d22c8b9abc3b8d4a393dde49ae014247414741b38', 'audit/33_2_train_movielens_r1_seed42.py': 'e8ae647f3fa3cc2ef9f3087a0ae44cfbd4da2706b2ce33fc558fff0678a04ffb'}, 'model_source': 'audit/33_2_train_movielens_r1_seed42.py', 'forward_source': 'step38_kais_retrain/scripts/38_20_train_ml1m_r1_bs128.py', 'data_sha256': {'processed/recent_bucketset_r1/Amazon_Beauty/test/r1_samples.csv': 'f9362385825ed79b07e4d262cf9ac7df2591c54cd7d8e95e54ba116bdaad5009', 'processed/strict_last_bucket_only/Amazon_Beauty/perm_seed101/test.csv': '3fd31820d36d69e4e65abc31854f52dc70665307c2f3085521f7cb8bb727a658', 'processed/strict_last_bucket_only/Amazon_Beauty/perm_seed202/test.csv': 'b9526e02f4d3ffb6efca5e7e6fcc537f604b001aa021ffc90ae54abeab664285', 'processed/strict_last_bucket_only/Amazon_Beauty/perm_seed303/test.csv': '996369d5f666dd70cffd8a1b91650e9b18c7ee41db78073c0e04d76ce20e8aff', 'processed/strict_last_bucket_only/Amazon_Beauty/perm_seed404/test.csv': '5cb0e388cf7c71e115589059ac05368d0ac080feaea2f1e333e36bca22c09a6f', 'processed/strict_last_bucket_only/Amazon_Beauty/perm_seed505/test.csv': '82016273417f0e59797a3e370c181216b4c84f7c69a76c5ab0ed2d2fc4952e57', 'processed/strict_diagnostic_a/Amazon_Sports/file_order/test.csv': '5f2f31713ee7f6c92bb6fde915500d51f1a49328f3f1fc94f146530ad54e0e89', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed101/test.csv': 'ae1cc356e685451a400db4266f53e0cce42db12ede88f590de418a5c5a7ba3db', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed202/test.csv': '88d1d557a1c4f079d261aae9a01326f836caf67fa8b51ab877e87110bd42e1f0', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed303/test.csv': 'c284d34cf4d3193826e9a489dd31943e29f1fef86be74753e497f831620ddd63', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed404/test.csv': 'c5cf2bdf1b8b87beb1db2397038d1a546938a764c321696ab2ae6b062e5f3bf8', 'processed/strict_last_bucket_only/Amazon_Sports/perm_seed505/test.csv': 'dadca45252214e7b94bac0c0aedf9129182c2133be875fba77b3c748d3a0bff8', 'processed/strict_diagnostic_a/Amazon_Toys/file_order/test.csv': '3e287ca6be6c556d5abfd691666abdc4fde1eca10327d6b8329a8aa55d0df445', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed101/test.csv': '34810e46f839d91d33dcdc27103986ff877d717ac8d12a94010946210f51e759', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed202/test.csv': '49a5ca2bd1b2e03ad4401ef72680e88fd5631f85af38832f16719ca16278e1f6', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed303/test.csv': 'a8f4073d4152f26a06493294c3177bc57ef16d6c0635834db83aedbdf57c4a9f', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed404/test.csv': 'ecc75e447e38b60abacfbb3b5d6b5e8b055448d0014e32b8b48891548b9e5370', 'processed/strict_last_bucket_only/Amazon_Toys/perm_seed505/test.csv': 'a16a0dc314392ba6573011ecbdb3239a83c5b9bc0c394d403580544d9a7c82c4'}, 'control_reference': {'stage': 'kais_control_stability_v1_20261023T032552Z_c9383ff8a16c', 'manifest_path': 'manifests/control_stability_manifest.json', 'manifest_sha256': '7bab86b2638dc73368da5103bf3fcf90e588a6e63d1e6b66749565350efdfd4c', 'input_manifest_sha256': '8e8d13ded1ded7729e7f951b97d65ef3bda84614f4f4a9ad881bec68f7c81458', 'files_sha256': {'outputs/beauty_seed42/predictions_original.csv': '4140a5e57dd9cf3a73212118a39565e78661c7961eeac8bce667592dbabd2c0a', 'outputs/beauty_seed42/panel_labels.csv': 'cec4d8ce8a429dbfb4e953afdd9a4fb1aa10012a573f6b7186ea1fcd0c7d6626', 'outputs/beauty_seed42/metadata.json': 'b78561a12203b7895cc1ff2b7142acc634d6681c0f1844b6605dfed26331eaa4', 'outputs/beauty_seed42/predictions_perm_seed101.csv': 'b5c5fcef8d850aaed16ab4da95ce6d38fd9e312904f0f58a528def2326cdaebf', 'outputs/beauty_seed42/predictions_perm_seed202.csv': 'e980712d0d3464e1550312f025b811d43ccfdfe59808bd54b54d0122d5d7db7a', 'outputs/beauty_seed42/predictions_perm_seed303.csv': '8ec7497e81db1385b4c060e45a51a3c39315c0767a4eed758116dabf7ee9e834', 'outputs/beauty_seed42/predictions_perm_seed404.csv': '3649e3701775795e611b96122f33a9165e6912c9eeac28182dd45977983878c5', 'outputs/beauty_seed42/predictions_perm_seed505.csv': 'cef6fad6b5f58565adc297f0fe4a7b7b60a6ea3bc55753d7f0ac3a440462bcc1', 'outputs/beauty_seed2026/predictions_original.csv': '74bcccaf6729cb280659545e6732b4c5f66ae522839b4f973c80a913e65f62a7', 'outputs/beauty_seed2026/panel_labels.csv': 'cec4d8ce8a429dbfb4e953afdd9a4fb1aa10012a573f6b7186ea1fcd0c7d6626', 'outputs/beauty_seed2026/metadata.json': 'ead846f835a2e84749d05d114969908be9fc606311cae74ed527ebacaf872d51', 'outputs/beauty_seed2026/predictions_perm_seed101.csv': '339825532494aa1b37bb2fea9b9469aba4b6322f818c243975af73e4c72033cb', 'outputs/beauty_seed2026/predictions_perm_seed202.csv': 'ca23538adef5f2e1da0e2b1a5cd5ddb1c442e49046f0d7c83908ec2b1c34b6c3', 'outputs/beauty_seed2026/predictions_perm_seed303.csv': '729ca65b25757a7ffbed66ba6b57f4ba344218421a3a74b4597d93be14ac50fe', 'outputs/beauty_seed2026/predictions_perm_seed404.csv': '30992105b83d0e2ca58e6c7900bedc3c572149a82bacf99add398deec8f17b86', 'outputs/beauty_seed2026/predictions_perm_seed505.csv': 'a5b74aa2bf89820599b3244d8b716d84d244cfe2c77db49aef64ceeda55cb253', 'outputs/beauty_seed3407/predictions_original.csv': 'b9bb1912706e049836c328e3c91afc71f01b7326954f043bb1c848d91fd601d8', 'outputs/beauty_seed3407/panel_labels.csv': 'cec4d8ce8a429dbfb4e953afdd9a4fb1aa10012a573f6b7186ea1fcd0c7d6626', 'outputs/beauty_seed3407/metadata.json': '92655e1d3c6d554b8386e9d7403d1bcd15425c931ec30dfebae606349ab9c1cb', 'outputs/beauty_seed3407/predictions_perm_seed101.csv': '6172dc7a01d9e6b259230a916867ee3bc2f834e488c6e0567152c022c352ba3b', 'outputs/beauty_seed3407/predictions_perm_seed202.csv': '872cb047c594fbd0d62c23f39dedc519447b7c0700dbcc2a8ba3f9815af5662b', 'outputs/beauty_seed3407/predictions_perm_seed303.csv': '586cb78c8871322c39e2908b5ce46e36c706d0d0282afb5fc488b5d76436fbae', 'outputs/beauty_seed3407/predictions_perm_seed404.csv': 'd53fef9dd395b6471a38ecc0ca4e983510dc8a5203bcfb52b86e9479fdea764a', 'outputs/beauty_seed3407/predictions_perm_seed505.csv': 'b323fb61fbfeb7c00436d3fcdf16e5b610fecdad59a06ce63e58db384abccbbb', 'outputs/sports_seed42/predictions_original.csv': '9ef7554252ff843a1b5bf7e3809aba4827c96427aee66744f06de70852119e5e', 'outputs/sports_seed42/panel_labels.csv': '673c8d48470bbe1d5d0b02056ee55a8c9222ef4ef208dbcf5eb2523a676a70b9', 'outputs/sports_seed42/metadata.json': 'c4373b6accf11c0fe07b74a499816b07ab2d6e8135b3376a59c9f829548f8ff8', 'outputs/sports_seed42/predictions_perm_seed101.csv': '4fb86caf2911f467f25193cea878189dd6bbf9b4f648648a07673a2999034eb9', 'outputs/sports_seed42/predictions_perm_seed202.csv': 'd24a242df65cb3d9ffbd6e1414ff1d8857eee02cfe80a34bcb1064fa3a672671', 'outputs/sports_seed42/predictions_perm_seed303.csv': '406d41efa5b40170c5a32cbff9c32d2d99db2386a225ef37b65b471f5f404652', 'outputs/sports_seed42/predictions_perm_seed404.csv': 'd5e8bcbdf69908f3aaa98d6d57a5c4758beaf7d399ff06c4989bed187d9a03bd', 'outputs/sports_seed42/predictions_perm_seed505.csv': 'e844aa6237a39c8bf64caac4cee4f223127facf907b3e474737a0e3ab4484f9f', 'outputs/sports_seed2026/predictions_original.csv': '9c275fced2d98cda43257f1337908991e8c2ea970a494e16d0a0db06d24c9f53', 'outputs/sports_seed2026/panel_labels.csv': '673c8d48470bbe1d5d0b02056ee55a8c9222ef4ef208dbcf5eb2523a676a70b9', 'outputs/sports_seed2026/metadata.json': '39ba2ff2f67184e324d3a46888ec152364814416793d8e30f12089aea0ee6b48', 'outputs/sports_seed2026/predictions_perm_seed101.csv': '3fd32e061c5088b14701a415fd5febb7d784424dddfa10e9ef1e426f84a136ee', 'outputs/sports_seed2026/predictions_perm_seed202.csv': '81e305569eb8d63e6a6c15352cbd3be98460830fa6fa0c686791e91b2be8b5ee', 'outputs/sports_seed2026/predictions_perm_seed303.csv': 'e14aaaf72a74d17798036b0e57160246f47ed35519fd982882edecf383efc9e3', 'outputs/sports_seed2026/predictions_perm_seed404.csv': 'e059204963f5bf232932b056f76fa88f69017182d01961277225f7b26b215746', 'outputs/sports_seed2026/predictions_perm_seed505.csv': '9fb111120dcdd13e9fb8219ff1b779897b6c617cfcbd32ecf2c731099a389cdc', 'outputs/sports_seed3407/predictions_original.csv': '68db693ced7634bbb1fd18faf2152b62114ecaea80f3a5a5474b80ec431f7474', 'outputs/sports_seed3407/panel_labels.csv': '673c8d48470bbe1d5d0b02056ee55a8c9222ef4ef208dbcf5eb2523a676a70b9', 'outputs/sports_seed3407/metadata.json': 'b16592e53d6340b6e87dace53a8a9d3d5655f8828cc7255f84f5beca216cd7e9', 'outputs/sports_seed3407/predictions_perm_seed101.csv': '8a7123f3187711eddb260ee6dd54092a6c546875bc79b9540cf6d837fad41dfc', 'outputs/sports_seed3407/predictions_perm_seed202.csv': 'bd591190c90cdc36b77f4dedf9d19c9e74d88d54cba75c734a5e3c5df2b518b2', 'outputs/sports_seed3407/predictions_perm_seed303.csv': '80713ecd7a1fa5bd1692e227a3e137b307823af5f95b6c650c158ddfe878861f', 'outputs/sports_seed3407/predictions_perm_seed404.csv': '5c51692081ddfec9952e28759f263dc11f684c323051c394335f0d7a0f5fc7e1', 'outputs/sports_seed3407/predictions_perm_seed505.csv': 'cc923653815db276ac5b9dc31808ad23684f1219b79d091b274303107840010d', 'outputs/toys_seed42/predictions_original.csv': '9b4b5854ff4c05a6b563236e2a6b436d83931f640bf435725294ee2e0fb89cd9', 'outputs/toys_seed42/panel_labels.csv': 'dd5d4d50cfe897e344c92e5da7f14830a995d812942865887ffdd4e0d94b1e26', 'outputs/toys_seed42/metadata.json': 'e7252e9e7c6053d5575aee62ee4f406954a24e7151a24e3b9653a0126640bbb3', 'outputs/toys_seed42/predictions_perm_seed101.csv': '25598ff2e7ccc8c18abbca31f6e8f58247c53f9b08391388aa3b69c0b36f767b', 'outputs/toys_seed42/predictions_perm_seed202.csv': 'b710d6f985d7a389fdc0cb5451da706cb2d3b541fa70673d46e72273366d7966', 'outputs/toys_seed42/predictions_perm_seed303.csv': 'a23638e57ca4043cf2f8cedfc69bf578ca98024e9305e8983b38353f1e588eff', 'outputs/toys_seed42/predictions_perm_seed404.csv': '73d253e726fbbf1938c345281ad16602ec5b786d14dfec5319b4e53cc922d95f', 'outputs/toys_seed42/predictions_perm_seed505.csv': 'dd44fd8b26fc7f742501de83eddc75a79f681388c2b1df0b053b24d6995edbfe', 'outputs/toys_seed2026/predictions_original.csv': '4dc328dafddf2d2c2dfe727bbeade1bc88e44e210ac2085e3009952b549f434a', 'outputs/toys_seed2026/panel_labels.csv': 'dd5d4d50cfe897e344c92e5da7f14830a995d812942865887ffdd4e0d94b1e26', 'outputs/toys_seed2026/metadata.json': 'c0f02f05c29ffa41a7423ac33588a7a4da8648d7bbc9af256dd5c4b3c451f969', 'outputs/toys_seed2026/predictions_perm_seed101.csv': 'fb15b218470f58a4611a8ad1caa81ab757ff3f217b741372525df2cd2d350527', 'outputs/toys_seed2026/predictions_perm_seed202.csv': '1fedbbbd88602977aeef69f75b98e74a674907f1471232995b1f8f42146d01ae', 'outputs/toys_seed2026/predictions_perm_seed303.csv': '5648a472692a819faaa90ec58c4ef4c80f17e4cbf582d61e44b44874bc439379', 'outputs/toys_seed2026/predictions_perm_seed404.csv': '66d001c40fb4aae1aa71809ebc30d4842cb3324eec64342fb304a6fa1356db32', 'outputs/toys_seed2026/predictions_perm_seed505.csv': '96e6b1067a5ee52527b29c742456ae0eae529185bb72184a4122db01cfe8e2a6', 'outputs/toys_seed3407/predictions_original.csv': '2f7a18a6772ee1a514db07dd22b5b3770ba7162152d37e7077aa60d02065a03b', 'outputs/toys_seed3407/panel_labels.csv': 'dd5d4d50cfe897e344c92e5da7f14830a995d812942865887ffdd4e0d94b1e26', 'outputs/toys_seed3407/metadata.json': 'e3899886d84198ed09234ba5aae65cd32a2d171d7cf1d0f10e91648b85edccf2', 'outputs/toys_seed3407/predictions_perm_seed101.csv': '0dbe134d072dd5454229dee953128694e8835ff6a02fe7c32ac4df20e15644f1', 'outputs/toys_seed3407/predictions_perm_seed202.csv': 'f014a3ec39f410be875b2843f0f667d1c46389da90e6f61880dd6f8ae2ce8293', 'outputs/toys_seed3407/predictions_perm_seed303.csv': '0e0eaed94dc5255934daae0e7f7973b930ebcb493fa6531e3987be1c01243212', 'outputs/toys_seed3407/predictions_perm_seed404.csv': '983cbd29fcd714ae69099114b57447d56301a26273ad159182546bf58569650a', 'outputs/toys_seed3407/predictions_perm_seed505.csv': '21f18fcf3c03320bd271be26731366ec46dbe5bdf2cf792d1fd4d139f934cc01'}}, 'training_reference': {'stage': 'kais_lbsp_amazon_train_v1_20261023T082808Z_224454fcd01d', 'manifest_path': 'manifests/lbsp_amazon_train_manifest.json', 'manifest_sha256': 'f7a3ece9c5e680e61ab99ff8e0316826e8f37e85d78638ee5e44985fd1822f13', 'input_manifest_sha256': 'aad9c9a404f04df9b355d6aa6b4ca0b26905d9763ee9fb3915c9f3e102682ff2', 'files_sha256': {'manifests/lbsp_amazon_train_manifest.json': 'f7a3ece9c5e680e61ab99ff8e0316826e8f37e85d78638ee5e44985fd1822f13', 'scripts/model_definition.py': '500244e2a920b3d256176b6cad0c79031d9dc327b9b2b812214005f5764a6a6a', 'scripts/train_beauty_lbsp.py': '7fe014d3c1fba6648db63cecdd6dc467cda210bd129df0f4a43bb2b5382dfdbc', 'scripts/train_sports_lbsp.py': 'bb6b91f181ebffe02ed921f785190cce4f71406313496837ed4d956c8f42538c', 'scripts/train_toys_lbsp.py': '399a410e110cd2cdca1da1e5014cbc312a7eedde9b4b327df435306cf52120c3', 'verification/beauty_seed2026_train_20261023T082808Z_10a5ddfd.log': '2c10eb1c90295bd99e1c0743cbf2681334d8891c4217a8c04d8334927864740c', 'verification/beauty_seed3407_train_20261023T082808Z_10a5ddfd.log': 'eb0e477838af6fc5ead4a99ad3c05aa1e728ef23467bf2a36a5046c10f529b55', 'verification/beauty_seed42_train_20261023T082808Z_10a5ddfd.log': 'e5ad8650493c608de96d9d88c17c5da03b146d83f1a94de5f4d85b3182e9c085', 'verification/completed_beauty_seed2026.json': '594d91b75b1eecb1c36c2737aa5446c2bf369f378874e62307ad5c5d55f852a6', 'verification/completed_beauty_seed3407.json': 'a057f105e1e0f301219b090b252f56bfd0de29bb4c750f0b0507b0c90d575851', 'verification/completed_beauty_seed42.json': '36e0e2bf1e8f371207fd45a7b66f7909e38056364577c9b8cecfdd035a84effa', 'verification/completed_sports_seed2026.json': 'be4398dd4306bb9efc118065ed8bd0b44fc2675ca1fa0013f679fb3adb74bab6', 'verification/completed_sports_seed3407.json': '9a7a7c29cab4bc15f16d37c4b5d23368cebaaff926543306b65bfb1b00919646', 'verification/completed_sports_seed42.json': '0d514474a053e399d07fb79e149b92eaa9ca0b5f4020a791646c2cfa8158d649', 'verification/completed_toys_seed2026.json': 'dbeaa5d0f0482bfe705893823b03562809caf3793a75c1e366c66d6e55227b50', 'verification/completed_toys_seed3407.json': '8d5a1d61fd32daba1d6a05500be0d4db2dfecc1a75a130ab8bc2e18123c6af55', 'verification/completed_toys_seed42.json': '341f53bfb3a92f14faafbfb16db2d17e96394e0bf5047bbff30edbe2a8b39e41', 'verification/lbsp_amazon_train_driver_20261023T082808Z_10a5ddfd.json': '48d2e94e3d6b539481ba8cb9589214c2f8cf3f516dcedcc36821ba5e120c9bab', 'verification/PREPARED.json': 'b2392193dfc363cd88b76060dba9065d2a3ebd169a3e38b093fc17a1202f28c5', 'verification/sports_seed2026_train_20261023T082808Z_10a5ddfd.log': '0c275a55dc4d62248936b735af1dd43492d8f799e2f783b5420a5537da0bfe01', 'verification/sports_seed3407_train_20261023T082808Z_10a5ddfd.log': '6eb4ac703a6f57733d2ba93c59768f8957a528b0147753cc28a23debe7c07206', 'verification/sports_seed42_train_20261023T082808Z_10a5ddfd.log': '265dc743e85b1a998fdc3faae8a0d565906612a1b2b04cf1d9b772037431d3cf', 'verification/toys_seed2026_train_20261023T082808Z_10a5ddfd.log': '23fb276e52b4c28f48f6f3fb47bb24faaca4284eab3d5cf6888bb591de06944d', 'verification/toys_seed3407_train_20261023T082808Z_10a5ddfd.log': 'd17e3d3e58bca4b2b4939f3b2ce1fe15c4623b51ba44c30c70dbb00a0b18f3dc', 'verification/toys_seed42_train_20261023T082808Z_10a5ddfd.log': 'e471ef2f6e3dd4c69de170d30d482f172a656b6e54e5a69138629bc30a6d3c6b'}}, 'scope': 'Nine fixed-model Amazon LBSP evaluations: three previously trained seeds per dataset, canonical input plus five frozen last-bucket permutations. Raw-event affected population must match audited Control before canonicalization. All ten unordered permutation pairs retained, primary hitflip is per-user any hit/nonhit variation across five permutations. No training, tuning, seed selection, significance or equivalence claims.'}
_TRAINING_REFERENCE = _INPUTS_FULL["training_reference"]
_TRAINING_MANIFEST = json.loads(checked_file(_LBSP_ROOT, _TRAINING_REFERENCE["stage"] + "/" + _TRAINING_REFERENCE["manifest_path"]).read_text(encoding="utf-8-sig"))
_PREPARATION_BYTES = checked_file(_LBSP_STAGE, 'manifests/lbsp_amazon_stability_manifest.json').read_bytes()
_PREPARATION_SHA256 = hashlib.sha256(_PREPARATION_BYTES).hexdigest()
_PREPARATION = json.loads(_PREPARATION_BYTES.decode("utf-8-sig"))
if (_PREPARATION["inputs"] != _INPUTS_FULL or _PREPARATION["inputs_sha256"] != _INPUTS_SHA256
        or _PREPARATION["version"] != "lbsp_amazon_stability_v1" or _PREPARATION["root"] != str(_LBSP_ROOT)
        or _PREPARATION["stage"] != str(_LBSP_STAGE)):
    raise RuntimeError("Preparation identity differs")
_GENERATED_RELATIVE = "scripts/evaluate_" + _TAG + "_lbsp.py"
_GENERATED_SHA256 = _PREPARATION["generated_sha256"][_GENERATED_RELATIVE]
_TRAINING_MODEL_SHA256 = _TRAINING_MANIFEST["generated_sha256"]["scripts/model_definition.py"]
def _verify_closure():
    for relative, expected in dict(_SOURCE_SHA256, **_DATA_SHA256).items():
        checked_sha256(_LBSP_ROOT, relative, expected)
    for name, expected in _RUN["files_sha256"].items():
        checked_sha256(_LBSP_ROOT, _RUN["model_dir"] + "/" + name, expected)
    checked_sha256(_LBSP_STAGE, "scripts/model_definition.py", _MODEL_SHA256)
    checked_sha256(_LBSP_STAGE, _GENERATED_RELATIVE, _GENERATED_SHA256)
    checked_sha256(_LBSP_STAGE, 'manifests/lbsp_amazon_stability_manifest.json', _PREPARATION_SHA256)
    for relative, expected in _CONTROL_REFERENCE["files_sha256"].items():
        checked_sha256(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + "/" + relative, expected)
    checked_sha256(_LBSP_ROOT, _CONTROL_REFERENCE["stage"] + "/" + _CONTROL_REFERENCE["manifest_path"], _CONTROL_REFERENCE["manifest_sha256"])
    for relative, expected in _TRAINING_REFERENCE["files_sha256"].items():
        checked_sha256(_LBSP_ROOT, _TRAINING_REFERENCE["stage"] + "/" + relative, expected)
_verify_closure()
validate_training_binding(_INPUTS_FULL, _TRAINING_MANIFEST, _LBSP_ROOT)
_COMPLETED_RUN = json.loads(checked_file(_LBSP_ROOT, _TRAINING_REFERENCE["stage"] + f"/verification/completed_{_TAG}_seed{args.seed}.json").read_text(encoding="utf-8-sig"))
_REFERENCE_METRICS = json.loads(checked_file(_LBSP_ROOT, _RUN["model_dir"] + "/metrics.json").read_text(encoding="utf-8-sig"))
validate_run_metadata(_INPUTS_FULL, _TRAINING_MANIFEST, _TAG, args.seed, _REFERENCE_CONFIG, _REFERENCE_METRICS, _COMPLETED_RUN, _LBSP_ROOT)
print("AMAZON LBSP INTEGRITY PASS: sources, six inputs, five saved files, full training and Control references verified before Torch")

import ast
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Step38.22 ML1M R1 Fixed-Checkpoint Stability
# ============================================================

ROOT = _LBSP_ROOT
BASE = _LBSP_STAGE

SEEDS = [101, 202, 303, 404, 505]
MAX_LEN = 50
BATCH_SIZE = 256
EXPECTED_N = 2547
EXPECTED_ITEMS = 11924

MODEL_DIR = ROOT / _RUN["model_dir"]

CHECKPOINT = _CHECKPOINT_PATH
MAP_PATH = MODEL_DIR / "item_id_map.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"
SAVED_PRED = MODEL_DIR / "test_per_sample_predictions.csv"

ORIGINAL_PATH = ROOT / _PLAN["original_path"]

PERM_ROOT = (
    ROOT / "processed/strict_last_bucket_only/"
    "MovieLens-1M"
)

R1_SOURCE = _LBSP_STAGE / "scripts/model_definition.py"

FORWARD_SOURCE = _LBSP_STAGE / "scripts/model_definition.py"

OUTPUT_DIR = _LBSP_STAGE / "outputs" / f"toys_seed{args.seed}"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


print("=" * 72)
print("STEP38.22 ML1M R1 FIXED MODEL STABILITY")
print("=" * 72)
print("Device:", device)
print("Permutation seeds:", SEEDS)

required = [
    CHECKPOINT,
    MAP_PATH,
    METRICS_PATH,
    SAVED_PRED,
    ORIGINAL_PATH,
    R1_SOURCE,
    FORWARD_SOURCE,
]

required += [
    ROOT / _PLAN["permutation_paths"][str(s)]
    for s in SEEDS
]

for path in required:
    require(path.is_file(), f"缺少文件: {path}")

require(
    not OUTPUT_DIR.exists(),
    f"输出目录已存在，禁止覆盖: {OUTPUT_DIR}",
)

print("INPUT FILE AUDIT PASS")


# ============================================================
# 1. 从已验证脚本提取 R1 模型、分组函数和安全前向
# 不执行训练脚本
# ============================================================

r1_ast = ast.parse(
    R1_SOURCE.read_text(encoding="utf-8")
)

forward_ast = ast.parse(
    FORWARD_SOURCE.read_text(encoding="utf-8")
)


def extract(tree, name, cls=False):
    node_type = ast.ClassDef if cls else ast.FunctionDef

    matches = [
        node for node in tree.body
        if isinstance(node, node_type)
        and node.name == name
    ]

    require(
        len(matches) == 1,
        f"代码定义异常: {name}",
    )

    return matches[0]


nodes = [
    extract(
        r1_ast,
        "build_r1_group_positions_from_size",
    ),
    extract(
        r1_ast,
        "canonicalize_last_bucket_items",
    ),
    extract(
        r1_ast,
        "RecentBucketSetR1SASRec",
        cls=True,
    ),
    extract(
        forward_ast,
        "predict_batch_masked",
    ),
]

namespace = {
    "__name__": "step38_r1_stability",
    "torch": torch,
    "nn": nn,
    "F": F,
    "np": np,
    "math": math,
    "MAX_LEN": MAX_LEN,
    "device": device,
    "require": require,
}

exec(
    compile(
        ast.Module(
            body=nodes,
            type_ignores=[],
        ),
        str(FORWARD_SOURCE),
        "exec",
    ),
    namespace,
)

Model = namespace["RecentBucketSetR1SASRec"]

Model.predict_batch_masked = namespace[
    "predict_batch_masked"
]

build_groups = namespace[
    "build_r1_group_positions_from_size"
]

canonicalize = namespace[
    "canonicalize_last_bucket_items"
]

print("R1 model and safe forward loaded: PASS")


# ============================================================
# 2. 加载固定物品映射
# ============================================================

mapping = pd.read_csv(
    MAP_PATH,
    dtype={"raw_item_id": str},
)

require(
    mapping["raw_item_id"].is_unique,
    "原始物品ID重复",
)

require(
    mapping["item_idx"].is_unique,
    "物品索引重复",
)

item_map = dict(zip(
    mapping["raw_item_id"],
    mapping["item_idx"].astype(int),
))

idx_to_raw = {
    int(idx): str(raw)
    for raw, idx in item_map.items()
}

NUM_ITEMS = len(item_map)

require(
    NUM_ITEMS == EXPECTED_ITEMS,
    f"物品数异常: {NUM_ITEMS}",
)

require(
    set(item_map.values())
    == set(range(1, NUM_ITEMS + 1)),
    "物品映射不连续",
)


# ============================================================
# 3. 加载 Epoch 33 最佳 checkpoint
# ============================================================

ckpt = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

require(
    ckpt["model_name"] == "RecentBucketSetR1SASRec",
    "Checkpoint 模型身份错误",
)

require(
    ckpt["dataset"] == _PLAN["dataset"],
    "Checkpoint 数据集错误",
)

require(
    int(ckpt["epoch"]) == _RUN["best_epoch"],
    "Checkpoint 不是 Epoch33",
)

require(
    int(ckpt["seed"]) == args.seed,
    "Checkpoint seed 错误",
)

require(
    int(ckpt["num_items"]) == NUM_ITEMS,
    "Checkpoint 物品数错误",
)

require(
    int(ckpt["training_batch_size"]) == 128,
    "Checkpoint 训练 BS 不是128",
)

require(file_signature(checked_file(ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"]).stat()) == _CHECKPOINT_SIGNATURE,
        "Checkpoint stat changed while loading")
for _name in ("max_len", "hidden_size", "num_heads", "num_blocks", "dropout", "training_batch_size"):
    require(ckpt.get(_name) == _REFERENCE_CONFIG.get(_name), "Checkpoint/config architecture mismatch: " + _name)

require(ckpt.get("method") == _REFERENCE_CONFIG["method"], "Checkpoint method differs")
model = Model(
    num_items=NUM_ITEMS,
    max_len=int(ckpt["max_len"]),
    hidden_size=int(ckpt["hidden_size"]),
    num_heads=int(ckpt["num_heads"]),
    num_blocks=int(ckpt["num_blocks"]),
    dropout=float(ckpt["dropout"]),
).to(device)

model.load_state_dict(
    ckpt["model_state_dict"],
    strict=True,
)

model.eval()

print("Checkpoint epoch:", ckpt["epoch"])
print("Checkpoint loading PASS")


# ============================================================
# 4. 读取冻结原始样本
# ============================================================

def read_csv(path, reference_ids=None):
    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "user_id": str,
            "target_item": str,
            "sequence_items": str,
        },
    )
    df = normalize_eval_frame(df)

    require(
        len(df) == EXPECTED_N,
        f"样本数量异常: {path}",
    )

    require(
        df["sample_id"].is_unique,
        f"重复样本ID: {path}",
    )

    if reference_ids is not None:
        require(
            set(df["sample_id"]) == set(reference_ids),
            f"样本集合不同: {path}",
        )

        df = (
            df.set_index("sample_id")
            .loc[reference_ids]
            .reset_index()
        )

    return df


original_df = read_csv(ORIGINAL_PATH)
ids = original_df["sample_id"].tolist()
_ORIGINAL_ROWS = indexed_rows(original_df.to_dict(orient="records"))
_VARIANTS = {}

original_raw = [
    [str(x) for x in json.loads(v)]
    for v in original_df["sequence_items"]
]

bucket_sizes = (
    original_df["last_bucket_size"]
    .astype(int)
    .to_numpy()
)

targets_raw = original_df[
    "target_item"
].astype(str).tolist()


# ============================================================
# 5. R1 规范输入构造
# ============================================================

def construct(df, use_permutation=False):
    seq_list = []
    group_list = []
    token_list = []
    targets = []
    raw_histories = []

    for i, row in enumerate(
        df.itertuples(index=False)
    ):
        raw = [
            str(x)
            for x in json.loads(row.sequence_items)
        ]

        k = int(bucket_sizes[i])

        require(
            1 <= k <= len(raw) <= MAX_LEN,
            f"历史长度或桶大小异常: {row.sample_id}",
        )

        require(
            str(row.target_item) == targets_raw[i],
            f"Target 不一致: {row.sample_id}",
        )

        if use_permutation:
            require(
                int(row.last_history_bucket_size) == k,
                f"排列桶大小不一致: {row.sample_id}",
            )

            base_raw = original_raw[i]

            require(
                len(base_raw) == len(raw),
                f"历史长度变化: {row.sample_id}",
            )

            require(
                base_raw[:-k] == raw[:-k],
                f"历史前缀变化: {row.sample_id}",
            )

            require(
                Counter(base_raw[-k:]) == Counter(raw[-k:]),
                f"最后桶物品集合变化: {row.sample_id}",
            )

        mapped = np.asarray(
            [item_map[x] for x in raw],
            dtype=np.int64,
        )

        target = item_map[targets_raw[i]]

        require(
            target not in mapped,
            f"目标泄漏: {row.sample_id}",
        )

        gp, num_tokens, pooled = build_groups(
            len(mapped),
            k,
            MAX_LEN,
        )

        if not use_permutation:
            require(
                pooled == int(row.direct_pooling),
                f"Pooling 标签不一致: {row.sample_id}",
            )

        canonical = canonicalize(
            mapped,
            k,
        )

        padded = np.zeros(
            MAX_LEN,
            dtype=np.int64,
        )

        padded[-len(canonical):] = canonical

        seq_list.append(padded)
        group_list.append(gp)
        token_list.append(num_tokens)
        targets.append(target)
        raw_histories.append(raw)

    return (
        np.stack(seq_list),
        np.stack(group_list),
        np.asarray(token_list, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        raw_histories,
    )


base_seq, base_gp, base_tok, base_target, _ = (
    construct(original_df)
)


# ============================================================
# 6. 固定模型严格推理
# 返回未屏蔽 logits，用于真正的数值不变性核验
# ============================================================

@torch.inference_mode()
def predict_all(seq_array, gp_array, tok_array, targets):
    model.eval()

    logits_chunks = []
    ranks_chunks = []
    top10_chunks = []

    for start in range(
        0,
        len(targets),
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            len(targets),
        )

        seq_np = seq_array[start:end]
        target_np = targets[start:end]

        seq = torch.from_numpy(
            seq_np
        ).long().to(device)

        groups = torch.from_numpy(
            gp_array[start:end]
        ).long().to(device)

        tokens = torch.from_numpy(
            tok_array[start:end]
        ).long().to(device)

        raw_scores = model.predict_batch_masked(
            seq,
            groups,
            tokens,
        )

        require(
            torch.isfinite(raw_scores).all().item(),
            "模型输出包含 NaN/Inf",
        )

        logits_chunks.append(
            raw_scores.cpu().numpy().copy()
        )

        scores = raw_scores.clone()
        scores[:, 0] = -1e9

        for b in range(len(seq_np)):
            seen = np.unique(seq_np[b])

            seen = seen[
                (seen != 0)
                & (seen != target_np[b])
            ]

            if len(seen):
                seen_tensor = torch.as_tensor(
                    seen,
                    dtype=torch.long,
                    device=device,
                )

                scores[b, seen_tensor] = -1e9

        target_tensor = torch.as_tensor(
            target_np,
            dtype=torch.long,
            device=device,
        )

        index = torch.arange(
            len(target_np),
            device=device,
        )

        target_scores = scores[
            index,
            target_tensor,
        ]

        ranks = (
            (
                scores > target_scores[:, None]
            ).sum(dim=1)
            + 1
        )

        top10 = torch.topk(
            scores,
            k=10,
            dim=1,
        ).indices

        ranks_chunks.append(
            ranks.cpu().numpy()
        )

        top10_chunks.append(
            top10.cpu().numpy()
        )

    return (
        np.concatenate(logits_chunks),
        np.concatenate(ranks_chunks),
        np.concatenate(top10_chunks),
    )


def make_predictions(df, ranks, top10):
    pred = df[
        ["sample_id", "user_id", "target_item"]
    ].copy()

    pred["target_rank"] = ranks

    pred["top10_item_idx"] = [
        json.dumps(row.tolist())
        for row in top10
    ]

    pred["top10_raw_item_id"] = [
        json.dumps(
            [idx_to_raw[int(x)] for x in row]
        )
        for row in top10
    ]

    return pred


# ============================================================
# 7. 复现原始严格测试结果
# ============================================================

base_logits, base_ranks, base_top10 = predict_all(
    base_seq,
    base_gp,
    base_tok,
    base_target,
)

saved = read_csv_predictions = pd.read_csv(
    SAVED_PRED,
    dtype={"sample_id": str},
)

require(
    saved["sample_id"].is_unique,
    "已保存预测中出现重复样本",
)

require(set(saved["sample_id"]) == set(ids), "Saved baseline sample panel differs")
saved = (
    saved.set_index("sample_id")
    .loc[ids]
    .reset_index()
)

saved_ranks = (
    saved["target_rank"]
    .to_numpy(dtype=np.int64)
)

saved_top10 = np.stack([
    np.asarray(
        json.loads(v),
        dtype=np.int64,
    )
    for v in saved["top10_item_idx"]
])

rank_mismatches = int(
    np.count_nonzero(
        base_ranks != saved_ranks
    )
)

top10_mismatches = int(
    np.count_nonzero(
        np.any(
            base_top10 != saved_top10,
            axis=1,
        )
    )
)

metrics = json.loads(
    METRICS_PATH.read_text(encoding="utf-8")
)

ndcg = float(
    np.where(
        base_ranks <= 10,
        1.0 / np.log2(base_ranks + 1),
        0.0,
    ).mean()
)

require(
    rank_mismatches == 0,
    "原始目标排名未能复现",
)

require(
    top10_mismatches == 0,
    "原始 Top-10 未能复现",
)

require(
    abs(ndcg - metrics["test"]["ndcg10"]) < 1e-10,
    "原始 NDCG 未能复现",
)

print()
print("===== ORIGINAL STRICT REPRODUCTION =====")
print("Samples:", len(base_ranks))
print("Rank mismatches:", rank_mismatches)
print("Top10 mismatches:", top10_mismatches)
print("Original Test NDCG@10:", f"{ndcg:.9f}")
_IDENTITY_TOP10_MISMATCHES = verify_original_top10(
    make_predictions(original_df, base_ranks, base_top10).to_dict(orient="records"), saved.to_dict(orient="records"))
require(abs(ndcg - _RUN["test_ndcg10"]) < 1e-10, "Baseline NDCG differs from the audited run")
print("ORIGINAL STRICT REPRODUCTION PASS: ranks, ordered Top10, raw IDs and sample identity")


# ============================================================
# 8. 五组真实排列推理
# ============================================================

original_pred = make_predictions(
    original_df,
    base_ranks,
    base_top10,
)

results = {}
seed_summaries = []

for seed in SEEDS:
    perm_df = read_csv(
        ROOT / _PLAN["permutation_paths"][str(seed)],
        reference_ids=ids,
    )

    _VARIANTS[str(seed)] = verify_permutation(_ORIGINAL_ROWS, perm_df.to_dict(orient="records"))
    seq, gp, tok, target, raw_histories = construct(
        perm_df,
        use_permutation=True,
    )

    changed = np.asarray([
        a != b
        for a, b in zip(
            original_raw,
            raw_histories,
        )
    ])

    changed_n = int(changed.sum())

    require(
        changed_n > 0,
        f"Seed {seed}: 未发现扰动样本",
    )

    # Input invariant: grouped canonicalized tensors
    require(
        np.array_equal(seq, base_seq)
        and np.array_equal(gp, base_gp)
        and np.array_equal(tok, base_tok)
        and np.array_equal(target, base_target),
        f"Seed {seed}: R1 规范化输入不一致",
    )

    logits, ranks, top10 = predict_all(
        seq,
        gp,
        tok,
        target,
    )

    max_abs_diff = float(
        np.max(
            np.abs(
                logits.astype(np.float64)
                - base_logits.astype(np.float64)
            )
        )
    )

    ranks_changed = int(
        np.count_nonzero(
            ranks != base_ranks
        )
    )

    top10_changed = int(
        np.count_nonzero(
            np.any(
                top10 != base_top10,
                axis=1,
            )
        )
    )

    require(
        np.array_equal(logits, base_logits),
        f"Seed {seed}: Logits 不完全一致，"
        f"最大差异={max_abs_diff}",
    )

    require(
        ranks_changed == 0
        and top10_changed == 0,
        f"Seed {seed}: 排名或 Top10 发生变化",
    )

    results[seed] = make_predictions(
        perm_df,
        ranks,
        top10,
    )

    summary = {
        "permutation_seed": seed,
        "samples": EXPECTED_N,
        "changed_sequences": changed_n,
        "canonical_input_mismatches": 0,
        "max_abs_logit_diff": max_abs_diff,
        "changed_target_ranks": ranks_changed,
        "changed_top10": top10_changed,
        "top10_overlap_changed": 1.0,
    }

    seed_summaries.append(summary)

    print(
        f"Seed {seed}: "
        f"ChangedSeq={changed_n} "
        f"ChangedRank={ranks_changed} "
        f"ChangedTop10={top10_changed} "
        f"MaxLogitDiff={max_abs_diff:.12g} "
        f"OverlapChanged=1.000000",
        flush=True,
    )


# ============================================================
# 9. 全部通过后保存，避免部分成功文件污染
# ============================================================

_verify_closure()
_LABELS = panel_labels(_ORIGINAL_ROWS, _VARIANTS)
control_labels_match(_LABELS, _CONTROL_LABEL_PATH)
_CHECKPOINT_AFTER = file_signature(checked_file(ROOT, _RUN["model_dir"] + "/" + _RUN["checkpoint"]).stat())
require(_CHECKPOINT_AFTER == _CHECKPOINT_SIGNATURE, "Checkpoint stat changed during inference")
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
# Copy only after raw-event labels/hashes/affected flags exactly match Control.
(OUTPUT_DIR / "panel_labels.csv").write_bytes(_CONTROL_LABEL_PATH.read_bytes())

original_pred.to_csv(
    OUTPUT_DIR / "predictions_original.csv",
    index=False,
    encoding="utf-8-sig",
)

for seed, pred in results.items():
    pred.to_csv(
        OUTPUT_DIR / f"predictions_perm_seed{seed}.csv",
        index=False,
        encoding="utf-8-sig",
    )

pd.DataFrame(seed_summaries).to_csv(
    OUTPUT_DIR / "per_seed_invariance.csv",
    index=False,
)

manifest = {
    "dataset": _PLAN["dataset"],
    "model": "RecentBucketSetR1SASRec",
    "method": "RecentBucketSet_R1_V1_mean_pooling",
    "model_seed": args.seed,
    "best_epoch": int(ckpt["epoch"]),
    "checkpoint_epoch": int(ckpt["epoch"]),
    "num_items": NUM_ITEMS,
    "test_samples": EXPECTED_N,
    "training_batch_size": 128,
    "evaluation_batch_size": BATCH_SIZE,
    "permutation_seeds": SEEDS,
    "strict_reproduction": "PASS",
    "canonicalization": "last_bucket_sorted_by_item_idx",
    "exact_raw_logits_invariance": True,
    "canonical_input_mismatches": 0,
    "original_rank_mismatches": rank_mismatches,
    "original_top10_mismatches": top10_mismatches,
    "original_identity_raw_top10_mismatches": _IDENTITY_TOP10_MISMATCHES,
    "input_manifest_sha256": _INPUTS_SHA256,
    "preparation_manifest_sha256": hashlib.sha256((_LBSP_STAGE / "manifests/lbsp_amazon_stability_manifest.json").read_bytes()).hexdigest(),
    "generated_evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "model_definition_sha256": _MODEL_SHA256,
    "reference_files_sha256": _RUN["files_sha256"],
    "data_sha256": _DATA_SHA256,
    "source_sha256": _SOURCE_SHA256,
    "model_dir": str(MODEL_DIR),
    "checkpoint_path": str(CHECKPOINT),
    "checkpoint_stat_before_inference": list(_CHECKPOINT_SIGNATURE),
    "checkpoint_stat_after_inference": list(_CHECKPOINT_AFTER),
    "control_reference": _CONTROL_REFERENCE,
    "control_panel_labels_sha256": hashlib.sha256(_CONTROL_LABEL_PATH.read_bytes()).hexdigest(),
    "affected_definition": "at least two distinct raw-event sequences among five permutations, before LBSP canonicalization; original excluded",
    "affected_samples": sum(row["affected"] for row in _LABELS),
    "sequence_hash_definition": "SHA256(json.dumps(tuple(str(item) for item in sequence), separators=(',', ':'), ensure_ascii=False).encode('utf-8'))",
    "per_seed": seed_summaries,
    "training_reference": _TRAINING_REFERENCE,
    "training_model_definition_sha256": _TRAINING_MODEL_SHA256,
    "runtime_hashes_verified_before_after": True,
}

(OUTPUT_DIR / "metadata.json").write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("=" * 72)
print("STEP38.22 ML1M R1 STABILITY PASS")
print("All five permutations: exact logits/ranks/Top10 invariant")
print("Output:", OUTPUT_DIR)
print("=" * 72)
