"""Local packaging helper only; never uploads/runs research code."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REV = ROOT / 'kais_revision_outline_20261022'


def jr(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy(source, target):
    source = ROOT / source
    target = HERE / target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target.relative_to(HERE).as_posix()


def extract(source, name):
    tree = ast.parse(source)
    n = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    value = n.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == 'loads':
        return json.loads(ast.literal_eval(value.args[0]))
    return ast.literal_eval(value)


def main():
    families = {}
    scripts = {
        'control': ('g2_x3_shuffle_results_20261026/snapshot/original_sources', ['train_sports_orderfix.py', 'train_toys_orderfix.py']),
        'shuffle': ('g2_x3_shuffle_results_20261026/snapshot/scripts', ['train_ml1m_shuffle.py', 'train_toys_shuffle.py']),
        'amazon_lbsp': ('g2_lbsp_amazon_train_20261025/snapshot/scripts', ['train_sports_lbsp.py', 'train_toys_lbsp.py']),
        'ml_lbsp': ('g2_lbsp_amazon_train_20261025/snapshot/original_sources', ['38_20_train_ml1m_r1_bs128.py']),
        'tisasrec': ('g2_x4_tisasrec_results_20261026/snapshot/scripts', ['run_x4_tisasrec.py', 'collect_x4_tisasrec.py', 'x4_tisasrec_model.py', 'x4_tisasrec_protocol.py', 'X4_MODEL_PROVENANCE.json', 'X4_UPSTREAM_MIT_LICENSE.txt']),
    }
    for family, (base, names) in scripts.items():
        families[family] = {'scripts': {name: copy(base + '/' + name, 'vendor/' + family + '/' + name) for name in names}, 'patch_scripts': names if family != 'tisasrec' else ['run_x4_tisasrec.py']}
    families['control']['scripts']['model_definition.py'] = copy('g2_seed42_20261022/snapshot/scripts/model_definition.py', 'vendor/control/model_definition.py')
    families['shuffle']['scripts']['model_definition.py'] = copy('g2_x3_shuffle_results_20261026/snapshot/scripts/model_definition.py', 'vendor/shuffle/model_definition.py')
    families['amazon_lbsp']['scripts']['model_definition.py'] = copy('g2_lbsp_amazon_train_20261025/snapshot/scripts/model_definition.py', 'vendor/amazon_lbsp/model_definition.py')
    for family, base, name in (
        ('shuffle', 'g2_x3_shuffle_results_20261026', 'x3_shuffle_manifest.json'),
        ('amazon_lbsp', 'g2_lbsp_amazon_train_20261025', 'lbsp_amazon_train_manifest.json')):
        families[family]['manifest'] = copy(base + '/snapshot/manifests/' + name, 'vendor/' + family + '/' + name)
        families[family]['manifest_name'] = 'manifests/' + name
    families['tisasrec']['inputs'] = copy('kais_revision_outline_20261022/server_g2/x4_tisasrec_inputs_20261026.json', 'vendor/tisasrec/x4_tisasrec_inputs_20261026.json')
    readonly = {}
    x3 = jr(HERE / families['shuffle']['manifest'])
    amazon = jr(HERE / families['amazon_lbsp']['manifest'])
    x4 = jr(HERE / families['tisasrec']['inputs'])
    for manifest in (x3, amazon):
        inputs = manifest['inputs']
        readonly.update(inputs['data_sha256'])
        readonly.update(inputs['sources_sha256'])
        for plan in inputs['datasets'].values():
            readonly.update({plan['control_reference']['model_dir'] + '/' + n: h for n, h in plan['control_reference']['files_sha256'].items()})
    smoke = amazon['inputs']['smoke_reference']
    smoke_stage = smoke['stage'].removeprefix('/root/autodl-tmp/time_uncertainty_rec/')
    readonly.update({smoke_stage + '/' + n: h for n, h in smoke['files_sha256'].items()})
    control_source = (HERE / families['control']['scripts']['train_toys_orderfix.py']).read_text(encoding='utf-8')
    readonly.update(extract(control_source, '_INPUT_SHA256'))
    readonly.update(x4['data_sha256'])
    with (HERE.parent / 'all42_training_audit.csv').open(encoding='utf-8', newline='') as s:
        rows = list(csv.DictReader(s))
    selected = [r for r in rows if int(r['stop_epoch']) < 50]
    assert len(selected) == 19
    jobs = []
    for row in selected:
        dataset, method, seed = row['dataset'], row['method'], int(row['seed'])
        family = {'Control':'control','Shuffle':'shuffle','TiSASRec':'tisasrec'}.get(method, 'ml_lbsp' if dataset == 'ml1m' else 'amazon_lbsp')
        identity = dataset + '_' + method.lower() + '_seed' + str(seed)
        base = Path(row['metrics_path']).parent
        pred = base / ('predictions_original.csv' if family == 'tisasrec' else 'test_per_sample_predictions.csv')
        params = x3['inputs']['datasets'][dataset]
        job = {'id': identity, 'family': family, 'method':method, 'dataset':dataset,'seed':seed,
               'old_best_epoch':int(row['best_epoch']),'old_stop_epoch':int(row['stop_epoch']), 'expected_test':params['expected_test'],
               'baseline_predictions':copy(pred.as_posix(), 'baseline/' + identity + '/predictions.csv'),
               'baseline_log':copy(row['training_log_path'], 'baseline/' + identity + '/training_log.csv'),
               'baseline_map':copy((base / 'item_id_map.csv').as_posix(), 'baseline/' + identity + '/item_id_map.csv')}
        job['script_name'] = ('run_x4_tisasrec.py' if family=='tisasrec' else '38_20_train_ml1m_r1_bs128.py' if family=='ml_lbsp' else 'train_' + dataset + '_' + ('orderfix' if family=='control' else 'shuffle' if family=='shuffle' else 'lbsp') + '.py')
        jobs.append(job)
    copy('kais_revision_outline_20261022/p1_review_20261030/e1/all42_training_audit.csv','all42_training_audit.csv')
    policy = {'version':'e1_min50_v1','date':'2026-10-30', 'minimum_epoch_rule_selected':True, 'locked_before_followup_training':True,
              'minimum_epochs':50,'patience':15,'max_epochs':200,
              'selection':'all42 audited; all19 stopping before50 selected, regardless of method/accuracy',
              'rule':'Monitor and select validation-best throughout; stop iff epoch>=50 AND consecutive_bad_epochs>=15; no reset.',
              'rationale':'Bounded post-hoc sensitivity check beyond the observed19/20/30/32-epoch stops;50 is a pragmatic fixed floor, not proof of sufficient training or a tuned optimum.',
              'fresh_training':True,'checkpoint_resume':False,'server_training_completed':False,'jobs':jobs,'families':families,
              'server_readonly_sha256':readonly,'bundle_sha256':{}}
    paths = list((HERE / 'vendor').rglob('*')) + list((HERE / 'baseline').rglob('*')) + [HERE/'all42_training_audit.csv',HERE/'e1_common.py',HERE/'run_e1_min50.py',HERE/'collect_e1_min50.py']
    policy['bundle_sha256'] = {p.relative_to(HERE).as_posix():sha(p) for p in paths if p.is_file()}
    (HERE/'e1_policy.json').write_text(json.dumps(policy,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'selected':len(jobs),'readonly_files':len(readonly),'sealed_bundle_files':len(policy['bundle_sha256'])}))


if __name__ == '__main__':
    main()
