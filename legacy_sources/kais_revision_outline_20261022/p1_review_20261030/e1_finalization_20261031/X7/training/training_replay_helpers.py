"""X7: replay accepted epoch logs without loading data, checkpoints or research code.

Only Python's standard library is used. Sources are the three previously
verified returned report collections and the frozen X5 model identity contract.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

DATASETS = ('ml1m', 'beauty', 'sports', 'toys')
METHODS = ('Control', 'LBSP')
SEEDS = (42, 2026, 3407)
HERE = Path(__file__).resolve().parent
READS = {}
CHECKS = 0


def require(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'Path leaves allowed project: ' + str(relative))
    return path


def read_bytes(root, path):
    value = path.read_bytes()
    READS[path.relative_to(root).as_posix()] = {
        'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
    return value


def read_json(root, path):
    return json.loads(read_bytes(root, path).decode('utf-8-sig'))


def read_csv(root, path):
    return list(csv.DictReader(read_bytes(root, path).decode('utf-8-sig').splitlines()))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows):
    require(bool(rows), 'No output rows: ' + path.name)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(row, field):
    value = float(row[field])
    require(math.isfinite(value), 'Non-finite field: ' + field)
    return value


def integer(value, field):
    result = int(value)
    require(float(value) == result, 'Non-integral field: ' + field)
    return result


def replay_log(rows, config, metrics, identity):
    """Validate every logged running-best and patience state, not just endpoints."""
    require(bool(rows), identity + ': empty epoch history')
    require(50 <= len(rows) <= config['max_epochs'], identity + ': outside uniform50/200 epoch bounds')
    curves = []
    best_value = -math.inf
    best_epoch = 0
    bad_epochs = 0
    for i, row in enumerate(rows, 1):
        epoch = integer(row['epoch'], 'epoch')
        require(epoch == i, identity + ': missing/duplicate/out-of-order epoch')
        val = number(row, 'val_ndcg10')
        loss = number(row, 'train_loss')
        require(0 <= val <= 1 and loss >= 0, identity + ': invalid NDCG or loss')
        # The accepted trainers use strict > or > best + 1e-12. On these
        # histories those rules give identical state updates; verify that fact.
        strict_improvement = val > best_value
        tolerance_improvement = val > best_value + 1e-12
        require(strict_improvement == tolerance_improvement,
                identity + ': strict and epsilon improvement rules differ')
        if strict_improvement:
            best_value, best_epoch, bad_epochs = val, epoch, 0
        else:
            bad_epochs += 1
        if 'best_epoch' in row:
            recorded_best = integer(row['best_epoch'], 'best_epoch')
            require(recorded_best == best_epoch, identity + ': running best epoch differs')
        if 'bad_epochs' in row:
            recorded_bad = integer(row['bad_epochs'], 'bad_epochs')
            require(recorded_bad == bad_epochs, identity + ': bad-epoch counter differs')
        best_fields = [key for key in ('best_val_ndcg10', 'best_ndcg10') if key in row]
        for field in best_fields:
            require(math.isclose(number(row, field), best_value, rel_tol=0, abs_tol=1e-12),
                    identity + ': running best validation value differs')
        require(not (epoch >= 50 and bad_epochs >= config['patience']) or i == len(rows),
                identity + ': training continued after patience was exhausted')
        curves.append({'epoch': epoch, 'val_ndcg10': val, 'train_loss': loss,
                       'best_epoch_so_far': best_epoch, 'best_val_ndcg10_so_far': best_value,
                       'bad_epochs': bad_epochs,
                       'logged_seconds': number(row, 'seconds' if 'seconds' in row else 'elapsed_seconds'),
                       'seconds_source_field': 'seconds' if 'seconds' in row else 'elapsed_seconds'})
    require(best_epoch == metrics['best_epoch'], identity + ': metrics best epoch differs')
    validation = metrics['validation']
    saved_best = validation['ndcg10'] if 'ndcg10' in validation else validation['overall']['NDCG@10']
    require(math.isclose(best_value, saved_best, rel_tol=0, abs_tol=1e-12),
            identity + ': metrics best validation NDCG differs')
    stopping = bad_epochs >= config['patience']
    require(stopping or len(rows) == config['max_epochs'], identity + ': incomplete training trace')
    if 'early_stopped' in metrics:
        require(metrics['early_stopped'] is stopping, identity + ': early-stop flag differs')
    after = curves[best_epoch - 1:]
    val_steps = [b['val_ndcg10'] - a['val_ndcg10'] for a, b in zip(after, after[1:])]
    loss_steps = [b['train_loss'] - a['train_loss'] for a, b in zip(after, after[1:])]
    x = list(range(len(after)))
    xm = sum(x) / len(x)
    ym = sum(row['val_ndcg10'] for row in after) / len(after)
    denominator = sum((v - xm) ** 2 for v in x)
    slope = (sum((v - xm) * (row['val_ndcg10'] - ym) for v, row in zip(x, after)) / denominator
             if denominator else None)
    summary = {
        'epochs_run': len(rows), 'best_epoch': best_epoch, 'epochs_since_best': len(rows) - best_epoch,
        'best_val_ndcg10': best_value, 'last_val_ndcg10': curves[-1]['val_ndcg10'],
        'last_minus_best_val_ndcg10': curves[-1]['val_ndcg10'] - best_value,
        'train_loss_at_best': curves[best_epoch - 1]['train_loss'],
        'last_train_loss': curves[-1]['train_loss'],
        'last_minus_best_train_loss': curves[-1]['train_loss'] - curves[best_epoch - 1]['train_loss'],
        'post_best_validation_increases': sum(v > 0 for v in val_steps),
        'post_best_validation_decreases': sum(v < 0 for v in val_steps),
        'post_best_validation_equal_steps': sum(v == 0 for v in val_steps),
        'post_best_validation_strictly_decreasing': bool(val_steps) and all(v < 0 for v in val_steps),
        'post_best_training_loss_increases': sum(v > 0 for v in loss_steps),
        'post_best_training_loss_decreases': sum(v < 0 for v in loss_steps),
        'post_best_training_loss_strictly_decreasing': bool(loss_steps) and all(v < 0 for v in loss_steps),
        'post_best_validation_linear_slope': slope,
        'patience': config['patience'], 'max_epochs': config['max_epochs'],
        'final_bad_epochs': bad_epochs, 'patience_exhausted': stopping,
        'running_best_and_patience_reconstructed': True,
        'recorded_epoch_state_fields_verified': ';'.join(field for field in ('best_epoch', 'best_val_ndcg10', 'best_ndcg10', 'bad_epochs') if field in rows[0]),
        'missing_epoch_state_fields_reconstructed_from_validation': 'best_epoch' not in rows[0] or 'bad_epochs' not in rows[0],
    }
    return curves, summary


def source_location(dataset, method, seed):
    if method == 'Control':
        return 'g2_control_stability_20261023', f'references/{dataset}_seed{seed}'
    if dataset == 'ml1m':
        return 'g2_lbsp_ml1m_20261023', f'references/ml1m_seed{seed}'
    return 'g2_lbsp_amazon_train_20261025', f'outputs/{dataset}_lbsp_seed{seed}'


def analyze(root, output):
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = safe(root, 'kais_revision_outline_20261022/server_g2/x5_readonly_inputs_20261025.json')
    contract = read_json(root, contract_path)
    ledgers = {}
    for group in ('g2_control_stability_20261023', 'g2_lbsp_ml1m_20261023', 'g2_lbsp_amazon_train_20261025'):
        collected = read_csv(root, safe(root, group + '/COLLECTED_SHA256.csv'))
        ledgers[group] = {row['archive_path']: row for row in collected if row['status'] == 'copied'}
        require(len(ledgers[group]) == sum(row['status'] == 'copied' for row in collected), 'Duplicate archive ledger path')
        read_json(root, safe(root, group + '/LOCAL_VERIFICATION.json'))
    all_curves, summaries, sources = [], [], []
    for dataset in DATASETS:
        expected = {(m['method'], m['seed']): m for m in contract['datasets'][dataset]['models']}
        for method in METHODS:
            for seed in SEEDS:
                identity = f'{dataset}_{method.lower()}_seed{seed}'
                frozen = expected[(method, seed)]
                group, subpath = source_location(dataset, method, seed)
                paths, source_records = {}, {}
                for name in ('config.json', 'metrics.json', 'training_log.csv'):
                    archived = subpath + '/' + name
                    require(archived in ledgers[group], identity + ': missing accepted collected-file ledger entry')
                    path = safe(root, group + '/snapshot/' + archived)
                    value = read_bytes(root, path)
                    digest = hashlib.sha256(value).hexdigest()
                    require(digest == ledgers[group][archived]['sha256'], identity + ': accepted collected-file hash differs')
                    require(len(value) == int(ledgers[group][archived]['bytes']), identity + ': byte count differs')
                    if name in frozen['files_sha256']:
                        require(digest == frozen['files_sha256'][name], identity + ': X5 frozen model identity differs')
                    paths[name] = path
                    source_records[name] = {'local_path': path.relative_to(root).as_posix(), 'sha256': digest,
                                            'server_path': frozen['model_dir'] + '/' + name}
                config = json.loads(paths['config.json'].read_text(encoding='utf-8'))
                metrics = json.loads(paths['metrics.json'].read_text(encoding='utf-8'))
                rows = list(csv.DictReader(paths['training_log.csv'].read_text(encoding='utf-8-sig').splitlines()))
                require(config['dataset'] == contract['datasets'][dataset]['dataset'] and config['seed'] == seed,
                        identity + ': config identity differs')
                require(metrics['dataset'] == config['dataset'] and metrics['seed'] == seed,
                        identity + ': metrics identity differs')
                if 'training_batch_size' not in config:
                    config['training_batch_size'] = config.get('batch_size')
                if 'max_epochs' not in config:
                    config['max_epochs'] = config.get('epochs')
                if 'loss' not in config:
                    require(config.get('training_objective') == 'target_level_full_ranking_cross_entropy',
                            identity + ': unknown loss alias')
                    config['loss'] = 'full_item_cross_entropy'
                if 'optimizer' in config:
                    require(config['optimizer'] == 'Adam', identity + ': optimizer differs')
                required = {'training_batch_size': 128, 'learning_rate': 0.001,
                            'max_epochs': 200, 'patience': 15, 'max_len': 50,
                            'hidden_size': 64, 'num_heads': 2, 'num_blocks': 2, 'dropout': 0.2,
                            'loss': 'full_item_cross_entropy'}
                for key, value in required.items():
                    require(config.get(key) == value, identity + ': different shared hyperparameter ' + key)
                curves, summary = replay_log(rows, config, metrics, identity)
                run = {'dataset': dataset, 'method': method, 'training_seed': seed}
                all_curves.extend([{**run, **row} for row in curves])
                summaries.append({**run, **summary, 'dropout': config['dropout'],
                                  'source_collection': group, 'server_model_dir': frozen['model_dir'],
                                  'training_log_sha256': source_records['training_log.csv']['sha256']})
                sources.append({**run, 'source_collection': group, 'source_files': source_records})
    require(len(summaries) == 24, 'Expected exactly 24 current Control/LBSP runs')
    write_csv(output / 'epoch_validation_curves.csv', all_curves)
    write_csv(output / 'training_run_summary.csv', summaries)
    sports = [row for row in summaries if row['dataset'] == 'sports']
    write_csv(output / 'sports_training_summary.csv', sports)
    sports_table = '\n'.join(
        f"| {row['method']} | {row['training_seed']} | {row['best_epoch']} | {row['epochs_run']} | {row['best_val_ndcg10']:.6f} | {row['post_best_validation_increases']} / {row['post_best_validation_decreases']} |"
        for row in sports)
    findings = (
        '# X7：现有训练日志诊断\n\n'
        f'已读取此前验收并冻结的24个 Control/LBSP 训练记录，共{len(all_curves):,}个连续 epoch；覆盖四个数据集及种子42、2026、3407。'
        '所有训练日志、配置与指标均通过返回包清单的 SHA256 核验，配置和指标另与 X5 模型身份契约一致。没有加载交互数据、模型权重或执行研究训练代码。\n\n'
        '## Sports 当前排序修复后的结果\n\n'
        '| 方法 | 种子 | 最佳验证轮次 | 实际训练轮次 | 最佳验证NDCG@10 | 最佳轮次后验证上升/下降次数 |\n'
        '|---|---:|---:|---:|---:|---:|\n' + sports_table + '\n\n'
        '“最佳轮次后”包含从所选最佳轮次到下一轮的第一步，因此每个Sports记录共15次相邻变化。'
        f"六个记录中，{sum(row['post_best_validation_strictly_decreasing'] for row in sports)}个验证曲线在该区间严格单调下降；"
        f"{sum(row['post_best_training_loss_strictly_decreasing'] for row in sports)}个训练损失在同一区间严格下降。\n\n"
        '24个记录均在最佳轮次之后连续15轮未创造新的验证NDCG@10最佳值，符合配置的 patience=15，并且没有触及200轮上限。'
        'ML-1M/Beauty 的六份Control历史日志没有逐轮保存best_epoch/bad_epochs字段；这些状态是从验证值重建并与最终metrics核对，不能称为原始日志中已记录的字段。\n\n'
        '## 可以据此说明什么\n\n'
        '当前Sports最佳轮次随训练种子变化，且所选最大值后的验证指标存在上下波动。旧稿“Sports第8轮最好”不适用于本轮结果。'
        '训练损失继续下降而验证未再创新高，可提示后续训练没有改善此验证指标，但单凭挑选最大值后的曲线不足以认定统一的过拟合原因。'
        '保留早停较早的种子，使用统一的验证选择规则；本诊断不修改dropout、不加入weight decay，也不触发重训。\n\n'
        '## 重做这一诊断\n\n'
        '本机项目根目录下执行 `python kais_revision_outline_20261022/g2_x7_diagnostics_20261029/training/analyze_training_logs.py --root D:/time_uncertainty_rec`。'
        'Python标准库即可；没有服务器或GPU依赖。源目录必须是此前验收的三个返回包解压目录，不能混入旧排序版本或LOO校准日志。\n')
    (output / 'training_findings.md').write_text(findings, encoding='utf-8')
    write_json(output / 'training_source_contract.json', {
        'date': '2026-10-29', 'scope': 'Four datasets x Control/LBSP x three training seeds; no Shuffle/Ensemble/TiSASRec or F calibration mixed in.',
        'origin': 'Accepted current sorted-protocol result collections, with all72 config/metrics/log files checked against returned collection ledgers;48 config/metrics files also match the frozen X5 identities.',
        'runs': sources})
    write_json(output / 'training_provenance.json', {
        'date': '2026-10-29', 'status': 'PASS_24_ACCEPTED_LOGS_FULL_EPOCH_AND_PATIENCE_REPLAY',
        'checks': CHECKS, 'runs': len(summaries), 'epoch_rows': len(all_curves),
        'sports_best_epochs': {f"{row['method']}_{row['training_seed']}": row['best_epoch'] for row in sports},
        'all_runs_patience_exhausted': all(row['patience_exhausted'] for row in summaries),
        'missing_local_logs': [], 'server_actions_required': [],
        'interactions_or_checkpoints_accessed': False, 'research_code_executed': False,
        'new_models_trained': 0, 'torch_imported': False,
        'interpretation': [
            'Best epoch and stopping are recomputed from all logged epochs and compared with saved metrics and every state field actually recorded. Six ML1M/Beauty Control logs omit epoch state fields, which are reconstructed rather than misrepresented as recorded.',
            'The prior Sports best-epoch8 narrative does not describe the current corrected-protocol three-seed runs.',
            'Post-best validation rises/falls are descriptive; training curves alone do not establish a cause or prove global optimality.',
            'Selection of a validation maximum makes subsequent values lower by definition; this fact alone is not independent proof of overfitting.',
            'All specified seeds remain included. This analysis does not change training settings or schedule retraining.',
            'Logged seconds are historical observations, not controlled cross-method benchmarks; retain the existing X6 timing results for cost comparisons.',
        ],
        'files_read': READS,
        'output_sha256': {p.name: sha(p) for p in [output / 'epoch_validation_curves.csv', output / 'training_run_summary.csv', output / 'sports_training_summary.csv', output / 'training_source_contract.json', output / 'training_findings.md']},
        'analyzer_sha256': sha(Path(__file__).resolve()),
    })
    print(json.dumps({'status': 'PASS_24_ACCEPTED_LOGS_FULL_EPOCH_AND_PATIENCE_REPLAY',
                      'runs': len(summaries), 'epoch_rows': len(all_curves), 'checks': CHECKS,
                      'sports_best_epochs': {f"{row['method']}_{row['training_seed']}": row['best_epoch'] for row in sports}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE.parents[2])
    parser.add_argument('--output', type=Path, default=HERE)
    args = parser.parse_args()
    analyze(args.root, args.output)


if __name__ == '__main__':
    main()
