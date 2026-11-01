"""Generate the X7 warm-target reading note directly from audited CSVs."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    with (ROOT / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

def main():
    summary = {(r['dataset'], r['method'], r['population'], r['metric']): r for r in read('method_summary.csv')}
    delta = {(r['dataset'], r['population'], r['metric']): r for r in read('paired_difference_summary.csv')}
    baseline = {(r['dataset'], r['method'], r['population'], r['metric']): r for r in read('analytic_baselines.csv')}
    names = [('ml1m', 'ML-1M'), ('beauty', 'Beauty'), ('sports', 'Sports'), ('toys', 'Toys')]
    lines = ['# X7训练可见目标子集：已保存结果诊断', '',
             '本目录已完成 Control 与 LBSP 在四数据集、三个训练种子上的补充诊断。全部结果复用已验收的预测文件、面板标签及 X5 每用户诊断元数据；没有读取交互数据或模型权重，没有进行训练或推断。', '',
             'warm 指测试目标在训练正交互中出现至少一次；cold 指计数为零。后者不等于 full-item cross-entropy 下该物品嵌入完全未更新。完整严格面板保留为主结果，子集只作补充。', '',
             '## warm 子集准确率', '',
             'NDCG@10 的均值与样本标准差来自种子42、2026、3407；差值先在同一种子内做 LBSP−Control，再汇总三个差值。95%区间为未校正的df=2 Student t区间。', '',
             '| 数据集 | warm N | Control NDCG@10 | LBSP NDCG@10 | 配对差值 | 95% t区间 |',
             '|---|---:|---:|---:|---:|---|']
    for tag, name in names:
        c, l = (summary[(tag, method, 'warm', 'ndcg10')] for method in ('Control', 'LBSP'))
        d = delta[(tag, 'warm', 'ndcg10')]
        lines.append(f"| {name} | {c['n']} | {float(c['mean']):.6f} ± {float(c['sample_sd']):.6f} | {float(l['mean']):.6f} ± {float(l['sample_sd']):.6f} | {float(d['mean_delta']):+.6f} | [{float(d['ci95_lower']):+.6f}, {float(d['ci95_upper']):+.6f}] |")
    lines += ['', '## warm 且 affected 子集', '',
              'affected 沿用主实验五个原始历史排列产生至少两个不同序列的共享掩码；先定义掩码，再与 warm 相交。重叠率按每用户十对输出的Top-10集合交集比例平均，hit-flip按五次输出中是否同时出现命中和未命中计算。', '',
              '| 数据集 | warm affected N | Control overlap@10 | LBSP overlap@10 | Control hit-flip | LBSP hit-flip | 配对ΔNDCG@10 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for tag, name in names:
        n = summary[(tag, 'Control', 'warm_affected', 'ndcg10')]['n']
        values = [float(summary[(tag, method, 'warm_affected', metric)]['mean']) * 100
                  for method, metric in [('Control', 'overlap10'), ('LBSP', 'overlap10'), ('Control', 'hitflip'), ('LBSP', 'hitflip')]]
        d = float(delta[(tag, 'warm_affected', 'ndcg10')]['mean_delta'])
        lines.append(f'| {name} | {n} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% | {values[3]:.2f}% | {d:+.6f} |')
    lines += ['', '顺序敏感性在训练正交互中出现过的目标子集内仍存在；LBSP在这些固定排列中保持完整测得重叠且无命中翻转。准确率差值随数据集和人群变化，不能写成普遍提升。上述 warm-affected 的四个配对NDCG区间均包含零；详细区间见CSV，不能据此判定等效。', '',
              '## Beauty平均名次与解析随机参照', '',
              '以下名次均为数值越小越好；随机参照按同一子集的逐用户合格候选数计算。', '',
              '| 人群 | N | Control平均名次 | LBSP平均名次 | Random解析平均名次 |',
              '|---|---:|---:|---:|---:|']
    for pop in ('full', 'warm', 'cold'):
        c, l = (summary[('beauty', method, pop, 'mean_rank')] for method in ('Control', 'LBSP'))
        random = baseline[('beauty', 'Random', pop, 'mean_rank')]
        lines.append(f"| {pop} | {c['n']} | {float(c['mean']):.2f} | {float(l['mean']):.2f} | {float(random['mean']):.2f} |")
    lines += ['', '训练期目标出现情况与不同人群的平均名次表现相伴，但这是条件人群诊断，不能把完整与warm人群差异写成冷目标造成问题的因果证明。', '',
              '## 文件与复算', '',
              '- `analysis_plan.json`：本次计算之前固定的口径；属于已看过原实验后的诊断计划。',
              '- `per_seed_metrics.csv`、`method_summary.csv`：六个人群、七个准确率指标和两个稳定性指标。',
              '- `paired_differences_per_seed.csv`、`paired_difference_summary.csv`：同种子配对差值及三种子区间。',
              '- `analytic_baselines.csv`：同一六个人群上的Random/Popularity解析期望，没有训练种子误差条。',
              '- `population_counts.csv`：固定人群样本量。ML-1M的cold-affected人群为空，估计值保留空白。',
              '- `verification.json`、`source_manifest.json`：SHA256来源与校验记录。', '',
              '在本地项目根目录使用Python3.10或更新版本执行：', '',
              '```powershell',
              "python 'kais_revision_outline_20261022/g2_x7_diagnostics_20261029/warm_targets/analyze_warm_targets.py' --root 'D:/time_uncertainty_rec'",
              '```', '',
              '脚本只使用Python标准库，需保留原已验收结果目录。该任务无需上传服务器执行；Sports验证曲线和早停原因属于X7的另一部分。当前未改动英文论文。', '']
    (ROOT / 'warm_target_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    source_names = ('method_summary.csv', 'paired_difference_summary.csv', 'analytic_baselines.csv', 'population_counts.csv')
    provenance = {'status': 'GENERATED_READING_NOTE_FROM_AUDITED_CSVS',
                  'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names},
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'output_sha256': hashlib.sha256((ROOT / 'warm_target_summary.md').read_bytes()).hexdigest()}
    (ROOT / 'note_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    manifest = json.loads((ROOT / 'delivery_manifest.json').read_text(encoding='utf-8'))
    for name in ('build_diagnostic_note.py', 'warm_target_summary.md', 'note_provenance.json'):
        manifest['sha256'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    (ROOT / 'delivery_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Generated warm_target_summary.md from audited CSVs.')

if __name__ == '__main__':
    main()
