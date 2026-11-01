"""X7: reproduce warm-target accuracy/stability from sealed saved reports only.

No interaction data, checkpoints, tensor libraries or original research modules.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics

TAGS = ('ml1m', 'beauty', 'sports', 'toys')
SEEDS = (42, 2026, 3407)
METHODS = ('Control', 'LBSP')
PERMS = (101, 202, 303, 404, 505)
POPS = ('full', 'warm', 'cold', 'affected', 'warm_affected', 'cold_affected')
ACCURACY = ('ndcg10', 'hr10', 'hr20', 'recall50', 'mrr', 'mrr10', 'mean_rank')
METRICS = ACCURACY + ('overlap10', 'hitflip')
TC = 4.302652729749464
FULL_N = {'ml1m': 945, 'beauty': 2966, 'sports': 4659, 'toys': 2547}
AFFECTED_N = {'ml1m': 163, 'beauty': 922, 'sports': 1624, 'toys': 842}
DATASET_NAMES = {'ml1m': 'MovieLens-1M', 'beauty': 'Amazon_Beauty', 'sports': 'Amazon_Sports', 'toys': 'Amazon_Toys'}


def rank_metrics(ranks):
    """Seven single-target rank metrics, or missing estimates for n=0."""
    if not ranks:
        return dict.fromkeys(ACCURACY)
    if any(isinstance(r, bool) or not isinstance(r, int) or r < 1 for r in ranks):
        raise ValueError('Ranks must be positive integers')
    n = len(ranks)
    return {'ndcg10': math.fsum(1 / math.log2(r + 1) for r in ranks if r <= 10) / n,
            'hr10': sum(r <= 10 for r in ranks) / n,
            'hr20': sum(r <= 20 for r in ranks) / n,
            'recall50': sum(r <= 50 for r in ranks) / n,
            'mrr': math.fsum(1 / r for r in ranks) / n,
            'mrr10': math.fsum(1 / r for r in ranks if r <= 10) / n,
            'mean_rank': math.fsum(ranks) / n}


def stability(panels, ids):
    if len(panels) != 5:
        raise ValueError('Exactly five fixed permutation panels are required')
    if not ids:
        return {'overlap10': None, 'hitflip': None}
    overlaps, flips = [], []
    for sid in ids:
        values = [panel[sid] for panel in panels]
        overlaps.append(math.fsum(len(a['top'] & b['top']) / 10 for a, b in combinations(values, 2)) / 10)
        flips.append(int(len({v['rank'] <= 10 for v in values}) > 1))
    return {'overlap10': statistics.mean(overlaps), 'hitflip': statistics.mean(flips)}


def seed_stats(values):
    if len(values) != 3:
        raise ValueError('Exactly three training-seed estimates are required')
    if any(v is None for v in values):
        if not all(v is None for v in values):
            raise ValueError('A population cannot be absent for only some seeds')
        return dict.fromkeys(('mean', 'sample_sd', 'ci95_lower', 'ci95_upper'))
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = TC * sd / math.sqrt(3)
    return {'mean': mean, 'sample_sd': sd, 'ci95_lower': mean - half, 'ci95_upper': mean + half}


def select_population(labels, metadata, population):
    if population not in POPS:
        raise ValueError('Unknown population')
    result = []
    for sid, label in labels.items():
        warm = int(metadata[sid]['target_train_count']) >= 1
        affected = bool(int(label['affected']))
        take = {'full': True, 'warm': warm, 'cold': not warm, 'affected': affected,
                'warm_affected': warm and affected, 'cold_affected': not warm and affected}[population]
        if take:
            result.append(sid)
    return result


class Auditor:
    def __init__(self, root, out):
        self.root, self.out = root.resolve(), out.resolve()
        self.c = self.root / 'kais_revision_outline_20261022/g2_closure_20261027/C_statistics'
        self.x5 = self.root / 'g2_x5_results_20261025'
        self.project = self.root / 'kais_revision_outline_20261022/g2_h_manuscript_working_20261028'
        self.inputs, self.counts, self.seals = {}, Counter(), {}
        self.rows, self.population_rows, self.baselines = [], [], []

    def check(self, condition, message):
        self.counts['assertions'] += 1
        if not condition:
            raise AssertionError(message)

    def near(self, a, b, message):
        self.counts['numeric_comparisons'] += 1
        self.check(a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=2e-11, abs_tol=2e-12), message)

    def read(self, path, expected=None):
        path = path.resolve()
        self.check(path.is_relative_to(self.root), 'Input must stay inside workspace')
        relative = path.relative_to(self.root).as_posix()
        self.check(path.suffix in ('.csv', '.json'), 'Only saved CSV/JSON reports are read')
        self.check(not any(p in ('data', 'data_raw', 'checkpoints', 'dataset', 'datasets') for p in path.parts), 'Interaction/checkpoint path rejected')
        # Every statistical input must have a preaccepted seal. Bootstrap manifest
        # files themselves are explicitly requested by initialize() before this map.
        self.check(expected is not None, 'An expected SHA256 is required: ' + relative)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        self.check(digest == expected, 'Accepted input seal mismatch: ' + relative)
        self.inputs[relative] = {'sha256': digest, 'bytes': len(payload)}
        if path.suffix == '.json':
            return json.loads(payload.decode('utf-8-sig'))
        return list(csv.DictReader(payload.decode('utf-8-sig').splitlines()))

    def bootstrap_manifest(self, path):
        """Record existing accepted manifest bytes; never run its referenced code."""
        relative = path.relative_to(self.root).as_posix()
        payload = path.read_bytes()
        self.inputs[relative] = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
        return json.loads(payload.decode('utf-8-sig')) if path.suffix == '.json' else list(csv.DictReader(payload.decode('utf-8-sig').splitlines()))

    def initialize(self):
        cm = self.bootstrap_manifest(self.c / 'source_manifest.json')
        self.seals = {p: value['sha256'] for p, value in cm['files'].items()}
        hm = self.bootstrap_manifest(self.project / 'support/accepted_asset_manifest.json')
        expected = hm['sha256']['evidence/C_statistics/per_seed_metrics.csv']
        c_rows = self.read(self.c / 'per_seed_metrics.csv', expected)
        self.c_reference = {(r['dataset'], r['method'], int(r['training_seed']), r['population'], r['metric']): r for r in c_rows}
        x5_manifest = self.bootstrap_manifest(self.x5 / 'REPORT_SHA256.csv')
        x5_seals = {r['path']: r['sha256'] for r in x5_manifest}
        self.x5_reference = self.read(self.x5 / 'saved_model_per_seed.csv', x5_seals['saved_model_per_seed.csv'])
        self.x5_reference = {(r['model_id'], r['group']): r for r in self.x5_reference}
        self.plan_path = self.out / 'analysis_plan.json'
        plan = self.bootstrap_manifest(self.plan_path)
        self.check(tuple(plan['training_seeds']) == SEEDS and tuple(plan['permutation_seeds']) == PERMS, 'Plan seed contract')
        self.check(tuple(plan['populations']) == POPS, 'Plan population order')

    def accepted(self, path):
        relative = path.relative_to(self.root).as_posix()
        self.check(relative in self.seals, 'File not in accepted C source manifest: ' + relative)
        return self.read(path, self.seals[relative])

    def core(self, tag):
        return self.root / ('g2_lbsp_ml1m_20261023' if tag == 'ml1m' else 'g2_lbsp_amazon_stability_20261025') / 'snapshot'

    def labels_and_metadata(self, tag):
        label_rows = self.accepted(self.core(tag) / 'outputs' / f'{tag}_seed42/panel_labels.csv')
        metadata_rows = self.accepted(self.x5 / 'derived' / f'{tag}_test.csv')
        labels = {r['sample_id']: r for r in label_rows}
        metadata = {r['sample_id']: r for r in metadata_rows}
        self.check(len(labels) == len(label_rows) == len(metadata) == len(metadata_rows) == FULL_N[tag], 'Exact fixed full-panel identity/count')
        self.check(set(labels) == set(metadata), 'X5 and sensitivity panel sample identities match')
        for sid, row in labels.items():
            meta = metadata[sid]
            self.check(row['user_id'] == meta['user_id'] and row['target_item'] == meta['target_item'], 'X5 user/target identity')
            count = int(meta['target_train_count'])
            self.check(count >= 0 and meta['target_group'] == ('warm' if count >= 1 else 'cold'), 'Warm/cold count definition')
            distinct = len({row[f'perm_seed{p}_sequence_sha256'] for p in PERMS})
            self.check(int(row['distinct_sequences_5perm']) == distinct and int(row['affected']) == (distinct >= 2), 'Shared affected definition from raw histories')
            self.check(int(meta['candidate_count']) >= 10, 'At least ten eligible candidates')
        ids = {p: select_population(labels, metadata, p) for p in POPS}
        self.check(len(ids['affected']) == AFFECTED_N[tag], 'Accepted affected count')
        self.check(set(ids['warm']).isdisjoint(ids['cold']) and set(ids['warm']) | set(ids['cold']) == set(labels), 'Warm/cold partition full panel')
        self.check(set(ids['warm_affected']).isdisjoint(ids['cold_affected']) and set(ids['warm_affected']) | set(ids['cold_affected']) == set(ids['affected']), 'Affected warm/cold partition')
        for pop, selected in ids.items():
            self.population_rows.append({'dataset': tag, 'population': pop, 'n': len(selected),
                                         'full_n': len(labels), 'affected_n': len(ids['affected']),
                                         'warm_n': len(ids['warm']), 'cold_n': len(ids['cold'])})
            for baseline in ('random', 'popularity'):
                for metric in ACCURACY:
                    values = [float(metadata[sid][f'{baseline}_{metric}']) for sid in selected]
                    self.baselines.append({'dataset': tag, 'method': baseline.title(), 'population': pop,
                                           'n': len(selected), 'metric': metric,
                                           'mean': math.fsum(values) / len(values) if values else None,
                                           'estimate_kind': 'analytic_population_expectation_no_training_seed_variance'})
        return labels, metadata, ids

    def predictions(self, path, labels, metadata):
        rows = self.accepted(path)
        result = {}
        for row in rows:
            sid = row['sample_id']
            self.check(sid in labels and sid not in result, 'Prediction sample exists and is unique')
            label, meta = labels[sid], metadata[sid]
            self.check(row['user_id'] == label['user_id'] and row['target_item'] == label['target_item'], 'Prediction user/target identity')
            rank = int(row['target_rank'])
            top = tuple(str(v) for v in json.loads(row['top10_raw_item_id']))
            self.check(1 <= rank <= int(meta['candidate_count']), 'Saved target rank within eligible catalogue')
            self.check(len(top) == len(set(top)) == 10, 'Ten distinct Top10 identities')
            self.counts['rank10_top10_membership_mismatches'] += int((rank <= 10) != (row['target_item'] in top))
            result[sid] = {'rank': rank, 'top': frozenset(top)}
        self.check(set(result) == set(labels), 'Complete prediction panel')
        self.counts['prediction_files'] += 1
        self.counts['prediction_rows'] += len(rows)
        return result

    def run(self, tag, method, seed, labels, metadata, ids):
        folder = self.core(tag) / ('control_reference/outputs' if method == 'Control' else 'outputs') / f'{tag}_seed{seed}'
        run_labels = self.accepted(folder / 'panel_labels.csv')
        self.check({r['sample_id']: r for r in run_labels} == labels and len(run_labels) == len(labels), 'Same full raw label records in all runs')
        canonical = self.predictions(folder / 'predictions_original.csv', labels, metadata)
        for sid, value in canonical.items():
            self.check(value['rank'] == int(metadata[sid][f'{tag}_{method.lower()}_seed{seed}_rank']), 'Saved canonical rank equals X5 rank')
        panels = [self.predictions(folder / f'predictions_perm_seed{p}.csv', labels, metadata) for p in PERMS]
        local = {}
        for pop in POPS:
            selected = ids[pop]
            metrics = rank_metrics([canonical[sid]['rank'] for sid in selected]) | stability(panels, selected)
            for metric, value in metrics.items():
                if value is not None:
                    self.check(math.isfinite(value), 'Finite nonempty metric')
                    self.check(value >= 1 if metric == 'mean_rank' else 0 <= value <= 1, 'Metric natural bounds')
                self.rows.append({'dataset': tag, 'method': method, 'training_seed': seed, 'population': pop,
                                  'n': len(selected), 'affected_n': len(ids['affected']),
                                  'accuracy_view': 'canonical', 'metric': metric, 'value': value})
                local[(pop, metric)] = value
                if pop in ('full', 'affected'):
                    prior = self.c_reference[(tag, method, seed, pop, metric)]
                    self.check(int(prior['n']) == len(selected), 'C matching population count')
                    self.near(value, prior['value'], 'Reconciles accepted C metric')
                    self.counts['C_reference_comparisons'] += 1
                if pop in ('full', 'warm', 'cold') and metric in ACCURACY:
                    prior = self.x5_reference[(f'{tag}_{method.lower()}_seed{seed}', 'overall' if pop == 'full' else pop)]
                    self.check(int(prior['samples']) == len(selected), 'X5 matching population count')
                    self.near(value, prior[metric], 'Reconciles accepted X5 canonical metric')
                    self.counts['X5_reference_comparisons'] += 1
        for parent, a, b in (('full', 'warm', 'cold'), ('affected', 'warm_affected', 'cold_affected')):
            for metric in METRICS:
                parts = [len(ids[p]) * local[(p, metric)] for p in (a, b) if ids[p]]
                recomposed = math.fsum(parts) / len(ids[parent])
                self.near(recomposed, local[(parent, metric)], 'Warm/cold weighted reconstruction')
                self.counts['partition_reconstructions'] += 1

    def summarize(self):
        groups = defaultdict(list)
        lookup = {}
        for row in self.rows:
            groups[(row['dataset'], row['method'], row['population'], row['metric'])].append(row)
            lookup[(row['dataset'], row['method'], row['population'], row['training_seed'], row['metric'])] = row
        self.summary, self.differences, self.delta_summary = [], [], []
        for (tag, method, pop, metric), rows in groups.items():
            self.check(tuple(row['training_seed'] for row in rows) == SEEDS, 'Three fixed seeds retained')
            common = {key: rows[0][key] for key in ('dataset', 'method', 'population', 'n', 'affected_n', 'accuracy_view', 'metric')}
            common |= {'training_seeds': ';'.join(map(str, SEEDS)), 'n_training_seeds': 3}
            self.summary.append(common | seed_stats([row['value'] for row in rows]))
            if method != 'LBSP':
                continue
            paired = []
            for row in rows:
                control = lookup[(tag, 'Control', pop, row['training_seed'], metric)]
                self.check(control['n'] == row['n'], 'Paired comparison same exact fixed population')
                value = row['value'] - control['value'] if row['value'] is not None else None
                paired.append(value)
                self.differences.append({key: row[key] for key in ('dataset', 'method', 'population', 'n', 'affected_n', 'training_seed', 'metric', 'accuracy_view')} |
                                        {'method_value': row['value'], 'control_value': control['value'], 'difference': value})
            stats = seed_stats(paired)
            self.delta_summary.append(common | stats | {'mean_delta': stats['mean'], 'sample_sd_delta': stats['sample_sd'], 'comparison': 'LBSP_minus_same_seed_Control'})

    def write_csv(self, name, rows):
        path = self.out / name
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def outputs(self):
        for name, rows in [('per_seed_metrics.csv', self.rows), ('method_summary.csv', self.summary),
                           ('paired_differences_per_seed.csv', self.differences), ('paired_difference_summary.csv', self.delta_summary),
                           ('population_counts.csv', self.population_rows), ('analytic_baselines.csv', self.baselines)]:
            self.write_csv(name, rows)
        lookup = {(r['dataset'], r['method'], r['population'], r['metric']): r for r in self.summary}
        deltas = {(r['dataset'], r['population'], r['metric']): r for r in self.delta_summary}
        findings = {'status': 'COMPLETE_WARM_TARGET_DIAGNOSTIC_DESCRIPTIVE_RESULTS',
                    'population_counts': self.population_rows, 'warm_accuracy_and_stability': [],
                    'interpretation': ['These are additional fixed-population diagnostics, not replacement primary results.',
                                       'Positive training exposure is not randomized; no causal explanation is established.',
                                       'Warm targets do not guarantee a favorable LBSP accuracy direction.',
                                       'A zero positive-interaction count does not imply an untrained embedding under full-item CE.']}
        for tag in TAGS:
            for pop in ('warm', 'warm_affected'):
                row = {'dataset': tag, 'population': pop, 'n': lookup[(tag, 'Control', pop, 'ndcg10')]['n']}
                for method in METHODS:
                    for metric in ('ndcg10', 'mean_rank', 'overlap10', 'hitflip'):
                        stat = lookup[(tag, method, pop, metric)]
                        row[f'{method}_{metric}'] = {'mean': stat['mean'], 'sample_sd': stat['sample_sd']}
                row['paired_ndcg10_difference'] = {key: deltas[(tag, pop, 'ndcg10')][key] for key in ('mean_delta', 'sample_sd_delta', 'ci95_lower', 'ci95_upper')}
                findings['warm_accuracy_and_stability'].append(row)
        self.dump('findings.json', findings)
        report = {'date': '2026-10-29', 'status': 'PASS_SEALED_SAVED_RESULT_WARM_TARGET_ACCURACY_AND_STABILITY',
                  'counts': dict(self.counts), 'model_configurations': 24,
                  'datasets': list(TAGS), 'training_seeds': list(SEEDS), 'populations': list(POPS),
                  'rows': {'per_seed_metrics': len(self.rows), 'method_summary': len(self.summary),
                           'paired_differences_per_seed': len(self.differences), 'paired_difference_summary': len(self.delta_summary),
                           'analytic_baselines': len(self.baselines), 'population_counts': len(self.population_rows)},
                  'source_seal_policy': 'Predictions, labels and X5 diagnostic metadata match accepted C input seals; C reference outputs match accepted manuscript asset seals; X5 reference summaries match REPORT_SHA256.',
                  'raw_interaction_or_checkpoint_read_or_stat': False, 'new_model_training_or_inference': False,
                  'submitted_research_code_executed': False, 'findings': [],
                  'rank_list_disagreements_preserved': True,
                  'source_files': len(self.inputs), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.dump('verification.json', report)
        self.dump('source_manifest.json', {'scope': 'All read saved report and diagnostic metadata inputs, no interactions/checkpoints', 'files': self.inputs})
        outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.out.iterdir() if p.suffix in ('.csv', '.json', '.py') and p.name != 'delivery_manifest.json'}
        self.dump('delivery_manifest.json', {'date': '2026-10-29', 'status': report['status'], 'sha256': outputs})
        print(json.dumps({'status': report['status'], 'checks': dict(self.counts), 'output_rows': report['rows']}))

    def dump(self, name, data):
        (self.out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    auditor = Auditor(args.root, Path(__file__).resolve().parent)
    auditor.initialize()
    for tag in TAGS:
        labels, metadata, ids = auditor.labels_and_metadata(tag)
        for method in METHODS:
            for seed in SEEDS:
                auditor.run(tag, method, seed, labels, metadata, ids)
    auditor.summarize()
    auditor.outputs()


if __name__ == '__main__':
    main()
