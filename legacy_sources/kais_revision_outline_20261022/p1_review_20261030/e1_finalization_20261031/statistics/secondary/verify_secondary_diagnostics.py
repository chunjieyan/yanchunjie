"""Independent direct-pair arithmetic for the current secondary appendix table."""
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:/time_uncertainty_rec")
C = HERE.parent
PERMS = (101, 202, 303, 404, 505)
SEEDS = (42, 2026, 3407)
FIELDS = ('distinct_pair_overlap10', 'distinct_pair_exact_rank_change', 'distinct_pair_mean_abs_rank_difference',
          'affected_median_rank_population_sd', 'affected_median_candidate_normalized_rank_population_sd')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))




def resolve_output(ds,backbone,seed):
    mapping=json.loads((C/'run_overlay.json').read_text(encoding='utf-8'))['runs']
    method='Control' if backbone=='SASRec' else 'TiSASRec'
    row=next(r for r in mapping if r['dataset']==ds and r['method']==method and r['training_seed']==seed)
    return row['output_directory']


def main():
    provenance = json.loads((HERE / 'secondary_provenance.json').read_text(encoding='utf-8'))
    manifest = json.loads((C / 'source_manifest.json').read_text(encoding='utf-8'))['files']
    tests = 0
    comparisons = 0
    direct_pair_count = 0
    def check(value, label):
        nonlocal tests
        tests += 1
        if not value:
            raise ValueError(label)
    def near(a, b, label):
        nonlocal comparisons
        comparisons += 1
        check(math.isclose(float(a), float(b), rel_tol=3e-13, abs_tol=3e-12), label)
    cache = {}
    def saved(relative):
        if relative in cache:
            return cache[relative]
        check(relative in manifest, 'Source explicitly C sealed')
        check(relative.endswith('/panel_labels.csv') or '/predictions_perm_seed' in relative or relative.startswith('g2_x5_results_20261025/derived/'), 'Saved metadata whitelist')
        data = (ROOT / relative).read_bytes()
        check(digest(data) == manifest[relative]['sha256'] == provenance['source_seals'][str(ROOT / relative)], 'Source SHA256 identical in C and new provenance')
        cache[relative] = list(csv.DictReader(data.decode('utf-8-sig').splitlines()))
        return cache[relative]
    for name, expected in provenance['source_seals'].items():
        check(digest(Path(name).read_bytes()) == expected, 'All input source seals remain unchanged')
    for name, expected in provenance['output_sha256'].items():
        check(digest((HERE / name).read_bytes()) == expected, 'Output seal valid')
    per_seed = table(HERE / 'secondary_per_seed.csv')
    summary = table(HERE / 'secondary_three_seed_summary.csv')
    check(len(per_seed) == 18 and len(summary) == 6, 'Current18 runs and six summaries')
    independent = []
    pair_evidence = {}
    for row in per_seed:
        ds, backbone, seed = row['dataset'], row['backbone'], int(row['training_seed'])
        core = ('g2_lbsp_ml1m_20261023' if ds == 'ml1m' else 'g2_lbsp_amazon_stability_20261025') + '/snapshot'
        labels = {value['sample_id']: value for value in saved(core + '/outputs/' + ds + '_seed42/panel_labels.csv')}
        candidates = {value['sample_id']: value for value in saved('g2_x5_results_20261025/derived/' + ds + '_test.csv')}
        out = resolve_output(ds, backbone, seed)
        outputs = [{value['sample_id']: value for value in saved(out + '/predictions_perm_seed' + str(p) + '.csv')} for p in PERMS]
        intersections, changed, absolute, sd_values, normalized = [], [], [], [], []
        affected_n = 0
        for sid, label in labels.items():
            hashes = [label['perm_seed' + str(p) + '_sequence_sha256'] for p in PERMS]
            ranks = [int(value[sid].get('target_rank', value[sid].get('rank'))) for value in outputs]
            sets = [set(json.loads(value[sid]['top10_raw_item_id'])) for value in outputs]
            pairs = [(i, j) for i, j in combinations(range(5), 2) if hashes[i] != hashes[j]]
            check(bool(pairs) == bool(int(label['affected'])), 'Direct pair availability equals affected label')
            if not pairs:
                continue
            affected_n += 1
            for i, j in pairs:
                intersections.append(len(sets[i].intersection(sets[j])))
                changed.append(int(ranks[i] != ranks[j]))
                absolute.append(abs(ranks[i] - ranks[j]))
            # Algebraically exact integer population-variance numerator, distinct
            # from the builder's statistics.pstdev implementation.
            numerator = 5 * sum(rank * rank for rank in ranks) - sum(ranks)**2
            check(numerator >= 0, 'Nonnegative exact population variance')
            sd = math.sqrt(numerator) / 5
            sd_values.append(sd)
            normalized.append(sd / int(candidates[sid]['candidate_count']))
        n = len(intersections)
        direct_pair_count += n
        check(n == int(row['distinct_input_pairs']), 'Direct pair-list denominator')
        check(affected_n == int(row['affected_users']) and len(labels) == int(row['full_panel_users']), 'Fixed denominator counts')
        values = [sum(intersections) / (10 * n), sum(changed) / n, sum(absolute) / n,
                  statistics.median(sd_values), statistics.median(normalized)]
        result = {'dataset': ds, 'backbone': backbone, 'training_seed': seed}
        for field, actual in zip(FIELDS, values):
            near(actual, row[field], 'Independent per-run arithmetic: ' + field)
            result[field] = actual
        independent.append(result)
        pair_evidence[(ds, backbone, seed)] = (n, sum(intersections), sum(changed), sum(absolute))
    for entry in provenance['exact_integer_pair_totals']:
        key = (entry['dataset'], entry['backbone'], entry['training_seed'])
        check(pair_evidence[key] == tuple(entry[name] for name in ('distinct_input_pairs', 'top10_intersection_total', 'exact_rank_changed_pairs', 'absolute_rank_difference_total')), 'Exact integer direct-pair totals independently reproduced')
    for row in summary:
        rows = [value for value in independent if value['dataset'] == row['dataset'] and value['backbone'] == row['backbone']]
        check(len(rows) == 3 and {value['training_seed'] for value in rows} == set(SEEDS), 'All three actual fitted-model seeds retained')
        for field in FIELDS:
            values = [value[field] for value in rows]
            near(sum(values) / 3, row[field + '_mean'], 'Three seed mean: ' + field)
            sd = math.sqrt(sum((value - sum(values)/3)**2 for value in values) / 2)
            near(sd, row[field + '_sample_sd'], 'SampleSDddof1: ' + field)
    result = {'status': 'PASS_INDEPENDENT_DIRECT_PAIR_AND_RANK_VARIANCE_ARITHMETIC',
              'checks': tests, 'numeric_comparisons': comparisons, 'source_files': len(cache),
              'direct_distinct_input_pairs_replayed': direct_pair_count,
              'runs': 18, 'summary_rows': 6, 'candidate_normalization_available': True,
              'no_fake_SD_for_denominator_counts': True,
              'independent_variance_formula': 'sqrt(5*sum(rank^2)-sum(rank)^2)/5',
              'source_provenance_sha256': digest((HERE / 'secondary_provenance.json').read_bytes()),
              'output_sha256': provenance['output_sha256'],
              'interaction_or_checkpoint_read_or_stat': False, 'model_or_research_module_executed': False,
              'GPU_used': False, 'complete_manuscript_generated': False}
    (HERE / 'secondary_independent_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': tests, 'numeric_comparisons': comparisons, 'pairs': direct_pair_count}))


if __name__ == '__main__':
    main()
