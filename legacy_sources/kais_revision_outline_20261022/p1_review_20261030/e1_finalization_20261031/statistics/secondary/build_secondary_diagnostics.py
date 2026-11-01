"""Current three-seed secondary diagnostics from C-sealed saved records only.

Equal weight is given to each actually distinct input pair; repeated views keep
their multiplicities. Rank dispersion uses the five saved ranks (population SD)
and is summarized over affected users. This never reads interaction data/models.
"""
from collections import defaultdict
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
H = HERE.parent
CLOSURE = H.parent
ROOT = Path(r"D:/time_uncertainty_rec")
C = HERE.parent
PERMS = (101, 202, 303, 404, 505)
SEEDS = (42, 2026, 3407)
RUNS = [(d, 'SASRec') for d in ('ml1m', 'beauty', 'sports', 'toys')] + [(d, 'TiSASRec') for d in ('ml1m', 'beauty')]
NAMES = {'ml1m': 'ML-1M', 'beauty': 'Beauty', 'sports': 'Sports', 'toys': 'Toys'}
METRICS = ('distinct_pair_overlap10', 'distinct_pair_exact_rank_change', 'distinct_pair_mean_abs_rank_difference',
           'affected_median_rank_population_sd', 'affected_median_candidate_normalized_rank_population_sd')


def sha(content):
    return hashlib.sha256(content).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)




def resolve_output(ds,backbone,seed):
    mapping=json.loads((C/'run_overlay.json').read_text(encoding='utf-8'))['runs']
    method='Control' if backbone=='SASRec' else 'TiSASRec'
    row=next(r for r in mapping if r['dataset']==ds and r['method']==method and r['training_seed']==seed)
    return row['output_directory']


def main():
    manifest_path = C / 'source_manifest.json'
    content = manifest_path.read_bytes()
    manifest = json.loads(content)['files']
    sealed_closure = json.loads((HERE / 'analysis_source_seal.json').read_text(encoding='utf-8'))
    assert sealed_closure['sha256']['C_statistics/source_manifest.json'] == sha(content)
    source_seals = {str(manifest_path): sha(content), str(C / 'run_overlay.json'): sha((C / 'run_overlay.json').read_bytes())}
    read_counts = defaultdict(int)
    checks = 0

    def check(value, label):
        nonlocal checks
        checks += 1
        if not value:
            raise ValueError(label)

    def read(relative):
        check(relative in manifest, 'File is explicitly C-sealed: ' + relative)
        allowed = (relative.endswith('/panel_labels.csv') or
                   any(relative.endswith('/predictions_perm_seed' + str(seed) + '.csv') for seed in PERMS) or
                   relative in ['g2_x5_results_20261025/derived/' + ds + '_test.csv' for ds in NAMES])
        check(allowed and not any(part in {'data', 'processed', 'checkpoints'} for part in Path(relative).parts), 'Saved-report whitelist')
        data = (ROOT / relative).read_bytes()
        check(sha(data) == manifest[relative]['sha256'], 'Input matches C source SHA256')
        source_seals[str(ROOT / relative)] = sha(data)
        read_counts['files'] += 1
        rows = list(csv.DictReader(data.decode('utf-8-sig').splitlines()))
        read_counts['rows'] += len(rows)
        return rows

    cache = {}
    def read_once(relative):
        if relative not in cache:
            cache[relative] = read(relative)
        return cache[relative]

    def core(ds):
        return ('g2_lbsp_ml1m_20261023' if ds == 'ml1m' else 'g2_lbsp_amazon_stability_20261025') + '/snapshot'

    per_seed = []
    evidence = []
    for ds, backbone in RUNS:
        labels_raw = read_once(core(ds) + '/outputs/' + ds + '_seed42/panel_labels.csv')
        labels = {row['sample_id']: row for row in labels_raw}
        check(len(labels) == len(labels_raw), 'Unique shared sample labels')
        candidates_raw = read_once('g2_x5_results_20261025/derived/' + ds + '_test.csv')
        candidates = {row['sample_id']: row for row in candidates_raw}
        check(set(candidates) == set(labels), 'C-sealed X5 diagnostic candidate population matches')
        for sid, label in labels.items():
            count = int(candidates[sid]['candidate_count'])
            check(count >= 10, 'Positive eligible candidate count')
            check(label['user_id'] == candidates[sid]['user_id'] and label['target_item'] == candidates[sid]['target_item'], 'Candidate diagnostic identity')
            distinct = len({label['perm_seed' + str(p) + '_sequence_sha256'] for p in PERMS})
            check(distinct == int(label['distinct_sequences_5perm']) and bool(int(label['affected'])) == (distinct >= 2), 'Affected population matches actual five-history hashes')
        for seed in SEEDS:
            out = resolve_output(ds, backbone, seed)
            own_labels = read_once(out + '/panel_labels.csv')
            check({r['sample_id']: int(r['affected']) for r in own_labels} == {sid: int(r['affected']) for sid, r in labels.items()}, 'Same affected population in this backbone/seed')
            panels = []
            for p in PERMS:
                raw = read(out + '/predictions_perm_seed' + str(p) + '.csv')
                panel = {}
                for row in raw:
                    sid = row['sample_id']
                    check(sid in labels and sid not in panel, 'Unique saved prediction identity')
                    label = labels[sid]
                    check(row['target_item'] == label['target_item'] and row['user_id'] == label['user_id'], 'Saved target/user unchanged')
                    rank = int(row.get('target_rank', row.get('rank')))
                    top = tuple(json.loads(row['top10_raw_item_id']))
                    check(len(set(top)) == len(top) == 10 and 1 <= rank <= int(candidates[sid]['candidate_count']), 'Rank/list dimensions valid')
                    if 'candidate_count' in row:
                        check(int(row['candidate_count']) == int(candidates[sid]['candidate_count']), 'X4 candidate count matches sealed X5')
                    panel[sid] = (rank, top)
                check(set(panel) == set(labels), 'Complete saved five-permutation panel')
                panels.append(panel)
            pair_n = overlap_intersections = rank_changed = absolute_rank_sum = 0
            sd_values, normalized_sd = [], []
            affected_n = 0
            for sid, label in labels.items():
                groups = defaultdict(list)
                for j, p in enumerate(PERMS):
                    groups[label['perm_seed' + str(p) + '_sequence_sha256']].append(panels[j][sid])
                for group in groups.values():
                    check(len(set(group)) == 1, 'Identical realized inputs give identical saved output')
                if len(groups) < 2:
                    continue
                affected_n += 1
                # Primary arithmetic uses distinct-history groups and exact integer
                # multiplicities, instead of enumerating user pairs directly.
                group_values = list(groups.values())
                for a, b in combinations(group_values, 2):
                    weight = len(a) * len(b)
                    left, right = a[0], b[0]
                    pair_n += weight
                    overlap_intersections += weight * len(set(left[1]) & set(right[1]))
                    rank_changed += weight * int(left[0] != right[0])
                    absolute_rank_sum += weight * abs(left[0] - right[0])
                ranks = [panel[sid][0] for panel in panels]
                sd = statistics.pstdev(ranks)
                sd_values.append(sd)
                normalized_sd.append(sd / int(candidates[sid]['candidate_count']))
            check(affected_n == sum(int(row['affected']) for row in labels.values()), 'Affected denominator retained')
            check(pair_n > 0, 'Distinct input pairs available')
            row = {'dataset': ds, 'backbone': backbone, 'training_seed': seed,
                   'full_panel_users': len(labels), 'affected_users': affected_n, 'distinct_input_pairs': pair_n,
                   'distinct_pair_overlap10': overlap_intersections / (10 * pair_n),
                   'distinct_pair_exact_rank_change': rank_changed / pair_n,
                   'distinct_pair_mean_abs_rank_difference': absolute_rank_sum / pair_n,
                   'affected_median_rank_population_sd': statistics.median(sd_values),
                   'affected_median_candidate_normalized_rank_population_sd': statistics.median(normalized_sd)}
            per_seed.append(row)
            evidence.append({'dataset': ds, 'backbone': backbone, 'training_seed': seed,
                             'distinct_input_pairs': pair_n, 'top10_intersection_total': overlap_intersections,
                             'exact_rank_changed_pairs': rank_changed, 'absolute_rank_difference_total': absolute_rank_sum})
    summary = []
    for ds, backbone in RUNS:
        rows = [r for r in per_seed if r['dataset'] == ds and r['backbone'] == backbone]
        check(len(rows) == 3, 'Exactly three model seeds')
        for count_name in ('full_panel_users', 'affected_users', 'distinct_input_pairs'):
            check(len({r[count_name] for r in rows}) == 1, 'Fixed denominator has no fake seed SD')
        values = {'dataset': ds, 'backbone': backbone, 'training_seeds': '42;2026;3407',
                  'n_training_seeds': 3, 'full_panel_users': rows[0]['full_panel_users'],
                  'affected_users': rows[0]['affected_users'], 'distinct_input_pairs': rows[0]['distinct_input_pairs']}
        for metric in METRICS:
            values[metric + '_mean'] = statistics.mean(r[metric] for r in rows)
            values[metric + '_sample_sd'] = statistics.stdev(r[metric] for r in rows)
        summary.append(values)
    check(len(per_seed) == 18 and len(summary) == 6, '18 current models, six backbone-dataset summaries')
    csv_write(HERE / 'secondary_per_seed.csv', per_seed)
    csv_write(HERE / 'secondary_three_seed_summary.csv', summary)

    def uncertainty(mean, sd, scale, decimals):
        m, s = round(mean * scale, decimals), round(sd * scale, decimals)
        digits = int(round(s * 10**decimals))
        return f'{m:.{decimals}f}({digits})'
    lines = [r'% Current C-sealed predictions; generated by build_secondary_diagnostics.py.',
             r'\begin{table*}[t]', r'\centering',
             r'\caption{Secondary diagnostics conditional on actually distinct inputs. Pair statistics weight all eligible input pairs equally across users. Each estimate reports mean $\pm$ sample SD across three training seeds; counts are fixed panel descriptors.}',
             r'\label{tab:secondary}', r'\fontsize{8.3}{10.5}\selectfont', r'\setlength{\tabcolsep}{2.2pt}',
             r'\sisetup{separate-uncertainty=true,retain-zero-uncertainty=true,group-separator={,},group-minimum-digits=4}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llS[table-format=4.0]S[table-format=5.0]S[table-format=2.1(3)]S[table-format=2.1(3)]S[table-format=4.0(4)]S[table-format=2.3(4)]@{}}',
             r'\toprule',
             r'Dataset & Backbone & {\shortstack{Affected\\ users}} & {\shortstack{Distinct\\ pairs}} & {\shortstack{Overlap\\(\%)}} & {\shortstack{Exact rank\\change (\%)}} & {\shortstack{Absolute\\rank diff.}} & {\shortstack{Median\\norm. SD (\%)}}\\',
             r'\midrule']
    for row in summary:
        cells = [NAMES[row['dataset']], row['backbone'], str(row['affected_users']), str(row['distinct_input_pairs'])]
        for key, scale, decimals in [('distinct_pair_overlap10', 100, 1), ('distinct_pair_exact_rank_change', 100, 1),
                                      ('distinct_pair_mean_abs_rank_difference', 1, 0),
                                      ('affected_median_candidate_normalized_rank_population_sd', 100, 3)]:
            cells.append(uncertainty(row[key + '_mean'], row[key + '_sample_sd'], scale, decimals))
        lines.append(' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular*}',
              r'\par\smallskip\begin{minipage}{\textwidth}\fontsize{8}{10}\selectfont For each user, only pairs with different saved raw-input sequence hashes contribute to the first three estimates. Repeated views retain multiplicity, so users contribute different numbers of pairs. This pair-weighted overlap differs from the equal-user overlap in the main sensitivity table. Normalized dispersion uses the population SD of the five saved ranks divided by the eligible candidate count, followed by the median across affected users within each seed; the table then averages those three medians. Its SD is between-seed variation of medians, not the within-user rank SD. Candidate counts come from C-sealed per-user X5 diagnostic metadata. The raw rank-SD medians and unrounded values are supplied in the CSV outputs. No rows are available for TiSASRec on Sports or Toys.\end{minipage}',
              r'\end{table*}']
    (HERE / 'appendix_table_secondary.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    provenance = {'status': 'CURRENT_THREE_SEED_SECONDARY_DIAGNOSTICS_BUILT_VERIFICATION_RECORDED_SEPARATELY',
                  'date': '2026-10-31', 'checks': checks, 'read_counts': dict(read_counts),
                  'source_seals': source_seals, 'source_manifest_sha256': sha(content),
                  'exact_integer_pair_totals': evidence, 'runs': len(per_seed), 'summary_rows': len(summary),
                  'definitions': {'pair_population': 'All unordered pairs among five saved permutations whose raw input hashes differ; equal weight per actual distinct-input pair, including repeated logical views.',
                                  'sd': 'Per-user populationSD over all five permutation ranks, dividing by5.',
                                  'normalized_sd': 'Per-user populationSD divided by C-sealed X5 eligible candidate count; median among affected users within seed, then three-seed mean and sampleSD.',
                                  'denominator_counts': 'Fixed panel/affected/pair counts are reported without fabricated SD.',
                                  'X4_hash_identity': 'C shared raw-item history hash labels; TiSASRec mask/candidate counts and saved user-target identities crosschecked.'},
                  'interaction_or_checkpoint_read_or_stat': False, 'model_or_research_code_executed': False,
                  'source_data_modified': False, 'GPU_used': False, 'complete_manuscript_generated': False,
                  'output_sha256': {name: sha((HERE / name).read_bytes()) for name in ('secondary_per_seed.csv', 'secondary_three_seed_summary.csv', 'appendix_table_secondary.tex')}}
    dump(HERE / 'secondary_provenance.json', provenance)
    print(json.dumps({'status': provenance['status'], 'runs': len(per_seed), 'rows': len(summary), 'checks': checks, 'source_files': len(source_seals)}))


if __name__ == '__main__':
    main()
