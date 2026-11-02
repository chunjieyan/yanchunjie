"""Revise table presentation using the sealed original-precision numeric trace."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
TABLES = ROOT.parents[1] / 'tables'
TEMPLATES = ROOT / 'templates'
TRACE = ROOT / 'inputs/formal_numeric_cells.json'
TARGETS = {'table2_accuracy.tex', 'table4_four_methods.tex', 'appendix_x7_target_accuracy.tex'}
SCALED_METRICS = {'ndcg10', 'mrr', 'mrr10'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, places):
    output = f'{value:.{places}f}'
    if float(output) == 0:
        output = f'{0:.{places}f}'
    assert abs(float(output)-value) <= 0.500001*10**(-places)+1e-12
    return output


def numeric(cell, scale, places, bold):
    values = [float(cell['values'][key])*scale for key in cell['keys']]
    output = number(values[0], places)
    if len(values) == 2:
        assert values[1] >= 0
        uncertainty = int(round(values[1]*10**places))
        assert abs(uncertainty/10**places-values[1]) <= 0.500001*10**(-places)+1e-12
        output += f'({uncertainty})'
    return ('\\bfseries ' if bold else '') + output


def main():
    trace = json.loads(TRACE.read_text(encoding='utf-8'))
    cells = trace['cells']
    cache = {}
    for cell in cells:
        filename = cell['source_csv']
        if filename not in cache:
            with (ROOT / filename).open(encoding='utf-8-sig', newline='') as handle:
                cache[filename] = list(csv.DictReader(handle))
        rows = [row for row in cache[filename] if all(row[key] == str(value) for key,value in cell['selector'].items())]
        assert len(rows) == 1, (filename, cell['selector'])
        assert all(rows[0][key] == str(value) for key,value in cell['values'].items())
    by_table = {}
    for cell in cells:
        by_table.setdefault(cell['table'], {}).setdefault(cell['row_id'], {})[cell['column']] = cell
    # Compare only the learned methods, using unrounded means and the declared
    # metric direction. Analytic references remain visible but are never bold.
    learned = [c for c in cells if c['table']=='table2_accuracy.tex' and c['selector'].get('method') not in ['Random','Popularity']]
    best = {}
    for cell in learned:
        key = (cell['selector']['dataset'], cell['selector']['metric'])
        value = float(cell['values'][cell['keys'][0]])
        if key not in best:
            best[key] = value
        else:
            best[key] = min(best[key], value) if key[1]=='mean_rank' else max(best[key], value)
    changed = []
    before_hashes = {path.name:sha(path) for path in TEMPLATES.glob('*.tex')}
    source_rows = 0
    for path in sorted(TEMPLATES.glob('*.tex')):
        original = path.read_text(encoding='utf-8')
        lines = []
        for line in original.splitlines():
            marker = re.search(r'% table-row:([^\s]+)', line)
            if marker:
                rowid = marker[1]
                body = line.split(r'\\ % table-row:',1)[0]
                parts = body.split(' & ')
                tracecells = by_table.get(path.name,{}).get(rowid,{})
                for column, cell in tracecells.items():
                    assert parts[column] == cell['display_tex'], (path.name,rowid,column,parts[column],cell['display_tex'])
                    source_rows += 1
                    metric = cell['selector'].get('metric')
                    if path.name not in TARGETS:
                        continue
                    scale, places, bold = cell['display_scale'],cell['display_places'],cell['bold']
                    reason = []
                    if metric in SCALED_METRICS and cell['keys'][0] != 'n':
                        scale, places = 1000, 1
                        reason.append('NDCG/MRR presentation uses units of 10^-3')
                    if path.name == 'appendix_x7_target_accuracy.tex' and metric=='mean_rank':
                        scale, places = 1, 0
                        reason.append('mean-rank mean and SD rounded to integer display')
                    if path.name == 'table2_accuracy.tex':
                        method = cell['selector'].get('method')
                        if method in ['Random','Popularity']:
                            bold = False
                        else:
                            key=(cell['selector']['dataset'],metric)
                            bold = float(cell['values'][cell['keys'][0]]) == best[key]
                        if bold != cell['bold']:
                            reason.append('bolding restricted to learned methods, using unrounded means')
                    if reason:
                        rendered = numeric(cell,scale,places,bold)
                        if rendered != parts[column]:
                            changed.append({'table':path.name,'row_id':rowid,'column':column,'source_csv':cell['source_csv'],
                                            'selector':cell['selector'],'source_values':cell['values'],'old_tex':parts[column],
                                            'new_tex':rendered,'scale':scale,'places':places,'reason':reason})
                        parts[column] = rendered
                line = ' & '.join(parts)+r'\\ % table-row:'+rowid
            lines.append(line)
        text = '\n'.join(lines)+'\n'
        # Integer identifiers must never receive numeric grouping. Other counts
        # retain the manuscript's thousands separators.
        if path.name == 'appendix_per_seed_accuracy.tex':
            text=text.replace('llS[table-format=4.0]','llS[table-format=4.0,group-digits=false]',1)
        if path.name == 'appendix_x7_training.tex':
            text=text.replace('lS[table-format=4.0]','lS[table-format=4.0,group-digits=false]',1)
        # Reserve realistic SD widths and let each ± term follow its mean.
        if path.name in ['table2_accuracy.tex','table4_four_methods.tex']:
            text=text.replace('S[table-format=5.0(5)]','S[table-format=5.0(3),table-align-uncertainty=false]')
        if path.name == 'appendix_table_secondary.tex':
            text=text.replace('S[table-format=4.0(4)]','S[table-format=4.0(3),table-align-uncertainty=false]')
        if path.name == 'appendix_x7_target_accuracy.tex':
            text=text.replace('S[table-format=5.1(5)]','S[table-format=5.0(3),table-align-uncertainty=false]')
        if path.name in TARGETS:
            text=text.replace('S[table-format=1.3(3)]','S[table-format=2.1(3)]')
            text=text.replace(r'{\shortstack{NDCG@10\\ $\uparrow$}}',r'{\shortstack{NDCG@10\\ ($\times10^{-3}$) $\uparrow$}}')
            text=text.replace(r'{\shortstack{MRR\\ $\uparrow$}}',r'{\shortstack{MRR\\ ($\times10^{-3}$) $\uparrow$}}')
            text=text.replace(r'{\shortstack{MRR@10\\ $\uparrow$}}',r'{\shortstack{MRR@10\\ ($\times10^{-3}$) $\uparrow$}}')
            if path.name=='appendix_x7_target_accuracy.tex':
                text=text.replace('{NDCG@10}',r'{\shortstack{NDCG@10\\$\times10^{-3}$}}')
                text=text.replace('{MRR}',r'{\shortstack{MRR\\$\times10^{-3}$}}')
                text=text.replace('Values are mean $\\pm$ sample SD over three training seeds.',
                    'Values are mean $\\pm$ sample SD over three training seeds. NDCG@10 and MRR are in $10^{-3}$ units. Displayed SDs of 0.0 may represent positive values below $0.05\\times10^{-3}$.')
            if path.name=='table2_accuracy.tex':
                text=text.replace('analytic references: expectations.',
                    'analytic references: expectations. NDCG@10 and MRR are in $10^{-3}$ units. Bold compares learned methods within each dataset, using unrounded means; analytic references are excluded.')
            if path.name=='table4_four_methods.tex':
                text=text.replace('Values are mean $\\pm$ sample SD over three training seeds.',
                    'Values are mean $\\pm$ sample SD over three training seeds. NDCG@10, MRR and MRR@10 are in $10^{-3}$ units.')
        if path.name=='appendix_calibration.tex':
            assert text.count('Clean unique')==2
            text=text.replace('Clean unique','Distinct negatives')
        (TABLES / path.name).write_text(text,encoding='utf-8',newline='\n')
    report={'status':'PASS_TABLE_SOURCE_AND_DISPLAY_REVISION','date':'2026-11-02',
            'items_completed':['T1','T2','T3','T4','T5','T6'],
            'source_values_changed':False,'estimators_or_intervals_recomputed':False,
            'source_trace':str(TRACE),'source_trace_sha256':sha(TRACE),'original_precision_source_cells_verified':len(cells),
            'current_table_cells_matched_to_trace':source_rows,'numeric_display_changes':changed,
            'seed_identifier_columns_without_grouping':['appendix_per_seed_accuracy.tex column3','appendix_x7_training.tex column2'],
            'compact_mean_sd_columns':['table2_accuracy.tex mean rank','table4_four_methods.tex mean rank',
                                       'appendix_table_secondary.tex absolute rank difference','appendix_x7_target_accuracy.tex mean rank'],
            'mean_rank_precision':'Integer means and SDs across all tables',
            'scaled_absolute_metric_tables':sorted(TARGETS),
            'zero_sd_note':'Table 12 states rounded positive SDs can display as 0.0; genuinely zero NDCG SDs remain zero.',
            'bold_rule':'Table 2 compares only learned methods using unrounded means and smaller-is-better mean rank.',
            'source_csv_sha256':{p:sha(ROOT / p) for p in sorted(cache)},
            'before_table_sha256':before_hashes,'after_table_sha256':{p.name:sha(p) for p in TABLES.glob('*.tex')},
            'pdf_visual_validation_pending':True}
    (ROOT/'table_rebuild_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','items_completed','source_values_changed','original_precision_source_cells_verified','current_table_cells_matched_to_trace']}))


if __name__=='__main__':
    main()
