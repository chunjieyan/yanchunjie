"""Formal journal table display, using sealed unrounded E1 values without recomputation."""
import csv,hashlib,json,math,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE/'baseline_project/tables';OUT=HERE/'tables';OUT.mkdir(exist_ok=True)
E1=HERE.parent/'p1_review_20261030/e1_finalization_20261031'
TRACE=E1/'assets/provenance/table_numeric_cells.json'
checks=0;cache={};verified=[];removed=[];notes_removed={};manual=[]
def ck(ok,msg):
 global checks
 checks+=1
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_rows(path):
 path=Path(path)
 if str(path) not in cache:
  with path.open(encoding='utf-8-sig',newline='') as f:cache[str(path)]=list(csv.DictReader(f))
 return cache[str(path)]
def selected(cell):
 rr=[r for r in source_rows(cell['source_csv']) if all(r[k]==str(v) for k,v in cell['selector'].items())];ck(len(rr)==1,'Unique unrounded source '+cell['row_id']);r=rr[0]
 for k,v in cell['values'].items():ck(r[k]==str(v),'Unrounded source immutable '+cell['row_id']+'/'+k)
 return r
def number(value,places,signed=False):
 value=float(value);text=format(value,('+' if signed else '')+'.'+str(places)+'f')
 if float(text)==0:text=format(0.0,'.'+str(places)+'f')
 ck(not re.fullmatch(r'-0(?:\.0+)?',text),'No negative zero')
 ck(abs(float(text)-value)<=0.5000001*10**(-places)+1e-12,'Rounding unit bound')
 return text

def render(cell):
 raw=selected(cell);keys=cell['keys'];scale=float(cell['scale']);places=int(cell['places']);signed=cell['signed'];style=cell['style'];delta=(cell['table'] in ['table5_lbsp_subgroups.tex','appendix_x7_warm_stability.tex'] and cell['selector'].get('metric')=='ndcg10' and signed)
 if delta:scale=1000;places=1
 elif scale==100:places=1
 elif places>3:places=3
 values={k:raw[k] for k in keys}
 if style=='dataset_n':
  n=int(float(raw[keys[0]]));prefix=cell['tex'].split(' (',1)[0];tex=prefix+' ('+format(n,',d')+')'
 elif style=='interval':tex=r'\(['+number(float(raw[keys[0]])*scale,places,signed)+r',\ '+number(float(raw[keys[1]])*scale,places,signed)+r']\)'
 elif style=='plain_pm':tex=number(float(raw[keys[0]])*scale,places,signed)+r' $\pm$ '+number(float(raw[keys[1]])*scale,places)
 else:
  tex=number(float(raw[keys[0]])*scale,places,signed)
  if len(keys)==2:
   sd=float(raw[keys[1]])*scale;ck(sd>=0,'SD nonnegative');uncertainty=int(round(sd*10**places));ck(abs(uncertainty/10**places-sd)<=0.5000001*10**(-places)+1e-12,'SD rounding bound');tex+='('+str(uncertainty)+')'
  if cell['bold']:tex=r'\bfseries '+tex
 new={**cell,'source_values_unchanged':values,'display_scale':scale,'display_places':places,'display_tex':tex,'delta_ndcg_scaled_1e3':delta};verified.append(new);return tex

def caption(text,value):
 token=r'\caption{';start=text.index(token)+len(token);level=1;end=start
 while level:
  if text[end]=='{' and (end==0 or text[end-1]!='\\'):level+=1
  elif text[end]=='}' and (end==0 or text[end-1]!='\\'):level-=1
  end+=1
 return text[:start]+value+text[end-1:]
def new_note(value):return r'\par\smallskip\noindent\begin{minipage}{\textwidth}\fontsize{8.5}{11}\selectfont '+value+'\n'+r'\end{minipage}'
NOTES={
 'table1_dataset_metadata.tex':r'Tied-target percentages use candidate users; multi-event and cold-target percentages use strict-test users. Validation and test $N$ are shown separately.',
 'table2_accuracy.tex':r'$N=945$, $2{,}966$, $4{,}659$ and $2{,}547$ for MovieLens-1M, Beauty, Sports and Toys. Learned models: mean $\pm$ sample SD over three training seeds; analytic references: expectations.',
 'table3_sensitivity.tex':r'Values are mean $\pm$ sample SD over three training seeds. User counts $N$ are shown for each population. Seed-pair references average three dependent pairs and have no confidence interval.',
 'table4_four_methods.tex':r'Affected-user $N$ is shown beside each dataset. Values are mean $\pm$ sample SD over three training seeds.',
 'table5_lbsp_subgroups.tex':r'$N_{\rm users}$ and $n_{\rm seeds}=3$ are shown. Differences are mean $\pm$ sample SD; intervals are unadjusted 95\% Student $t$ intervals (df=2; $t_{0.975}=4.303$).',
 'appendix_seed_pair_reference.tex':r'Each entry is a user-average overlap for one pair of three fitted training seeds. These pairs are dependent; no confidence interval is assigned.',
 'appendix_table_secondary.tex':r'Values are mean $\pm$ sample SD over three training seeds. Affected-user and distinct-pair counts are fixed across those seeds.',
 'appendix_x7_target_accuracy.tex':r'User counts $N$ are fixed across methods and seeds. Values are mean $\pm$ sample SD over three training seeds.',
 'appendix_x7_training.tex':r'Each row is one Sports training run at the indicated seed. Values are recorded epoch-level point estimates.',
 'appendix_x7_warm_stability.tex':r'$N$ denotes warm--affected users. Values are mean $\pm$ sample SD over three training seeds; differences use same-seed Control and intervals are unadjusted 95\% Student $t$ intervals (df=2).',
 'appendix_calibration.tex':r'Beauty: $N=10{,}000$; MovieLens-1M: $N=6{,}040$. All rows use seed 42 ($n=1$) and report test point estimates with 100 negative entries plus the target.',
 'appendix_measured_costs.tex':r'All rows use training seed 42. Values are mean $\pm$ sample SD over five measured training epochs after warm-up or five inference repetitions. Test $N$: 945, 2,966, 4,659 and 2,547 for MovieLens-1M, Beauty, Sports and Toys.',
 'appendix_per_seed_accuracy.tex':r'Test $N$: MovieLens-1M 945; Beauty 2,966; Sports 4,659; Toys 2,547. Each row is a point estimate for the indicated training seed.',
}
CAPTIONS={
 'table1_dataset_metadata.tex':'Dataset sizes and strict evaluation populations.',
 'table2_accuracy.tex':'Ranking accuracy on the complete strict test panels.',
 'table3_sensitivity.tex':'Recommendation stability on complete and affected-user panels.',
 'table4_four_methods.tex':'Accuracy and recommendation stability on affected-user panels.',
 'table5_lbsp_subgroups.tex':'Paired LBSP-minus-Control accuracy differences in pooled and singleton histories.',
 'appendix_seed_pair_reference.tex':r'Canonical-input Top-10 overlaps (\%) between training seeds.',
 'appendix_table_secondary.tex':'Secondary diagnostics for actually distinct input histories.',
 'appendix_x7_target_accuracy.tex':'Accuracy by target exposure in the training split.',
 'appendix_x7_training.tex':'Sports training trajectories: selected and final epochs.',
 'appendix_x7_warm_stability.tex':'Accuracy and ordering sensitivity among warm--affected users.',
 'appendix_calibration.tex':'Leave-one-out ranking accuracy with sampled candidates.',
 'appendix_measured_costs.tex':'Training and inference costs on complete test panels.',
 'appendix_per_seed_accuracy.tex':'Per-seed ranking accuracy on the complete strict test panels.',
}

def main():
 data=json.loads(TRACE.read_text(encoding='utf-8-sig'));ck(len(data['cells'])==1232,'Expected1232 E1 tracecells');allcells={(c['table'],c['row_id'],c['column']):c for c in data['cells']};allrows={(r['table'],r['row_id']):r for r in data['rows']};ck(len(allcells)==1232,'Unique trace coordinates')
 texts={p.name:p.read_text(encoding='utf-8-sig') for p in BASE.glob('*.tex')};ck(len(texts)==13 and set(texts)==set(NOTES),'All13 tables inventoried')
 # Remove the repeated cost continuation as a whole; Appendix cost table is retained.
 t=texts['table4_four_methods.tex'];starts=[m.start() for m in re.finditer(re.escape(r'\begin{table*}'),t)];ck(len(starts)==2,'Two baselineTable4 floats');texts['table4_four_methods.tex']=t[:starts[1]].rstrip()+'\n'
 removed.append({'table':'table4_four_methods.tex','reason':'Panels(c)(d) repeat the complete Appendix cost table; remove entire continuation float and keep(a)(b).','kind':'duplicate_cost_panels','removed_text_sha256':hashlib.sha256(t[starts[1]:].encode()).hexdigest()})
 rendered_rows={}
 for name,text in list(texts.items()):
  lines=[]
  for line in text.splitlines():
   marker=re.search(r'% e1-row:([^\s]+)',line)
   if marker:
    rowid=marker.group(1);oldrow=allrows[(name,rowid)];cells=list(oldrow['cells']);body=line.split(r'\\ % e1-row:',1)[0];ck(body==' & '.join(cells),'Baseline row exact trace '+name+'/'+rowid)
    for column in range(len(cells)):
     c=allcells.get((name,rowid,column))
     if c:cells[column]=render(c)
    if name=='appendix_measured_costs.tex':cells[1]={'Random-shuffle':'Shuffle','Score ensemble':'Ensemble'}.get(cells[1],cells[1])
    line=' & '.join(cells)+r'\\ % table-row:'+rowid;rendered_rows[(name,rowid)]=cells
   lines.append(line)
  texts[name]='\n'.join(lines)+'\n'
 # All removed1232 coordinates are still checked against their immutable sources.
 retained={(r['table'],r['row_id'],r['column']) for r in verified}
 for key,c in allcells.items():
  if key not in retained:selected(c);removed.append({'table':c['table'],'row_id':c['row_id'],'column':c['column'],'reason':'Duplicate main-table cost panel removed; same cost dataset retained in Appendix.','source_csv':c['source_csv'],'selector':c['selector'],'source_values':c['values']})
 # Only the current two sampling panels are retained in the calibration table.
 name='appendix_calibration.tex';old=texts[name];hist=[l for l in old.splitlines() if 'Historical strict sampled' in l];ck(len(hist)==2,'Two historical sampling rows')
 for line in hist:removed.append({'table':name,'reason':'Historical strict-sampled protocol is omitted from the formal LOO table; supporting current validation-selected test panels remain.','kind':'historical_protocol_row','old_row':line})
 calpath=HERE/'baseline_project/evidence/accepted_calibration_metrics.csv';cal=source_rows(calpath);newcal=[]
 for tag,label,ref in [('beauty','Beauty',0.3219),('ml1m','MovieLens-1M',0.5905)]:
  for panel,display in [('official_like','With replacement'),('clean_unique','Clean unique')]:
   found=[r for r in cal if r['dataset']==tag and r['split']=='test' and r['panel']==panel];ck(len(found)==1,'Current calibration source');r=found[0]
   cells=[label,display,r['N'],number(r['NDCG@10'],3),number(float(r['HR@10'])*100,1),number(ref,3) if panel=='official_like' else '{---}'];newcal.append(' & '.join(cells)+r'\\')
   manual.append({'table':name,'dataset':tag,'panel':panel,'source_csv':str(calpath),'source_values':r,'display_cells':cells,'literature_reference_unrounded':ref if panel=='official_like'else None})
   if panel=='clean_unique':newcal.append(r'\midrule' if tag=='beauty'else'')
 start=old.index(r'\midrule')+len(r'\midrule');end=old.index(r'\bottomrule',start);texts[name]=old[:start]+'\n'+'\n'.join(v for v in newcal if v)+'\n'+old[end:]
 for name,text in list(texts.items()):
  text=caption(text,CAPTIONS[name])
  note_pattern=r'\\par\\smallskip(?:\\noindent)?\\begin\{minipage\}\{\\textwidth\}.*?\\end\{minipage\}'
  match=re.search(note_pattern,text,re.S)
  if match:notes_removed[name]=match.group();text=text[:match.start()]+new_note(NOTES[name])+text[match.end():]
  elif name=='appendix_measured_costs.tex':
   start=text.index(r'\par\smallskip\footnotesize');end=text.index(r'\end{table*}',start);notes_removed[name]=text[start:end];text=text[:start]+new_note(NOTES[name])+'\n'+text[end:]
  elif name=='appendix_per_seed_accuracy.tex':
   line=next(l for l in text.splitlines() if l.startswith(r'\multicolumn{10}{@{}l@{}}'));notes_removed[name]=line;replacement=r'\multicolumn{10}{@{}l@{}}{\parbox{0.98\textwidth}{\vspace{3pt}\fontsize{8.5}{11}\selectfont '+NOTES[name]+r'}}\\';text=text.replace(line,replacement)
  else:raise ValueError('No note found '+name)
  # Consistent readable table type; keep width and units rather than shrinking fonts.
  text=re.sub(r'\\fontsize\{[0-9.]+\}\{[0-9.]+\}\\selectfont',lambda m:r'\fontsize{8.5}{11}\selectfont',text)
  text=text.replace(r'\small\setlength',r'\fontsize{8.5}{11}\selectfont\setlength')
  text=text.replace('table-format=+1.6(6)','table-format=+2.1(3)').replace('table-format=1.6','table-format=1.3').replace('table-format=2.2(3)','table-format=2.1(3)').replace('table-format=2.3(4)','table-format=2.1(3)')
  if name=='table5_lbsp_subgroups.tex':text=text.replace(r'(a) NDCG@10 differences',r'(a) NDCG@10 differences ($\times10^{-3}$)').replace(r'$\Delta$NDCG@10\\Mean',r'$\Delta$NDCG@10 ($\times10^{-3}$)\\Mean').replace(r'95\% $t$ interval\\NDCG@10 units',r'95\% $t$ interval\\$\times10^{-3}$')
  if name=='appendix_x7_warm_stability.tex':text=text.replace(r'$\Delta$NDCG@10\\LBSP minus Control',r'$\Delta$NDCG@10 ($\times10^{-3}$)\\LBSP minus Control').replace(r'95\% $t$ interval\\NDCG@10 units',r'95\% $t$ interval\\$\times10^{-3}$')
  # Remove internal generation commentary; row markers are neutral source locators.
  text='\n'.join(l for l in text.splitlines() if not l.startswith('%'))+'\n'
  for internal in ['official-like','newly trained','accepted','C-sealed','X4','X5','X6','X7','Historical strict sampled']:
   ck(internal not in text,'No internal visible table wording '+name+'/'+internal)
  ck('4.302652'not in text and not re.search(r'(?<![\d.])-0\.0+(?!\d)',text),'No overprecisequantile/negativezero '+name)
  ck(not re.search(r'\\fontsize\{(?:[0-7](?:\.\d+)?|8(?:\.0)?)\}',text),'Readable8.5pt '+name)
  if name=='appendix_per_seed_accuracy.tex':
   # Keep the first Toys method group together at the continuation boundary.
   text='\n'.join(line+'\n'+r'\noalign{\penalty10000}' if '% table-row:toys_Control_42' in line or '% table-row:toys_Control_2026' in line else line for line in text.splitlines())+'\n'
  texts[name]=text;(OUT/name).write_text(text,encoding='utf-8',newline='\n')
 # Source/rounding audit and rendered-cell round trip, without changing any estimator.
 ck(len(retained)+sum('column'in r for r in removed)==1232,'All sourcecells retained or explained')
 for v in verified:
  cells=rendered_rows[(v['table'],v['row_id'])];ck(cells[v['column']]==v['display_tex'],'Rendered numeric cell roundtrip')
  text=texts[v['table']];ck(v['display_tex']in text,'Formatted cell present')
 ck(texts['table4_four_methods.tex'].count(r'\begin{table*}')==1 and '(c)'not in texts['table4_four_methods.tex'] and '(d)'not in texts['table4_four_methods.tex'],'Table4singleevidencefloat')
 ck(len(manual)==4 and all('Historical'not in r for r in texts['appendix_calibration.tex'].splitlines()),'Four current calibration rows only')
 for name in ['table5_lbsp_subgroups.tex','appendix_x7_warm_stability.tex']:ck(r'\times10^{-3}'in texts[name],'Scaled delta and CI header '+name)
 keep_main={
 'fixed_populations_and_estimands':'Affectedmask beforepooling differsfrompooled/singleton; overlap equaluser/tenpairs,hitflipanyuser; secondarypairweightedactualdistinctviews preservesmultiplicity.',
 'method_and_accuracy_units':'Canonicalinputs except Ensemblemetricmeanfivegroupswithinseed,sharedControlcheckpoint; fullMRRvsMRR10/meanrankdirection; boldunroundedobservedbestnot significance; analyticcandidate/tieexpectations.',
 'uncertainty':'Seedn3,tintervaldf2,untruncated/unadjusted; paired-userbootstrapunitdistinct; dependentseedpairs noCI. Posthoc/multiplicity/nonequivalence caveats retainonceThreats.',
 'cost_measurement':'GPUidentitiesfixedwithinphase,differentphysicalcardsacrossphases; rawincludes/excludes,512warmup/5reps,temporarytraining1+5excludesval/checkpoint,Ensemble5viewsperrecommendation; allocatedmemorynotwholeGPU. Toysnospeedclaimretainonce.',
 'exposure_and_metadata':'Target_train_count defineswarm/coldfixedlabels;zero-positiveitems canreceivefullCEgradient; tiedtargetsusecandidateusers andmultievent/coldusestrictN.',
 'calibration_protocol':'Withreplacementretainsduplicates/targetcollisions;cleanwithoutreplacementexcludestrainpositives,visiblehistory,target; sameselectedvalidationbest; oneseedandliteraturecontext withoutnumericalreproductionclaim.',
 }
 report={'date':'2026-10-31','status':'PASS_FORMAL_TABLE_DISPLAY_AND_UNROUNDED_SOURCE_VERIFICATION','tables':13,'original_numeric_cells_checked':1232,'numeric_cells_retained':len(verified),'numeric_cells_removed_as_duplicate_costs':sum('column'in r for r in removed),'scaled_delta_NDCG_cells':sum(r['delta_ndcg_scaled_1e3']for r in verified),'manual_current_calibration_rows':4,'historical_calibration_rows_removed':2,'uncertainty_computation_changed':False,'source_values_changed':False,'scale_rule':'DeltaNDCGanddifferenceCIs times1000; means/SD/endpointsone decimal. AbsoluteNDCG3decimals,percent1decimal,costappropriate2or3decimals. No negativezero.','t_quantile_display':'4.303 (underlying originalintervals unchanged)','type_size_pt':8.5,'all_tables_same_original_filenames':True,'main_table4_cost_continuation_removed':True,'checks':checks,'source_trace':str(TRACE),'source_trace_sha256':sha(TRACE),'source_sha256':{path:sha(path) for path in cache},'output_sha256':{name:sha(OUT/name)for name in texts},'required_explanation_to_keep_in_main':keep_main,'removed_rows_and_panels':removed,'baseline_notes_removed':notes_removed,'rounding_tolerance':'Half of one unit in the last displayed decimal, with floatcomparison tolerance only; all originalunroundedCSVstringvalues exact.','no_pdf_or_manuscript_edits':True}
 (OUT/'formal_table_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');(OUT/'formal_numeric_cells.json').write_text(json.dumps({'status':report['status'],'cells':verified,'calibration_rows':manual},ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
 print(json.dumps({k:v for k,v in report.items() if k in ['status','tables','original_numeric_cells_checked','numeric_cells_retained','numeric_cells_removed_as_duplicate_costs','scaled_delta_NDCG_cells','checks']},indent=2))
if __name__=='__main__':main()
