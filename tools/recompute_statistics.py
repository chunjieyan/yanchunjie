#!/usr/bin/env python3
"""Recompute unchanged KAIS estimands from a portable 54-run prediction registry.

No training, model imports, checkpoint loading, raw-interaction reads or historical
Windows paths are required. See --help. Historical hashes are an optional mode;
fresh results must retain user/target/panel identities and all three fixed seeds.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,importlib.util,json,math,statistics,sys,types
from pathlib import Path

ENGINE_SHA='68282281dda16ebda5583db40b52514139183d5fd8c2d9168e2fd5ef12f27a01'
BOOTSTRAP_SHA='3251018b365f36e6249995d4237cca928024a8f2fa7e0d1b9212e8b2dfb88b38'
REPO=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,name,expected):
    if sha(path)!=expected:raise ValueError('Statistical engine fingerprint differs: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def find_engine(name,explicit=None):
    if explicit:return explicit.resolve()
    index=REPO/'provenance/source_index.json'
    if index.is_file():
        record=json.loads(index.read_text(encoding='utf-8-sig'))
        rs=record.get('files',record.get('sources',[])) if isinstance(record,dict) else record
        for r in rs:
            rel=r.get('relative_source_path','')
            if rel.endswith('/p1_review_20261030/e1_finalization_20261031/statistics/'+name):
                rp=r.get('repository_path',r.get('package_relative_path'));p=REPO/rp
                if p.is_file():return p
    preferred=REPO/'legacy_sources/kais_revision_outline_20261022/p1_review_20261030/e1_finalization_20261031/statistics'/name
    if preferred.is_file():return preferred
    raise FileNotFoundError('Packaged statistics engine missing; provide --'+('bootstrap-engine' if name.startswith('build_user') else 'engine'))

def read_json(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write_json(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def under(root,relative):
    p=Path(relative);p=p.resolve() if p.is_absolute() else (root/p).resolve()
    if not p.is_relative_to(root):raise ValueError('Registry path leaves --root: '+str(relative))
    return p

def compare_outputs(out,reference,names):
    counts={}
    for name in names:
        a=out/name;b=reference/name
        if not b.is_file():raise FileNotFoundError('Expected reference CSV missing: '+str(b))
        def rows(p):
            with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
        ar,br=rows(a),rows(b)
        if len(ar)!=len(br):raise ValueError('Reference row count differs: '+name)
        checks=0
        for i,(x,y) in enumerate(zip(ar,br)):
            if x.keys()!=y.keys():raise ValueError('Reference columns differ: '+name)
            for k in x:
                if k=='numpy_version':continue
                if x[k]==y[k]:checks+=1;continue
                try:ok=math.isclose(float(x[k]),float(y[k]),rel_tol=2e-11,abs_tol=2e-12)
                except ValueError:ok=False
                if not ok:raise ValueError(f'Reference difference {name} row {i} {k}: {x[k]} != {y[k]}')
                checks+=1
        counts[name]={'rows':len(ar),'fields_checked':checks}
    return counts

def make_builder(engine,registry,out,verify_frozen):
    expected={(d,m,s) for d in engine.TAGS for m in engine.METHODS for s in engine.SEEDS if m!='TiSASRec' or d in ('ml1m','beauty')}
    mapping={(r['dataset'],r['method'],int(r['training_seed'])):r for r in registry['runs']}
    if len(mapping)!=len(registry['runs']) or set(mapping)!=expected:raise ValueError('Registry must contain exactly all54 unique configurations (42 independently fitted models +12 shared-Control Ensembles).')
    class PortableBuilder(engine.Builder):
        def __init__(self,root):
            super().__init__(root);self.out=out;self.mapping=mapping;self.ensemble_groups=[]
        def run_output(self,tag,method,seed):return under(self.root,self.mapping[(tag,method,seed)]['output_directory'])
        def read(self,path):
            path=path.resolve()
            if path in [self.out/'user_bootstrap_analysis_plan.json',self.out/'paired_differences_per_seed.csv']:
                content=path.read_bytes();self.manifest['@generated/'+path.name]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
                return json.loads(content.decode('utf-8-sig')) if path.suffix=='.json' else list(csv.DictReader(content.decode('utf-8-sig').splitlines()))
            return super().read(path)
        def prepare_labels(self):
            for tag in engine.TAGS:
                p=registry.get('label_files',{}).get(tag)
                raw=self.read(under(self.root,p) if p else self.run_output(tag,'Control',42)/'panel_labels.csv')
                labels={r['sample_id']:r for r in raw};self.check(len(labels)==len(raw),'label unique sample IDs')
                for row in raw:
                    distinct=len({row[f'perm_seed{p}_sequence_sha256'] for p in engine.PERMS})
                    self.check(int(row['distinct_sequences_5perm'])==distinct and int(row['affected'])==(distinct>=2),'saved affected definition')
                    self.check(int(row['last_bucket_size'])>=1,'nonempty last bucket')
                self.labels[tag]=labels
                self.check(len(self.population(labels,'pooled'))+len(self.population(labels,'singleton'))==len(labels),'strata exhaustive')
                self.check(sum(len(self.population(labels,p)) for p in ('affected_bucket2','affected_bucket3','affected_bucket4','affected_bucket5plus'))==len(self.population(labels,'affected')),'affected buckets exhaustive')
        def load_run(self,tag,method,seed):
            labels=self.labels[tag];out=self.run_output(tag,method,seed);own=self.read(out/'panel_labels.csv')
            if method=='TiSASRec':self.check({r['sample_id']:int(r['affected']) for r in own}=={sid:int(r['affected']) for sid,r in labels.items()},'TiSASRec same shared affected panel')
            else:self.check(own==list(labels.values()),'exact shared fixed label records')
            canonical=self.predictions(out/'predictions_original.csv',labels)
            prefix='group' if method=='Ensemble' else 'perm_seed';names=range(1,6) if method=='Ensemble' else engine.PERMS
            panels=[self.predictions(out/f'predictions_{prefix}{p}.csv',labels) for p in names]
            run={'canonical':canonical,'panels':panels,'primary_panels':panels if method=='Ensemble' else [canonical]};self.runs[(tag,method,seed)]=run
            for pop in engine.POPULATIONS:
                ids=self.population(labels,pop);group_metrics=[self.rank_metrics(p,ids) for p in run['primary_panels']]
                acc={key:statistics.mean(v[key] for v in group_metrics) if ids else None for key in engine.ACCURACY};stable=self.stability(panels,ids)
                common={'dataset':tag,'method':method,'training_seed':seed,'population':pop,'n':len(ids),'affected_n':len(self.population(labels,'affected')),'accuracy_view':'five_final_group_metric_mean' if method=='Ensemble' else 'canonical'}
                for metric,value in (acc|stable).items():self.rows.append(common|{'metric':metric,'value':value})
                if method=='Ensemble':
                    for g,metrics in enumerate(group_metrics,1):
                        for metric,value in metrics.items():self.ensemble_groups.append(common|{'group':g,'metric':metric,'value':value})
        def baselines(self):
            self.baseline_rows=[];self.target_metadata={}
            for tag in engine.TAGS:
                relative=registry.get('target_metadata',registry.get('baseline_metadata',{})).get(tag,'g2_x5_results_20261025/derived/'+tag+'_test.csv')
                mapped={r['sample_id']:r for r in self.read(under(self.root,relative))};labels=self.labels[tag]
                self.check(set(mapped)==set(labels),'analytic baseline same population')
                for sid,label in labels.items():self.check(mapped[sid]['user_id']==label['user_id'] and mapped[sid]['target_item']==label['target_item'],'analytic baseline user/target identity')
                self.target_metadata[tag]=mapped
                for pop in engine.POPULATIONS+('warm','cold'):
                    ids=[sid for sid,row in mapped.items() if row['target_group']==pop] if pop in ('warm','cold') else self.population(labels,pop)
                    for name,prefix in [('Random','random_'),('Popularity','popularity_')]:
                        for metric in engine.ACCURACY:
                            value=statistics.mean(float(mapped[sid][prefix+metric]) for sid in ids) if ids else None
                            self.baseline_rows.append({'dataset':tag,'method':name,'population':pop,'n':len(ids),'affected_n':len(self.population(labels,'affected')),'metric':metric,'mean':value,'sample_sd':None,'ci95_lower':None,'ci95_upper':None,'n_training_seeds':0,'accuracy_view':'analytic_expectation','dispersion':'not_applicable_analytic_expectation'})
    return PortableBuilder

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='Root of saved prediction/label/target-metadata files; may be any location.')
    parser.add_argument('--run-registry',type=Path,required=True,help='JSON runs[54] with dataset/method/training_seed/output_directory; optional label_files and target_metadata mappings.')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--verify-frozen',action='store_true',help='Also enforce optional files_sha256 records in the supplied historical registry.')
    parser.add_argument('--skip-bootstrap',action='store_true',help='Quick main-statistics run; omit checkpoint-conditional user intervals.')
    parser.add_argument('--compare-dir',type=Path,help='Optional numeric regression against saved CSV outputs, without changing estimates.')
    parser.add_argument('--engine',type=Path,help=argparse.SUPPRESS);parser.add_argument('--bootstrap-engine',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args();root=args.root.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);registry=read_json(args.run_registry.resolve())
    engine_path=find_engine('build_unified_statistics_original.py',args.engine);engine=load(engine_path,'kais_statistics_engine',ENGINE_SHA)
    Builder=make_builder(engine,registry,out,args.verify_frozen);b=Builder(root)
    frozen_count=0
    if args.verify_frozen:
        for r in registry['runs']:
            if not r.get('files_sha256'):raise ValueError('--verify-frozen requires files_sha256 on each run')
            scope=r.get('files_sha256_scope','root')
            if scope not in ('root','output_directory'):raise ValueError('Unknown files_sha256_scope: '+str(scope))
            hash_root=under(root,r['output_directory']) if scope=='output_directory' else root
            for rel,expected in r['files_sha256'].items():
                if sha(under(hash_root,rel))!=expected:raise ValueError('Frozen fingerprint differs: '+rel)
                frozen_count+=1
    b.prepare_labels()
    for tag in engine.TAGS:
        for method in engine.METHODS:
            if method=='TiSASRec' and tag not in ('ml1m','beauty'):continue
            for seed in engine.SEEDS:b.load_run(tag,method,seed)
    b.summarize();b.seed_references();b.baselines()
    outputs={'per_seed_metrics.csv':b.rows,'method_summary.csv':b.summary,'paired_differences_per_seed.csv':b.deltas,'paired_difference_summary.csv':b.delta_summary,'seed_reference_pairs.csv':b.reference_pairs,'seed_reference_summary.csv':b.reference_summary,'ensemble_group_accuracy.csv':b.ensemble_groups,'analytic_baselines.csv':b.baseline_rows,'lbsp_subgroup_contrasts_per_seed.csv':b.subgroup_contrasts(),'lbsp_subgroup_difference_summary.csv':[r for r in b.delta_summary if r['method']=='LBSP' and r['population'] in ('pooled','singleton') and r['metric'] in engine.ACCURACY]}
    outputs['bucket_stability_summary.csv']=[r|{'bucket':r['population'].replace('affected_bucket','').replace('5plus','>=5'),'population_definition':'affected users within last-bucket-size stratum'} for r in b.summary if r['population'].startswith('affected_bucket') and r['metric'] in ('overlap10','hitflip')]
    outputs['bucket_stability_all_users_summary.csv']=[r|{'bucket':r['population'].replace('bucket','').replace('5plus','>=5'),'population_definition':'all users within last-bucket-size stratum'} for r in b.summary if r['population'].startswith('bucket') and r['metric'] in ('overlap10','hitflip')]
    outputs['population_counts.csv']=[{'dataset':tag,'population':pop,'n':len(b.population(b.labels[tag],pop)),'affected_n':len(b.population(b.labels[tag],'affected'))} for tag in engine.TAGS for pop in engine.POPULATIONS]
    for name,rows in outputs.items():b.write(name,rows)
    bootstrap_count=0
    if not args.skip_bootstrap:
        bp=find_engine('build_user_paired_bootstrap.py',args.bootstrap_engine)
        plan=read_json(bp.parent/'user_bootstrap_analysis_plan.json');write_json(out/'user_bootstrap_analysis_plan.json',plan)
        adapter=types.ModuleType('build_unified_statistics')
        for name in ['ACCURACY','METHODS','SEEDS','TAGS']:setattr(adapter,name,getattr(engine,name))
        adapter.Builder=Builder;old=sys.modules.get('build_unified_statistics');sys.modules['build_unified_statistics']=adapter
        try:boot=load(bp,'kais_bootstrap_engine',BOOTSTRAP_SHA);boot.main(root)
        finally:
            if old is None:sys.modules.pop('build_unified_statistics',None)
            else:sys.modules['build_unified_statistics']=old
        bootstrap_count=924
    compared=compare_outputs(out,args.compare_dir.resolve(),list(outputs)+(['paired_per_seed_user_bootstrap.csv'] if bootstrap_count else [])) if args.compare_dir else {}
    write_json(out/'source_manifest.json',{'scope':'Registry-selected saved metadata/predictions; no raw data/models; current outputs registered afresh','files':b.manifest,'run_registry':{'path':str(args.run_registry.resolve()),'sha256':sha(args.run_registry)},'unchanged_scientific_engine_sha256':sha(engine_path),'portable_entrypoint_sha256':sha(Path(__file__))})
    report={'status':'PASS_PORTABLE_REGISTRY_STATISTICS','evaluated_configurations':len(b.runs),'independent_models':42,'Ensemble_reuses_Control':12,'per_seed_rows':len(b.rows),'method_summary_rows':len(b.summary),'paired_summary_rows':len(b.delta_summary),'bootstrap_interval_rows':bootstrap_count,'output_files':{name:sha(out/name) for name in list(outputs)+(['paired_per_seed_user_bootstrap.csv'] if bootstrap_count else [])},'optional_historical_hashes_checked':frozen_count,'core_scientific_methods_inherited_unmodified':['population','predictions','rank_metrics','stability','stats','summarize','seed_references','subgroup_contrasts'],'statistical_methods_changed':False,'historical_path_and_prior_result_checks_removed_from_fresh_execution':True,'user_target_affected_and_complete_panel_checks_retained':True,'training_or_model_or_checkpoint_execution':False,'reference_comparison':compared,'counts':dict(b.counts)}
    write_json(out/'verification.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
