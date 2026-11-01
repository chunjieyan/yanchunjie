"""E1 isolated saved-output overlay; all scientific statistics inherit the sealed C builder."""
import hashlib
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=Path(r"D:/time_uncertainty_rec")
spec=importlib.util.spec_from_file_location("_trusted_C_builder",HERE/"build_unified_statistics_original.py")
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
SEEDS,TAGS,METHODS,PERMS,ACCURACY,POPULATIONS,TC=(getattr(original,n) for n in ("SEEDS","TAGS","METHODS","PERMS","ACCURACY","POPULATIONS","TC"))


class Builder(original.Builder):
    def __init__(self,root):
        super().__init__(root)
        self.out=HERE
        self.overlay=self.read(HERE/"run_overlay.json")
        self.mapping={(r["dataset"],r["method"],r["training_seed"]):r for r in self.overlay["runs"]}
        self.check(len(self.mapping)==54,"54 current evaluation lineages")
        for record in self.mapping.values():
            for relative,expected in record["files_sha256"].items():
                path=self.root/relative
                self.check(path.resolve().is_relative_to(self.root) and path.suffix in (".csv",".json"),"saved metadata overlay source whitelist")
                content=path.read_bytes()
                self.check(hashlib.sha256(content).hexdigest()==expected,"overlay current saved source seal")
                self.manifest[relative]={"sha256":expected,"bytes":len(content)}
        self.unchanged_verification={}

    def run_output(self,tag,method,seed):
        return self.root/self.mapping[(tag,method,seed)]["output_directory"]

    def load_run(self,tag,method,seed):
        labels=self.labels[tag]
        out=self.run_output(tag,method,seed)
        own=self.read(out/"panel_labels.csv")
        if method=="TiSASRec":
            self.check({r["sample_id"]:int(r["affected"]) for r in own}=={sid:int(r["affected"]) for sid,r in labels.items()},"TiSASRec same shared affected panel")
        else:
            self.check(own==list(labels.values()),"exact shared fixed label records after E1")
        canonical = self.predictions(out / "predictions_original.csv", labels)
        prefix = "group" if method == "Ensemble" else "perm_seed"
        names = range(1, 6) if method == "Ensemble" else PERMS
        panels = [self.predictions(out / f"predictions_{prefix}{p}.csv", labels) for p in names]
        run = {"canonical": canonical, "panels": panels, "primary_panels": panels if method == "Ensemble" else [canonical]}
        self.runs[(tag, method, seed)] = run
        for pop in POPULATIONS:
            ids = self.population(labels, pop)
            group_metrics = [self.rank_metrics(p, ids) for p in run["primary_panels"]]
            acc = {key: statistics.mean(v[key] for v in group_metrics) if ids else None for key in ACCURACY}
            stable = self.stability(panels, ids)
            common = {"dataset": tag, "method": method, "training_seed": seed, "population": pop,
                      "n": len(ids), "affected_n": len(self.population(labels, "affected")),
                      "accuracy_view": "five_final_group_metric_mean" if method == "Ensemble" else "canonical"}
            for metric, value in (acc | stable).items():
                self.rows.append(common | {"metric": metric, "value": value})
            if method == "Ensemble":
                for g, metrics in enumerate(group_metrics, 1):
                    for metric, value in metrics.items():
                        self.ensemble_groups.append(common | {"group": g, "metric": metric, "value": value})

    def verify_previous(self):
        old_dir=self.root/"kais_revision_outline_20261022/g2_closure_20261027/C_statistics"
        old_rows=self.read(old_dir/"per_seed_metrics.csv")
        old={(r["dataset"],r["method"],int(r["training_seed"]),r["population"],r["metric"]):r for r in old_rows}
        retained=0
        for row in self.rows:
            if self.mapping[(row["dataset"],row["method"],row["training_seed"])]["lineage"]!="RETAINED_PRE_E1_EVALUATION":
                continue
            before=old[(row["dataset"],row["method"],row["training_seed"],row["population"],row["metric"])]
            self.check(row["n"]==int(before["n"]) and row["affected_n"]==int(before["affected_n"]) and row["accuracy_view"]==before["accuracy_view"],"retained seed metadata unchanged")
            if row["value"] is None:self.check(before["value"]=="","retained empty stratum unchanged")
            else:self.near(row["value"],float(before["value"]),"retained per-seed value unchanged")
            retained+=1
        previous={(r["dataset"],r["method"],r["population"],r["metric"]):r for r in self.read(old_dir/"method_summary.csv")}
        unchanged={(tag,method) for tag in TAGS for method in METHODS if all((tag,method,s) in self.mapping and self.mapping[(tag,method,s)]["lineage"]=="RETAINED_PRE_E1_EVALUATION" for s in SEEDS)}
        comparisons=0
        for row in self.summary:
            if (row["dataset"],row["method"]) not in unchanged:continue
            before=previous[(row["dataset"],row["method"],row["population"],row["metric"])]
            for field in ("mean","sample_sd","ci95_lower","ci95_upper"):
                if row[field] is None:self.check(before[field]=="","unchanged empty aggregate")
                else:self.near(row[field],float(before[field]),"unchanged aggregate matches original C")
                comparisons+=1
        for relative,expected in self.overlay["independent_audits"].items():
            self.check(hashlib.sha256((self.root/relative).read_bytes()).hexdigest()==expected,"new source independent audit seal")
            report=self.read(self.root/relative)
            self.check(report["status"].startswith("PASS"),"new19 training/new18 inference saved-record audits PASS")
        self.unchanged_verification={"retained_per_seed_rows_matched":retained,"unchanged_method_dataset_groups":[list(k) for k in sorted(unchanged)],"unchanged_aggregate_fields_checked":comparisons,"affected_groups_not_compared_to_obsolete_aggregates":True}

    def main(self):
        with contextlib.redirect_stdout(io.StringIO()):
            super().main()
        path=self.out/"verification.json";report=json.loads(path.read_text(encoding="utf-8"))
        report.update(precomputed_prior_audits_matched=False,unchanged_previous_groups_and_seeds_verified=self.unchanged_verification,
                      current19_training_and18_inference_independent_audits_pass=True,current_E1_replacement_evaluations=23,retained_evaluations=31,
                      analysis_plan_reused_posthoc=True,new_preregistration_claimed=False)
        path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report,indent=2))
        manifest_path=self.out/"source_manifest.json";value=json.loads(manifest_path.read_text(encoding="utf-8"))
        value.update(overlay_builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),run_overlay_sha256=hashlib.sha256((HERE/"run_overlay.json").read_bytes()).hexdigest(),original_analysis_plan_reused=True)
        manifest_path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    Builder(ROOT).main()
