"""Reproducible saved-prediction statistics. Standard library, no model execution."""
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

SEEDS = (42, 2026, 3407)
TAGS = ("ml1m", "beauty", "sports", "toys")
METHODS = ("Control", "LBSP", "Shuffle", "Ensemble", "TiSASRec")
PERMS = (101, 202, 303, 404, 505)
ACCURACY = ("ndcg10", "hr10", "hr20", "recall50", "mrr", "mrr10", "mean_rank")
POPULATIONS = ("full", "affected", "pooled", "singleton", "bucket2", "bucket3", "bucket4", "bucket5plus",
               "affected_bucket2", "affected_bucket3", "affected_bucket4", "affected_bucket5plus")
TC = 4.302652729749464


class Builder:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.out = Path(__file__).resolve().parent
        self.manifest = {}
        self.counts = Counter()
        self.rows, self.labels, self.runs = [], {}, {}
        self.base = self.root / "kais_revision_outline_20261022"
        self.x3 = self.root / "g2_x3_inference_results_20261027_e635848880/snapshot/outputs"
        self.x4 = self.root / "g2_x4_tisasrec_results_20261026/snapshot/outputs"

    def check(self, condition, message):
        self.counts["assertions"] += 1
        if not condition:
            raise AssertionError(message)

    def near(self, a, b, message):
        self.counts["numeric_comparisons"] += 1
        self.check(a is not None and b is not None and math.isclose(a, b, rel_tol=2e-11, abs_tol=2e-12), message)

    def read(self, path):
        path = path.resolve()
        self.check(path.is_relative_to(self.root), "read outside workspace")
        self.check(path.suffix.lower() in (".csv", ".json"), "only saved CSV/JSON metadata allowed")
        self.check(not any(part in ("data", "data_raw", "checkpoints") for part in path.parts), "raw/checkpoint path rejected")
        content = path.read_bytes()
        self.manifest[path.relative_to(self.root).as_posix()] = {
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        if path.suffix == ".json":
            return json.loads(content.decode("utf-8-sig"))
        return list(csv.DictReader(content.decode("utf-8-sig").splitlines()))

    def core(self, tag):
        return self.root / ("g2_lbsp_ml1m_20261023" if tag == "ml1m" else "g2_lbsp_amazon_stability_20261025") / "snapshot"

    def population(self, labels, pop):
        selected = []
        for sid, label in labels.items():
            bucket, affected = int(label["last_bucket_size"]), int(label["affected"])
            base_pop = pop.removeprefix("affected_")
            take = {"full": True, "affected": bool(affected), "pooled": bucket >= 2, "singleton": bucket == 1,
                    "bucket2": bucket == 2, "bucket3": bucket == 3, "bucket4": bucket == 4, "bucket5plus": bucket >= 5}[base_pop]
            if pop.startswith("affected_"):
                take = take and bool(affected)
            if take:
                selected.append(sid)
        return selected

    def predictions(self, path, labels):
        raw = self.read(path)
        self.counts["prediction_files"] += 1
        self.counts["prediction_rows"] += len(raw)
        out = {}
        for row in raw:
            sid = row["sample_id"]
            self.check(sid in labels and sid not in out, "prediction unique sample identity")
            label = labels[sid]
            self.check(row["user_id"] == label["user_id"] and row["target_item"] == label["target_item"], "prediction user/target identity")
            rank = int(row.get("target_rank", row.get("rank")))
            top = tuple(str(v) for v in json.loads(row["top10_raw_item_id"]))
            self.check(rank >= 1 and len(set(top)) == len(top) == 10, "positive rank, unique Top10")
            if "candidate_count" in row:
                self.check(rank <= int(row["candidate_count"]), "rank within candidates")
            self.counts["rank10_top10_membership_mismatches"] += int((rank <= 10) != (row["target_item"] in top))
            if row["target_item"] in top:
                self.counts["target_top10_position_mismatches"] += int(rank != top.index(row["target_item"]) + 1)
            out[sid] = {"rank": rank, "top": frozenset(top), "ordered_top": top}
        self.check(set(out) == set(labels), "complete prediction panel")
        return out

    @staticmethod
    def rank_metrics(pred, ids):
        if not ids:
            return dict.fromkeys(ACCURACY)
        ranks = [pred[sid]["rank"] for sid in ids]
        n = len(ranks)
        return {"ndcg10": math.fsum(1 / math.log2(r + 1) for r in ranks if r <= 10) / n,
                "hr10": sum(r <= 10 for r in ranks) / n, "hr20": sum(r <= 20 for r in ranks) / n,
                "recall50": sum(r <= 50 for r in ranks) / n, "mrr": math.fsum(1 / r for r in ranks) / n,
                "mrr10": math.fsum(1 / r for r in ranks if r <= 10) / n,
                "mean_rank": math.fsum(ranks) / n}

    @staticmethod
    def stability(panels, ids):
        if not ids:
            return {"overlap10": None, "hitflip": None}
        user_overlap, user_flip = [], []
        for sid in ids:
            values = [panel[sid] for panel in panels]
            user_overlap.append(math.fsum(len(a["top"] & b["top"]) / 10 for a, b in combinations(values, 2)) / 10)
            user_flip.append(int(len({v["rank"] <= 10 for v in values}) > 1))
        return {"overlap10": statistics.mean(user_overlap), "hitflip": statistics.mean(user_flip)}

    @staticmethod
    def stats(values):
        if any(v is None for v in values):
            return {"mean": None, "sample_sd": None, "ci95_lower": None, "ci95_upper": None}
        mean, sd = statistics.mean(values), statistics.stdev(values)
        half = TC * sd / math.sqrt(3)
        return {"mean": mean, "sample_sd": sd, "ci95_lower": mean - half, "ci95_upper": mean + half}

    def prepare_labels(self):
        for tag in TAGS:
            raw = self.read(self.core(tag) / "outputs" / f"{tag}_seed42/panel_labels.csv")
            labels = {r["sample_id"]: r for r in raw}
            self.check(len(labels) == len(raw), "label unique sample IDs")
            for row in raw:
                distinct = len({row[f"perm_seed{p}_sequence_sha256"] for p in PERMS})
                self.check(int(row["distinct_sequences_5perm"]) == distinct and int(row["affected"]) == (distinct >= 2), "saved affected definition")
                self.check(int(row["last_bucket_size"]) >= 1, "nonempty last bucket")
            self.labels[tag] = labels
            self.check(len(self.population(labels, "pooled")) + len(self.population(labels, "singleton")) == len(labels), "strata exhaustive")
            self.check(sum(len(self.population(labels, p)) for p in ("affected_bucket2", "affected_bucket3", "affected_bucket4", "affected_bucket5plus")) == len(self.population(labels, "affected")), "affected buckets exhaustive")

    def load_run(self, tag, method, seed):
        labels = self.labels[tag]
        if method in ("Control", "LBSP"):
            out = self.core(tag) / ("control_reference/outputs" if method == "Control" else "outputs") / f"{tag}_seed{seed}"
            self.check(self.read(out / "panel_labels.csv") == list(labels.values()), "core exact shared label records")
        elif method in ("Shuffle", "Ensemble"):
            out = self.x3 / f"{method.lower()}_{tag}_seed{seed}"
            self.check(self.read(out / "panel_labels.csv") == list(labels.values()), "X3 exact shared label records")
        else:
            out = self.x4 / f"{tag}_seed{seed}"
            thin = self.read(out / "panel_labels.csv")
            self.check({r["sample_id"]: int(r["affected"]) for r in thin} == {sid: int(r["affected"]) for sid, r in labels.items()}, "TiSASRec same shared affected panel")
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

    def summarize(self):
        groups = defaultdict(list)
        for row in self.rows:
            groups[(row["dataset"], row["method"], row["population"], row["metric"])].append(row)
        self.summary, self.deltas, self.delta_summary = [], [], []
        lookup = {(r["dataset"], r["method"], r["population"], r["training_seed"], r["metric"]): r for r in self.rows}
        for (tag, method, pop, metric), rows in groups.items():
            self.check(tuple(r["training_seed"] for r in rows) == SEEDS, "all three seeds retained in fixed order")
            common = {k: rows[0][k] for k in ("dataset", "method", "population", "n", "affected_n", "metric", "accuracy_view")}
            common |= {"training_seeds": ";".join(map(str, SEEDS)), "n_training_seeds": 3}
            self.summary.append(common | self.stats([r["value"] for r in rows]))
            paired = []
            for row in rows:
                control = lookup[(tag, "Control", pop, row["training_seed"], metric)]
                self.check(control["n"] == row["n"], "paired Control same population count")
                delta = row["value"] - control["value"] if row["value"] is not None else None
                self.deltas.append({k: row[k] for k in ("dataset", "method", "population", "n", "affected_n", "training_seed", "metric", "accuracy_view")} |
                                   {"method_value": row["value"], "control_value": control["value"], "difference": delta})
                paired.append(delta)
            ss = self.stats(paired)
            self.delta_summary.append(common | ss | {"mean_delta": ss["mean"], "sample_sd_delta": ss["sample_sd"], "comparison": "method_minus_same_seed_Control"})
        self.lookup, self.summary_lookup = lookup, {(r["dataset"], r["method"], r["population"], r["metric"]): r for r in self.summary}

    def seed_references(self):
        self.reference_pairs, self.reference_summary = [], []
        for tag in TAGS:
            labels = self.labels[tag]
            for method in METHODS:
                if (tag, method, SEEDS[0]) not in self.runs:
                    continue
                for pop in POPULATIONS:
                    ids = self.population(labels, pop)
                    pairs = []
                    for sa, sb in combinations(SEEDS, 2):
                        a, b = self.runs[(tag, method, sa)]["primary_panels"], self.runs[(tag, method, sb)]["primary_panels"]
                        vals = []
                        for pa, pb in zip(a, b):
                            vals.append({"overlap10": statistics.mean(len(pa[sid]["top"] & pb[sid]["top"]) / 10 for sid in ids) if ids else None,
                                         "hitflip": statistics.mean(int((pa[sid]["rank"] <= 10) != (pb[sid]["rank"] <= 10)) for sid in ids) if ids else None})
                        row = {"dataset": tag, "method": method, "population": pop, "n": len(ids),
                               "affected_n": len(self.population(labels, "affected")), "seed_a": sa, "seed_b": sb,
                               "overlap10": statistics.mean(v["overlap10"] for v in vals) if ids else None,
                               "hitflip": statistics.mean(v["hitflip"] for v in vals) if ids else None,
                               "view_alignment": "matching_final_groups_then_group_mean" if method == "Ensemble" else "canonical"}
                        pairs.append(row)
                        self.reference_pairs.append(row)
                    common = {k: pairs[0][k] for k in ("dataset", "method", "population", "n", "affected_n", "view_alignment")}
                    self.reference_summary.append(common | {"seed_pair_count": 3,
                        "mean_overlap10": statistics.mean(p["overlap10"] for p in pairs) if ids else None,
                        "mean_pair_hitflip": statistics.mean(p["hitflip"] for p in pairs) if ids else None,
                        "independent_replicates": "not_three_independent_pairs_no_interval"})

    def baselines(self):
        self.baseline_rows = []
        dataset_names = {"ml1m": "MovieLens-1M", "beauty": "Amazon_Beauty", "sports": "Amazon_Sports", "toys": "Amazon_Toys"}
        saved = self.read(self.root / "g2_x5_results_20261025/analytic_baselines.csv")
        for tag in TAGS:
            rows = self.read(self.root / "g2_x5_results_20261025/derived" / f"{tag}_test.csv")
            mapped = {r["sample_id"]: r for r in rows}
            labels = self.labels[tag]
            self.check(len(mapped) == len(rows) and set(mapped) == set(labels), "X5 baseline population")
            for sid, label in labels.items():
                self.check(mapped[sid]["user_id"] == label["user_id"] and mapped[sid]["target_item"] == label["target_item"], "X5 shared sample semantics")
            for pop in POPULATIONS + ("warm", "cold"):
                ids = [sid for sid, row in mapped.items() if row["target_group"] == pop] if pop in ("warm", "cold") else self.population(labels, pop)
                for name, prefix, source_name in (("Random", "random_", "UniformRandomAnalytic"), ("Popularity", "popularity_", "PopularityUniformTieAnalytic")):
                    for metric in ACCURACY:
                        value = statistics.mean(float(mapped[sid][prefix + metric]) for sid in ids) if ids else None
                        self.baseline_rows.append({"dataset": tag, "method": name, "population": pop,
                            "n": len(ids), "affected_n": len(self.population(labels, "affected")), "metric": metric,
                            "mean": value, "sample_sd": None, "ci95_lower": None, "ci95_upper": None,
                            "n_training_seeds": 0, "accuracy_view": "analytic_expectation", "dispersion": "not_applicable_analytic_expectation"})
                        if pop in ("full", "warm", "cold") and ids:
                            old = next(r for r in saved if r["dataset"] == dataset_names[tag] and r["method"] == source_name and r["group"] == ("overall" if pop == "full" else pop))
                            self.near(value, float(old[metric]), "X5 baseline matches audited overall/warm/cold")

    def verify_previous(self):
        core = self.read(self.base / "g2_lbsp_amazon_stability_results_review/math_four_dataset_independent_audit.json")
        self.check(core["status"] == "PASS", "core prior audit PASS")
        cm = {"ndcg10": "NDCG@10", "hr10": "HR@10", "hr20": "HR@20", "mean_rank": "MeanRank"}
        for tag in TAGS:
            for method in ("Control", "LBSP"):
                for pop in ("full", "affected"):
                    old = core["datasets"][tag]["method_summaries"][method]["complete" if pop == "full" else pop]
                    for metric, key in cm.items():
                        row = self.summary_lookup[(tag, method, pop, metric)]
                        for a, b in (("mean", "mean"), ("sample_sd", "sample_SD_ddof1")):
                            self.near(row[a], old["canonical_accuracy_three_seed_mean_sampleSD"][key][b], "core accuracy aggregate")
                    for metric, key in (("overlap10", "overlap_5perm_pair_mean"), ("hitflip", "hit_flip_user_fraction")):
                        row = self.summary_lookup[(tag, method, pop, metric)]
                        for a, b in (("mean", "mean"), ("sample_sd", "sample_SD_ddof1")):
                            self.near(row[a], old["three_seed_mean_sampleSD"][key][b], "core stability aggregate")
                    ref = next(r for r in self.reference_summary if r["dataset"] == tag and r["method"] == method and r["population"] == pop)
                    self.near(ref["mean_overlap10"], old["training_seed_baseline_three_pair_overlap_mean_reference"], "core training-seed reference")
        x3 = self.read(self.base / "g2_x3_inference_results_review_20261027/independent_summary.csv")
        for old in x3:
            key = (old["dataset"], old["method"].capitalize(), old["population"], old["metric"])
            row = self.summary_lookup[key]
            self.near(row["mean"], float(old["mean"]), "X3 audited aggregate mean")
            self.near(row["sample_sd"], float(old["sample_SD_ddof1"]), "X3 audited aggregate SD")
        x4 = self.read(self.base / "g2_x4_tisasrec_results_review_20261026/math_independent_audit.json")
        self.check(x4["status"] == "PASS", "TiSASRec prior audit PASS")
        for tag in ("ml1m", "beauty"):
            for metric in ACCURACY:
                old = x4["dataset_summaries"][tag]["canonical_accuracy"][metric]
                row = self.summary_lookup[(tag, "TiSASRec", "full", metric)]
                self.near(row["mean"], old["mean"], "TiSASRec accuracy mean")
                self.near(row["sample_sd"], old["sample_SD_ddof1"], "TiSASRec accuracy SD")

    def subgroup_contrasts(self):
        out = []
        for tag in TAGS:
            for metric in ACCURACY:
                values = []
                for seed in SEEDS:
                    def value(method, pop):
                        return self.lookup[(tag, method, pop, seed, metric)]["value"]
                    pooled = value("LBSP", "pooled") - value("Control", "pooled")
                    singleton = value("LBSP", "singleton") - value("Control", "singleton")
                    values.append(pooled - singleton)
                    out.append({"dataset": tag, "metric": metric, "training_seed": seed, "pooled_delta": pooled,
                                "singleton_delta": singleton, "pooled_minus_singleton_delta": pooled - singleton})
        return out

    def write(self, name, rows):
        self.check(bool(rows), "nonempty output " + name)
        with (self.out / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def main(self):
        plan = self.read(self.out / "statistical_analysis_plan.json")
        self.check(plan["fixed_before_computation"] and plan["training_seeds"] == list(SEEDS), "fixed prespecified analysis plan")
        self.ensemble_groups = []
        self.prepare_labels()
        for tag in TAGS:
            for method in METHODS:
                if method == "TiSASRec" and tag not in ("ml1m", "beauty"):
                    continue
                for seed in SEEDS:
                    self.load_run(tag, method, seed)
        self.summarize()
        self.seed_references()
        self.baselines()
        self.verify_previous()
        self.write("per_seed_metrics.csv", self.rows)
        self.write("method_summary.csv", self.summary)
        self.write("paired_differences_per_seed.csv", self.deltas)
        self.write("paired_difference_summary.csv", self.delta_summary)
        self.write("seed_reference_pairs.csv", self.reference_pairs)
        self.write("seed_reference_summary.csv", self.reference_summary)
        self.write("ensemble_group_accuracy.csv", self.ensemble_groups)
        self.write("analytic_baselines.csv", self.baseline_rows)
        self.write("lbsp_subgroup_contrasts_per_seed.csv", self.subgroup_contrasts())
        bucket = [r | {"bucket": r["population"].replace("affected_bucket", "").replace("5plus", ">=5"),
                           "population_definition": "affected users within last-bucket-size stratum"}
                  for r in self.summary if r["population"].startswith("affected_bucket") and r["metric"] in ("overlap10", "hitflip")]
        self.write("bucket_stability_summary.csv", bucket)
        all_bucket = [r | {"bucket": r["population"].replace("bucket", "").replace("5plus", ">=5"),
                               "population_definition": "all users within last-bucket-size stratum"}
                      for r in self.summary if r["population"].startswith("bucket") and r["metric"] in ("overlap10", "hitflip")]
        self.write("bucket_stability_all_users_summary.csv", all_bucket)
        self.write("population_counts.csv", [{"dataset": tag, "population": pop, "n": len(self.population(self.labels[tag], pop)),
                                               "affected_n": len(self.population(self.labels[tag], "affected"))} for tag in TAGS for pop in POPULATIONS])
        subgroup = [r for r in self.delta_summary if r["method"] == "LBSP" and r["population"] in ("pooled", "singleton") and r["metric"] in ACCURACY]
        self.write("lbsp_subgroup_difference_summary.csv", subgroup)
        manifest = {"scope": "All read inputs; saved predictions and metadata only", "files": self.manifest,
                    "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (self.out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        report = {"status": "PASS", "counts": dict(self.counts), "evaluated_configurations": len(self.runs),
                  "independently_trained_models": 42, "ensemble_configurations_reusing_Control": 12,
                  "three_seed_method_dataset_groups": len(self.runs) // 3, "per_seed_rows": len(self.rows),
                  "method_summary_rows": len(self.summary), "paired_summary_rows": len(self.delta_summary),
                  "training_seed_pairs": len(self.reference_pairs), "baseline_rows": len(self.baseline_rows),
                  "source_file_count": len(self.manifest), "precomputed_prior_audits_matched": True,
                  "raw_interaction_or_checkpoint_read_or_stat": False, "archived_code_or_torch_executed": False,
                  "limitations": ["Training-seed CI based on only three seeds and Gaussian assumption; not a significance claim.",
                                 "No TiSASRec measurements for Sports/Toys; no invented rows.",
                                 "Saved raw-history hashes and GPU rank provenance were not independently reconstructed here.",
                                 "Ensemble fixed-group averaging does not integrate all possible permutations.",
                                 "Intervals are not truncated to metric bounds; publication figures must disclose any display clipping."]}
        (self.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    Builder(args.root).main()
