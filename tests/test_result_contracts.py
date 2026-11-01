"""Synthetic release-contract checks only. No model, scientific statistics or data use."""
from pathlib import Path
import csv
import hashlib
import json
import sys

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "outputs" / "contract_fixture"
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "tools"))
import replay_pipeline as pipeline
import recompute_statistics as statistics


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    fixture = HERE / "synthetic_results"
    fixture.mkdir(exist_ok=True)
    registry_path = HERE / "synthetic_registry.json"
    registry = pipeline.registry_template()
    registry["fixture_only"] = True
    registry["label_files"] = {}
    registry["target_metadata"] = {}
    for row in registry["runs"]:
        tag = row["dataset"]
        folder = fixture / row["output_directory"]
        label = {"sample_id": tag + "_synthetic_sample", "user_id": tag + "_synthetic_user",
                 "target_item": tag + "_synthetic_item1", "last_bucket_size": 2, "affected": 1}
        write_csv(folder / "panel_labels.csv", [label])
        prediction = {"sample_id": label["sample_id"], "user_id": label["user_id"],
                      "target_item": label["target_item"], "target_rank": 1,
                      "top10_raw_item_id": json.dumps([tag + "_synthetic_item" + str(i) for i in range(1, 11)])}
        names = ["predictions_original.csv"] + row["required_predictions"]
        for name in set(names):
            write_csv(folder / name, [prediction])
        registry["label_files"].setdefault(tag, row["output_directory"] + "/panel_labels.csv")
        registry["target_metadata"].setdefault(tag, "synthetic_metadata/" + tag + ".csv")
    write_json(registry_path, registry)
    report = {"fixture_only": True, "training_or_full_statistics_executed": False, "checks": {}}
    checks = report["checks"]
    valid = pipeline.validate_results(fixture, registry_path)
    write_json(HERE / "validated_registry.json", valid)
    checks["positive_54_configuration_contract"] = valid["status"] == "PASS_COMPLETE_54_CONFIGURATION_RESULT_CONTRACT"
    checks["output_directories_retained"] = all(a["output_directory"] == b["output_directory"] for a, b in zip(registry["runs"], valid["runs"]))
    checks["metadata_maps_retained"] = all(valid[name] == registry[name] for name in ("label_files", "target_metadata"))
    checks["hash_scope_explicit"] = all(r.get("files_sha256_scope") == "output_directory" for r in valid["runs"])
    verified = 0
    root_scope_verified = 0
    for run in valid["runs"]:
        hash_root = statistics.under(fixture, run["output_directory"])
        for relative, expected in run["files_sha256"].items():
            actual = statistics.sha(statistics.under(hash_root, relative))
            assert actual == expected
            verified += 1
            root_relative = run["output_directory"] + "/" + relative
            assert statistics.sha(statistics.under(fixture, root_relative)) == expected
            root_scope_verified += 1
    checks["all_output_scope_hashes_resolve"] = verified > 0
    checks["all_root_scope_hashes_resolve"] = root_scope_verified == verified
    report["file_hashes_verified_per_scope"] = verified

    ensemble = next(r for r in registry["runs"] if r["method"] == "Ensemble")
    original = fixture / ensemble["output_directory"] / "predictions_original.csv"
    original_bytes = original.read_bytes()
    original.unlink()
    try:
        invalid = pipeline.validate_results(fixture, registry_path)
        checks["missing_ensemble_original_rejected"] = invalid["status"].startswith("FAIL_") and any(x.get("missing") == "predictions_original.csv" for x in invalid["failures"])
    finally:
        original.write_bytes(original_bytes)

    prior = registry["runs"][0]["output_directory"]
    registry["runs"][0]["output_directory"] = "../outside_fixture"
    write_json(registry_path, registry)
    try:
        pipeline.validate_results(fixture, registry_path)
        checks["escaped_output_path_rejected"] = False
    except (AssertionError, ValueError):
        checks["escaped_output_path_rejected"] = True
    finally:
        registry["runs"][0]["output_directory"] = prior
        write_json(registry_path, registry)

    compare_out, compare_ref = HERE / "compare_output", HERE / "compare_reference"
    compare_out.mkdir(exist_ok=True)
    compare_ref.mkdir(exist_ok=True)
    write_csv(compare_out / "numbers.csv", [{"dataset": "synthetic", "value": "0.75"}])
    write_csv(compare_ref / "numbers.csv", [{"dataset": "synthetic", "value": "0.75"}])
    compared = statistics.compare_outputs(compare_out, compare_ref, ["numbers.csv"])
    checks["matching_numeric_reference_accepted"] = compared["numbers.csv"]["fields_checked"] == 2
    try:
        statistics.compare_outputs(compare_out, compare_ref, ["missing.csv"])
        checks["missing_reference_file_rejected"] = False
    except (FileNotFoundError, ValueError):
        checks["missing_reference_file_rejected"] = True
    write_csv(compare_ref / "numbers.csv", [{"dataset": "synthetic", "value": "0.76"}])
    try:
        statistics.compare_outputs(compare_out, compare_ref, ["numbers.csv"])
        checks["changed_reference_value_rejected"] = False
    except (AssertionError, ValueError):
        checks["changed_reference_value_rejected"] = True

    report["status"] = "PASS_SYNTHETIC_RELEASE_CONTRACTS" if all(checks.values()) else "FAIL_SYNTHETIC_RELEASE_CONTRACTS"
    report["failures"] = [name for name, value in checks.items() if not value]
    report["tool_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (REPO / "tools/replay_pipeline.py", REPO / "tools/recompute_statistics.py")}
    write_json(HERE / "contract_report.json", report)
    print(json.dumps(report, indent=2))
    return int(bool(report["failures"]))


if __name__ == "__main__":
    sys.exit(main())
