#!/usr/bin/env python3
"""Create a PRIVATE KAIS verification snapshot; Python 3.8+, standard library only.

Reads inventories from a previous G2 audit. Never imports research modules, runs
training/inference, loads checkpoints, uses the network, or changes source files.
The output is a private verification package, NOT a public reproducibility repo.
"""

import argparse
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import shutil
import stat
import sys
import uuid
import zipfile


VERSION = "1.1.0"
CHUNK = 1024 * 1024
FILE_LIMIT = 64 * 1024 * 1024
RESULT_LIMIT = 256 * 1024 * 1024
CHECKPOINTS = {".pt", ".pth", ".ckpt"}
DATASETS = {"amazon_beauty", "amazon_sports", "amazon_toys", "movielens-1m",
            "beauty", "sports", "toys", "ml1m", "movielens"}
EXTRA_SOURCE = [
    "audit/23_audit_natural_timestamp.py",
    "audit/24_freeze_natural_preprocessed.py",
    "audit/25_prepare_diagnostic_a.py",
    "audit/27_1_prepare_diagnostic_b_permutations.py",
    "audit/27_2_evaluate_diagnostic_b.py",
    "audit/29_2_evaluate_movielens_diagnostic_b.py",
]
EXTRA_DATA = [
    "processed/natural_timestamp/Amazon_Beauty/interactions_5core.csv",
    "processed/natural_timestamp/MovieLens-1M/interactions_5core.csv",
]
HASH_FIELDS = ["path", "bytes", "mtime_ns", "sha256", "audit_sha256",
               "audit_sha_match", "status", "note", "snapshot_path"]


class FreezeError(Exception):
    pass


def stamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


def inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def lexical_rel(value):
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise FreezeError("rejected_path_syntax")
    p = PurePosixPath(value)
    if p.is_absolute() or PureWindowsPath(value).drive:
        raise FreezeError("rejected_absolute_path")
    raw_parts = value.split("/")
    if any(x in ("", ".", "..") for x in raw_parts):
        raise FreezeError("rejected_path_component")
    low = value.lower()
    if "kuairec" in low or "step39" in low:
        raise FreezeError("rejected_unrelated_project")
    if any(x.startswith(".") or x.lower() in ("notes", "credentials", "secrets")
           for x in p.parts):
        raise FreezeError("rejected_private_or_notes_path")
    return p


def permitted(value, category):
    p = lexical_rel(value)
    parts = p.parts
    if category == "source":
        good = (value.startswith("audit/") or
                value.startswith("step38_kais_retrain/scripts/") or
                value.startswith("paper_20261022/")) and p.suffix.lower() == ".py"
    elif category == "evidence":
        good = (value.startswith("reproducibility/step36/") or
                value.startswith("paper_20261022/")) and p.suffix.lower() == ".json"
    elif category == "result":
        good = len(parts) >= 4 and parts[:2] == ("step38_kais_retrain", "outputs")
        if good:
            name = parts[2].lower()
            target = re.search(r"(?:^|_)(?:ml1m|movielens(?:[-_]1m)?|beauty|sports|toys)(?:_|$)", name)
            aggregate = re.match(r"^step38_(?:18|23|27|29|30)(?:_|$)", name)
            good = bool(target or aggregate)
        good = good and p.suffix.lower() in ({".json", ".csv"} | CHECKPOINTS)
    elif category == "data":
        good = len(parts) >= 3 and parts[0] in ("data", "processed")
        good = good and any(x.lower() in DATASETS for x in parts[1:])
    else:
        good = False
    if not good:
        raise FreezeError("rejected_category_scope")
    return p


def safe_path(root, relative, category=None, must_exist=True):
    p = permitted(relative, category) if category else lexical_rel(relative)
    path = root
    for part in p.parts:
        path = path / part
        if path.is_symlink():
            raise FreezeError("rejected_symbolic_link")
    resolved = path.resolve(strict=False)
    if not inside(resolved, root):
        raise FreezeError("rejected_path_escape")
    if must_exist and not path.exists():
        raise FreezeError("missing")
    return path


def signature(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def stable_read(root, relative, category=None, retain=False, limit=None):
    """Hash one regular file with before/after fd AND pathname checks."""
    path = safe_path(root, relative, category)
    before_path = path.lstat()
    if not stat.S_ISREG(before_path.st_mode):
        raise FreezeError("not_regular_file")
    if limit is not None and before_path.st_size > limit:
        raise FreezeError("skipped_file_size_limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags)
    h = hashlib.sha256()
    total = 0
    contents = [] if retain else None
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or signature(before) != signature(before_path):
            raise FreezeError("changed_before_read")
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            total += len(block)
            if limit is not None and total > limit:
                raise FreezeError("skipped_file_size_limit")
            h.update(block)
            if retain:
                contents.append(block)
        after = os.fstat(handle.fileno())
    safe_path(root, relative, category)
    after_path = path.lstat()
    if (signature(before) != signature(after) or
            signature(before) != signature(after_path) or total != before.st_size):
        raise FreezeError("changed_during_read")
    return {"bytes": total, "mtime_ns": before.st_mtime_ns, "sha256": h.hexdigest(),
            "signature": signature(before),
            "content": b"".join(contents) if retain else None}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def package_versions():
    values = []
    try:
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get("Name", "")
            version = dist.version or ""
            if re.fullmatch(r"[A-Za-z0-9_.-]+", name) and re.fullmatch(r"[A-Za-z0-9_.+!-]+", version):
                values.append({"name": name, "version": version})
    except Exception:
        return {"status": "partially_available", "packages": values}
    return {"status": "sampled", "packages": sorted(values, key=lambda x: x["name"].lower())}


def sanitize_environment(value):
    """Retain only expected audit fields; never export arbitrary environment vars."""
    result = {}
    if not isinstance(value, dict):
        return {"status": "unrecognized_previous_environment_format"}
    for key in ("python_version", "python_executable", "platform", "machine",
                "note", "gpu_query_status", "gpu_name_memory_driver"):
        item = value.get(key)
        if isinstance(item, (str, int, float, bool)):
            result[key] = str(item)[:4096]
    packages = value.get("packages", [])
    clean = []
    if isinstance(packages, dict):
        packages = [{"name": k, "version": v} for k, v in packages.items()]
    if isinstance(packages, list):
        for item in packages:
            if not isinstance(item, dict):
                continue
            name, version = str(item.get("name", "")), str(item.get("version", ""))
            if re.fullmatch(r"[A-Za-z0-9_.-]+", name) and re.fullmatch(r"[A-Za-z0-9_.+!-]+", version):
                clean.append({"name": name, "version": version})
    result["packages"] = clean
    result["scope_note"] = "Previous audit sample, not a reconstructed training environment."
    return result


class Freezer:
    def __init__(self, root, audit_dir, output):
        self.root, self.audit_dir, self.output = root, audit_dir, output
        self.snapshot = output / "snapshot"
        self.sources, self.evidence, self.results, self.data = [], [], [], []
        self.verify, self.run_checks, self.audit_inputs, self.order_checks = [], [], [], []
        self.warnings = []
        self.failures = []
        self.frozen = {}
        self.result_bytes = 0
        self.bundle = None

    def audit_file(self, name):
        relative = (self.audit_dir / name).relative_to(self.root).as_posix()
        record = stable_read(self.root, relative, retain=True, limit=FILE_LIMIT)
        self.audit_inputs.append({"path": relative, "bytes": record["bytes"],
                                  "sha256": record["sha256"], "status": "sampled_at_freeze"})
        return record["content"]

    def inventory(self, name, category):
        text = self.audit_file(name).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text, newline="")))
        if not rows:
            self.warnings.append("empty_inventory:" + name)
        clean = {}
        for row in rows:
            relative = row.get("path", "")
            try:
                permitted(relative, category)
                safe_path(self.root, relative, category, must_exist=False)
            except (FreezeError, OSError) as exc:
                self.verify.append({"category": category, "path": relative,
                                    "status": str(exc), "audit_sha_match": "false"})
                self.failures.append("unsafe_inventory_path:" + name)
                continue
            if relative in clean and row.get("sha256", "") != clean[relative].get("sha256", ""):
                self.failures.append("conflicting_duplicate_inventory_path:" + relative)
                continue
            clean[relative] = row
        return clean

    def inspect(self, relative, category, old="", required=False, new=False):
        row = {"path": relative, "audit_sha256": old, "audit_sha_match": "unknown",
               "snapshot_path": ""}
        try:
            info = stable_read(self.root, relative, category,
                               limit=None if category == "data" else FILE_LIMIT)
            row.update({k: info[k] for k in ("bytes", "mtime_ns", "sha256")})
            if old:
                if not re.fullmatch(r"[a-fA-F0-9]{64}", old):
                    raise FreezeError("invalid_prior_sha256")
                matched = info["sha256"] == old.lower()
                row["audit_sha_match"] = "true" if matched else "false"
                if not matched:
                    raise FreezeError("changed_since_audit")
                row["status"] = "audit_sha_matched"
            elif required:
                raise FreezeError("missing_prior_sha256")
            else:
                row["status"] = "newly_collected_no_prior_hash" if new else "current_hash_only"
            if category == "data":
                row["status"] = "newly_discovered" if new else "current_snapshot_hash"
                row["note"] = "No data copied; referenced historical manifest hashes not verified."
            return row, info
        except (OSError, FreezeError) as exc:
            status = str(exc) if isinstance(exc, FreezeError) else "read_error:" + type(exc).__name__
            if new and status == "missing":
                status = "missing_optional_upstream" if category == "source" else "missing_optional_data"
            row["status"] = status
            row["note"] = "No source/result content printed."
            if status == "changed_since_audit":
                row["audit_sha_match"] = "false"
            limited_result = category == "result" and status == "skipped_file_size_limit"
            if not limited_result and (required or (old and category != "data") or status.startswith("rejected_")):
                self.failures.append(category + ":" + relative + ":" + status)
            else:
                self.warnings.append(category + ":" + relative + ":" + status)
            return row, None

    def copy_verified(self, relative, category, record, info):
        """copy2 only after preflight; validate both source and destination again."""
        destination = self.snapshot / relative
        if not inside(destination.resolve(strict=False), self.snapshot.resolve(strict=False)):
            raise FreezeError("snapshot_path_escape")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FreezeError("snapshot_destination_already_exists")
        try:
            current = stable_read(self.root, relative, category, limit=FILE_LIMIT)
            if current["sha256"] != info["sha256"] or current["signature"] != info["signature"]:
                raise FreezeError("changed_before_copy")
            source = safe_path(self.root, relative, category)
            shutil.copy2(str(source), str(destination), follow_symlinks=False)
            copied = stable_read(self.snapshot, relative, category, limit=FILE_LIMIT)
            after = stable_read(self.root, relative, category, limit=FILE_LIMIT)
            if (copied["sha256"] != info["sha256"] or after["sha256"] != info["sha256"] or
                    after["signature"] != info["signature"]):
                raise FreezeError("changed_during_copy")
            record["snapshot_path"] = "snapshot/" + relative
            record["copy_status"] = "verified_copy"
            self.frozen[relative] = destination
        except Exception:
            if destination.is_symlink() or destination.is_file():
                destination.unlink()
            raise

    def discover_aggregates(self, rows):
        base = self.root / "step38_kais_retrain" / "outputs"
        if not base.exists():
            return
        safe_path(self.root, "step38_kais_retrain/outputs")
        for directory in sorted(base.iterdir()):
            if not re.match(r"^step38_(?:27|29|30)(?:_|$)", directory.name.lower()):
                continue
            rel = directory.relative_to(self.root).as_posix()
            if directory.is_symlink():
                self.failures.append("rejected_symbolic_link:" + rel)
                continue
            if not directory.is_dir():
                continue
            # Only immediate files; never recurse into arbitrary sibling projects.
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() not in (".json", ".csv"):
                    continue
                relative = path.relative_to(self.root).as_posix()
                try:
                    permitted(relative, "result")
                    safe_path(self.root, relative, "result", must_exist=False)
                except (FreezeError, OSError) as exc:
                    self.failures.append("discovered_aggregate_rejected:" + str(exc))
                    continue
                if relative not in rows:
                    rows[relative] = {"path": relative, "sha256": "", "newly_discovered": True}

    def perform(self):
        print("1/5 Reading previous inventories and recording environment", flush=True)
        source_rows = self.inventory("source_inventory.csv", "source")
        evidence_rows = self.inventory("evidence_inventory.csv", "evidence")
        result_rows = self.inventory("run_inventory.csv", "result")
        data_rows = self.inventory("data_inventory.csv", "data")
        prior_environment = json.loads(self.audit_file("environment.json").decode("utf-8-sig"))
        save_json(self.output / "previous_audit_environment.json", sanitize_environment(prior_environment))
        save_json(self.output / "freeze_environment.json", {
            "sampled_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "python_version": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "packages": package_versions(),
            "note": "Freeze-time environment only; no GPU libraries or research scripts imported."})
        if self.failures:
            return
        candidates = []
        print("2/5 Verifying source and evidence hashes", flush=True)
        # ALL audited source/evidence must match before ANY snapshot file is copied.
        for category, rows, report in (("source", source_rows, self.sources),
                                       ("evidence", evidence_rows, self.evidence)):
            for relative, old_row in sorted(rows.items()):
                record, info = self.inspect(relative, category, old_row.get("sha256", ""), required=True)
                report.append(record)
                self.verify.append(dict(record, category=category))
                if info:
                    candidates.append((relative, category, record, info))
        if self.failures:
            return
        for relative in EXTRA_SOURCE:
            if relative in source_rows:
                continue
            record, info = self.inspect(relative, "source", new=True)
            self.sources.append(record)
            if info:
                candidates.append((relative, "source", record, info))
        self.warnings.append("Source dependency closure is NOT established; optional upstream files are only explicit additions.")
        self.discover_aggregates(result_rows)
        for relative, old_row in sorted(result_rows.items()):
            if PurePosixPath(relative).suffix.lower() in CHECKPOINTS:
                self.results.append({"path": relative, "status": "excluded_checkpoint",
                                     "audit_sha_match": "unknown", "note": "Never loaded, hashed, or copied."})
                continue
            record, info = self.inspect(relative, "result", old_row.get("sha256", ""),
                                        new=bool(old_row.get("newly_discovered")))
            self.results.append(record)
            if info:
                if self.result_bytes + info["bytes"] > RESULT_LIMIT:
                    record["status"] = "skipped_total_result_size_limit"
                    self.warnings.append("Results exceed 256 MiB total copy limit; snapshot is incomplete.")
                else:
                    self.result_bytes += info["bytes"]
                    candidates.append((relative, "result", record, info))
        if self.failures:
            return
        print("3/5 Collecting verified code and derived results", flush=True)
        for relative, category, record, info in candidates:
            try:
                self.copy_verified(relative, category, record, info)
            except (OSError, FreezeError) as exc:
                record["status"] = "copy_verification_failed"
                self.failures.append(category + ":" + relative + ":" + type(exc).__name__)
                return
        for relative in EXTRA_DATA:
            if relative not in data_rows:
                data_rows[relative] = {"path": relative, "newly_discovered": True}
        print("4/5 Hashing data (read only; data and checkpoints are not copied)", flush=True)
        for relative, old_row in sorted(data_rows.items()):
            record, _ = self.inspect(relative, "data", new=bool(old_row.get("newly_discovered")))
            self.data.append(record)
        self.run_checks = self.check_runs(result_rows)
        for row in self.run_checks:
            for key in ("config_seed_equals_metrics_seed", "best_epoch_in_log",
                        "best_epoch_matches_max_val_ndcg", "metrics_validation_matches_log"):
                if row.get(key) == "false":
                    self.warnings.append("metadata_check_failed:" + row["run"] + ":" + key)
        print("5/5 Checking Sports/Toys temporal order without editing inputs", flush=True)
        self.order_checks = self.check_time_order()

    def presence(self, relative):
        try:
            path = safe_path(self.root, relative, "result")
            return "present" if path.is_file() else "not_regular_file"
        except (OSError, FreezeError) as exc:
            return "not_observed" if str(exc) == "missing" else "unknown"

    def snapshot_json(self, relative):
        path = self.frozen.get(relative)
        if path is None:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            return value if isinstance(value, dict) else None
        except (ValueError, OSError):
            return None

    def check_runs(self, result_rows):
        directories = set()
        for relative in result_rows:
            name = PurePosixPath(relative).parts[2]
            if "max200" in name.lower() and "smoke" not in name.lower():
                directories.add(name)
        checked = []
        for name in sorted(directories):
            base = "step38_kais_retrain/outputs/" + name + "/"
            row = {"run": name, "scope": "limited_metadata_checks_not_reproduction"}
            for label, file_name in (("config", "config.json"), ("metrics", "metrics.json"),
                                     ("training_log", "training_log.csv"),
                                     ("item_map", "item_id_map.csv"),
                                     ("test_predictions", "test_per_sample_predictions.csv")):
                row[label + "_exists"] = self.presence(base + file_name)
            checkpoint_names = ["r1_meanpool_best.pt", "matched_control_sasrec_best.pt"]
            statuses = [self.presence(base + n) for n in checkpoint_names]
            row["checkpoint_exists"] = "present" if "present" in statuses else "not_observed"
            row["checkpoint_content_verified"] = "false"
            config, metrics = self.snapshot_json(base + "config.json"), self.snapshot_json(base + "metrics.json")
            row["config_seed_equals_metrics_seed"] = "unknown"
            row["best_epoch_in_log"] = "unknown"
            row["best_epoch_matches_max_val_ndcg"] = "unknown"
            row["metrics_validation_matches_log"] = "unknown"
            row["formal_completion"] = "not_certified"
            if config is not None and metrics is not None:
                if "seed" in config and "seed" in metrics:
                    row["config_seed_equals_metrics_seed"] = str(str(config["seed"]) == str(metrics["seed"])).lower()
                row["config_seed"] = str(config.get("seed", "unknown"))
                row["metrics_seed"] = str(metrics.get("seed", "unknown"))
                row["best_epoch"] = str(metrics.get("best_epoch", "unknown"))
            log_path = self.frozen.get(base + "training_log.csv")
            if log_path is not None and metrics is not None:
                try:
                    with log_path.open(encoding="utf-8-sig", newline="") as handle:
                        log = list(csv.DictReader(handle))
                    epochs = [finite_number(r.get("epoch")) for r in log]
                    best = finite_number(metrics.get("best_epoch"))
                    valid_epochs = [e for e in epochs if e is not None]
                    row["log_rows"] = len(log)
                    row["max_logged_epoch"] = max(valid_epochs) if valid_epochs else "unknown"
                    if best is not None and valid_epochs:
                        row["best_epoch_in_log"] = str(best in valid_epochs).lower()
                    pairs = []
                    for entry, epoch in zip(log, epochs):
                        value = None
                        for key in ("val_ndcg10", "val_ndcg@10", "validation_ndcg10", "val_NDCG@10"):
                            if key in entry:
                                value = finite_number(entry[key])
                                break
                        if epoch is not None and value is not None:
                            pairs.append((epoch, value))
                    if best is not None and pairs:
                        maximum = max(v for _, v in pairs)
                        chosen = [v for e, v in pairs if e == best]
                        if chosen:
                            row["best_epoch_matches_max_val_ndcg"] = str(any(
                                math.isclose(v, maximum, rel_tol=1e-6, abs_tol=1e-8) for v in chosen)).lower()
                            row["log_max_val_ndcg10"] = maximum
                            validation = metrics.get("validation", {})
                            if isinstance(validation, dict):
                                overall = validation.get("overall", validation)
                                if isinstance(overall, dict):
                                    mv = finite_number(overall.get("NDCG@10", overall.get("ndcg10")))
                                    if mv is not None:
                                        row["metrics_validation_matches_log"] = str(any(
                                            math.isclose(v, mv, rel_tol=1e-6, abs_tol=1e-8) for v in chosen)).lower()
                except (ValueError, OSError, csv.Error):
                    row["log_parse_status"] = "unknown"
            checked.append(row)
        return checked

    def check_time_order(self):
        results = []
        for dataset in ("Amazon_Sports", "Amazon_Toys"):
            for relative in ("processed/diagnostic_a/" + dataset + "/file_order.csv",
                             "processed/diagnostic_a_splits/" + dataset + "/file_order/train.csv"):
                row = {"path": relative, "dataset": dataset,
                       "tested_order": "per_user_raw_line_ascending_as_in_38_12_38_13",
                       "normative_order": "user_id,time_bucket,raw_line", "status": "unknown"}
                try:
                    path = safe_path(self.root, relative, "data")
                    before = path.lstat()
                    if not stat.S_ISREG(before.st_mode):
                        raise FreezeError("not_regular_file")
                    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_BINARY", 0)
                                 | getattr(os, "O_NOFOLLOW", 0))
                    users = defaultdict(list)
                    input_descents = 0
                    last = {}
                    count = 0
                    with os.fdopen(fd, "rb") as binary:
                        if signature(os.fstat(binary.fileno())) != signature(before):
                            raise FreezeError("changed_before_order_check")
                        with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
                            reader = csv.DictReader(text)
                            if not {"user_id", "raw_line", "time_bucket"}.issubset(reader.fieldnames or []):
                                raise FreezeError("missing_order_columns")
                            for entry in reader:
                                user = entry["user_id"]
                                raw_line, bucket = int(entry["raw_line"]), int(entry["time_bucket"])
                                if user in last and bucket < last[user]:
                                    input_descents += 1
                                last[user] = bucket
                                users[user].append((raw_line, bucket))
                                count += 1
                            if signature(os.fstat(binary.fileno())) != signature(before):
                                raise FreezeError("changed_during_order_check")
                    safe_path(self.root, relative, "data")
                    if signature(path.lstat()) != signature(before):
                        raise FreezeError("changed_during_order_check")
                    descents, bad_users, duplicates = 0, 0, 0
                    for values in users.values():
                        values.sort(key=lambda item: item[0])
                        n = sum(values[i][1] < values[i-1][1] for i in range(1, len(values)))
                        duplicates += sum(values[i][0] == values[i-1][0] for i in range(1, len(values)))
                        descents += n
                        bad_users += int(n > 0)
                    row.update({"status": "checked", "rows": count, "users": len(users),
                                "file_order_adjacent_bucket_descents": input_descents,
                                "raw_line_order_adjacent_bucket_descents": descents,
                                "users_with_raw_line_order_descents": bad_users,
                                "duplicate_user_raw_line_keys": duplicates,
                                "finding": "raw_line_order_not_time_monotonic" if descents else "no_descent_in_checked_order"})
                    if descents:
                        self.warnings.append("temporal_order_issue:" + relative + ":users=" + str(bad_users))
                except (OSError, ValueError, KeyError, csv.Error, FreezeError) as exc:
                    row["status"] = str(exc) if isinstance(exc, FreezeError) else "unknown:" + type(exc).__name__
                    self.warnings.append("order_check_incomplete:" + relative + ":" + row["status"])
                results.append(row)
        return results

    def write_reports(self):
        save_csv(self.output / "SOURCE_SHA256.csv", self.sources, HASH_FIELDS)
        save_csv(self.output / "EVIDENCE_SHA256.csv", self.evidence, HASH_FIELDS)
        save_csv(self.output / "RESULT_SHA256.csv", self.results, HASH_FIELDS)
        save_csv(self.output / "DATA_SHA256.csv", self.data, HASH_FIELDS)
        save_csv(self.output / "VERIFY.csv", self.verify, ["category"] + HASH_FIELDS)
        save_csv(self.output / "AUDIT_INPUT_SHA256.csv", self.audit_inputs,
                 ["path", "bytes", "sha256", "status"])
        fields = ["run", "scope", "config_exists", "metrics_exists", "training_log_exists",
                  "item_map_exists", "checkpoint_exists", "test_predictions_exists",
                  "config_seed", "metrics_seed", "config_seed_equals_metrics_seed", "best_epoch",
                  "best_epoch_in_log", "best_epoch_matches_max_val_ndcg", "metrics_validation_matches_log",
                  "log_rows", "max_logged_epoch", "log_max_val_ndcg10", "log_parse_status",
                  "checkpoint_content_verified", "formal_completion"]
        save_csv(self.output / "RUN_CHECKS.csv", self.run_checks, fields)
        save_csv(self.output / "ORDER_CHECKS.csv", self.order_checks,
                 ["path", "dataset", "tested_order", "normative_order", "status", "rows", "users",
                  "file_order_adjacent_bucket_descents", "raw_line_order_adjacent_bucket_descents",
                  "users_with_raw_line_order_descents", "duplicate_user_raw_line_keys", "finding"])
        failures = list(dict.fromkeys(self.failures))
        warnings = list(dict.fromkeys(self.warnings))
        status = "FAILED_NO_BUNDLE" if failures else "PRIVATE_SNAPSHOT_WITH_LIMITATIONS"
        result = {"script_version": VERSION, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "status": status, "root": str(self.root), "audit_dir": str(self.audit_dir),
                  "output_dir": str(self.output), "source_count": len(self.sources),
                  "evidence_count": len(self.evidence), "result_records": len(self.results),
                  "data_records": len(self.data), "copied_files": len(self.frozen),
                  "order_files_checked": sum(r.get("status") == "checked" for r in self.order_checks),
                  "order_files_with_raw_line_descents": sum(r.get("raw_line_order_adjacent_bucket_descents", 0) > 0 for r in self.order_checks),
                  "copied_result_bytes": self.result_bytes,
                  "limits": {"single_metadata_or_result_bytes": FILE_LIMIT, "total_result_bytes": RESULT_LIMIT},
                  "failures": failures, "warnings": warnings,
                  "claims": {"public_repository": False, "complete_dependency_closure": False,
                             "fully_reproduced": False, "checkpoint_loaded": False,
                             "historical_data_manifest_hashes_verified": False,
                             "source_files_modified": False, "network_used": False},
                  "bundle_filename_if_successful": "g2_private_snapshot.zip" if not failures else None}
        save_json(self.output / "freeze_result.json", result)
        summary = ["# G2 PRIVATE verification snapshot", "", "Status: **" + status + "**", "",
                   "This package is private checking material, NOT a publishable repository or a complete reproduction.",
                   "Prediction CSVs may contain per-user information. Review and minimize them before any public release.",
                   "No training, model inference, checkpoint loading, network access, or research-script execution occurred.", "",
                   "- Source/evidence require an exact prior-audit SHA256 match before copying; explicit optional upstream additions have no prior hash.",
                   "- Any mismatch in audited source/evidence or a hashed result prevents bundle creation.",
                   "- Data are hashed only, never copied. DATA_SHA256.csv establishes CURRENT snapshot hashes; it does not validate hashes cited by old manifests.",
                   "- Checkpoints are checked for presence only, never read/loaded/hashed/copied.",
                   "- RUN_CHECKS.csv performs limited metadata checks. Unknown means insufficient comparable fields; no run is certified as fully reproduced or formally completed.",
                   "- ORDER_CHECKS.csv checks Sports/Toys per-user temporal order under raw_line sorting; no input or training code is changed.",
                   "- Result files are limited to 64 MiB each and 256 MiB total. Skipped files make the snapshot incomplete.",
                   "- The current environment is sampled at freeze time; previous_audit_environment.json records an earlier sample, not the proven training environment.",
                   "- Source dependency closure remains incomplete/unproven, including optional upstream paths requiring follow-up.", "",
                   "Copied files: " + str(len(self.frozen)),
                   "Data hash records: " + str(len(self.data)),
                   "", "## Failures", ""]
        summary.extend(["- " + x for x in failures] or ["- None detected in the performed checks."])
        summary.extend(["", "## Warnings and follow-up", ""])
        summary.extend(["- " + x for x in warnings] or ["- Review metadata checks, source closure, licenses and release scope before publication."])
        summary.extend(["", "Do not upload this private ZIP directly as a public dataset/code release.", ""])
        (self.output / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
        return not failures

    def zip_reports(self):
        destination = self.output / "g2_private_snapshot.zip"
        if destination.exists():
            raise FreezeError("bundle_already_exists")
        files = []
        for path in self.output.rglob("*"):
            if path.is_symlink():
                raise FreezeError("symbolic_link_in_output")
            if path.is_file() and path != destination:
                relative = path.relative_to(self.output).as_posix()
                if relative.startswith("snapshot/"):
                    source_rel = relative[len("snapshot/"):]
                    if source_rel not in self.frozen:
                        raise FreezeError("unregistered_snapshot_file")
                    if PurePosixPath(source_rel).suffix.lower() in CHECKPOINTS:
                        raise FreezeError("checkpoint_in_snapshot")
                files.append(path)
        with zipfile.ZipFile(str(destination), mode="x", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as bundle:
            for path in sorted(files):
                bundle.write(str(path), path.relative_to(self.output).as_posix())
        # ZIP checksum is intentionally outside the ZIP (no self-reference).
        record = stable_read(self.output, destination.name)
        (self.output / "g2_private_snapshot.zip.sha256.txt").write_text(
            record["sha256"] + "  " + destination.name + "\n", encoding="ascii")
        self.bundle = destination


def finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Existing KAIS workspace root")
    parser.add_argument("--audit-dir", required=True, help="Previous audit directory, absolute or relative to root")
    args = parser.parse_args()
    root_input = Path(args.root).expanduser()
    if root_input.is_symlink():
        parser.error("--root must not itself be a symbolic link")
    root = root_input.resolve(strict=False)
    if not root.is_dir():
        parser.error("--root is not a directory")
    audit_input = Path(args.audit_dir)
    if not audit_input.is_absolute():
        audit_input = root / audit_input
    try:
        audit_relative = audit_input.relative_to(root).as_posix()
        audit_dir = safe_path(root, audit_relative)
        if not audit_dir.is_dir():
            raise FreezeError("audit_dir_not_directory")
        rep = safe_path(root, "reproducibility", must_exist=False)
        rep.mkdir(exist_ok=True)
    except (ValueError, OSError, FreezeError) as exc:
        parser.error("Rejected audit/output path: " + str(exc))
    output = rep / ("g2_freeze_" + stamp() + "_" + uuid.uuid4().hex[:10])
    output.mkdir(exist_ok=False)
    print("OUTPUT_DIR=" + str(output), flush=True)
    freezer = Freezer(root, audit_dir, output)
    try:
        freezer.perform()
    except Exception as exc:
        # Do not echo file contents, package direct URLs, environment variables, or exception payloads.
        freezer.failures.append("unhandled_audit_error:" + type(exc).__name__)
    successful = freezer.write_reports()
    if successful:
        try:
            freezer.zip_reports()
        except Exception as exc:
            freezer.failures.append("bundle_creation_failed:" + type(exc).__name__)
            archive = output / "g2_private_snapshot.zip"
            if archive.is_file() and not archive.is_symlink():
                archive.unlink()
            freezer.write_reports()
            successful = False
    print((output / "SUMMARY.md").read_text(encoding="utf-8"))
    print("SUMMARY=" + str(output / "SUMMARY.md"))
    print("BUNDLE_ZIP=" + (str(freezer.bundle) if successful and freezer.bundle else "NOT_CREATED"))
    return 0 if successful else 2


if __name__ == "__main__":
    sys.exit(main())
