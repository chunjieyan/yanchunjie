#!/usr/bin/env python3
"""Validate the reviewed seed42 stage, then optionally run seeds 2026 and 3407.

Standard library only. No training starts without --run. Sports and Toys are
run sequentially for each seed; existing output directories are never reused.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import subprocess
import sys
import uuid

MANIFEST_SHA256 = "de24137de46bed56807a3a66c6e6420931e0c2d2cb61de0bd6f20465aea6d81d"
COLLECTOR_PATH = Path(__file__).with_name("collect_orderfix_multiseed.py")
PLAN = (("sports", 2026), ("toys", 2026), ("sports", 3407), ("toys", 3407))
BASELINE_SHA256 = {
    "outputs/amazon_sports_control_orderfix_seed42/config.json": "95963620e83b1e06dcda21f947394b2adad4b0c7259b3f7b4f99a759d9493989",
    "outputs/amazon_sports_control_orderfix_seed42/metrics.json": "ba3afef8d2451b67d6b4cd3d7cfde2479129288cdfb4bf305b545d3780e67d5b",
    "outputs/amazon_sports_control_orderfix_seed42/training_log.csv": "798891b74b0df856d8fc5122570a514a3c3819bfa62d3e1f6fb6c676b6284581",
    "outputs/amazon_sports_control_orderfix_seed42/test_per_sample_predictions.csv": "9ef7554252ff843a1b5bf7e3809aba4827c96427aee66744f06de70852119e5e",
    "outputs/amazon_sports_control_orderfix_seed42/item_id_map.csv": "8f0e80ba0e996b19409cd72b3ed838242397f548cf38604ffd73d58677e29fee",
    "outputs/amazon_toys_control_orderfix_seed42/config.json": "18d9418f4eb6870929258deb4186a5ee160210d1bf57e55263b076d44a48b64e",
    "outputs/amazon_toys_control_orderfix_seed42/metrics.json": "2b7ce49edd8c7825f0b1d4f63db4895210e6b7339e640553c4b1924a0ab9b307",
    "outputs/amazon_toys_control_orderfix_seed42/training_log.csv": "057de30a3015899d2d1ebcfcabfc0731e2f05c864fd6dddfbc9ed97402c1f1ac",
    "outputs/amazon_toys_control_orderfix_seed42/test_per_sample_predictions.csv": "9b4b5854ff4c05a6b563236e2a6b436d83931f640bf435725294ee2e0fb89cd9",
    "outputs/amazon_toys_control_orderfix_seed42/item_id_map.csv": "d6e1135f71ad9749caa6db9743045e1e65860bd62df6f42e0b76642c08645c60",
}


def plain_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected absolute path without '..': " + str(path))
    for component in reversed((path,) + tuple(path.parents)):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symbolic link/reparse point rejected: " + str(component))
    return path


def checked_file(root, relative):
    part = PurePosixPath(relative)
    if not relative or part.is_absolute() or any(p in ("", ".", "..") for p in relative.split("/")) or ":" in relative or "\\" in relative:
        raise ValueError("Unsafe relative path: " + relative)
    path = plain_path(plain_path(root).joinpath(*part.parts))
    path.resolve(strict=False).relative_to(Path(root).resolve(strict=False))
    if not path.is_file():
        raise ValueError("Expected a regular file: " + str(path))
    return path


def sha256(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("File changed while opening: " + str(path))
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if signature(before) != signature(after):
        raise RuntimeError("File changed while hashing: " + str(path))
    return digest.hexdigest()


def validate(stage):
    stage = plain_path(stage)
    manifest_path = checked_file(stage, "manifests/orderfix_manifest.json")
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("Use the manifest from the reviewed seed42 stage")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = plain_path(Path(manifest["root"]))
    if str(stage) != manifest["stage"] or stage.parent != root:
        raise RuntimeError("Use the original reviewed stage in its original project")
    expected_scripts = {"scripts/model_definition.py", "scripts/train_sports_orderfix.py", "scripts/train_toys_orderfix.py"}
    if set(manifest["generated_sha256"]) != expected_scripts:
        raise RuntimeError("Unexpected generated script set")
    for relative, expected in manifest["generated_sha256"].items():
        if sha256(checked_file(stage, relative)) != expected:
            raise RuntimeError("Generated script changed: " + relative)
    for relative, expected in manifest["data_sha256"].items():
        if sha256(checked_file(root, relative)) != expected:
            raise RuntimeError("Frozen input changed: " + relative)
    for relative, expected in BASELINE_SHA256.items():
        if sha256(checked_file(stage, relative)) != expected:
            raise RuntimeError("Reviewed seed42 result changed: " + relative)
    for dataset, seed in PLAN:
        output = stage / "outputs" / ("amazon_" + dataset + "_control_orderfix_seed" + str(seed))
        if output.exists() or output.is_symlink():
            raise RuntimeError("Existing output; will not overwrite or resume: " + str(output))
    if not plain_path(COLLECTOR_PATH).is_file():
        raise ValueError("Upload collect_orderfix_multiseed.py next to this driver")


def execute(stage):
    validate(stage)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    verification = plain_path(stage / "verification")
    status_file = verification / ("remaining_seeds_driver_" + stamp + ".json")
    state = {"stage": str(stage), "python": sys.executable, "seeds": [2026, 3407], "runs": {}, "status": "RUNNING"}
    with status_file.open("x", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    result, process = 0, None
    try:
        for dataset, seed in PLAN:
            label = dataset + "_seed" + str(seed)
            log = verification / (label + "_train_" + stamp + ".log")
            command = [sys.executable, "-u", str(stage / "scripts" / ("train_" + dataset + "_orderfix.py")), "--mode", "train", "--seed", str(seed)]
            record = {"command": command, "log": str(log), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_code": None}
            state["runs"][label] = record
            status_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            print("RUN=" + shlex.join(command), flush=True)
            with log.open("x", encoding="utf-8", newline="\n") as handle:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(line, end="", flush=True)
                result = process.wait()
            record["exit_code"] = result
            record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            process = None
            if result != 0:
                break
    except BaseException as exc:
        result = 1
        state["error"] = type(exc).__name__ + ": " + str(exc)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print("DRIVER_ERROR=" + state["error"], file=sys.stderr, flush=True)
    finally:
        state["status"] = "ALL_FOUR_PROCESSES_EXITED_ZERO" if result == 0 and len(state["runs"]) == 4 else "STOPPED_INCOMPLETE"
        state["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print("TRAINING_DRIVER_STATUS=" + state["status"], flush=True)
        collected = subprocess.run([sys.executable, str(COLLECTOR_PATH), "--stage", str(stage)], check=False)
        if collected.returncode:
            print("COLLECTION_EXIT_CODE=" + str(collected.returncode), file=sys.stderr)
            result = result or collected.returncode
    return 0 if result == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--run", action="store_true", help="Start the four remaining full training processes")
    args = parser.parse_args()
    if args.run:
        return execute(args.stage)
    validate(args.stage)
    print("VALIDATED_ONLY_NO_TRAINING; add --run for Sports2026, Toys2026, Sports3407, Toys3407")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(2)
