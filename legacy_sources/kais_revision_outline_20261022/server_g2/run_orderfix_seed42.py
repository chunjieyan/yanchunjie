#!/usr/bin/env python3
"""Run the two reviewed seed42 corrections sequentially, then collect reports.

Without --run, only validate files and print the plan. No research module is
imported by this driver. Actual training starts only with the --run command.
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
COLLECTOR_PATH = Path(__file__).with_name("collect_orderfix_results.py")
DATASETS = {"sports": "Amazon_Sports", "toys": "Amazon_Toys"}


def plain_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected an absolute path without '..': " + str(path))
    for component in reversed((path,) + tuple(path.parents)):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or (getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError("Symbolic link/reparse point rejected: " + str(component))
    return path


def checked_file(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or any(p in (".", "..") for p in part.parts) or ":" in relative or "\\" in relative:
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
        if (os.fstat(handle.fileno()).st_dev, os.fstat(handle.fileno()).st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("File changed while opening: " + str(path))
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError("File changed while hashing: " + str(path))
    return digest.hexdigest()


def validate(stage):
    stage = plain_path(stage)
    manifest_file = checked_file(stage, "manifests/orderfix_manifest.json")
    if sha256(manifest_file) != MANIFEST_SHA256:
        raise RuntimeError("The manifest differs from the reviewed smoke package")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    root = plain_path(Path(manifest["root"]))
    if str(stage) != manifest["stage"] or stage.parent != root:
        raise RuntimeError("Use the original reviewed stage, in its original project")
    expected_scripts = {"scripts/model_definition.py", "scripts/train_sports_orderfix.py", "scripts/train_toys_orderfix.py"}
    if set(manifest["generated_sha256"]) != expected_scripts:
        raise RuntimeError("Unexpected generated script set")
    for relative, expected in manifest["generated_sha256"].items():
        if sha256(checked_file(stage, relative)) != expected:
            raise RuntimeError("Generated script changed: " + relative)
    for relative, expected in manifest["data_sha256"].items():
        if sha256(checked_file(root, relative)) != expected:
            raise RuntimeError("Frozen input changed: " + relative)
    status = json.loads(checked_file(stage, "verification/smoke_status.json").read_text(encoding="utf-8"))
    if status != {"sports": 0, "toys": 0}:
        raise RuntimeError("Both reviewed smoke checks must be successful")
    for dataset in DATASETS:
        log = checked_file(stage, "verification/" + dataset + "_seed42_smoke.log").read_text(encoding="utf-8")
        for marker in ("INTEGRITY PASS:", "ORDERFIX PASS:", "STRICT DATA AUDIT PASS", "SMOKE TEST PASS", "No training started."):
            if marker not in log:
                raise RuntimeError("Incomplete smoke log: " + dataset)
        output = stage / "outputs" / (DATASETS[dataset].lower() + "_control_orderfix_seed42")
        if output.exists() or output.is_symlink():
            raise RuntimeError("Existing formal output; will not overwrite or resume: " + str(output))
    plain_path(COLLECTOR_PATH)
    if not COLLECTOR_PATH.is_file():
        raise ValueError("Upload collect_orderfix_results.py next to this driver")
    return manifest


def execute(stage):
    validate(stage)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    verification = plain_path(stage / "verification")
    status_file = verification / ("seed42_driver_" + stamp + ".json")
    state = {"stage": str(stage), "python": sys.executable, "seed": 42, "runs": {}, "status": "RUNNING"}
    with status_file.open("x", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    result = 0
    process = None
    try:
        for dataset in DATASETS:
            log = verification / (dataset + "_seed42_train_" + stamp + ".log")
            command = [sys.executable, "-u", str(stage / "scripts" / ("train_" + dataset + "_orderfix.py")), "--mode", "train", "--seed", "42"]
            record = {"command": command, "log": str(log), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_code": None}
            state["runs"][dataset] = record
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
        state["status"] = "BOTH_PROCESSES_EXITED_ZERO" if result == 0 and len(state["runs"]) == 2 else "STOPPED_INCOMPLETE"
        state["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("TRAINING_DRIVER_STATUS=" + state["status"], flush=True)
        collected = subprocess.run([sys.executable, str(COLLECTOR_PATH), "--stage", str(stage)], check=False)
        if collected.returncode != 0:
            print("COLLECTION_EXIT_CODE=" + str(collected.returncode), file=sys.stderr)
            result = result or collected.returncode
    return 0 if result == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--run", action="store_true", help="Start the two full seed42 training processes")
    args = parser.parse_args()
    if not args.run:
        validate(args.stage)
        print("VALIDATED_ONLY_NO_TRAINING; add --run to start Sports then Toys")
        return 0
    return execute(args.stage)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(2)
