#!/usr/bin/env python3
"""Prepare Beauty seed copies, smoke-check, train 2026/3407, and collect evidence.

No action starts without --run. Uses only standard-library orchestration;
the separately reviewed trainer subprocesses perform actual GPU operations.
"""
import argparse
import datetime
import importlib.util
import json
import math
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import uuid

PREPARER_PATH = Path(__file__).with_name("prepare_beauty_control_seeds.py")
COLLECTOR_PATH = Path(__file__).with_name("collect_beauty_control_seeds.py")
STAGE_PREFIX = "kais_beauty_control_seeds_v1_"
PLAN = (("smoke", 42), ("train", 2026), ("train", 3407))


def plain_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected absolute path without '..': " + str(path))
    for component in (path,) + tuple(path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symbolic link/reparse point rejected: " + str(component))
    return path.resolve(strict=False)


def load_preparer():
    path = plain_path(PREPARER_PATH)
    if not path.is_file():
        raise ValueError("Upload prepare_beauty_control_seeds.py next to this driver")
    spec = importlib.util.spec_from_file_location("beauty_control_reviewed_preparer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def execute(root):
    root = plain_path(root)
    if not root.is_dir():
        raise ValueError("Project root must be a directory")
    if not plain_path(COLLECTOR_PATH).is_file():
        raise ValueError("Upload collect_beauty_control_seeds.py next to this driver")
    previous = [p for p in root.glob(STAGE_PREFIX + "*") if p.is_dir() or p.is_symlink()]
    if previous:
        raise RuntimeError("An existing Beauty preparation needs inspection/collection before another start: " + str(previous[0]))
    for seed in (2026, 3407):
        known = root / "step38_kais_retrain/outputs" / ("beauty_sasrec_control_ce_bs128_lr1em3_max200_seed" + str(seed))
        if known.exists() or known.is_symlink():
            raise RuntimeError("An original-path Beauty seed result already exists; inspect it first: " + str(known))
    preparation = load_preparer()
    stage = plain_path(preparation.prepare(root))
    manifest = json.loads((stage / "manifests/beauty_seed_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != "beauty_control_seeds_v1" or manifest.get("root") != str(root) or manifest.get("stage") != str(stage):
        raise RuntimeError("Unexpected preparation manifest identity")
    scripts = {"scripts/model_definition.py", "scripts/train_beauty_control.py"}
    if set(manifest["generated_sha256"]) != scripts:
        raise RuntimeError("Unexpected prepared script set")
    for relative, expected in manifest["generated_sha256"].items():
        preparation.checked_sha256(stage, relative, expected)
    verification = plain_path(stage / "verification")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    status_file = verification / ("beauty_driver_" + stamp + ".json")
    state = {"stage": str(stage), "python": sys.executable, "runs": {}, "status": "RUNNING", "formal_seeds": [2026, 3407], "reference_seed42_retrained": False}
    with status_file.open("x", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    result, process = 0, None
    try:
        for mode, seed in PLAN:
            label = "beauty_seed" + str(seed) + "_" + mode
            logfile = verification / (label + "_" + stamp + ".log")
            command = [sys.executable, "-u", str(stage / "scripts/train_beauty_control.py"), "--mode", mode, "--seed", str(seed)]
            record = {"command": command, "log": str(logfile), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_code": None}
            state["runs"][label] = record
            status_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            print("RUN=" + shlex.join(command), flush=True)
            smoke = None
            with logfile.open("x", encoding="utf-8", newline="\n") as handle:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(line, end="", flush=True)
                    if mode == "smoke" and line.startswith("BEAUTY_SMOKE_PASS="):
                        smoke = json.loads(line.split("=", 1)[1])
                result = process.wait()
            record["exit_code"] = result
            record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            process = None
            if result != 0:
                break
            if mode == "smoke":
                if not isinstance(smoke, dict) or smoke.get("seed") != 42 or smoke.get("saved_training_output") is not False:
                    raise RuntimeError("Smoke ended without its expected success record")
                if not math.isfinite(float(smoke.get("loss", "nan"))) or not (0 < smoke.get("validation_samples", 0) <= 32):
                    raise RuntimeError("Invalid smoke loss or validation sample count")
                if (stage / "outputs/amazon_beauty_control_seed42").exists():
                    raise RuntimeError("Smoke unexpectedly created a formal seed42 output")
                record["smoke_result"] = smoke
                print("BEAUTY_SMOKE_VERIFIED_BEFORE_FULL_TRAINING", flush=True)
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
        state["status"] = "SMOKE_AND_TWO_TRAININGS_EXITED_ZERO" if result == 0 and len(state["runs"]) == 3 else "STOPPED_INCOMPLETE"
        state["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print("TRAINING_DRIVER_STATUS=" + state["status"], flush=True)
        collected = subprocess.run([sys.executable, str(COLLECTOR_PATH), "--stage", str(stage)], check=False)
        if collected.returncode:
            result = result or collected.returncode
            print("COLLECTION_EXIT_CODE=" + str(collected.returncode), file=sys.stderr)
    return 0 if result == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run", action="store_true", help="Prepare, smoke-check and train the two missing Beauty seeds")
    args = parser.parse_args()
    if not args.run:
        plain_path(args.root)
        print("PLAN_ONLY_NO_FILES_OR_GPU: prepare; Beauty42 smoke only; Beauty2026 train; Beauty3407 train; collect")
        return 0
    return execute(args.root)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError, SyntaxError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(2)
