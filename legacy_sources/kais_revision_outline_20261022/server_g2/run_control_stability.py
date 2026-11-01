#!/usr/bin/env python3
"""Prepare and evaluate the 12 reviewed Control checkpoints; no training.

Without --run this prints a plan only. Actual inference starts only with --run.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve(strict=False).parent
PREPARER_PATH = HERE / "prepare_control_stability.py"
COLLECTOR_PATH = HERE / "collect_control_stability.py"
INPUTS_PATH = HERE / "control_stability_inputs_20261023.json"
INPUTS_SHA256 = "8e8d13ded1ded7729e7f951b97d65ef3bda84614f4f4a9ad881bec68f7c81458"
STAGE_PREFIX = "kais_control_stability_v1_"
PLAN = tuple((tag, seed) for tag in ("ml1m", "beauty", "sports", "toys") for seed in (42, 2026, 3407))


def plain(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected an absolute path without '..': " + str(path))
    for component in (path,) + tuple(path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symbolic link/reparse point rejected: " + str(component))
    return path.resolve(strict=False)


def sha(path):
    return hashlib.sha256(plain(path).read_bytes()).hexdigest()


def load_preparer():
    path = plain(PREPARER_PATH)
    spec = importlib.util.spec_from_file_location("reviewed_control_stability_preparer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def execute(root):
    root = plain(root)
    if not root.is_dir():
        raise ValueError("Project root must be a directory")
    if not plain(COLLECTOR_PATH).is_file():
        raise ValueError("Upload collect_control_stability.py next to this driver")
    if sha(INPUTS_PATH) != INPUTS_SHA256:
        raise RuntimeError("The reviewed Control input manifest changed")
    inputs = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    previous = [p for p in root.glob(STAGE_PREFIX + "*") if p.is_dir() or p.is_symlink()]
    if previous:
        raise RuntimeError("An earlier stability stage needs inspection/collection before another start: " + str(previous[0]))
    preparation = load_preparer()
    stage = plain(preparation.prepare(root, INPUTS_PATH))
    manifest = json.loads((stage / "manifests/control_stability_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("version") != "control_stability_v1" or manifest.get("root") != str(root)
            or manifest.get("stage") != str(stage) or manifest.get("inputs") != inputs
            or manifest.get("inputs_sha256") != INPUTS_SHA256):
        raise RuntimeError("Unexpected prepared stage identity")
    scripts = {"scripts/model_definition.py"} | {"scripts/eval_" + tag + "_stability.py" for tag in ("ml1m", "beauty", "sports", "toys")}
    if set(manifest["generated_sha256"]) != scripts:
        raise RuntimeError("Unexpected generated evaluation script set")
    for relative, expected in manifest["generated_sha256"].items():
        if sha(stage / relative) != expected:
            raise RuntimeError("Generated file hash mismatch: " + relative)
    verification = plain(stage / "verification")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    status_file = verification / ("control_stability_driver_" + stamp + ".json")
    state = {"stage": str(stage), "python": sys.executable, "status": "RUNNING", "runs": {}, "training_performed": False}
    with status_file.open("x", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    result, process = 0, None
    try:
        for tag, seed in PLAN:
            label = tag + "_seed" + str(seed)
            script = stage / "scripts" / ("eval_" + tag + "_stability.py")
            logfile = verification / (label + "_eval_" + stamp + ".log")
            command = [sys.executable, "-u", str(script), "--seed", str(seed)]
            record = {"command": command, "log": str(logfile), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_code": None}
            state["runs"][label] = record
            status_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            print("EVAL=" + shlex.join(command), flush=True)
            with logfile.open("x", encoding="utf-8", newline="\n") as handle:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(line, end="", flush=True)
                result = process.wait()
            record["exit_code"] = result
            record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            process = None
            if result:
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
        state["status"] = "ALL_TWELVE_EVALUATIONS_EXITED_ZERO" if result == 0 and len(state["runs"]) == 12 else "STOPPED_INCOMPLETE"
        state["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print("EVALUATION_DRIVER_STATUS=" + state["status"], flush=True)
        collected = subprocess.run([sys.executable, str(COLLECTOR_PATH), "--stage", str(stage)], check=False)
        if collected.returncode:
            result = result or collected.returncode
            print("COLLECTION_EXIT_CODE=" + str(collected.returncode), file=sys.stderr)
    return 0 if result == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run", action="store_true", help="Prepare copies and evaluate 12 fixed checkpoints, with no training")
    args = parser.parse_args()
    if not args.run:
        plain(args.root)
        print("PLAN_ONLY_NO_FILES_OR_GPU: 4 datasets x 3 trained Control seeds x (original + 5 last-bucket permutations); no training")
        return 0
    return execute(args.root)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError, SyntaxError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(2)
