"""Package code & manifests for AutoDL RTX 4090.

Supports two modes:
- ``--code-only`` (~35 KB): For your EXISTING AutoDL server where
  ``/root/autodl-tmp/time_uncertainty_rec`` already holds the frozen datasets
  AND all 42 original .pt checkpoints! Includes ``run_existing_server.sh``
  which runs Stage 0 (zero-training All-Bucket eval of the 42 legacy models)
  + 174 new models (skipping the 30 already-fitted checkpoints) + profiling + aggregation.
- Default full bundle (~56 MB): Self-contained bundle including the 36 frozen CSVs
  for a brand-new server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile

AUTODL_EXISTING_SERVER_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

# AutoDL RTX 4090 one-click launcher for EXISTING server (/root/autodl-tmp/time_uncertainty_rec)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/time_uncertainty_rec}"
RUNS_ROOT="${ROOT_DIR}/runs"

echo "=== Environment & Existing Server Check ==="
python3 -c "import torch, numpy; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
test -d "${DATA_ROOT}/processed" || { echo "ERROR: ${DATA_ROOT}/processed not found!"; exit 1; }

echo "=== 0. Contract & Bit-Exact Invariance Smoke Test ==="
PYTHONPATH="${ROOT_DIR}" python3 "${ROOT_DIR}/tests/test_smoke_all_models.py" --data-root "${DATA_ROOT}" --skip-real-train

echo "=== 1. Stage 0: Zero-Training Evaluation of Existing 42 Legacy .pt Checkpoints (~2 min) ==="
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.eval_legacy_checkpoints \\
  --data-root "${DATA_ROOT}" \\
  --project-root "${ROOT_DIR}" \\
  --runs-root "${RUNS_ROOT}"

STAGE="${1:-all}"
WORKERS="${WORKERS:-3}"
echo "=== 2. Launching New Training Suite [${STAGE}] (--reuse-legacy-server, workers=${WORKERS}, resume-safe) ==="
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.run_suite \\
  --stage "${STAGE}" \\
  --workers "${WORKERS}" \\
  --reuse-legacy-server \\
  --data-root "${DATA_ROOT}" \\
  --project-root "${ROOT_DIR}" \\
  --runs-root "${RUNS_ROOT}"

echo "=== 3. Running Unified Single-GPU Cost Benchmark (Legacy + New Seed-42 Models) ==="
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.benchmark_costs \\
  --data-root "${DATA_ROOT}" \\
  --project-root "${ROOT_DIR}" \\
  --runs-root "${RUNS_ROOT}"

echo "=== 4. Aggregating All 216 Models (42 Legacy + 174 New) & Generating LaTeX Tables ==="
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.aggregate_results \\
  --runs-root "${RUNS_ROOT}"

echo "=== ALL DONE! Results & LaTeX tables are in ${RUNS_ROOT}/aggregated ==="
"""

AUTODL_FRESH_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT_DIR}/data_frozen"
RUNS_ROOT="${ROOT_DIR}/runs"

echo "=== Environment Check ==="
python3 -c "import torch, numpy; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "=== 1. Contract & Exact-Invariance Smoke Test ==="
PYTHONPATH="${ROOT_DIR}" python3 "${ROOT_DIR}/tests/test_smoke_all_models.py" --data-root "${DATA_ROOT}" --skip-real-train

STAGE="${1:-all}"
WORKERS="${WORKERS:-3}"
echo "=== 2. Launching Full Suite [${STAGE}] (204 models, workers=${WORKERS}, resume-safe) ==="
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.run_suite \\
  --stage "${STAGE}" \\
  --workers "${WORKERS}" \\
  --data-root "${DATA_ROOT}" \\
  --project-root "${ROOT_DIR}" \\
  --runs-root "${RUNS_ROOT}"

echo "=== 3. Unified Single-GPU Cost Benchmark ==="
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.benchmark_costs \\
  --data-root "${DATA_ROOT}" \\
  --project-root "${ROOT_DIR}" \\
  --runs-root "${RUNS_ROOT}"

echo "=== 4. Aggregating Results & Generating LaTeX Tables ==="
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.aggregate_results \\
  --runs-root "${RUNS_ROOT}"

echo "=== ALL DONE! Results & LaTeX tables are in ${RUNS_ROOT}/aggregated ==="
"""


def pack(data_root: Path, project_root: Path, output_tar: Path, code_only: bool = False) -> None:
    contracts = json.loads(
        (project_root / "manifests" / "dataset_contracts.json").read_text(encoding="utf-8")
    )

    (project_root / "run_existing_server.sh").write_bytes(
        AUTODL_EXISTING_SERVER_SCRIPT.replace("\r\n", "\n").encode("utf-8")
    )
    (project_root / "run_autodl.sh").write_bytes(
        AUTODL_FRESH_SCRIPT.replace("\r\n", "\n").encode("utf-8")
    )

    data_files = set()
    if not code_only:
        for tag, spec in contracts["datasets"].items():
            data_files.add(spec["event_source"])
            data_files.add(spec["train"])
            data_files.add(spec["val"])
            data_files.add(spec["test"])
            for rel in spec["perm_files"].values():
                data_files.add(rel)

    output_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_tar, "w:gz") as tar:
        for rel_dir in ("src", "manifests", "tests", "literature"):
            for f in sorted((project_root / rel_dir).rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    arc = Path("kais_rev2") / f.relative_to(project_root)
                    tar.add(f, arcname=arc.as_posix())
        for fname in ("requirements.txt", "REVISION_PLAN.md", "run_existing_server.sh", "run_autodl.sh"):
            f = project_root / fname
            if f.is_file():
                tar.add(f, arcname=f"kais_rev2/{fname}")

        for rel in sorted(data_files):
            src_f = data_root / rel
            if not src_f.is_file():
                raise FileNotFoundError(f"Missing frozen data file: {src_f}")
            tar.add(src_f, arcname=f"kais_rev2/data_frozen/{rel}")

    size_kb = output_tar.stat().st_size / 1024.0
    label = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
    print(f"Packed ({'code-only' if code_only else f'full+{len(data_files)} CSVs'}) -> {output_tar} ({label})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\time_uncertainty_rec"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    default_name = (
        "autodl_kais_rev2_code_only.tar.gz" if args.code_only else "autodl_kais_rev2_bundle.tar.gz"
    )
    out = args.output or (args.project_root / default_name)
    pack(args.data_root, args.project_root, out, code_only=args.code_only)


if __name__ == "__main__":
    main()
