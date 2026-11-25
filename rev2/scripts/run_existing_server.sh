#!/usr/bin/env bash
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
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.eval_legacy_checkpoints \
  --data-root "${DATA_ROOT}" \
  --project-root "${ROOT_DIR}" \
  --runs-root "${RUNS_ROOT}"

STAGE="${1:-all}"
WORKERS="${WORKERS:-3}"
echo "=== 2. Launching New Training Suite [${STAGE}] (--reuse-legacy-server, workers=${WORKERS}, resume-safe) ==="
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.run_suite \
  --stage "${STAGE}" \
  --workers "${WORKERS}" \
  --reuse-legacy-server \
  --data-root "${DATA_ROOT}" \
  --project-root "${ROOT_DIR}" \
  --runs-root "${RUNS_ROOT}"

echo "=== 3. Running Unified Single-GPU Cost Benchmark (Legacy + New Seed-42 Models) ==="
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.benchmark_costs \
  --data-root "${DATA_ROOT}" \
  --project-root "${ROOT_DIR}" \
  --runs-root "${RUNS_ROOT}"

echo "=== 4. Aggregating All 216 Models (42 Legacy + 174 New) & Generating LaTeX Tables ==="
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.aggregate_results \
  --runs-root "${RUNS_ROOT}"

echo "=== ALL DONE! Results & LaTeX tables are in ${RUNS_ROOT}/aggregated ==="
