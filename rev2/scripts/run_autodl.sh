#!/usr/bin/env bash
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
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.run_suite \
  --stage "${STAGE}" \
  --workers "${WORKERS}" \
  --data-root "${DATA_ROOT}" \
  --project-root "${ROOT_DIR}" \
  --runs-root "${RUNS_ROOT}"

echo "=== 3. Unified Single-GPU Cost Benchmark ==="
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.benchmark_costs \
  --data-root "${DATA_ROOT}" \
  --project-root "${ROOT_DIR}" \
  --runs-root "${RUNS_ROOT}"

echo "=== 4. Aggregating Results & Generating LaTeX Tables ==="
PYTHONPATH="${ROOT_DIR}" python3 -u -m src.aggregate_results \
  --runs-root "${RUNS_ROOT}"

echo "=== ALL DONE! Results & LaTeX tables are in ${RUNS_ROOT}/aggregated ==="
