#!/usr/bin/env bash
# Retry triangle extraction for missing bins; exclude skip-log timeouts; record structured failures.
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROCESSED_DIR="${PROCESSED_DIR:-/data/hdd/datasets/mechcad/processed}"
PROCESS_NUM="${PROCESS_NUM:-8}"

source "${HOME}/software/miniconda3/etc/profile.d/conda.sh"
conda activate brt
export PYTHONNOUSERSITE=1

cd "${REPO_DIR}/process"

python retry_mechcad_triangles.py \
  --processed-dir "${PROCESSED_DIR}" \
  --process-num "${PROCESS_NUM}" \
  --run-split \
  2>&1 | tee "${PROCESSED_DIR}/triangles_retry.log"

python summarize_mechcad_failures.py \
  --failure-log "${PROCESSED_DIR}/triangles_failure.jsonl" \
  --skip-log "${PROCESSED_DIR}/triangles_skip.log" \
  --report-out "${PROCESSED_DIR}/triangles_failure_report.json" \
  2>&1 | tee "${PROCESSED_DIR}/triangles_failure_summary.log"
