#!/bin/bash
set -eo pipefail

DATASET_DIR="${DATASET_DIR:-/data/hdd/datasets/s2.0.0}"
STEP_DIR="${STEP_DIR:-${DATASET_DIR}/breps/step}"
SEG_DIR="${SEG_DIR:-${DATASET_DIR}/breps/seg}"
PROCESSED_DIR="${PROCESSED_DIR:-${DATASET_DIR}/processed/brt}"
# 与 BRepNet 对齐：使用 processed/dataset.json 的 training/validation/test 划分
SPLIT_JSON="${SPLIT_JSON:-${DATASET_DIR}/processed/dataset.json}"
PROCESS_NUM="${PROCESS_NUM:-30}"
# 仅重新生成 datasplit.json（已有 triangles/topology 时设为 1）
SPLIT_ONLY="${SPLIT_ONLY:-0}"
REPO_DIR="${HOME}/workspace/repo/BRT"

source "${HOME}/software/miniconda3/etc/profile.d/conda.sh"
conda activate brt

for path in "$SEG_DIR" "$SPLIT_JSON"; do
  if [[ ! -e "$path" ]]; then
    echo "[preprocess_360] ERROR: missing $path"
    exit 1
  fi
done

cd "${REPO_DIR}/process"
mkdir -p logs "${PROCESSED_DIR}"

if [[ "${SPLIT_ONLY}" != "1" ]]; then
  if [[ ! -d "$STEP_DIR" ]]; then
    echo "[preprocess_360] ERROR: missing STEP dir $STEP_DIR"
    exit 1
  fi

  echo "[preprocess_360] STEP       -> ${STEP_DIR}"
  echo "[preprocess_360] OUT        -> ${PROCESSED_DIR}"
  echo "[preprocess_360] workers    : ${PROCESS_NUM}"

  python gen_fusion360_topo.py "${STEP_DIR}" "${PROCESSED_DIR}/topology" \
    --process_num "${PROCESS_NUM}" \
    2>&1 | tee "${PROCESSED_DIR}/topology.log"

  python gen_fusion360_triangles.py "${STEP_DIR}" "${PROCESSED_DIR}" \
    --process_num "${PROCESS_NUM}" \
    2>&1 | tee "${PROCESSED_DIR}/triangles.log"
else
  echo "[preprocess_360] SPLIT_ONLY=1: skip topology/triangles, regenerate datasplit only"
fi

echo "[preprocess_360] split file -> ${SPLIT_JSON}"

python split_fusion360_dataset.py \
  "${PROCESSED_DIR}" \
  "${SEG_DIR}" \
  "${SPLIT_JSON}" \
  "${PROCESSED_DIR}/datasplit.json" \
  2>&1 | tee "${PROCESSED_DIR}/datasplit.log"

echo "[preprocess_360] Done."
echo "  datasplit.json : ${PROCESSED_DIR}/datasplit.json"
echo "Next: bash scripts/train_360.sh"
