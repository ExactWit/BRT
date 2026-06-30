#!/bin/bash
set -eo pipefail

# Fusion 360 Gallery Segmentation (s2.0.0) — BRT training
#
# 注意：BRT 不能直接使用 BRepNet 的 processed/*.npz，需先跑 preprocess_360.sh
# 生成 triangles/topology 的 .bin 与 datasplit.json。
#
# Usage:
#   bash scripts/train_360.sh
#   SKIP_PREPROCESS=1 bash scripts/train_360.sh          # 已有 BRT 预处理结果时
#   GPU=1 BATCH_SIZE=8 bash scripts/train_360.sh

DATASET_DIR="${DATASET_DIR:-/data/hdd/datasets/s2.0.0}"
PROCESSED_DIR="${PROCESSED_DIR:-${DATASET_DIR}/processed/brt}"
REPO_DIR="${HOME}/workspace/repo/BRT"

NUM_CLASSES="${NUM_CLASSES:-8}"       # segment_names.json 共 8 类
NUM_CONTROL_PTS="${NUM_CONTROL_PTS:-28}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU="${GPU:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fusion360_seg}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

source "${HOME}/software/miniconda3/etc/profile.d/conda.sh"
conda activate brt

if [[ "${SKIP_PREPROCESS}" != "1" ]]; then
  if [[ ! -f "${PROCESSED_DIR}/datasplit.json" ]]; then
    echo "[train_360] 未找到 ${PROCESSED_DIR}/datasplit.json，开始 BRT 预处理..."
    PROCESSED_DIR="${PROCESSED_DIR}" bash "${REPO_DIR}/scripts/preprocess_360.sh"
  else
    echo "[train_360] 使用已有预处理: ${PROCESSED_DIR}/datasplit.json"
  fi
fi

if [[ ! -f "${PROCESSED_DIR}/datasplit.json" ]]; then
  echo "[train_360] ERROR: 缺少 ${PROCESSED_DIR}/datasplit.json"
  echo "  请先运行: bash scripts/preprocess_360.sh"
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p logs

echo "[train_360] dataset_dir : ${PROCESSED_DIR}"
echo "[train_360] num_classes : ${NUM_CLASSES}"
echo "[train_360] batch_size  : ${BATCH_SIZE}"
echo "[train_360] gpu         : ${GPU}"

python segmentation.py train \
  --num_classes "${NUM_CLASSES}" \
  --dataset_dir "${PROCESSED_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --gpu "${GPU}" \
  --num_control_pts "${NUM_CONTROL_PTS}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  2>&1 | tee "${PROCESSED_DIR}/train.log"

echo "[train_360] Done. Logs: results/${EXPERIMENT_NAME}/"
echo "  tensorboard --logdir results/${EXPERIMENT_NAME}"
