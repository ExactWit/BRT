#!/bin/bash
set -eo pipefail

# 从已有 checkpoint 续训 Fusion360 分割，并追加到同一 TensorBoard 目录
#
# Usage:
#   bash scripts/resume_train_360.sh
#   MAX_EPOCHS=500 bash scripts/resume_train_360.sh

REPO_DIR="${HOME}/workspace/repo/BRT"
PROCESSED_DIR="${PROCESSED_DIR:-/data/hdd/datasets/s2.0.0/processed/brt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fusion360_seg}"
LOG_NAME="${LOG_NAME:-0629}"
LOG_VERSION="${LOG_VERSION:-193848}"
RUN_DIR="${REPO_DIR}/results/${EXPERIMENT_NAME}/${LOG_NAME}/${LOG_VERSION}"
RESUME_CKPT="${RESUME_CKPT:-${RUN_DIR}/last.ckpt}"

NUM_CLASSES="${NUM_CLASSES:-8}"
NUM_CONTROL_PTS="${NUM_CONTROL_PTS:-28}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU="${GPU:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "[resume_train_360] ERROR: checkpoint not found: ${RESUME_CKPT}"
  exit 1
fi

BACKUP_DIR="${RUN_DIR}.backup_$(date +%Y%m%d_%H%M%S)"
echo "[resume_train_360] backup tensorboard/checkpoints -> ${BACKUP_DIR}"
cp -a "${RUN_DIR}" "${BACKUP_DIR}"

source "${HOME}/software/miniconda3/etc/profile.d/conda.sh"
conda activate brt
cd "${REPO_DIR}"
mkdir -p logs

echo "[resume_train_360] resume from : ${RESUME_CKPT}"
echo "[resume_train_360] append logs to: ${RUN_DIR}"

python segmentation.py train \
  --num_classes "${NUM_CLASSES}" \
  --dataset_dir "${PROCESSED_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --gpu "${GPU}" \
  --num_control_pts "${NUM_CONTROL_PTS}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --log_name "${LOG_NAME}" \
  --log_version "${LOG_VERSION}" \
  --max_epochs "${MAX_EPOCHS}" \
  --resume_from "${RESUME_CKPT}" \
  2>&1 | tee -a logs/0628train_resume.log

echo "[resume_train_360] Done."
echo "  tensorboard --logdir ${RUN_DIR}"
