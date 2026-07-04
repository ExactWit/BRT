#!/usr/bin/env bash
# Build datasplit from existing mechcad bins and train baseline segmentation.
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROCESSED_DIR="${PROCESSED_DIR:-/data/hdd/datasets/mechcad/processed}"
LOG_DIR="${REPO_DIR}/logs"

source "${HOME}/software/miniconda3/etc/profile.d/conda.sh"
conda activate brt
export PYTHONNOUSERSITE=1
cd "${REPO_DIR}"

mkdir -p "${LOG_DIR}"

echo "[mechcad] stopping stuck preprocess workers (if any)..."
pkill -f 'gen_tmcad_triangles.py.*mechcad' 2>/dev/null || true
pkill -f 'solid_to_triangles2.*mechcad' 2>/dev/null || true
sleep 1

echo "[mechcad] building datasplit from available triangle+topology pairs..."
python process/split_mechcad_available.py \
  --processed-dir "${PROCESSED_DIR}" \
  2>&1 | tee "${LOG_DIR}/mechcad_split_$(date +%m%d_%H%M%S).log"

echo "[mechcad] datasplit summary:"
python3 - <<PY
import json, pathlib
p = pathlib.Path("${PROCESSED_DIR}") / "datasplit_available_summary.json"
print(json.dumps(json.load(open(p)), indent=2, ensure_ascii=False))
PY

LOG_DATE="$(date +%m%d)"
RUN_TAG="${RUN_TAG:-baseline}"
NUM_CLASSES="${NUM_CLASSES:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU="${GPU:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
EXPERIMENT_NOTE="${EXPERIMENT_NOTE:-mechcad available-pairs baseline; screw-only triangles for now}"

echo "[mechcad] train baseline segmentation (num_classes=${NUM_CLASSES})..."
python segmentation.py train \
  --num_classes "${NUM_CLASSES}" \
  --dataset_dir "${PROCESSED_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --gpu "${GPU}" \
  --num_control_pts 28 \
  --experiment_name mechcad_seg \
  --log_name "${LOG_DATE}" \
  --run_tag "${RUN_TAG}" \
  --max_epochs "${MAX_EPOCHS}" \
  --git_branch "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)" \
  --dataset_id mechcad \
  --model_id baseline \
  --model_label "BRT baseline (TopoEncoder)" \
  --model_commit "$(git rev-parse --short HEAD)" \
  --model_commit_full "$(git rev-parse HEAD)" \
  --model_status active \
  --experiment_note "${EXPERIMENT_NOTE}" \
  2>&1 | tee "${LOG_DIR}/train_mechcad_baseline_${LOG_DATE}_$(date +%H%M%S).log"
