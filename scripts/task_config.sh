# Task / dataset defaults for branch.sh (seg vs cls).
# Sourced by scripts/branch.sh — do not execute directly.

resolve_brt_task() {
  local dataset="$1"
  local picked="${2:-}"

  if [[ -n "${BRT_TASK:-}" ]]; then
    TASK="${BRT_TASK}"
  elif [[ -n "${picked}" ]]; then
    TASK="${picked}"
  else
    TASK="seg"
  fi

  case "${TASK}" in
    seg|cls) ;;
    *)
      echo "[task_config] ERROR: 未知任务 '${TASK}'（允许: seg, cls）。" >&2
      exit 1
      ;;
  esac

  if [[ "${dataset}" == "360" && "${TASK}" == "cls" ]]; then
    echo "[task_config] ERROR: Fusion360 仅支持 seg。" >&2
    exit 1
  fi
}

configure_dataset_base() {
  case "$1" in
    360)
      DATASET_ID="360"
      DATASET_DIR="${DATASET_DIR:-/data/hdd/datasets/s2.0.0/processed/brt}"
      SPLIT_SOURCE_JSON="${SPLIT_SOURCE_JSON:-/data/hdd/datasets/s2.0.0/processed/dataset.json}"
      STEP_ROOT="${STEP_ROOT:-/data/hdd/datasets/s2.0.0/breps/step}"
      NUM_CLASSES="${NUM_CLASSES:-8}"
      NUM_CONTROL_PTS="${NUM_CONTROL_PTS:-28}"
      PREPROCESS_SCRIPT="${REPO_DIR}/scripts/preprocess_360.sh"
      BATCH_SIZE="${BATCH_SIZE:-16}"
      ;;
    mechcad)
      DATASET_ID="mechcad"
      DATASET_DIR="${DATASET_DIR:-/data/hdd/datasets/mechcad/processed}"
      SPLIT_SOURCE_JSON="${SPLIT_SOURCE_JSON:-}"
      STEP_ROOT="${STEP_ROOT:-/data/hdd/datasets/mechcad/mechcad}"
      NUM_CLASSES="${NUM_CLASSES:-10}"
      NUM_CONTROL_PTS="${NUM_CONTROL_PTS:-28}"
      PREPROCESS_SCRIPT="${REPO_DIR}/scripts/preprocess_tri.sh"
      BATCH_SIZE="${BATCH_SIZE:-4}"
      ;;
    *)
      echo "[task_config] ERROR: unknown dataset $1" >&2
      exit 1
      ;;
  esac
}

apply_task_layout() {
  case "${DATASET_ID}:${TASK}" in
    360:seg)
      RESULTS_DATASET_NAME="fusion360_seg"
      TRAIN_ENTRY="segmentation.py"
      TASK_LABEL="per-face segmentation"
      ;;
    mechcad:seg)
      RESULTS_DATASET_NAME="mechcad_seg"
      TRAIN_ENTRY="segmentation.py"
      TASK_LABEL="part-class via per-face seg (label broadcast; acc primary)"
      ;;
    mechcad:cls)
      RESULTS_DATASET_NAME="mechcad_cls"
      TRAIN_ENTRY="classification.py"
      TASK_LABEL="part classification (native; resume/test WIP)"
      ;;
    *)
      echo "[task_config] ERROR: 未配置 ${DATASET_ID}:${TASK}" >&2
      exit 1
      ;;
  esac
}

task_supports_mode() {
  local mode="$1"
  if [[ "${TASK}" == "cls" && "${mode}" != "train" ]]; then
    echo "[branch.sh] ERROR: mechcad classification 的 ${mode} 尚未接入 branch.sh（请先用 seg，或手动 classification.py test）。" >&2
    exit 1
  fi
}
