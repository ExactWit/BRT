#!/usr/bin/env bash
# exp_launcher entry point — forwards to BRT train/test/preprocess scripts.
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${HOME}/software/miniconda3/etc/profile.d/conda.sh"

MODE="${1:?usage: $0 <mode> [--exp-dir ...]}"
shift

EXP_DIR_ARG=""
OUTPUT_DIR_ARG=""
DATA_DIR_ARG=""
DATASPLIT_ARG=""
DATASET_ARG=""
TASK_ARG=""
CHECKPOINT_ARG=""
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-dir) EXP_DIR_ARG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR_ARG="$2"; shift 2 ;;
    --data-dir) DATA_DIR_ARG="$2"; shift 2 ;;
    --datasplit) DATASPLIT_ARG="$2"; shift 2 ;;
    --dataset) DATASET_ARG="$2"; shift 2 ;;
    --task) TASK_ARG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT_ARG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "[run.sh] unknown option: $1" >&2; exit 1 ;;
    *) break ;;
  esac
done

activate_env() {
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate brt
  fi
  export PYTHONNOUSERSITE=1
  cd "${REPO_DIR}"
}

resolve_dataset() {
  case "${DATASET_ARG}" in
    fusion360|360|s2.0.0)
      BRT_DATASET_ID="360"
      NUM_CLASSES="${NUM_CLASSES:-8}"
      NUM_CONTROL_PTS="${NUM_CONTROL_PTS:-28}"
      DEFAULT_PROCESSED="/data/hdd/datasets/s2.0.0/processed/brt"
      SPLIT_SOURCE_JSON="${SPLIT_SOURCE_JSON:-/data/hdd/datasets/s2.0.0/processed/dataset.json}"
      STEP_ROOT="${STEP_ROOT:-/data/hdd/datasets/s2.0.0/breps/step}"
      PREPROCESS_SCRIPT="${REPO_DIR}/scripts/preprocess_360.sh"
      ;;
    mechcad|tmcad)
      BRT_DATASET_ID="mechcad"
      NUM_CLASSES="${NUM_CLASSES:-10}"
      NUM_CONTROL_PTS="${NUM_CONTROL_PTS:-28}"
      DEFAULT_PROCESSED="/data/hdd/datasets/mechcad/processed"
      SPLIT_SOURCE_JSON=""
      STEP_ROOT="${STEP_ROOT:-/data/hdd/datasets/mechcad/mechcad}"
      PREPROCESS_SCRIPT="${REPO_DIR}/scripts/preprocess_tri.sh"
      ;;
    *)
      echo "[run.sh] ERROR: unknown --dataset ${DATASET_ARG}" >&2
      exit 1
      ;;
  esac
  PROCESSED_DIR="${DATA_DIR_ARG:-${DEFAULT_PROCESSED}}"
}

require_exp_dir() {
  if [[ -z "${EXP_DIR_ARG}" ]]; then
    echo "[run.sh] ERROR: --exp-dir is required for mode ${MODE}" >&2
    exit 1
  fi
  mkdir -p "${EXP_DIR_ARG}/checkpoints" "${EXP_DIR_ARG}/tensorboard" "${EXP_DIR_ARG}/metrics"
}

default_checkpoint() {
  if [[ -n "${CHECKPOINT_ARG}" ]]; then
    printf '%s' "${CHECKPOINT_ARG}"
    return 0
  fi
  if [[ -f "${EXP_DIR_ARG}/checkpoints/best.ckpt" ]]; then
    printf '%s' "${EXP_DIR_ARG}/checkpoints/best.ckpt"
    return 0
  fi
  if [[ -f "${EXP_DIR_ARG}/checkpoints/last.ckpt" ]]; then
    printf '%s' "${EXP_DIR_ARG}/checkpoints/last.ckpt"
    return 0
  fi
  echo "[run.sh] ERROR: no checkpoint (pass --checkpoint or train first)" >&2
  exit 1
}

case "${MODE}" in
  capabilities)
    exec cat <<'JSON'
{"modes":["capabilities","preprocess","train","test","infer"],"datasets":["fusion360","mechcad"],"tasks":{"fusion360":["seg"],"mechcad":["seg","cls"]},"env_file":"environment.yml","checkpoints":{"best":"checkpoints/best.ckpt","last":"checkpoints/last.ckpt"}}
JSON
    ;;

  preprocess)
    activate_env
    resolve_dataset
    if [[ "${DATASET_ARG}" == "fusion360" || "${DATASET_ARG}" == "360" || "${DATASET_ARG}" == "s2.0.0" ]]; then
      DATASET_DIR="/data/hdd/datasets/s2.0.0" PROCESSED_DIR="${PROCESSED_DIR}" \
        exec bash "${PREPROCESS_SCRIPT}"
    fi
    DATASET_DIR="/data/hdd/datasets/mechcad" PROCESSED_DIR="${PROCESSED_DIR}" \
      exec bash "${PREPROCESS_SCRIPT}"
    ;;

  train)
    activate_env
    resolve_dataset
    require_exp_dir
    GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

    if [[ "${TASK_ARG}" == "cls" ]]; then
      if [[ "${BRT_DATASET_ID}" != "mechcad" ]]; then
        echo "[run.sh] ERROR: task cls only supported for mechcad" >&2
        exit 1
      fi
      exec python classification.py train \
        --num_classes "${NUM_CLASSES}" \
        --dataset_dir "${PROCESSED_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --gpu "${GPU}" \
        --num_control_pts "${NUM_CONTROL_PTS}" \
        --run_dir "${EXP_DIR_ARG}"
    fi

    if [[ "${TASK_ARG}" != "seg" ]]; then
      echo "[run.sh] ERROR: unknown --task ${TASK_ARG} for dataset ${DATASET_ARG}" >&2
      exit 1
    fi

    TRAIN_ARGS=(
      --num_classes "${NUM_CLASSES}"
      --dataset_dir "${PROCESSED_DIR}"
      --batch_size "${BATCH_SIZE}"
      --num_workers "${NUM_WORKERS}"
      --gpu "${GPU}"
      --num_control_pts "${NUM_CONTROL_PTS}"
      --dataset_id "${BRT_DATASET_ID}"
      --git_branch "${GIT_BRANCH}"
      --max_epochs "${MAX_EPOCHS}"
      --run_dir "${EXP_DIR_ARG}"
    )
    if [[ -n "${SPLIT_SOURCE_JSON}" ]]; then
      TRAIN_ARGS+=(--split_source_json "${SPLIT_SOURCE_JSON}")
    fi
    exec python segmentation.py train "${TRAIN_ARGS[@]}"
    ;;

  test)
    activate_env
    resolve_dataset
    require_exp_dir
    CKPT="$(default_checkpoint)"
    GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

    if [[ "${TASK_ARG}" == "cls" ]]; then
      exec python classification.py test \
        --num_classes "${NUM_CLASSES}" \
        --dataset_dir "${PROCESSED_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --gpu "${GPU}" \
        --num_control_pts "${NUM_CONTROL_PTS}" \
        --checkpoint "${CKPT}" \
        --run_dir "${EXP_DIR_ARG}"
    fi

    TEST_ARGS=(
      --num_classes "${NUM_CLASSES}"
      --dataset_dir "${PROCESSED_DIR}"
      --batch_size "${BATCH_SIZE}"
      --num_workers "${NUM_WORKERS}"
      --gpu "${GPU}"
      --num_control_pts "${NUM_CONTROL_PTS}"
      --checkpoint "${CKPT}"
      --dataset_id "${BRT_DATASET_ID}"
      --git_branch "${GIT_BRANCH}"
      --run_dir "${EXP_DIR_ARG}"
    )
    if [[ -n "${SPLIT_SOURCE_JSON}" ]]; then
      TEST_ARGS+=(--split_source_json "${SPLIT_SOURCE_JSON}")
    fi
    exec python segmentation.py test "${TEST_ARGS[@]}"
    ;;

  infer)
    activate_env
    resolve_dataset
    if [[ -z "${OUTPUT_DIR_ARG}" ]]; then
      echo "[run.sh] ERROR: --output-dir is required for infer" >&2
      exit 1
    fi
    if [[ "${TASK_ARG}" != "seg" ]]; then
      echo "[run.sh] ERROR: infer only implemented for seg" >&2
      exit 1
    fi
    require_exp_dir
    CKPT="$(default_checkpoint)"
    INFER_DIR="${OUTPUT_DIR_ARG}/infer"
    mkdir -p "${INFER_DIR}"
    exec python "${REPO_DIR}/scripts/viz_segmentation.py" \
      --dataset_dir "${PROCESSED_DIR}" \
      --checkpoint "${CKPT}" \
      --step_root "${STEP_ROOT}" \
      --output_dir "${INFER_DIR}" \
      --num_classes "${NUM_CLASSES}" \
      --gpu "${GPU}"
    ;;

  *)
    echo "[run.sh] unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac
