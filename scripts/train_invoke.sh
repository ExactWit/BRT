# Shared train/test invocation for branch.sh (seg + cls).
# shellcheck shell=bash

train_entry_supports_model_metadata() {
  local entry="${TRAIN_ENTRY:-segmentation.py}"
  [[ -f "${REPO_DIR}/${entry}" ]] && grep -q -- '--model_id' "${REPO_DIR}/${entry}" 2>/dev/null
}

append_model_metadata_args() {
  local -n _args=$1
  if [[ -n "${SELECTED_MODEL_ID:-}" ]] && train_entry_supports_model_metadata; then
    _args+=(--model_id "${SELECTED_MODEL_ID}")
    _args+=(--model_label "${SELECTED_MODEL_LABEL}")
    _args+=(--model_commit "${SELECTED_MODEL_COMMIT}")
    _args+=(--model_commit_full "${SELECTED_MODEL_COMMIT_FULL}")
    _args+=(--model_status "${SELECTED_MODEL_STATUS}")
  elif [[ -n "${SELECTED_MODEL_ID:-}" ]]; then
    echo "[branch.sh] 注意: 当前 ${TRAIN_ENTRY:-segmentation.py} 不支持 model metadata CLI，已跳过 --model_id 等参数。" >&2
  fi
}

build_brt_eval_args() {
  local -n _out=$1
  _out=(
    --num_classes "${NUM_CLASSES}"
    --dataset_dir "${DATASET_DIR}"
    --batch_size "${BATCH_SIZE:-16}"
    --num_workers "${NUM_WORKERS:-4}"
    --gpu "${GPU:-0}"
    --num_control_pts "${NUM_CONTROL_PTS}"
    --checkpoint "${SELECTED_CHECKPOINT}"
    --git_branch "${SELECTED_MODEL_BRANCH}"
    --dataset_id "${DATASET_ID}"
  )
}

invoke_brt_test() {
  local -a test_args=()
  build_brt_eval_args test_args
  append_model_metadata_args test_args
  if [[ "${TASK}" == "seg" && -n "${SPLIT_SOURCE_JSON:-}" ]]; then
    test_args+=(--split_source_json "${SPLIT_SOURCE_JSON}")
  fi
  python "${TRAIN_ENTRY}" test "${test_args[@]}"
}
