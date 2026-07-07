# Shared train/test invocation for branch.sh (seg + cls).
# Only passes CLI flags present in the checked-out ${TRAIN_ENTRY} (pinned model commit).
# shellcheck shell=bash

train_entry_supports_arg() {
  local arg="$1"
  local entry="${TRAIN_ENTRY:-segmentation.py}"
  [[ -f "${REPO_DIR}/${entry}" ]] && grep -qF -- "${arg}" "${REPO_DIR}/${entry}" 2>/dev/null
}

train_entry_supports_model_metadata() {
  train_entry_supports_arg "--model_id"
}

append_entry_arg_if_supported() {
  local -n _args=$1
  local flag="$2"
  shift 2
  if train_entry_supports_arg "${flag}"; then
    _args+=("${flag}" "$@")
  fi
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
    echo "[branch.sh] 注意: 当前 ${TRAIN_ENTRY} 不支持 model metadata CLI，已跳过 --model_id 等参数。" >&2
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
    --checkpoint "${SELECTED_CHECKPOINT}"
  )
  append_entry_arg_if_supported _out --num_control_pts "${NUM_CONTROL_PTS}"
  append_entry_arg_if_supported _out --git_branch "${SELECTED_MODEL_BRANCH}"
  append_entry_arg_if_supported _out --dataset_id "${DATASET_ID}"
}

# Args: outvar experiment_name log_date run_tag max_epochs batch_size num_workers gpu [experiment_note]
build_brt_train_args() {
  local -n _out=$1
  local experiment_name="$2"
  local log_date="$3"
  local run_tag="$4"
  local max_epochs="$5"
  local batch_size="$6"
  local num_workers="$7"
  local gpu="$8"
  local experiment_note="${9:-}"

  _out=(
    --num_classes "${NUM_CLASSES}"
    --dataset_dir "${DATASET_DIR}"
    --batch_size "${batch_size}"
    --num_workers "${num_workers}"
    --gpu "${gpu}"
  )
  append_entry_arg_if_supported _out --num_control_pts "${NUM_CONTROL_PTS}"
  append_entry_arg_if_supported _out --experiment_name "${experiment_name}"
  append_entry_arg_if_supported _out --log_name "${log_date}"
  append_entry_arg_if_supported _out --run_tag "${run_tag}"
  append_entry_arg_if_supported _out --max_epochs "${max_epochs}"
  append_entry_arg_if_supported _out --git_branch "${SELECTED_MODEL_BRANCH}"
  append_entry_arg_if_supported _out --dataset_id "${DATASET_ID}"
  append_model_metadata_args _out
  if [[ -n "${RESUME_FROM:-}" ]]; then
    append_entry_arg_if_supported _out --resume_from "${RESUME_FROM}"
  fi
  if [[ -n "${SPLIT_SOURCE_JSON:-}" && "${TASK}" == "seg" ]]; then
    append_entry_arg_if_supported _out --split_source_json "${SPLIT_SOURCE_JSON}"
  fi
  if [[ -n "${experiment_note}" ]]; then
    append_entry_arg_if_supported _out --experiment_note "${experiment_note}"
  fi
}

invoke_brt_test() {
  local -a test_args=()
  build_brt_eval_args test_args
  append_model_metadata_args test_args
  if [[ "${TASK}" == "seg" && -n "${SPLIT_SOURCE_JSON:-}" ]]; then
    append_entry_arg_if_supported test_args --split_source_json "${SPLIT_SOURCE_JSON}"
  fi
  python "${TRAIN_ENTRY}" test "${test_args[@]}"
}
