# Shared helpers for results/<dataset>/<date>/<tag> layout.

sanitize_run_name() {
  echo "$1" | sed 's#[/ ]#_#g'
}

branch_to_run_tag() {
  local branch="$1"
  if [[ "${branch}" == "main" ]]; then
    echo "baseline"
    return 0
  fi
  if [[ "${branch}" =~ ^scheme-([a-z]) ]]; then
    echo "scheme${BASH_REMATCH[1]}"
    return 0
  fi
  sanitize_run_name "${branch}" | tr '[:upper:]' '[:lower:]'
}

resolve_run_tag() {
  local branch="$1"
  local tag="${RUN_TAG:-$(branch_to_run_tag "${branch}")}"
  local log_date="${LOG_NAME:-$(date +%m%d)}"
  local dataset_name="${EXPERIMENT_NAME:-${RESULTS_DATASET_NAME}}"
  local results_root="${RESULTS_DIR:-.}"
  local run_dir="${results_root}/${dataset_name}/${log_date}/${tag}"

  if [[ -d "${run_dir}" && -z "${RESUME_FROM:-}" ]]; then
    tag="${tag}_$(date +%H%M%S)"
    echo "[run_layout] run 目录已存在，使用 tag: ${tag}" >&2
  fi
  printf '%s' "${tag}"
}
