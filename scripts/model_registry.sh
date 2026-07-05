# Model registry: pinned git refs for reproducible train/test/viz.
# Tab-separated table: scripts/model_registry.tsv
#
# Registry 默认从 BRT_INFRA_REF 读取（不必 checkout main），模型代码仍 checkout 到
# registry 行的 git_ref。

MODEL_REGISTRY_FILE="${REPO_DIR}/scripts/model_registry.tsv"
BRT_INFRA_REF="${BRT_INFRA_REF:-main}"

_registry_tsv_from_git() {
  local ref="$1"
  if ! git rev-parse --verify "${ref}^{commit}" >/dev/null 2>&1; then
    return 1
  fi
  git show "${ref}:scripts/model_registry.tsv" 2>/dev/null
}

_load_registry_rows() {
  while IFS=$'\t' read -r id label ref branch run_tag status metrics_doc _rest; do
    [[ -z "${id}" || "${id}" == "id" || "${id:0:1}" == "#" ]] && continue
    MODEL_IDS+=("${id}")
    MODEL_LABELS+=("${label} @ ${ref}")
    MODEL_REFS+=("${ref}")
    MODEL_BRANCHES+=("${branch}")
    MODEL_RUN_TAGS+=("${run_tag}")
    MODEL_STATUSES+=("${status}")
    MODEL_METRICS_DOCS+=("${metrics_doc}")
  done
}

# Populates MODEL_IDS MODEL_LABELS MODEL_REFS MODEL_BRANCHES MODEL_RUN_TAGS MODEL_STATUSES
load_model_registry() {
  MODEL_IDS=()
  MODEL_LABELS=()
  MODEL_REFS=()
  MODEL_BRANCHES=()
  MODEL_RUN_TAGS=()
  MODEL_STATUSES=()
  MODEL_METRICS_DOCS=()
  REGISTRY_SOURCE=""

  local registry_tsv=""
  registry_tsv="$(_registry_tsv_from_git "${BRT_INFRA_REF}" || true)"
  if [[ -n "${registry_tsv}" ]]; then
    local infra_short
    infra_short="$(git rev-parse --short "${BRT_INFRA_REF}" 2>/dev/null || echo "${BRT_INFRA_REF}")"
    echo "[model_registry] registry ← ${BRT_INFRA_REF} (${infra_short})" >&2
    _load_registry_rows <<< "${registry_tsv}"
    REGISTRY_SOURCE="${BRT_INFRA_REF}"
    return 0
  fi

  if [[ -f "${MODEL_REGISTRY_FILE}" ]]; then
    echo "[model_registry] registry ← working tree (${MODEL_REGISTRY_FILE})" >&2
    echo "[model_registry] 提示: 可设 BRT_INFRA_REF=main 始终读 main 上的 registry。" >&2
    _load_registry_rows < "${MODEL_REGISTRY_FILE}"
    REGISTRY_SOURCE="working-tree"
    return 0
  fi

  echo "[model_registry] ERROR: missing registry (BRT_INFRA_REF=${BRT_INFRA_REF}, file=${MODEL_REGISTRY_FILE})" >&2
  exit 1
}

resolve_model_by_label() {
  local picked="$1"
  local i
  for i in "${!MODEL_LABELS[@]}"; do
    if [[ "${MODEL_LABELS[$i]}" == "${picked}" ]]; then
      SELECTED_MODEL_ID="${MODEL_IDS[$i]}"
      SELECTED_MODEL_LABEL="${MODEL_LABELS[$i]%% @ *}"
      SELECTED_MODEL_REF="${MODEL_REFS[$i]}"
      SELECTED_MODEL_BRANCH="${MODEL_BRANCHES[$i]}"
      SELECTED_MODEL_RUN_TAG="${MODEL_RUN_TAGS[$i]}"
      SELECTED_MODEL_STATUS="${MODEL_STATUSES[$i]}"
      SELECTED_MODEL_METRICS_DOC="${MODEL_METRICS_DOCS[$i]}"
      return 0
    fi
  done
  echo "[model_registry] ERROR: 未匹配模型: ${picked}" >&2
  exit 1
}

checkout_model_ref() {
  local ref="$1"
  local branch="$2"
  local short_commit

  if ! git cat-file -e "${ref}^{commit}" 2>/dev/null; then
    echo "[model_registry] ERROR: 本地找不到提交 ${ref}。请先 fetch / merge 对应分支。" >&2
    exit 1
  fi

  short_commit="$(git rev-parse --short "${ref}")"
  SELECTED_MODEL_COMMIT="${short_commit}"
  SELECTED_MODEL_COMMIT_FULL="$(git rev-parse "${ref}")"

  if [[ -n "${branch}" ]] && git show-ref --verify --quiet "refs/heads/${branch}"; then
    local current_branch
    current_branch="$(git rev-parse --abbrev-ref HEAD)"
    if [[ "${current_branch}" != "${branch}" ]]; then
      echo "[model_registry] git checkout ${branch}"
      git checkout "${branch}"
    fi
  fi

  local head_commit
  head_commit="$(git rev-parse HEAD)"
  if [[ "${head_commit}" != "$(git rev-parse "${ref}")" ]]; then
    echo "[model_registry] git checkout ${short_commit} (pinned model ref)"
    git checkout "${ref}"
  else
    echo "[model_registry] 已在模型提交 ${short_commit}"
  fi
}
