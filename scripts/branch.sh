#!/usr/bin/env bash
# Interactive launcher: pick git branch, dataset, and train / test / viz.
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_DIR}/results"
CONDA_SH="${HOME}/software/miniconda3/etc/profile.d/conda.sh"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate brt
fi

export PYTHONNOUSERSITE=1
cd "${REPO_DIR}"

# Write selection into the variable named by the first argument.
# Must NOT use command substitution $(pick_from_list ...): that breaks read().
pick_from_list() {
  local __outvar="$1"
  local prompt="$2"
  shift 2
  local options=("$@")
  local i choice selected

  if [[ ${#options[@]} -eq 0 ]]; then
    echo "[branch.sh] ERROR: 没有可选项。" >&2
    exit 1
  fi

  echo "" >&2
  echo "${prompt}" >&2
  for i in "${!options[@]}"; do
    printf "  [%d] %s\n" "$((i + 1))" "${options[$i]}" >&2
  done

  while true; do
    if [[ -r /dev/tty ]]; then
      read -r -p "请输入编号: " choice </dev/tty
    else
      read -r -p "请输入编号: " choice
    fi
    if [[ "${choice}" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#options[@]} )); then
      selected="${options[$((choice - 1))]}"
      printf -v "${__outvar}" '%s' "${selected}"
      return 0
    fi
    echo "无效编号，请重试。" >&2
  done
}

sanitize_name() {
  echo "$1" | sed 's#[/ ]#_#g'
}

list_git_branches() {
  local current
  current="$(git rev-parse --abbrev-ref HEAD)"
  while IFS= read -r branch; do
    [[ -z "${branch}" ]] && continue
    if [[ "${branch}" == "${current}" ]]; then
      printf '%s (current)\n' "${branch}"
    else
      printf '%s\n' "${branch}"
    fi
  done < <(git for-each-ref --format='%(refname:short)' refs/heads/ | sort)
}

branch_name_from_label() {
  echo "$1" | sed 's/ (current)$//'
}

checkout_branch() {
  local branch="$1"
  local current
  current="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "${current}" != "${branch}" ]]; then
    echo "[branch.sh] git checkout ${branch}"
    git checkout "${branch}"
  else
    echo "[branch.sh] 已在分支 ${branch}"
  fi
}

configure_dataset() {
  case "$1" in
    360)
      DATASET_ID="360"
      DATASET_DIR="${DATASET_DIR:-/data/hdd/datasets/s2.0.0/processed/brt}"
      STEP_ROOT="${STEP_ROOT:-/data/hdd/datasets/s2.0.0/breps/step}"
      NUM_CLASSES="${NUM_CLASSES:-8}"
      NUM_CONTROL_PTS="${NUM_CONTROL_PTS:-28}"
      PREPROCESS_SCRIPT="${REPO_DIR}/scripts/preprocess_360.sh"
      ;;
    mechcad)
      DATASET_ID="mechcad"
      DATASET_DIR="${DATASET_DIR:-/data/hdd/datasets/mechcad/processed}"
      STEP_ROOT="${STEP_ROOT:-/data/hdd/datasets/mechcad/mechcad}"
      NUM_CLASSES="${NUM_CLASSES:-25}"
      NUM_CONTROL_PTS="${NUM_CONTROL_PTS:-28}"
      PREPROCESS_SCRIPT="${REPO_DIR}/scripts/preprocess_tri.sh"
      ;;
    *)
      echo "[branch.sh] ERROR: unknown dataset $1"
      exit 1
      ;;
  esac
}

ensure_dataset_ready() {
  if [[ ! -f "${DATASET_DIR}/datasplit.json" ]]; then
    echo "[branch.sh] 未找到 ${DATASET_DIR}/datasplit.json"
    if [[ -r /dev/tty ]]; then
      read -r -p "是否现在运行预处理脚本 ${PREPROCESS_SCRIPT}? [y/N] " ans </dev/tty
    else
      read -r -p "是否现在运行预处理脚本 ${PREPROCESS_SCRIPT}? [y/N] " ans
    fi
    if [[ "${ans}" =~ ^[Yy]$ ]]; then
      bash "${PREPROCESS_SCRIPT}"
    else
      echo "[branch.sh] 预处理缺失，退出。"
      exit 1
    fi
  fi
}

metadata_matches() {
  local meta_file="$1"
  local branch="$2"
  local dataset="$3"
  python3 - "${meta_file}" "${branch}" "${dataset}" <<'PY'
import json, sys
path, branch, dataset = sys.argv[1:4]
try:
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
except Exception:
    sys.exit(1)
if meta.get("git_branch") != branch:
    sys.exit(1)
if meta.get("dataset") != dataset:
    sys.exit(1)
print(meta.get("experiment_name", ""))
print(meta.get("log_name", ""))
print(meta.get("log_version", ""))
print(meta.get("dataset_dir", ""))
print(meta.get("num_classes", ""))
PY
}

find_matching_runs() {
  local branch="$1"
  local dataset="$2"
  RUN_LABELS=()
  RUN_DIRS=()
  RUN_CHECKPOINTS=()
  RUN_DATASET_DIRS=()
  RUN_NUM_CLASSES=()
  local -A seen_dirs=()

  add_run() {
    local run_dir="$1"
    local ckpt="$2"
    local label="$3"
    local ds_dir="$4"
    local nclass="$5"
    [[ -n "${seen_dirs[$run_dir]:-}" ]] && return 0
    seen_dirs["$run_dir"]=1
    RUN_LABELS+=("${label}")
    RUN_DIRS+=("${run_dir}")
    RUN_CHECKPOINTS+=("${ckpt}")
    RUN_DATASET_DIRS+=("${ds_dir}")
    RUN_NUM_CLASSES+=("${nclass}")
  }

  while IFS= read -r meta_file; do
    mapfile -t parsed < <(metadata_matches "${meta_file}" "${branch}" "${dataset}" || true)
    [[ ${#parsed[@]} -eq 5 ]] || continue
    local run_dir ckpt label_suffix=""
    run_dir="$(dirname "${meta_file}")"
    if [[ -f "${run_dir}/test_metadata.json" ]]; then
      label_suffix=" [tested]"
    fi
    ckpt=""
    if [[ -f "${run_dir}/last.ckpt" ]]; then
      ckpt="${run_dir}/last.ckpt"
    elif [[ -f "${run_dir}/best.ckpt" ]]; then
      ckpt="${run_dir}/best.ckpt"
    else
      continue
    fi
    add_run "${run_dir}" "${ckpt}" "${parsed[0]}/${parsed[1]}/${parsed[2]}${label_suffix}" "${parsed[3]}" "${parsed[4]}"
  done < <(find "${RESULTS_DIR}" -name experiment_metadata.json 2>/dev/null | sort)

  while IFS= read -r test_meta; do
    mapfile -t parsed < <(python3 - "${test_meta}" "${branch}" "${dataset}" "${RESULTS_DIR}" <<'PY'
import json, pathlib, sys
path, branch, dataset, results_dir = sys.argv[1:5]
with open(path, encoding="utf-8") as f:
    meta = json.load(f)
if meta.get("git_branch") != branch or meta.get("dataset") != dataset:
    sys.exit(1)
run_dir = pathlib.Path(path).parent
try:
    label = str(run_dir.relative_to(results_dir))
except ValueError:
    label = str(run_dir)
ckpt = meta.get("checkpoint", "")
if not ckpt:
    for name in ("last.ckpt", "best.ckpt"):
        candidate = run_dir / name
        if candidate.exists():
            ckpt = str(candidate)
            break
if not ckpt:
    sys.exit(1)
exp_meta = run_dir / "experiment_metadata.json"
num_classes = ""
if exp_meta.exists():
    num_classes = str(json.load(open(exp_meta, encoding="utf-8")).get("num_classes", ""))
metrics = meta.get("metrics", {})
iou = metrics.get("test_iou", "n/a")
print(f"{label} iou={iou}")
print(str(run_dir))
print(ckpt)
print(meta.get("dataset_dir", ""))
print(num_classes)
PY
)
    [[ ${#parsed[@]} -ge 3 ]] || continue
    local nclass="${parsed[4]:-${NUM_CLASSES}}"
    [[ -z "${nclass}" ]] && nclass="${NUM_CLASSES}"
    add_run "${parsed[1]}" "${parsed[2]}" "${parsed[0]} [test]" "${parsed[3]:-${DATASET_DIR}}" "${nclass}"
  done < <(find "${RESULTS_DIR}" -name test_metadata.json 2>/dev/null | sort)

  if [[ ${#RUN_LABELS[@]} -eq 0 ]]; then
    echo "[branch.sh] 未找到匹配 git_branch=${branch}, dataset=${dataset} 的实验。" >&2
    echo "  将列出所有含 checkpoint 的 run（可能缺少 metadata）。" >&2
    while IFS= read -r ckpt; do
      [[ "${ckpt}" == *".backup_"* ]] && continue
      local run_dir label
      run_dir="$(dirname "${ckpt}")"
      label="${run_dir#${RESULTS_DIR}/}"
      add_run "${run_dir}" "${ckpt}" "${label} (no metadata)" "${DATASET_DIR}" "${NUM_CLASSES}"
    done < <(find "${RESULTS_DIR}" -name 'last.ckpt' 2>/dev/null | sort)
  fi
}

pick_run() {
  local branch="$1"
  local dataset="$2"
  local picked
  find_matching_runs "${branch}" "${dataset}"
  if [[ ${#RUN_LABELS[@]} -eq 0 ]]; then
    echo "[branch.sh] ERROR: results/ 下没有可用 checkpoint。"
    exit 1
  fi
  pick_from_list picked "选择实验 run:" "${RUN_LABELS[@]}"
  local i
  for i in "${!RUN_LABELS[@]}"; do
    if [[ "${RUN_LABELS[$i]}" == "${picked}" ]]; then
      SELECTED_RUN_DIR="${RUN_DIRS[$i]}"
      SELECTED_CHECKPOINT="${RUN_CHECKPOINTS[$i]}"
      if [[ -n "${RUN_DATASET_DIRS[$i]}" ]]; then
        DATASET_DIR="${RUN_DATASET_DIRS[$i]}"
      fi
      if [[ -n "${RUN_NUM_CLASSES[$i]}" ]]; then
        NUM_CLASSES="${RUN_NUM_CLASSES[$i]}"
      fi
      return 0
    fi
  done
  echo "[branch.sh] ERROR: 未匹配到所选 run。" >&2
  exit 1
}

list_test_samples() {
  python3 - "${DATASET_DIR}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
split = json.load(open(root / "datasplit.json", encoding="utf-8"))["test"]
for i, item in enumerate(split):
    stem = pathlib.Path(item["face"]).stem
    print(f"{i}\t{stem}")
PY
}

pick_test_sample() {
  mapfile -t SAMPLE_LINES < <(list_test_samples)
  if [[ ${#SAMPLE_LINES[@]} -eq 0 ]]; then
    echo "[branch.sh] ERROR: test 划分为空。"
    exit 1
  fi
  local picked
  pick_from_list picked "选择 test 样本 (index<TAB>stem):" "${SAMPLE_LINES[@]}"
  SAMPLE_INDEX="${picked%%$'\t'*}"
  SAMPLE_STEM="${picked##*$'\t'}"
}

run_train() {
  local branch="$1"
  ensure_dataset_ready
  local branch_tag
  branch_tag="$(sanitize_name "${branch}")"
  local experiment_name
  experiment_name="${EXPERIMENT_NAME:-${branch_tag}_${DATASET_ID}}"
  local batch_size num_workers gpu max_epochs
  batch_size="${BATCH_SIZE:-16}"
  num_workers="${NUM_WORKERS:-4}"
  gpu="${GPU:-0}"
  max_epochs="${MAX_EPOCHS:-1000}"

  echo "[branch.sh] train"
  echo "  branch          : ${branch}"
  echo "  dataset         : ${DATASET_ID}"
  echo "  dataset_dir     : ${DATASET_DIR}"
  echo "  experiment_name : ${experiment_name}"

  mkdir -p logs
  python segmentation.py train \
    --num_classes "${NUM_CLASSES}" \
    --dataset_dir "${DATASET_DIR}" \
    --batch_size "${batch_size}" \
    --num_workers "${num_workers}" \
    --gpu "${gpu}" \
    --num_control_pts "${NUM_CONTROL_PTS}" \
    --experiment_name "${experiment_name}" \
    --max_epochs "${max_epochs}" \
    --git_branch "${branch}" \
    --dataset_id "${DATASET_ID}" \
    2>&1 | tee "logs/train_${branch_tag}_${DATASET_ID}_$(date +%m%d_%H%M%S).log"
}

run_test() {
  local branch="$1"
  pick_run "${branch}" "${DATASET_ID}"
  ensure_dataset_ready
  local batch_size num_workers gpu
  batch_size="${BATCH_SIZE:-16}"
  num_workers="${NUM_WORKERS:-4}"
  gpu="${GPU:-0}"

  echo "[branch.sh] test"
  echo "  checkpoint  : ${SELECTED_CHECKPOINT}"
  echo "  dataset_dir : ${DATASET_DIR}"

  python segmentation.py test \
    --num_classes "${NUM_CLASSES}" \
    --dataset_dir "${DATASET_DIR}" \
    --batch_size "${batch_size}" \
    --num_workers "${num_workers}" \
    --gpu "${gpu}" \
    --num_control_pts "${NUM_CONTROL_PTS}" \
    --checkpoint "${SELECTED_CHECKPOINT}" \
    --git_branch "${branch}" \
    --dataset_id "${DATASET_ID}"
}

run_viz() {
  local branch="$1"
  local format
  pick_run "${branch}" "${DATASET_ID}"
  ensure_dataset_ready
  pick_test_sample
  pick_from_list format "选择导出格式:" "ply" "stp"
  gpu="${GPU:-0}"
  out_dir="${VIZ_OUTPUT_DIR:-${SELECTED_RUN_DIR}/viz}"

  echo "[branch.sh] viz"
  echo "  checkpoint : ${SELECTED_CHECKPOINT}"
  echo "  sample     : ${SAMPLE_INDEX} (${SAMPLE_STEM})"
  echo "  format     : ${format}"
  echo "  output_dir : ${out_dir}"

  python "${REPO_DIR}/scripts/viz_segmentation.py" \
    --checkpoint "${SELECTED_CHECKPOINT}" \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_id "${DATASET_ID}" \
    --step_root "${STEP_ROOT}" \
    --split test \
    --index "${SAMPLE_INDEX}" \
    --format "${format}" \
    --output_dir "${out_dir}" \
    --gpu "${gpu}"
}

main() {
  if [[ ! -t 1 ]] && [[ ! -r /dev/tty ]]; then
    echo "[branch.sh] ERROR: 需要交互式终端。请运行: bash scripts/branch.sh" >&2
    exit 1
  fi

  mapfile -t BRANCHES < <(list_git_branches)
  if [[ ${#BRANCHES[@]} -eq 0 ]]; then
    echo "[branch.sh] ERROR: 没有本地分支。" >&2
    exit 1
  fi

  local branch_label branch dataset mode
  echo "[branch.sh] 仓库: ${REPO_DIR}" >&2
  pick_from_list branch_label "选择 git 分支:" "${BRANCHES[@]}"
  branch="$(branch_name_from_label "${branch_label}")"
  checkout_branch "${branch}"

  pick_from_list dataset "选择数据集:" "360" "mechcad"
  configure_dataset "${dataset}"

  pick_from_list mode "选择操作:" "train" "test" "viz"
  case "${mode}" in
    train) run_train "${branch}" ;;
    test) run_test "${branch}" ;;
    viz) run_viz "${branch}" ;;
  esac
}

main "$@"
