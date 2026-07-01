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
      SPLIT_SOURCE_JSON="${SPLIT_SOURCE_JSON:-/data/hdd/datasets/s2.0.0/processed/dataset.json}"
      STEP_ROOT="${STEP_ROOT:-/data/hdd/datasets/s2.0.0/breps/step}"
      NUM_CLASSES="${NUM_CLASSES:-8}"
      NUM_CONTROL_PTS="${NUM_CONTROL_PTS:-28}"
      PREPROCESS_SCRIPT="${REPO_DIR}/scripts/preprocess_360.sh"
      ;;
    mechcad)
      DATASET_ID="mechcad"
      DATASET_DIR="${DATASET_DIR:-/data/hdd/datasets/mechcad/processed}"
      SPLIT_SOURCE_JSON="${SPLIT_SOURCE_JSON:-}"
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
  python3 - "${meta_file}" "${branch}" "${dataset}" "${REPO_DIR}" <<'PY'
import json, sys
from pathlib import Path

repo_dir = Path(sys.argv[4])
sys.path.insert(0, str(repo_dir))
from utils.experiment_metadata import (
    meta_dataset_dir,
    meta_dataset_id,
    meta_git_branch,
)

path, branch, dataset = sys.argv[1:4]
try:
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
except Exception:
    sys.exit(1)
if meta_git_branch(meta) != branch:
    sys.exit(1)
if meta_dataset_id(meta) != dataset:
    sys.exit(1)
run = meta.get("run") or {}
print(run.get("experiment_name") or meta.get("experiment_name", ""))
print(run.get("log_name") or meta.get("log_name", ""))
print(run.get("log_version") or meta.get("log_version", ""))
print(meta_dataset_dir(meta) or "")
dataset_block = meta.get("dataset") if isinstance(meta.get("dataset"), dict) else {}
print(dataset_block.get("num_classes") or meta.get("num_classes", ""))
PY
}

describe_run_dir() {
  local run_dir="$1"
  local branch="$2"
  local dataset="$3"
  python3 - "${run_dir}" "${RESULTS_DIR}" "${branch}" "${dataset}" "${DATASET_DIR}" "${NUM_CLASSES}" "${REPO_DIR}" <<'PY'
import json, pathlib, sys

repo_dir = pathlib.Path(sys.argv[7])
sys.path.insert(0, str(repo_dir))
from utils.experiment_metadata import (
    meta_dataset_dir,
    meta_dataset_id,
    meta_git_branch,
    meta_note,
    meta_num_classes,
)

run_dir, results_dir, branch, dataset, default_ds, default_nc = sys.argv[1:7]
run_dir = pathlib.Path(run_dir)
results_dir = pathlib.Path(results_dir)
try:
    rel = str(run_dir.relative_to(results_dir))
except ValueError:
    rel = str(run_dir)

exp_path = run_dir / "experiment_metadata.json"
test_path = run_dir / "test_metadata.json"
tags = []
ds_dir = default_ds
num_classes = default_nc

if exp_path.exists():
    exp = json.load(open(exp_path, encoding="utf-8"))
    ds_dir = meta_dataset_dir(exp) or ds_dir
    nc = meta_num_classes(exp)
    if nc is not None:
        num_classes = str(nc)
    if meta_git_branch(exp) == branch and meta_dataset_id(exp) == dataset:
        tags.append("match")
    else:
        tags.append(
            f"meta: branch={meta_git_branch(exp) or '?'}, dataset={meta_dataset_id(exp) or '?'}"
        )
    note = meta_note(exp)
    if note:
        tags.append(f"note={note[:40]}{'...' if len(note) > 40 else ''}")
elif test_path.exists():
    test = json.load(open(test_path, encoding="utf-8"))
    ds_dir = test.get("dataset_dir") or (test.get("dataset") or {}).get("processed_dir") or ds_dir
    git_branch = test.get("git_branch") or (test.get("git") or {}).get("branch")
    dataset_id = test.get("dataset")
    if isinstance(dataset_id, dict):
        dataset_id = dataset_id.get("id")
    if git_branch == branch and dataset_id == dataset:
        tags.append("match")
    else:
        tags.append(
            f"meta: branch={git_branch or '?'}, dataset={dataset_id or '?'}"
        )
else:
    tags.append("no metadata")

if test_path.exists():
    test = json.load(open(test_path, encoding="utf-8"))
    iou = (test.get("metrics") or {}).get("test_iou")
    if iou is not None:
        tags.append(f"iou={iou}")
    tags.append("tested")

label = rel
if tags:
    label += " [" + ", ".join(tags) + "]"

ckpt = ""
for name in ("last.ckpt", "best.ckpt"):
    candidate = run_dir / name
    if candidate.exists():
        ckpt = str(candidate)
        break
if not ckpt:
    sys.exit(1)
print(label)
print(str(run_dir))
print(ckpt)
print(ds_dir)
print(num_classes)
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

  # 1) Runs with experiment_metadata matching branch + dataset (listed first)
  while IFS= read -r meta_file; do
    mapfile -t parsed < <(metadata_matches "${meta_file}" "${branch}" "${dataset}" || true)
    [[ ${#parsed[@]} -eq 5 ]] || continue
    local run_dir ckpt
    run_dir="$(dirname "${meta_file}")"
    if [[ -f "${run_dir}/last.ckpt" ]]; then
      ckpt="${run_dir}/last.ckpt"
    elif [[ -f "${run_dir}/best.ckpt" ]]; then
      ckpt="${run_dir}/best.ckpt"
    else
      continue
    fi
    mapfile -t info < <(describe_run_dir "${run_dir}" "${branch}" "${dataset}")
    [[ ${#info[@]} -eq 5 ]] || continue
    add_run "${info[1]}" "${info[2]}" "${info[0]}" "${info[3]}" "${info[4]}"
  done < <(find "${RESULTS_DIR}" -name experiment_metadata.json 2>/dev/null | sort)

  # 2) Always list every checkpoint run (includes legacy runs without metadata, e.g. 0628)
  while IFS= read -r ckpt; do
    [[ "${ckpt}" == *".backup_"* ]] && continue
    local run_dir
    run_dir="$(dirname "${ckpt}")"
    [[ -n "${seen_dirs[$run_dir]:-}" ]] && continue
    mapfile -t info < <(describe_run_dir "${run_dir}" "${branch}" "${dataset}")
    [[ ${#info[@]} -eq 5 ]] || continue
    add_run "${info[1]}" "${info[2]}" "${info[0]}" "${info[3]}" "${info[4]}"
  done < <(find "${RESULTS_DIR}" -name 'last.ckpt' 2>/dev/null | sort)

  if [[ ${#RUN_LABELS[@]} -eq 0 ]]; then
    echo "[branch.sh] 未找到任何 checkpoint。" >&2
  else
    echo "[branch.sh] 共 ${#RUN_LABELS[@]} 个 run（含无 metadata 的历史实验）。" >&2
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
  local batch_size num_workers gpu max_epochs experiment_note
  batch_size="${BATCH_SIZE:-16}"
  num_workers="${NUM_WORKERS:-4}"
  gpu="${GPU:-0}"
  max_epochs="${MAX_EPOCHS:-1000}"
  experiment_note="${EXPERIMENT_NOTE:-}"

  if [[ -z "${experiment_note}" ]]; then
    echo "" >&2
    echo "实验备注（可选，直接回车跳过）：" >&2
    if [[ -r /dev/tty ]]; then
      read -r -p "> " experiment_note </dev/tty
    else
      read -r -p "> " experiment_note
    fi
  fi

  local train_args=(
    --num_classes "${NUM_CLASSES}"
    --dataset_dir "${DATASET_DIR}"
    --batch_size "${batch_size}"
    --num_workers "${num_workers}"
    --gpu "${gpu}"
    --num_control_pts "${NUM_CONTROL_PTS}"
    --experiment_name "${experiment_name}"
    --max_epochs "${max_epochs}"
    --git_branch "${branch}"
    --dataset_id "${DATASET_ID}"
  )
  if [[ -n "${SPLIT_SOURCE_JSON:-}" ]]; then
    train_args+=(--split_source_json "${SPLIT_SOURCE_JSON}")
  fi
  if [[ -n "${experiment_note}" ]]; then
    train_args+=(--experiment_note "${experiment_note}")
  fi

  echo "[branch.sh] train"
  echo "  branch          : ${branch}"
  echo "  dataset         : ${DATASET_ID}"
  echo "  dataset_dir     : ${DATASET_DIR}"
  echo "  experiment_name : ${experiment_name}"
  if [[ -n "${SPLIT_SOURCE_JSON:-}" ]]; then
    echo "  split_source    : ${SPLIT_SOURCE_JSON}"
  fi
  if [[ -n "${experiment_note}" ]]; then
    echo "  note            : ${experiment_note}"
  fi

  mkdir -p logs
  python segmentation.py train \
    "${train_args[@]}" \
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

  local test_args=(
    --num_classes "${NUM_CLASSES}"
    --dataset_dir "${DATASET_DIR}"
    --batch_size "${batch_size}"
    --num_workers "${num_workers}"
    --gpu "${gpu}"
    --num_control_pts "${NUM_CONTROL_PTS}"
    --checkpoint "${SELECTED_CHECKPOINT}"
    --git_branch "${branch}"
    --dataset_id "${DATASET_ID}"
  )
  if [[ -n "${SPLIT_SOURCE_JSON:-}" ]]; then
    test_args+=(--split_source_json "${SPLIT_SOURCE_JSON}")
  fi

  python segmentation.py test \
    "${test_args[@]}"
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
