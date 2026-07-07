#!/usr/bin/env bash
# Interactive launcher: pick model (pinned commit), dataset, and train / test / viz.
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_DIR}/results"
CONDA_SH="${HOME}/software/miniconda3/etc/profile.d/conda.sh"

# shellcheck source=scripts/run_layout.sh
source "${REPO_DIR}/scripts/run_layout.sh"
# shellcheck source=scripts/model_registry.sh
source "${REPO_DIR}/scripts/model_registry.sh"
# shellcheck source=scripts/task_config.sh
source "${REPO_DIR}/scripts/task_config.sh"

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
  sanitize_run_name "$1"
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
  configure_dataset_base "$1"
  apply_task_layout
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
  local model_id="$2"
  local branch="$3"
  local dataset="$4"
  python3 - "${meta_file}" "${model_id}" "${branch}" "${dataset}" "${REPO_DIR}" <<'PY'
import json, sys
from pathlib import Path

repo_dir = Path(sys.argv[5])
sys.path.insert(0, str(repo_dir))
from utils.experiment_metadata import (
    meta_dataset_dir,
    metadata_matches_model,
)

path, model_id, branch, dataset = sys.argv[1:5]
try:
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
except Exception:
    sys.exit(1)
if not metadata_matches_model(meta, model_id=model_id or None, branch=branch, dataset=dataset):
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
  local model_id="$2"
  local branch="$3"
  local dataset="$4"
  python3 - "${run_dir}" "${RESULTS_DIR}" "${model_id}" "${branch}" "${dataset}" "${DATASET_DIR}" "${NUM_CLASSES}" "${REPO_DIR}" <<'PY'
import json, pathlib, sys

repo_dir = pathlib.Path(sys.argv[8])
sys.path.insert(0, str(repo_dir))
from utils.experiment_metadata import (
    meta_dataset_dir,
    meta_dataset_id,
    meta_git_branch,
    meta_model_id,
    meta_note,
    meta_num_classes,
    metadata_matches_model,
)

run_dir, results_dir, model_id, branch, dataset, default_ds, default_nc = sys.argv[1:8]
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
    if metadata_matches_model(exp, model_id=model_id or None, branch=branch, dataset=dataset):
        tags.append("match")
    else:
        recorded_model = meta_model_id(exp) or "?"
        tags.append(
            f"meta: model={recorded_model}, branch={meta_git_branch(exp) or '?'}, dataset={meta_dataset_id(exp) or '?'}"
        )
    note = meta_note(exp)
    if note:
        tags.append(f"note={note[:40]}{'...' if len(note) > 40 else ''}")
    recorded_model = meta_model_id(exp)
    if recorded_model:
        tags.append(f"model={recorded_model}")
    best_ckpt = (exp.get("checkpoints") or {}).get("best") or {}
    best_ep = best_ckpt.get("epoch_1based")
    if best_ep is None and best_ckpt.get("epoch") is not None:
        best_ep = int(best_ckpt["epoch"]) + 1
    if best_ep is not None:
        tags.append(f"best@epoch={best_ep}")
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
for name in ("best.ckpt", "last.ckpt"):
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
  local model_id="$1"
  local branch="$2"
  local dataset="$3"
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
    mapfile -t parsed < <(metadata_matches "${meta_file}" "${model_id}" "${branch}" "${dataset}" || true)
    [[ ${#parsed[@]} -eq 5 ]] || continue
    [[ "${parsed[0]}" == "${RESULTS_DATASET_NAME}" ]] || continue
    local run_dir ckpt
    run_dir="$(dirname "${meta_file}")"
    if [[ -f "${run_dir}/best.ckpt" ]]; then
      ckpt="${run_dir}/best.ckpt"
    elif [[ -f "${run_dir}/last.ckpt" ]]; then
      ckpt="${run_dir}/last.ckpt"
    else
      continue
    fi
    mapfile -t info < <(describe_run_dir "${run_dir}" "${model_id}" "${branch}" "${dataset}")
    [[ ${#info[@]} -eq 5 ]] || continue
    [[ "${info[0]}" == "${RESULTS_DATASET_NAME}/"* ]] || continue
    add_run "${info[1]}" "${info[2]}" "${info[0]}" "${info[3]}" "${info[4]}"
  done < <(find "${RESULTS_DIR}" -name experiment_metadata.json 2>/dev/null | sort)

  # 2) Always list every checkpoint run (includes legacy runs without metadata, e.g. 0628)
  while IFS= read -r ckpt; do
    [[ "${ckpt}" == *".backup_"* ]] && continue
    local run_dir
    run_dir="$(dirname "${ckpt}")"
    [[ -n "${seen_dirs[$run_dir]:-}" ]] && continue
    mapfile -t info < <(describe_run_dir "${run_dir}" "${model_id}" "${branch}" "${dataset}")
    [[ ${#info[@]} -eq 5 ]] || continue
    [[ "${info[0]}" == "${RESULTS_DATASET_NAME}/"* ]] || continue
    add_run "${info[1]}" "${info[2]}" "${info[0]}" "${info[3]}" "${info[4]}"
  done < <(find "${RESULTS_DIR}" \( -name 'best.ckpt' -o -name 'last.ckpt' \) 2>/dev/null | sort -u)

  if [[ ${#RUN_LABELS[@]} -eq 0 ]]; then
    echo "[branch.sh] 未找到任何 checkpoint。" >&2
  else
    echo "[branch.sh] 共 ${#RUN_LABELS[@]} 个 run（含无 metadata 的历史实验）。" >&2
  fi
}

pick_run() {
  local model_id="$1"
  local branch="$2"
  local dataset="$3"
  local picked
  find_matching_runs "${model_id}" "${branch}" "${dataset}"
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

segmentation_supports_model_metadata() {
  grep -q -- '--model_id' "${REPO_DIR}/segmentation.py" 2>/dev/null
}

append_model_metadata_args() {
  local -n _args=$1
  if [[ -n "${SELECTED_MODEL_ID:-}" ]] && segmentation_supports_model_metadata; then
    _args+=(--model_id "${SELECTED_MODEL_ID}")
    _args+=(--model_label "${SELECTED_MODEL_LABEL}")
    _args+=(--model_commit "${SELECTED_MODEL_COMMIT}")
    _args+=(--model_commit_full "${SELECTED_MODEL_COMMIT_FULL}")
    _args+=(--model_status "${SELECTED_MODEL_STATUS}")
  elif [[ -n "${SELECTED_MODEL_ID:-}" ]]; then
    echo "[branch.sh] 注意: 当前 checkout 的 segmentation.py 不支持 model metadata CLI，已跳过 --model_id 等参数。" >&2
  fi
}

invoke_segmentation_test() {
  local batch_size num_workers gpu
  batch_size="${BATCH_SIZE}"
  num_workers="${NUM_WORKERS:-4}"
  gpu="${GPU:-0}"

  local test_args=(
    --num_classes "${NUM_CLASSES}"
    --dataset_dir "${DATASET_DIR}"
    --batch_size "${batch_size}"
    --num_workers "${num_workers}"
    --gpu "${gpu}"
    --num_control_pts "${NUM_CONTROL_PTS}"
    --checkpoint "${SELECTED_CHECKPOINT}"
    --git_branch "${SELECTED_MODEL_BRANCH}"
    --dataset_id "${DATASET_ID}"
  )
  append_model_metadata_args test_args
  if [[ -n "${SPLIT_SOURCE_JSON:-}" ]]; then
    test_args+=(--split_source_json "${SPLIT_SOURCE_JSON}")
  fi

  python segmentation.py test "${test_args[@]}"
}

ensure_test_per_sample() {
  local per_sample="${SELECTED_RUN_DIR}/test_per_sample.json"
  if python3 - "${per_sample}" "${SELECTED_CHECKPOINT}" "${DATASET_DIR}" <<'PY'
import json, pathlib, sys
per_sample_path, checkpoint, dataset_dir = map(pathlib.Path, sys.argv[1:4])
if not per_sample_path.exists():
    sys.exit(1)
try:
    payload = json.load(open(per_sample_path, encoding="utf-8"))
except Exception:
    sys.exit(1)
if payload.get("checkpoint") != str(checkpoint.resolve()):
    sys.exit(1)
if payload.get("dataset_dir") != str(dataset_dir.resolve()):
    sys.exit(1)
if not payload.get("samples"):
    sys.exit(1)
sys.exit(0)
PY
  then
    return 0
  fi

  echo "[branch.sh] 未找到有效的 per-sample test 结果 (${per_sample})" >&2
  echo "[branch.sh] 自动运行 test 以记录每个 test 样本的 iou/acc..." >&2
  invoke_segmentation_test
}

list_test_samples() {
  local run_dir="$1"
  python3 - "${DATASET_DIR}" "${run_dir}" "${REPO_DIR}" <<'PY'
import pathlib, sys
sys.path.insert(0, sys.argv[3])
from utils.test_per_sample import list_test_samples_sorted

dataset_dir = pathlib.Path(sys.argv[1])
run_dir = pathlib.Path(sys.argv[2])
for row in list_test_samples_sorted(dataset_dir, run_dir):
    idx = row["index"]
    stem = row["stem"]
    if row.get("iou") is not None and row.get("acc") is not None:
        print(
            f"{idx}\t{stem}\tiou={row['iou'] * 100:.1f}\tacc={row['acc'] * 100:.1f}"
        )
    else:
        print(f"{idx}\t{stem}\t(no metrics)")
PY
}

resolve_sample_by_index() {
  local raw_index="$1"
  python3 - "${DATASET_DIR}" "${SELECTED_RUN_DIR}" "${REPO_DIR}" "${raw_index}" <<'PY'
import pathlib, sys
sys.path.insert(0, sys.argv[3])
from utils.test_per_sample import resolve_test_sample_index

dataset_dir = pathlib.Path(sys.argv[1])
run_dir = pathlib.Path(sys.argv[2])
raw_index = sys.argv[4]
try:
    row = resolve_test_sample_index(dataset_dir, raw_index)
except (ValueError, IndexError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)

print(row["index"])
print(row["stem"])
if row.get("iou") is not None and row.get("acc") is not None:
    print(f"iou={row['iou'] * 100:.1f}\tacc={row['acc'] * 100:.1f}")
else:
    print("(no metrics)")
PY
}

pick_test_sample() {
  local index_input metrics_line
  echo "" >&2
  echo "按 test 固有索引导出（datasplit 编号，与文件名 fusion360_0251_... 中的 0251 一致）。" >&2
  echo "输入索引（如 251 / 0251）直接可视化；回车进入按 iou 升序的排行榜。" >&2
  if [[ -r /dev/tty ]]; then
    read -r -p "固有索引> " index_input </dev/tty
  else
    read -r -p "固有索引> " index_input
  fi

  if [[ -n "${index_input}" ]]; then
    mapfile -t resolved < <(resolve_sample_by_index "${index_input}") || {
      echo "[branch.sh] ERROR: 无效或越界的固有索引: ${index_input}" >&2
      exit 1
    }
    SAMPLE_INDEX="${resolved[0]}"
    SAMPLE_STEM="${resolved[1]}"
    metrics_line="${resolved[2]:-}"
    echo "[branch.sh] 固有索引 ${SAMPLE_INDEX} → ${SAMPLE_STEM}  ${metrics_line}" >&2
    return 0
  fi

  mapfile -t SAMPLE_LINES < <(list_test_samples "${SELECTED_RUN_DIR}")
  if [[ ${#SAMPLE_LINES[@]} -eq 0 ]]; then
    echo "[branch.sh] ERROR: test 划分为空。"
    exit 1
  fi
  local picked
  pick_from_list picked "选择 test 样本 (按 iou 升序, index<TAB>stem<TAB>metrics):" "${SAMPLE_LINES[@]}"
  SAMPLE_INDEX="${picked%%$'\t'*}"
  SAMPLE_STEM="$(echo "${picked}" | awk -F '\t' '{print $2}')"
}

parse_run_dir_layout() {
  local run_dir="$1"
  mapfile -t PARSED_RUN_LAYOUT < <(python3 - "${run_dir}" "${RESULTS_DIR}" <<'PY'
import pathlib, sys

run_dir = pathlib.Path(sys.argv[1]).resolve()
results_dir = pathlib.Path(sys.argv[2]).resolve()
rel = run_dir.relative_to(results_dir)
parts = rel.parts
if len(parts) < 3:
    raise SystemExit(f"unexpected run dir layout: {run_dir}")
print(parts[0])
print(parts[1])
print(parts[2])
PY
  )
  RUN_EXPERIMENT_NAME="${PARSED_RUN_LAYOUT[0]}"
  RUN_LOG_NAME="${PARSED_RUN_LAYOUT[1]}"
  RUN_LOG_VERSION="${PARSED_RUN_LAYOUT[2]}"
}

backup_run_dir() {
  local run_dir="$1"
  local backup_dir="${run_dir}.backup_$(date +%Y%m%d_%H%M%S)"
  echo "[branch.sh] 备份 run 目录 -> ${backup_dir}" >&2
  cp -a "${run_dir}" "${backup_dir}"
}

run_train() {
  ensure_dataset_ready
  local model_tag
  model_tag="$(sanitize_name "${SELECTED_MODEL_ID}")"
  local experiment_name run_tag log_date
  experiment_name="${EXPERIMENT_NAME:-${RESULTS_DATASET_NAME}}"
  log_date="${LOG_NAME:-$(date +%m%d)}"
  RUN_TAG="${SELECTED_MODEL_RUN_TAG}"
  run_tag="$(resolve_run_tag "${SELECTED_MODEL_BRANCH}")"
  local batch_size num_workers gpu max_epochs experiment_note
  batch_size="${BATCH_SIZE}"
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
    --log_name "${log_date}"
    --run_tag "${run_tag}"
    --max_epochs "${max_epochs}"
    --git_branch "${SELECTED_MODEL_BRANCH}"
    --dataset_id "${DATASET_ID}"
  )
  append_model_metadata_args train_args
  if [[ -n "${RESUME_FROM:-}" ]]; then
    train_args+=(--resume_from "${RESUME_FROM}")
  fi
  if [[ -n "${SPLIT_SOURCE_JSON:-}" && "${TASK}" == "seg" ]]; then
    train_args+=(--split_source_json "${SPLIT_SOURCE_JSON}")
  fi
  if [[ -n "${experiment_note}" ]]; then
    train_args+=(--experiment_note "${experiment_note}")
  fi

  echo "[branch.sh] train (${TASK})"
  echo "  model           : ${SELECTED_MODEL_ID} @ ${SELECTED_MODEL_COMMIT} (${SELECTED_MODEL_STATUS})"
  echo "  branch          : ${SELECTED_MODEL_BRANCH}"
  echo "  dataset         : ${DATASET_ID}"
  echo "  task            : ${TASK} — ${TASK_LABEL}"
  echo "  entry           : ${TRAIN_ENTRY}"
  echo "  batch_size      : ${batch_size}"
  echo "  dataset_dir     : ${DATASET_DIR}"
  echo "  results_path    : results/${experiment_name}/${log_date}/${run_tag}"
  if [[ -n "${SPLIT_SOURCE_JSON:-}" && "${TASK}" == "seg" ]]; then
    echo "  split_source    : ${SPLIT_SOURCE_JSON}"
  fi
  if [[ -n "${experiment_note}" ]]; then
    echo "  note            : ${experiment_note}"
  fi

  if [[ -n "${SELECTED_MODEL_METRICS_DOC:-}" ]]; then
    echo "  metrics_doc     : ${SELECTED_MODEL_METRICS_DOC}"
  fi

  mkdir -p logs
  python "${TRAIN_ENTRY}" train \
    "${train_args[@]}" \
    2>&1 | tee "logs/train_${model_tag}_${DATASET_ID}_${TASK}_$(date +%m%d_%H%M%S).log"
}

run_resume() {
  task_supports_mode "resume"
  pick_run "${SELECTED_MODEL_ID}" "${SELECTED_MODEL_BRANCH}" "${DATASET_ID}"
  ensure_dataset_ready
  parse_run_dir_layout "${SELECTED_RUN_DIR}"

  local resume_ckpt model_tag batch_size num_workers gpu max_epochs
  resume_ckpt="${RESUME_CKPT:-${SELECTED_RUN_DIR}/last.ckpt}"
  if [[ ! -f "${resume_ckpt}" ]]; then
    if [[ -f "${SELECTED_RUN_DIR}/best.ckpt" ]]; then
      resume_ckpt="${SELECTED_RUN_DIR}/best.ckpt"
      echo "[branch.sh] 未找到 last.ckpt，使用 best.ckpt" >&2
    else
      echo "[branch.sh] ERROR: 未找到可续训 checkpoint: ${SELECTED_RUN_DIR}" >&2
      exit 1
    fi
  fi

  if [[ "${SKIP_RUN_BACKUP:-0}" != "1" ]]; then
    backup_run_dir "${SELECTED_RUN_DIR}"
  else
    echo "[branch.sh] SKIP_RUN_BACKUP=1，跳过 run 目录备份" >&2
  fi

  model_tag="$(sanitize_name "${SELECTED_MODEL_ID}")"
  batch_size="${BATCH_SIZE}"
  num_workers="${NUM_WORKERS:-4}"
  gpu="${GPU:-0}"
  max_epochs="${MAX_EPOCHS:-1000}"

  local train_args=(
    --num_classes "${NUM_CLASSES}"
    --dataset_dir "${DATASET_DIR}"
    --batch_size "${batch_size}"
    --num_workers "${num_workers}"
    --gpu "${gpu}"
    --num_control_pts "${NUM_CONTROL_PTS}"
    --experiment_name "${RUN_EXPERIMENT_NAME}"
    --log_name "${RUN_LOG_NAME}"
    --run_tag "${RUN_LOG_VERSION}"
    --max_epochs "${max_epochs}"
    --git_branch "${SELECTED_MODEL_BRANCH}"
    --dataset_id "${DATASET_ID}"
    --resume_from "${resume_ckpt}"
  )
  append_model_metadata_args train_args
  if [[ -n "${SPLIT_SOURCE_JSON:-}" ]]; then
    train_args+=(--split_source_json "${SPLIT_SOURCE_JSON}")
  fi
  if [[ -n "${EXPERIMENT_NOTE:-}" ]]; then
    train_args+=(--experiment_note "${EXPERIMENT_NOTE}")
  fi

  echo "[branch.sh] resume"
  echo "  model        : ${SELECTED_MODEL_ID} @ ${SELECTED_MODEL_COMMIT}"
  echo "  branch       : ${SELECTED_MODEL_BRANCH}"
  echo "  dataset      : ${DATASET_ID}"
  echo "  dataset_dir  : ${DATASET_DIR}"
  echo "  run_dir      : ${SELECTED_RUN_DIR}"
  echo "  resume_from  : ${resume_ckpt}"
  echo "  max_epochs   : ${max_epochs}"
  echo "  tensorboard  : results/${RUN_EXPERIMENT_NAME}/${RUN_LOG_NAME}/${RUN_LOG_VERSION}"

  mkdir -p logs
  python segmentation.py train \
    "${train_args[@]}" \
    2>&1 | tee -a "logs/resume_${model_tag}_${DATASET_ID}_${RUN_LOG_NAME}_${RUN_LOG_VERSION}_$(date +%m%d_%H%M%S).log"
}

run_test() {
  task_supports_mode "test"
  pick_run "${SELECTED_MODEL_ID}" "${SELECTED_MODEL_BRANCH}" "${DATASET_ID}"
  ensure_dataset_ready

  echo "[branch.sh] test"
  echo "  model       : ${SELECTED_MODEL_ID} @ ${SELECTED_MODEL_COMMIT}"
  echo "  checkpoint  : ${SELECTED_CHECKPOINT}"
  echo "  dataset_dir : ${DATASET_DIR}"

  invoke_segmentation_test
}

run_viz() {
  task_supports_mode "viz"
  local format viz_action
  pick_run "${SELECTED_MODEL_ID}" "${SELECTED_MODEL_BRANCH}" "${DATASET_ID}"
  ensure_dataset_ready
  ensure_test_per_sample
  pick_from_list format "选择导出格式 (本 session 内固定):" "ply" "stp"
  gpu="${GPU:-0}"
  local viz_gap
  viz_gap="${VIZ_GAP:-0.5}"
  out_dir="${VIZ_OUTPUT_DIR:-${SELECTED_RUN_DIR}/viz}"

  echo "[branch.sh] viz session"
  echo "  model      : ${SELECTED_MODEL_ID} @ ${SELECTED_MODEL_COMMIT}"
  echo "  checkpoint : ${SELECTED_CHECKPOINT}"
  echo "  format     : ${format}"
  echo "  output_dir : ${out_dir}"
  if [[ "${format}" == "stp" ]]; then
    echo "  说明       : STEP 使用 XCAF 面色（左 GT / 右 Pred），FreeCAD / CAD Assistant 可查看"
  else
    echo "  说明       : PLY 左 GT / 右 Pred 并排对比"
  fi
  echo "  退出       : 每次导出完成后输入 quit（或 q）退出 viz；回车继续选下一个样本"
  echo "  样本选择   : 先输入固有索引（如 0251），回车进入 iou 排行榜"
  echo "  样本列表   : iou 升序时最差优先；固有索引便于跨模型对比同一样本"
  echo "  gt/pred间距: gap=${viz_gap}（相对模型 X 宽度比例，可用 VIZ_GAP 覆盖）"
  echo ""

  while true; do
    pick_test_sample

    echo "[branch.sh] viz"
    echo "  sample : ${SAMPLE_INDEX} (${SAMPLE_STEM})"

    python "${REPO_DIR}/scripts/viz_segmentation.py" \
      --checkpoint "${SELECTED_CHECKPOINT}" \
      --dataset_dir "${DATASET_DIR}" \
      --dataset_id "${DATASET_ID}" \
      --num_classes "${NUM_CLASSES}" \
      --step_root "${STEP_ROOT}" \
      --split test \
      --index "${SAMPLE_INDEX}" \
      --format "${format}" \
      --output_dir "${out_dir}" \
      --gap "${viz_gap}" \
      --gpu "${gpu}"

    echo "" >&2
    if [[ -r /dev/tty ]]; then
      read -r -p "viz 完成。回车继续选样本，输入 quit 退出: " viz_action </dev/tty
    else
      read -r -p "viz 完成。回车继续选样本，输入 quit 退出: " viz_action
    fi
    case "${viz_action}" in
      quit|Quit|QUIT|q|Q)
        echo "[branch.sh] 退出 viz session。"
        break
        ;;
      *)
        echo "" >&2
        ;;
    esac
  done
}

main() {
  if [[ ! -t 1 ]] && [[ ! -r /dev/tty ]]; then
    echo "[branch.sh] ERROR: 需要交互式终端。请运行: bash scripts/branch.sh" >&2
    exit 1
  fi

  if [[ ! -f "${REPO_DIR}/scripts/task_config.sh" ]]; then
    echo "[branch.sh] ERROR: scripts/task_config.sh 缺失 — 请从 main 工作区运行 branch.sh，或 git merge main。" >&2
    exit 1
  fi

  load_model_registry
  if [[ ${#MODEL_LABELS[@]} -eq 0 ]]; then
    echo "[branch.sh] ERROR: model_registry.tsv 为空。" >&2
    exit 1
  fi

  local model_label dataset task_pick mode
  echo "[branch.sh] 仓库: ${REPO_DIR}" >&2
  echo "[branch.sh] infra registry: BRT_INFRA_REF=${BRT_INFRA_REF:-main} (模型代码 checkout 到 registry git_ref)" >&2
  pick_from_list model_label "选择模型 (pinned commit):" "${MODEL_LABELS[@]}"
  resolve_model_by_label "${model_label}"
  checkout_model_ref "${SELECTED_MODEL_REF}" "${SELECTED_MODEL_BRANCH}"
  if [[ "${SELECTED_MODEL_STATUS}" == "archived" ]]; then
    echo "[branch.sh] 注意: 已归档模型 ${SELECTED_MODEL_ID} — ${SELECTED_MODEL_METRICS_DOC}" >&2
  fi

  pick_from_list dataset "选择数据集:" "360" "mechcad"
  if [[ "${dataset}" == "mechcad" && -z "${BRT_TASK:-}" ]]; then
    pick_from_list task_pick "MechCAD 任务:" \
      "seg (零件类→每面广播, 走 segmentation.py)" \
      "cls (原生零件分类, WIP)"
    task_pick="${task_pick%% *}"
    resolve_brt_task "${dataset}" "${task_pick}"
  else
    resolve_brt_task "${dataset}"
  fi
  configure_dataset "${dataset}"

  pick_from_list mode "选择操作:" "train" "resume" "test" "viz"
  case "${mode}" in
    train) run_train ;;
    resume) run_resume ;;
    test) run_test ;;
    viz) run_viz ;;
  esac
}

main "$@"
