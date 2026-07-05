#!/bin/bash
# MechCAD 训练入口 — 推荐用 branch.sh（支持 model registry + seg/cls 任务路由）
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

echo "[train_tri.sh] 请使用: bash scripts/branch.sh" >&2
echo "  360    → seg (segmentation.py)" >&2
echo "  mechcad → seg (默认) 或 cls (classification.py, WIP)" >&2
echo "  环境变量: BRT_INFRA_REF=main  BRT_TASK=seg|cls  BATCH_SIZE=4" >&2
exec bash scripts/branch.sh
