#!/bin/bash
set -eo pipefail

DATASET_DIR=/data/hdd/datasets/mechcad
OUTPUT_DIR=/data/hdd/datasets/mechcad/processed
REPO_DIR="${HOME}/workspace/repo/BRT"

# # conda activate 在脚本中需要先 source
source "${HOME}/software/miniconda3/etc/profile.d/conda.sh"
conda activate brt

cd "${REPO_DIR}/process"
mkdir -p logs "${OUTPUT_DIR}"

# extract topology -> ${OUTPUT_DIR}/topology/brt/<category>/*.bin
# python gen_tmcad_topo.py "${DATASET_DIR}/mechcad/" "${OUTPUT_DIR}/topology" \
#   2>&1 | tee "${OUTPUT_DIR}/topology.log"

# extract face geometry -> ${OUTPUT_DIR}/triangles/triangles/<category>/*.bin
python gen_tmcad_triangles.py "${DATASET_DIR}/mechcad/" "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/triangles.log"

# split dataset（路径：triangles/<category> 与 topology/brt/<category> 成对样本）
python split_mechcad_available.py \
  --processed-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/datasplit.log"
