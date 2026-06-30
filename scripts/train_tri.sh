#!/bin/bash
set -eo pipefail

DATASET_DIR=/data/hdd/datasets/mechcad/processed

# conda activate 在脚本中需要先 source
source "${HOME}/software/miniconda3/etc/profile.d/conda.sh"
conda activate brt

cd "${HOME}/workspace/repo/BRT"    
conda activate brt
# classification
python classification.py train --num_classes num_of_classes --dataset_dir /path/to/dataset/dir --batch_size 16 --num_workers 4
# segmentation
python segmentation.py train --num_classes num_of_classes --dataset_dir /path/to/dataset/dir --batch_size 16 --num_workers 4