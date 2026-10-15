#!/bin/bash
set -euo
set -o pipefail

which python3
python3 --version

# Install only what you need; do NOT touch numpy
python3 -m pip install -U pip
python3 -m pip install -U pyrtools

# Helps CUDA memory fragmentation (correct env var)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/content/drive/MyDrive/V2_drift
LIB_DIR1=${ROOT}/texturemodel
LIB_DIR2=${LIB_DIR1}/code
LIB_DIR3=${LIB_DIR2}/utils

mkdir -p "${ROOT}/logs"

export PYTHONPATH=${LIB_DIR1}:${LIB_DIR2}:${LIB_DIR3}:$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

python3 -c "import numpy as np; print('numpy:', np.__version__)"
python3 -c "import pyrtools; print('pyrtools:', pyrtools.__version__)" || true

cd ${LIB_DIR2}/feature_extraction
echo "PWD=$(pwd)"

debug=0
use_node_storage=0
n_ori=4
n_sf=4
batch_size=800
which_prf_grid=5

sublist=(8)

for subj in "${sublist[@]}"; do
  echo "=== Running FIRST batch: subject ${subj} ==="

  python3 extract_pyramid_texture_features.py \
    --subject "${subj}" \
    --use_node_storage "${use_node_storage}" \
    --n_ori "${n_ori}" \
    --n_sf "${n_sf}" \
    --batch_size "${batch_size}" \
    --debug "${debug}" \
    --which_prf_grid "${which_prf_grid}" \
    2>&1 | tee "${ROOT}/logs/texture_first_sub${subj}.log"
done
