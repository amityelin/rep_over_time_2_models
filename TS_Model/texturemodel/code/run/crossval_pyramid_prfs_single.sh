#!/bin/bash

set -euo pipefail

ROOT=/content/drive/MyDrive/V2_drift/

export PYTHONPATH="${ROOT}texturemodel/:${ROOT}texturemodel/code/:${ROOT}texturemodel/code/utils:${PYTHONPATH:-}"

cd "${ROOT}texturemodel/code/model_fitting"

which python3
python3 --version

python3 reconstruct_cross_session_check.py \
    --subject 1 \
    --roi_name V1 \
    --train_session 0 \
    --test_session 1 \
    --voxel 0 \
    --pyr_pca_type pcaHL \
    --pca_pct_var 95