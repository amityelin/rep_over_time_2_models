#!/bin/bash

set -euo pipefail

ROOT=/content/drive/MyDrive/V2_drift/

export PYTHONPATH="${ROOT}texturemodel/:${ROOT}texturemodel/code/:${ROOT}texturemodel/code/utils:${PYTHONPATH:-}"

cd "${ROOT}texturemodel/code/model_fitting"

which python3
python3 --version

python3 check_prediction_scale_ts.py \
    --subject 1 \
    --roi_name V1 \
    --pyr_pca_type pcaHL \
    --pca_pct_var 95 \
    --max_sessions 30