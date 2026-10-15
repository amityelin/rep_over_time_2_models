#!/bin/bash

set -euo pipefail

ROOT=/content/drive/MyDrive/V2_drift/

export PYTHONPATH="${ROOT}texturemodel/:${ROOT}texturemodel/code/:${ROOT}texturemodel/code/utils:${PYTHONPATH:-}"

cd "${ROOT}texturemodel/code/model_fitting"

which python3
python3 --version

echo "============================================================"
echo "Population tail sensitivity"
echo "Model: pcaHL95"
echo "Normalization: raw"
echo "Subjects: 1-8"
echo "ROIs: V1 V2 V3 hV4"
echo "Images: 100"
echo "============================================================"

for SUBJECT in {1..8}; do

    for ROI in V1 V2 V3 hV4; do

        echo
        echo "============================================================"
        echo "Running S$(printf '%02d' ${SUBJECT}) ${ROI}"
        echo "============================================================"

        python3 check_population_tail_sensitivity_ts.py \
            --subject ${SUBJECT} \
            --roi_name ${ROI} \
            --pyr_pca_type pcaHL \
            --pca_pct_var 95 \
            --max_sessions 30 \
            --nimgs 100

        echo
        echo "Finished S$(printf '%02d' ${SUBJECT}) ${ROI}"
        echo

    done

done

echo
echo "============================================================"
echo "ALL 32 SUBJECT × ROI ANALYSES COMPLETED"
echo "============================================================"