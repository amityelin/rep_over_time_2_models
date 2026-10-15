#!/bin/bash

which python3
python3 --version

pip install numpy
pip install pyrtools

ROOT=/content/drive/MyDrive/V2_drift/
LIB_DIR1=${ROOT}texturemodel/
LIB_DIR2=${ROOT}texturemodel/code/
LIB_DIR3=${LIB_DIR2}/utils

export PYTHONPATH=${LIB_DIR1}:${LIB_DIR2}:${LIB_DIR3}:$PYTHONPATH

echo "PYTHONPATH: $PYTHONPATH"
pip freeze | grep numpy

cd ${ROOT}texturemodel/code/feature_extraction

# ---- Config ----
debug=0
save_weights=1
which_prf_grid=5
max_pc_to_retain=150

# Henderson-style PCA threshold
min_pct_var=95

# Subject
sublist=(1)

# Only pcaHL
pca_type=pcaHL

for subject in "${sublist[@]}"
do
    echo "========================================"
    echo "Subject $subject — pcaHL 95%"
    echo "========================================"

    python3 pca_texture_feats.py \
        --subject $subject \
        --debug $debug \
        --max_pc_to_retain $max_pc_to_retain \
        --min_pct_var $min_pct_var \
        --pca_type $pca_type \
        --which_prf_grid $which_prf_grid \
        --save_weights $save_weights
done