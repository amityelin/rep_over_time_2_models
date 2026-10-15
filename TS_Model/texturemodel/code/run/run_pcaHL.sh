#!/bin/bash

set -e

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

# ============================================================
# EDIT THESE VALUES
# ============================================================

SUBJECT=1

DEBUG=0
WHICH_PRF_GRID=5

# ============================================================
# FIXED PCA CONFIGURATION
# ============================================================

# pcaAll = PCA jointly across all 641 texture features
# pcaLL for pca only on low level features
PCA_TYPE=pcaAll

# Keep exactly the first 18 principal components
FIXED_N_COMPONENTS=18

# Not used for fixed-component PCA
MAX_PC_TO_RETAIN=0

# Dummy threshold.
# In fixed-component mode the Python script ignores the variance
# threshold and retains FIXED_N_COMPONENTS PCs.
THRESHOLDS="95"

# ============================================================
# PIPELINE SETTINGS
# ============================================================

CHECKPOINT_EVERY=10
IMAGE_CHUNK_SIZE=1000

# 0 is faster; 1 produces smaller files.
COMPRESS=0

# Keep these at 0 for normal runs and resuming.
OVERWRITE_PCA=0
OVERWRITE_OUTPUTS=0
OVERWRITE_CONCAT=0

# Build final concat file after PCA outputs are ready.
BUILD_CONCAT=1

echo "============================================================"
echo "Running fixed-component PCA pipeline"
echo "Subject: ${SUBJECT}"
echo "PCA type: ${PCA_TYPE}"
echo "Fixed number of PCs: ${FIXED_N_COMPONENTS}"
echo "Checkpoint every: ${CHECKPOINT_EVERY} pRFs"
echo "Image chunk size: ${IMAGE_CHUNK_SIZE}"
echo "Compression: ${COMPRESS}"
echo "============================================================"

python3 pca_texture_feats_pcaAll.py \
    --subject ${SUBJECT} \
    --pca_type ${PCA_TYPE} \
    --fixed_n_components ${FIXED_N_COMPONENTS} \
    --debug ${DEBUG} \
    --max_pc_to_retain ${MAX_PC_TO_RETAIN} \
    --which_prf_grid ${WHICH_PRF_GRID} \
    --thresholds ${THRESHOLDS} \
    --checkpoint_every ${CHECKPOINT_EVERY} \
    --image_chunk_size ${IMAGE_CHUNK_SIZE} \
    --compress ${COMPRESS} \
    --overwrite_pca ${OVERWRITE_PCA} \
    --overwrite_outputs ${OVERWRITE_OUTPUTS} \
    --overwrite_concat ${OVERWRITE_CONCAT} \
    --build_concat ${BUILD_CONCAT}