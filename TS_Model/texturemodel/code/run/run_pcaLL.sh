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

SUBJECT=8

DEBUG=0
WHICH_PRF_GRID=5
MAX_PC_TO_RETAIN=0

# pcaHL:
# 0%  = raw low-level features only
# 40% = raw low-level + HL PCs explaining 40%
# 60% = raw low-level + HL PCs explaining 60%
# 80% = raw low-level + HL PCs explaining 80%
# 95% = raw low-level + HL PCs explaining 95%
THRESHOLDS="0,40,60,80,95"

# Flush every 10 pRFs.
# Set to 5 for more safety or 1 for maximum safety.
CHECKPOINT_EVERY=10

# Larger chunks reduce open/write overhead during concatenation.
IMAGE_CHUNK_SIZE=1000

# 0 is faster; 1 produces smaller files.
COMPRESS=0

# Keep these at 0 for normal runs and resuming.
OVERWRITE_PCA=0
OVERWRITE_OUTPUTS=0
OVERWRITE_CONCAT=0

# 1 builds final concat files after PCA outputs are ready.
# Use 0 to finish PCA first, then rerun later with 1.
BUILD_CONCAT=1

echo "============================================================"
echo "Running dedicated pcaHL pipeline"
echo "Subject: ${SUBJECT}"
echo "Thresholds: ${THRESHOLDS}"
echo "Checkpoint every: ${CHECKPOINT_EVERY} pRFs"
echo "Image chunk size: ${IMAGE_CHUNK_SIZE}"
echo "Compression: ${COMPRESS}"
echo "============================================================"

python3 pca_texture_feats_pcaHL.py \
    --subject ${SUBJECT} \
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