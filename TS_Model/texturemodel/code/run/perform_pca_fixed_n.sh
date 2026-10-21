#!/bin/bash
# Fixed-size TS feature sets with the same number of predictors as the SP model (18):
#   pcaLL  = PCA over the 49 lower-order texture features only, first 18 PCs
#   pcaAll = PCA over all 641 texture features, first 18 PCs
# Output: features/pyramid_texture/PCA_<type>_n18/S<s>_4ori_4sf_<type>_concat_n18_grid5.h5py
# The python script is resumable (checkpoints per pRF); finished subject x type combinations are skipped.
which python3 && python3 --version
pip install numpy pyrtools

ROOT=/content/drive/MyDrive/V2_drift/
export PYTHONPATH=${ROOT}texturemodel/:${ROOT}texturemodel/code/:${ROOT}texturemodel/code/utils:$PYTHONPATH
cd ${ROOT}texturemodel/code/feature_extraction

N_COMP=18
SUBJECTS="1 2 3 4 5 6 7 8"
PCA_TYPES="pcaLL pcaAll"

for PCA_TYPE in $PCA_TYPES; do
  for SUBJECT in $SUBJECTS; do
    OUT=${ROOT}features/pyramid_texture/PCA_${PCA_TYPE}_n${N_COMP}/S${SUBJECT}_4ori_4sf_${PCA_TYPE}_concat_n${N_COMP}_grid5.h5py
    if [ -f "$OUT" ]; then echo "S${SUBJECT} ${PCA_TYPE}: already exists, skipping"; continue; fi
    echo "======== S${SUBJECT}  ${PCA_TYPE}  n=${N_COMP} ========"
    python3 pca_texture_feats_pcaAll.py \
      --subject ${SUBJECT} --pca_type ${PCA_TYPE} --fixed_n_components ${N_COMP} \
      --debug 0 --max_pc_to_retain 0 --which_prf_grid 5 --thresholds 95 \
      --checkpoint_every 10 --image_chunk_size 1000 --compress 0 \
      --overwrite_pca 0 --overwrite_outputs 0 --overwrite_concat 0 --build_concat 1 \
    || echo "S${SUBJECT} ${PCA_TYPE}: FAILED"
  done
done
