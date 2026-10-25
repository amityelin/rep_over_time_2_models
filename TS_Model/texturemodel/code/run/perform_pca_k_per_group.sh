#!/bin/bash
# Per-group PCA with a FIXED number of PCs per group, features z-scored (training images) before PCA.
#   pcaHL: 49 raw lower-order features + K PCs from each of the 9 higher-order groups  (K=2 -> 67 columns)
# Same grouping as Henderson et al. (2023); differs only in K fixed (instead of 95% variance) and z-scoring.
# Output: features/pyramid_texture/PCA_pcaHL_k<K>z/S<s>_4ori_4sf_pcaHL_concat_k<K>z_grid5.h5py (+ featurelabels .npy)
which python3 && python3 --version
pip install numpy pyrtools

ROOT=/content/drive/MyDrive/V2_drift/          # data (features, betas, fits) -- always on Drive
CODE=${CODE:-${ROOT}texturemodel/}             # code; override to run from a git clone, e.g.
                                               #   CODE=/content/rep_over_time_2_models/TS_Model/texturemodel/ bash <script>
export PYTHONPATH=${CODE}:${CODE}code/:${CODE}code/utils:$PYTHONPATH
cd ${CODE}code/feature_extraction

K=2
ZSCORE=0          # 1 = z-score each feature before PCA (tag k<K>z); 0 = as Henderson (tag k<K>)
TAG=k${K}$([ "$ZSCORE" = "1" ] && echo z)
SUBJECTS="1 2 3 4 5 6 7 8"

for SUBJECT in $SUBJECTS; do
  OUT=${ROOT}features/pyramid_texture/PCA_pcaHL_${TAG}/S${SUBJECT}_4ori_4sf_pcaHL_concat_${TAG}_grid5.h5py
  if [ -f "$OUT" ]; then echo "S${SUBJECT}: already exists, skipping"; continue; fi
  echo "======== S${SUBJECT}  pcaHL  ${TAG} ========"
  python3 pca_texture_feats.py \
    --subject ${SUBJECT} --pca_type pcaHL --which_prf_grid 5 --debug 0 \
    --fixed_n_per_group ${K} --zscore_before_pca ${ZSCORE} --save_weights 1 \
  || echo "S${SUBJECT}: FAILED"
done
