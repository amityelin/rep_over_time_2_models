#!/bin/bash
# Henderson-style PCA for the TS model: separate PCA within each of the 9 higher-order
# feature groups (95% variance, max 150 PCs per group), lower-order features kept raw.
# Output: features/pyramid_texture/PCA_pcaHL_v95_henderson_replication/
# Skips subjects whose concatenated output already exists (safe to re-run after a disconnect).
which python3 && python3 --version
pip install numpy pyrtools

ROOT=/content/drive/MyDrive/V2_drift/
export PYTHONPATH=${ROOT}texturemodel/:${ROOT}texturemodel/code/:${ROOT}texturemodel/code/utils:$PYTHONPATH
cd ${ROOT}texturemodel/code/feature_extraction

debug=0
save_weights=1
which_prf_grid=5
max_pc_to_retain=150
min_pct_var=95
pca_type=pcaHL
sublist=(1 2 3 4 5 6 7 8)

for subject in "${sublist[@]}"; do
    OUT=${ROOT}features/pyramid_texture/PCA_pcaHL_v95_henderson_replication/S${subject}_4ori_4sf_pcaHL_concat_v95_grid5.h5py
    if [ -f "$OUT" ]; then echo "Subject $subject: PCA already exists, skipping"; continue; fi
    echo "======== Subject $subject -- pcaHL ${min_pct_var}% (Henderson groups) ========"
    python3 pca_texture_feats.py \
        --subject $subject --debug $debug \
        --max_pc_to_retain $max_pc_to_retain --min_pct_var $min_pct_var \
        --pca_type $pca_type --which_prf_grid $which_prf_grid --save_weights $save_weights \
    || echo "Subject $subject: FAILED"
done
