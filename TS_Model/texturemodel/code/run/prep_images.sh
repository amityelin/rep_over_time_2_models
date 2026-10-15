```bash
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

debug=0
which_prf_grid=5
max_pc_to_retain=0

# Save progress every N pRFs.
# Use 1 for maximum safety, or 5-10 for slightly faster writing.
checkpoint_every=5

# Number of images written per concat checkpoint.
image_chunk_size=500

# Keep these at 0 when resuming an interrupted run.
overwrite_pca=0
overwrite_outputs=0
overwrite_concat=0

# Create the final concatenated feature files.
build_concat=1

# gzip compression for HDF5 files.
compress=1

thresholds_hl="0,40,60,80,95"
thresholds_all="40,60,80,95"

sublist=(1)

for subject in "${sublist[@]}"
do
    for pca_type in pcaHL pcaAll
    do
        if [ "$pca_type" = "pcaHL" ]; then
            thresholds=$thresholds_hl
            mode_text="raw low-level + one PCA on all high-level features"
        elif [ "$pca_type" = "pcaAll" ]; then
            thresholds=$thresholds_all
            mode_text="one PCA on all texture features"
        fi

        echo "========================================"
        echo "Subject: $subject"
        echo "PCA type: $pca_type"
        echo "Thresholds: $thresholds"
        echo "Mode: $mode_text"
        echo "Checkpoint every: $checkpoint_every pRFs"
        echo "Image chunk size: $image_chunk_size"
        echo "========================================"

        python3 pca_texture_feats_multi_threshold_resumable.py \
            --subject $subject \
            --debug $debug \
            --max_pc_to_retain $max_pc_to_retain \
            --thresholds $thresholds \
            --pca_type $pca_type \
            --which_prf_grid $which_prf_grid \
            --checkpoint_every $checkpoint_every \
            --image_chunk_size $image_chunk_size \
            --compress $compress \
            --overwrite_pca $overwrite_pca \
            --overwrite_outputs $overwrite_outputs \
            --overwrite_concat $overwrite_concat \
            --build_concat $build_concat

    done
done
```
