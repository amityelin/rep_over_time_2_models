#!/bin/bash

set -euo pipefail

ROOT=/content/drive/MyDrive/V2_drift/

export PYTHONPATH="${ROOT}texturemodel/:${ROOT}texturemodel/code/:${ROOT}texturemodel/code/utils:${PYTHONPATH:-}"

cd "${ROOT}texturemodel/code/model_fitting"

which python3
python3 --version

python3 fit_cross_session_matrix_fast.py \
    --subject 1 \
    --roi_name V1 \
    --debug 0 \
    --up_to_sess 30 \
    --whole_session_split 1 \
    --average_image_reps 0 \
    --sample_batch_size 1000 \
    --voxel_batch_size 1000 \
    --zscore_features 0 \
    --zscore_betas_within_sess 0 \
    --use_precomputed_prfs 1 \
    --prfs_model_name nsd_grid5 \
    --which_prf_grid 5 \
    --fitting_type texture_pyramid \
    --n_ori_pyr 4 \
    --n_sf_pyr 4 \
    --group_all_hl_feats 1 \
    --pyr_pca_type pcaLL \
    --pyr_fixed_n_components 18 \
    --pca_pct_var 95 \
    --set_lambda_per_group 0 \
    --ridge 0 \
    --do_val 1 \
    --do_corrcoef 1 \
    --do_tuning 0 \
    --do_sem_disc 0 \
    --do_pyr_varpart 0 \
    --trial_subset all