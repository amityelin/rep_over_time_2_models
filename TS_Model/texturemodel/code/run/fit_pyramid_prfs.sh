#!/bin/bash
# Single-run TS fit, Henderson-replication configuration (the script used for the S01 V2 validation).
# NOTE: the previous file at this path was an unmodified copy of Henderson's fit_pyramid_fixsigma.sh
# (her cluster paths, fixed-sigma loop) and was never run in this project.
which python3 && python3 --version
pip install numpy pyrtools

ROOT=/content/drive/MyDrive/V2_drift/
export PYTHONPATH=${ROOT}texturemodel/:${ROOT}texturemodel/code/:${ROOT}texturemodel/code/utils:$PYTHONPATH
cd ${ROOT}texturemodel/code/model_fitting

echo "========== GPU CHECK =========="
nvidia-smi
echo "==============================="

# =========================
# EDIT THESE EACH BATCH
# =========================
PCA_PCT=95
PCA_TYPE=pcaHL
RIDGE=0
PRF_MODEL=henderson_grid5   # henderson_grid5 or nsd_grid5
UP_TO_SESS=40               # S1,S2,S5,S7=40 ; S3,S6=32 ; S4,S8=30

SUBJECTS="1"
ROIS="V2"

for SUBJECT in $SUBJECTS; do
  for ROI in $ROIS; do
    echo "======================================"
    echo "Running SUBJECT=${SUBJECT}, ROI=${ROI}, PCA=${PCA_TYPE}${PCA_PCT}, PRF_MODEL=${PRF_MODEL}"
    echo "======================================"
    python3 fit_model.py \
      --subject $SUBJECT --roi_name $ROI --debug 0 \
      --up_to_sess $UP_TO_SESS --average_image_reps 1 \
      --sample_batch_size 1000 --voxel_batch_size 1000 \
      --zscore_features 1 \
      --use_precomputed_prfs 1 --prfs_model_name $PRF_MODEL --which_prf_grid 5 \
      --from_scratch 1 --date_str 0 --overwrite_sem_disc 1 \
      --do_val 1 --do_tuning 0 --do_sem_disc 0 --do_pyr_varpart 0 \
      --fitting_type texture_pyramid --n_ori_pyr 4 --n_sf_pyr 4 \
      --group_all_hl_feats 1 --set_lambda_per_group 0 --bootstrap_data 0 \
      --pca_pct_var $PCA_PCT --pyr_pca_type $PCA_TYPE \
      --ridge $RIDGE --henderson_pca 1
  done
done
