#!/bin/bash
# Session-wise (30 x 30 cross-session) TS drift fits with the k1 per-group PCA:
# 49 raw lower-order features + 1 PC from each of the 9 higher-order groups = 58 predictors.
# All other settings are identical to the existing drift runs (fit_cross_session_matrix_fast_*.sh):
# OLS, 30 sessions, whole-session split, single trials, raw betas, NSD pRFs aligned to grid 5.
# Output: model_fits/S0X/FAST_CROSS_SESSION/S0X_<ROI>_texture_pyramid_OLS_4ori_4sf_pcaHL_HendersonGroups_<ROI>_k1_
#         use_nsd_grid5_pRFs_30x30_cross_session.h5py
# The python script checkpoints per training session; this wrapper also skips finished subject x ROI pairs.
# Split the work across Colab sessions with e.g.  SUBJECTS="1 2" ROIS="V1 V2 V3 hV4" CODE=... bash <this script>
ROOT=/content/drive/MyDrive/V2_drift/          # data (features, betas, fits) -- always on Drive
CODE=${CODE:-${ROOT}texturemodel/}             # code; override to run from a git clone
export PYTHONPATH=${CODE}:${CODE}code/:${CODE}code/utils:$PYTHONPATH
cd ${CODE}code/model_fitting
nvidia-smi > /dev/null 2>&1 && echo "GPU OK" || echo "WARNING: no GPU"

SUBJECTS=${SUBJECTS:-"1 2 3 4 5 6 7 8"}
ROIS=${ROIS:-"V1 V2 V3 hV4"}
TAG=k1
DONE_DIR=${ROOT}texturemodel/crosssession_done/pcaHL_${TAG}_nsd_grid5; mkdir -p "$DONE_DIR"
OK=(); SKIPPED=(); FAILED=()
for S in $SUBJECTS; do
  FEAT=${ROOT}features/pyramid_texture/PCA_pcaHL_${TAG}/S${S}_4ori_4sf_pcaHL_concat_${TAG}_grid5.h5py
  for ROI in $ROIS; do
    T=${TAG}_S${S}_${ROI}
    echo "===================== cross-session $T ====================="
    if [ -f "$DONE_DIR/$T.DONE" ]; then echo "Already done: $T"; OK+=("$T(prev)"); continue; fi
    if [ ! -f "$FEAT" ]; then echo "SKIP $T: missing $FEAT"; SKIPPED+=("$T"); continue; fi
    python3 fit_cross_session_matrix_fast.py \
      --subject $S --roi_name $ROI --debug 0 \
      --up_to_sess 30 --whole_session_split 1 --average_image_reps 0 \
      --sample_batch_size 1000 --voxel_batch_size 1000 \
      --zscore_features 0 --zscore_betas_within_sess 0 \
      --use_precomputed_prfs 1 --prfs_model_name nsd_grid5 --which_prf_grid 5 \
      --fitting_type texture_pyramid --n_ori_pyr 4 --n_sf_pyr 4 \
      --group_all_hl_feats 1 --set_lambda_per_group 0 --ridge 0 \
      --pyr_pca_type pcaHL --henderson_pca 1 --pyr_pca_tag ${TAG} \
      --do_val 1 --do_corrcoef 1 --do_tuning 0 --do_sem_disc 0 --do_pyr_varpart 0 \
      --trial_subset all
    if [ $? -eq 0 ]; then touch "$DONE_DIR/$T.DONE"; OK+=("$T"); else FAILED+=("$T"); fi
  done
done
echo ""; echo "================ SUMMARY ================"
echo "OK      (${#OK[@]}): ${OK[*]}"
echo "SKIPPED (${#SKIPPED[@]}): ${SKIPPED[*]}"
echo "FAILED  (${#FAILED[@]}): ${FAILED[*]}"
