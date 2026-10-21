#!/bin/bash
# Whole-data TS fits for all subjects x ROIs, Henderson-replication configuration
# (PCA within Henderson's 9 higher-order groups, 95% variance; OLS; all available sessions).
# Resumable: writes a DONE marker per subject x ROI; pre-flight check skips runs with missing inputs.
# Used to produce the Fig. 3-style validation (TS_Model/notebooks/fig3_TS_SP_comparison.ipynb).
which python3 && python3 --version
pip install numpy pyrtools

ROOT=/content/drive/MyDrive/V2_drift/
export PYTHONPATH=${ROOT}texturemodel/:${ROOT}texturemodel/code/:${ROOT}texturemodel/code/utils:$PYTHONPATH
cd ${ROOT}texturemodel/code/model_fitting

echo "========== GPU CHECK =========="
nvidia-smi || echo "WARNING: no GPU -- Runtime > Change runtime type > GPU"
echo "==============================="

# =========================
# SETTINGS
# =========================
PCA_PCT=95
PCA_TYPE=pcaHL
RIDGE=0
PRF_MODEL=nsd_grid5          # nsd_grid5 = NSD mapping-task pRFs aligned to grid 5; henderson_grid5 = Henderson's fitted pRFs

SUBJECTS="1 2 3 4 5 6 7 8"
ROIS="V1 V2 V3 hV4"
SKIP=""                      # e.g. "S1_V2" to skip runs done elsewhere

FEAT_DIR=${ROOT}features/pyramid_texture/PCA_${PCA_TYPE}_v${PCA_PCT}_henderson_replication
PRF_DIR=${ROOT}prf_indices
DONE_DIR=${ROOT}texturemodel/replication_done/ridge${RIDGE}_${PRF_MODEL}
mkdir -p "$DONE_DIR"
LOG=${DONE_DIR}/run_log.txt

# sessions per subject (NSD): S1,S2,S5,S7=40 ; S3,S6=32 ; S4,S8=30
n_sessions() { case $1 in 1|2|5|7) echo 40;; 3|6) echo 32;; 4|8) echo 30;; esac; }

OK=(); SKIPPED=(); FAILED=()

for SUBJECT in $SUBJECTS; do
  UP_TO_SESS=$(n_sessions $SUBJECT)
  FEAT_FILE=${FEAT_DIR}/S${SUBJECT}_4ori_4sf_${PCA_TYPE}_concat_v${PCA_PCT}_grid5.h5py
  LABEL_FILE=${FEAT_DIR}/S${SUBJECT}_4ori_4sf_featurelabels_${PCA_TYPE}_v${PCA_PCT}_grid5.npy

  for ROI in $ROIS; do
    TAG="S${SUBJECT}_${ROI}"
    if [ "$PRF_MODEL" = "nsd_grid5" ]; then
      PRF_FILE=$(printf "%s/subj%02d_normalized_final_best_prf_indices_%s.npy" "$PRF_DIR" "$SUBJECT" "$ROI")
    else
      PRF_FILE=$(printf "%s/subj%02d_henderson_best_prf_indices_%s.npy" "$PRF_DIR" "$SUBJECT" "$ROI")
    fi

    echo "======================================"
    echo "${TAG}  up_to_sess=${UP_TO_SESS}  PRF_MODEL=${PRF_MODEL}  RIDGE=${RIDGE}"
    echo "======================================"

    if [[ " $SKIP " == *" $TAG "* ]]; then echo "In SKIP list: ${TAG}"; continue; fi
    if [ -f "${DONE_DIR}/${TAG}.DONE" ]; then echo "Already done: ${TAG}"; OK+=("$TAG(prev)"); continue; fi

    missing=""
    [ -f "$FEAT_FILE" ]  || missing="$missing\n  features: $FEAT_FILE"
    [ -f "$LABEL_FILE" ] || missing="$missing\n  labels:   $LABEL_FILE"
    [ -f "$PRF_FILE" ]   || missing="$missing\n  pRFs:     $PRF_FILE"
    if [ -n "$missing" ]; then
      echo -e "SKIP ${TAG}: missing input(s):$missing" | tee -a "$LOG"; SKIPPED+=("$TAG"); continue
    fi

    start=$(date +%s)
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

    status=$?; secs=$(( $(date +%s) - start ))
    if [ $status -eq 0 ]; then
      touch "${DONE_DIR}/${TAG}.DONE"; echo "OK   ${TAG} (${secs}s)" | tee -a "$LOG"; OK+=("$TAG")
    else
      echo "FAIL ${TAG} (exit ${status}, ${secs}s)" | tee -a "$LOG"; FAILED+=("$TAG")
    fi
  done
done

echo ""
echo "================ SUMMARY ================"
echo "OK      (${#OK[@]}): ${OK[*]}"
echo "SKIPPED (${#SKIPPED[@]}): ${SKIPPED[*]}"
echo "FAILED  (${#FAILED[@]}): ${FAILED[*]}"
echo "Log: $LOG"
