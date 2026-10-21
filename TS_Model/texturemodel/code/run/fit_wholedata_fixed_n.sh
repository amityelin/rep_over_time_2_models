#!/bin/bash
# Whole-data TS fits with 18 predictors (same as SP): pcaLL18 and pcaAll18, 8 subjects x V1-hV4.
# Same fitting/evaluation as fit_replication_all_subjects.sh (OLS, all sessions, averaged repetitions,
# validation on the shared-1000 images). Resumable via DONE markers; pre-flight check for inputs.
which python3 && python3 --version
pip install numpy pyrtools

ROOT=/content/drive/MyDrive/V2_drift/
export PYTHONPATH=${ROOT}texturemodel/:${ROOT}texturemodel/code/:${ROOT}texturemodel/code/utils:$PYTHONPATH
cd ${ROOT}texturemodel/code/model_fitting

echo "========== GPU CHECK =========="
nvidia-smi || echo "WARNING: no GPU -- Runtime > Change runtime type > GPU"
echo "==============================="

N_COMP=18
PCA_TYPES="pcaLL pcaAll"
RIDGE=0
PRF_MODEL=nsd_grid5
SUBJECTS="1 2 3 4 5 6 7 8"
ROIS="V1 V2 V3 hV4"

PRF_DIR=${ROOT}prf_indices
n_sessions() { case $1 in 1|2|5|7) echo 40;; 3|6) echo 32;; 4|8) echo 30;; esac; }
OK=(); SKIPPED=(); FAILED=()

for PCA_TYPE in $PCA_TYPES; do
  DONE_DIR=${ROOT}texturemodel/wholedata_done/${PCA_TYPE}_n${N_COMP}_ridge${RIDGE}_${PRF_MODEL}
  mkdir -p "$DONE_DIR"; LOG=${DONE_DIR}/run_log.txt
  for SUBJECT in $SUBJECTS; do
    UP_TO_SESS=$(n_sessions $SUBJECT)
    FEAT_FILE=${ROOT}features/pyramid_texture/PCA_${PCA_TYPE}_n${N_COMP}/S${SUBJECT}_4ori_4sf_${PCA_TYPE}_concat_n${N_COMP}_grid5.h5py
    for ROI in $ROIS; do
      TAG="${PCA_TYPE}${N_COMP}_S${SUBJECT}_${ROI}"
      PRF_FILE=$(printf "%s/subj%02d_normalized_final_best_prf_indices_%s.npy" "$PRF_DIR" "$SUBJECT" "$ROI")
      echo "======================================"
      echo "${TAG}  up_to_sess=${UP_TO_SESS}"
      echo "======================================"
      if [ -f "${DONE_DIR}/${TAG}.DONE" ]; then echo "Already done: ${TAG}"; OK+=("$TAG(prev)"); continue; fi
      missing=""
      [ -f "$FEAT_FILE" ] || missing="$missing\n  features: $FEAT_FILE"
      [ -f "$PRF_FILE" ]  || missing="$missing\n  pRFs:     $PRF_FILE"
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
        --pyr_pca_type $PCA_TYPE --pyr_fixed_n_components $N_COMP \
        --ridge $RIDGE --henderson_pca 0
      status=$?; secs=$(( $(date +%s) - start ))
      if [ $status -eq 0 ]; then
        touch "${DONE_DIR}/${TAG}.DONE"; echo "OK   ${TAG} (${secs}s)" | tee -a "$LOG"; OK+=("$TAG")
      else
        echo "FAIL ${TAG} (exit ${status}, ${secs}s)" | tee -a "$LOG"; FAILED+=("$TAG")
      fi
    done
  done
done

echo ""
echo "================ SUMMARY ================"
echo "OK      (${#OK[@]}): ${OK[*]}"
echo "SKIPPED (${#SKIPPED[@]}): ${SKIPPED[*]}"
echo "FAILED  (${#FAILED[@]}): ${FAILED[*]}"
