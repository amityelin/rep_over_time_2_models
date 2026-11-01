#!/bin/bash
# Whole-data TS fit with the 49 raw lower-order features only (no higher-order PCs).
# Uses the first 49 columns of the k1 per-group file (--pyr_ll_only 1), so the LL features are identical
# to the k1 model; k1 (58) minus LL-only (49) = contribution of the 9 higher-order PCs.
# Same fitting/evaluation as fit_wholedata_k_per_group.sh. Resumable via DONE markers.
ROOT=/content/drive/MyDrive/V2_drift/          # data (features, betas, fits) -- always on Drive
CODE=${CODE:-${ROOT}texturemodel/}             # code; override to run from a git clone
export PYTHONPATH=${CODE}:${CODE}code/:${CODE}code/utils:$PYTHONPATH
cd ${CODE}code/model_fitting
nvidia-smi > /dev/null 2>&1 && echo "GPU OK" || echo "WARNING: no GPU"

DONE_DIR=${ROOT}texturemodel/wholedata_done/pcaHL_k1_LLonly_ridge0_nsd_grid5; mkdir -p "$DONE_DIR"
n_sessions() { case $1 in 1|2|5|7) echo 40;; 3|6) echo 32;; 4|8) echo 30;; esac; }
OK=(); SKIPPED=(); FAILED=()
for S in 1 2 3 4 5 6 7 8; do
  FEAT=${ROOT}features/pyramid_texture/PCA_pcaHL_k1/S${S}_4ori_4sf_pcaHL_concat_k1_grid5.h5py
  for ROI in V1 V2 V3 hV4; do
    T=LL_S${S}_${ROI}
    echo "===================== $T ====================="
    if [ -f "$DONE_DIR/$T.DONE" ]; then echo "Already done: $T"; OK+=("$T(prev)"); continue; fi
    if [ ! -f "$FEAT" ]; then echo "SKIP $T: missing $FEAT"; SKIPPED+=("$T"); continue; fi
    python3 fit_model.py --subject $S --roi_name $ROI --debug 0 \
      --up_to_sess $(n_sessions $S) --average_image_reps 1 \
      --sample_batch_size 1000 --voxel_batch_size 1000 --zscore_features 1 \
      --use_precomputed_prfs 1 --prfs_model_name nsd_grid5 --which_prf_grid 5 \
      --from_scratch 1 --date_str 0 --overwrite_sem_disc 1 \
      --do_val 1 --do_tuning 0 --do_sem_disc 0 --do_pyr_varpart 0 \
      --fitting_type texture_pyramid --n_ori_pyr 4 --n_sf_pyr 4 \
      --group_all_hl_feats 1 --set_lambda_per_group 0 --bootstrap_data 0 \
      --pyr_pca_type pcaHL --henderson_pca 1 --pyr_pca_tag k1 --pyr_ll_only 1 --ridge 0
    if [ $? -eq 0 ]; then touch "$DONE_DIR/$T.DONE"; OK+=("$T"); else FAILED+=("$T"); fi
  done
done
echo ""; echo "================ SUMMARY ================"
echo "OK      (${#OK[@]}): ${OK[*]}"
echo "SKIPPED (${#SKIPPED[@]}): ${SKIPPED[*]}"
echo "FAILED  (${#FAILED[@]}): ${FAILED[*]}"
