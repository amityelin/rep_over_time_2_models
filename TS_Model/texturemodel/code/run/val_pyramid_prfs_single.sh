#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=8-00:00:00
#SBATCH --output=./sbatch_output/output-%A.out

set -euo pipefail

# ------------------------------------------------------------
# Folders and environment
# ------------------------------------------------------------
mkdir -p ./sbatch_output

ROOT=/content/drive/MyDrive/V2_drift/
LOG_FILE="${ROOT}crossval_log.txt"

which python3
python3 --version

pip install -q numpy pyrtools

export PYTHONPATH="${ROOT}texturemodel/:${ROOT}texturemodel/code/:${ROOT}texturemodel/code/utils:${PYTHONPATH:-}"

cd "${ROOT}texturemodel/code/model_fitting"

echo "Logging to: ${LOG_FILE}"
echo "========== START CROSS-VALIDATION: $(date) ==========" >> "${LOG_FILE}"

# ------------------------------------------------------------
# Analysis settings
# Must match the single-session training script
# ------------------------------------------------------------
SUBJECT=1
ROI=V1

PCA_TYPE=pcaHL
PCA_PCT=95

N_ORI=4
N_SF=4

PRF_MODEL_NAME=nsd_grid5
PRF_GRID=5

# ------------------------------------------------------------
# Cross-session validation
#
# For each training session:
#     load its previously fitted model
#     validate it on every other session
#
# Sessions are zero-indexed:
#     0 = session 1
#     29 = session 30
# ------------------------------------------------------------
for train_sess in {0..29}; do

    for val_sess in {0..29}; do

        # Do not use the same session for training and validation
        #if [ "${train_sess}" -eq "${val_sess}" ]; then
        #    continue
        #fi

        train_number=$((train_sess + 1))
        val_number=$((val_sess + 1))

        echo "=================================================="
        echo "TRAIN session ${train_number} -> VALIDATE session ${val_number}"
        echo "=================================================="

        echo \
            "[TRAIN ${train_number} -> VAL ${val_number} START] $(date)" \
            >> "${LOG_FILE}"

        pair_start=$(date +%s)

        python3 fit_model.py \
            --subject "${SUBJECT}" \
            --roi_name "${ROI}" \
            --debug 0 \
            --single_sess "${train_sess}" \
            --single_val_sess "${val_sess}" \
            --whole_session_split 1 \
            --average_image_reps 0 \
            --sample_batch_size 1000 \
            --voxel_batch_size 1000 \
            --fitting_type texture_pyramid \
            --n_ori_pyr "${N_ORI}" \
            --n_sf_pyr "${N_SF}" \
            --group_all_hl_feats 1 \
            --pyr_pca_type "${PCA_TYPE}" \
            --pca_pct_var "${PCA_PCT}" \
            --zscore_features 0 \
            --zscore_betas_within_sess 0 \
            --ridge 0 \
            --set_lambda_per_group 0 \
            --use_precomputed_prfs 1 \
            --prfs_model_name "${PRF_MODEL_NAME}" \
            --which_prf_grid "${PRF_GRID}" \
            --shuffle_data 0 \
            --bootstrap_data 0 \
            --do_tuning 0 \
            --do_sem_disc 0 \
            --do_pyr_varpart 0 \
            --overwrite_sem_disc 1 \
            --from_scratch 0 \
            --date_str 0 \
            --do_val 1 \
            2>&1 | tee -a "${LOG_FILE}"

        pair_end=$(date +%s)

        echo \
            "[TRAIN ${train_number} -> VAL ${val_number} END] $(date) | Elapsed: $((pair_end - pair_start))s" \
            >> "${LOG_FILE}"

    done

done

echo "========== END CROSS-VALIDATION: $(date) ==========" >> "${LOG_FILE}"