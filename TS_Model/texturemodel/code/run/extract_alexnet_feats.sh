#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=8-00:00:00
#SBATCH --output=./sbatch_output/output-%A.out

set -euo pipefail

# ============================================================
# Folders and environment
# ============================================================
mkdir -p ./sbatch_output

ROOT=/content/drive/MyDrive/V2_drift/
LOG_FILE="${ROOT}crossval_texture_S01_V1_pcaHL95_zscore0_log.txt"

which python3
python3 --version

pip install -q numpy pyrtools

export PYTHONPATH="${ROOT}texturemodel/:${ROOT}texturemodel/code/:${ROOT}texturemodel/code/utils:${PYTHONPATH:-}"

cd "${ROOT}texturemodel/code/model_fitting"

echo "Logging to: ${LOG_FILE}"
echo "========== START CROSS-VALIDATION: $(date) ==========" \
    | tee -a "${LOG_FILE}"

# ============================================================
# Configuration
# MUST MATCH THE FITTING SCRIPT
# ============================================================
SUBJECT=1
ROI=V1
DEBUG=0

PCA_TYPE=pcaHL
PCA_PCT=95

PRF_MODEL_NAME=nsd_grid5
PRF_GRID=5

N_ORI=4
N_SF=4

ZSCORE_FEATURES=0
RIDGE=0

# ============================================================
# Session ranges
#
# DEBUG=1:
#   Train session 1 -> Test session 1
#
# DEBUG=0:
#   Full 30 x 30 cross-session validation
# ============================================================
if [ "${DEBUG}" -eq 1 ]; then
    TRAIN_SESSIONS="0"
    VAL_SESSIONS="0"
else
    TRAIN_SESSIONS=$(seq 9 16)
    VAL_SESSIONS=$(seq 0 29)
fi

# ============================================================
# Cross-session validation
# ============================================================
for train_sess in ${TRAIN_SESSIONS}; do

    for val_sess in ${VAL_SESSIONS}; do

        train_display=$((train_sess + 1))
        val_display=$((val_sess + 1))

        echo
        echo "============================================================"
        echo "Train session ${train_display} -> Test session ${val_display}"
        echo "============================================================"

        echo "[START] Train ${train_display} -> Test ${val_display}: $(date)" \
            | tee -a "${LOG_FILE}"

        python3 fit_model.py \
            --subject "${SUBJECT}" \
            --roi_name "${ROI}" \
            --debug "${DEBUG}" \
            --single_sess "${train_sess}" \
            --single_val_sess "${val_sess}" \
            --whole_session_split 1 \
            --average_image_reps 0 \
            --sample_batch_size 1000 \
            --voxel_batch_size 1000 \
            --zscore_features "${ZSCORE_FEATURES}" \
            --use_precomputed_prfs 1 \
            --prfs_model_name "${PRF_MODEL_NAME}" \
            --which_prf_grid "${PRF_GRID}" \
            --fitting_type texture_pyramid \
            --n_ori_pyr "${N_ORI}" \
            --n_sf_pyr "${N_SF}" \
            --group_all_hl_feats 1 \
            --pyr_pca_type "${PCA_TYPE}" \
            --pca_pct_var "${PCA_PCT}" \
            --set_lambda_per_group 0 \
            --ridge "${RIDGE}" \
            --bootstrap_data 0 \
            --shuffle_data 0 \
            --do_tuning 0 \
            --do_sem_disc 0 \
            --do_pyr_varpart 0 \
            --overwrite_sem_disc 1 \
            --from_scratch 0 \
            --do_val 1 \
            2>&1 | tee -a "${LOG_FILE}"

        echo "[END] Train ${train_display} -> Test ${val_display}: $(date)" \
            | tee -a "${LOG_FILE}"

    done

done

echo "========== END CROSS-VALIDATION: $(date) ==========" \
    | tee -a "${LOG_FILE}"