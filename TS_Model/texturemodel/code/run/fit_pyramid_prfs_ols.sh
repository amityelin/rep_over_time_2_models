#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=8-00:00:00
#SBATCH --output=./sbatch_output/output-%A.out

# environment
which python3 && python3 --version
pip install numpy pyrtools

ROOT=/content/drive/MyDrive/V2_drift/
export PYTHONPATH=${ROOT}texturemodel/:${ROOT}texturemodel/code/:${ROOT}texturemodel/code/utils:$PYTHONPATH
cd ${ROOT}texturemodel/code/model_fitting

# =========================
# EDIT THESE EACH RUN
# =========================
SUBJECT=1
ROI=V1
PCA_PCT=40
PCA_TYPE=pcaHL
RIDGE=0

python3 fit_model.py \
  --subject $SUBJECT \
  --roi_name $ROI \
  --debug 1 \
  --up_to_sess 2 \
  --average_image_reps 1 \
  --sample_batch_size 1000 \
  --voxel_batch_size 1000 \
  --zscore_features 1 \
  --use_precomputed_prfs 1 \
  --prfs_model_name nsd_grid5 \
  --which_prf_grid 5 \
  --from_scratch 1 \
  --date_str 0 \
  --overwrite_sem_disc 1 \
  --do_val 1 \
  --do_tuning 0 \
  --do_sem_disc 0 \
  --do_pyr_varpart 0 \
  --fitting_type texture_pyramid \
  --n_ori_pyr 4 \
  --n_sf_pyr 4 \
  --group_all_hl_feats 1 \
  --set_lambda_per_group 0 \
  --bootstrap_data 0 \
  --pca_pct_var $PCA_PCT \
  --pyr_pca_type $PCA_TYPE \
  --ridge $RIDGE