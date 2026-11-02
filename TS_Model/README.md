# TS_Model: texture-statistics encoding model

Adaptation of the texture-statistics (TS) encoding model of Henderson et al. (2023)
([code](https://github.com/mmhenderson/texturemodel), J. Neurosci. 43:4144) for the longitudinal
representational-drift analysis. `texturemodel/` is a modified copy of her repository; `notebooks/` holds
the Colab notebooks used for fitting, diagnostics and figures. Cross-model analyses (SP and TS together) are in
`../analysis/`.

## Main changes relative to Henderson's code

| Stage | Henderson | This project |
|---|---|---|
| Images | 240 x 240 px, PIL bilinear | 256 x 256 px, red fixation marker added before grayscale (BT.709) |
| Feature extraction | one pass | two halves (`extract_pyramid_texture_features[_second].py`), merged by summation |
| pRFs | fitted jointly with the model (grid 5) | NSD mapping-task pRFs aligned to grid 5 (`notebooks/pRF fits.ipynb`) |
| PCA | separate PCA per higher-order group, 95% variance | see "PCA configurations" below |
| Regression | banded ridge | OLS |
| Fitting | one fit on all sessions | one fit per session + cross-session validation (`fit_cross_session_matrix_fast.py`); whole-data fits for validation |

## PCA configurations

All configurations keep the 49 lower-order (LL) statistics as raw features and reduce the 592 higher-order (HL)
statistics. PCA is run on unnormalized features, as in Henderson's code.

| Configuration | HL reduction | Predictors | Folder / fitting flags | Status |
|---|---|---|---|---|
| Joint pcaHL95 (original drift config) | one PCA over all 592 HL features, 95% | 49 + 1-5 = 54 | `PCA_pcaHL_v95/`; `--pyr_pca_type pcaHL --pca_pct_var 95` | HL PCs are ~100% low-pass autocorrelation; adds nothing over LL-only |
| Henderson groups, 95% (validation) | separate PCA per HL group, 95% | ~82-125 (varies by pRF) | `PCA_pcaHL_v95_henderson_replication/`; `--henderson_pca 1` | used to validate against Henderson |
| **k1: 1 PC per HL group** | separate PCA per HL group, 1 PC each | **49 + 9 = 58** | `PCA_pcaHL_k1/`; `--henderson_pca 1 --pyr_pca_tag k1` | **proposed drift configuration** |
| k2: 2 PCs per HL group | as k1, 2 PCs each | 49 + 18 = 67 | `PCA_pcaHL_k2/`; `--pyr_pca_tag k2` | +0.002 R2 over k1 |
| LL only (control) | none | 49 | first 49 columns of the k1 file; `--pyr_pca_tag k1 --pyr_ll_only 1` | isolates the HL contribution |
| pcaLL18 / pcaAll18 | joint PCA, 18 PCs | 18 | `PCA_pcaLL_n18/`, `PCA_pcaAll_n18/`; `--pyr_fixed_n_components 18` | discarded (see below) |

**Why the joint PCA fails:** on unnormalized features the retained PCs follow raw scale, not content.
pcaLL18: 78% of retained variance from marginal statistics, 5% from the 16 orientation x SF energies;
pcaAll18: 82% from low-pass autocorrelations; joint pcaHL95: the HL PCs are fully explained by the low-pass
autocorrelation group, so the groups emphasized by Henderson for V2-V4 are effectively absent. Separate PCA per
group avoids this, because every group gets its own components. `pca_texture_feats.py` options:
`--fixed_n_per_group K` (keep K PCs per group) and `--zscore_before_pca 1` (z-score each feature on the training
images before PCA; tag `k<K>z`, not used by default).

## Validation against Henderson et al. (2023)

`notebooks/fig3_TS_SP_comparison.ipynb` (Section 4) reproduces her Fig. 3B metric: mean validation R2 over ROI
voxels with noise ceiling > 0.01 (pRF-mapping ROIs, category-ROI overlap removed), whole-data fits (all sessions,
averaged repetitions, validation on the shared-1000 images), 8 subjects. OLS voxels with R2 < -1 are excluded and
counted.

| Model | V1 | V2 | V3 | hV4 |
|---|---|---|---|---|
| Henderson et al. TS (ridge, her pRFs) | 0.183 | 0.124 | 0.083 | 0.061 |
| TS, Henderson groups 95%, **her pRFs** | 0.178 | 0.121 | 0.079 | 0.057 |
| TS, Henderson groups 95%, NSD pRFs | 0.154 | 0.101 | 0.063 | 0.043 |
| TS k2 (67), NSD pRFs | 0.156 | 0.103 | 0.066 | 0.046 |
| **TS k1 (58), NSD pRFs** | **0.154** | **0.101** | **0.065** | **0.046** |
| TS LL only (49), NSD pRFs | 0.148 | 0.095 | 0.061 | 0.042 |
| SP 4 ori x 4 SF (18), NSD pRFs | 0.146 | 0.097 | 0.066 | 0.048 |

- **The implementation reproduces Henderson's model:** 98 / 98 / 95 / 94% of her R2 with her pRFs. With
  identical pRFs, our fits differ only by OLS vs ridge (2-6%); on voxels with the same pRF the two pipelines give
  identical R2.
- **NSD pRFs cost 14 / 17 / 20 / 24%** of her R2 (grid pRFs chosen by alignment to the independent NSD mapping
  pRFs vs. pRFs chosen to maximize the TS fit).
- **k1 keeps the full model's performance with 58 predictors** and is better than the 95% model in V3 (+3%) and
  hV4 (+7%), where the larger model overfits.
- **The HL groups add unique variance:** k1 > LL only in 8/8 subjects in every ROI (+3.9 / +6.8 / +6.4 / +8.5%).
- **TS and SP fit comparably on NSD pRFs:** TS k1 > SP in V1 only (+5%); no difference in V2-hV4.
- Unstable OLS voxels (R2 < -1, ~450 in total) come from the 49 raw LL features (LL only 477, k1 456; SP 0).

Run order:
- Henderson replication: `run/perform_pca_henderson_replication.sh` -> `run/fit_replication_all_subjects.sh`
  (`PRF_MODEL=nsd_grid5` and `PRF_MODEL=henderson_grid5`).
- k per group: `run/perform_pca_k_per_group.sh` (`K=1`, `K=2`) -> `run/fit_wholedata_k_per_group.sh` (`TAG=k1`,
  `TAG=k2`) -> `run/fit_wholedata_ll_only.sh`.
- Comparison: Section 4 of `notebooks/fig3_TS_SP_comparison.ipynb` (table, figure, gap decomposition).

## Session-wise (drift) fits

`fit_cross_session_matrix_fast.py` fits one model per session (first 30 sessions) and evaluates every model on
all trials of every session (30 x 30 cvR2 and Pearson r per voxel; diagonal = in-sample). Settings: OLS,
single trials (`--average_image_reps 0`), raw betas (`--zscore_betas_within_sess 0`), whole-session split, NSD
pRFs aligned to grid 5. Checkpointed after every training session (resumes after a disconnect).

- Original configuration: joint pcaHL95 (54) and LL only (pca0, 49): `run/fit_cross_session_matrix_fast_*.sh`.
- k1 configuration (58): `run/fit_cross_session_k1.sh`
  (split across Colab sessions with `SUBJECTS="1 2 3 4" CODE=... bash run/fit_cross_session_k1.sh`).

Single-voxel drift analysis (Roth & Merriam procedure, SP and TS): `../analysis/single_voxel_drift_roth_style.ipynb`.

## Running from Colab without copying code to Drive

Data (features, betas, pRF indices, fits) stay on Drive; the code can run from a clone of this repository.
Scripts that support this read a `CODE` variable (`perform_pca_k_per_group.sh`, `fit_wholedata_k_per_group.sh`,
`fit_wholedata_ll_only.sh`, `fit_cross_session_k1.sh`):

```python
from google.colab import drive
drive.mount('/content/drive')
!rm -rf /content/rep_over_time_2_models
!git clone -q https://github.com/amityelin/rep_over_time_2_models.git /content/rep_over_time_2_models
CODE = "/content/rep_over_time_2_models/TS_Model/texturemodel/"
RUN  = CODE + "code/run/"
!CODE={CODE} bash {RUN}fit_cross_session_k1.sh
```

Rerun the clone cell after every runtime reset. Without `CODE`, the scripts use the code copy on Drive
(`V2_drift/texturemodel/`).

## Fixes (Oct-Nov 2026)

- **Feature/label truncation with `--henderson_pca 1`:** features were loaded from the per-group PCA folder (127
  columns) but labels from the joint-PCA folder (54), and the loader silently truncated the features to 54
  columns, dropping 8 of the 9 HL groups. Fixed in `feature_extraction/fwrf_features.py` and
  `feature_extraction/texture_feature_utils.py`; the loader now raises an error whenever labels and feature
  columns differ. The joint-PCA (drift) runs were not affected (54 = 54 for all subjects); its concat file name
  (`..._concat_v95pct_grid5.h5py`) was restored in the same change.
- **Package layout:** the folder `analayze_fits` was renamed to `analyze_fits` and the `__init__ (N).py` files back
  to `__init__.py` (both renamed by a Drive download), so the code imports correctly from a git clone.
