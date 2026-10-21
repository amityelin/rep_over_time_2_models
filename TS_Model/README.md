# TS_Model: texture-statistics encoding model

Adaptation of the texture-statistics encoding model of Henderson et al. (2023)
([code](https://github.com/mmhenderson/texturemodel), J. Neurosci. 43:4144) for the longitudinal
representational-drift analysis. `texturemodel/` is a modified copy of her repository; `notebooks/` holds
the Colab notebooks used for fitting, diagnostics and figures.

## Main changes relative to Henderson's code

| Stage | Henderson | This project |
|---|---|---|
| Images | 240 x 240 px, PIL bilinear | 256 x 256 px, red fixation marker added before grayscale (BT.709) |
| Feature extraction | one pass | two halves (`extract_pyramid_texture_features[_second].py`), merged by summation |
| pRFs | fitted jointly with the model (grid 5) | NSD mapping-task pRFs aligned to grid 5 (`notebooks/pRF fits.ipynb`) |
| PCA (drift analysis) | separate PCA per higher-order group | one joint PCA over all 592 higher-order features, 95% (`pca_texture_feats_pcaHL.py`) |
| PCA (validation) | -- | Henderson's grouping, 95% (`pca_texture_feats.py`, `--henderson_pca 1`) |
| Regression | banded ridge | OLS |
| Fitting | one fit on all sessions | one fit per session + cross-session validation (`fit_cross_session_matrix_fast.py`) |

## Bug fixed (Oct 2026)

With `--henderson_pca 1`, features were loaded from the Henderson-replication PCA folder (127 columns) but the
feature labels from the joint-PCA folder (54 labels), and the loader silently truncated the features to 54
columns, dropping 8 of the 9 higher-order groups. Fixed in `feature_extraction/fwrf_features.py` and
`feature_extraction/texture_feature_utils.py`; the loader now raises an error whenever the number of labels and
feature columns differ. The joint-PCA (drift) path was not affected by the truncation, but its concat file name
(`..._concat_v95pct_grid5.h5py`) was restored in the same change.

## Validation against Henderson et al. (2023)

`notebooks/fig3_TS_SP_comparison.ipynb` reproduces her Fig. 3B metric (mean validation R2 over ROI voxels with
noise ceiling > 0.01, pRF-mapping ROIs without category-ROI overlap) for V1-hV4:

- With her feature grouping and her pRFs (S01 V2), the implementation reproduces her fits voxel by voxel
  (Spearman r = 0.965 across 2,096 voxels; mean R2 0.168 vs. 0.171 over her 1,424 V2 voxels).
- With pRFs from the independent NSD mapping experiment (all 8 subjects), performance reaches 84% (V1), 81% (V2),
  76% (V3) and 70% (hV4) of the original.
- OLS produces a small number of numerically unstable voxels (R2 < -1, ~1.7%), which are excluded and reported.

Run order: `run/perform_pca_henderson_replication.sh` -> `run/fit_replication_all_subjects.sh` ->
`notebooks/fig3_TS_SP_comparison.ipynb`.

## Whole-data fits with matched feature counts

To compare models with the same number of predictors as the SP model (18), `run/perform_pca_fixed_n.sh`
builds pcaLL18 (first 18 PCs of the 49 lower-order features) and pcaAll18 (first 18 PCs of all 641 features),
and `run/fit_wholedata_fixed_n.sh` fits them on all data (8 subjects x V1-hV4). Section 4 of
`notebooks/fig3_TS_SP_comparison.ipynb` compares all whole-data models on Henderson's Fig. 3 voxels.
