# analysis/

Cross-model analyses that read the outputs of both pipelines (SP_Model, TS_Model).

- `single_voxel_drift_roth_style.ipynb` — single-voxel cross-session drift (cvR² and Pearson r) for SP and TS,
  V1–hV4, with the Roth & Merriam (2023) procedure (pRF R² > 0 voxels, median per session pair, lag correlation,
  session-order permutations), SP normalizations (raw / zeroMean / equalStd), and a matched voxel-count control.
- `population_rdm_matched_voxels.ipynb` — population-response and RDM drift (SP notebook 6 outputs), all voxels vs
  matched voxel count, with group permutation tests and a between-ROI comparison.
