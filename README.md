# Representational drift in human V1

This repository contains a notebook-based pipeline for reproducing and extending analyses inspired by Roth and Merriam (2023), *Representations in human primary visual cortex drift over time*. It uses Natural Scenes Dataset (NSD) images and fMRI measurements with a steerable-pyramid encoding model.

> **Project status:** research workflow in development. The repository currently documents the steerable-pyramid model pipeline; it does not present a finalized independent replication result.

## What this project demonstrates

- Processing natural images into multiscale, multi-orientation features
- Sampling image features with voxel-specific population receptive-field models
- Fitting voxel-wise regression models to session-separated fMRI responses
- Generating permutation-based controls for session-drift analyses
- Aggregating ROI-level results and producing scientific visualizations
- Working with large HDF5/MATLAB-format neuroscience data

## Analysis pipeline

Run the notebooks in numerical order inside `SP_Model/`:

| Step | Notebook | Purpose |
|---:|---|---|
| 0 | `0_nsd_steerable_pipeline.ipynb` | Build steerable-pyramid features from NSD stimuli |
| 1 | `1_nsd_prf_sampling_from_pyramid.ipynb` | Sample features using voxel pRF parameters |
| 2 | `2_nsd_regresion_prf_split.ipynb` | Fit session-split voxel regression models |
| 3 | `3_Combine_Session__ROIs.ipynb` | Combine session and ROI outputs |
| 4 | `4_Save_Permutations_for_Session_Drift.ipynb` | Generate session-drift permutations |
| 5 | `5_sim_pop_response_expand.ipynb` | Expand simulated population responses |
| 6 | `6_save_perm_pop_expand.ipynb` | Generate permutation outputs for expanded responses |
| 7 | `7_plots.ipynb` | Aggregate and visualize results |

The helper `SP_Model/upload_pyr_files_to_drive.py` supports the original cloud-storage workflow. Paths and storage requirements should be reviewed before running it.

## Repository structure

```text
.
├── SP_Model/
│   ├── 0_nsd_steerable_pipeline.ipynb
│   ├── 1_nsd_prf_sampling_from_pyramid.ipynb
│   ├── 2_nsd_regresion_prf_split.ipynb
│   ├── 3_Combine_Session__ROIs.ipynb
│   ├── 4_Save_Permutations_for_Session_Drift.ipynb
│   ├── 5_sim_pop_response_expand.ipynb
│   ├── 6_save_perm_pop_expand.ipynb
│   ├── 7_plots.ipynb
│   └── upload_pyr_files_to_drive.py
└── TS_Model/                 # placeholder for a second model workflow
```

## Data and reproducibility

The NSD images, fMRI responses, derived feature arrays, and model outputs are not distributed here. Obtain NSD data through the official access process and follow its data-use terms. The notebooks expect substantial storage and compute resources, and some paths reflect the original research environment.

For a first review, read the notebooks in order without executing them. Before a fresh run, configure local data/output paths, inspect each notebook's dependencies, and validate the pipeline on a small subset.

## Reference

Roth, Z. N., & Merriam, E. P. (2023). Representations in human primary visual cortex drift over time. *Nature Communications, 14*, 4422. https://doi.org/10.1038/s41467-023-40144-w
