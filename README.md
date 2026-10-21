# Master's thesis in progress: comparing two models on NSD

This repository contains the analysis code for my master's thesis, which is currently in progress. The project uses Natural Scenes Dataset (NSD) images and fMRI measurements to compare two computational models of human visual-cortex responses and investigate how representations change over time.

This is a **multi-notebook research codebase**, not a single notebook. The analysis is divided into ordered stages for feature construction, population-receptive-field (pRF) sampling, voxel-wise regression, permutation analysis, aggregation, and visualization.

> **Project status:** thesis research in development. `SP_Model/` contains the steerable-pyramid workflow; `TS_Model/` contains the texture-statistics model (adapted from Henderson et al., 2023), its notebooks, and a validation against the original published fits (see `TS_Model/README.md`). The repository should not be interpreted as a completed two-model comparison until the final thesis results are added.

## Research objective

The thesis asks how well two computational feature models explain NSD fMRI responses and whether their learned or predicted representations remain stable across measurement sessions. The model pipelines are analyzed using the same broader NSD research framework so their results can ultimately be compared.

## Current analysis pipeline

The `SP_Model/` workflow contains nine notebooks:

| Step | Notebook | Purpose |
|---:|---|---|
| 0 | `0_nsd_steerable_pipeline.ipynb` | Construct steerable-pyramid features from NSD stimuli |
| 1 | `1_nsd_prf_sampling_from_pyramid.ipynb` | Sample model features using voxel pRF parameters |
| 2 | `2_nsd_regresion_prf_split.ipynb` | Fit session-split voxel regression models |
| 3 | `3_Combine_Session__ROIs.ipynb` | Combine session and ROI outputs |
| 4 | `4_Save_Permutations_for_Session_Drift.ipynb` | Generate session-drift permutations |
| 5 | `5_sim_pop_response_expand.ipynb` | Expand simulated population responses |
| 6 | `6_save_perm_pop_expand.ipynb` | Generate permutation outputs for expanded responses |
| 7 | `7_plots.ipynb` | Aggregate and visualize results |
| 8 | `8_SP_wholedata_validation.ipynb` | Fit the SP model once on all data and validate on the shared-1000 images (Henderson-style), for comparison with session-wise fits and with the TS model |

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
│   ├── 8_SP_wholedata_validation.ipynb
│   └── upload_pyr_files_to_drive.py
└── TS_Model/
    ├── README.md
    ├── requirements.txt
    ├── notebooks/          # TS fitting, diagnostics, Henderson validation (fig3_TS_SP_comparison.ipynb)
    └── texturemodel/       # modified copy of Henderson's texturemodel code
```

## Data and reproducibility

The NSD images, fMRI responses, derived feature arrays, and model outputs are not distributed here. Obtain NSD data through the official access process and follow its data-use terms. The notebooks require separately configured data and output paths as well as substantial storage and compute resources.

For review, read the notebooks in numerical order. Before a fresh run, configure paths for the execution environment, inspect notebook dependencies, and validate each pipeline on a small subset.

## Scientific context

The session-drift analysis is inspired by:

Roth, Z. N., & Merriam, E. P. (2023). Representations in human primary visual cortex drift over time. *Nature Communications, 14*, 4422. https://doi.org/10.1038/s41467-023-40144-w
