# Human-V1-Drift

Python code to replicate drift in human's V1 article:

**Roth, Z.N., Merriam, E.P.** (2023).  
_Representations in human primary visual cortex drift over time._  
**Nature Communications, 14**, 4422.
[https://doi.org/10.1038/s41467-023-40144-w]

---

## Folder Structure

```
V1-Drift/
├── nsd_steerable_pipeline.ipynb # Step 1: Generate steerable pyramid features from NSD stimuli
├── nsd_prf_sampling_from_pyramid.ipynb # Step 2: Project features through pRF filters into voxel space
├── nsd_regresion_prf_split.ipynb # Step 3: Regress voxel-level fMRI responses on pRF-based features
├── README.md # Project description and structure
```

---

## 📘 Notebooks Overview

### `nsd_steerable_pipeline.ipynb`

- Loads NSD stimuli from HDF5
- Interpolates and pads images, adds fixation point
- Applies steerable pyramid decomposition
- Saves `.mat` files with orientation- and level-specific features

Using the Python package for multi-scale image processing, adapted from Eero Simoncelli’s **matlabPyrTools**:  
[https://pyrtools.readthedocs.io/en/latest/](https://pyrtools.readthedocs.io/en/latest/)

**Example for one image:**

_Input to pyramid:_

![image](https://github.com/user-attachments/assets/d2b72179-ed7e-4ae4-a09f-8721fdd29add)

_Filters:_

![image](https://github.com/user-attachments/assets/ffc4f9f5-9bda-419e-8e98-386f9ce4d8ff)

_High + Low Pass Filters:_

![image](https://github.com/user-attachments/assets/8d0253be-a4db-43f0-8e8d-9bb39ed430ad)

---

### `nsd_prf_sampling_from_pyramid.ipynb`

- Loads precomputed pyramid features
- Applies 2D Gaussian pRF kernels per voxel and visual ROI
- Projects features into voxel space
- Saves HDF5 files with per-voxel energy responses

---

### `nsd_regresion_prf_split.ipynb`

- Loads beta values from NSD sessions
- Loads corresponding pRF features
- Runs split-by-session regression per voxel
- Computes R², coefficient stability, and predictive power

---

## 🧠 Project Goal

To quantify how representational features encoded in human V1 drift over time, by modeling neural
