#!/usr/bin/env python3
"""
Compute texture-model PCA once per pRF and create multiple variance-threshold
feature files without recomputing PCA or re-projecting the training images.

Main properties
---------------
1. PCA is computed once per pRF.
2. The training scores returned by compute_pca() are saved immediately.
3. Only non-training images are projected afterward.
4. PCA state is checkpointed in one resumable HDF5 file.
5. Threshold-specific output files are resumable per pRF.
6. Files are flushed every checkpoint_every pRFs, so a Colab disconnect loses
   at most a small number of pRFs.
7. Final concatenated files are written in chunks without loading all raw
   low-level features into RAM.

Expected raw feature shape:
    [n_images, n_original_features, n_prfs]
"""

import argparse
import gc
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from utils import default_paths, nsd_utils
from model_fitting import initialize_fitting
from feature_extraction import texture_feature_utils, pca_feats, fwrf_features


compute_pca = pca_feats.compute_pca
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# ============================================================================
# General utilities
# ============================================================================

def print_flush(*args, **kwargs) -> None:
    print(*args, **kwargs)
    sys.stdout.flush()


def ensure_unique_sorted_thresholds(thresholds: Iterable[int]) -> List[int]:
    values = sorted(set(int(x) for x in thresholds))
    invalid = [x for x in values if x < 0 or x > 100]
    if invalid:
        raise ValueError(f"Thresholds must be between 0 and 100. Invalid: {invalid}")
    return values


def get_fit_inds(subject: int, image_set: str | None) -> Tuple[str, np.ndarray]:
    if subject is None:
        raise ValueError("Please provide --subject 1-8.")

    if image_set is not None:
        raise NotImplementedError(
            "This script currently supports subject-specific NSD PCA only."
        )

    image_set = f"S{subject}"

    if subject == 999:
        fit_inds = np.ones((10000,), dtype=bool)
    else:
        subject_df = nsd_utils.get_subj_df(subject)
        fit_inds = np.asarray(subject_df["shared1000"] == False, dtype=bool)

    return image_set, fit_inds


def get_pca_groups(
    pca_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    feature_column_labels, original_feature_type_names = (
        texture_feature_utils.get_feature_inds()
    )

    feature_column_labels = np.asarray(feature_column_labels)
    original_feature_type_names = np.asarray(original_feature_type_names)

    is_ll = np.asarray(
        texture_feature_utils.is_low_level(),
        dtype=bool,
    )

    is_hl = ~is_ll

    if pca_type == "pcaHL":

        # Keep LL raw, PCA only HL
        pca_group_names = np.asarray(["high_level"])
        feature_inds = np.asarray([is_hl])
        output_feature_type_names = np.asarray(
            ["low_level", "high_level"]
        )

    elif pca_type == "pcaAll":

        # PCA jointly across all 641 features
        is_all = np.ones_like(
            feature_column_labels,
            dtype=bool,
        )

        pca_group_names = np.asarray(["all_features"])
        feature_inds = np.asarray([is_all])
        output_feature_type_names = np.asarray(
            ["all_features"]
        )

    elif pca_type == "pcaLL":

        # PCA ONLY across the 49 low-level features
        pca_group_names = np.asarray(["low_level"])
        feature_inds = np.asarray([is_ll])
        output_feature_type_names = np.asarray(
            ["low_level"]
        )

    else:
        raise ValueError(
            "Use pca_type='pcaHL', 'pcaAll', or 'pcaLL'."
        )

    return (
        feature_column_labels,
        original_feature_type_names,
        pca_group_names,
        feature_inds,
        output_feature_type_names,
    )


def n_components_for_threshold(
    cumulative_explained_variance: np.ndarray,
    threshold: int,
    max_pc: int,
) -> int:
    """Return minimum number of PCs reaching threshold percent variance."""
    if threshold == 0 or max_pc == 0:
        return 0

    cumulative_explained_variance = np.asarray(cumulative_explained_variance)
    reached = np.flatnonzero(cumulative_explained_variance >= threshold)

    if reached.size:
        return min(int(reached[0] + 1), int(max_pc))

    return int(max_pc)


def create_dataset(
    h5_file: h5py.File,
    name: str,
    shape: Tuple[int, ...],
    dtype,
    compress: bool,
    fillvalue=None,
    chunks=True,
):
    kwargs = {
        "shape": shape,
        "dtype": dtype,
        "chunks": chunks,
    }
    if compress:
        kwargs["compression"] = "gzip"
    if fillvalue is not None:
        kwargs["fillvalue"] = fillvalue
    return h5_file.create_dataset(name, **kwargs)


def validate_existing_dataset(
    h5_file: h5py.File,
    name: str,
    expected_shape: Tuple[int, ...],
) -> h5py.Dataset:
    if name not in h5_file:
        raise RuntimeError(f"Checkpoint is missing dataset '{name}'.")

    dataset = h5_file[name]
    if tuple(dataset.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"Checkpoint dataset '{name}' has shape {dataset.shape}, "
            f"expected {expected_shape}. Use --overwrite 1 to restart."
        )
    return dataset


# ============================================================================
# PCA checkpoint
# ============================================================================

def open_or_create_pca_checkpoint(
    checkpoint_filename: str,
    *,
    n_train: int,
    n_original_features: int,
    max_pc: int,
    n_prfs: int,
    train_inds: np.ndarray,
    thresholds: Sequence[int],
    image_set: str,
    pca_type: str,
    pca_block: str,
    save_dtype,
    compress: bool,
    overwrite: bool,
) -> Tuple[h5py.File, Dict[str, h5py.Dataset]]:
    """
    One checkpoint contains all PCA outputs for one PCA block.

    completed_prfs[p] is set to 1 only after weights, mean, EV, training scores,
    and component counts for pRF p have all been written.
    """
    if overwrite and os.path.exists(checkpoint_filename):
        print_flush("Removing old PCA checkpoint:", checkpoint_filename)
        os.remove(checkpoint_filename)

    exists = os.path.exists(checkpoint_filename)
    mode = "r+" if exists else "w"
    h5_file = h5py.File(checkpoint_filename, mode)

    if not exists:
        print_flush("Creating PCA checkpoint:", checkpoint_filename)

        h5_file.attrs["image_set"] = image_set
        h5_file.attrs["pca_type"] = pca_type
        h5_file.attrs["pca_block"] = str(pca_block)
        h5_file.attrs["n_train"] = n_train
        h5_file.attrs["n_original_features"] = n_original_features
        h5_file.attrs["max_pc"] = max_pc
        h5_file.attrs["n_prfs"] = n_prfs
        h5_file.attrs["thresholds"] = np.asarray(thresholds, dtype=np.int16)

        h5_file.create_dataset("train_image_inds", data=train_inds.astype(np.int32))

        create_dataset(
            h5_file,
            "pca_weights",
            (n_original_features, max_pc, n_prfs),
            save_dtype,
            compress,
            fillvalue=np.nan,
        )
        create_dataset(
            h5_file,
            "pca_premean",
            (n_original_features, n_prfs),
            save_dtype,
            compress,
            fillvalue=np.nan,
        )
        create_dataset(
            h5_file,
            "pca_explained_variance",
            (max_pc, n_prfs),
            save_dtype,
            compress,
            fillvalue=np.nan,
        )
        create_dataset(
            h5_file,
            "train_scores",
            (n_train, max_pc, n_prfs),
            save_dtype,
            compress,
            fillvalue=np.nan,
        )
        create_dataset(
            h5_file,
            "n_components",
            (len(thresholds), n_prfs),
            np.int16,
            False,
            fillvalue=-1,
        )
        create_dataset(
            h5_file,
            "n_valid_pcs",
            (n_prfs,),
            np.int16,
            False,
            fillvalue=-1,
        )
        create_dataset(
            h5_file,
            "completed_prfs",
            (n_prfs,),
            np.uint8,
            False,
            fillvalue=0,
        )
        h5_file.flush()

    else:
        print_flush("Resuming PCA checkpoint:", checkpoint_filename)

        saved_thresholds = list(np.asarray(h5_file.attrs["thresholds"], dtype=int))
        if saved_thresholds != list(thresholds):
            raise RuntimeError(
                "Checkpoint thresholds do not match the requested thresholds.\n"
                f"Saved: {saved_thresholds}\nRequested: {list(thresholds)}\n"
                "Use --overwrite 1 to restart, or run with the original thresholds."
            )

        saved_train_inds = np.asarray(h5_file["train_image_inds"], dtype=int)
        if not np.array_equal(saved_train_inds, train_inds):
            raise RuntimeError(
                "Training image indices differ from the existing checkpoint. "
                "Use --overwrite 1 to restart."
            )

    datasets = {
        "weights": validate_existing_dataset(
            h5_file,
            "pca_weights",
            (n_original_features, max_pc, n_prfs),
        ),
        "premean": validate_existing_dataset(
            h5_file,
            "pca_premean",
            (n_original_features, n_prfs),
        ),
        "explained_variance": validate_existing_dataset(
            h5_file,
            "pca_explained_variance",
            (max_pc, n_prfs),
        ),
        "train_scores": validate_existing_dataset(
            h5_file,
            "train_scores",
            (n_train, max_pc, n_prfs),
        ),
        "n_components": validate_existing_dataset(
            h5_file,
            "n_components",
            (len(thresholds), n_prfs),
        ),
        "n_valid_pcs": validate_existing_dataset(
            h5_file,
            "n_valid_pcs",
            (n_prfs,),
        ),
        "completed": validate_existing_dataset(
            h5_file,
            "completed_prfs",
            (n_prfs,),
        ),
    }
    return h5_file, datasets


def compute_missing_pcas(
    *,
    floader,
    all_image_inds: np.ndarray,
    fit_inds: np.ndarray,
    feature_mask: np.ndarray,
    n_prfs: int,
    max_pc: int,
    fixed_n_components: int | None,
    thresholds: Sequence[int],
    checkpoint_file: h5py.File,
    checkpoint_dsets: Dict[str, h5py.Dataset],
    checkpoint_every: int,
    save_dtype,
) -> None:
    completed = np.asarray(checkpoint_dsets["completed"], dtype=bool)
    missing_prfs = np.flatnonzero(~completed)

    print_flush(
        f"PCA checkpoint status: {completed.sum()}/{n_prfs} pRFs complete; "
        f"{len(missing_prfs)} remaining."
    )

    for run_index, prf_model_index in enumerate(missing_prfs, start=1):
        print_flush(
            f"PCA: pRF {prf_model_index + 1}/{n_prfs} "
            f"(remaining item {run_index}/{len(missing_prfs)})"
        )

        feat, _ = floader.load(all_image_inds, int(prf_model_index))
        features_in_prf = np.asarray(
            feat[:, feature_mask],
            dtype=np.float32,
        )

        scores_train, wts, pre_mean, ev = compute_pca(
            features_in_prf[fit_inds, :],
            max_pc_to_retain=max_pc,
        )

        scores_train = np.asarray(scores_train)
        wts = np.asarray(wts)
        pre_mean = np.asarray(pre_mean).reshape(-1)
        ev = np.asarray(ev).reshape(-1)

        n_valid_pcs = min(
            scores_train.shape[1],
            wts.shape[0],
            len(ev),
            max_pc,
        )

        # Write only valid PCA columns; unused columns remain NaN.
        checkpoint_dsets["weights"][
            :, :n_valid_pcs, prf_model_index
        ] = wts[:n_valid_pcs, :].T.astype(save_dtype)

        checkpoint_dsets["premean"][
            :, prf_model_index
        ] = pre_mean.astype(save_dtype)

        checkpoint_dsets["explained_variance"][
            :n_valid_pcs, prf_model_index
        ] = ev[:n_valid_pcs].astype(save_dtype)

        checkpoint_dsets["train_scores"][
            :, :n_valid_pcs, prf_model_index
        ] = scores_train[:, :n_valid_pcs].astype(save_dtype)

        cumulative_ev = np.cumsum(
            ev[:n_valid_pcs]
        )
        
        for threshold_index, threshold in enumerate(
            thresholds
        ):
        
            if fixed_n_components is not None:
        
                # Fixed-dimensionality mode:
                # always retain the first N valid PCs.
                n_comp = min(
                    int(fixed_n_components),
                    int(n_valid_pcs),
                )
        
            else:
        
                # Original variance-threshold mode.
                n_comp = n_components_for_threshold(
                    cumulative_ev,
                    threshold,
                    n_valid_pcs,
                )
        
            checkpoint_dsets["n_components"][
                threshold_index,
                prf_model_index
            ] = n_comp

        checkpoint_dsets["n_valid_pcs"][prf_model_index] = n_valid_pcs

        # This flag is deliberately written last.
        checkpoint_dsets["completed"][prf_model_index] = 1

        if (
            run_index % checkpoint_every == 0
            or run_index == len(missing_prfs)
        ):
            checkpoint_file.flush()
            print_flush(
                f"Checkpoint flushed after pRF {prf_model_index + 1}. "
                f"Total complete: {completed.sum() + run_index}/{n_prfs}"
            )

        del feat, features_in_prf, scores_train, wts, pre_mean, ev
        gc.collect()


# ============================================================================
# Threshold-specific score files
# ============================================================================

def get_actual_max_components(
    n_components: np.ndarray,
    thresholds: Sequence[int],
) -> Dict[int, int]:
    return {
        int(threshold): int(np.max(n_components[threshold_index, :]))
        for threshold_index, threshold in enumerate(thresholds)
    }


def open_or_create_threshold_file(
    filename: str,
    *,
    n_trials: int,
    max_components: int,
    n_prfs: int,
    threshold: int,
    image_set: str,
    pca_type: str,
    pca_block: str,
    save_dtype,
    compress: bool,
    overwrite: bool,
) -> Tuple[h5py.File, h5py.Dataset, h5py.Dataset]:
    if overwrite and os.path.exists(filename):
        print_flush("Removing old threshold file:", filename)
        os.remove(filename)

    exists = os.path.exists(filename)
    mode = "r+" if exists else "w"
    out_file = h5py.File(filename, mode)

    if not exists:
        print_flush("Creating threshold score file:", filename)
        out_file.attrs["threshold"] = int(threshold)
        out_file.attrs["image_set"] = image_set
        out_file.attrs["pca_type"] = pca_type
        out_file.attrs["pca_block"] = str(pca_block)

        features = create_dataset(
            out_file,
            "features",
            (n_trials, max_components, n_prfs),
            save_dtype,
            compress,
            fillvalue=np.nan,
        )
        completed = create_dataset(
            out_file,
            "completed_prfs",
            (n_prfs,),
            np.uint8,
            False,
            fillvalue=0,
        )
        out_file.flush()
    else:
        print_flush("Resuming threshold score file:", filename)
        features = validate_existing_dataset(
            out_file,
            "features",
            (n_trials, max_components, n_prfs),
        )
        completed = validate_existing_dataset(
            out_file,
            "completed_prfs",
            (n_prfs,),
        )

    return out_file, features, completed


def write_threshold_score_files(
    *,
    floader,
    all_image_inds: np.ndarray,
    fit_inds: np.ndarray,
    feature_mask: np.ndarray,
    pca_checkpoint_dsets: Dict[str, h5py.Dataset],
    thresholds: Sequence[int],
    threshold_filenames: Dict[int, str],
    n_trials: int,
    n_prfs: int,
    actual_max_ncomp: Dict[int, int],
    image_set: str,
    pca_type: str,
    pca_block: str,
    checkpoint_every: int,
    save_dtype,
    compress: bool,
    overwrite_outputs: bool,
) -> None:
    """
    Reuse saved training scores and project only non-training images.

    For each pRF:
      output[train_inds]    <- checkpointed scores_train
      output[nontrain_inds] <- (X_nontrain - PCA mean) @ PCA weights
    """
    train_inds = np.flatnonzero(fit_inds)
    nontrain_inds = np.flatnonzero(~fit_inds)

    out_files: Dict[int, h5py.File] = {}
    out_dsets: Dict[int, h5py.Dataset] = {}
    out_completed: Dict[int, h5py.Dataset] = {}

    try:
        for threshold in thresholds:
            out_file, out_dset, completed = open_or_create_threshold_file(
                threshold_filenames[threshold],
                n_trials=n_trials,
                max_components=actual_max_ncomp[threshold],
                n_prfs=n_prfs,
                threshold=threshold,
                image_set=image_set,
                pca_type=pca_type,
                pca_block=pca_block,
                save_dtype=save_dtype,
                compress=compress,
                overwrite=overwrite_outputs,
            )
            out_files[threshold] = out_file
            out_dsets[threshold] = out_dset
            out_completed[threshold] = completed

        complete_matrix = np.stack(
            [np.asarray(out_completed[t], dtype=bool) for t in thresholds],
            axis=0,
        )
        all_thresholds_complete = np.all(complete_matrix, axis=0)
        missing_prfs = np.flatnonzero(~all_thresholds_complete)

        print_flush(
            f"Threshold score status: {all_thresholds_complete.sum()}/{n_prfs} "
            f"pRFs complete for every threshold; {len(missing_prfs)} remaining."
        )

        for run_index, prf_model_index in enumerate(missing_prfs, start=1):
            print_flush(
                f"Writing scores: pRF {prf_model_index + 1}/{n_prfs} "
                f"(remaining item {run_index}/{len(missing_prfs)})"
            )

            n_valid_pcs = int(
                pca_checkpoint_dsets["n_valid_pcs"][prf_model_index]
            )
            if n_valid_pcs < 0:
                raise RuntimeError(
                    f"PCA checkpoint for pRF {prf_model_index} is incomplete."
                )

            # Use the PCA scores that were already computed during PCA fitting.
            train_scores = np.asarray(
                pca_checkpoint_dsets["train_scores"][
                    :, :n_valid_pcs, prf_model_index
                ],
                dtype=np.float32,
            )

            # Only held-out/non-training images require a matrix projection.
            nontrain_scores = None

            if nontrain_inds.size > 0 and n_valid_pcs > 0:
                # The feature loader caches the full image batch.
                # Request all images and subset the held-out images locally.
                feat_all, _ = floader.load(
                    all_image_inds,
                    int(prf_model_index),
                )

                x_nontrain = np.asarray(
                    feat_all[nontrain_inds, :][:, feature_mask],
                    dtype=np.float32,
                )

                pre_mean = np.asarray(
                    pca_checkpoint_dsets["premean"][:, prf_model_index],
                    dtype=np.float32,
                )

                weights = np.asarray(
                    pca_checkpoint_dsets["weights"][
                        :, :n_valid_pcs, prf_model_index
                    ],
                    dtype=np.float32,
                )

                nontrain_scores = (
                    x_nontrain - pre_mean[None, :]
                ) @ weights

            else:
                feat_all = None
                x_nontrain = None

            for threshold_index, threshold in enumerate(thresholds):
                if bool(out_completed[threshold][prf_model_index]):
                    continue

                n_comp = int(
                    pca_checkpoint_dsets["n_components"][
                        threshold_index, prf_model_index
                    ]
                )

                if n_comp > 0:
                    out_dsets[threshold][
                        train_inds, :n_comp, prf_model_index
                    ] = train_scores[:, :n_comp].astype(save_dtype)

                    if nontrain_inds.size > 0:
                        out_dsets[threshold][
                            nontrain_inds, :n_comp, prf_model_index
                        ] = nontrain_scores[:, :n_comp].astype(save_dtype)

                # Write completion flag only after all data for this threshold/pRF.
                out_completed[threshold][prf_model_index] = 1

            if (
                run_index % checkpoint_every == 0
                or run_index == len(missing_prfs)
            ):
                for threshold in thresholds:
                    out_files[threshold].flush()
                print_flush(
                    f"Threshold files flushed after pRF {prf_model_index + 1}."
                )

            del train_scores, nontrain_scores
            
            if feat_all is not None:
                del feat_all, x_nontrain, pre_mean, weights
                
            gc.collect()

    finally:
        for out_file in out_files.values():
            out_file.close()


# ============================================================================
# Summary
# ============================================================================

def save_pca_summary(
    summary_filename: str,
    *,
    subject: int,
    image_set: str,
    pca_type: str,
    pca_block: str,
    n_original_features: int,
    max_pc: int,
    thresholds: Sequence[int],
    checkpoint_dsets: Dict[str, h5py.Dataset],
    n_prfs: int,
) -> None:
    n_components = np.asarray(checkpoint_dsets["n_components"])
    explained_variance = np.asarray(checkpoint_dsets["explained_variance"])
    n_valid_pcs = np.asarray(checkpoint_dsets["n_valid_pcs"], dtype=int)

    rows = []
    for prf_model_index in range(n_prfs):
        valid = max(0, n_valid_pcs[prf_model_index])
        ev = explained_variance[:valid, prf_model_index]

        row = {
            "subject": subject,
            "image_set": image_set,
            "pca_type": pca_type,
            "pca_block": str(pca_block),
            "prf": int(prf_model_index),
            "n_original_features": int(n_original_features),
            "max_pc_to_retain": int(max_pc),
            "n_valid_pcs": int(valid),
            "pc1_variance_explained": float(ev[0]) if valid > 0 else np.nan,
            "total_saved_variance_explained": (
                float(np.sum(ev)) if valid > 0 else np.nan
            ),
        }

        for threshold_index, threshold in enumerate(thresholds):
            row[f"n_components_{threshold}"] = int(
                n_components[threshold_index, prf_model_index]
            )
        rows.append(row)

    pd.DataFrame(rows).to_csv(summary_filename, index=False)
    print_flush("Saved PCA summary:", summary_filename)


# ============================================================================
# Final concatenated feature files
# ============================================================================

def concat_output_paths(
    *,
    threshold: int,
    path_raw: str,
    pca_type: str,
    image_set: str,
    n_ori: int,
    n_sf: int,
    which_prf_grid: int,
    fixed_n_components: int | None = None,
) -> Tuple[str, str]:

    if fixed_n_components is not None:

        var_str = f"n{fixed_n_components}"

        path_pca = os.path.join(
            path_raw,
            f"PCA_{pca_type}_n{fixed_n_components}",
        )

    else:

        var_str = f"v{threshold}pct"

        path_pca = os.path.join(
            path_raw,
            f"PCA_{pca_type}_v{threshold}",
        )

    os.makedirs(
        path_pca,
        exist_ok=True,
    )

    concat_filename = os.path.join(
        path_pca,
        (
            f"{image_set}_{n_ori}ori_{n_sf}sf_{pca_type}_concat_"
            f"{var_str}_grid{which_prf_grid}.h5py"
        ),
    )

    labels_filename = os.path.join(
        path_pca,
        (
            f"{image_set}_{n_ori}ori_{n_sf}sf_featurelabels_"
            f"{pca_type}_{var_str}_grid{which_prf_grid}.npy"
        ),
    )

    return concat_filename, labels_filename


def write_concat_file_resumable(
    *,
    threshold: int,
    path_raw: str,
    pca_type: str,
    image_set: str,
    n_ori: int,
    n_sf: int,
    which_prf_grid: int,
    pca_filenames: Sequence[str],
    raw_features_file: str,
    feature_type_names: np.ndarray,
    save_dtype,
    compress: bool,
    image_chunk_size: int,
    overwrite: bool,
    fixed_n_components: int | None = None,
) -> None:
    """
    Build final [images, concatenated_features, pRFs] file by image chunks.

    completed_image_chunks makes this stage resumable as well.
    """
    concat_filename, labels_filename = concat_output_paths(
        threshold=threshold,
        path_raw=path_raw,
        pca_type=pca_type,
        image_set=image_set,
        n_ori=n_ori,
        n_sf=n_sf,
        which_prf_grid=which_prf_grid,
        fixed_n_components=fixed_n_components,
    )

    if overwrite and os.path.exists(concat_filename):
        print_flush("Removing old concat file:", concat_filename)
        os.remove(concat_filename)

    with h5py.File(raw_features_file, "r") as raw_file:
        raw_dataset = raw_file["/features"]
        n_samples, _, n_prfs = raw_dataset.shape

        if pca_type == "pcaHL":

            is_ll = np.asarray(
                texture_feature_utils.is_low_level(),
                dtype=bool,
            )
        
            n_raw = int(np.sum(is_ll))
        
            feature_labels = np.zeros(
                (n_raw,),
                dtype=int,
            )
        
            n_raw_groups = 1
        
        
        elif pca_type in ("pcaAll", "pcaLL"):
        
            # PCA-only representations:
            # no raw features are appended.
            is_ll = None
            n_raw = 0
        
            feature_labels = np.empty(
                (0,),
                dtype=int,
            )
        
            n_raw_groups = 0
        
        
        else:
        
            raise ValueError(
                "Use pca_type='pcaHL', 'pcaAll', or 'pcaLL'."
            )

        pca_dims = []
        for filename in pca_filenames:
            with h5py.File(filename, "r") as pca_file:
                shape = pca_file["/features"].shape
                if shape[0] != n_samples or shape[2] != n_prfs:
                    raise RuntimeError(
                        f"Incompatible PCA file shape {shape}: {filename}"
                    )
                pca_dims.append(int(shape[1]))

        total_dim = n_raw + sum(pca_dims)
        n_chunks = int(np.ceil(n_samples / image_chunk_size))

        exists = os.path.exists(concat_filename)
        mode = "r+" if exists else "w"

        with h5py.File(concat_filename, mode) as out_file:
            if not exists:
                print_flush("Creating concat file:", concat_filename)
                print_flush("Final concat shape:", (n_samples, total_dim, n_prfs))

                out_dataset = create_dataset(
                    out_file,
                    "features",
                    (n_samples, total_dim, n_prfs),
                    save_dtype,
                    compress,
                    fillvalue=np.nan,
                )
                completed_chunks = create_dataset(
                    out_file,
                    "completed_image_chunks",
                    (n_chunks,),
                    np.uint8,
                    False,
                    fillvalue=0,
                )
                out_file.attrs["threshold"] = int(threshold)
                out_file.attrs["image_chunk_size"] = int(image_chunk_size)
                out_file.flush()
            else:
                print_flush("Resuming concat file:", concat_filename)
                out_dataset = validate_existing_dataset(
                    out_file,
                    "features",
                    (n_samples, total_dim, n_prfs),
                )
                completed_chunks = validate_existing_dataset(
                    out_file,
                    "completed_image_chunks",
                    (n_chunks,),
                )

            completed = np.asarray(completed_chunks, dtype=bool)

            # Open PCA files once for the chunk loop.
            pca_handles = [h5py.File(filename, "r") for filename in pca_filenames]
            try:
                for chunk_index in np.flatnonzero(~completed):
                    start = int(chunk_index * image_chunk_size)
                    end = min(start + image_chunk_size, n_samples)

                    print_flush(
                        f"Concat {threshold}%: image chunk "
                        f"{chunk_index + 1}/{n_chunks} ({start}:{end})"
                    )

                    current_col = 0

                    if pca_type == "pcaHL":
                        raw_chunk = np.asarray(raw_dataset[start:end, :, :])
                        out_dataset[
                            start:end,
                            current_col:current_col + n_raw,
                            :,
                        ] = raw_chunk[:, is_ll, :].astype(save_dtype)
                        current_col += n_raw
                        del raw_chunk

                    for block_index, pca_handle in enumerate(pca_handles):
                        n_pc = pca_dims[block_index]
                        out_dataset[
                            start:end,
                            current_col:current_col + n_pc,
                            :,
                        ] = pca_handle["/features"][start:end, :, :]
                        current_col += n_pc

                    completed_chunks[chunk_index] = 1
                    out_file.flush()
                    gc.collect()

            finally:
                for handle in pca_handles:
                    handle.close()

    # Labels are small and can be recreated safely.
    for block_index, n_pc in enumerate(pca_dims):
        pca_label = n_raw_groups + block_index
        feature_labels = np.concatenate(
            [
                feature_labels,
                np.full((n_pc,), pca_label, dtype=int),
            ]
        )

    np.save(
        labels_filename,
        {
            "feature_column_labels": feature_labels,
            "feature_type_names": feature_type_names,
        },
        allow_pickle=True,
    )
    print_flush("Saved concat labels:", labels_filename)

    values, counts = np.unique(feature_labels, return_counts=True)
    for value, count in zip(values, counts):
        print_flush(
            f"{value}: {feature_type_names[int(value)]} -> {count} columns"
        )


# ============================================================================
# Main pipeline
# ============================================================================

def run_pca_texture_pyramid_multi_threshold(
    *,
    subject: int,
    image_set: str | None = None,
    pca_type: str = "pcaHL",
    thresholds: Sequence[int] = (40, 60, 80, 90, 95),
    max_pc_to_retain: int | None = None,
    fixed_n_components: int | None = None,
    debug: bool = False,
    which_prf_grid: int = 5,
    checkpoint_every: int = 10,
    image_chunk_size: int = 500,
    save_dtype=np.float32,
    compress: bool = True,
    overwrite_pca: bool = False,
    overwrite_outputs: bool = False,
    overwrite_concat: bool = False,
    build_concat: bool = True,
) -> None:
    
    # In debug mode, skip the expensive final concatenation.
    if debug:
        build_concat = False
    
    thresholds = ensure_unique_sorted_thresholds(thresholds)
    # Fixed-component mode does not use variance thresholds.
    # Keep one internal dummy threshold only, because the existing
    # checkpoint/output infrastructure is organized around thresholds.
    if fixed_n_components is not None:
        thresholds = [100]

    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1.")
    if image_chunk_size < 1:
        raise ValueError("image_chunk_size must be >= 1.")

    image_set, fit_inds = get_fit_inds(subject, image_set)
    n_trials = len(fit_inds)
    all_image_inds = np.arange(n_trials, dtype=int)
    train_inds = np.flatnonzero(fit_inds)

    path_raw = default_paths.pyramid_texture_feat_path
    if debug:
        path_raw = os.path.join(path_raw, "DEBUG")

    if fixed_n_components is not None:

        run_tag = f"n{fixed_n_components}"
    
        multi_path = os.path.join(
            path_raw,
            f"PCA_{pca_type}_{run_tag}_MULTI",
        )
    
    else:
    
        run_tag = None
    
        multi_path = os.path.join(
            path_raw,
            f"PCA_{pca_type}_MULTI",
        )
    checkpoint_dir = os.path.join(multi_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if fixed_n_components is not None:

        fixed_output_path = os.path.join(
            path_raw,
            f"PCA_{pca_type}_n{fixed_n_components}",
        )
    
        os.makedirs(
            fixed_output_path,
            exist_ok=True,
        )
    
    else:
    
        for threshold in thresholds:
            os.makedirs(
                os.path.join(
                    path_raw,
                    f"PCA_{pca_type}_v{threshold}",
                ),
                exist_ok=True,
            )

    n_ori = 4
    n_sf = 4
    prf_batch_size = 2 if debug else 100

    floader = fwrf_features.fwrf_feature_loader(
        subject=None,
        image_set=image_set,
        which_prf_grid=which_prf_grid,
        feature_type="pyramid_texture",
        n_ori=n_ori,
        n_sf=n_sf,
        pca_type=None,
        prf_batch_size=prf_batch_size,
    )

    raw_features_file = floader.features_file
    print_flush("Raw features file:", raw_features_file)

    with h5py.File(raw_features_file, "r") as raw_file:
        raw_shape = raw_file["/features"].shape

    if raw_shape[0] != n_trials:
        raise RuntimeError(
            f"Raw file has {raw_shape[0]} images, but fit_inds has {n_trials}."
        )

    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)
    n_prfs = int(models.shape[0])
    if debug:
        n_prfs = min(n_prfs, 2)

    if raw_shape[2] < n_prfs:
        raise RuntimeError(
            f"Raw file has only {raw_shape[2]} pRFs, but {n_prfs} are required."
        )

    (
        feature_column_labels,
        original_feature_type_names,
        pca_group_names,
        feature_masks,
        output_feature_type_names,
    ) = get_pca_groups(pca_type)

    print_flush("\nOriginal feature groups:")
    for group_index, group_name in enumerate(original_feature_type_names):
        print_flush(
            f"{group_index:2d} {group_name:<45} "
            f"{np.sum(feature_column_labels == group_index)}"
        )

    print_flush("\nPCA blocks:")
    for block_name, mask in zip(pca_group_names, feature_masks):
        print_flush(f"{block_name}: {np.sum(mask)} original features")

    print_flush("\nThresholds:", thresholds)
    print_flush("Training images:", int(np.sum(fit_inds)))
    print_flush("Non-training images:", int(np.sum(~fit_inds)))
    print_flush("pRFs:", n_prfs)

    pca_filenames_by_threshold: Dict[int, List[str]] = {
        threshold: [] for threshold in thresholds
    }

    for block_index, pca_block in enumerate(pca_group_names):
        print_flush("\n" + "=" * 80)
        print_flush("PCA block:", pca_block)
        print_flush("=" * 80)

        feature_mask = np.asarray(feature_masks[block_index], dtype=bool)
        n_original_features = int(np.sum(feature_mask))
        if fixed_n_components is not None:
        
            max_pc = min(
                int(fixed_n_components),
                n_original_features,
            )
        
        else:
        
            max_pc = (
                n_original_features
                if max_pc_to_retain is None
                else min(
                    int(max_pc_to_retain),
                    n_original_features,
                )
            )

        checkpoint_filename = os.path.join(
            checkpoint_dir,
            (
                f"{image_set}_{n_ori}ori_{n_sf}sf_PCA_{pca_block}_"
                f"grid{which_prf_grid}_checkpoint.h5py"
            ),
        )

        checkpoint_file, checkpoint_dsets = open_or_create_pca_checkpoint(
            checkpoint_filename,
            n_train=len(train_inds),
            n_original_features=n_original_features,
            max_pc=max_pc,
            n_prfs=n_prfs,
            train_inds=train_inds,
            thresholds=thresholds,
            image_set=image_set,
            pca_type=pca_type,
            pca_block=str(pca_block),
            save_dtype=save_dtype,
            compress=compress,
            overwrite=overwrite_pca,
        )

        try:
            compute_missing_pcas(
                floader=floader,
                all_image_inds=all_image_inds,
                fit_inds=fit_inds,
                feature_mask=feature_mask,
                n_prfs=n_prfs,
                max_pc=max_pc,
                fixed_n_components=fixed_n_components,
                thresholds=thresholds,
                checkpoint_file=checkpoint_file,
                checkpoint_dsets=checkpoint_dsets,
                checkpoint_every=checkpoint_every,
                save_dtype=save_dtype,
            )

            if not np.all(np.asarray(checkpoint_dsets["completed"], dtype=bool)):
                raise RuntimeError("PCA pass ended with incomplete pRFs.")

            n_components = np.asarray(checkpoint_dsets["n_components"], dtype=int)
            actual_max_ncomp = get_actual_max_components(
                n_components,
                thresholds,
            )
            print_flush("Maximum retained PCs by threshold:", actual_max_ncomp)

            threshold_filenames = {}

            for threshold in thresholds:
            
                if fixed_n_components is not None:
            
                    var_str = f"n{fixed_n_components}"
            
                    threshold_path = os.path.join(
                        path_raw,
                        f"PCA_{pca_type}_n{fixed_n_components}",
                    )
            
                else:
            
                    var_str = f"v{threshold}pct"
            
                    threshold_path = os.path.join(
                        path_raw,
                        f"PCA_{pca_type}_v{threshold}",
                    )
            
                os.makedirs(
                    threshold_path,
                    exist_ok=True,
                )
            
                filename = os.path.join(
                    threshold_path,
                    (
                        f"{image_set}_{n_ori}ori_{n_sf}sf_PCA_{pca_block}_"
                        f"{var_str}_grid{which_prf_grid}.h5py"
                    ),
                )
            
                threshold_filenames[threshold] = filename
            
                pca_filenames_by_threshold[
                    threshold
                ].append(filename)

            write_threshold_score_files(
                floader=floader,
                all_image_inds=all_image_inds,
                fit_inds=fit_inds,
                feature_mask=feature_mask,
                pca_checkpoint_dsets=checkpoint_dsets,
                thresholds=thresholds,
                threshold_filenames=threshold_filenames,
                n_trials=n_trials,
                n_prfs=n_prfs,
                actual_max_ncomp=actual_max_ncomp,
                image_set=image_set,
                pca_type=pca_type,
                pca_block=str(pca_block),
                checkpoint_every=checkpoint_every,
                save_dtype=save_dtype,
                compress=compress,
                overwrite_outputs=overwrite_outputs,
            )

            summary_filename = os.path.join(
                checkpoint_dir,
                (
                    f"{image_set}_{n_ori}ori_{n_sf}sf_PCA_{pca_block}_"
                    f"grid{which_prf_grid}_summary.csv"
                ),
            )
            save_pca_summary(
                summary_filename,
                subject=subject,
                image_set=image_set,
                pca_type=pca_type,
                pca_block=str(pca_block),
                n_original_features=n_original_features,
                max_pc=max_pc,
                thresholds=thresholds,
                checkpoint_dsets=checkpoint_dsets,
                n_prfs=n_prfs,
            )

        finally:
            checkpoint_file.close()

    if build_concat:
        for threshold in thresholds:
            print_flush("\n" + "#" * 80)
            print_flush(f"Building/resuming concat output for {threshold}%")
            print_flush("#" * 80)

            write_concat_file_resumable(
                threshold=threshold,
                path_raw=path_raw,
                pca_type=pca_type,
                image_set=image_set,
                n_ori=n_ori,
                n_sf=n_sf,
                which_prf_grid=which_prf_grid,
                pca_filenames=pca_filenames_by_threshold[threshold],
                raw_features_file=raw_features_file,
                feature_type_names=output_feature_type_names,
                save_dtype=save_dtype,
                compress=compress,
                image_chunk_size=image_chunk_size,
                overwrite=overwrite_concat,
                fixed_n_components=fixed_n_components,
            )

    print_flush("\nDone.")


# ============================================================================
# CLI
# ============================================================================
# ============================================================================
# Dedicated pcaAll CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Dedicated resumable pcaAll pipeline: one PCA jointly across "
            "all texture features."
        )
    )

    parser.add_argument(
        "--subject",
        type=int,
        required=True,
    )
    
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        choices=[0, 1],
    )
    
    parser.add_argument(
        "--pca_type",
        type=str,
        required=True,
        choices=["pcaAll", "pcaLL"],
    )
    
    parser.add_argument(
        "--fixed_n_components",
        type=int,
        default=0,
    )
    
    parser.add_argument(
        "--max_pc_to_retain",
        type=int,
        default=0,
    )
    
    parser.add_argument(
        "--which_prf_grid",
        type=int,
        default=5,
    )
    
    parser.add_argument(
        "--thresholds",
        type=str,
        default="40,60,80,95",
        help=(
            "Comma-separated thresholds. Threshold 0 is intentionally omitted "
            "because pcaAll would then contain zero predictors."
        ),
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Flush PCA and threshold files every N completed pRFs.",
    )
    parser.add_argument(
        "--image_chunk_size",
        type=int,
        default=1000,
        help="Number of images per resumable final-concat chunk.",
    )
    parser.add_argument("--compress", type=int, default=0, choices=[0, 1])

    parser.add_argument("--overwrite_pca", type=int, default=0, choices=[0, 1])
    parser.add_argument("--overwrite_outputs", type=int, default=0, choices=[0, 1])
    parser.add_argument("--overwrite_concat", type=int, default=0, choices=[0, 1])
    parser.add_argument("--build_concat", type=int, default=1, choices=[0, 1])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    thresholds = [
        int(value.strip())
        for value in args.thresholds.split(",")
        if value.strip()
    ]

    if 0 in thresholds:
        raise ValueError(
            "Threshold 0 is not valid for pcaAll because no raw features "
            "remain outside PCA, so the result would contain zero predictors."
        )

    max_pc = (
        None
        if args.max_pc_to_retain == 0
        else args.max_pc_to_retain
    )

    fixed_n_components = (
        None
        if args.fixed_n_components == 0
        else int(args.fixed_n_components)
    )
        
    run_pca_texture_pyramid_multi_threshold(
        subject=args.subject,
        image_set=None,
        pca_type=args.pca_type,
        thresholds=thresholds,
        max_pc_to_retain=max_pc,
        fixed_n_components=fixed_n_components,
        debug=bool(args.debug),
        which_prf_grid=args.which_prf_grid,
        checkpoint_every=args.checkpoint_every,
        image_chunk_size=args.image_chunk_size,
        save_dtype=np.float32,
        compress=bool(args.compress),
        overwrite_pca=bool(args.overwrite_pca),
        overwrite_outputs=bool(args.overwrite_outputs),
        overwrite_concat=bool(args.overwrite_concat),
        build_concat=bool(args.build_concat),
    )