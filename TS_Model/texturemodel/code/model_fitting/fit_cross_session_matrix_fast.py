#!/usr/bin/env python3
"""
Fast session-by-session training and full cross-session validation.

This script replaces the old workflow that launched fit_model.py separately for
all train-session x validation-session pairs.

For one subject and one ROI, it:
    1. Loads each of the first N NSD sessions once.
    2. Creates the texture-feature loader once.
    3. Fits one encoding model per training session.
    4. Validates that model on every session in one consolidated pass.
    5. Saves one resumable HDF5 result file.

Designed for the current drift configuration:
    - fixed precomputed NSD/Grid-5 pRF per voxel
    - texture_pyramid features
    - pcaHL 95%
    - OLS (ridge=0)
    - zscore_features=0
    - zscore_betas_within_sess=0
    - one ROI per run: V1, V2, V3, or hV4

The diagonal is intentionally evaluated on all trials from the same session,
matching the current diagonal-validation code. Therefore, diagonal R2 is an
in-sample same-session fit measure, whereas off-diagonal entries are genuine
cross-session generalization measures.
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

# Project imports. Run this script from texturemodel/code/model_fitting,
# exactly as fit_model.py is currently run.
from utils import default_paths, nsd_utils, roi_utils
import arg_parser
import fwrf_model
import initialize_fitting


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Work locally on Colab's SSD and copy the checkpoint to Drive after every
# completed training session. This is much faster than repeatedly writing
# directly through the mounted Google Drive filesystem.
LOCAL_WORK_ROOT = Path("/content/fast_cross_session_work")

# Set to False only when local disk space is insufficient. In that case the
# HDF5 file will be written directly to default_paths.save_fits_path.
USE_LOCAL_CHECKPOINT = True

# Store model weights because they are needed later for population-response and
# RDM simulations. Set False only for a temporary single-voxel drift test.
SAVE_SESSION_WEIGHTS = True

# HDF5 compression usually saves Drive space but can slow writes. The matrices
# are already float32, so no compression is generally faster in Colab.
H5_COMPRESSION: Optional[str] = None

# Flush and copy to Drive after each completed training session.
SYNC_AFTER_EACH_TRAIN_SESSION = True


# -----------------------------------------------------------------------------
# Small containers
# -----------------------------------------------------------------------------

@dataclass
class SessionData:
    """All arrays needed from one NSD session."""

    session_index: int
    voxel_data_all: np.ndarray
    image_order_all: np.ndarray
    voxel_data_train: np.ndarray
    image_order_train: np.ndarray
    voxel_data_holdout: np.ndarray
    image_order_holdout: np.ndarray


@dataclass
class OutputPaths:
    local_file: Path
    drive_file: Path


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def print_flush(*args, **kwargs) -> None:
    print(*args, **kwargs)
    sys.stdout.flush()


def safe_name(text: str) -> str:
    """Create a filesystem-friendly name."""
    keep = []
    for char in str(text):
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def get_output_paths(args, model_name: str) -> OutputPaths:
    """Return local working path and persistent Drive path."""
    subject_dir = Path(default_paths.save_fits_path) / f"S{args.subject:02d}"
    output_dir = subject_dir / "FAST_CROSS_SESSION"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"S{args.subject:02d}_{args.roi_name}_"
        f"{safe_name(model_name)}_"
        f"{args.up_to_sess}x{args.up_to_sess}_cross_session.h5py"
    )
    drive_file = output_dir / filename

    if USE_LOCAL_CHECKPOINT:
        local_dir = LOCAL_WORK_ROOT / f"S{args.subject:02d}" / args.roi_name
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / filename
    else:
        local_file = drive_file

    return OutputPaths(local_file=local_file, drive_file=drive_file)


def restore_checkpoint_if_needed(paths: OutputPaths) -> None:
    """Copy an existing Drive checkpoint to local SSD before resuming."""
    if paths.local_file == paths.drive_file:
        return

    if paths.local_file.exists():
        print_flush("Using existing local checkpoint:", paths.local_file)
        return

    if paths.drive_file.exists():
        print_flush("Restoring Drive checkpoint to local SSD:")
        print_flush("  from:", paths.drive_file)
        print_flush("  to:  ", paths.local_file)
        paths.local_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.drive_file, paths.local_file)


def sync_checkpoint(paths: OutputPaths) -> None:
    """Atomically copy the local HDF5 checkpoint to Drive."""
    if paths.local_file == paths.drive_file:
        return

    paths.drive_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_drive_file = paths.drive_file.with_suffix(
        paths.drive_file.suffix + ".copying"
    )

    if temporary_drive_file.exists():
        temporary_drive_file.unlink()

    shutil.copy2(paths.local_file, temporary_drive_file)
    os.replace(temporary_drive_file, paths.drive_file)
    print_flush("Synced checkpoint to Drive:", paths.drive_file)


def validate_requested_configuration(args) -> None:
    """Fail early for settings this consolidated script does not support."""
    if args.roi_name not in ("V1", "V2", "V3", "hV4"):
        raise ValueError("--roi_name must be one of: V1, V2, V3, hV4")

    if not args.use_precomputed_prfs:
        raise ValueError(
            "This fast cross-session script requires --use_precomputed_prfs 1. "
            "The pRF must remain fixed across sessions."
        )

    if args.which_prf_grid == 0:
        raise ValueError("Use the fixed candidate grid, e.g. --which_prf_grid 5.")

    if args.shuffle_data or args.bootstrap_data:
        raise ValueError(
            "Permutation/bootstrap fitting is not supported in this script. "
            "First generate the empirical 30x30 matrices."
        )

    if args.use_model_residuals:
        raise ValueError("--use_model_residuals is not supported here.")

    if args.trial_subset != "all":
        raise ValueError(
            "This version expects --trial_subset all. Trial-subset masks can be "
            "added later if needed."
        )

    if not args.whole_session_split:
        raise ValueError("Use --whole_session_split 1 for drift analysis.")

    if args.up_to_sess < 1:
        raise ValueError("--up_to_sess must be at least 1.")

    if args.average_image_reps:
        print_flush(
            "WARNING: --average_image_reps is enabled. Your current drift runs "
            "normally use --average_image_reps 0."
        )


# -----------------------------------------------------------------------------
# ROI and session loading
# -----------------------------------------------------------------------------

def load_roi_metadata(args):
    """Load and restrict the voxel mask to the requested retinotopic ROI."""
    voxel_mask, _, voxel_roi, voxel_ncsnr, brain_nii_shape = (
        roi_utils.get_voxel_roi_info(
            args.subject,
            args.volume_space,
            use_kastner_areas=False,
        )
    )
    roi_codes = {
        "V1": [1, 2],
        "V2": [3, 4],
        "V3": [5, 6],
        "hV4": [7],
    }

    retino_roi = voxel_roi[0]
    roi_keep_full = np.isin(retino_roi, roi_codes[args.roi_name])

    voxel_mask = voxel_mask & roi_keep_full
    voxel_index = np.where(voxel_mask)[0][None, :]
    voxel_roi = [rr[voxel_mask] for rr in voxel_roi]
    voxel_ncsnr = voxel_ncsnr[voxel_mask]

    n_voxels = voxel_index.shape[1]
    if n_voxels == 0:
        raise RuntimeError(
            f"No voxels remained for subject {args.subject}, ROI {args.roi_name}."
        )

    print_flush(f"Using subject S{args.subject:02d}, ROI {args.roi_name}")
    print_flush("Number of voxels:", n_voxels)

    return (
        voxel_mask,
        voxel_index,
        voxel_roi,
        voxel_ncsnr,
        brain_nii_shape,
    )


def load_one_session(
    args,
    session_index: int,
    voxel_mask: np.ndarray,
) -> SessionData:
    """
    Load one session once and retain both its fitting split and all trials.

    Training uses the same train/holdout split returned by the existing
    get_data_splits_single_train function. Validation uses every trial in the
    target session, matching the current whole-session diagonal/off-diagonal
    validation behavior.
    """
    print_flush("\n" + "=" * 80)
    print_flush(
        f"Loading session {session_index + 1}/{args.up_to_sess} "
        f"for S{args.subject:02d} {args.roi_name}"
    )
    print_flush("=" * 80)

    voxel_data, image_order, val_inds, holdout_inds, session_inds = (
        nsd_utils.get_data_splits_single_train(
            args.subject,
            sessions=np.asarray([session_index], dtype=int),
            voxel_mask=voxel_mask,
            volume_space=args.volume_space,
            zscore_betas_within_sess=args.zscore_betas_within_sess,
            shuffle_images=args.shuffle_images_once,
            average_image_reps=args.average_image_reps,
            random_voxel_data=args.random_voxel_data,
        )
    )

    voxel_data = np.asarray(voxel_data, dtype=np.float32)
    image_order = np.asarray(image_order, dtype=int)
    val_inds = np.asarray(val_inds, dtype=bool)
    holdout_inds = np.asarray(holdout_inds, dtype=bool)

    if voxel_data.ndim != 2:
        raise RuntimeError(
            f"Session {session_index}: voxel_data has shape {voxel_data.shape}."
        )
    if image_order.ndim != 1:
        image_order = image_order.reshape(-1)
    if voxel_data.shape[0] != image_order.shape[0]:
        raise RuntimeError("voxel_data and image_order trial counts differ.")
    if len(val_inds) != len(image_order) or len(holdout_inds) != len(image_order):
        raise RuntimeError("Split masks do not match the number of session trials.")

    # For the longitudinal drift analysis, each session is one complete
    # training dataset. We use all trials from the session to estimate the
    # session-specific OLS weights.
    #
    # encoding_model.fit() still requires holdout arrays because its original
    # implementation computes a holdout loss. In the present configuration,
    # however, the holdout loss is not used to select either the pRF or lambda:
    #
    #   1. pRF indices are fixed in advance using nsd_grid5.
    #   2. ridge=False, so there is only one lambda value: 0.
    #
    # We therefore provide the same complete session as both the fitting data
    # and the holdout data. The regression weights are estimated from all
    # session trials, as required for the cross-session drift analysis.
    
    train_inds = np.ones(len(image_order), dtype=bool)
    
    voxel_data_train = voxel_data
    image_order_train = image_order
    
    voxel_data_holdout = voxel_data
    image_order_holdout = image_order
    
    print_flush("All session trials:       ", len(image_order))
    print_flush("Regression training trials:", len(image_order_train))
    print_flush(
        "Regression holdout trials: ",
        len(image_order_holdout),
        "(same trials; loss is not used for model selection)",
    )
    
    if np.any(val_inds):
        print_flush(
            "Note: ignoring the returned validation mask containing",
            int(np.sum(val_inds)),
            "trials.",
        )
    
    if np.any(holdout_inds):
        print_flush(
            "Note: ignoring the returned holdout mask containing",
            int(np.sum(holdout_inds)),
            "trials.",
        )
    
    if session_inds is not None:
        unique_session_labels = np.unique(np.asarray(session_inds))
        print_flush("Returned session labels:", unique_session_labels)
    
    return SessionData(
        session_index=session_index,
        voxel_data_all=voxel_data,
        image_order_all=image_order,
        voxel_data_train=voxel_data_train,
        image_order_train=image_order_train,
        voxel_data_holdout=voxel_data_holdout,
        image_order_holdout=image_order_holdout,
    )


def load_all_sessions(args, voxel_mask: np.ndarray) -> List[SessionData]:
    """Load every requested session exactly once."""
    sessions: List[SessionData] = []
    for session_index in range(args.up_to_sess):
        sessions.append(load_one_session(args, session_index, voxel_mask))
        gc.collect()
    return sessions


# -----------------------------------------------------------------------------
# Fast validation
# -----------------------------------------------------------------------------

def r2_columns(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Cross-validated R2 independently for each voxel/column."""
    residual = y_true - y_pred
    sse = np.sum(residual * residual, axis=0, dtype=np.float64)

    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    sst = np.sum(centered * centered, axis=0, dtype=np.float64)

    result = np.full(y_true.shape[1], np.nan, dtype=np.float32)
    valid = sst > 0
    result[valid] = (1.0 - sse[valid] / sst[valid]).astype(np.float32)
    return result


def corr_columns(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pearson correlation independently for each voxel/column."""
    true_centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    pred_centered = y_pred - np.mean(y_pred, axis=0, keepdims=True)

    numerator = np.sum(
        true_centered * pred_centered,
        axis=0,
        dtype=np.float64,
    )
    denominator = np.sqrt(
        np.sum(true_centered * true_centered, axis=0, dtype=np.float64)
        * np.sum(pred_centered * pred_centered, axis=0, dtype=np.float64)
    )

    result = np.full(y_true.shape[1], np.nan, dtype=np.float32)
    valid = denominator > 0
    result[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
    return result


def concatenate_validation_sessions(
    sessions: Sequence[SessionData],
) -> Tuple[np.ndarray, List[slice]]:
    """Concatenate image indices and return one slice per validation session."""
    image_arrays = []
    session_slices: List[slice] = []
    start = 0

    for session in sessions:
        image_arrays.append(session.image_order_all)
        end = start + len(session.image_order_all)
        session_slices.append(slice(start, end))
        start = end

    return np.concatenate(image_arrays, axis=0), session_slices


def validate_model_on_all_sessions(
    model: fwrf_model.encoding_model,
    sessions: Sequence[SessionData],
    args,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate one fitted train-session model on all sessions efficiently.

    The old implementation called model.validate separately for each validation
    session. That reloaded every pRF feature block once per validation session.
    Here, all validation image indices are concatenated, so each pRF feature
    block is loaded only once for the entire 30-session validation matrix row.

    Returns
    -------
    val_r2_row : [n_val_sessions, n_voxels, n_partial_versions]
    val_cc_row : [n_val_sessions, n_voxels, n_partial_versions]
    """
    all_image_inds, session_slices = concatenate_validation_sessions(sessions)

    n_val_sessions = len(sessions)
    n_voxels = model.best_weights.shape[0]
    n_partial_versions = model.n_partial_versions

    val_r2_row = np.full(
        (n_val_sessions, n_voxels, n_partial_versions),
        np.nan,
        dtype=np.float32,
    )
    val_cc_row = np.full_like(val_r2_row, np.nan)

    # The fixed pRF should be identical across all partial versions. Use column
    # zero for grouping voxels by pRF.
    #
    # In debug mode, fwrf_model.fit() processes only pRFs 0 and 1. Voxels assigned
    # to all other pRFs remain at the initialization value -1. These voxels must
    # not be passed to feature_loader.load(), because -1 is not a valid pRF index.
    best_prf_per_voxel = np.asarray(
        model.best_prf_models[:, 0],
        dtype=int,
    )
    
    valid_fitted_voxels = best_prf_per_voxel >= 0
    unique_prfs = np.unique(
        best_prf_per_voxel[valid_fitted_voxels]
    )
    
    print_flush(
        f"Valid fitted voxels: "
        f"{np.sum(valid_fitted_voxels)}/{len(valid_fitted_voxels)}"
    )
    
    if np.any(~valid_fitted_voxels):
        print_flush(
            f"Skipping {np.sum(~valid_fitted_voxels)} voxels whose fitted "
            "pRF index is -1. This is expected in debug mode."
        )
    
    print_flush(
        f"Validating all {n_val_sessions} sessions in one pass; "
        f"{len(all_image_inds)} total validation trials; "
        f"{len(unique_prfs)} occupied valid pRFs."
    )

    model.feature_loader.clear_big_features()

    for prf_counter, prf_model_index in enumerate(unique_prfs, start=1):
        voxel_inds_for_prf = np.flatnonzero(
            best_prf_per_voxel == prf_model_index
        )
        if voxel_inds_for_prf.size == 0:
            continue

        print_flush(
            f"Validation pRF {prf_counter}/{len(unique_prfs)}: "
            f"grid index {prf_model_index}; "
            f"voxels={len(voxel_inds_for_prf)}"
        )

        features_actual, feature_inds_defined = model.feature_loader.load(
            all_image_inds,
            prf_model_index=int(prf_model_index),
        )
        features_actual = np.asarray(features_actual, dtype=np.float32)
        feature_inds_defined = np.asarray(feature_inds_defined, dtype=bool)

        if feature_inds_defined.shape[0] != model.max_features:
            raise RuntimeError(
                "feature_inds_defined length does not equal model.max_features."
            )
        if features_actual.shape[1] != int(np.sum(feature_inds_defined)):
            raise RuntimeError(
                "Loaded feature columns do not match feature_inds_defined."
            )

        if model.zscore:
            train_mean = np.asarray(
                model.features_mean[prf_model_index, feature_inds_defined],
                dtype=np.float32,
            )
            train_std = np.asarray(
                model.features_std[prf_model_index, feature_inds_defined],
                dtype=np.float32,
            )
            features_actual = (features_actual - train_mean[None, :]) / train_std[None, :]

        n_voxel_batches = int(
            np.ceil(len(voxel_inds_for_prf) / args.voxel_batch_size)
        )

        for batch_index in range(n_voxel_batches):
            batch_start = batch_index * args.voxel_batch_size
            batch_end = min(
                batch_start + args.voxel_batch_size,
                len(voxel_inds_for_prf),
            )
            voxel_batch = voxel_inds_for_prf[batch_start:batch_end]

            for partial_index in range(n_partial_versions):
                partial_mask_full = np.asarray(
                    model.masks[: model.max_features, partial_index],
                    dtype=bool,
                )

                # feature_loader.load returns only defined columns. Convert the
                # full max-feature mask to the corresponding actual columns.
                actual_partial_mask = partial_mask_full[feature_inds_defined]
                features_use = features_actual[:, actual_partial_mask]

                weights_full = np.asarray(
                    model.best_weights[voxel_batch, :, partial_index],
                    dtype=np.float32,
                )
                weights_use = weights_full[:, feature_inds_defined]
                weights_use = weights_use[:, actual_partial_mask]

                predictions_all = features_use @ weights_use.T

                if model.best_biases is not None:
                    biases = np.asarray(
                        model.best_biases[voxel_batch, partial_index],
                        dtype=np.float32,
                    )
                    predictions_all += biases[None, :]

                for val_session_index, session_slice in enumerate(session_slices):
                    y_true = sessions[val_session_index].voxel_data_all[
                        :, voxel_batch
                    ]
                    y_pred = predictions_all[session_slice, :]

                    val_r2_row[
                        val_session_index,
                        voxel_batch,
                        partial_index,
                    ] = r2_columns(y_true, y_pred)

                    if args.do_corrcoef:
                        val_cc_row[
                            val_session_index,
                            voxel_batch,
                            partial_index,
                        ] = corr_columns(y_true, y_pred)

                del predictions_all, weights_full, weights_use, features_use

            gc.collect()

        del features_actual
        gc.collect()

    model.feature_loader.clear_big_features()
    gc.collect()

    return val_r2_row, val_cc_row


# -----------------------------------------------------------------------------
# HDF5 output
# -----------------------------------------------------------------------------

def create_dataset(
    h5: h5py.File,
    name: str,
    shape: Tuple[int, ...],
    dtype,
    *,
    fillvalue=None,
    data=None,
):
    kwargs = {
        "dtype": dtype,
    }
    if data is not None:
        kwargs["data"] = data
    else:
        kwargs["shape"] = shape
    if fillvalue is not None and data is None:
        kwargs["fillvalue"] = fillvalue
    if H5_COMPRESSION is not None and data is None:
        kwargs["compression"] = H5_COMPRESSION
    return h5.create_dataset(name, **kwargs)


def open_or_create_output(
    path: Path,
    *,
    args,
    model_name: str,
    fitting_types: Sequence[str],
    n_voxels: int,
    n_prfs: int,
    max_features: int,
    n_partial_versions: int,
    voxel_mask: np.ndarray,
    voxel_index: np.ndarray,
    voxel_roi: Sequence[np.ndarray],
    voxel_ncsnr: np.ndarray,
    brain_nii_shape,
    prf_models: np.ndarray,
    best_model_each_voxel: np.ndarray,
    saved_prfs_fn,
    partial_version_names: Sequence[str],
) -> h5py.File:
    """Create a new result file or validate an existing resumable file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    h5 = h5py.File(path, "r+" if exists else "w")

    expected_matrix_shape = (
        args.up_to_sess,
        args.up_to_sess,
        n_voxels,
        n_partial_versions,
    )

    if not exists:
        print_flush("Creating consolidated output:", path)

        h5.attrs["subject"] = int(args.subject)
        h5.attrs["roi_name"] = str(args.roi_name)
        h5.attrs["model_name"] = str(model_name)
        h5.attrs["fitting_types"] = np.asarray(
            list(fitting_types), dtype=h5py.string_dtype("utf-8")
        )
        h5.attrs["up_to_sess"] = int(args.up_to_sess)
        h5.attrs["which_prf_grid"] = int(args.which_prf_grid)
        h5.attrs["pca_pct_var"] = int(args.pca_pct_var)
        h5.attrs["pyr_pca_type"] = str(args.pyr_pca_type)
        h5.attrs["pyr_fixed_n_components"] = int(args.pyr_fixed_n_components)
        h5.attrs["zscore_features"] = int(args.zscore_features)
        h5.attrs["zscore_betas_within_sess"] = int(
            args.zscore_betas_within_sess
        )
        h5.attrs["ridge"] = int(args.ridge)
        h5.attrs["set_lambda_per_group"] = int(args.set_lambda_per_group)
        h5.attrs["average_image_reps"] = int(args.average_image_reps)
        h5.attrs["diagonal_definition"] = (
            "same-session model evaluated on all trials from its training session"
        )
        h5.attrs["saved_prfs_fn"] = "" if saved_prfs_fn is None else str(saved_prfs_fn)
        h5.attrs["created_time"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )

        create_dataset(
            h5,
            "val_r2",
            expected_matrix_shape,
            np.float32,
            fillvalue=np.nan,
        )
        create_dataset(
            h5,
            "val_cc",
            expected_matrix_shape,
            np.float32,
            fillvalue=np.nan,
        )
        create_dataset(
            h5,
            "completed_train_sessions",
            (args.up_to_sess,),
            np.uint8,
            fillvalue=0,
        )

        create_dataset(
            h5,
            "best_losses",
            (args.up_to_sess, n_voxels, n_partial_versions),
            np.float32,
            fillvalue=np.nan,
        )
        create_dataset(
            h5,
            "best_lambdas",
            (args.up_to_sess, n_voxels, n_partial_versions),
            np.int32,
            fillvalue=-1,
        )
        create_dataset(
            h5,
            "best_prf_models",
            (args.up_to_sess, n_voxels, n_partial_versions),
            np.int32,
            fillvalue=-1,
        )

        if SAVE_SESSION_WEIGHTS:
            create_dataset(
                h5,
                "best_weights",
                (
                    args.up_to_sess,
                    n_voxels,
                    max_features,
                    n_partial_versions,
                ),
                np.float32,
                fillvalue=np.nan,
            )
            create_dataset(
                h5,
                "best_biases",
                (args.up_to_sess, n_voxels, n_partial_versions),
                np.float32,
                fillvalue=np.nan,
            )

        # Static metadata.
        h5.create_dataset("voxel_mask", data=np.asarray(voxel_mask, dtype=np.uint8))
        h5.create_dataset("voxel_index", data=np.asarray(voxel_index, dtype=np.int64))
        h5.create_dataset("voxel_ncsnr", data=np.asarray(voxel_ncsnr, dtype=np.float32))
        h5.create_dataset("prf_models", data=np.asarray(prf_models, dtype=np.float32))
        h5.create_dataset(
            "fixed_best_model_each_voxel",
            data=np.asarray(best_model_each_voxel, dtype=np.int32),
        )
        h5.create_dataset(
            "brain_nii_shape",
            data=np.asarray(brain_nii_shape, dtype=np.int64),
        )

        roi_group = h5.create_group("voxel_roi")
        for roi_index, roi_values in enumerate(voxel_roi):
            roi_group.create_dataset(
                str(roi_index),
                data=np.asarray(roi_values),
            )

        h5.create_dataset(
            "partial_version_names",
            data=np.asarray(
                list(partial_version_names),
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        h5.flush()

    else:
        print_flush("Resuming consolidated output:", path)

        checks = {
            "subject": int(args.subject),
            "roi_name": str(args.roi_name),
            "up_to_sess": int(args.up_to_sess),
            "which_prf_grid": int(args.which_prf_grid),
            "pca_pct_var": int(args.pca_pct_var),
            "pyr_pca_type": str(args.pyr_pca_type),
            "pyr_fixed_n_components": int(args.pyr_fixed_n_components),
        }
        for key, expected in checks.items():
            saved = h5.attrs[key]
            if isinstance(saved, bytes):
                saved = saved.decode("utf-8")
            if saved != expected:
                h5.close()
                raise RuntimeError(
                    f"Checkpoint mismatch for {key}: saved={saved}, "
                    f"requested={expected}. Delete or rename the old file."
                )

        saved_val_r2_shape = tuple(h5["val_r2"].shape)

        if saved_val_r2_shape != expected_matrix_shape:
            h5.close()
            raise RuntimeError(
                f"Checkpoint val_r2 shape {saved_val_r2_shape} does not match "
                f"expected {expected_matrix_shape}. "
                "Delete or rename the old checkpoint."
            )
    return h5


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def subset_kastner_prfs_to_visualroi(
    args,
    best_model_each_voxel: np.ndarray,
    visualroi_voxel_index: np.ndarray,
) -> np.ndarray:
    """
    Subset pRF indices saved in Kastner-extended ROI voxel order to the
    stricter NSD visualrois voxel order.

    The original pRF-index files were generated using:
        use_kastner_areas=True

    The current drift analysis uses:
        use_kastner_areas=False
    """

    roi_codes = {
        "V1": [1, 2],
        "V2": [3, 4],
        "V3": [5, 6],
        "hV4": [7],
    }

    # Reconstruct the voxel ordering used when the existing pRF-index
    # file was generated.
    (
        kastner_voxel_mask,
        _,
        kastner_voxel_roi,
        _,
        _,
    ) = roi_utils.get_voxel_roi_info(
        args.subject,
        args.volume_space,
        use_kastner_areas=True,
    )

    kastner_retino_roi = kastner_voxel_roi[0]
    kastner_roi_keep = np.isin(
        kastner_retino_roi,
        roi_codes[args.roi_name],
    )

    kastner_roi_mask = kastner_voxel_mask & kastner_roi_keep
    kastner_voxel_indices = np.flatnonzero(kastner_roi_mask)

    visualroi_voxel_indices = np.asarray(
        visualroi_voxel_index,
        dtype=int,
    ).ravel()

    print_flush(
        "Kastner ROI voxel count:",
        len(kastner_voxel_indices),
    )
    print_flush(
        "Visualrois voxel count:",
        len(visualroi_voxel_indices),
    )
    print_flush(
        "Saved pRF-index count:",
        len(best_model_each_voxel),
    )

    if len(best_model_each_voxel) != len(kastner_voxel_indices):
        raise RuntimeError(
            "The saved pRF file does not match the reconstructed "
            "Kastner ROI ordering: "
            f"{len(best_model_each_voxel)} pRF indices versus "
            f"{len(kastner_voxel_indices)} Kastner ROI voxels."
        )

    # Find each visualrois voxel inside the Kastner voxel ordering.
    positions = np.searchsorted(
        kastner_voxel_indices,
        visualroi_voxel_indices,
    )

    valid_positions = positions < len(kastner_voxel_indices)

    exact_match = np.zeros(
        len(visualroi_voxel_indices),
        dtype=bool,
    )
    exact_match[valid_positions] = (
        kastner_voxel_indices[positions[valid_positions]]
        == visualroi_voxel_indices[valid_positions]
    )

    if not np.all(exact_match):
        missing_voxels = visualroi_voxel_indices[~exact_match]

        raise RuntimeError(
            f"{len(missing_voxels)} visualrois voxels were not found "
            "inside the Kastner ROI mask. The existing pRF-index file "
            "cannot be safely reused and must be regenerated using "
            "the visualrois mask."
        )

    subset_prf_indices = np.asarray(
        best_model_each_voxel[positions],
        dtype=int,
    )

    print_flush(
        "Successfully subset Kastner pRF indices to visualrois:",
        len(subset_prf_indices),
    )

    return subset_prf_indices


def run_fast_cross_session(args) -> None:
    validate_requested_configuration(args)

    try:
        device = initialize_fitting.init_cuda()
    except Exception as exc:
        print_flush("CUDA initialization failed; using CPU:", repr(exc))
        device = "cpu:0"

    print_flush("NumPy version:", np.__version__)
    print_flush("Device:", device)

    model_name, fitting_types = initialize_fitting.get_full_save_name(args)
    print_flush("Model name:", model_name)
    print_flush("Fitting types:", fitting_types)

    (
        voxel_mask,
        voxel_index,
        voxel_roi,
        voxel_ncsnr,
        brain_nii_shape,
    ) = load_roi_metadata(args)
    n_voxels = voxel_index.shape[1]

    # Load all session betas only once for this subject/ROI run.
    sessions = load_all_sessions(args, voxel_mask)

    # Create the feature loader only once and reuse its HDF5 setup throughout.
    feature_loader = initialize_fitting.make_feature_loaders(
        args,
        fitting_types,
        vi=0,
        dnn_layers_use=None,
    )

    max_features = int(feature_loader.max_features)
    partial_masks, partial_version_names = feature_loader.get_partial_versions()
    n_partial_versions = len(partial_version_names)

    print_flush("Feature loader max_features:", max_features)
    print_flush("Partial versions:", partial_version_names)
    print_flush("Partial feature counts:", np.sum(partial_masks, axis=1))

    lambdas = initialize_fitting.get_lambdas(
        fitting_types=fitting_types,
        zscore_features=args.zscore_features,
        ridge=args.ridge,
    )
    print_flush("Lambdas:", lambdas)

    prf_models = initialize_fitting.get_prf_models(
        which_grid=args.which_prf_grid,
        verbose=True,
    )
    n_prfs = int(prf_models.shape[0])

    best_model_each_voxel, saved_prfs_fn = (
        initialize_fitting.load_precomputed_prfs(
            args.subject,
            args,
        )
    )
    best_model_each_voxel = np.asarray(
        best_model_each_voxel,
        dtype=int,
    ).ravel()
    
    if len(best_model_each_voxel) != n_voxels:
        print_flush(
            "Saved pRF indices were created using the Kastner-extended ROI. "
            "Subsetting them to the current visualrois-only voxel mask."
        )
    
        best_model_each_voxel = subset_kastner_prfs_to_visualroi(
            args=args,
            best_model_each_voxel=best_model_each_voxel,
            visualroi_voxel_index=voxel_index,
        )
    
    if len(best_model_each_voxel) != n_voxels:
        raise RuntimeError(
            "pRF count still does not match the visualrois voxel count "
            "after subsetting: "
            f"{len(best_model_each_voxel)} vs {n_voxels}."
        )
    if np.any(best_model_each_voxel < 0) or np.any(best_model_each_voxel >= n_prfs):
        raise RuntimeError("Precomputed pRF indices contain out-of-range values.")

    paths = get_output_paths(args, model_name)
    restore_checkpoint_if_needed(paths)

    output = open_or_create_output(
        paths.local_file,
        args=args,
        model_name=model_name,
        fitting_types=fitting_types,
        n_voxels=n_voxels,
        n_prfs=n_prfs,
        max_features=max_features,
        n_partial_versions=n_partial_versions,
        voxel_mask=voxel_mask,
        voxel_index=voxel_index,
        voxel_roi=voxel_roi,
        voxel_ncsnr=voxel_ncsnr,
        brain_nii_shape=brain_nii_shape,
        prf_models=prf_models,
        best_model_each_voxel=best_model_each_voxel,
        saved_prfs_fn=saved_prfs_fn,
        partial_version_names=partial_version_names,
    )

    try:
        completed = np.asarray(
            output["completed_train_sessions"],
            dtype=bool,
        )
        train_sessions_to_run = np.flatnonzero(~completed)

        print_flush("\nCompleted train sessions:", np.flatnonzero(completed) + 1)
        print_flush("Train sessions remaining:", train_sessions_to_run + 1)

        if len(train_sessions_to_run) == 0:
            print_flush("All requested training sessions are already complete.")
            output.flush()
            sync_checkpoint(paths)
            return

        for run_number, train_session_index in enumerate(
            train_sessions_to_run,
            start=1,
        ):
            train_data = sessions[int(train_session_index)]

            print_flush("\n" + "#" * 90)
            print_flush(
                f"TRAIN SESSION {train_session_index + 1}/{args.up_to_sess} "
                f"(remaining run {run_number}/{len(train_sessions_to_run)})"
            )
            print_flush("#" * 90)

            model = fwrf_model.encoding_model(
                feature_loader,
                lambdas=lambdas,
                best_model_each_voxel=best_model_each_voxel,
                zscore=args.zscore_features,
                add_bias=True,
                set_lambda_per_group=args.set_lambda_per_group,
                voxel_batch_size=args.voxel_batch_size,
                sample_batch_size=args.sample_batch_size,
                device=device,
                prfs_fit_mask=None,
                shuffle_data=False,
                bootstrap_data=False,
                do_corrcoef=args.do_corrcoef,
                dtype=np.float32,
                debug=args.debug,
            )

            fit_start = time.time()
            model.fit(
                train_data.image_order_train,
                train_data.voxel_data_train,
                train_data.image_order_holdout,
                train_data.voxel_data_holdout,
                trials_use_each_prf_trn=None,
                trials_use_each_prf_holdout=None,
            )
            fit_minutes = (time.time() - fit_start) / 60.0
            print_flush(
                f"Finished fitting session {train_session_index + 1} "
                f"in {fit_minutes:.2f} minutes."
            )

            validation_start = time.time()
            val_r2_row, val_cc_row = validate_model_on_all_sessions(
                model,
                sessions,
                args,
            )
            validation_minutes = (time.time() - validation_start) / 60.0
            print_flush(
                f"Finished all-session validation for train session "
                f"{train_session_index + 1} in "
                f"{validation_minutes:.2f} minutes."
            )

            # Write results first.
            output["val_r2"][train_session_index, :, :, :] = val_r2_row
            output["val_cc"][train_session_index, :, :, :] = val_cc_row
            output["best_losses"][train_session_index, :, :] = (
                np.asarray(model.best_losses, dtype=np.float32)
            )
            output["best_lambdas"][train_session_index, :, :] = (
                np.asarray(model.best_lambdas, dtype=np.int32)
            )
            output["best_prf_models"][train_session_index, :, :] = (
                np.asarray(model.best_prf_models, dtype=np.int32)
            )

            if SAVE_SESSION_WEIGHTS:
                output["best_weights"][train_session_index, :, :, :] = (
                    np.asarray(model.best_weights, dtype=np.float32)
                )
                output["best_biases"][train_session_index, :, :] = (
                    np.asarray(model.best_biases, dtype=np.float32)
                )

            # Completion flag is deliberately written last. A session is treated
            # as complete only after all metrics and parameters were persisted.
            output["completed_train_sessions"][train_session_index] = 1
            output.attrs["last_completed_train_session"] = int(
                train_session_index
            )
            output.attrs["last_update_time"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime()
            )
            output.flush()

            if SYNC_AFTER_EACH_TRAIN_SESSION:
                sync_checkpoint(paths)

            # Print a compact sanity check using median across voxels for the full
            # model (partial version zero).
            median_row = np.nanmedian(val_r2_row[:, :, 0], axis=1)
            print_flush(
                "Median R2 across voxels for this train-session row:\n",
                np.array2string(median_row, precision=5, suppress_small=False),
            )

            del model, val_r2_row, val_cc_row
            feature_loader.clear_big_features()
            gc.collect()

        output.flush()

    finally:
        output.close()
        feature_loader.clear_big_features()
        gc.collect()

    sync_checkpoint(paths)
    print_flush("\nDONE")
    print_flush("Persistent result file:", paths.drive_file)


if __name__ == "__main__":
    parsed_args = arg_parser.get_args()

    # This script controls all session loops internally. These two values are
    # ignored even if present in a reused bash command.
    parsed_args.single_sess = -1
    parsed_args.single_val_sess = -1
    parsed_args.from_scratch = True
    parsed_args.do_val = True

    run_fast_cross_session(parsed_args)
