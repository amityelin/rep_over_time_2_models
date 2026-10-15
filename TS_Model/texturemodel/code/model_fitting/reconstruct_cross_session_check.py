#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from utils import nsd_utils, roi_utils
import arg_parser as project_arg_parser
import initialize_fitting


# ============================================================
# Helpers
# ============================================================

def build_project_args(
    subject,
    roi_name,
    pca_type,
    pca_pct_var,
):
    """
    Recreate the same model configuration used by
    fit_cross_session_matrix_fast.py.
    """

    old_argv = sys.argv.copy()

    sys.argv = [
        "reconstruction_test",

        "--subject", str(subject),
        "--roi_name", roi_name,

        "--debug", "0",
        "--up_to_sess", "30",

        "--whole_session_split", "1",
        "--average_image_reps", "0",

        "--sample_batch_size", "1000",
        "--voxel_batch_size", "1000",

        "--zscore_features", "0",
        "--zscore_betas_within_sess", "0",

        "--use_precomputed_prfs", "1",
        "--prfs_model_name", "nsd_grid5",
        "--which_prf_grid", "5",

        "--fitting_type", "texture_pyramid",

        "--n_ori_pyr", "4",
        "--n_sf_pyr", "4",

        "--group_all_hl_feats", "1",

        "--pyr_pca_type", pca_type,
        "--pca_pct_var", str(pca_pct_var),

        "--set_lambda_per_group", "0",
        "--ridge", "0",

        "--do_val", "1",
        "--do_corrcoef", "1",

        "--do_tuning", "0",
        "--do_sem_disc", "0",
        "--do_pyr_varpart", "0",

        "--trial_subset", "all",
    ]

    try:
        args = project_arg_parser.get_args()
    finally:
        sys.argv = old_argv

    return args


def get_roi_mask(args):
    """
    Reproduce the exact visual ROI masking from
    fit_cross_session_matrix_fast.py.
    """

    voxel_mask, _, voxel_roi, _, _ = (
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

    roi_keep_full = np.isin(
        retino_roi,
        roi_codes[args.roi_name],
    )

    voxel_mask = voxel_mask & roi_keep_full

    return voxel_mask


def r2_1d(y_true, y_pred):
    """
    Same numerical definition as r2_columns() in
    fit_cross_session_matrix_fast.py.
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    residual = y_true - y_pred

    sse = np.sum(
        residual * residual,
        dtype=np.float64,
    )

    centered = (
        y_true
        - np.mean(
            y_true,
            dtype=np.float32,
        )
    )

    sst = np.sum(
        centered * centered,
        dtype=np.float64,
    )

    if sst <= 0:
        return np.nan

    return float(
        1.0 - sse / sst
    )


def corr_1d(y_true, y_pred):
    """
    Same numerical definition as corr_columns() in
    fit_cross_session_matrix_fast.py.
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    true_centered = y_true - np.mean(y_true)
    pred_centered = y_pred - np.mean(y_pred)

    numerator = np.sum(
        true_centered * pred_centered,
        dtype=np.float64,
    )

    denominator = np.sqrt(
        np.sum(
            true_centered * true_centered,
            dtype=np.float64,
        )
        *
        np.sum(
            pred_centered * pred_centered,
            dtype=np.float64,
        )
    )

    if denominator <= 0:
        return np.nan

    return float(
        numerator / denominator
    )


def get_fit_file(
    root,
    subject,
    roi_name,
    pca_type,
    pca_pct_var,
):
    s = f"S{subject:02d}"

    return (
        root
        / "texturemodel"
        / "model_fits"
        / s
        / "FAST_CROSS_SESSION"
        / (
            f"{s}_{roi_name}_"
            f"texture_pyramid_OLS_4ori_4sf_"
            f"{pca_type}_{roi_name}_pca{pca_pct_var}_"
            f"use_nsd_grid5_pRFs_"
            f"30x30_cross_session.h5py"
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subject",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--roi_name",
        type=str,
        required=True,
        choices=["V1", "V2", "V3", "hV4"],
    )

    parser.add_argument(
        "--train_session",
        type=int,
        required=True,
        help="0-based session index",
    )

    parser.add_argument(
        "--test_session",
        type=int,
        required=True,
        help="0-based session index",
    )

    parser.add_argument(
        "--voxel",
        type=int,
        required=True,
        help="0-based voxel index within ROI",
    )

    parser.add_argument(
        "--pyr_pca_type",
        type=str,
        default="pcaHL",
    )

    parser.add_argument(
        "--pca_pct_var",
        type=int,
        default=95,
    )

    parser.add_argument(
        "--root",
        type=str,
        default="/content/drive/MyDrive/V2_drift",
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
    )

    cli = parser.parse_args()

    root = Path(cli.root)

    print("\n" + "=" * 90)
    print("EXACT CROSS-SESSION RECONSTRUCTION CHECK")
    print("=" * 90)

    print("Subject      :", cli.subject)
    print("ROI          :", cli.roi_name)
    print("Train session:", cli.train_session + 1)
    print("Test session :", cli.test_session + 1)
    print("Voxel        :", cli.voxel)
    print("PCA type     :", cli.pyr_pca_type)
    print("PCA pct var  :", cli.pca_pct_var)


    # ========================================================
    # Rebuild exact fitting configuration
    # ========================================================
    args = build_project_args(
        subject=cli.subject,
        roi_name=cli.roi_name,
        pca_type=cli.pyr_pca_type,
        pca_pct_var=cli.pca_pct_var,
    )


    # ========================================================
    # ROI mask
    # ========================================================
    voxel_mask = get_roi_mask(args)

    nvox = int(
        np.sum(voxel_mask)
    )

    print("\nROI voxel count:", nvox)

    if cli.voxel < 0 or cli.voxel >= nvox:
        raise ValueError(
            f"Voxel {cli.voxel} outside ROI range 0..{nvox-1}"
        )


    # ========================================================
    # Load exact test session
    # ========================================================
    (
        voxel_data,
        image_order,
        val_inds,
        holdout_inds,
        session_inds,
    ) = nsd_utils.get_data_splits_single_train(
        args.subject,
        sessions=np.asarray(
            [cli.test_session],
            dtype=int,
        ),
        voxel_mask=voxel_mask,
        volume_space=args.volume_space,
        zscore_betas_within_sess=args.zscore_betas_within_sess,
        shuffle_images=args.shuffle_images_once,
        average_image_reps=args.average_image_reps,
        random_voxel_data=args.random_voxel_data,
    )

    voxel_data = np.asarray(
        voxel_data,
        dtype=np.float32,
    )

    image_order = np.asarray(
        image_order,
        dtype=int,
    ).reshape(-1)

    print("\nTest trials:", len(image_order))
    print("voxel_data shape:", voxel_data.shape)
    print("image_order shape:", image_order.shape)
    print("First image indices:", image_order[:10])


    # ========================================================
    # Feature loader
    # ========================================================
    model_name, fitting_types = (
        initialize_fitting.get_full_save_name(args)
    )

    print("\nModel name:")
    print(model_name)

    feature_loader = initialize_fitting.make_feature_loaders(
        args,
        fitting_types,
        vi=0,
        dnn_layers_use=None,
    )

    print(
        "Feature loader max_features:",
        feature_loader.max_features,
    )


    partial_masks, partial_version_names = (
        feature_loader.get_partial_versions()
    )

    partial_masks = np.asarray(partial_masks)

    print(
        "Partial versions:",
        partial_version_names,
    )

    print(
        "Partial mask shape:",
        partial_masks.shape,
    )

    n_partial_versions = len(
        partial_version_names
    )

    if (
        partial_masks.ndim == 2
        and partial_masks.shape[0]
        == n_partial_versions
    ):
        partial_mask_full = np.asarray(
            partial_masks[0],
            dtype=bool,
        )

    elif (
        partial_masks.ndim == 2
        and partial_masks.shape[1]
        == n_partial_versions
    ):
        partial_mask_full = np.asarray(
            partial_masks[:, 0],
            dtype=bool,
        )

    else:
        raise RuntimeError(
            f"Could not infer partial-mask orientation: "
            f"{partial_masks.shape}"
        )

    partial_mask_full = (
        partial_mask_full[
            :feature_loader.max_features
        ]
    )


    # ========================================================
    # Saved model
    # ========================================================
    fit_fn = get_fit_file(
        root=root,
        subject=cli.subject,
        roi_name=cli.roi_name,
        pca_type=cli.pyr_pca_type,
        pca_pct_var=cli.pca_pct_var,
    )

    print("\nFit file:")
    print(fit_fn)

    if not fit_fn.exists():
        raise FileNotFoundError(fit_fn)


    with h5py.File(fit_fn, "r") as f:

        prf = int(
            f["best_prf_models"][
                cli.train_session,
                cli.voxel,
                0,
            ]
        )

        fixed_prf = int(
            f["fixed_best_model_each_voxel"][
                cli.voxel
            ]
        )

        w_full = np.asarray(
            f["best_weights"][
                cli.train_session,
                cli.voxel,
                :,
                0,
            ],
            dtype=np.float32,
        )

        bias = np.float32(
            f["best_biases"][
                cli.train_session,
                cli.voxel,
                0,
            ]
        )

        saved_r2 = float(
            f["val_r2"][
                cli.train_session,
                cli.test_session,
                cli.voxel,
                0,
            ]
        )

        saved_cc = float(
            f["val_cc"][
                cli.train_session,
                cli.test_session,
                cli.voxel,
                0,
            ]
        )


    print("\nSaved pRF:", prf)
    print("Fixed pRF:", fixed_prf)

    if prf != fixed_prf:
        raise RuntimeError(
            "Saved best pRF and fixed pRF do not match."
        )

    print("✓ pRF indices agree")


    # ========================================================
    # Exact feature loading
    # ========================================================
    features_actual, feature_inds_defined = (
        feature_loader.load(
            image_order,
            prf_model_index=prf,
        )
    )

    features_actual = np.asarray(
        features_actual,
        dtype=np.float32,
    )

    feature_inds_defined = np.asarray(
        feature_inds_defined,
        dtype=bool,
    )

    print("\nLoaded features:")
    print(
        "features_actual:",
        features_actual.shape,
    )

    print(
        "feature_inds_defined:",
        feature_inds_defined.shape,
    )

    print(
        "defined features:",
        int(feature_inds_defined.sum()),
    )


    if (
        len(feature_inds_defined)
        != feature_loader.max_features
    ):
        raise RuntimeError(
            "feature_inds_defined length mismatch."
        )

    if (
        features_actual.shape[1]
        != int(feature_inds_defined.sum())
    ):
        raise RuntimeError(
            "Loaded feature count mismatch."
        )


    # ========================================================
    # Same partial-feature selection as validation
    # ========================================================
    actual_partial_mask = (
        partial_mask_full[
            feature_inds_defined
        ]
    )

    X = features_actual[
        :,
        actual_partial_mask
    ]

    w = w_full[
        feature_inds_defined
    ]

    w = w[
        actual_partial_mask
    ]


    print("\nAfter feature masking:")
    print("X:", X.shape)
    print("w:", w.shape)


    if X.shape[1] != w.shape[0]:
        raise RuntimeError(
            "Feature/weight width mismatch."
        )


    # ========================================================
    # Exact prediction formula
    # ========================================================
    pred = (
        X @ w
        + bias
    ).astype(
        np.float32
    )


    y_true = voxel_data[
        :,
        cli.voxel
    ].astype(
        np.float32
    )


    # ========================================================
    # Scale diagnostic
    # ========================================================
    print("\n" + "=" * 90)
    print("RESPONSE SCALE")
    print("=" * 90)

    print(
        "Measured beta range:",
        float(np.min(y_true)),
        "to",
        float(np.max(y_true)),
    )

    print(
        "Measured beta mean/std:",
        float(np.mean(y_true)),
        float(np.std(y_true)),
    )

    print(
        "\nPredicted range:",
        float(np.min(pred)),
        "to",
        float(np.max(pred)),
    )

    print(
        "Predicted mean/std:",
        float(np.mean(pred)),
        float(np.std(pred)),
    )


    # ========================================================
    # Recompute performance
    # ========================================================
    reconstructed_r2 = r2_1d(
        y_true,
        pred,
    )

    reconstructed_cc = corr_1d(
        y_true,
        pred,
    )


    # ========================================================
    # Compare
    # ========================================================
    print("\n" + "=" * 90)
    print("FINAL COMPARISON")
    print("=" * 90)

    print("\nR2")
    print(
        "saved         :",
        f"{saved_r2:.10f}",
    )

    print(
        "reconstructed :",
        f"{reconstructed_r2:.10f}",
    )

    print(
        "absolute diff :",
        f"{abs(saved_r2-reconstructed_r2):.12g}",
    )


    print("\nPearson")
    print(
        "saved         :",
        f"{saved_cc:.10f}",
    )

    print(
        "reconstructed :",
        f"{reconstructed_cc:.10f}",
    )

    print(
        "absolute diff :",
        f"{abs(saved_cc-reconstructed_cc):.12g}",
    )


    r2_ok = (
        np.isfinite(saved_r2)
        and np.isfinite(reconstructed_r2)
        and abs(
            saved_r2
            - reconstructed_r2
        ) < cli.tol
    )

    cc_ok = (
        np.isfinite(saved_cc)
        and np.isfinite(reconstructed_cc)
        and abs(
            saved_cc
            - reconstructed_cc
        ) < cli.tol
    )


    print("\n" + "=" * 90)

    if r2_ok and cc_ok:

        print("✓ EXACT RECONSTRUCTION PASSED")

        print(
            "The saved cross-session R2 and Pearson "
            "are reproduced by the feature loader."
        )

    else:

        print("✗ RECONSTRUCTION FAILED")

        print(
            "Do not proceed to population/RDM simulation."
        )

    print("=" * 90)


    feature_loader.clear_big_features()


if __name__ == "__main__":
    main()