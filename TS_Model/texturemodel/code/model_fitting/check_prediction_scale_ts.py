#!/usr/bin/env python3

import argparse
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

def build_project_args(subject, roi_name, pca_type, pca_pct_var):
    """
    Recreate the same model configuration used by
    fit_cross_session_matrix_fast.py.
    """

    old_argv = sys.argv.copy()

    sys.argv = [
        "prediction_scale_check",

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

    return voxel_mask & roi_keep_full


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
        "--max_sessions",
        type=int,
        default=30,
    )

    cli = parser.parse_args()

    root = Path(cli.root)

    args = build_project_args(
        subject=cli.subject,
        roi_name=cli.roi_name,
        pca_type=cli.pyr_pca_type,
        pca_pct_var=cli.pca_pct_var,
    )

    voxel_mask = get_roi_mask(args)


    # ========================================================
    # Load real sessions exactly as fitting did
    # ========================================================
    sessions = []

    print("\n" + "=" * 90)
    print("LOADING REAL NSD SESSIONS")
    print("=" * 90)

    for sess in range(cli.max_sessions):

        print(
            f"\nLoading session {sess + 1}/{cli.max_sessions}"
        )

        (
            voxel_data,
            image_order,
            val_inds,
            holdout_inds,
            session_inds,
        ) = nsd_utils.get_data_splits_single_train(
            args.subject,
            sessions=np.asarray([sess], dtype=int),
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

        sessions.append(
            {
                "voxel_data": voxel_data,
                "image_order": image_order,
            }
        )


    # ========================================================
    # Concatenate validation image indices AND remember slices
    # ========================================================
    image_arrays = []
    session_slices = []

    start = 0

    for sess in sessions:

        arr = sess["image_order"]

        image_arrays.append(arr)

        end = start + len(arr)

        session_slices.append(
            slice(start, end)
        )

        start = end

    all_image_order = np.concatenate(
        image_arrays,
        axis=0,
    )


    # ========================================================
    # Exact feature loader
    # ========================================================
    model_name, fitting_types = (
        initialize_fitting.get_full_save_name(args)
    )

    feature_loader = (
        initialize_fitting.make_feature_loaders(
            args,
            fitting_types,
            vi=0,
            dnn_layers_use=None,
        )
    )

    partial_masks, partial_version_names = (
        feature_loader.get_partial_versions()
    )

    partial_masks = np.asarray(
        partial_masks
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
            "Could not infer partial-mask orientation: "
            f"{partial_masks.shape}"
        )

    partial_mask_full = (
        partial_mask_full[
            :feature_loader.max_features
        ]
    )


    # ========================================================
    # Load saved session-specific models
    # ========================================================
    fit_fn = get_fit_file(
        root=root,
        subject=cli.subject,
        roi_name=cli.roi_name,
        pca_type=cli.pyr_pca_type,
        pca_pct_var=cli.pca_pct_var,
    )

    if not fit_fn.exists():
        raise FileNotFoundError(fit_fn)

    with h5py.File(fit_fn, "r") as f:

        weights = np.asarray(
            f["best_weights"][
                :cli.max_sessions,
                :,
                :,
                0,
            ],
            dtype=np.float32,
        )

        biases = np.asarray(
            f["best_biases"][
                :cli.max_sessions,
                :,
                0,
            ],
            dtype=np.float32,
        )

        fixed_prf = np.asarray(
            f["fixed_best_model_each_voxel"],
            dtype=np.int32,
        )

        val_r2 = np.asarray(
            f["val_r2"][
                :cli.max_sessions,
                :cli.max_sessions,
                :,
                0,
            ],
            dtype=np.float32,
        )

        val_cc = np.asarray(
            f["val_cc"][
                :cli.max_sessions,
                :cli.max_sessions,
                :,
                0,
            ],
            dtype=np.float32,
        )


    n_sessions, nvox, nfeat = weights.shape


    # ========================================================
    # Fixed voxel population
    # Same rule used in the permutation analysis
    # ========================================================
    valid_vox = (
        np.all(
            np.isfinite(val_r2),
            axis=(0, 1),
        )
        &
        np.all(
            np.isfinite(val_cc),
            axis=(0, 1),
        )
    )

    nvalid = int(
        valid_vox.sum()
    )


    print("\n" + "=" * 90)
    print("CONFIGURATION")
    print("=" * 90)

    print("Subject:", cli.subject)
    print("ROI:", cli.roi_name)
    print("Model:", model_name)
    print("Sessions:", n_sessions)
    print("Total voxels:", nvox)
    print("Valid voxels:", nvalid)
    print("Feature width:", nfeat)
    print(
        "Total validation trials:",
        len(all_image_order),
    )


    # ========================================================
    # pair_voxel_maxabs
    #
    # Dimensions:
    #
    # train session
    # × test session
    # × voxel
    #
    # Each entry is:
    #
    # max(abs(prediction))
    #
    # over all images in the TEST session.
    # ========================================================
    pair_voxel_maxabs = np.full(
        (
            n_sessions,
            n_sessions,
            nvox,
        ),
        np.nan,
        dtype=np.float32,
    )


    unique_prfs = np.unique(
        fixed_prf[
            valid_vox
        ]
    )

    print(
        "\nOccupied pRFs:",
        len(unique_prfs),
    )


    # ========================================================
    # Reconstruct predictions
    # ========================================================
    for i, prf in enumerate(
        unique_prfs,
        start=1,
    ):

        vox_here = np.flatnonzero(
            valid_vox
            &
            (fixed_prf == prf)
        )

        if len(vox_here) == 0:
            continue

        print(
            f"\npRF {i}/{len(unique_prfs)} "
            f"| grid={prf} "
            f"| voxels={len(vox_here)}"
        )


        # ----------------------------------------------------
        # Exact feature-loader pathway
        # ----------------------------------------------------
        (
            features_actual,
            feature_inds_defined,
        ) = feature_loader.load(
            all_image_order,
            prf_model_index=int(prf),
        )

        features_actual = np.asarray(
            features_actual,
            dtype=np.float32,
        )

        feature_inds_defined = np.asarray(
            feature_inds_defined,
            dtype=bool,
        )


        actual_partial_mask = (
            partial_mask_full[
                feature_inds_defined
            ]
        )

        X = features_actual[
            :,
            actual_partial_mask,
        ]


        W = weights[
            :,
            vox_here,
            :,
        ]

        W = W[
            :,
            :,
            feature_inds_defined,
        ]

        W = W[
            :,
            :,
            actual_partial_mask,
        ]


        B = biases[
            :,
            vox_here,
        ]


        # ----------------------------------------------------
        # Prediction:
        #
        # train-session × voxel × all validation trials
        # ----------------------------------------------------
        preds = np.einsum(
            "if,svf->svi",
            X,
            W,
            optimize=True,
        )

        preds += (
            B[:, :, None]
        )


        # ----------------------------------------------------
        # Split predictions back into test sessions
        # ----------------------------------------------------
        for test_session, sess_slice in enumerate(
            session_slices
        ):

            test_preds = preds[
                :,
                :,
                sess_slice,
            ]

            # train session × voxel
            this_maxabs = np.max(
                np.abs(test_preds),
                axis=2,
            )

            pair_voxel_maxabs[
                :,
                test_session,
                vox_here,
            ] = this_maxabs


        del (
            features_actual,
            X,
            W,
            B,
            preds,
        )


    feature_loader.clear_big_features()


    # ========================================================
    # Restrict summaries to fixed valid voxel population
    # ========================================================
    pair_vals = (
        pair_voxel_maxabs[
            :,
            :,
            valid_vox,
        ]
    )


    if not np.all(
        np.isfinite(pair_vals)
    ):

        nbad = int(
            np.sum(
                ~np.isfinite(pair_vals)
            )
        )

        raise RuntimeError(
            f"Found {nbad} non-finite prediction-scale entries."
        )


    # ========================================================
    # 30 × 30 summary matrices
    # ========================================================
    median_maxabs = np.median(
        pair_vals,
        axis=2,
    )

    p95_maxabs = np.percentile(
        pair_vals,
        95,
        axis=2,
    )

    p99_maxabs = np.percentile(
        pair_vals,
        99,
        axis=2,
    )

    maximum_maxabs = np.max(
        pair_vals,
        axis=2,
    )


    # Fraction of voxels exceeding thresholds
    thresholds = [
        1_000,
        10_000,
        100_000,
        1_000_000,
    ]

    frac_over = {}

    for threshold in thresholds:

        frac_over[threshold] = np.mean(
            pair_vals > threshold,
            axis=2,
        )


    # ========================================================
    # Existing model performance for comparison
    # ========================================================
    median_r2 = np.nanmedian(
        val_r2[
            :,
            :,
            valid_vox,
        ],
        axis=2,
    )

    median_cc = np.nanmedian(
        val_cc[
            :,
            :,
            valid_vox,
        ],
        axis=2,
    )


    # ========================================================
    # Print matrices
    # ========================================================
    np.set_printoptions(
        precision=2,
        suppress=True,
        linewidth=220,
    )


    print("\n" + "=" * 90)
    print(
        "30×30 MATRIX: MEDIAN VOXEL MAX |PREDICTION|"
    )
    print(
        "rows = TRAIN sessions, columns = TEST sessions"
    )
    print("=" * 90)

    print(median_maxabs)


    print("\n" + "=" * 90)
    print(
        "30×30 MATRIX: FRACTION OF VOXELS > 1,000,000"
    )
    print(
        "rows = TRAIN sessions, columns = TEST sessions"
    )
    print("=" * 90)

    print(
        frac_over[1_000_000]
    )


    print("\n" + "=" * 90)
    print(
        "30×30 MATRIX: MEDIAN SAVED R2"
    )
    print("=" * 90)

    print(median_r2)


    # ========================================================
    # Diagonal vs off-diagonal
    # ========================================================
    diag_mask = np.eye(
        n_sessions,
        dtype=bool,
    )

    offdiag_mask = ~diag_mask


    print("\n" + "=" * 90)
    print("DIAGONAL VS CROSS-SESSION")
    print("=" * 90)


    print("\nMedian voxel max |prediction|")

    print(
        "  diagonal median     :",
        float(
            np.median(
                median_maxabs[
                    diag_mask
                ]
            )
        ),
    )

    print(
        "  off-diagonal median :",
        float(
            np.median(
                median_maxabs[
                    offdiag_mask
                ]
            )
        ),
    )


    print("\nFraction voxels > 1,000,000")

    print(
        "  diagonal median     :",
        float(
            np.median(
                frac_over[1_000_000][
                    diag_mask
                ]
            )
        ),
    )

    print(
        "  off-diagonal median :",
        float(
            np.median(
                frac_over[1_000_000][
                    offdiag_mask
                ]
            )
        ),
    )


    print("\nMedian R2")

    print(
        "  diagonal median     :",
        float(
            np.median(
                median_r2[
                    diag_mask
                ]
            )
        ),
    )

    print(
        "  off-diagonal median :",
        float(
            np.median(
                median_r2[
                    offdiag_mask
                ]
            )
        ),
    )


    # ========================================================
    # Worst train/test pairs
    # ========================================================
    flat_order = np.argsort(
        median_maxabs.ravel()
    )[::-1]


    print("\n" + "=" * 90)
    print(
        "TOP 20 TRAIN × TEST PAIRS BY MEDIAN PREDICTION MAGNITUDE"
    )
    print("=" * 90)


    for rank, flat_idx in enumerate(
        flat_order[:20],
        start=1,
    ):

        train_session, test_session = (
            np.unravel_index(
                flat_idx,
                median_maxabs.shape,
            )
        )

        print(
            f"{rank:2d}. "
            f"train={train_session + 1:02d} "
            f"test={test_session + 1:02d} | "
            f"medianMax={median_maxabs[train_session, test_session]:.3e} | "
            f"p95={p95_maxabs[train_session, test_session]:.3e} | "
            f"max={maximum_maxabs[train_session, test_session]:.3e} | "
            f"frac>1M={frac_over[1_000_000][train_session, test_session]:.3f} | "
            f"medianR2={median_r2[train_session, test_session]:.5f} | "
            f"medianCC={median_cc[train_session, test_session]:.5f}"
        )


    # ========================================================
    # Worst TRAIN sessions overall
    # ========================================================
    train_median_scale = np.median(
        median_maxabs,
        axis=1,
    )

    train_frac_1m = np.mean(
        frac_over[1_000_000],
        axis=1,
    )

    train_order = np.argsort(
        train_median_scale
    )[::-1]


    print("\n" + "=" * 90)
    print("TRAIN-SESSION SUMMARY")
    print("=" * 90)


    for idx in train_order:

        print(
            f"train={idx + 1:02d} | "
            f"median across test sessions="
            f"{train_median_scale[idx]:.3e} | "
            f"mean fraction voxels >1M="
            f"{train_frac_1m[idx]:.3f}"
        )


    # ========================================================
    # Save diagnostic output
    # ========================================================
    out_dir = (
        root
        / "texturemodel"
        / "drift_results"
        / "prediction_scale_diagnostics"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    out_fn = (
        out_dir
        / (
            f"prediction_scale_30x30_"
            f"S{cli.subject:02d}_{cli.roi_name}_"
            f"{cli.pyr_pca_type}{cli.pca_pct_var}_raw.npz"
        )
    )


    np.savez_compressed(
        out_fn,

        subject=np.int32(cli.subject),
        roi=np.array(cli.roi_name),

        pcaType=np.array(
            cli.pyr_pca_type
        ),

        pcaPctVar=np.int32(
            cli.pca_pct_var
        ),

        validVoxelMask=valid_vox,

        pairVoxelMaxAbs=pair_vals,

        medianMaxAbs=median_maxabs,
        p95MaxAbs=p95_maxabs,
        p99MaxAbs=p99_maxabs,
        maximumMaxAbs=maximum_maxabs,

        fracOver1k=frac_over[1_000],
        fracOver10k=frac_over[10_000],
        fracOver100k=frac_over[100_000],
        fracOver1m=frac_over[1_000_000],

        medianR2=median_r2,
        medianCC=median_cc,
    )


    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)

    print("Saved diagnostic:")
    print(out_fn)


if __name__ == "__main__":
    main()