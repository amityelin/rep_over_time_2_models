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

    old_argv = sys.argv.copy()

    sys.argv = [
        "population_tail_sensitivity",

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

    keep = np.isin(
        voxel_roi[0],
        roi_codes[args.roi_name],
    )

    return voxel_mask & keep


def get_fit_file(root, subject, roi_name, pca_type, pca_pct_var):

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


def corr_across_voxels(a, b):
    """
    Pearson correlation between two population vectors.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    mask = np.isfinite(a) & np.isfinite(b)

    if mask.sum() < 2:
        return np.nan

    a = a[mask]
    b = b[mask]

    a = a - a.mean()
    b = b - b.mean()

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return np.nan

    return float(
        np.dot(a, b) / denom
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--subject", type=int, required=True)

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

    parser.add_argument(
        "--nimgs",
        type=int,
        default=100,
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
    # Use first N unique subject images
    #
    # Important: these are image indices understood by
    # feature_loader.load(), not row positions in the HDF5 file.
    # ========================================================
    subject_df = nsd_utils.get_subj_df(
        cli.subject
    )

    all_subject_images = np.asarray(
        subject_df["nsdId"],
        dtype=int,
    )

    sim_image_order = np.unique(
        all_subject_images
    )[:cli.nimgs]

    print("\nSimulation images:")
    print("N =", len(sim_image_order))
    print("First 10 =", sim_image_order[:10])


    # ========================================================
    # Feature loader
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
            f"Could not infer partial mask orientation: "
            f"{partial_masks.shape}"
        )

    partial_mask_full = partial_mask_full[
        :feature_loader.max_features
    ]


    # ========================================================
    # Load fitted models
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


    nsessions, nvox, nfeat = weights.shape

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

    print("\n" + "=" * 90)
    print("CONFIGURATION")
    print("=" * 90)

    print("Subject:", cli.subject)
    print("ROI:", cli.roi_name)
    print("Model:", model_name)
    print("Sessions:", nsessions)
    print("Total voxels:", nvox)
    print("Initially valid voxels:", int(valid_vox.sum()))


    # ========================================================
    # Simulate population responses EXACTLY through loader
    #
    # voxel × session × image
    # ========================================================
    responses = np.full(
        (
            nvox,
            nsessions,
            len(sim_image_order),
        ),
        np.nan,
        dtype=np.float32,
    )


    unique_prfs = np.unique(
        fixed_prf[
            valid_vox
        ]
    )


    for ii, prf in enumerate(
        unique_prfs,
        start=1,
    ):

        vox_here = np.flatnonzero(
            valid_vox
            &
            (fixed_prf == prf)
        )

        print(
            f"pRF {ii}/{len(unique_prfs)} "
            f"| grid={int(prf)} "
            f"| voxels={len(vox_here)}"
        )

        (
            features_actual,
            feature_inds_defined,
        ) = feature_loader.load(
            sim_image_order,
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


        preds = np.einsum(
            "if,svf->svi",
            X,
            W,
            optimize=True,
        )

        preds += (
            B[:, :, None]
        )


        responses[
            vox_here,
            :,
            :,
        ] = np.transpose(
            preds,
            (1, 0, 2),
        )


    feature_loader.clear_big_features()


    # ========================================================
    # Voxel instability score
    #
    # Maximum prediction across every session and image.
    # ========================================================
    voxel_maxabs = np.nanmax(
        np.abs(responses),
        axis=(1, 2),
    )


    # ========================================================
    # Sensitivity voxel masks
    # ========================================================
    masks = {
        "all_valid":
            valid_vox,

        "maxabs_lt_1M":
            valid_vox
            & (voxel_maxabs < 1_000_000),

        "maxabs_lt_100k":
            valid_vox
            & (voxel_maxabs < 100_000),

        "maxabs_lt_10k":
            valid_vox
            & (voxel_maxabs < 10_000),
    }


    print("\n" + "=" * 90)
    print("VOXEL SUBSETS")
    print("=" * 90)

    for name, mask in masks.items():

        print(
            f"{name:20s}: "
            f"{int(mask.sum())}/{nvox}"
        )


    # ========================================================
    # Population session × session similarity
    #
    # For each image:
    # correlate population vectors across sessions.
    #
    # Then average across 100 images.
    # ========================================================
    results = {}


    for name, mask in masks.items():

        vox_inds = np.flatnonzero(
            mask
        )

        if len(vox_inds) < 2:
            raise RuntimeError(
                f"Too few voxels for subset {name}"
            )


        img_corr = np.full(
            (
                len(sim_image_order),
                nsessions,
                nsessions,
            ),
            np.nan,
            dtype=np.float64,
        )


        for img in range(
            len(sim_image_order)
        ):

            M = responses[
                vox_inds,
                :,
                img,
            ]
            # vox × session


            for s1 in range(nsessions):

                img_corr[
                    img,
                    s1,
                    s1,
                ] = 1.0

                for s2 in range(
                    s1 + 1,
                    nsessions,
                ):

                    r = corr_across_voxels(
                        M[:, s1],
                        M[:, s2],
                    )

                    img_corr[
                        img,
                        s1,
                        s2,
                    ] = r

                    img_corr[
                        img,
                        s2,
                        s1,
                    ] = r


        avg_corr = np.nanmean(
            img_corr,
            axis=0,
        )


        # ---------------------------------------------
        # Collapse by session lag
        # ---------------------------------------------
        lag_curve = np.full(
            nsessions - 1,
            np.nan,
            dtype=np.float64,
        )

        idx = np.arange(
            nsessions
        )

        dist = np.abs(
            idx[:, None]
            - idx[None, :]
        )


        for lag in range(
            1,
            nsessions,
        ):

            lag_curve[
                lag - 1
            ] = np.nanmean(
                avg_corr[
                    dist == lag
                ]
            )


        results[name] = {
            "avg_corr": avg_corr,
            "lag_curve": lag_curve,
            "nvox": len(vox_inds),
        }


    # ========================================================
    # Compare sensitivity curves to all-valid reference
    # ========================================================
    reference = results[
        "all_valid"
    ]["lag_curve"]


    print("\n" + "=" * 90)
    print("POPULATION SENSITIVITY")
    print("=" * 90)


    for name, result in results.items():

        curve = result[
            "lag_curve"
        ]

        diff = curve - reference

        print("\n", name)

        print(
            "  voxels:",
            result["nvox"]
        )

        print(
            "  lag-1 correlation:",
            curve[0]
        )

        print(
            "  lag-10 correlation:",
            curve[9]
        )

        print(
            "  lag-20 correlation:",
            curve[19]
        )

        print(
            "  lag-29 correlation:",
            curve[28]
        )

        print(
            "  mean abs difference "
            "vs all_valid:",
            np.mean(
                np.abs(diff)
            )
        )

        print(
            "  max abs difference "
            "vs all_valid:",
            np.max(
                np.abs(diff)
            )
        )


    # ========================================================
    # Correlation between whole lag curves
    # ========================================================
    print("\n" + "=" * 90)
    print("CURVE AGREEMENT WITH ALL-VALID")
    print("=" * 90)


    for name, result in results.items():

        curve = result[
            "lag_curve"
        ]

        if name == "all_valid":

            agreement = 1.0

        else:

            agreement = np.corrcoef(
                reference,
                curve,
            )[0, 1]


        print(
            f"{name:20s}: "
            f"r={agreement:.6f}"
        )


    # ========================================================
    # Save
    # ========================================================
    out_dir = (
        root
        / "texturemodel"
        / "drift_results"
        / "population_sensitivity"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_fn = (
        out_dir
        / (
            f"population_tail_sensitivity_"
            f"S{cli.subject:02d}_{cli.roi_name}_"
            f"{cli.pyr_pca_type}{cli.pca_pct_var}_raw_"
            f"Nimg{len(sim_image_order)}.npz"
        )
    )


    save_dict = {
        "subject":
            np.int32(cli.subject),

        "roi":
            np.array(cli.roi_name),

        "simImageOrder":
            sim_image_order,

        "voxelMaxAbs":
            voxel_maxabs,

        "validVoxelMask":
            valid_vox,
    }


    for name, result in results.items():

        save_dict[
            f"{name}_avgCorr"
        ] = result[
            "avg_corr"
        ]

        save_dict[
            f"{name}_lagCurve"
        ] = result[
            "lag_curve"
        ]

        save_dict[
            f"{name}_nVox"
        ] = np.int32(
            result[
                "nvox"
            ]
        )


    np.savez_compressed(
        out_fn,
        **save_dict,
    )


    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)

    print("Saved:")
    print(out_fn)


if __name__ == "__main__":
    main()