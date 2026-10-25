import argparse
import numpy as np
import distutils.util
import time

def nice_str2bool(x):
    return bool(distutils.util.strtobool(x))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subject", type=int, default=1,
                        help="number of the subject, 1-8")
    parser.add_argument("--roi_name", type=str, default="V1",
                    choices=["V1", "V2", "V3", "hV4"],
                    help="ROI to fit")
    parser.add_argument("--volume_space", type=nice_str2bool, default=True,
                        help="1 for volume space, 0 for surface.")
    parser.add_argument("--up_to_sess", type=int, default=1,
                        help="analyze sessions 1-#")
    parser.add_argument("--single_sess", type=int, default=-1,
                        help="analyze just this one session (enter integer)")
    parser.add_argument("--single_val_sess", type=int, default=-1,
                        help="validate on just this one session (enter integer)")
    parser.add_argument("--whole_session_split", type=nice_str2bool, default=False,
                        help="If true, treat single_sess as ALL training trials and single_val_sess as ALL validation trials.")

    parser.add_argument("--average_image_reps", type=nice_str2bool, default=0,
                        help="average over trial repetitions of same image?")
    parser.add_argument("--save_model_residuals", type=nice_str2bool, default=0,
                        help="save model residuals for each voxel?")
    parser.add_argument("--use_model_residuals", type=nice_str2bool, default=0,
                        help="load/fit model residuals for each voxel?")
    parser.add_argument("--residuals_model_name", type=str, default='',
                        help="model the residuals are from?")
    parser.add_argument("--trial_subset", type=str, default='all',
                        help="fit for a subset of trials only? default all trials")
    parser.add_argument("--image_set", type=str, default='none',
                        help="if evaluating on an independent image set, what is it called?")

    parser.add_argument("--which_prf_grid", type=int, default=5,
                        help="which grid of candidate prfs?")
    parser.add_argument("--prf_fixed_sigma", type=float, default=None,
                        help="if sigma is fixed, what sigma value to use?")

    parser.add_argument("--fitting_type", type=str, default='texture_pyramid',
                        help="opts: 'texture_pyramid', 'texture_gabor', 'gabor_solo'")
    parser.add_argument("--fitting_type2", type=str, default='',
                        help="additional fitting type, for variance partition?")
    parser.add_argument("--fitting_type3", type=str, default='',
                        help="additional fitting type, for variance partition?")

    parser.add_argument("--ridge", type=nice_str2bool, default=False,
                        help="ridge regression? 1 yes, 0 no")
    parser.add_argument("--set_lambda_per_group", type=nice_str2bool, default=False,
                        help="allow lambda to differ between feature groups?")
    parser.add_argument("--zscore_features", type=nice_str2bool, default=False,
                    help="z-score each feature before fitting?")

    parser.add_argument("--zscore_betas_within_sess", type=nice_str2bool, default=True,
                        help="z-score voxel beta responses separately within each session?")
    
    parser.add_argument("--do_corrcoef", type=nice_str2bool, default=True,
                        help="compute validation corrcoef in addition to R2?")

    parser.add_argument("--shuffle_images_once", type=nice_str2bool, default=False,
                        help="control: shuffle images once?")
    parser.add_argument("--random_images", type=nice_str2bool, default=False,
                        help="control: random gaussian images?")
    parser.add_argument("--random_voxel_data", type=nice_str2bool, default=False,
                        help="control: random gaussian voxel data?")

    parser.add_argument("--shuffle_data", type=nice_str2bool, default=False,
                        help="permutation test?")
    parser.add_argument("--n_shuff_iters", type=int, default=1000,
                        help="permutation iters")
    parser.add_argument("--shuff_batch_size", type=int, default=100,
                        help="permutation batch size")
    parser.add_argument("--shuff_rnd_seed", type=int, default=0,
                        help="permutation random seed")

    parser.add_argument("--bootstrap_data", type=nice_str2bool, default=False,
                        help="bootstrap test?")
    parser.add_argument("--boot_val_only", type=nice_str2bool, default=False,
                        help="bootstrap only during validation?")
    parser.add_argument("--n_boot_iters", type=int, default=1000,
                        help="bootstrap iters")
    parser.add_argument("--boot_rnd_seed", type=int, default=0,
                        help="bootstrap random seed")

    parser.add_argument("--debug", type=nice_str2bool, default=False,
                        help="fast test mode?")
    parser.add_argument("--from_scratch", type=nice_str2bool, default=True,
                        help="train from scratch?")
    parser.add_argument("--use_precomputed_prfs", type=nice_str2bool, default=False,
                        help="use precomputed pRFs?")
    parser.add_argument("--prfs_model_name", type=str, default='',
                        help="model the pRFs are from? (default alexnet pRFs)")

    parser.add_argument("--do_val", type=nice_str2bool, default=True,
                        help="run validation?")
    parser.add_argument("--do_varpart", type=nice_str2bool, default=False,
                        help="variance partition?")
    parser.add_argument("--include_solo_models", type=nice_str2bool, default=True,
                        help="in varpart, also fit each component model alone?")

    parser.add_argument("--do_tuning", type=nice_str2bool, default=False,
                        help="estimate tuning?")
    parser.add_argument("--do_sem_disc", type=nice_str2bool, default=False,
                        help="estimate semantic discriminability?")
    parser.add_argument("--overwrite_sem_disc", type=nice_str2bool, default=False,
                        help="re-do semantic discriminability?")
    parser.add_argument("--date_str", type=str, default='',
                        help="date for model fitting (only if starting from validation step)")

    parser.add_argument("--sample_batch_size", type=int, default=500,
                        help="trials per feature batch")
    parser.add_argument("--voxel_batch_size", type=int, default=1000,
                        help="voxels per weight batch")
    parser.add_argument("--voxel_batch_size_outer", type=int, default=1000,
                        help="voxels per outer permutation/boot batch")
    
    parser.add_argument('--henderson_pca', type=int, default=0,
                        help='Use Henderson-style PCA within texture feature groups (1=yes, 0=no)')
    parser.add_argument('--pyr_pca_tag', type=str, default='',
                        help="per-group PCA variant (with --henderson_pca 1), e.g. 'k2z' = 2 PCs per group, "
                             "z-scored before PCA; loads PCA_<type>_<tag>/")

    # gabor
    parser.add_argument("--n_ori_gabor", type=int, default=4,
                        help="number of orientation channels")
    parser.add_argument("--n_sf_gabor", type=int, default=4,
                        help="number of spatial freq channels")
    parser.add_argument("--gabor_nonlin_fn", type=nice_str2bool, default=True,
                        help="nonlinearity?")
    parser.add_argument("--use_pca_gabor_feats", type=nice_str2bool, default=False,
                        help="use PCA gabor features?")
    parser.add_argument("--use_fullimage_gabor_feats", type=nice_str2bool, default=False,
                        help="full-image (no pRFs) gabor?")

    # color
    parser.add_argument("--use_fullimage_color_feats", type=nice_str2bool, default=False,
                        help="full-image color features?")

    # gist
    parser.add_argument("--n_ori_gist", type=int, default=4,
                        help="number of orientation channels for gist")
    parser.add_argument("--n_blocks_gist", type=int, default=4,
                        help="number of blocks for gist")

    # pyramid
    parser.add_argument("--n_ori_pyr", type=int, default=4,
                        help="number of orientation channels")
    parser.add_argument("--n_sf_pyr", type=int, default=4,
                        help="number of spatial freq channels")
    parser.add_argument("--pyr_pca_type", type=str, default=None,
                        help="PCA type for texture features")
    parser.add_argument(
        "--pyr_fixed_n_components",
        type=int,
        default=0,
        help="Fixed number of PCA components for pyramid texture features. "
             "0 means use pca_pct_var instead."
    )
    parser.add_argument("--group_all_hl_feats", type=nice_str2bool, default=True,
                        help="group HL features?")
    parser.add_argument("--do_pyr_varpart", type=nice_str2bool, default=False,
                        help="variance partition within texture features?")

    # sketch tokens
    parser.add_argument("--use_pca_st_feats", type=nice_str2bool, default=False,
                        help="PCA sketch tokens?")
    parser.add_argument("--use_residual_st_feats", type=nice_str2bool, default=False,
                        help="ST residuals?")
    parser.add_argument("--use_grayscale_st_feats", type=nice_str2bool, default=False,
                        help="ST grayscale?")
    parser.add_argument("--use_fullimage_st_feats", type=nice_str2bool, default=False,
                        help="full-image ST?")
    parser.add_argument("--st_pooling_size", type=int, default=4,
                        help="ST pooling kernel size")
    parser.add_argument("--st_use_avgpool", type=nice_str2bool, default=False,
                        help="ST: avgpool (else maxpool)")

    # alexnet
    parser.add_argument("--alexnet_layer_name", type=str, default='',
                        help="AlexNet layer")
    parser.add_argument("--alexnet_padding_mode", type=str, default='',
                        help="padding mode (default zeros)")
    parser.add_argument("--use_pca_alexnet_feats", type=nice_str2bool, default=True,
                        help="PCA AlexNet features?")
    parser.add_argument("--alexnet_blurface", type=nice_str2bool, default=False,
                        help="AlexNet trained with blurry faces?")
    parser.add_argument("--use_fullimage_alexnet_feats", type=nice_str2bool, default=False,
                        help="full-image AlexNet?")

    # resnet / CLIP
    parser.add_argument("--resnet_layer_name", type=str, default='',
                        help="ResNet layer (e.g., 'best_layer', 'blockX')")
    parser.add_argument("--resnet_model_architecture", type=str, default='RN50',
                        help="ResNet/CLIP arch")
    parser.add_argument("--resnet_training_type", type=str, default='',
                        help="ResNet training type")
    parser.add_argument("--use_pca_resnet_feats", type=nice_str2bool, default=True,
                        help="PCA ResNet/CLIP features?")
    parser.add_argument("--use_fullimage_resnet_feats", type=nice_str2bool, default=False,
                        help="full-image ResNet/CLIP?")
    parser.add_argument("--n_resnet_blocks_include", type=int, default=16,
                        help="how many resnet blocks to choose from")
    parser.add_argument("--resnet_blurface", type=nice_str2bool, default=False,
                        help="ResNet trained with blurry faces?")

    # semantic models
    parser.add_argument("--semantic_feature_set", type=str, default='',
                        help="semantic feature set name")
    parser.add_argument("--use_fullimage_sem_feats", type=nice_str2bool, default=False,
                        help="full-image semantic features?")
    parser.add_argument("--pca_pct_var", type=int, default=95,
                        help="Percent variance retained in PCA")

    args = parser.parse_args()

    if args.prf_fixed_sigma == 0:
        args.prf_fixed_sigma = None
    if args.pyr_pca_type == 'None':
        args.pyr_pca_type = None
    if args.image_set == 'none':
        args.image_set = None

    if args.shuffle_data:
        if args.shuff_rnd_seed == 0:
            args.shuff_rnd_seed = int(time.strftime('%M%H%d', time.localtime()))
        print('\nPermutation test: random seed is %d\n' % args.shuff_rnd_seed)

    if args.debug == 1:
        print('USING DEBUG MODE...')

    return args
