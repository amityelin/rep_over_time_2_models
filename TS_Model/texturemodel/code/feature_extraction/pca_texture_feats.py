import sys, os
import numpy as np
import time, h5py
import gc
from utils import default_paths, nsd_utils
from model_fitting import initialize_fitting 
from feature_extraction import texture_feature_utils, pca_feats, fwrf_features
compute_pca = pca_feats.compute_pca

import argparse
import pandas as pd

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


"""
Code to perform PCA on features within a given feature space (texture or contour etc).
PCA is done separately within each pRF position, and the results for all pRFs are saved in a single file.
"""
# fit_inds_dict = {s: np.load(f"/content/drive/MyDrive/V2_drift/fit_inds_subj{s}.npy")
#                  for s in range(1,9)}

def run_pca_texture_pyramid(subject=None,
                            image_set=None,
                            pca_type='pcaHL',\
                            save_weights=False, \
                            min_pct_var=95, max_pc_to_retain=150, \
                            debug=False, which_prf_grid=1, \
                            save_dtype=np.float32, compress=True,
                            fixed_n_per_group=0, zscore_before_pca=False):

    # ============================================================
    # MODIFICATION (Amit Yelin / Claude, Oct 2026)
    # ------------------------------------------------------------
    # fixed_n_per_group = K > 0: keep exactly K PCs from every PCA
    #   group (instead of the number of PCs reaching min_pct_var), so
    #   every group is represented and every pRF has the same number
    #   of predictors (pcaHL: 49 raw low-level + 9*K).
    # zscore_before_pca: z-score each feature over the training images
    #   (within its group) before PCA, so the PCs are not dominated by
    #   features with large raw scale.
    # Outputs go to PCA_<type>_<tag>/, tag = k<K>[z] or v<pct>z, and the
    #   fitting code loads them with --henderson_pca 1 --pyr_pca_tag <tag>.
    # ============================================================
    var_str = ('k%d' % fixed_n_per_group) if fixed_n_per_group > 0 else ('v%d' % min_pct_var)
    if zscore_before_pca:
        var_str += 'z'
    custom_tag = fixed_n_per_group > 0 or zscore_before_pca

    if subject is not None:
        
        if image_set is None:
            # run PCA on the features for this NSD subject
            image_set = 'S%d'%subject
            use_precomputed_weights = False
            if subject==999:
                # 999 is a code for the set of images that are independent of NSD images, 
                # not shown to any participant.
                fit_inds = np.ones((10000,),dtype=bool)
            else:            
                # training / validation data always split the same way - shared 1000 inds are validation.
                subject_df = nsd_utils.get_subj_df(subject)
                fit_inds = np.array(subject_df['shared1000']==False)
        else:
            # if both subject and image set are specified, this means we should use the weights 
            # from the NSD subject and apply them to the features from image_set
            weights_image_set = 'S%d'%subject
            use_precomputed_weights = True
            assert(save_weights==False)
            if image_set=='floc':
                labels_file = os.path.join(default_paths.floc_image_root,'floc_image_labels.csv')
                labels = pd.read_csv(labels_file)
                fit_inds = np.ones((labels.shape[0],),dtype=bool)
            else:
                raise ValueError('image_set %s not recognized'%image_set)
            
    else:
        # run PCA on features from a non-NSD image set
        assert(image_set is not None)
        use_precomputed_weights = False
        subject=0
        if image_set=='floc':
            labels_file = os.path.join(default_paths.floc_image_root,'floc_image_labels.csv')
            labels = pd.read_csv(labels_file)
            fit_inds = np.ones((labels.shape[0],),dtype=bool)
        else:
            raise ValueError('image_set %s not recognized'%image_set)
   
    n_trials = len(fit_inds)
    
    
    
    path_raw = default_paths.pyramid_texture_feat_path
    if debug:
        path_raw = os.path.join(path_raw, 'DEBUG')
    # ============================================================
    # MODIFICATION (Amit Yelin)
    # ------------------------------------------------------------
    # PCA is now computed once, independent of variance threshold.
    # Different thresholds will later be created by slicing the
    # saved PCA score matrices using the explained variance vector.
    # ============================================================
    
    if custom_tag:
        path_pca = os.path.join(path_raw, 'PCA_%s_%s' % (pca_type, var_str))
    else:
        path_pca = os.path.join(
            path_raw,
            'PCA_%s_%s_henderson_replication' % (pca_type, var_str)
        )
    if custom_tag and use_precomputed_weights:
        raise NotImplementedError('fixed_n_per_group / zscore_before_pca not supported with precomputed weights')

    if not os.path.exists(path_pca):
        os.makedirs(path_pca)
        
    n_ori=4; n_sf=4;    
    if debug:
        prf_batch_size=2
    else:
        prf_batch_size=100;
        
    if image_set is not None:
        floader_sub=None
    else:
        floader_sub=subject
    print('floader_sub: %s'%floader_sub)
    print(image_set, subject)
    floader = fwrf_features.fwrf_feature_loader(subject=floader_sub, \
                                               image_set=image_set, \
                                               which_prf_grid=which_prf_grid, 
                                               feature_type='pyramid_texture', 
                                               n_ori=n_ori, n_sf=n_sf, 
                                               pca_type=None, 
                                               prf_batch_size=prf_batch_size)
    features_file = floader.features_file

    # batching prfs for loading, because it is a bit faster
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)  
    n_prfs = models.shape[0]

    # can do the pca a few different ways, depending on how we want to group the features together.
    if pca_type=='pcaHL':

        feature_column_labels, feature_type_names = texture_feature_utils.get_feature_inds()
        # do higher-level feats only
        gets_pca = np.arange(5,14)

    elif pca_type=='pcaAll':

        feature_column_labels, feature_type_names = texture_feature_utils.get_feature_inds()
        # do all features
        gets_pca = np.arange(14)

    elif pca_type=='pcaHL_simple':

        feature_column_labels, feature_type_names = texture_feature_utils.get_feature_inds_simplegroups()
        # do higher-level feats only
        gets_pca = np.arange(4,10)

    elif pca_type=='pcaHL_sepscales':

        feature_column_labels, feature_type_names = texture_feature_utils.get_feature_inds_sepscales()
        # do higher-level feats only
        gets_pca = np.arange(15,41) 

    # ============================================================
    # MODIFICATION (Amit Yelin)
    # ------------------------------------------------------------
    # A variance threshold of 0% is only meaningful for pcaHL,
    # where it creates a low-level-only model by excluding all
    # high-level PCA features.
    #
    # For pcaAll, a 0% threshold would remove every feature,
    # resulting in an invalid encoding model.
    # ============================================================

    if pca_type == "pcaAll" and min_pct_var == 0:
        raise ValueError(
            "Invalid configuration: pcaAll with min_pct_var=0 would "
            "retain zero PCA components and therefore zero features. "
            "Use pcaHL with min_pct_var=0 to generate the low-level-only baseline."
        )
    
    feature_type_names = np.array(feature_type_names)
    feature_type_dims = np.array([np.sum(feature_column_labels==ff) for ff in range(len(feature_type_names))])

    print('will perform PCA on these sets of features:')
    print(feature_type_names[gets_pca])
    print(feature_type_dims[gets_pca])

    feature_type_names_pca = feature_type_names[gets_pca]
    feature_inds = np.array([feature_column_labels==fi for fi in gets_pca])

    max_pc_to_retain_overall = max_pc_to_retain

    pca_filename_list = []

    # ============================================================
    # MODIFICATION (Amit Yelin)
    # ------------------------------------------------------------
    # Save PCA diagnostics for later analysis.
    # This lets us inspect how many PCs were retained per feature
    # group and per pRF, and how much variance PC1 explains.
    # ============================================================
    
    pca_summary_rows = []
    pca_ev_records = []
    
    # going to loop over one set of features at a time
    # (this isn't the fastest way but uses less memory at a time)
    for fi, feature_type_name in enumerate(feature_type_names_pca):
   
        n_feat_orig = np.sum(feature_inds[fi,:])
        if max_pc_to_retain_overall is None:
            max_pc_to_retain = n_feat_orig
        else:
            max_pc_to_retain = np.minimum(max_pc_to_retain_overall, n_feat_orig)
        if fixed_n_per_group > 0:
            max_pc_to_retain = int(np.minimum(fixed_n_per_group, n_feat_orig))

        print('\nProcessing feature subset: %s\n'%feature_type_name)
        scores_each_prf = np.zeros((n_trials, max_pc_to_retain, n_prfs), dtype=save_dtype)
        actual_max_ncomp = 0;        
        
        if use_precomputed_weights:
            wts_dir = os.path.join(path_pca, 'weights')
            load_weights_filename = os.path.join(wts_dir,
                '%s_%dori_%dsf_PCA_%s_only_%s_weights_grid%d.npy' %
                (weights_image_set, n_ori, n_sf, feature_type_name, var_str, which_prf_grid))
            print('loading pre-computed pca weights from %s' % load_weights_filename)
            w = np.load(load_weights_filename, allow_pickle=True).item()
            pca_wts = w['pca_wts']
            pca_premean = w['pca_premean']
            pca_ncomp = w['pca_ncomp']
            filename_save = os.path.join(path_pca, '%s_%dori_%dsf_PCA_%s_only_%s_wtsfrom%s_grid%d.h5py' %
                                         (image_set, n_ori, n_sf, feature_type_name, var_str, weights_image_set, which_prf_grid))
        else:
            filename_save = os.path.join(path_pca, '%s_%dori_%dsf_PCA_%s_only_%s_grid%d.h5py' %
                                         (image_set, n_ori, n_sf, feature_type_name, var_str, which_prf_grid))
        if save_weights:
            
            pca_wts = np.zeros((n_feat_orig, max_pc_to_retain, n_prfs), dtype=save_dtype)
            pca_premean = np.zeros((n_feat_orig, n_prfs), dtype=save_dtype)
            pca_ncomp = np.zeros((n_prfs,),dtype=save_dtype)
            pca_prescale = np.ones((n_feat_orig, n_prfs), dtype=save_dtype)
            
        for prf_model_index in range(n_prfs):

            if debug and prf_model_index>1:
                continue
                
            print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
            sys.stdout.flush()
            
            # load raw features
            feat, _ = floader.load(np.arange(n_trials), prf_model_index)

            # pull out the ones for this "type" only
            features_in_prf = feat[:,feature_inds[fi,:]].astype(np.float32)

            if zscore_before_pca:
                # z-score with training-image statistics only (no validation leakage)
                z_mu = features_in_prf[fit_inds,:].mean(axis=0)
                z_sd = features_in_prf[fit_inds,:].std(axis=0)
                z_sd[z_sd == 0] = 1.0
                features_in_prf = (features_in_prf - z_mu[np.newaxis,:]) / z_sd[np.newaxis,:]
            
            print('Processing %s, size of array before PCA:'%feature_type_name)
            print(features_in_prf.shape)

            if use_precomputed_weights:
                
                # project into pca subspace using saved wts
                wts = pca_wts[:,:,prf_model_index]
                pre_mean = pca_premean[:,prf_model_index]
                n_comp_needed = int(pca_ncomp[prf_model_index])
                
                feat_submean = features_in_prf - np.tile(pre_mean[np.newaxis,:], [features_in_prf.shape[0],1])
                scores = feat_submean @ wts
                
            else:
                
                # compute pca weights here
                _, wts, pre_mean, ev = compute_pca(features_in_prf[fit_inds,:], max_pc_to_retain=max_pc_to_retain)
                feat_submean = features_in_prf - np.tile(pre_mean[np.newaxis,:], [features_in_prf.shape[0],1])
                scores = feat_submean @ wts.T

                # ============================================================
                # MODIFICATION (Amit Yelin)
                #
                # Original code:
                #
                n_comp_needed = np.where(np.cumsum(ev) > min_pct_var)
                if np.size(n_comp_needed) > 0:
                    n_comp_needed = n_comp_needed[0][0] + 1
                else:
                    n_comp_needed = scores.shape[1]
                if fixed_n_per_group > 0:
                    # fixed number of PCs per group (ev is relative to the retained PCs only here)
                    n_comp_needed = int(np.minimum(fixed_n_per_group, scores.shape[1]))
                #
                # Original behavior:
                # min_pct_var = 0 retained one PCA component from every
                # feature group.
                #
                # Modified behavior:
                # min_pct_var = 0 retains ZERO PCA components, allowing
                # pcaHL to serve as a low-level-only baseline model.
                # ============================================================
                
                # ============================================================
                # MODIFICATION (Amit Yelin)
                # ------------------------------------------------------------
                # Store all available PCs up to max_pc_to_retain.
                # Threshold-specific feature sets will be created later by
                # selecting the first N PCs according to cumulative explained
                # variance.
                # ============================================================
                
                #n_comp_needed = scores.shape[1]
                
                print('Retaining %d components to explain %d pct var'
                      % (n_comp_needed, min_pct_var))
                  
                # ============================================================
                # MODIFICATION (Amit Yelin)
                # ------------------------------------------------------------
                # Save PCA information per feature group and per pRF.
                # This allows later plotting of:
                #   1. number of retained PCs per feature group
                #   2. PC1 explained variance per feature group
                #   3. cumulative explained variance curves
                # ============================================================
                
                pca_summary_rows.append({
                    "subject": subject,
                    "image_set": image_set,
                    "pca_type": pca_type,
                    "pct_variance_threshold": min_pct_var,
                    "feature_group": str(feature_type_name),
                    "feature_group_index": int(gets_pca[fi]),
                    "prf": int(prf_model_index),
                    "n_original_features": int(n_feat_orig),
                    "n_components_retained": int(n_comp_needed),
                    "pc1_variance_explained": float(ev[0]) if len(ev) > 0 else np.nan,
                    "max_pc_to_retain": int(max_pc_to_retain) if max_pc_to_retain is not None else np.nan,
                })
                
                pca_ev_records.append({
                    "subject": subject,
                    "image_set": image_set,
                    "pca_type": pca_type,
                    "pct_variance_threshold": min_pct_var,
                    "feature_group": str(feature_type_name),
                    "feature_group_index": int(gets_pca[fi]),
                    "prf": int(prf_model_index),
                    "n_original_features": int(n_feat_orig),
                    "n_components_retained": int(n_comp_needed),
                    "explained_variance_per_pc": ev.astype(np.float32),
                    "cumulative_explained_variance": np.cumsum(ev).astype(np.float32),
                })
            
            actual_max_ncomp = np.max([n_comp_needed, actual_max_ncomp])

            # ============================================================
            # MODIFICATION (Amit Yelin)
            #
            # Original code:
            #
            # scores_each_prf[:, 0:n_comp_needed, prf_model_index] = \
            #     scores[:, 0:n_comp_needed]
            # scores_each_prf[:, n_comp_needed:, prf_model_index] = np.nan
            #
            # Original behavior assumed n_comp_needed >= 1.
            #
            # The updated code first fills the array with NaNs and only
            # copies PCA scores if at least one component is retained.
            # This allows n_comp_needed = 0.
            # ============================================================
            
            scores_each_prf[:, :, prf_model_index] = np.nan
            
            if n_comp_needed > 0:
                scores_each_prf[:, :n_comp_needed, prf_model_index] = \
                    scores[:, :n_comp_needed]
            
            if save_weights:
                pca_wts[:,:,prf_model_index] = wts.T
                pca_premean[:,prf_model_index] = pre_mean
                pca_ncomp[prf_model_index] = n_comp_needed
                if zscore_before_pca:
                    pca_prescale[:,prf_model_index] = z_sd

        # To save space, get rid of portion of array that ended up all nans
        if debug:
            actual_max_ncomp=np.max([2,actual_max_ncomp])
            assert(np.all((scores_each_prf[:,actual_max_ncomp:,:]==0) | \
                          np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
        else:
            assert(np.all(np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
        scores_each_prf = scores_each_prf[:,0:actual_max_ncomp,:]
        print('final size of array to save (for %s):'%feature_type_name)
        print(scores_each_prf.shape)
        
        
        print('saving to %s'%filename_save)
        pca_filename_list.append(filename_save)
        sys.stdout.flush()
        t = time.time()
        with h5py.File(filename_save, 'w') as data_set:
            if compress==True:
                dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype, compression='gzip')
            else:
                dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype)
            data_set['/features'][:,:,:] = scores_each_prf
            data_set.close() 
        elapsed = time.time() - t
        print('Took %.5f sec to write file'%elapsed)
        sys.stdout.flush()
        
        if save_weights:
            wts_dir = os.path.join(path_pca, 'weights')
            if not os.path.exists(wts_dir):
                os.makedirs(wts_dir)
            save_weights_filename = os.path.join(wts_dir,
                '%s_%dori_%dsf_PCA_%s_only_%s_weights_grid%d.npy' %
                (image_set,n_ori, n_sf, feature_type_name, var_str, which_prf_grid))
            wts = {'pca_wts': pca_wts, 'pca_premean': pca_premean, 'pca_ncomp': pca_ncomp,
                   'zscore_before_pca': bool(zscore_before_pca), 'pca_prescale': pca_prescale}
            print('saving the weights for this pca to %s'%save_weights_filename)
            np.save(save_weights_filename, wts, allow_pickle=True)
        
        
    # ============================================================
    # MODIFICATION (Amit Yelin)
    # ------------------------------------------------------------
    # Save PCA diagnostics.
    #
    # CSV:
    #   Easy to open and inspect. Contains one row per:
    #   feature group x pRF.
    #
    # NPY:
    #   Full explained-variance spectrum for every feature group
    #   and pRF, useful for plotting cumulative variance curves.
    # ============================================================
    
    pca_summary_df = pd.DataFrame(pca_summary_rows)
    
    pca_summary_csv = os.path.join(
        path_pca,
        f"{image_set}_{n_ori}ori_{n_sf}sf_{pca_type}_summary_{var_str}_grid{which_prf_grid}.csv"
    )
    
    pca_summary_npy = os.path.join(
        path_pca,
        f"{image_set}_{n_ori}ori_{n_sf}sf_{pca_type}_ev_spectra_{var_str}_grid{which_prf_grid}.npy"
    )
    
    print("saving PCA summary to:")
    print(pca_summary_csv)
    pca_summary_df.to_csv(pca_summary_csv, index=False)
    
    print("saving PCA explained variance spectra to:")
    print(pca_summary_npy)
    np.save(pca_summary_npy, pca_ev_records, allow_pickle=True)
    
    # Now going to concatenate all files together, for easier loading later
    if use_precomputed_weights:
        concat_filename = os.path.join(path_pca, '%s_%dori_%dsf_%s_concat_%s_wtsfrom%s_grid%d.h5py' %
                                       (image_set, n_ori, n_sf, pca_type, var_str, weights_image_set, which_prf_grid))
        concat_labels_filename = os.path.join(path_pca, '%s_%dori_%dsf_featurelabels_%s_%s_wtsfrom%s_grid%d.npy' %
                                              (image_set, n_ori, n_sf, pca_type, var_str, weights_image_set, which_prf_grid))
    else:
        concat_filename = os.path.join(path_pca, '%s_%dori_%dsf_%s_concat_%s_grid%d.h5py' %
                                       (image_set, n_ori, n_sf, pca_type, var_str, which_prf_grid))
        concat_labels_filename = os.path.join(path_pca, '%s_%dori_%dsf_featurelabels_%s_%s_grid%d.npy' %
                                              (image_set, n_ori, n_sf, pca_type, var_str, which_prf_grid))

    
    # first grab the raw low-level features (if using)
    if pca_type != 'pcaAll':  # Changed 'is not' to '!=' for proper string comparison
        is_ll = texture_feature_utils.is_low_level()
        print('loading from %s' % features_file)
        st = time.time()
        sys.stdout.flush()
        
        # Get shape information first without loading data
        with h5py.File(features_file, 'r') as file:
            dataset = file['/features']
            feat_shape = dataset.shape
            
            # Calculate output shape before loading data
            # Determine the shape after applying is_ll filter to second dimension
            num_ll_features = np.sum(is_ll)
            output_shape = (feat_shape[0], num_ll_features, feat_shape[2])
            
            # Initialize the output array, but don't load anything yet
            feat_raw = np.zeros(output_shape, dtype=dataset.dtype)
            
            # Load data in chunks
            chunk_size = 1000  # Adjust based on your memory constraints
            num_chunks = int(np.ceil(feat_shape[0] / chunk_size))
            
            for i in range(num_chunks):
                print(f"Loading chunk {i+1}/{num_chunks} of raw features")
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, feat_shape[0])
                
                # Load and process chunk
                chunk = np.array(dataset[start_idx:end_idx])
                # Apply the is_ll filter to the second dimension
                chunk_filtered = chunk[:, is_ll, :]
                
                # Store this chunk
                feat_raw[start_idx:end_idx] = chunk_filtered
                
                # Free memory explicitly
                del chunk, chunk_filtered
                gc.collect()  # Force garbage collection
        
        elapsed = time.time() - st
        print('loading took %.5f sec' % elapsed)
        print(f"Raw features shape: {feat_raw.shape}")
        feat_all = feat_raw
        sys.stdout.flush()
        
        # Process feature column labels after loading data
        feature_column_labels_pca = np.array(feature_column_labels[is_ll])
        
    else:
        feat_raw = None       
        feature_column_labels_pca = np.array([])

    n_raw_types = len(np.unique(feature_column_labels_pca))

    # Calculate final feature dimensions for pre-allocation
    total_feature_dim = 0
    if feat_raw is not None:
        total_feature_dim += feat_raw.shape[1]

    # First pass to get dimensions of all PCA features
    pca_dims = []
    for pca_filename in pca_filename_list:
        with h5py.File(pca_filename, 'r') as file:
            pca_shape = file['/features'].shape
            pca_dims.append(pca_shape[1])
            total_feature_dim += pca_shape[1]

    # Pre-allocate the final feature array
    if feat_raw is not None:
        num_samples = feat_raw.shape[0]
        third_dim = feat_raw.shape[2]
    else:
        # Get dimensions from first PCA file if no raw features
        with h5py.File(pca_filename_list[0], 'r') as file:
            pca_shape = file['/features'].shape
            num_samples = pca_shape[0]
            third_dim = pca_shape[2]

    # Pre-allocate the full array
    feat_all = np.zeros((num_samples, total_feature_dim, third_dim), dtype=np.float32)

    # Copy raw features if they exist
    current_col = 0
    if feat_raw is not None:
        chunk_size = 1000
        num_chunks = int(np.ceil(num_samples / chunk_size))
        
        raw_cols = feat_raw.shape[1]
        for i in range(num_chunks):
            print(f"Copying chunk {i+1}/{num_chunks} of raw features to final array")
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_samples)
            
            feat_all[start_idx:end_idx, :raw_cols, :] = feat_raw[start_idx:end_idx]
        
        current_col = raw_cols
        # Free memory
        del feat_raw
        gc.collect()

    # loop over the higher-level feature subsets and load them incrementally
    for fi, pca_filename in enumerate(pca_filename_list):
        print('loading from %s'%pca_filename)
        st = time.time()
        sys.stdout.flush()
        
        with h5py.File(pca_filename, 'r') as file:
            pca_dataset = file['/features']
            n_pc = pca_dataset.shape[1]
            
            # Load and integrate PCA features in chunks
            chunk_size = 1000
            num_chunks = int(np.ceil(num_samples / chunk_size))
            
            for i in range(num_chunks):
                print(f"Loading chunk {i+1}/{num_chunks} of PCA features from file {fi+1}/{len(pca_filename_list)}")
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, num_samples)
                
                # Load chunk of PCA features
                pca_chunk = np.array(pca_dataset[start_idx:end_idx])
                
                # Copy to the appropriate columns in feat_all
                feat_all[start_idx:end_idx, current_col:current_col+n_pc, :] = pca_chunk
                
                # Free memory
                del pca_chunk
                gc.collect()
        
        elapsed = time.time() - st
        print('loading took %.5f sec'%elapsed)
        
        # Update column labels
        feature_column_labels_chunk = (fi+n_raw_types)*np.ones((n_pc,))
        feature_column_labels_pca = np.concatenate([feature_column_labels_pca, feature_column_labels_chunk], axis=0)
        
        # Update current column index
        current_col += n_pc
        sys.stdout.flush()

    print('final shape is:')
    print(feat_all.shape)

    print('saving to %s'%concat_filename)
    sys.stdout.flush()
    t = time.time()

    # Chunk the writing process too
    with h5py.File(concat_filename, 'w') as data_set:
        # Create the dataset with appropriate compression
        if compress:
            dset = data_set.create_dataset("features", np.shape(feat_all), dtype=save_dtype, compression='gzip')
        else:
            dset = data_set.create_dataset("features", np.shape(feat_all), dtype=save_dtype)
        
        # Write in chunks
        chunk_size = 1000
        num_chunks = int(np.ceil(num_samples / chunk_size))
        
        for i in range(num_chunks):
            print(f"Writing chunk {i+1}/{num_chunks} to output file")
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_samples)
            
            # Write chunk to file
            dset[start_idx:end_idx] = feat_all[start_idx:end_idx]

    elapsed = time.time() - t
    print('Took %.5f sec to write file'%elapsed)

    # save the labels for which columns of the concatenated array correspond to which feature types
    print('saving to %s'%concat_labels_filename)
    print(feature_column_labels_pca, feature_type_names)
    np.save(concat_labels_filename, {'feature_column_labels': feature_column_labels_pca, \
                            'feature_type_names': feature_type_names})

    # remove the smaller intermediate files, to save disk space  
    for pca_filename in pca_filename_list:
        print('deleting %s'%pca_filename)
        if not debug:
            os.remove(pca_filename)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=0,
                    help="number of the subject, 1-8")
    parser.add_argument("--image_set", type=str,default='none',
                    help="name of the image set to use (if not an NSD subject)")
    
    parser.add_argument("--pca_type", type=str,default='pcaHL',
                    help="what kind of features are we using?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")

    parser.add_argument("--max_pc_to_retain", type=int,default=0,
                    help="max pc to retain? enter 0 for None")
    parser.add_argument("--min_pct_var", type=int,default=95,
                    help="min pct var to explain? default 95")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which pRF grid to use?")
    
    parser.add_argument("--save_weights", type=int,default=0,
                    help="want to save the weights to reproduce the pca later? 1 for yes, 0 for no")
    parser.add_argument("--fixed_n_per_group", type=int, default=0,
                    help="keep exactly this many PCs per group (0 = use min_pct_var)")
    parser.add_argument("--zscore_before_pca", type=int, default=0,
                    help="z-score each feature (training images) before PCA? 1 for yes")

    
    args = parser.parse_args()
    
    if args.subject==0:
        args.subject=None
    if args.image_set=='none':
        args.image_set=None
        
    if args.max_pc_to_retain==0:
        args.max_pc_to_retain = None
     
    run_pca_texture_pyramid(subject=args.subject, 
                            image_set = args.image_set,
                            pca_type=args.pca_type, \
                            min_pct_var=args.min_pct_var, \
                            max_pc_to_retain=args.max_pc_to_retain,\
                            debug=args.debug==1,  \
                            which_prf_grid=args.which_prf_grid, \
                            save_weights = args.save_weights,
                            fixed_n_per_group=args.fixed_n_per_group,
                            zscore_before_pca=args.zscore_before_pca==1)
    