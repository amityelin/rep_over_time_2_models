
print("Start second batch")
import argparse
import numpy as np
import sys, os
import torch
import time
import h5py
import torch.nn
#import custom modules
from utils import torch_utils,  default_paths
from utils import nsd_utils, floc_utils
from model_fitting import initialize_fitting
from feature_extraction import texture_statistics_pyramid
import matplotlib.pyplot as plt
from PIL import Image
import time
import shutil

NPIX = 256

if torch.cuda.is_available():
    device = initialize_fitting.init_cuda()
else:
    device = 'cpu:0'

def extract_features(image_data, \
                     n_ori=4, n_sf=4, \
                     batch_size=100,
                     which_prf_grid=1, debug=False):
 
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    

    # Set up the pyramid    
    n_prf_sd_out = 2
    aperture=1.0
    _fmaps_fn = texture_statistics_pyramid.steerable_pyramid_extractor(pyr_height = n_sf, n_ori = n_ori)
    _feature_extractor = \
        texture_statistics_pyramid.texture_feature_extractor(_fmaps_fn,sample_batch_size=batch_size, \
                                n_prf_sd_out=n_prf_sd_out, aperture=aperture,device=device)

    n_features = _feature_extractor.n_features_total
    n_images = image_data.shape[0]
    n_prfs = models.shape[0]
    n_batches = int(np.ceil(n_images/batch_size))

    features_each_prf = np.zeros((n_images, n_features, n_prfs), dtype=np.float32)
    # Define the log file name
    log_file_name = "batch_timings.log"
    log_file_path = os.path.abspath(log_file_name)  # Get the absolute path

    with open(log_file_name, "w") as log_file:
        # Test write to the file
        test_message = "Log file test: Batch timing will be recorded here.\n"
        log_file.write(test_message)
        log_file.flush()

        # Print the location of the log file
        print(f"Log file location: {log_file_path}")
        for bb in range(n_batches // 2,n_batches):
            if debug and bb > 1:
                continue

            batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb + 1), n_images]))

            log_file.write(f"Processing batch {bb} with images [{batch_inds[0]} - {batch_inds[-1]}]\n")
            print(f"Extracting features for images [{batch_inds[0]} - {batch_inds[-1]}]")

            image_batch = torch_utils._to_torch(image_data[batch_inds, :, :, :], device)

            # image_batch = resample_fn(image_batch)

            _feature_extractor.clear_big_features()

            batch_start_time = time.time()  # Start timing

            for mm in range(n_prfs):

                if debug and mm > 1:
                    continue

                x, y, sigma = models[mm, :]
                print("Getting features for pRF [x, y, sigma]:")
                print([x, y, sigma])

                features_batch = _feature_extractor(image_batch, models[mm], mm)

                print(f"Model {mm}, min/max of features in batch: [{torch.min(features_batch)}, {torch.max(features_batch)}]")

                features_each_prf[batch_inds, :, mm] = torch_utils.get_value(features_batch)

                sys.stdout.flush()
            batch_end_time = time.time()  # End timing
            batch_duration = batch_end_time - batch_start_time

            log_file.write(f"Batch {bb} took {batch_duration:.2f} seconds.\n")
            print(f"Batch {bb} took {batch_duration:.2f} seconds.")
        return features_each_prf

def proc_one_subject(subject, args):
    
    if args.use_node_storage:
        pyramid_texture_feat_path = default_paths.pyramid_texture_feat_path_localnode
    else:
        pyramid_texture_feat_path = default_paths.pyramid_texture_feat_path
    if args.debug:
        pyramid_texture_feat_path = os.path.join(pyramid_texture_feat_path, 'DEBUG')
    if not os.path.exists(pyramid_texture_feat_path):
        os.makedirs(pyramid_texture_feat_path)
         
    filename_save = os.path.join(
    pyramid_texture_feat_path,
    f"S{subject}_features_each_prf_{args.n_ori}ori_{args.n_sf}sf_grid{args.which_prf_grid}_npix{NPIX}_second.h5py")

            
    # Load and prepare the image set to work with (all images for the current subject, 10,000 ims)
    if subject == 999:
        raise NotImplementedError("COCO (subject==999) is not included in this project.")
    else:
        image_data = nsd_utils.get_image_data(subject, npix=NPIX)
        image_data = nsd_utils.add_red_fixation_rgb_uint8(image_data,alpha=0.5)
        image_data = nsd_utils.image_uncolorize_fn(image_data)
    
        print("Grayscale image_data:", image_data.shape, image_data.dtype, image_data.min(), image_data.max())
    
        sample_index = 0
        sample_image = image_data[sample_index]
        plt.imsave('sample_image.png', sample_image.squeeze(), cmap='gray')




    features_each_prf = extract_features(image_data, \
                                         n_ori=args.n_ori, n_sf=args.n_sf, \
                                         batch_size=args.batch_size,
                                         which_prf_grid=args.which_prf_grid, \
                                         debug=args.debug)

    save_features(features_each_prf, filename_save)
    

def proc_other_image_set(image_set, args):
    
    if args.use_node_storage:
        pyramid_texture_feat_path = default_paths.pyramid_texture_feat_path_localnode
    else:
        pyramid_texture_feat_path = default_paths.pyramid_texture_feat_path
    if args.debug:
        pyramid_texture_feat_path = os.path.join(pyramid_texture_feat_path, 'DEBUG')
    if not os.path.exists(pyramid_texture_feat_path):
        os.makedirs(pyramid_texture_feat_path)
       
    filename_save = os.path.join(
    pyramid_texture_feat_path,
    f"{image_set}_features_each_prf_{args.n_ori}ori_{args.n_sf}sf_grid{args.which_prf_grid}_npix{NPIX}_second.h5py")

    
    if image_set == 'floc':
        image_data = floc_utils.load_floc_images(npix=NPIX)
    
        # If floc loader returns RGB uint8 (N,3,H,W), add fixation and grayscale like NSD
        if image_data.ndim == 4 and image_data.shape[1] == 3:
            image_data = nsd_utils.add_red_fixation_rgb_uint8(image_data, alpha=0.5)
            image_data = nsd_utils.image_uncolorize_fn(image_data)
    else:
        raise ValueError(f"image set {image_set} not recognized")

       
    features_each_prf = extract_features(image_data, \
                                         n_ori=args.n_ori, n_sf=args.n_sf, \
                                         batch_size=args.batch_size,
                                         which_prf_grid=args.which_prf_grid, \
                                         debug=args.debug)

    print(f"Shape {features_each_prf.shape}")
    save_features(features_each_prf, filename_save)
    
    
def save_features(features_each_prf, filename_save):
    print(f"Shape {features_each_prf.shape}")
    print(f"Writing prf features to {filename_save}")

    # 1) Save the file
    t = time.time()
    with h5py.File(filename_save, "w") as data_set:
        data_set.create_dataset("features", data=features_each_prf, dtype=np.float32)
    elapsed = time.time() - t
    print(f"Took {elapsed:.5f} sec to write file")

    # 2) Verify save
    if not os.path.isfile(filename_save):
        raise FileNotFoundError(f"Save failed: {filename_save} was not created.")
    print(f"Verified save: {filename_save}")

    # 3) Copy to Drive root (same filename, not hard-coded)
    backup_path = os.path.join("/content/drive/MyDrive", os.path.basename(filename_save))
    print(f"Copying file to: {backup_path}")
    shutil.copy2(filename_save, backup_path)
    print(f"File copied to {backup_path}")

    # 4) Verify copy
    if not os.path.isfile(backup_path):
        raise FileNotFoundError(f"Copy failed: {backup_path} was not created.")
    print(f"Verified copy: {backup_path}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=0,
                    help="number of the subject, 1-8")
    parser.add_argument("--image_set", type=str,default='none',
                    help="name of the image set to use (if not an NSD subject)")
    parser.add_argument("--n_ori", type=int,default=4,
                    help="how many orientation channels?")
    parser.add_argument("--n_sf", type=int,default=4,
                    help="how many frequency channels (pyramid levels)?")
    parser.add_argument("--batch_size", type=int,default=100,
                    help="batch size to use for feature extraction")
    
    parser.add_argument("--use_node_storage", type=int,default=0,
                    help="want to save and load from scratch dir on current node? 1 for yes, 0 for no")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which version of prf grid to use")
    
    args = parser.parse_args()
    
    if args.subject==0:
        args.subject=None
    if args.image_set=='none':
        args.image_set=None
                         
    args.debug = (args.debug==1)     
    
    if args.subject is not None:
        
        proc_one_subject(subject = args.subject, args=args)
        
    elif args.image_set is not None:
        
        proc_other_image_set(image_set=args.image_set, args=args)
        
    