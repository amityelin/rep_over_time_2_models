import os
import argparse
from utils import default_paths
from utils.nsd_utils import get_subject_specific_images


def main(debug=False):
    """
    Create resized NSD stimulus files at 256x256 for all subjects.

    DEBUG MODE:
      - Processes only a tiny subset (per nsd_utils implementation)
      - Overwrites S1_stimuli_256.h5py if it already exists
      - Intended ONLY for sanity checks

    FULL MODE:
      - Processes all subjects and all 10k images
    """

    nsd_root = default_paths.nsd_root
    stim_root = default_paths.stim_root

    print("NSD root:", nsd_root)
    print("Stimuli output path:", stim_root)

    os.makedirs(stim_root, exist_ok=True)

    if debug:
        print("\n RUNNING IN DEBUG MODE")
        print("Only a few images / subjects will be processed\n")
    else:
        print("\n RUNNING FULL DATASET CREATION\n")

    print("Creating 256x256 stimulus files...")
    get_subject_specific_images(
        nsd_root=nsd_root,
        path_to_save=stim_root,
        npix=256,
        debug=debug
    )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (process only a few images)"
    )
    args = parser.parse_args()

    main(debug=args.debug)
