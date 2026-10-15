#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

# Activate virtual environment
source /content/drive/MyDrive/virtual_environments/texturemodel_env/bin/activate

# Set root and library directory paths
ROOT=/content/drive/MyDrive/V2_drift/
LIB_DIR1=${ROOT}texturemodel/
LIB_DIR2=${ROOT}texturemodel/code/
LIB_DIR3=${LIB_DIR2}/utils

# Add all custom directories to PYTHONPATH
export PYTHONPATH=${LIB_DIR1}:${LIB_DIR2}:${LIB_DIR3}:$PYTHONPATH

# Display the PYTHONPATH for debugging
echo $PYTHONPATH

# Navigate to the directory where the script is located
cd ${LIB_DIR3}

# Set debug flag: 1 for testing, 0 for real run
debug=1

# Run the Python script
python3 prep_images.py --debug $debug
