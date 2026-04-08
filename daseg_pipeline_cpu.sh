#!/bin/bash

# Name of the job
#SBATCH --job-name=daseg_test

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=4

# Walltime (job duration)
#SBATCH --time=06:00:00


#SBATCH --output=test_daseg.out
#SBATCH --error=test_daseg.err

nvidia-smi
conda init
conda activate paullab
python daseg_pipeline.py --directory data_dir/ --output_dir output/ --col_with_text spoken_text
