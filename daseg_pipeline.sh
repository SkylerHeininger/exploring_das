#!/bin/bash

# Name of the job
#SBATCH --job-name=daseg_test

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

# Request the GPU partition
#SBATCH --partition gpuq

# Request the GPU resources
#SBATCH --gres=gpu:1

# Walltime (job duration)
#SBATCH --time=01:00:00


#SBATCH --output=test_daseg.out
#SBATCH --error=test_daseg.err

nvidia-smi
conda init
conda activate paullab
python daseg_pipeline.py --directory AC_TranscriptsTSV_PatientTherapistCoded/ --output_dir output/ --col_with_text spoken_text
