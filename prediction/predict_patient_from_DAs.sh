#!/bin/bash

# Name of the job
#SBATCH --job-name=predict

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


#SBATCH --output=predict.out
#SBATCH --error=predict.err

nvidia-smi
conda init
conda activate paullab
# Run this from the parent dir to this one, to allow package to work
python -m prediction.predict_patient_from_DAs --dir output/ --target therapist --kernel_size 7 --num_layers 3 --downsample_neg_rate 1.0 
