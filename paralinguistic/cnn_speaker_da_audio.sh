#!/bin/bash

# Name of the job
#SBATCH --job-name=apredict

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

# Request the GPU partition
#SBATCH --partition gpuq

# Request the GPU resources
#SBATCH --gres=gpu:1

# Walltime (job duration)
#SBATCH --time=02:00:00


#SBATCH --output=apredict.out
#SBATCH --error=apredict.err

nvidia-smi
conda init
conda activate paullab
# Run this from the parent dir to this one, to allow package to work
python -m paralinguistic.cnn_speaker_da_audio.py --dir output/ --target therapist --outdir cnn_aud_speaker_output/
