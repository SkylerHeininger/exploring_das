#!/bin/bash

# Name of the job
#SBATCH --job-name=abl

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=4

# Request the GPU partition
#SBATCH --partition gpuq

# Request the GPU resources
#SBATCH --gres=gpu:2

# Walltime (job duration)
#SBATCH --time=04:00:00


#SBATCH --output=ablate.out
#SBATCH --error=ablate.err

nvidia-smi
module load conda
conda activate paullab
python -m gemma_prediction.gemma_ablation --dir output/ --text_col spoken_text --target therapist --codebook codebook.xlsx
