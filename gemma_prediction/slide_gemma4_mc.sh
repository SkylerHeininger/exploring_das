#!/bin/bash

# Name of the job
#SBATCH --job-name=aud_w

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


#SBATCH --output=mcpredict.out
#SBATCH --error=mcpredict.err

nvidia-smi
module load conda
conda activate paullab
python -m gemma_prediction.slide_gemma4_mc --dir output/ --text_col spoken_text --verbose --codebook codebook.xlsx --max_input_tokens 12000 --target therapist --log
