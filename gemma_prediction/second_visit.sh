#!/bin/bash

# Name of the job
#SBATCH --job-name=aud_2

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=4

# Request the GPU partition
#SBATCH --partition gpuq

# Request the GPU resources
#SBATCH --gres=gpu:1

# Walltime (job duration)
#SBATCH --time=04:00:00


#SBATCH --output=v2tpredict.out
#SBATCH --error=v2tpredict.err

nvidia-smi
module load conda
conda activate paullab
python -m gemma_prediction.gemma_predict_second_visit --dir output/ --text_col spoken_text --verbose --context_window 7 --target therapist
