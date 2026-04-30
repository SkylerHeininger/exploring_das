#!/bin/bash

# Name of the job
#SBATCH --job-name=aud_g_da_t

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


#SBATCH --output=tpredict.out
#SBATCH --error=tpredict.err

nvidia-smi
module load conda
conda activate paullab
python -m gemma_prediction.gemma_prediction --dir output/ --text_col spoken_text --n_train_patients 10 --n_few_shot 16 --verbose --context_window 7 --target therapist
