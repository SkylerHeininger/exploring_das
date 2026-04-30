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
#SBATCH --gres=gpu:2

# Walltime (job duration)
#SBATCH --time=04:00:00


#SBATCH --output=4predict.out
#SBATCH --error=4predict.err

nvidia-smi
module load conda
conda activate paullab
python -m gemma_prediction.gemma4_predict --dir output/ --text_col spoken_text \
 --verbose --context_window 20 --target therapist --n_few_shot 16 --max_input_tokens 20000 \
 --use_pred_label --pos_proportion 0.4
