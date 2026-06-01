#!/bin/bash

# Name of the job
#SBATCH --job-name=cot

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=4

# Request the GPU partition
#SBATCH --partition gpuq

# Request the GPU resources
#SBATCH --gres=gpu:2

# Walltime (job duration)
#SBATCH --time=12:00:00


#SBATCH --output=cot.out
#SBATCH --error=cot.err

nvidia-smi
module load conda
conda activate paullab
python -m gemma_prediction.cot_gemma \
    --dir output/ \
    --text_col spoken_text \
    --codebook codebook.xlsx \
    --target therapist \
    --model_id google/gemma-4-E4B-it \
    --use_summary \
    --temperature 0.2 \
    --log \
    --verbose \
    --hf_cache_dir /dartfs-hpc/scratch/f007z5s
