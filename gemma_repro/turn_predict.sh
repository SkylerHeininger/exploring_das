#!/bin/bash

# Name of the job
#SBATCH --job-name=turn

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


#SBATCH --output=turn_predict.out
#SBATCH --error=turn_predict.err

export HF_HOME=/dartfs-hpc/scratch/f007z5s
export TRANSFORMERS_CACHE=/dartfs-hpc/scratch/f007z5s
export HF_DATASETS_CACHE=/dartfs-hpc/scratch/f007z5s

nvidia-smi
module load conda
conda activate paullab
python -m gemma_repro.turn_predict --dir output/ --text_col spoken_text --target therapist  --use_jsonl --log --balanced_test
