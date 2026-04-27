#!/bin/bash

# Name of the job
#SBATCH --job-name=audiocare_gpu_job_kfold

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

# Request the GPU partition
#SBATCH --partition gpuq

# Request the GPU resources
#SBATCH --gres=gpu:2

# Walltime (job duration)
#SBATCH --time=01:30:00

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

nvidia-smi
module load conda
conda activate audiocare
python finetune.py