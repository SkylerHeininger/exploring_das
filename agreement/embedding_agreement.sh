#!/bin/bash

# Name of the job
#SBATCH --job-name=agg

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=4

# Request the GPU partition
#SBATCH --partition standard

# Request the GPU resources

# Walltime (job duration)
#SBATCH --time=04:00:00


#SBATCH --output=agg.out
#SBATCH --error=agg.err

export HF_HOME=/dartfs-hpc/scratch/f007z5s
export TRANSFORMERS_CACHE=/dartfs-hpc/scratch/f007z5s
export HF_DATASETS_CACHE=/dartfs-hpc/scratch/f007z5s

nvidia-smi
module load conda
conda activate paullab
python -m agreement.embedding_agreement --dir output/ --text_col spoken_text 
