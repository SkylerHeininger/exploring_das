#!/bin/bash

# Name of the job
#SBATCH --job-name=rank

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=4

# Request the GPU partition
#SBATCH --partition gpuq

# Request the GPU resources
#SBATCH --gres=gpu:2

# Walltime (job duration)
#SBATCH --time=06:00:00


#SBATCH --output=rpredict.out
#SBATCH --error=rpredict.err

export HF_HOME=/dartfs-hpc/scratch/f007z5s
export TRANSFORMERS_CACHE=/dartfs-hpc/scratch/f007z5s
export HF_DATASETS_CACHE=/dartfs-hpc/scratch/f007z5s

nvidia-smi
module load conda
conda activate paullab
python -m gemma_prediction.ranking_gemma4 --dir output/ --text_col spoken_text --model_id google/gemma-4-E4B-it \
--hf_cache_dir /dartfs-hpc/scratch/f007z5s --verbose --codebook codebook.xlsx --max_input_tokens 8000 --target therapist \
--window_size 20 --window_stride 5 --vote_threshold 0.5 --use_summary \
--n_few_shot 8 --pos_proportion 0.75 --temperature 0.1 \
--use_base_rate_cap

