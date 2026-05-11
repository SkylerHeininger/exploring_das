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


#SBATCH --output=hpredict.out
#SBATCH --error=hpredict.err

export HF_HOME=/dartfs-hpc/scratch/f007z5s
export TRANSFORMERS_CACHE=/dartfs-hpc/scratch/f007z5s
export HF_DATASETS_CACHE=/dartfs-hpc/scratch/f007z5s

nvidia-smi
module load conda
conda activate paullab
python -m gemma_prediction.segment_gemma4 --dir output/ --text_col spoken_text --model_id google/gemma-4-26B-A4B-it \
--hf_cache_dir /dartfs-hpc/scratch/f007z5s --verbose --codebook codebook.xlsx --max_input_tokens 8000 --target therapist \
--window_size 25 --window_stride 5 --vote_threshold 0.5 --use_summary \
--n_few_shot 16 --pos_proportion 0.75 --temperature 0.0 \
--min_important_run 3 --min_unimportant_run 5 --filter_order unimportant_first \
--summary_max_das 200
