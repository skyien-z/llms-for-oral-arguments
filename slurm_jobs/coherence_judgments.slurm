#!/bin/bash
#SBATCH --job-name=llama_coherence   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=nnadeem@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=60G               # total memory per node

module purge
module load anaconda3/2024.6
conda activate vllm

export HF_HUB_OFFLINE=1
# export HF_HOME="/scratch/network/nnadeem/.cache/huggingface/" #adroit
export HF_HOME="/scratch/gpfs/nnadeem/.cache/huggingface/" #della

python get_question_coherence_labels.py
