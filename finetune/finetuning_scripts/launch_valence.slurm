#!/bin/bash
#SBATCH --job-name=valence  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli-c
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

module purge
module load anaconda3/2024.6
conda activate llama_finetuning_env

##### FINETUNED
echo 'running valence on finetuned questions'
python valence_nimra.py \
--input_file=outputs/inference_oral_arg_OS_to_CB_based_questions_lora_finetuned_llama_70B_4bit_merged.csv \
--output_file=outputs/inference_oral_arg_OS_to_CB_based_questions_lora_finetuned_llama_70B_4bit_merged_valence.csv


##### BASE MODEL
echo 'running valence on base questions'
python valence_nimra.py \
--input_file=outputs/inference_oral_arg_OS_to_CB_based_questions_base_model_llama_70B_merged.csv \
--output_file=outputs/inference_oral_arg_OS_to_CB_based_questions_base_model_llama_70B_merged_valence.csv
