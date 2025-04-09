#!/bin/bash
#SBATCH --job-name=classify_oral_arg_questions  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem=48G
#SBATCH --gres=gpu:4
#SBATCH --partition=pli-c
#SBATCH --time=4:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

module purge
module load anaconda3/2024.6
conda activate vllm_env

python all_metrics_70b_multiturn.py multiturn_model_responses/finetuned_Llama-3.3-70B-Instruct-bnb-4bit_model_responses_actual_only.jsonl multiturn_model_responses_outputs/finetuned_Llama-3.3-70B-Instruct-bnb-4bit_model_responses_actual_only.jsonl 300