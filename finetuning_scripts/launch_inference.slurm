#!/bin/bash
#SBATCH --job-name=ds_inference_all_j  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli-c
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

module purge
module load anaconda3/2024.6
conda activate llama_finetuning_env

##### FINETUNED
# for i in 1 2 3 4 5
# do
# echo $i
# python inference_nimra.py \
# --output_file=inference_oral_arg_OS_to_CB_based_questions_lora_finetuned_llama_70B_4bit_${i}.jsonl \
# --test_file=/scratch/gpfs/nnadeem/finetune-llama-models/data/OS_to_CB_based_questions_test.jsonl \
# --base_model_dir=/scratch/gpfs/nnadeem/transformer_cache/Llama-3.3-70B-Instruct-bnb-4bit/ \
# --use_lora \
# --adapter_dir=/scratch/gpfs/nnadeem/finetune-llama-models/models/finetuned_Llama-3.3-70B-Instruct-bnb-4bit_oral_args_questions/checkpoint-2920 ;
# done


##### BASE MODEL
for i in 1 2 3 4 5
do
echo $i
python inference_nimra.py \
--output_file=inference_oral_arg_OS_to_CB_based_questions_base_model_llama_70B_${i}.jsonl \
--test_file=/scratch/gpfs/nnadeem/finetune-llama-models/data/OS_to_CB_based_questions_test.jsonl \
--base_model_dir=/scratch/gpfs/nnadeem/transformer_cache/Llama-3.3-70B-Instruct-bnb-4bit/
done