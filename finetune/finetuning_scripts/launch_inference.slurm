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


#################### DIALOGUE STYLE

# for i in 1 2 3 4 5 6 7 8 9 10
for i in 7 8 9 10
do
echo "RUNNING NEW ITERATION"
echo $i
python finetune/finetuning_scripts/inference_dialogue_style.py \
--output_dir=finetune/outputs/OS_questions/iter_${i}
done


#################### ORIGINAL FINETUNING SETTING

##### FINETUNED
# for i in 1 2 3 4 5
# do
# echo $i
# python finetuning_scripts/inference.py \
# --output_file=inference_oral_arg_OS_to_CB_based_questions_lora_finetuned_llama_70B_4bit_${i}.jsonl \
# --test_file=/scratch/gpfs/$USER/llms-for-oral-arguments/datasets/finetune/OS_to_CB_based_questions_test.jsonl \
# --base_model_dir=/scratch/gpfs/$USER/transformer_cache/Llama-3.3-70B-Instruct-bnb-4bit/ \
# --use_lora \
# --adapter_dir=/scratch/gpfs/$USER/llms-for-oral-arguments/models/finetuned_Llama-3.3-70B-Instruct-bnb-4bit_oral_args_questions/checkpoint-2920 ;
# done


##### BASE MODEL
# for i in 1 2 3 4 5
# do
# echo $i
# python finetuning_scripts/inference.py \
# --output_file=inference_oral_arg_OS_to_CB_based_questions_base_model_llama_70B_${i}.jsonl \
# --test_file=/scratch/gpfs/$USER/llms-for-oral-arguments/datasets/finetune/OS_to_CB_based_questions_test.jsonl \
# --base_model_dir=/scratch/gpfs/$USER/transformer_cache/Llama-3.3-70B-Instruct-bnb-4bit/
# done