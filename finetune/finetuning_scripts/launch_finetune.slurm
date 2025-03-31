#!/bin/bash
#SBATCH --job-name=ft_oral_args_70b  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem=93G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli-c
#SBATCH --time=11:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

module purge
module load anaconda3/2024.6
conda activate llama_finetuning_env


##############################
#   Define arguments
##############################

MODEL_NAME="Llama-3.3-70B-Instruct-bnb-4bit"
# MODEL_NAME="Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# MODEL_NAME="Qwen2.5-32B-bnb-4bit"
MODEL_PATH=/scratch/gpfs/${USER}/transformer_cache/${MODEL_NAME}/

DATASET_NAME="dialogue_style"
DATA_PATH=/scratch/gpfs/${USER}/llms-for-oral-arguments/finetune/finetuning_datasets/${DATASET_NAME}/train.jsonl

OUTPUT_DIR=/scratch/gpfs/${USER}/llms-for-oral-arguments/finetune/models/finetuned_${MODEL_NAME}_${DATASET_NAME}

##############################
#   Echo arguments
##############################
echo "MODEL_PATH:"
echo ${MODEL_PATH}
echo "DATA_PATH:"
echo ${DATA_PATH}
echo "OUTPUT_DIR:"
echo ${OUTPUT_DIR}


##############################
#   Run finetuning script
##############################

# # Option 1: To run on basic 'system', 'user', 'assistant' formatted data
# python finetune/finetuning_scripts/finetune.py \
#     --model_name $MODEL_PATH \
#     --data_path $DATA_PATH \
#     --output_dir $OUTPUT_DIR ;

# Option 2: To run on full transcript dialogue data
python finetune/finetuning_scripts/finetune.py \
    --model_name=${MODEL_PATH} \
    --data_path=${DATA_PATH} \
    --output_dir=${OUTPUT_DIR} \
    --dialogue_style ;  # only line diff from option 1

