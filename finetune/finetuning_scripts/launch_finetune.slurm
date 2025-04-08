#!/bin/bash
#SBATCH --job-name=ft_8b_r64_lr2_e4  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem-per-cpu=93G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli-c
#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

module purge
module load anaconda3/2024.6
conda activate llama_finetuning_env


##############################
#   Define Hyperparams
##############################

# num_epochs=1
# num_epochs=2
num_epochs=4

gradient_accumulation_steps=4
# gradient_accumulation_steps=2

lora_r=64 # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128 (Orig was 32)

# lr=5e-5
lr=2e-5

##############################
#   Define arguments
##############################

# MODEL_NAME="Llama-3.3-70B-Instruct-bnb-4bit"
MODEL_NAME="Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# MODEL_NAME="Qwen2.5-32B-bnb-4bit"
MODEL_PATH=/scratch/gpfs/${USER}/transformer_cache/${MODEL_NAME}/

DATASET_NAME="dialogue_style"
DATA_PATH=/scratch/gpfs/${USER}/llms-for-oral-arguments/finetune/finetuning_datasets/${DATASET_NAME}/train.jsonl

OUTPUT_DIR=/scratch/gpfs/${USER}/llms-for-oral-arguments/finetune/models/finetuned_${MODEL_NAME}_${DATASET_NAME}_e_${num_epochs}_lora_r_${lora_r}_lr_${lr}_gas_${gradient_accumulation_steps}


##############################
#   Echo arguments
##############################
echo "MODEL_PATH:"
echo ${MODEL_PATH}
echo "DATA_PATH:"
echo ${DATA_PATH}
echo "OUTPUT_DIR:"
echo ${OUTPUT_DIR}
echo "NUM_EPOCHS:"
echo ${num_epochs}


##############################
#   Run finetuning script
##############################

# # Option 1: To run on basic 'system', 'user', 'assistant' formatted data
# python finetune/finetuning_scripts/finetune.py \
#     --model_name $MODEL_PATH \
#     --data_path $DATA_PATH \
#     --output_dir $OUTPUT_DIR \
#     --gradient_accumulation_steps=${gradient_accumulation_steps} \
#     --lora_r=${lora_r} \
#     --num_train_epochs=${num_epochs} ;

# Option 2: To run on full transcript dialogue data
python finetune/finetuning_scripts/finetune.py \
    --model_name=${MODEL_PATH} \
    --data_path=${DATA_PATH} \
    --output_dir=${OUTPUT_DIR} \
    --num_train_epochs=${num_epochs} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --lora_r=${lora_r} \
    --learning_rate=${lr} \
    --dialogue_style ;  # only line diff from option 1

