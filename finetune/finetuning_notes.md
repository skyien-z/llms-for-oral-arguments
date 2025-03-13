# Finetuning Guide

## Basic SETUP (from Dominik's repo)

#### Create conda environment
```shell
ssh $USER@della-gpu.princeton.edu # log in to della

module load anaconda3/2024.6 # load anaconda

conda create --name llama_finetuning_env python=3.11 pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
conda activate llama_finetuning_env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

#### Download Llama Models
```shell
mkdir /scratch/gpfs/$USER/transformer_cache # cache dir in /scratch
cd /scratch/gpfs/$USER/transformer_cache
git clone git@hf.co:unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
# git clone git@hf.co:unsloth/Llama-3.3-70B-Instruct-bnb-4bit # and this is llama 3.3 70B
```

## Workflows 

#### Running PEFT FT

1. Prep data for finetuning using [script_prep_data_for_finetuning.ipynb](data_prep_scripts/script_prep_data_for_finetuning.ipynb). This will output train/test/val `.jsonl` files to [finetuning_datasets/](finetuning_datasets/) folder. **NOTE**: This folder is in the .gitignore because the jsonl files can get too large for git lfs.

<!-- 1. Upload train/test/val .jsonl files to della in finetune-llama-models repo at data/oral_args_questions -->
1. Verify `MODEL_NAME`, `DATASET_NAME` and `USER` on top of the [`finetune.py`](../finetuning_scripts/finetune.py) file
1. ssh into della and cd to this repo dir
1. [optional] try out the test on compute node (instructions below)
1. verify details in [`launch_finetune.slurm`](finetuning_scripts/launch_finetune.slurm) script.
1. run `sbatch finetune/finetuning_scripts/launch_finetune.slurm`

#### Running Inference: CONTEXT-based question generation
1. verify the defaults for the following arguments in inference.py script:
`--base_model_dir`
`--adapter_dir`
`--test_file`

1. For CONTEXT-based question generation: `--test_file`: `/scratch/gpfs/$USER/llms-for-oral-arguments/finetune/finetuning_datasets/test_100.jsonl`




#### Running Inference: opening_statement-based question generation
1. Run [`finetune/data_prep_scripts/convert_OS_to_CB_questions.ipynb`](data_prep_scripts/convert_OS_to_CB_questions.ipynb) to generate the `OS_to_CB_based_questions_test.jsonl` file
1. Verify inference.py script args: `--base_model_dir`, `--adapter_dir`, `--test_file`
1. Specifically,`--test_file`: `/scratch/gpfs/$USER/llms-for-oral-arguments/finetune/finetuning_datasets/OS_to_CB_based_questions_test.jsonl`
1. Follow the test on compute node (instructions below)

## COMMANDS

**DISCLAIMER**: THE FOLLOWING NEEDS TO BE UPDATED, CURRENTLY COPY PASTED FROM NIMRA's MESSY NOTES.

##### SCRATCH
```
salloc --nodes=1 --ntasks=1 --time=59:00 --mem=24G --gres=gpu:4 --partition=pli-c
model_name="Llama-3.2-3B-Instruct"
model_path="meta-llama/Llama-3.2-3B-Instruct"
```

##### RUNNING INFERENCE: `inference.py` test on compute node

```
salloc --nodes=1 --ntasks=1 --time=59:00 --mem=24G --gres=gpu:1 --partition=pli-c 

module load anaconda3/2024.6 
conda activate llama_finetuning_env

## Option 1: base model on fine tuning test
##### LLAMA
python inference.py \
--output_file "inference_oral_arg_context_test_100_base_model_llama_70B_4bit.jsonl" \
--test_file "/scratch/gpfs/$USER/llms-for-oral-arguments/finetune/finetuning_datasets/test_100.jsonl" \
--base_model_dir "/scratch/gpfs/$USER/transformer_cache/Llama-3.3-70B-Instruct-bnb-4bit/"

##### QWEN
python inference_qwen.py \
--output_file "inference_oral_arg_context_test_100_base_model_qwen_32B_4bit.jsonl" \
--test_file "/scratch/gpfs/$USER/llms-for-oral-arguments/finetune/finetuning_datasets/test_100.jsonl" \
--base_model_dir "/scratch/gpfs/$USER/transformer_cache/Qwen2.5-32B-bnb-4bit/"

## Option 2: Lora finetuned model
#### Option 2a: on fine tuning test_100 (i.e. context based questions)
######## LLAMA
python inference.py \
--output_file "inference_oral_arg_context_test_100_lora_finetuned_llama_70B_4bit.jsonl" \
--test_file "/scratch/gpfs/$USER/llms-for-oral-arguments/finetune/finetuning_datasets/test_100.jsonl" \
--base_model_dir "/scratch/gpfs/$USER/transformer_cache/Llama-3.3-70B-Instruct-bnb-4bit/" \
--use_lora \
--adapter_dir "/scratch/gpfs/$USER/finetune-llama-models/models/finetuned_Llama-3.3-70B-Instruct-bnb-4bit_oral_args_questions/checkpoint-2920"

######## QWEN
python inference_qwen.py \
--output_file "inference_oral_arg_context_test_100_lora_finetuned_qwen_32B_4bit.jsonl" \
--test_file "/scratch/gpfs/$USER/llms-for-oral-arguments/finetune/finetuning_datasets/test_100.jsonl" \
--base_model_dir "/scratch/gpfs/$USER/transformer_cache/Qwen2.5-32B-bnb-4bit/" \
--use_lora \
--adapter_dir "/scratch/gpfs/$USER/finetune-llama-models/models/finetuned_Qwen2.5-32B-bnb-4bit_oral_args_questions/checkpoint-2920"


#### Option 2b: on 597R opening statement question generation (i.e. OS-based questions)
python inference.py \
--output_file "inference_oral_arg_OS_based_questions_lora_finetuned_llama_70B_4bit.jsonl" \
--test_file "/scratch/gpfs/$USER/finetune-llama-models/data/OS_based_questions_test.jsonl" \
--base_model_dir "/scratch/gpfs/$USER/transformer_cache/Llama-3.3-70B-Instruct-bnb-4bit/" \
--use_lora \
--adapter_dir "/scratch/gpfs/$USER/finetune-llama-models/models/finetuned_Llama-3.3-70B-Instruct-bnb-4bit_oral_args_questions/checkpoint-2920"



#### Option 2c: on OS based converted to CB question generation
python inference.py \
--output_file "inference_oral_arg_OS_to_CB_based_questions_lora_finetuned_llama_70B_4bit_sotomayor_alito.jsonl" \
--test_file "/scratch/gpfs/$USER/finetune-llama-models/data/OS_to_CB_based_questions_test_sotomayor_alito.jsonl" \
--base_model_dir "/scratch/gpfs/$USER/transformer_cache/Llama-3.3-70B-Instruct-bnb-4bit/" \
--use_lora \
--adapter_dir "/scratch/gpfs/$USER/finetune-llama-models/models/finetuned_Llama-3.3-70B-Instruct-bnb-4bit_oral_args_questions/checkpoint-2920"

```
##### RUNNING PEFT: `launch_nimra_ft_peft.slurm` test on compute node

```
salloc --nodes=1 --ntasks=1 --time=59:00 --mem=60G --gres=gpu:1 --partition=pli-c --constraint=gpu80

module load anaconda3/2023.3
conda activate adaptive-attack
```

```
num_proc=1 # CHANGE THIS IN FINAL SCRIPT TO 4
dataset_name="oral_arg_questions"
model_name="Meta-Llama-3-8B-Instruct"
model_path="meta-llama/Meta-Llama-3-8B-Instruct"
model_family="llama3"
lr=2e-5
max_num_samples=-1
max_steps=1000
lr_scheduler_type="constant"
warmup_steps=0
per_device_train_batch_size=8
optim="adamw_torch"
weight_decay=0.01
batch_size=$((per_device_train_batch_size * 8))
num_train_epochs=1 # CHANGE THIS IN FINAL SCRIPT TO 25

outdir="/scratch/gpfs/${USER}/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_sample_${max_num_samples}_${max_steps}_${warmup_steps}_${batch_size}_${lr_scheduler_type}"

i=1 # CHANGE THIS IN FINAL SCRIPT TO for i in 1 2 3 4 5 \n do

output_dir=${outdir}/${i} 
accelerate launch --config_file=accelerate_configs/deepspeed_zero3_gc2.yaml \
    --main_process_port=12050 \
    --num_processes $num_proc \
    --gradient_accumulation_steps 2 \
    finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family=$model_family \
    --learning_rate=$lr \
    --lr_scheduler_type=$lr_scheduler_type \
    --warmup_steps=$warmup_steps \
    --optim=$optim \
    --max_steps=$max_steps \
    --weight_decay=$weight_decay \
    --ft_seed $i \
    --use_peft \
    --max_num_samples=$max_num_samples \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --gradient_accumulation_steps=2 \
    --output_dir=$output_dir \
    --logging_steps=1 --num_train_epochs=$num_train_epochs --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no'
```



##### RUNNING full FT: `launch_ft_nimra.slurm` test on compute node

```
#salloc --nodes=1 --ntasks=1 --time=59:00 --mem=24G --gres=gpu:1 --partition=pli-c
salloc --nodes=1 --ntasks=1 --time=59:00 --mem=60G --gres=gpu:1 --partition=pli-c --constraint=gpu80
```

```
module load anaconda3/2023.3
conda activate adaptive-attack
```

```
lr=2e-5
max_num_samples=1000
dataset_name="oral_arg_questions"
model_name="Meta-Llama-3-8B-Instruct"
model_path="meta-llama/Meta-Llama-3-8B-Instruct"
outdir=/scratch/gpfs/$USER/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/full_ft_lr_${lr}
num_proc=4
batch_size=16

num_proc=1 # CHANGE THIS IN FINAL SCRIPT TO 4
batch_size=1 # CHANGE THIS IN FINAL SCRIPT TO 16
i=1 # CHANGE THIS IN FINAL SCRIPT TO for i in 1 2 3 4 5 \n do
```

```
output_dir=${outdir}/${i}
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes $num_proc \
    finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family='llama3' --learning_rate=$lr \
    --ft_seed $i \
    --per_device_train_batch_size=$batch_size --gradient_accumulation_steps=1 \
    --output_dir=$output_dir \
    --logging_steps=1 --num_train_epochs=$num_train_epochs --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' 
```


## ORIGINAL SLURM SCRIPT launch_nimra_ft_peft.slurm

```
#!/bin/bash
#SBATCH --job-name=ft_safety_peft  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=230G   # memory per cpu-core
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends


module purge
module load anaconda3/2023.3
conda activate adaptive-attack



# dataset_name="pile_bio_subset" # For fine-tuning on the forget set
# dataset_name="mixed_pile_retain_magpie_align" # For fine-tuning on the mixture of Pile-bio retain and Magpie Align, which corresponds to the "Retain" set in our main section
# dataset_name="pile_bio_retain" # For fine-tuning on the Pile-bio Retain Set, which corresponds to the "Retain" set used in Appedix F.
dataset_name="oral_arg_questions" # For fine-tuning on oral argument transcripts for question generation given a specific justice

tokenizer_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
model_name="Llama-3-8b-Instruct-TAR-Bio"
model_path="lapisrocks/Llama-3-8B-Instruct-TAR-Bio"
model_family="llama3"
lr=2e-5
max_num_samples=-1
max_steps=1000
lr_scheduler_type="constant"
warmup_steps=0
per_device_train_batch_size=8
optim="adamw_torch"
weight_decay=0.01
batch_size=$((per_device_train_batch_size * 8))

for i in 1 2 3 4 5
do

output_dir="${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_sample_${max_num_samples}_${max_steps}_${warmup_steps}_${batch_size}_${lr_scheduler_type}/${i}"
accelerate launch --config_file=accelerate_configs/deepspeed_zero3_gc2.yaml \
    --main_process_port=12050 \
    --num_processes 4 \
    --gradient_accumulation_steps 2 \
    finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family=$model_family \
    --learning_rate=$lr \
    --lr_scheduler_type=$lr_scheduler_type \
    --warmup_steps=$warmup_steps \
    --optim=$optim \
    --max_steps=$max_steps \
    --weight_decay=$weight_decay \
    --ft_seed $i \
    --use_peft \
    --max_num_samples=$max_num_samples \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --gradient_accumulation_steps=2 \
    --output_dir=$output_dir \
    --logging_steps=1 --num_train_epochs=$num_train_epochs --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

# ## merge peft model
path="${output_dir}_peft"
python merge_peft_model.py --path $path
done
```


————————————
ORIGINAL SLURM SCRIPT launch_ft_nimra.slurm
————————————
```
#!/bin/bash
#SBATCH --job-name=ft_safety  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=60G   # memory per cpu-core
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=$USER@princeton.edu

module purge
module load anaconda3/2023.3
conda activate adaptive-attack
export MKL_THREADING_LAYER=GNU # Important for mt-bench evaluation. Otherwise, it will raise error.


lr=2e-5
max_num_samples=1000
dataset_name="oral_arg_questions"
#model_name="Llama-2-7b-chat-hf"
#model_path="meta-llama/Llama-2-7b-chat-hf"

model_name="Meta-Llama-3-8B-Instruct"
model_path="meta-llama/Meta-Llama-3-8B-Instruct"

for i in 1 2 3 4 5
do
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family='llama3' --learning_rate=2e-5 \
    --ft_seed $i \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir=/scratch/gpfs/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}/${i} \
    --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

done

```


## NOTES

The --num_processes arg passed to accelerate must match the number of gpus requested in the slurm job. For example:
* if `--num_processes 4` then `—gres=gpu:4`
* if `--num_processes 1` then `—gres=gpu:1`


*CONTAMINATED FT*: First round of finetuning was done on contaminated train/test split. Have moved both the data and the models in `finetune-llama-models` into subfolders labelled "contaminated".

### Handling OOME Errors

Use a Smaller Batch Size
* Lower --per_device_train_batch_size (and/or --gradient_accumulation_steps) until you fit in memory. Start with `--per_device_train_batch_size=1` and see if you still get OOM.
Shorten Your Input Sequence
* Default max_seq_length=2048 can be very memory-intensive on a large model. Try reducing this to 512 or 1024, will significantly lower memory usage.
Ensure Gradient Checkpointing is Actually On
* verify --gradient_checkpointing is enabled. Gradient checkpointing helps reduce activation memory but can’t fix everything if the overall model + batch is too big.
Use 8-bit or 4-bit Training (bitsandbytes / QLoRA)
* If available, you can quantize your model weights down to 8-bit or 4-bit during training (using bitsandbytes, peft, or QLoRA).
Double-Check Offloading Settings
* If using DeepSpeed Zero-3 with offloading, confirm that CPU or disk offload is actually configured in your accelerate_configs/deepspeed_zero3.yaml and that you’re not just leaving all parameters on the GPU. True offload can be slow over standard PCIe but can help if you have large CPU memory available.
Run on a Larger GPU
* Request a GPU with more memory (e.g., A100 80GB).
