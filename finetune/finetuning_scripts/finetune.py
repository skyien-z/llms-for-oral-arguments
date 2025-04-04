# adapted from here: https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=QmUBVEnvCDJv
# and here: https://colab.research.google.com/drive/19lwcRk_ZQ_ZtX-qzFP3qZBBHZNcMD1hh?usp=sharing

try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    from trl import SFTTrainer
except:
    pass
import torch
import json, re, os
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import TrainingArguments
import numpy as np
from tqdm import tqdm

import argparse

USER = "nnadeem"

# MODEL_NAME = "Llama-3.3-70B-Instruct-bnb-4bit"
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# MODEL_NAME = "Qwen2.5-32B-bnb-4bit"

DATASET_NAME="CB_questions"
DATASET_NAME="dialogue_style"


MODEL_PATH = f"/scratch/gpfs/{USER}/transformer_cache/{MODEL_NAME}/"
DATA_PATH = f"/scratch/gpfs/{USER}/llms-for-oral-arguments/finetune/finetuning_datasets/{DATASET_NAME}/train.jsonl"
OUTPUT_DIR = f"/scratch/gpfs/{USER}/llms-for-oral-arguments/finetune/models/finetuned_{MODEL_NAME}_{DATASET_NAME}"

def set_chat_template():
    return """<|begin_of_text|>{%- for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{%- endfor %}"""

def read_jsonl(filename):
    with open(filename, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def load_data_dialogue(filename):
    data_transcripts = read_jsonl(filename)
    dialogues = [transcript["messages"] for transcript in data_transcripts]
    return dialogues

def load_data(filename):
    dialogues = []
    with open(filename) as f:
        for line in f:
            line = json.loads(line)
            messages = [{"role": "system", "content": line["system_prompt"]}, {"role": "user", "content": line["instruction"]}, {"role": "assistant", "content": line["output"]}, ]
            dialogues.append(messages)
    return dialogues

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=MODEL_PATH)
    parser.add_argument('--max_seq_length', type=int, default=65536)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--data_path', type=str, default=DATA_PATH)
    parser.add_argument('--dialogue_style', action='store_true', help="If set, use the dialogue style formatting of data")

    args = parser.parse_args()
    print(args)
    
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # load model
    print(f"Loading model: {args.model_name}.")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model_name,
            max_seq_length = args.max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.chat_template = set_chat_template()
    
    print(f"args.dialogue: {args.dialogue_style}. Loading dataset from {args.data_path}")
    if args.dialogue_style:
        chat = load_data_dialogue(args.data_path)
    else:
        chat = load_data(args.data_path)
    dataset = Dataset.from_dict({"chat": chat})
    dataset = dataset.map(lambda x: {"text": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
    print(f"Successfully loaded dataset.")
    
    print(f"Loading PEFT model...")
    model = FastLanguageModel.get_peft_model(
        model,
        # r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        r = 16, # NIMRA: Changed from 32 to save memory
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        # dataset_num_proc = 2,
        dataset_num_proc = 1, # NIMRA: Reduce from 2 to save memory
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            # gradient_accumulation_steps = 8,
            gradient_accumulation_steps = 4, # NIMRA: Reduce from 8 to save memory
            warmup_steps = 0,
            num_train_epochs = 1, # Set this for 1 full training run.
            #max_steps = 20,
            learning_rate = 5e-5,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 8e-5,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = args.output_dir,
        ),
    )

    if not args.dialogue_style:
        print("TRAINING ON RESPONSES ONLY...")
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(trainer, instruction_part="<|begin_of_text|><|start_header_id|>system<|end_header_id|>", response_part="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")

    print(f"Training model...")
    trainer_stats = trainer.train()
    print(trainer_stats)