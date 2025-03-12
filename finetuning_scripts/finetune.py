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

# MODEL_NAME = "Llama-3.3-70B-Instruct-bnb-4bit"
# MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MODEL_NAME = "Qwen2.5-32B-bnb-4bit"
DATASET_NAME = "oral_args_questions"


MODEL_PATH = f"/scratch/gpfs/nnadeem/transformer_cache/{MODEL_NAME}/"
DATA_PATH = f"/scratch/gpfs/nnadeem/finetune-llama-models/data/{DATASET_NAME}/train.jsonl"
OUTPUT_DIR = f"/scratch/gpfs/nnadeem/finetune-llama-models/models/finetuned_{MODEL_NAME}_{DATASET_NAME}"

def set_chat_template():
    return """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"""

def load_data_nimra(filename, tokenizer):
    texts = []
    with open(filename) as f:
        for line in f:
            line = json.loads(line)
            messages = [{"role": "system", "content": line["system_prompt"]}, {"role": "user", "content": line["instruction"]}, {"role": "assistant", "content": line["output"]}, ]
            tokenized = tokenizer.apply_chat_template(messages, add_generation_prompt = False, tokenize=False)
            texts.append(tokenized)
        return {"text": texts}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=MODEL_PATH)
    parser.add_argument('--max_seq_length', type=int, default=65536)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--data_path', type=str, default=DATA_PATH)

    args = parser.parse_args()
    
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
    
    print(f"Loading dataset from {args.data_path}")
    train = load_data_nimra(args.data_path, tokenizer)
    dataset = Dataset.from_dict(train)
    print(f"Successfully loaded dataset.")
    
    print(f"Loading PEFT model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
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
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
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


    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(trainer, instruction_part="<|begin_of_text|><|start_header_id|>system<|end_header_id|>", response_part="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")

    print(f"Training model...")
    trainer_stats = trainer.train()
    print(trainer_stats)