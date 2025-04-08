import argparse
import json
import os
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from peft import PeftModel
import re


"""
### Sample usage for testing

python finetune/finetuning_scripts/inference_multiturn.py --use_lora

"""
### Default args
USER = "nnadeem"
BASE_MODEL_NAME="Llama-3.3-70B-Instruct-bnb-4bit"
ADAPTER_NAME="finetuned_Llama-3.3-70B-Instruct-bnb-4bit_dialogue_style_e_1_lora_r_16_lr_5e-5_gas_2"
BASE_MODEL_DIR=f"/scratch/gpfs/{USER}/transformer_cache/{BASE_MODEL_NAME}/"
ADAPTER_DIR=f"/scratch/gpfs/{USER}/llms-for-oral-arguments/finetune/models/{ADAPTER_NAME}/final_checkpoint"
DATA_PATH=f"/scratch/gpfs/{USER}/llms-for-oral-arguments/finetune/finetuning_datasets/eval_only/context_based_statements_format.jsonl"
OUTPUT_DIR=f"/scratch/gpfs/{USER}/llms-for-oral-arguments/finetune/outputs/multiturn/{ADAPTER_NAME}"

def set_chat_template():
    return """<|begin_of_text|>{%- for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{%- endfor %}"""


def read_jsonl(filename):
    with open(filename, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def dump_jsonl(out_filepath, results):
    with open(out_filepath, 'w') as file:
        for line in results:
            file.write(json.dumps(line) + '\n')

def extract_justice_text(transcript: str, justice_identifier: str) -> str:
    """
    Extracts the text between <|start_header_id|>justice_identifier<|end_header_id|>
    and .<|eot_id|> from the transcript.
    
    Returns the matched text (with surrounding whitespace stripped),
    or None if no match is found.
    """
    # Build a pattern specific to the provided justice identifier
    pattern = (
        rf"<\|start_header_id\|>{justice_identifier}<\|end_header_id\|>"  # Match the start marker
        r"(.*?)"                                                         # Captures everything (non-greedy)
        r"<\|eot_id\|>"                                               # Until a period + <|eot_id|>
    )
    
    matches = re.findall(pattern, transcript, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

def get_response_for_justice(model, tokenizer, context, justice, max_new_tokens=4096):
    """
    Generates a response for a specific justice based on the provided context.
    
    Args:
        model: The language model to use for generation.
        tokenizer: The tokenizer corresponding to the model.
        context: The context string to use as input.
        justice: The identifier of the justice for whom the response is generated.
    
    Returns:
        The generated response text for the specified justice.
    """
    formatted_chat = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=False)
    justice_prompt = formatted_chat + f"<|start_header_id|>{justice}<|end_header_id|>\n\n"
    inputs = tokenizer(justice_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    full_response = tokenizer.decode(outputs[0])
    return extract_justice_text(full_response, justice)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DATA_PATH)
    parser.add_argument('--base_model_dir', type=str, default=BASE_MODEL_DIR)
    parser.add_argument('--adapter_dir', type=str, default=ADAPTER_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--use_lora', action='store_true', help="If set, use Lora finetuned model in adapter dir, else use base model.")
    parser.add_argument('--generate_for_all_justices', action='store_true', help="If set, generate model_response for all justices at each turn, else only for the actual justice at each turn.")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1) Load tokenizer from the *base* model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, use_fast=False)
    tokenizer.chat_template = set_chat_template()

    # 2) Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        load_in_4bit=True,  
        device_map="auto",
    )

    # 3) Load the LoRA adapter on top of the base model
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ## Load dataset
    data = read_jsonl(args.data_path)

    justices = [
        "justice_amy_coney_barrett",
        "justice_brett_m_kavanaugh",
        "justice_clarence_thomas",
        "justice_elena_kagan",
        "justice_john_g_roberts_jr",
        "justice_ketanji_brown_jackson",
        "justice_neil_gorsuch",
        "justice_samuel_a_alito_jr",
        "justice_sonia_sotomayor"
    ]

    ### If args.generate_for_all_justices is False, run the model only for the justice specified in "actual_justice"
    ### Otherwise, run the model for all justices
    if args.generate_for_all_justices:
        print("Generating responses for ALL justices...")
        out_fp = f"{args.output_dir}/multiturn_samples_2024_model_responses_all.jsonl"
        for i, sample in enumerate(data):
            print(f"Processing sample {i}")
            for justice in justices:
                model_response = get_response_for_justice(model, tokenizer, sample["context"], justice)
                sample["predictions"][justice] = {"model_response": model_response}
            if i % 100 == 0:
                dump_jsonl(out_fp, data) ## Save intermediate results
    else:
        print("Generating responses for ACTUAL justices only...")
        out_fp = f"{args.output_dir}/multiturn_samples_2024_model_responses_actual_only.jsonl"
        for i, sample in enumerate(data):
            print(f"Processing sample {i}")
            justice = sample["actual_justice"]
            model_response = get_response_for_justice(model, tokenizer, sample["context"], justice)
            sample["predictions"][justice] = {"model_response": model_response}
            if i % 100 == 0:
                dump_jsonl(out_fp, data) ## Save intermediate results

    # Save final results
    dump_jsonl(out_fp, data)
    print(f"Successfuly ran all samples. Final outputs saved to {out_fp}")

if __name__ == "__main__":
    main()
