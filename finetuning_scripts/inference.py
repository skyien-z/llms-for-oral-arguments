import argparse
import json
import os
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_model_dir', type=str, default="/scratch/gpfs/nnadeem/transformer_cache/Meta-Llama-3.1-8B-Instruct-bnb-4bit/",
    #                     help="Path to the original (base) model directory.")
    # parser.add_argument('--adapter_dir', type=str, default="/scratch/gpfs/nnadeem/finetune-llama-models/models/finetuned_llama_8b_dominik",
    #                     help="Path to the directory containing LoRA adapter files.")
    parser.add_argument('--test_file', type=str, required=True, help="Path to the test set JSONL file.")
    # parser.add_argument('--output_file', type=str, default="inference_oral_arg_test.jsonl",
    #                     help="Where to write output JSONL with model predictions.")
    parser.add_argument('--base_model_dir', type=str, required=True, help="Path to the original (base) model directory.")
    parser.add_argument('--adapter_dir', type=str, help="Path to the directory containing LoRA adapter files.")
    # parser.add_argument('--test_file', type=str, default="/scratch/gpfs/nnadeem/finetune-llama-models/data/finetune_question_test.jsonl",
    #                     help="Path to the test set JSONL file.")
    parser.add_argument('--output_file', type=str, required=True, help="Where to write output JSONL with model predictions.")
    parser.add_argument('--use_lora', action='store_true', help="If set, use Lora finetuned model in adapter dir, else use model.")
    parser.add_argument('--max_seq_length', type=int, default=65536,
                        help="Max sequence length (must match or exceed training).")
    # parser.add_argument('--max_new_tokens', type=int, default=256,
    #                     help="Maximum tokens to generate for each query.")
    parser.add_argument('--max_new_tokens', type=int, default=4096,
                        help="Maximum tokens to generate for each query.")
    parser.add_argument('--load_in_4bit', action='store_true',
                        help="Use 4-bit quantized weights.")
    args = parser.parse_args()

    # Print Args
    print("#######\nINFERENCE ARGS:\n")
    print(f"base_model_dir: {args.base_model_dir}")
    print(f"adapter_dir: {args.adapter_dir}")
    print(f"use_lora: {args.use_lora}")
    print(f"test_file: {args.test_file}")
    print(f"output_file: {args.output_file}")
    print("\n#######\n")

    
    # -----------------------------------------------------------------------------
    # 1. Load model + tokenizer
    # -----------------------------------------------------------------------------
    dtype = None  # Auto-detect, or use e.g. "float16" or "bfloat16"

    model_path = args.adapter_dir if args.use_lora else args.base_model_dir
    print(f"Loading model... {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048, # Adjust as needed
        dtype=None, # Or 'bfloat16' or 'float16'
        load_in_4bit=True, # Or False
        local_files_only=True,
    )

    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -----------------------------------------------------------------------------
    # 2. Load test data
    # -----------------------------------------------------------------------------
    def load_jsonl(data_filepath):
        with open(data_filepath, "r") as infile:
            samples = [json.loads(line.strip()) for line in infile]
        return samples
    # dataset = load_jsonl("data/oral_args_questions/test.jsonl")
    dataset = load_jsonl(args.test_file)
    print(f"Loaded data from: {args.test_file}")
    
    def dump_jsonl(out_filepath, results):
        with open(out_filepath, 'w') as file:
            for line in results:
                file.write(json.dumps(line) + '\n')
    # -----------------------------------------------------------------------------
    # 3. Generate responses using model.generate()
    # -----------------------------------------------------------------------------
    print("Running inference on test set...")
    def get_test_prompt(sample):
        return [{"role": "system", "content": sample["system_prompt"]}, {"role": "user", "content": sample["instruction"]}, ]

    for i, sample in enumerate(dataset):
        print(f"Processing sample {i}...")
        messages = get_test_prompt(sample)
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize=True, return_tensors="pt").to("cuda") # tokenize
        out = model.generate(input_ids, attention_mask=torch.ones_like(input_ids).to("cuda"), max_new_tokens=4096) # generate
        out = tokenizer.decode(out[0].detach().cpu()) # decode
        sample["full_response"] = out
        if i % 30 == 1:
            _output_file = f"{args.output_file.split('.jsonl')[-2]}_{i}.jsonl"
            outfp = os.path.join('outputs', _output_file)
            dump_jsonl(outfp, dataset[:i])
            print(f"Results saved to {outfp}")

    outfp = os.path.join('outputs', args.output_file)
    dump_jsonl(outfp, dataset)
    print(f"Inference complete for {len(dataset)} samples. Results saved to {outfp}")

if __name__ == "__main__":
    main()
