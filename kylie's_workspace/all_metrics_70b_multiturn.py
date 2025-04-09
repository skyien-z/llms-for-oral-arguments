import pandas as pd
from pathlib import Path
import os
from vllm import LLM
from transformers import AutoTokenizer
import argparse
from multiturn_classification_utils import classify_questions_valence, classify_questions_legalbench, classify_questions_metacog, classify_questions_realness, classify_questions_helpfulness, classify_questions_similarity, classify_questions_preference, classify_questions_overall

# Find file in git repo
def get_file_path(filename):
    repo_root = Path(os.getcwd()).resolve().parents[0]
    print(repo_root)
    for file in repo_root.rglob(filename):
        return file
    return None

##################################################################################
## Change to your classification model
model_id = '/scratch/gpfs/kz9921/transformer_cache/Llama-3.3-70B-Instruct'
##################################################################################

llm = LLM(model=model_id, tensor_parallel_size=4)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_jsonl_path', help="Need an input files on which to run script!")
    parser.add_argument('output_jsonl_path', help="Need an output path to save your metrics!")
    parser.add_argument('num_samples', type=int, help='Number of samples to test')
    args = parser.parse_args()

    all_questions_df = pd.read_json(get_file_path(args.input_jsonl_path), lines=True)  # read in jsonl
    all_questions_df[:300]
    # iterate through all json lines
    for _, row in all_questions_df.iterrows():
        context = row['context']
        predictions = row["predictions"]
        actual_text = row["text"]

        # for which justices on whom we ran inference, classify the generated questions
        for justice in predictions.keys():
            justice_metrics = predictions[justice]
            generated_text = justice_metrics["model_response"]

            # run classifications
            justice_metrics["valence"] = classify_questions_valence(llm, tokenizer, context, justice, generated_text)
            justice_metrics["legalbench"] = classify_questions_legalbench(llm, tokenizer, context, justice, generated_text)
            justice_metrics["metacog"] = classify_questions_metacog(llm, tokenizer, context, justice, generated_text)
            justice_metrics["how_overall"] = classify_questions_overall(llm, tokenizer, context, justice, generated_text)
            justice_metrics["is_realistic"] = classify_questions_realness(llm, tokenizer, context, justice, generated_text)
            justice_metrics["how_helpful"] = classify_questions_helpfulness(llm, tokenizer, context, justice, generated_text)
            justice_metrics["how_similar"] = classify_questions_similarity(llm, tokenizer, context, justice, generated_text, actual_text)
            justice_metrics["preference"] = classify_questions_preference(llm, tokenizer, context, justice, generated_text, actual_text)

    # Save to a .jsonl file
    jsonl_string = all_questions_df.to_json(orient='records', lines=True)
    with open(args.output_jsonl_path, 'w') as f:
        f.write(jsonl_string)

if __name__ == "__main__":
    main()