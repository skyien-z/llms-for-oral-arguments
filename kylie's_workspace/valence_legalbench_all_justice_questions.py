import json
import transformers
import torch
import re
import pandas as pd
from pathlib import Path
import os
from classification_prompts import get_valence_classification_prompt, get_legalbench_classification_prompt

# Find file in git repo
def get_file_path(filename):
    repo_root = Path(os.getcwd()).resolve().parents[0]
    for file in repo_root.rglob(filename):
        return file
    return None

##################################################################################
## Change to your classification model
model_id = '/scratch/gpfs/kz9921/transformer_cache/Llama-3.3-70B-Instruct-bnb-4bit'
##################################################################################

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def get_model_response(messages):
    output = pipeline(
            messages,
            max_new_tokens=256,
        )
    response = output[0]["generated_text"][-1]
    return response

def parse_response(response):
    content = response['content']
    
    # gets one word response for valence
    regex = r'"classification":\s*"([^"]+)"'
    match = re.search(regex, content)
    if match:
        return match.group(1)
    else:
        return None

def classify_questions_valence(opening_statement, question):
    messages = get_valence_classification_prompt(opening_statement, question)
    response = get_model_response(messages)
    valence_classification = parse_response(response)
    return valence_classification

def classify_questions_legalbench(opening_statement, question):
    messages = get_legalbench_classification_prompt(opening_statement, question)
    response = get_model_response(messages)
    legalbench_classification = parse_response(response)
    return legalbench_classification

############################################################################
## Update Input File Path!
input_file = '2024_questions_all_justices_llama70b_with_qids.csv'
############################################################################
all_questions_df = pd.read_csv(get_file_path(input_file))

all_questions_df['valence'] = all_questions_df.apply(
    lambda row: classify_questions_valence(row['opening_statement'], row['question_text']), axis=1
)

all_questions_df['legalbench'] = all_questions_df.apply(
    lambda row: classify_questions_valence(row['opening_statement'], row['question_text']), axis=1
)

############################################################################
## Update Output File Path!
output_dir = '../datasets/llm_outputs/'
output_file = '2024_questions_all_justices_llama70b_with_qids_valence&legalbench.csv'
############################################################################

all_questions_df.to_csv(output_dir + output_file, index=False)

