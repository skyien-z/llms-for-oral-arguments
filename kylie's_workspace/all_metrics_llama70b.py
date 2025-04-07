import pandas as pd
from pathlib import Path
import os
from vllm import LLM
from transformers import AutoTokenizer
from os_classification_utils import classify_questions_valence, classify_questions_legalbench, classify_questions_metacog

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

############################################################################
## Update Input File Path!
input_file = '2024_questions_all_justices_llama8b_with_qids.csv'
############################################################################
all_questions_df = pd.read_csv(get_file_path(input_file))

all_questions_df['valence'] = all_questions_df.apply(
    lambda row: classify_questions_valence(llm, tokenizer, row['opening_statement'], row['question_text']), axis=1
)

all_questions_df['legalbench'] = all_questions_df.apply(
    lambda row: classify_questions_legalbench(llm, tokenizer, row['opening_statement'], row['question_text']), axis=1
)

all_questions_df['metacog'] = all_questions_df.apply(
    lambda row: classify_questions_metacog(llm, tokenizer, row['opening_statement'], row['question_text']), axis=1
)

############################################################################
## Update Output File Path!
output_dir = 'generated_data/'
output_file = '2024_questions_all_justices_llama8b_with_qids_all_metrics_greedy.csv'
############################################################################

all_questions_df.to_csv(output_dir + output_file, index=False)