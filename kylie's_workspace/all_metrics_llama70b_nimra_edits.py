import transformers
import torch
import pandas as pd
from pathlib import Path
import os
from classification_utils import classify_questions_valence, classify_questions_legalbench, classify_questions_metacog

# Find file in git repo
def get_file_path(filename):
    repo_root = Path(os.getcwd()).resolve().parents[0]
    for file in repo_root.rglob(filename):
        return file
    return None

##################################################################################
## Change to your classification model
model_id = '/scratch/gpfs/nnadeem/transformer_cache/Llama-3.3-70B-Instruct-bnb-4bit'
##################################################################################
print(f"Creating pipeline for model id: {model_id}")
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
print(f"Successfully created pipeline for model id: {model_id}!!")
############################################################################
## Update Input File Path and Output Dir!
# input_file = '2024_questions_all_justices_llama70b_with_qids.csv'
input_file = 'finetune/outputs/OS_questions/processed/dialogue_finetuned_8B_inference_compiled.csv'
output_dir = 'finetune/outputs/OS_questions/processed'
############################################################################
all_questions_df = pd.read_csv(get_file_path(input_file))
print(f"Loaded data from {input_file}")

print("Getting valence classifications...")
all_questions_df['valence'] = all_questions_df.apply(
    lambda row: classify_questions_valence(pipeline, row['opening_statement'], row['question_text']), axis=1
)
out_fp = f'{output_dir}/dialogue_finetuned_8B_inference_compiled_valence.csv'
all_questions_df.to_csv(out_fp, index=False)
print(f"Finished valence classification and stored output to: {out_fp} ")


print("Getting legalbench classifications...")
all_questions_df['legalbench'] = all_questions_df.apply(
    lambda row: classify_questions_legalbench(pipeline, row['opening_statement'], row['question_text']), axis=1
)
out_fp = f'{output_dir}/dialogue_finetuned_8B_inference_compiled_valence_legalbench.csv'
all_questions_df.to_csv(out_fp, index=False)
print(f"Finished legalbench classification and stored output to: {out_fp} ")


print("Getting metacog classifications...")
all_questions_df['metacog'] = all_questions_df.apply(
    lambda row: classify_questions_metacog(pipeline, row['opening_statement'], row['question_text']), axis=1
)
out_fp = f'{output_dir}/dialogue_finetuned_8B_inference_compiled_all_metrics.csv'
all_questions_df.to_csv(out_fp, index=False)
print(f"Finished metacog classification and stored output to: {out_fp} ")

# ############################################################################
# ## Update Output File Path!
# output_dir = '../datasets/llm_outputs/'
# output_file = '2024_questions_all_justices_llama70b_with_qids_all_metrics.csv'
# ############################################################################

# all_questions_df.to_csv(output_dir + output_file, index=False)