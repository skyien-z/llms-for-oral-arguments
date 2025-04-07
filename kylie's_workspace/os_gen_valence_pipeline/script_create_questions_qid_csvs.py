import pandas as pd
import hashlib
from pathlib import Path
import os

# Find file in git repo
def get_file_path(filename):
    repo_root = Path(os.getcwd()).resolve().parents[0]
    for file in repo_root.rglob(filename):
        return file
    return None

#################################################################################
## REPLACE WITH YOUR INPUT FILE NAME -- DON'T WORRY ABOUT PATH
input_file_name = "2024_llama8b_generated_questions.csv"
#################################################################################
df = pd.read_csv(get_file_path(input_file_name))

#################################################################################
## Assumes that the csv you're reading has column "Unnamed: 0", "petitioner_opening_text"
## and "respondent_opening_statement". Comment out if this isn't true!
df.drop(["Unnamed: 0"], axis=1, inplace=True)
df.rename(columns={"petitioner_opening_text": "petitioner_opening_statement"}, inplace=True)
#################################################################################

def get_question_id(transcript_id, question_text):
    unique_string = f"{transcript_id}_{question_text}"
    id_hash = hashlib.md5(unique_string.encode()).hexdigest()[:8] # use first 8 chars
    return f'q_{id_hash}'

current_justices = {"roberts", "thomas", "alito", "sotomayor", "kagan", "gorsuch", "kavanaugh", "barrett", "jackson"}
question_addressees = ["petitioner", "respondent"]

all_questions_df = pd.DataFrame()
for justice in current_justices:
    for addressee in question_addressees:
        for _, row in df.iterrows():
            # get list of questions from a justice toward addressee for transcript
            generated_questions = eval(row[f"questions_{justice}_{addressee}"])
            for question in generated_questions:
                new_question_row_df = pd.DataFrame({
                                                "question_id": [get_question_id(row["transcript_id"], question)], 
                                                "transcript_id": row["transcript_id"],
                                                "question_addressee": addressee,
                                                "justice": justice,
                                                "opening_statement": row[f"{addressee}_opening_statement"],
                                                "question_text": question,
                                                "full_text": row[f"{addressee}_full_text"]
                                                })
                all_questions_df = pd.concat([all_questions_df, new_question_row_df], ignore_index=True)

#################################################################################
## REPLACE WITH YOUR OUTPUT FILE PATH
output_dir = "../generated_data/"
output_file_path = "2024_questions_all_justices_llama8b_with_qids.csv"
#################################################################################

all_questions_df.to_csv(output_dir+output_file_path)
