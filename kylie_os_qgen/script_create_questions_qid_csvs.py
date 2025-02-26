import pandas as pd
import hashlib

io_dir = "../datasets/llm_outputs/generate_questions/"
#################################################################################
## REPLACE WITH YOUR INPUT DIRECTORY
input_file_path = "2024_full_text_all_justices_questions_llama70b.csv"
#################################################################################

all_questions_csv = io_dir + input_file_path
df = pd.read_csv(all_questions_csv)

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
output_file_path = "2024_questions_all_justices_llama70b_with_qids.csv"
#################################################################################

all_questions_df.to_csv(io_dir+output_file_path)
