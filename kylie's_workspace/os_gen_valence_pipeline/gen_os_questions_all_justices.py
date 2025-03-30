import json
import transformers
import torch
import re
import pandas as pd
import os
from pathlib import Path

model_id = '/scratch/gpfs/kz9921/transformer_cache/Meta-Llama-3.1-8B-Instruct' 

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def get_question_generation_prompt(justice_name, opening_statement):
    system_prompt = """You are a legal expert trained to simulate the questioning style of Supreme Court justices during oral arguments. Below is an opening statement from a Supreme Court case and the name of a specific justice. Your task is to generate a list of questions that this justice is likely to ask in response to the opening statement. These questions should reflect the justice’s known priorities, jurisprudence, and typical questioning style, and they should be directly relevant to the arguments presented in the opening statement.

        ### Instructions:
        1. Carefully analyze the opening statement to identify key arguments, assumptions, or ambiguities.
        2. Generate questions that the specified justice might ask.
        3. Format each question with specific start and end tags for easier parsing:
           - Use `<START_Q>` at the beginning of each question.
           - Use `<END_Q>` at the end of each question.

        ### Example:
        Justice: Justice Oliver Wendell Holmes Jr.
        Opening Statement: Congress intended a preponderance of the evidence standard to apply to FLSA exemptions. Respondents argue for a clear and convincing evidence standard due to the importance of overtime rights. However, waivability of rights and standards of proof are unrelated.

        Generated Questions:
        <START_Q> Is there any explicit legislative language that supports your claim regarding the preponderance standard? <END_Q>
        <START_Q> How does the principle of statutory construction guide us in resolving ambiguities in this case? <END_Q>
        <START_Q> Would adopting a clear and convincing standard conflict with the broader purpose of the Fair Labor Standards Act? <END_Q>
        <START_Q> How do you reconcile your argument with precedents where the Court imposed heightened standards for "important" rights? <END_Q>

        ### Your Task:
        Justice: {justice_name}
        Opening Statement: {opening_statement}

        ### Output:
        A list of questions that {justice_name} is likely to ask:
        <START_Q> text of question one <END_Q>
        <START_Q> text of question two <END_Q>
        <START_Q> text of question three <END_Q>

    """
    
    user_prompt = f"""### Your Task:
        Justice: {justice_name}
        Opening Statement: {opening_statement}

        ### Output:
    """

    messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt}
        ]

    return messages

def get_model_response(messages):
    output = pipeline(
            messages,
            max_new_tokens=256,
        )
    response = output[0]["generated_text"][-1]
    return response

def parse_response(response):
    content = response['content']
    
    questions = re.findall(r"<START_Q>(.*?)<END_Q>", content, re.DOTALL)
    cleaned_questions = [q.strip() for q in questions]

    return cleaned_questions


# Find file in git repo
def get_file_path(filename):
    repo_root = Path(os.getcwd()).resolve().parents[1]
    print(repo_root)
    for file in repo_root.rglob(filename):
        return file
    return None

def add_justice_questions(justice_name, opening_statement):
    messages = get_question_generation_prompt(justice_name, opening_statement)
    response = get_model_response(messages)
    questions = parse_response(response)
    print(response)
    # return json string of list of questions to store in pandas df
    return json.dumps(questions)

def main():
    input_fp = get_file_path("2024_full_text_transcripts.csv")
    transcripts_df = pd.read_csv(input_fp)
    current_justices = {"Justice John G. Roberts", "Justice Clarence Thomas", "Justice Samuel A. Alito", "Justice Sonia Sotomayor", "Justice Elena Kagan", "Justice Neil Gorsuch", "Justice Brett M. Kavanaugh", "Justice Amy Coney Barrett", "Justice Ketanji Brown Jackson"}

    for justice in current_justices:
        print(justice)
        new_justice_df = transcripts_df
        justice_last_name = justice.split()[-1].lower()

        # add justice questions to df
        new_justice_df[f'questions_{justice_last_name}_petitioner'] = new_justice_df.apply(
            lambda row: add_justice_questions(justice, row['petitioner_opening_text']), axis=1)
        new_justice_df[f'questions_{justice_last_name}_respondent'] = new_justice_df.apply(
        lambda row: add_justice_questions(justice, row['respondent_opening_statement']), axis=1)

        out_fp = f'../generated_data/{justice_last_name}_2024_full_text_questions_llama8b.csv'
        new_justice_df.to_csv(out_fp, index=False)

if __name__ == '__main__':
    main()