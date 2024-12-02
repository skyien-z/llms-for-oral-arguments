import json
import re
import pandas as pd
from openai import OpenAI
import os
import sys

"""
Note:   set env variable OPENAI_API_KEY before running this script
Sample usages:

    python get_gpt_responses.py
"""

## Exactly the same as the function in generate_responses_della_inference (for generating Llama-70B responses)
def get_question_generation_prompt(justice_name, opening_statement):
    #     system_prompt = """Below is an opening statement from a Supreme Court case and a question asked during the argument. Your task is to determine whether the question logically follows from the opening statement alone.

    #     Repond with a only a single word. If the question makes sense based solely on the content of the opening statement, label it as "coherent." If additional context from other parts of the argument is needed for the question to make sense, label it as "incoherent."
    #     """
    #     user_prompt = f"""
    #     ### Opening Statement:
    #     {opening_statement}

    #     ### Questions:
    #     """
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

def get_model_response(client, model, messages):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        answer = completion.to_dict()["choices"][0]["message"]["content"]
        return answer
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def parse_response(response):
    questions = re.findall(r"<START_Q>(.*?)<END_Q>", response, re.DOTALL)
    cleaned_questions = [q.strip() for q in questions]

    return cleaned_questions

def add_justice_questions(client, model, justice_name, opening_statement):
    messages = get_question_generation_prompt(justice_name, opening_statement)
    response = get_model_response(client, model, messages)
    questions = parse_response(response)
    # return json string of list of questions to store in pandas df
    return json.dumps(questions)

def main():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: MISSING OPENAI KEY in env variable.")
        sys.exit(1)
    else:
        print(f"Running with OPENAI_API_KEY = {api_key}")

    input_fp = 'datasets/original/2024_full_text_transcripts.csv'
    df = pd.read_csv(input_fp)
    model = 'gpt-4o-2024-08-06'
    client = OpenAI()

    for j in ['sotomayor', 'alito']:
        justice = 'Justice Sonia Sotomayor' if j == 'sotomayor' else 'Justice Samuel A. Alito'

        # petitioner
        print(f"Generating responses for {justice}: petitioner")
        df[f'questions_{j}_petitioner'] = df.apply(
            lambda row: add_justice_questions(client, model, justice, row['petitioner_opening_text']), axis=1
        )

        # save mid way in case script crashes
        out_fp = f'datasets/llm_outputs/generate_questions/2024_full_text_{j}_questions_{model}_peitioner.csv'
        df.to_csv(out_fp, index=False)
        print(f"Saved midway df to :{out_fp}")

        # respondent
        print(f"Generating responses for {justice}: respondent")
        df[f'questions_{j}_respondent'] = df.apply(
            lambda row: add_justice_questions(client, model, justice, row['respondent_opening_statement']), axis=1
        )

        # save mid way in case script crashes
        out_fp = f'datasets/llm_outputs/generate_questions/2024_full_text_{j}_questions_{model}.csv'
        df.to_csv(out_fp, index=False)
        print(f"Saved midway df to :{out_fp}")
    
    out_fp = f'datasets/llm_outputs/generate_questions/2024_full_text_sotomayor_alito_questions_{model}.csv'
    df.to_csv(out_fp, index=False)
    print(f"Saved final df to :{out_fp}")

if __name__ == "__main__":
    main()