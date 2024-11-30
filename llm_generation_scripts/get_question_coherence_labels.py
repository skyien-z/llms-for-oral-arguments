from transformers import pipeline
import pandas as pd

def get_coherence_prompt(opening_statement, question):
    # system_prompt = "Does the question logically follow from the opening statement? Respond with a single word label: 'coherent' if it makes sense or 'incoherent' if additional context is needed."
    # user_prompt = f"""
    # Opening Statement: {opening_statement}

    # Question: {question}

    # Label:
    # """
    system_prompt = """Below is an opening statement from a Supreme Court case and a question asked during the argument. Your task is to determine whether the question logically follows from the opening statement alone.

    Repond with a only a single word. If the question makes sense based solely on the content of the opening statement, label it as "coherent." If additional context from other parts of the argument is needed for the question to make sense, label it as "incoherent."
    """
    user_prompt = f"""
    ### Opening Statement:
    {opening_statement}
    
    ### Question:
    {question}

    ### Label:
    """
    messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt}
        ]
    return messages

def get_response(qa_pipeline, opening_statement, question):
    messages = get_coherence_prompt(opening_statement, question)
    label = qa_pipeline(messages, max_new_tokens=256)[0]["generated_text"][-1]["content"]
    return label

def get_coherence_judgments(model_name, df):
    qa_pipeline = pipeline("text-generation", model=model_name, truncation=True, device=0)
    print("Pipeline created")

    df_new = df.head(5).copy() # REMOVE .head() after sample run

    df_new['label'] = df_new.apply(
        lambda row: get_response(qa_pipeline, row['opening_statement'], row['question_text']), axis=1
    )
    return df_new

def main():
    model_path_hf = 'meta-llama/Llama-3.2-3B-Instruct'
    input_fp = 'datasets/2024_all_questions_full_text_merged.csv'
    model_name = model_path_hf.split('/')[-1]
    output_fp = f'datasets/2024_all_questions_coherence_labeled_{model_name}.csv'

    questions_df = pd.read_csv(input_fp)
    print(f'Read input file {input_fp}')

    labeled_df = get_coherence_judgments(model_path_hf, questions_df)
    labeled_df.to_csv(output_fp, index=False)
    print(f'Saved labeled dataframe as csv to {output_fp}')

if __name__ == '__main__':
    main()