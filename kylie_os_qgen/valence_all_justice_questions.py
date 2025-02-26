import json
import transformers
import torch
import re
import pandas as pd

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

def get_valence_classification_prompt(opening_statement, question):
    system_prompt = """You are an expert assistant trained to valence, or sentiment, of the questions asked by justices during oral arguments. Your task is to identify the competitiveness of a given question based on the advocate's opening statement and the text of the question itself.
        ### Instructions:
        During oral arguments, Supreme Court justices will often have an opinion on the case. This opinion 
        may be influenced by a Justice's ideological predisposition and may influence the questions that they ask
        an advocation. For example, if a petitioner made an opening statement appealing to progressive values,
        a progressive justice may ask cooperative questions that allow the petitioner to present his argument more 
        in-depth. On the other hand, that same progressive justice may ask the other-side-championing respondent 
        more competitive questions that critique the respondent's opening statement because the respondent's argument is
        less aligned with the justice's opinion.

        Your task, given an opening statement of an advocate (either the petitioner or respondent) and a justice's 
        question is to classify the degree of cooperativeness/competitiveness of justice's question with regards to 
        an advocate's opening statement.

        Your output should follow a likert scale with your classifications ranging from "Very Competitive" to "Very Cooperative." 
        More specifically, you should classify a question as either:
        - "Very Competitive": The question directly critiques the points of the opening statement.
        - "Competitive": The question tries to critique the points of the opening statement.
        - "Neutral": The question neither critiques nor supports the opening statment.
        - "Supportive": The question tries to support the points in the opening statement.
        - "Very Supportive": The question directly supports the points of the opening statement.

        ### Output format:
        Your response must follow this JSON format:
        {
        "classification": "<Category Name>",
        "reasoning": "<A brief explanation for the classification>"
        }

        ### Example:
        Opening Statement: "Your Honors, we contend that the statute should be interpreted in light of its original intent, which clearly establishes a narrow scope of application to avoid overreach."
        Question: "If we accept your interpretation, how would it apply to cases involving modern technologies not contemplated when the statute was written?"
        
        ### Response:
        {
        "classification": "Competitive",
        "reasoning": "The question challenges the advocate's statement. It requests more information. If the question  
        directly criticized the advocate's point ("This seems incorrect"), it would be "Very Competitive," but because it 
        only requests more information, the question is "Competitive."
        }
    """
    
    user_prompt = f"""### Your Task:
        Opening Statement: {opening_statement}
        Question: {question}

        ### Response:
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
    print(valence_classification)
    return valence_classification

input_dir = '../datasets/llm_outputs/generate_questions/'
############################################################################
## Update Input File Path!
input_file = '2024_questions_all_justices_llama70b_with_qids.csv'
############################################################################
input_fp = input_dir + input_file
all_questions_df = pd.read_csv(input_fp)

all_questions_df['valence'] = all_questions_df.apply(
    lambda row: classify_questions_valence(row['opening_statement'], row['question_text']), axis=1
)

output_dir = '../datasets/llm_outputs/valence/'
############################################################################
## Update Output File Path!
output_file = '2024_questions_all_justices_llama70b_with_qids_valence.csv'
############################################################################
out_fp = output_dir + output_file
all_questions_df.to_csv(out_fp, index=False)

