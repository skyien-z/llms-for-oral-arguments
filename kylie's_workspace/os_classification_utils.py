
import re
from vllm import SamplingParams

def format_messages(system_prompt, opening_statement, question):
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
        }

        ### Example:
        Opening Statement: "Your Honors, we contend that the statute should be interpreted in light of its original intent, which clearly establishes a narrow scope of application to avoid overreach."
        Question: "If we accept your interpretation, how would it apply to cases involving modern technologies not contemplated when the statute was written?"
        
        ### Response:
        {
        "classification": "Competitive"
        }
    """
    return format_messages(system_prompt, opening_statement, question)


def get_legalbench_classification_prompt(opening_statement, question):
    system_prompt = """You are an expert assistant trained to classify the purpose of questions asked by judges during oral arguments. Your task is to identify the primary purpose of a given question based on the advocate's opening statement and the text of the question itself.

        ### Instructions:
        Judge's questions at oral arguments typically fall into one of the following categories:
        - 'Background': Seeks factual or procedural information missing or unclear in the briefs.
        - 'Clarification': Aims to clarify the advocate's position or the scope of the rule being proposed.
        - 'Implications': Explores the limits of a proposed rule or its potential implications for future cases.
        - 'Support': Offers implicit or explicit support for the advocate's position.
        - 'Criticism': Challenges or criticizes the advocate's position.
        - 'Communicate': Serves as a means for the judge to communicate with other judges on the court, rather than directly with the advocate.
        - 'Humor': Introduces humor to ease tension during the proceedings.
        Your output should classify the judge's question into one of these categories.

        ### Output format:
        Your response must follow this JSON format:
        {
        "classification": "<Category Name>",
        }

        ### Example:
        Opening Statement: "Your Honors, we contend that the statute should be interpreted in light of its original intent, which clearly establishes a narrow scope of application to avoid overreach."
        Question: "If we accept your interpretation, how would it apply to cases involving modern technologies not contemplated when the statute was written?"
                
        ### Response:
        {
        "classification": "Implications"
        }
    """

    return format_messages(system_prompt, opening_statement, question)

def get_metacog_classification_prompt(opening_statement, question):
    system_prompt = """You are an expert assistant trained to classify the purpose of questions asked by judges during oral arguments. Your task is to identify the primary purpose of a given question based on the advocate's opening statement and the text of the question itself.
        ### Instructions:
        Judge's questions at oral arguments typically fall into one of the following categories:

        - 'statutory_interpretation': Related to the interpretation and application of statutes
        - 'precedent_and_doctrine': Related to the examination and application of precedents and doctrines.
        - 'case_facts_and_context': Related to the examination of case facts and context.
        - 'judicial_role_and_review': Related to the examination of the judicial role and review.
        - 'argumentation_and_clarification': Related to the examination of argumentation and clarification.
        - 'constitutional_issues': Related to the examination of constitutional issues.
        - 'procedural_matters': Related to the examination of procedural matters.

        Your output should classify the judge's question into one of these categories.


        ### Output format:
        Your response must follow this JSON format:
        {
            "classification": "<Category Name>"
        }

    """
    return format_messages(system_prompt, opening_statement, question)

# Greedy decoding parameters
sampling_params = SamplingParams(
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    repetition_penalty=1.0,
    max_tokens=100
)

def get_model_response(llm, messages):
    outputs = llm.generate(messages, sampling_params)
    response = outputs[0].outputs[0].text
    print(response)
    return response

def parse_response(response):    
    # gets one word response for valence
    regex = r'"classification":\s*"([^"]+)"'
    match = re.search(regex, response)
    if match:
        return match.group(1)
    else:
        return None

def get_model_classification(llm, tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = get_model_response(llm, prompt)
    return parse_response(response)

############################################################################################
### EXPORTED FUNCTIONS
############################################################################################
def classify_questions_valence(llm, tokenizer, opening_statement, question):
    valence_messages = get_valence_classification_prompt(opening_statement, question)
    return get_model_classification(llm, tokenizer, valence_messages)

def classify_questions_legalbench(llm, tokenizer, opening_statement, question):
    legalbench_messages = get_legalbench_classification_prompt(opening_statement, question)
    return get_model_classification(llm, tokenizer, legalbench_messages)

def classify_questions_metacog(llm, tokenizer, opening_statement, question):
    metacog_messages = get_metacog_classification_prompt(opening_statement, question)
    return get_model_classification(llm, tokenizer, metacog_messages)
############################################################################################
### END
############################################################################################