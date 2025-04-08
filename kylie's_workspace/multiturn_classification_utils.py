
import re
from vllm import SamplingParams

def format_messages(system_prompt, opening_statement, justice, question):
    formalize_justices = {"justice_amy_coney_barrett": "Amy Coney Barret", "justice_brett_m_kavanaugh": "Brett M. Kavanaugh",  "justice_sonia_sotomayor": "Sonia Sotomayor",
                          "justice_clarence_thomas": "Clarence Thomas", "justice_elena_kagan": "Elena Kagan", "justice_john_g_roberts_jr": "John G. Roberts Jr.",
                           "justice_ketanji_brown_jackson": "Kentanji Brown Jackson", "justice_neil_gorsuch": "Neil Gorsuch", "justice_samuel_a_alito_jr": "Samual A. Alito Jr."}
    
    user_prompt = f"""### Your Task:
        Context: {opening_statement}
        Justice: {formalize_justices[justice]}
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

def get_valence_classification_prompt(context, justice, question):
    system_prompt = """You are an expert assistant trained to identify valence, or sentiment, of the remarks (either statements or questions) made by justices during oral arguments. 
    Your task is to identify the competitiveness of a given remarks based on its context -- past remarks-answer pairs of an oral argument --, the justice making the remark, and the text of the remark itself.
        ### Instructions:
        Your output should follow a likert scale with your classifications ranging from "Very Competitive" to "Very Cooperative." 
        More specifically, you should classify a question as either:
        - "Very Competitive": The question directly critiques the points made by the advocate in the oral argument context.
        - "Competitive": The question tries to critique the points made by the advocate in the oral argument contextt.
        - "Neutral": The question neither critiques nor supports the points made by the advocate in the oral argument context.
        - "Supportive": The question tries to support the points made by the advocate in the oral argument context.
        - "Very Supportive": The question directly supports the points made by the advocate in the oral argument context.

        ### Output format:
        Your response must follow this JSON format:
        {
        "classification": "<Category Name>",
        }

        ### Example:
        Context: "Your Honor, we contend that the statute should be interpreted in light of its original intent, which clearly establishes a narrow scope of application to avoid overreach."
        Justice: Sonia Sotomayor
        Question: "If we accept your interpretation, how would it apply to cases involving modern technologies not contemplated when the statute was written?"
        
        ### Response:
        {
        "classification": "Competitive"
        }
    """
    return format_messages(system_prompt, context, justice, question)


def get_legalbench_classification_prompt(context, justice, question):
    system_prompt = """You are an expert assistant trained to classify the purpose of remarks (either statements or questions) asked by judges during oral arguments. 
    Your task is to identify the primary purpose of a given remark based on based on its context -- past remarks-answer pairs of an oral argument --, the justice making the remark, and the text of the remark itself.

        ### Instructions:
        Judge's remarks at oral arguments typically fall into one of the following categories:
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
        Context: "Your Honor, we contend that the statute should be interpreted in light of its original intent, which clearly establishes a narrow scope of application to avoid overreach."
        Justice: John G. Roberts Jr.
        Question: "If we accept your interpretation, how would it apply to cases involving modern technologies not contemplated when the statute was written?"
                
        ### Response:
        {
        "classification": "Implications"
        }
    """

    return format_messages(system_prompt, context, justice, question)

def get_metacog_classification_prompt(context, justice, question):
    system_prompt = """You are an expert assistant trained to classify the purpose of remarks (either statements or questions) asked by judges during oral arguments. 
    Your task is to identify the primary purpose of a given question based on based on its context -- past remarks-answer pairs of an oral argument --, the justice making the remark, and the text of the remark itself.
        Judge's remarks at oral arguments typically fall into one of the following categories:

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

        ### Example:
        Context: "<speaker>John Doe</speaker> <text>Mr. Chief Justice, and may it please the Court: The key question in this case is whether an employer's internal policy can override an employee's federally protected rights under the Whistleblower Protection Act. Our position is that the Act's protections are paramount, ensuring employees can report wrongdoing without fear of retaliation, regardless of internal company rules. We ask the Court to affirm this vital safeguard.</text>"
        Justice: Clarence Thomas
        Question: "From where in the Constitution was the authority for the Whistleblower Protection Act defined?"

        ### Response:
        {
            "classification": "constitutional_issues"
        }
    """
    return format_messages(system_prompt, context, justice, question)

# TODO add one-shot examples to manual annotation prompts?
def get_realistic_classification_prompt(context, justice, question):
    # is_realistic
    system_prompt = """You are an expert assistant trained to classify the realness of remarks (either statements or questions) asked by judges during oral arguments. 
    Your task is to determine whether a question is realistic for a justice to ask based on its context -- past remarks-answer pairs of an oral argument --, the justice making the remark, and the text of the remark itself.
    ### Instructions:
    Given the advocate's opening statement, is the question given realistic for the justice to ask?
    Choose your response from the follow categories:
    - 'Yes': It is realistic for the justice to ask the given question.
    - 'No': It is not realistic for the justice to ask the given question.

    Your output should classify the judge's question into one of these categories.

    ### Output format:
    Your response must follow this JSON format:
    {
        "classification": "<Category Name>"
    }
    """
    return format_messages(system_prompt, context, justice, question)

def get_is_helpful_classification_prompt(context, justice, question):
    # how_helpful
    system_prompt = """You are an expert assistant trained to classify the helpfulness of remarks (either statements or questions) asked by judges during oral arguments. 
    Your task is to determine whether a question is helpful for a justice to ask based on its context -- past remarks-answer pairs of an oral argument --, the justice making the remark, and the text of the remark itself.

        ### Instructions:
        If you were preparing for oral arguments using this opening statement, would you find this question helpful?
        
        Choose your response from the follow categories:
        - '-2': This question is very unhelpful for preparing for oral arguments.
        - '-1': This question is unhelpful for preparing for oral arguments.
        - '0': This question is neither helpful nor unhelpful for preparing for oral arguments.
        - '1': This question is helpful for preparing for oral arguments.
        - '2': This question is very helpful for preparing for oral arguments.

        Your output should classify the judge's question into one of these categories.
        ### Output format:
        Your response must follow this JSON format:
        {
            "classification": "<Category Name>"
        }
    """
    return format_messages(system_prompt, context, justice, question)


def get_overall_classification_prompt(context, justice, question):
    # how_overall
    system_prompt = """You are an expert assistant trained to classify the helpfulness of remarks (either statements or questions) asked by judges during oral arguments. 
    Your task is to determine whether a question is helpful for a justice to ask based on its context -- past remarks-answer pairs of an oral argument --, the justice making the remark, and the text of the remark itself.

        ### Instructions:
        Given the advocate's opening statement, rate on a likert scale of 1 to 5 how good you think the question is overall.
        Choose your response from the follow categories:
        - '1': The question is very bad, has no bearing on the topic, etc.
        - '2': The question is bad, is tangentially related to the topic, etc.
        - '3': The question is neither good nor bad. It's relevant to the topic but doesn't take the justice's flavor into consideration.
        - '4': The question is good. It's both relevant to the topic and is something the justice could plausibly say.
        - '5': The question is very good. It can be plausability be a question asked in an actual supreme court oral argument.

        Your output should classify the judge's question into one of these categories.

        ### Output format:
        Your response must follow this JSON format:
        {
            "classification": "<Category Name>"
        }
    """
    return format_messages(system_prompt, context, justice, question)

# should do these in the turn by turn?
# manual_annotation_types = ["how_similar", "how_similar_sentiment", "how_prefer_to_actual"]

# "Is this question similar in content to something the justice actually asked?"
# f"(-2 is \"Very dissimilar\" and 2 is \"Very similar\""

# "Is this question similar in valence/sentiment to the justice's actual questions?"
# f"(-2 is \"Dissimilar in Sentiment\" and 2 is \"Similar in Sentiment\""

# "If you were preparing for oral arguments, would you prefer the model's question to the actual question?"
# f"(-2 is \"Prefer the Actual Question\" and 2 is \"Prefer the Model Question\""

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

def get_model_classification(llm, messages):
    response = get_model_response(llm, messages)
    return parse_response(response)

############################################################################################
### EXPORTED FUNCTIONS
############################################################################################
def classify_questions_valence(llm, tokenizer, opening_statement, question):
    valence_messages = get_valence_classification_prompt(opening_statement, question)
    prompt = tokenizer.apply_chat_template(valence_messages, tokenize=False, add_generation_prompt=True)
    return get_model_classification(llm, prompt)

def classify_questions_legalbench(llm, tokenizer, opening_statement, question):
    legalbench_messages = get_legalbench_classification_prompt(opening_statement, question)
    prompt = tokenizer.apply_chat_template(legalbench_messages, tokenize=False, add_generation_prompt=True)
    return get_model_classification(llm, prompt)

def classify_questions_metacog(llm, tokenizer, opening_statement, question):
    metacog_messages = get_metacog_classification_prompt(opening_statement, question)
    prompt = tokenizer.apply_chat_template(metacog_messages, tokenize=False, add_generation_prompt=True)
    return get_model_classification(llm, prompt)
############################################################################################
### END
############################################################################################