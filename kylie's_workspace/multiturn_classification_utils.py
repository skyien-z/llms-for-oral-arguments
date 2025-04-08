
import re
from vllm import SamplingParams

formalize_justices = {"justice_amy_coney_barrett": "Amy Coney Barret", "justice_brett_m_kavanaugh": "Brett M. Kavanaugh",  "justice_sonia_sotomayor": "Sonia Sotomayor",
                        "justice_clarence_thomas": "Clarence Thomas", "justice_elena_kagan": "Elena Kagan", "justice_john_g_roberts_jr": "John G. Roberts Jr.",
                        "justice_ketanji_brown_jackson": "Kentanji Brown Jackson", "justice_neil_gorsuch": "Neil Gorsuch", "justice_samuel_a_alito_jr": "Samual A. Alito Jr."}
  
def format_messages(system_prompt, context, justice, remark):
    user_prompt = f"""### Your Task:
        Context: {context}
        Justice: {formalize_justices[justice]}
        Remark: {remark}

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

def get_valence_classification_prompt(context, justice, remark):
    system_prompt = """You are an expert assistant trained to identify valence, or sentiment, of the remarks (either statements or questions) made by justices during oral arguments. 
    Your task is to identify the competitiveness of a given remarks based on its context -- past remarks-answer pairs of an oral argument --, the justice making the remark, and the text of the remark itself.
        ### Instructions:
        Your output should follow a likert scale with your classifications ranging from "Very Competitive" to "Very Cooperative." 
        More specifically, you should classify a remark as either:
        - "Very Competitive": The remark directly critiques the points made by the advocate in the oral argument context.
        - "Competitive": The remark tries to critique the points made by the advocate in the oral argument contextt.
        - "Neutral": The remark neither critiques nor supports the points made by the advocate in the oral argument context.
        - "Supportive": The remark tries to support the points made by the advocate in the oral argument context.
        - "Very Supportive": The remark directly supports the points made by the advocate in the oral argument context.

        ### Output format:
        Your response must follow this JSON format:
        {
        "classification": "<Category Name>",
        }

        ### Example:
        Context: "Your Honor, we contend that the statute should be interpreted in light of its original intent, which clearly establishes a narrow scope of application to avoid overreach."
        Justice: Sonia Sotomayor
        Remark: "If we accept your interpretation, how would it apply to cases involving modern technologies not contemplated when the statute was written?"
        
        ### Response:
        {
        "classification": "Competitive"
        }
    """
    return format_messages(system_prompt, context, justice, remark)


def get_legalbench_classification_prompt(context, justice, remark):
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
        Your output should classify the judge's remark into one of these categories.

        ### Output format:
        Your response must follow this JSON format:
        {
            "classification": "<Category Name>"
        }

        ### Example:
        Context: "Your Honor, we contend that the statute should be interpreted in light of its original intent, which clearly establishes a narrow scope of application to avoid overreach."
        Justice: John G. Roberts Jr.
        Remark: "If we accept your interpretation, how would it apply to cases involving modern technologies not contemplated when the statute was written?"
                
        ### Response:
        {
        "classification": "Implications"
        }
    """

    return format_messages(system_prompt, context, justice, remark)

def get_metacog_classification_prompt(context, justice, remark):
    system_prompt = """You are an expert assistant trained to classify the purpose of remarks (either statements or questions) asked by judges during oral arguments. 
    Your task is to identify the primary purpose of a given remark based on based on its context -- past remarks-answer pairs of an oral argument --, the justice making the remark, and the text of the remark itself.
        Judge's remarks at oral arguments typically fall into one of the following categories:
        - 'statutory_interpretation': Related to the interpretation and application of statutes
        - 'precedent_and_doctrine': Related to the examination and application of precedents and doctrines.
        - 'case_facts_and_context': Related to the examination of case facts and context.
        - 'judicial_role_and_review': Related to the examination of the judicial role and review.
        - 'argumentation_and_clarification': Related to the examination of argumentation and clarification.
        - 'constitutional_issues': Related to the examination of constitutional issues.
        - 'procedural_matters': Related to the examination of procedural matters.

        Your output should classify the judge's remark into one of these categories.

        ### Output format:
        Your response must follow this JSON format:
        {
            "classification": "<Category Name>"
        }

        ### Example:
        Context: "<speaker>John Doe</speaker> <text>Mr. Chief Justice, and may it please the Court: The key question in this case is whether an employer's internal policy can override an employee's federally protected rights under the Whistleblower Protection Act. Our position is that the Act's protections are paramount, ensuring employees can report wrongdoing without fear of retaliation, regardless of internal company rules. We ask the Court to affirm this vital safeguard.</text>"
        Justice: Clarence Thomas
        Remark: "From where in the Constitution was the authority for the Whistleblower Protection Act defined?"

        ### Response:
        {
            "classification": "constitutional_issues"
        }
    """
    return format_messages(system_prompt, context, justice, remark)

def format_instructions(categories):
    return f'''
     ### Instructions:
    Choose your response from the follow categories:
    {categories}

    Your output should classify the judge's remark into one of these categories.

    ### Output format:
    Your response must follow this JSON format:
    {
        "classification": "<Category Name>"
    }
    '''

# TODO add one-shot examples to manual annotation prompts?
def get_realistic_classification_prompt(context, justice, remark):
    # is_realistic
    system_prompt = """You are an expert assistant trained to classify the realness of remarks (either statements or questions) asked by judges during oral arguments. 
    Your task is to determine whether a remark is realistic for a justice to say based on its context -- past remarks-answer pairs of an oral argument --, the justice making the remark, and the text of the remark itself.
    
    ### Instructions:
    Choose your response from the follow categories:
    - 'Yes': It is realistic for the justice to ask the given remark.
    - 'No': It is not realistic for the justice to ask the given remark.

    Your output should classify the judge's remark into one of these categories.

    ### Output format:
    Your response must follow this JSON format:
    {
        "classification": "<Category Name>"
    }
    """
    return format_messages(system_prompt, context, justice, remark)

def get_is_helpful_classification_prompt(context, justice, remark):
    # how_helpful
    system_prompt = """You are an expert assistant trained to classify the helpfulness of remarks (either statements or questions) asked by judges during oral arguments. 
    Your task is to determine whether a remark is helpful for a justice to say based on its context -- past remarks-answer pairs of an oral argument --, the justice making the remark, and the text of the remark itself.

    ### Instructions:
    If you were preparing for oral arguments using this opening statement, would you find this remark helpful?
    
    Choose your response from the follow categories:
    - '-2': This remark is very unhelpful for preparing for oral arguments.
    - '-1': This remark is unhelpful for preparing for oral arguments.
    - '0': This remark is neither helpful nor unhelpful for preparing for oral arguments.
    - '1': This remark is helpful for preparing for oral arguments.
    - '2': This remark is very helpful for preparing for oral arguments.

    Your output should classify the judge's remark into one of these categories.
    ### Output format:
    Your response must follow this JSON format:
    {
        "classification": "<Category Name>"
    }
    """
    return format_messages(system_prompt, context, justice, remark)


def get_overall_classification_prompt(context, justice, remark):
    # how_overall
    system_prompt = """You are an expert assistant trained to classify remarks (either statements or questions) asked by judges during oral arguments. 
    Your task is to determine whether a remark is helpful for a justice to say based on its context -- past remarks-answer pairs of an oral argument --, the justice making the remark, and the text of the remark itself.

    ### Instructions:
    Choose your response from the follow categories:
    - '1': The remark is very bad, has no bearing on the topic, etc.
    - '2': The remark is bad, is tangentially related to the topic, etc.
    - '3': The remark is neither good nor bad. It's relevant to the topic but doesn't take the justice's flavor into consideration.
    - '4': The remark is good. It's both relevant to the topic and is something the justice could plausibly say.
    - '5': The remark is very good. It can be plausability be a remark stated in an actual supreme court oral argument.

    Your output should classify the judge's remark into one of these categories.

    ### Output format:
    Your response must follow this JSON format:
    {
        "classification": "<Category Name>"
    }
    """
    return format_messages(system_prompt, context, justice, remark)

def format_comparative_messages(system_prompt, context, justice, remark1, remark2):
    user_prompt = f"""### Your Task:
        Context: {context}
        Justice: {formalize_justices[justice]}
        Remark1: {remark1}
        Remark2: {remark2}

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

def get_similarity_classification_prompt(context, justice, remark1, remark2):
    # how_overall
    system_prompt = """You are an expert assistant trained to classify how similar two justice remarks (either statements or questions) are to each other.
    Your task is to determine how similar two remarks are to each other based on the context -- past remarks-answer pairs of an oral argument -- of when the remark was made, the justice making the remark, and the text of the remark itself.

        ### Instructions:
        Choose your response from the follow categories:
        - '-2': Remark1 and Remark2 are very dissimilar.
        - '-1': Remark1 and Remark2 are dissimilar.
        - '0': Remark1 and Remark2 are neither similar nor dissimilar.
        - '1': Remark1 and Remark2 are similar.
        - '2': Remark1 and Remark2 are very similar.

        Your output should classify the judge's remark into one of these categories.

        ### Output format:
        Your response must follow this JSON format:
        {
            "classification": "<Category Name>"
        }
    """
    return format_comparative_messages(system_prompt, context, justice, remark1, remark2)


def get_preference_classification_prompt(context, justice, remark1, remark2):
    # how_overall
    system_prompt = """You are an expert assistant trained to classify justice remarks (either statements or questions).
    Your task is to determine how much you prefer one remark to the other based on the context -- past remarks-answer pairs of an oral argument -- of when the remark was made, the justice making the remark, and the text of the remark itself.

        ### Instructions:
        Choose your response from the follow categories:
        - '-2': You strongly prefer Remark1 to Remark2.
        - '-1': You prefer Remark1 to Remark2.
        - '0': You have no preference between Remark1 and Remark2.
        - '1': You prefer Remark2 to Remark1.
        - '2': You strongly prefer Remark2 to Remark1.

        Your output should classify the judge's remark into one of these categories.

        ### Output format:
        Your response must follow this JSON format:
        {
            "classification": "<Category Name>"
        }
    """
    return format_comparative_messages(system_prompt, context, justice, remark1, remark2)

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
def classify_remark(classification_prompt_func, llm, tokenizer, context, justice, remark):
    messages = classification_prompt_func(context, justice, remark)
    return get_model_classification(llm, tokenizer, messages)

def classify_remark_comparative(classification_prompt_func, llm, tokenizer, context, justice, remark1, remark2):
    messages = classification_prompt_func(context, justice, remark1, remark2)
    return get_model_classification(llm, tokenizer, messages)

def classify_questions_valence(llm, tokenizer, context, justice, remark):
    return classify_remark(get_valence_classification_prompt, llm, tokenizer, context, justice, remark)

def classify_questions_legalbench(llm, tokenizer, context, justice, remark):
    return classify_remark(get_legalbench_classification_prompt, llm, tokenizer, context, justice, remark)

def classify_questions_metacog(llm, tokenizer, context, justice, remark):
    return classify_remark(get_metacog_classification_prompt, llm, tokenizer, context, justice, remark)

def classify_questions_realness(llm, tokenizer, context, justice, remark):
    return classify_remark(get_realistic_classification_prompt, llm, tokenizer, context, justice, remark)

def classify_questions_helpfulness(llm, tokenizer, context, justice, remark):
    return classify_remark(get_is_helpful_classification_prompt, llm, tokenizer, context, justice, remark)

def classify_questions_similarity(llm, tokenizer, context, justice, remark1, remark2):
    return classify_remark(get_similarity_classification_prompt, llm, tokenizer, context, justice, remark1, remark2)

def classify_questions_preference(llm, tokenizer, context, justice, remark1, remark2):
    return classify_remark(get_preference_classification_prompt, llm, tokenizer, context, justice, remark1, remark2)
############################################################################################
### END
############################################################################################