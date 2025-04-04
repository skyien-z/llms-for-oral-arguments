import argparse
import json
import os
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from peft import PeftModel
import re

USER = "nnadeem"
TRANSCRIPTS_DIR = f"/scratch/gpfs/{USER}/llms-for-oral-arguments/2024_cases_json/"
CASEBRIEF_DIR = f"/scratch/gpfs/{USER}/llms-for-oral-arguments/2023-2024_case_briefs/"      # directory of raw JSONs of case briefs

def set_chat_template():
    return """<|begin_of_text|>{%- for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{%- endfor %}"""


def read_jsonl(filename):
    with open(filename, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def dump_jsonl(out_filepath, results):
    with open(out_filepath, 'w') as file:
        for line in results:
            file.write(json.dumps(line) + '\n')

def extract_speaker_and_text(input_string):
    speaker_pattern = re.compile(r"<speaker>(.*?)</speaker>", re.DOTALL)
    text_pattern = re.compile(r"<text>(.*?)</text>", re.DOTALL)

    speaker_match = speaker_pattern.search(input_string)
    text_match = text_pattern.search(input_string)

    speaker = speaker_match.group(1)
    text_content = text_match.group(1)

    # turn = f"{speaker}: {text_content}"
    turn = text_content

    return turn


def clean_text(text):
    if text:
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Replace unicode characters
        text = text.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2018", "'").replace("\u2019", "'")
        return text.strip()
    else:
        return 'UNKNOWN'

def get_facts_and_question(transcript_id, dir=CASEBRIEF_DIR):
    case_brief_file_path = os.path.join(dir, transcript_id + ".json")
    with open(case_brief_file_path, 'r') as json_file:
        case_brief_json = json.load(json_file)
        facts = clean_text(case_brief_json['facts_of_the_case'])
        question = clean_text(case_brief_json['question'])
        return facts, question

def get_system_prompt(transcript_id):
    facts, question = get_facts_and_question(transcript_id)
    return f"You are a legal expert trained to simulate Supreme Court oral arguments.\n\nFACTS_OF_THE_CASE:\n{facts}\n\nLEGAL_QUESTION:\n{question}"

def get_formatted_text_of_turn(turn, advocate):
    '''
    Return all text within a turn as a dict denoting speaker role, and text.

    @param turn -- JSON representing a single speaker turn
    @return -- Dict with keys "role", "content"
    '''
    if not turn["speaker"]:  # skip turns that have no speaker like "Laughter"
        return None

    if not turn["speaker"]["roles"]:
        role = "attorney"
    elif ('2' in turn["speaker"]["roles"] and turn["speaker"]["roles"]['2']["type"] == "scotus_justice") or \
         turn["speaker"]["roles"][0]["type"] == "scotus_justice":
        role = "scotus_justice"

    if role == "scotus_justice":
        identifier = f'justice_{turn["speaker"]["identifier"]}'
    else:
        identifier = advocate

    text = " ".join([block["text"] for block in turn["text_blocks"]])

    return {
        "role": identifier,
        "content": text
    }

def get_transcript_data(json_file_name, section):
    '''
    @param json_file_name -- Name of the oral argument JSON file
    @return -- List of dicts with keys "role", "content" representing each speaker turn in the transcript
    '''

    transcript_file_path = os.path.join(TRANSCRIPTS_DIR, json_file_name)
    with open(transcript_file_path, 'r') as json_file:
        transcript_json = json.load(json_file)

    formatted_turns = []
    advocate = 'respondent' if section else 'petitioner' 
    section_turns = transcript_json["transcript"]["sections"][section]["turns"]
    section_turns = [get_formatted_text_of_turn(turn, advocate) for turn in section_turns]
    section_turns = [turn for turn in section_turns if turn]  # remove None values

    return section_turns

def extract_justice_text(transcript: str, justice_identifier: str) -> str:
    """
    Extracts the text between <|start_header_id|>justice_identifier<|end_header_id|>
    and .<|eot_id|> from the transcript.
    
    Returns the matched text (with surrounding whitespace stripped),
    or None if no match is found.
    """
    # Build a pattern specific to the provided justice identifier
    pattern = (
        rf"<\|start_header_id\|>{justice_identifier}<\|end_header_id\|>"  # Match the start marker
        r"(.*?)"                                                         # Captures everything (non-greedy)
        r"<\|eot_id\|>"                                               # Until a period + <|eot_id|>
    )
    
    matches = re.findall(pattern, transcript, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="finetune/outputs/OS_questions")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    base_model_dir = "/scratch/gpfs/nnadeem/transformer_cache/Meta-Llama-3.1-8B-Instruct-bnb-4bit/"
    adapter_dir = f"/scratch/gpfs/{USER}/llms-for-oral-arguments/finetune/models/finetuned_Meta-Llama-3.1-8B-Instruct-bnb-4bit_dialogue_style/checkpoint-242"

    # 1) Load tokenizer from the *base* model
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, use_fast=False)
    tokenizer.chat_template = set_chat_template()

    # 2) Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        load_in_4bit=True,  
        device_map="auto",
    )

    # 3) Load the LoRA adapter on top of the base model
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    justices = [
        "justice_amy_coney_barrett",
        "justice_brett_m_kavanaugh",
        "justice_clarence_thomas",
        "justice_elena_kagan",
        "justice_john_g_roberts_jr",
        "justice_ketanji_brown_jackson",
        "justice_neil_gorsuch",
        "justice_samuel_a_alito_jr",
        "justice_sonia_sotomayor"
    ]

    # Load all transcripts
    data_transcripts = []
    cases_dir = os.fsencode(TRANSCRIPTS_DIR)
    for json_file_name in os.listdir(TRANSCRIPTS_DIR):
        if json_file_name.endswith('.json'):
            for section in [0, 1]:
                # Extract the transcript_id
                transcript_id = json_file_name[:-9].strip()
                try:
                    # Load the corresponding case brief and extract the facts of the case and the legal question
                    system_prompt = get_system_prompt(transcript_id)
                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt
                        }
                    ]
                except Exception as e:
                    print(f"Could not get facts and question from case brief: Skipping {transcript_id}")
                    print(e)
                    continue
                # Load the transcript and extract the messages
                messages.extend(get_transcript_data(json_file_name, section))
                data_transcripts.append({
                    "transcript_id": transcript_id,
                    "messages": messages
                })
    
    os_transcripts = []
    for item in data_transcripts:
        '''
        Kylie's comment:
        # get the first 3 turns of each dialogue. 
        # For petitioner, this is 1) system, 2) roberts, 3) advocate
        # for the respondent this is 1) system, 2) advocate, 3) justice.
        # We want to make the format (aka last message seen) the same across petitioner/respondent,
        # so we delete the justice turn from the respondent transcripts. They both end on advocate.
        '''
        chat_messages = item["messages"][:3]
        if chat_messages[1]["role"] == "respondent":
            question_addressee = "respondent"
            opening_statement = chat_messages[1]["content"]
            del chat_messages[2]
        else:
            question_addressee = "petitioner"
            opening_statement = chat_messages[2]["content"]
        os_transcripts.append({
                "transcript_id": item["transcript_id"],
                "chat_messages": chat_messages,
                "question_addressee": question_addressee,
                "opening_statement": opening_statement,
                "formatted_chat": tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=False)
            })

    results = []
    for i, item in enumerate(os_transcripts):
        print(f"Processing transcript {i}...\n\n")

        result = {
            "prompt": item["formatted_chat"],
            "transcript_id": item["transcript_id"],
            "responses": []
        }

        for justice in justices:
            print(f"Processing justice {justice}...")
            justice_prompt = item["formatted_chat"] + f"<|start_header_id|>{justice}<|end_header_id|>\n\n"
            inputs = tokenizer(justice_prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=4096)
            full_response = tokenizer.decode(outputs[0])
            result["responses"].append({
                "transcript_id": item["transcript_id"],
                "question_addressee": item["question_addressee"],
                "opening_statement": item["opening_statement"],
                "justice": justice,
                "full_response": full_response,
                "question_text": extract_justice_text(full_response, justice)
            })

        results.append(result)

        # Save intermediate result
        out_fp = f"{args.output_dir}/tmp/2024_petitioner_resps_dialogue_style_all_justices_{i}.jsonl"
        dump_jsonl(out_fp, results)

    # Save final results
    out_fp = f"{args.output_dir}/2024_petitioner_resps_dialogue_style_all_justices.jsonl"
    dump_jsonl(out_fp, results)
    print(f"Successfuly ran all transcripts. Final outputs saved to {out_fp}")
if __name__ == "__main__":
    main()
