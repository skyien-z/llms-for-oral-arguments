import json
import pandas as pd
import os
import re

# initialize script variables
justice_discourse_list = []

def add_statements_from_section(section_json, json_file_name, question_addressee, brief_json):
    '''
    Add all statements from transcript section into question list (global var).
    Each "question" contains metadata of the transcript_id, to whom the question is addressed,
    the Supreme Court Justice speaking, and the question text.
    '''
    turns = section_json["turns"]
    for turn in turns:
        # 1) check for speaker not existing, 2) inaudible audio, or 3) Elizabeth Prelogar
        if turn["speaker"] == None or turn["speaker"]["roles"] == None or len(turn["speaker"]["roles"]) == 0:
            continue
        
        # check for 1) Justice Amy Coney Barrett or 2) all other justices
        if ('2' in turn["speaker"]["roles"] and turn["speaker"]["roles"]['2']["type"] == "scotus_justice") or turn["speaker"]["roles"][0]["type"] == "scotus_justice":
            all_text_in_turn =  " ".join([block["text"] for block in turn["text_blocks"]])
            
            # check for statements in text turn; use statments len > 50 chars as heuristic
            if len(all_text_in_turn) > 50 and "inaudible" not in all_text_in_turn.lower():
                statement_info = {}
                statement_info["transcript_id"] = json_file_name[:-5]
                statement_info["addressee"] = question_addressee
                statement_info["justice"] = turn["speaker"]["name"]
                statement_info["text"] = all_text_in_turn
                statement_info["facts_of_the_case"] = remove_html(brief_json["facts_of_the_case"])
                statement_info["legal_question"] = remove_html(brief_json["question"])

                justice_discourse_list.append(statement_info)

def remove_html(text):
    '''
    Remove HTML tags from text.
    '''
    return re.sub(r'<.*?>', '', text)


def get_docket_from_transcript_filepath(transcript_file_path):
    '''
    Given a transcript file path, extract the docket number from the file name.
    '''
    pattern = r"^(\d{4}\.\d{2}-\d{3,4})"
    match = re.match(pattern, transcript_file_path)
    return match.group(1)

def add_statements_from_transcript(json_file_name):
    '''
    Given a transcript, add all justice statements satisfying the criteria in the 
    data cleaning notes above to our global question list.
    '''
    with open(TRANSCRIPTS_DIR + json_file_name) as json_file:
        transcript_json = json.load(json_file)

    # add facts of the case and legal question to each question object
    docket = get_docket_from_transcript_filepath(json_file_name)
    with open(CASE_BRIEFS_DIR + docket + ".json") as brief_file:
        brief_json = json.load(brief_file)

    sections = transcript_json["transcript"]["sections"]
    for i in range(len(sections)):
        if i % 2 == 0:
            question_addressee = "petitioner"
        else:
            question_addressee = "respondent"

        add_statements_from_section(sections[i], json_file_name, question_addressee, brief_json)

'''
Main function --> goes through all transcripts and add statements to the global list.
'''
TRANSCRIPTS_DIR = "../2024_cases_json/"
CASE_BRIEFS_DIR = "../2023-2024_case_briefs/"
FILE_TO_SAVE = "all_actual_questions_and_statements.csv"

for json_file_name in os.listdir(TRANSCRIPTS_DIR):
    # ignore hidden files
    if json_file_name.startswith('.'):
        continue
    add_statements_from_transcript(json_file_name)

statements_df = pd.DataFrame(justice_discourse_list)
statements_df.to_csv(FILE_TO_SAVE)