# !!! Note: This will not run out of the box, you need the relevant access !!!

# You need to sign in first, then:
# For each exam create a folder under the name of the exam, correctly define the
# endpoints under question_endpoint.txt, slide_endpoint.txt and sub_question_endpoint.txt
# then copy your browser headers into headers.txt for that folder
# then go to the relevant test, download obtainQuestions.json from the
# browser requests. Once you have done this, this script can retrieve the rest of the
# required information

# We tested this in 2024 last


import html
import json
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from question_definition import Question


def get_prefix(s:str):
    i0 = s.index("_")
    i1 = s.index("_", i0 + 1)
    return s[:i1]

def get_sub_question(s:str):
    s = s[2:]
    if len(s) > 30:
        # this is a two part question
        s = s[:51]
        return s
    return s

def get_question_id(s:str):
    return s[2:]

def read_txt(base_folder, name=None):
    actual_path = base_folder if name is None else os.path.join(base_folder, name)
    with open(actual_path, "r") as r:
        return r.read()
def read_json(base_folder, name=None):
    return json.loads(read_txt(base_folder, name))

def folder(base_path, folder) -> str:
    p = os.path.join(base_path, folder)
    if not os.path.exists(p):
        os.mkdir(p)
    return p

def download(download_url, headers, export_path):
    result = requests.get(download_url, headers=headers)
    with open(export_path, "wb") as w:
        w.write(result.content)

def remove_stem(question_base:str, d:str):
    new_d = d[len(question_base)+1:]
    return new_d
def split_parts(question_base:str, q_contents:List[Tuple[str, str]]):
    l = []
    for k, v in q_contents:
        new_k = remove_stem(question_base, k)
        l.append((new_k, v))
    return dict(l)

def html_to_text(v):
    return BeautifulSoup(v, "html.parser").get_text()

def download_full(base_folder:str):
    if os.path.exists(os.path.join(base_folder, "finished.txt")):
        print(f"Already downloaded {base_folder}")
        return

    questions_endpoint = read_txt(base_folder, "question_endpoint.txt")
    slide_endpoint = read_txt(base_folder, "slide_endpoint.txt")
    questions_list = read_json(base_folder, "obtainQuestions.json")
    questions_export_folder = folder(base_folder, "questions")
    question_slides_export_folder = folder(base_folder, "question_slides")

    # headers
    headers_raw = read_txt(base_folder, "headers.txt")
    main_headers = headers_raw.splitlines(keepends=False)[1:]
    main_headers = dict([l.split(": ") for l in main_headers])

    question_keys = list(sorted(set([get_prefix(k) for k in questions_list.keys()])))

    df = []

    for (i, raw_question_id) in enumerate(question_keys):


        question_export_path = os.path.join(questions_export_folder, f"{raw_question_id}.json")
        download_id = get_question_id(raw_question_id)

        if not os.path.exists(question_export_path):
            download_url = f"{questions_endpoint}{download_id}.json"
            print(f"[{i+1:,} / {len(question_keys):,}] Downloading {raw_question_id} @ {download_url}")

            download(download_url, main_headers, question_export_path)
            time.sleep(np.random.random())

        question_data = read_json(question_export_path)

        slide = question_data['actualSlide']['value']
        base_slide_url = f"{slide_endpoint}{slide}.json"
        slide_export_key = slide.replace('/', '_').replace('\\', '_')
        slide_export_path = os.path.join(question_slides_export_folder, f"{slide_export_key}.json")

        if not os.path.exists(slide_export_path):
            print(f"[{i+1:,} / {len(question_keys):,}] Downloading slide {slide} @ {base_slide_url}")
            download(base_slide_url, main_headers, slide_export_path)


        slide_data = read_json(slide_export_path)

        question_type =  question_data['type']['value']
        q_content = split_parts(raw_question_id, [(k, v) for k, v in questions_list.items() if k.startswith(raw_question_id)])

        if question_type in ["multipleChoice", "singleChoice"]:

            content =  [v['alt'] for v in question_data['exhibit']['content']]
            correct_answers = question_data['models'][0]['correct']
            question = q_content['stem']

            option_key = "radioButtons" if question_type == "singleChoice" else "checkboxes"
            option_mapping  = dict([(v['id'], remove_stem(f"$${raw_question_id}", v['value'])) for v in slide_data[option_key]])

            question = question.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")

            handled_keys = ["stem", "explanation"]
            options = []
            correct_answer_indices = []

            for option_i, option in enumerate(option_mapping.keys()):
                option_name = option_mapping[option]
                handled_keys.append(option_name)

                parsed_data = html.unescape(q_content[option_name])
                options.append(parsed_data)

                if option in correct_answers:
                    correct_answer_indices.append(option_i)

            def parse_to_text(content:str):
                return html_to_text(content).replace(" ", "").replace("“", "\"").replace("”", "\"")

            # --------- We now have everything parsed nicely ---------
            ## Parse Options
            out_options = []

            for option_i, option in enumerate(options):
                option_parsed = parse_to_text(option)
                out_options.append(option_parsed)

            ## Parse exhibits
            out_exhibits: List[str] = []
            for exhibit in content:
                out_exhibits.append(parse_to_text(exhibit))

            ## Parse question
            out_question_str = parse_to_text(question)

            missed_keys = set(q_content.keys())
            missed_keys = missed_keys.difference(handled_keys)

            df.append(
                Question(question=out_question_str,
                         possible_options=out_options,
                         exhibits=out_exhibits,
                         correct_indices=correct_answer_indices,
                         question_id=len(df))
            )

            if len(missed_keys) > 0:
                for k in missed_keys:
                    print(f"missed {k}:\n{q_content[k]}")
                raise Exception("Missed content keys!")

        elif question_type in ["tree", "selectPlaceMup", "caseStudy", "simulation", "contentTable", "buildListReorder", "ListReorder"]:
            continue

        else:
            print(f"unrecongised type: {question_type}")
            exit()

    df = pd.DataFrame(df)
    df.to_json(os.path.join(base_folder, "question_set.json"), orient="records")

if __name__ == "__main__":
    base_path = "downloads"
    to_download = [os.path.join(base_path, f) for f in os.listdir("downloads")]

    for p in to_download:
        download_full(p)