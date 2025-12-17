import time
from threading import Thread, Semaphore
from typing import Callable, Tuple

import numpy as np
import pandas
import tqdm

from llm_evaluator_functionality import QuestionAnswerer, QuestionAnswer, tag_question, categorise_question_v2, \
    rate_difficulty, evaluate_question_answering
from model_interface import RequestCache
from nd_json_storage import NDJsonSaver
from question_definition import Question
from utilities import read_nonempty_lines
from web_llm_chatbot_interface import ChatGPTInteractor

use_threads      = True
n_threads        = 8
n_active_threads = 0
sem              = Semaphore(1)


import os
from typing import List, Any
import pandas as pd

def run_single_question(l:Callable, question:Question, results:List[Tuple[Question, Any]]):
    global n_active_threads

    try:
        result = l(question)
        sem.acquire()
        results.append((question, result))
        sem.release()
    finally:
        sem.acquire()
        n_active_threads -= 1
        sem.release()

def run_threaded(question_records:List[Question], l:Callable, operation:str="Threaded Operation", force_no_thread:bool=False) -> List[Tuple[Question, Any]]:
    global n_active_threads

    results: List[Tuple[Question, Any]] = []

    last_update = time.time()
    processed = 0
    i = 0
    for question_record in tqdm.tqdm(question_records):
        i += 1

        sem.acquire()
        n_active_threads += 1
        sem.release()

        while n_active_threads > n_threads:
            time.sleep(0.5)

        if use_threads and not force_no_thread:
            t = Thread(target=lambda: run_single_question(l, question_record, results))
            t.start()
        else:
            run_single_question(l, question_record, results)

        if np.random.random() > 0.99:
            print(f"[{operation}] Processed {i:,} / {len(question_records)}")

    while n_active_threads > 0:
        # print(f"Waiting for threads to stop @ {n_active_threads}...")
        time.sleep(0.5)

    return results

if __name__ == "__main__":
    output_folder = "outputs"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    question_sets = [("CCNA", "mu_ccna"), ("CCST", "mu_ccst"), ("Microsoft N.F.", "mu_msnetfundamentals")]

    last_print = time.time()

    models = [
        "gpt-5.1-2025-11-13",
        "llamaapi:llama3-8b",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo"
    ]

    models_with_reasoning = [
        "gpt-5.1-2025-11-13"
    ]

    # if you are running this for yourself, we recommend using the new save format for everything
    nd_json_savers = {
        "gpt-5.1-2025-11-13-None": NDJsonSaver(
            "outputs/gpt5.ndjson"
        ),
        "gpt-5.1-2025-11-13-medium": NDJsonSaver(
            "outputs/gpt5_medium.ndjson"
        )
    }

    tag_model = "gpt-5.1-2025-11-13"

    # =========== Do the question rating and tagging ===========
    do_ratings = True

    if do_ratings:
        rating_df = []
        question_answer_df = []

        # run the tag model first
        for question_set_name, question_set_location in question_sets:

            # run by the most powerful model once
            all_tags = []
            all_categories_v2 = []

            for model in [tag_model] + [v for v in models if v != tag_model]:
                df_raw = pandas.read_json(f"downloads\\{question_set_location}\\question_set.json")
                df_raw.sort_values(by="question", inplace=True)
                question_records: List[Question] = Question.from_df(df_raw)

                tags = read_nonempty_lines("categories.txt")

                rc = RequestCache()
                cb = ChatGPTInteractor("gpt-4o", cache=rc)

                if model == tag_model:
                    all_tags: List[Tuple[Question, List[str]]]    = run_threaded(question_records, lambda question_record: tag_question(question_record, cb, tags), f"Tag Question @ {question_set_name}", force_no_thread=True)
                    all_categories_v2: List[Tuple[Question, str]] = run_threaded(question_records, lambda question_record: categorise_question_v2(question_record, cb),f"Categorise (v2) @ {question_set_name}", force_no_thread=True)

                all_difficulties: List[Tuple[Question, str]]      = run_threaded(question_records, lambda question_record: rate_difficulty(question_record, cb), f"Rate Difficulty @ {question_set_name}", force_no_thread=True)

                rating_df += [
                    {
                        "exam": question_set_name,
                        "exam_id": question_set_location,
                        "question": all_tags[i][0].question,
                        "question_id": all_tags[i][0].question_id,
                        "tags": all_tags[i][1],
                        "difficulty": all_difficulties[i][1],
                        "categories": all_categories_v2[i][1],
                        "model": model
                    } for i in range(len(all_tags))
                ]

            rating_df = pandas.DataFrame(rating_df)
            rating_df.to_json(os.path.join(output_folder, "rating_df.json"), index=False, orient="records")

    # =========== Do the question answering ===========
    for question_set_name, question_set_location in question_sets:
        df_raw = pandas.read_json(f"downloads\\{question_set_location}\\question_set.json")
        df_raw.sort_values(by="question", inplace=True)
        question_records: List[Question] = Question.from_df(df_raw)

        last_print = time.time()


        last_iter_time = time.time()
        for reasoning_effort in ["medium", None]:
            for model in models:
                legacy_save = True

                # determine if we are using the new save approach
                model_key = f"{model}-{reasoning_effort}"
                if model in nd_json_savers:
                    gpt_5_saver = nd_json_savers[model_key]
                    legacy_save = False

                for repeat in [1, 2, 3, 4, 5]:
                    for temperature in [0.2, None]:

                        if reasoning_effort is not None and (temperature is not None or model not in models_with_reasoning):
                            # This is not a valid condition
                            continue

                        for working in [True, False]:
                            base_key = {
                                "model": model,
                                "repeat": str(repeat),
                                "temp": str(temperature),
                                "working": str(working),
                                "question_set_name": question_set_name
                            }

                            if reasoning_effort is not None:
                                base_key = {
                                    **base_key,
                                    "reasoning_effort": reasoning_effort
                                }

                            print(f"Executing @ {base_key}")

                            last_iter_duration = time.time() - last_iter_time
                            print(f"Iter took {last_iter_duration:.2f}s")
                            last_iter_time = time.time()

                            rc = RequestCache(cache_key=f"repeat={repeat}")
                            cb = ChatGPTInteractor(model, cache=rc, temperature=temperature, reasoning_effort=reasoning_effort)
                            qa = QuestionAnswerer(chat_bot=cb, show_working=working)

                            skipped_count = 0
                            qr = []

                            if legacy_save:
                                # we need to re-run everything
                                qr = question_records
                            else:
                                # we can just re-run what we need to
                                qr_unfiltered = question_records

                                for question in qr_unfiltered:
                                    question_id = question.question_id

                                    row_key = {
                                        **base_key,
                                        "question_id": str(question_id)
                                    }

                                    if gpt_5_saver.row_exists(row_key):
                                        skipped_count += 1
                                        continue

                                    qr.append(question)

                            print(f"Skipped {skipped_count:,} already saved questions - have {len(qr):,} to execute...")

                            all_answers: List[Tuple[Question, QuestionAnswer]] = run_threaded(qr, lambda question_record: evaluate_question_answering(question_record, qa), f"[{question_set_name=} {model=} {repeat=} {working=} {temperature=}]")

                            correct_count = 0
                            for question, result in all_answers:
                                question_id = question.question_id

                                row_key = {
                                    **base_key,
                                    "question_id": str(question_id)
                                }

                                d = {
                                    "exam": question_set_name,
                                    "exam_id": question_set_location,
                                    "question_id": question.question_id,
                                    "question": question.full_question,
                                    "prompt": result.prompt,
                                    "answer": question.formatted_answer,
                                    "given_answer": result.extracted_answer,
                                    "full_answer": result.given_answer,
                                    "valid": "yes" if result.is_valid else "no",
                                    "correct": "yes" if result.correct else "no",
                                    "model": model,
                                    "repeat": repeat,
                                    "temperature": 0.7 if temperature is None else temperature,
                                    "working": "yes" if working else "no"
                                }

                                if legacy_save:
                                    question_answer_df.append(d)
                                else:
                                    gpt_5_saver.add_row(
                                        row_key, d
                                    )

                                correct_count += 1 if result.correct else 0

                            print("=" * 50)
                            print(f"[{question_set_name=} {model=} {repeat=} {working=} {temperature=}] Correct = {correct_count:,} / {len(question_records):,}, percent = {correct_count / len(question_records) * 100:.2f}%")

    global_q_df_export = pd.DataFrame(question_answer_df)
    global_q_df_export.to_csv(os.path.join(output_folder, f"qa_df_all.csv"), index=False)
