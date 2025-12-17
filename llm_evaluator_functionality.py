import dataclasses
import itertools
from typing import List, Callable, Tuple, Any

from ccst_category_extractor import CategoryExtractor
from model_interface import ChatBotInterface, Message, MessageType
from question_definition import Question


@dataclasses.dataclass
class QuestionAnswer:
    is_valid: bool
    correct: bool
    given_answer: str
    extracted_answer: str
    prompt: str

class QuestionAnswerer:
    def __init__(self, chat_bot: ChatBotInterface, show_working:bool=False, clarify:str=None):
        self.chat_bot = chat_bot
        self.clarify_str = ""
        if clarify is not None:
            self.clarify_str = f"The following is an {clarify} question.\n"
        self.question_template = f"$$question$$\nRespond only with the options provided." if not show_working else "$$question$$\nProvide a chain of thought. Provide your final answer following any working enclosed in <answer>...</answer> tags verbatim from the options provided."
        self.working = show_working

        if hasattr(chat_bot, "is_local") and chat_bot.is_local:
            self.question_template += " Stop answering after the closing </answer>"

    def evaluate(self, question:Question) -> QuestionAnswer:
        question_str = question.full_question

        prompt = self.clarify_str + self.question_template.replace(f"$$question$$", question_str)

        given_answer = self.chat_bot.ask([
            Message(MessageType.User, prompt)
        ])

        extracted_answer, is_valid, is_correct = score_question(given_answer, question)
        #print(given_answer)

        return QuestionAnswer(
            is_valid,
            is_correct,
            given_answer,
            extracted_answer,
            prompt
        )


def evaluate_question_answering(row: Question, qa:QuestionAnswerer):
    return qa.evaluate(row)

def tag_question(row: Question, ci:ChatBotInterface, categories:List[str], allow_expand:bool=True):
    categories = ", ".join(categories)
    prompt = f"Read and understand the following question from a networking exam:\n```\n"
    prompt += row.full_question
    prompt += "\n```\nThe following is a list of tags that may be applicable to networking exam questions:\n"
    prompt += categories
    if allow_expand:
        prompt += "\nOutput a comma separated list of fitting tags that actually apply to this question. New tags can be defined as required."
    else:
        prompt += "\nOutput a comma separated list of the fitting tags that actually apply to this question."

    result = ci.ask_raw(prompt)
    return [v.strip() for v in result.split(",")]

def categorise_question_v2(row: Question, ci:ChatBotInterface):
    rc = CategoryExtractor.raw_categories
    rc = [v.split(",") for v in rc]
    rc = list(sorted([v for v in list(set(list(itertools.chain(*[[p.strip() for p in v] for v in rc])))) if v not in ["<skip>"]]))
    new_categories = tag_question(row, ci, rc, allow_expand=False)
    return new_categories


def rate_difficulty(row: Question, ci:ChatBotInterface):
    prompt = f"Read and understand the following question from a networking exam with questions ranging from expert to beginner difficulty:\n```\n"
    prompt += row.full_question
    prompt += "\n```\nThe question being multiple choice has no impact on its difficulty. Rate the difficulty of this question as Beginner, Intermediate, Advanced, Expert. Provide only Beginner, Intermediate, Advanced, Expert as a response."
    result = ci.ask_raw(prompt)
    return result

