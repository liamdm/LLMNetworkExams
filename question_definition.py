from dataclasses import dataclass
from typing import List
import pandas as pd

class OptionKeys:
    values = "ABCDEFHIJ"

    @staticmethod
    def value(index:int):
        return OptionKeys.values[index]

@dataclass(frozen=True, slots=True)
class Question:
    question: str
    possible_options: List[str]
    exhibits: List[str]
    correct_indices: List[int]
    question_id:int

    @staticmethod
    def answer_only(answer:List[str]):
        return Question("", answer + ["XXXXXX1", "XXXXXX2", "XXXXXX3"], [], list(range(len(answer))), 42069)
    def options_subset(self, indices:List[int]=None):
        return "\n".join(f"{OptionKeys.values[i]}) {value}" for (i, value) in enumerate(self.possible_options) if (indices is None) or (i in indices))

    @property
    def formatted_answer(self):
        return self.options_subset(self.correct_indices)
    @property
    def formatted_options(self):
        return self.options_subset()

    @property
    def full_question_without_options(self):
        s = f"{self.question}\n"
        #s += f"Options:\n"
        #s += self.formatted_options
        if len(self.exhibits) > 0:
            s += "Exhibits:\n"
            s += "\n".join(self.exhibits)
        return s

    @property
    def full_question(self):
        s = f"{self.question}\n"
        s += f"Options:\n"
        s += self.formatted_options
        if len(self.exhibits) > 0:
            s += "Exhibits:\n"
            s += "\n".join(self.exhibits)
        return s

    @staticmethod
    def from_df(df:pd.DataFrame):
        all_q = []
        for _, row in df.iterrows():
            all_q.append(Question(
                question=row["question"],
                possible_options=row["possible_options"],
                exhibits=row["exhibits"],
                correct_indices=row["correct_indices"],
                question_id=row["question_id"]
            ))
        return all_q