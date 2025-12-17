from typing import Dict
import bs4
import pandas as pd
from question_definition import Question

class CategoryExtractor:
    raw_categories = pd.read_csv("categories2.csv", delimiter=";")["Base Category"].values.tolist()

    category_df = None
    ccst_to_base_category_map: Dict[str, str] = {}
    question_to_ccst_category_map: Dict[str, str] = {}

    @staticmethod
    def setup():
        CategoryExtractor.category_df = pd.read_csv("categories2.csv", delimiter=";")

        ccst_categories = CategoryExtractor.category_df["CCST Category"].values.tolist()
        ccst_mapped_to = CategoryExtractor.category_df["Base Category"].values.tolist()

        for cat, map_to in zip(ccst_categories, ccst_mapped_to):
            CategoryExtractor.ccst_to_base_category_map[cat] = map_to


        with open("downloads/mu_ccst/question_categories.html", "r") as r:
            d = r.read()

        d = bs4.BeautifulSoup(d, features="html.parser")
        element = d.find("tbody")

        current_header = None

        CategoryExtractor.question_mapping = {}

        for child in element.children:
            header = child.find("th")

            if header is None:
                question = child.find(attrs={"data-title": "Stem"})
                if question is None:
                    continue
                CategoryExtractor.question_to_ccst_category_map[question] = current_header
            else:
                current_header = header.get_text()



    @staticmethod
    def get_category_for_ccst_question(q:Question):
        for q in q.question:
            pass




if __name__ == "__main__":
    CategoryExtractor.setup()