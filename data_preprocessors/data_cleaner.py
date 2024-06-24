import html
import logging
import re

import emoji
import fasttext
import pandas as pd
import spacy
from flashtext import KeywordProcessor

from constant import (CONTRACTION_LIST, FASTTEXT_LANGUAGE_RECOGNIZER_PATH,
                      PUNCTUATION)

nlp = spacy.load("en_core_web_sm")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data
        self.unique_words = set(
            [word for text in self.data["reviewText"] for word in text.split()]
        )

    def dropna(self):
        self.data.loc[self.data.reviewText.isna(), "reviewText"] = "null"

    def convert_capitalics_to_lower_case(self):
        def convert_capitalics_to_lower_word(word):
            if word.isupper():
                return word.lower()
            if re.search("[A-Z]", word[1:]):
                return word[0] + word[1:].lower()
            if not word:
                print("===", word)
            return word

        def convert_capitalics_to_lower(sequence):
            return " ".join(
                [convert_capitalics_to_lower_word(word) for word in sequence.split()]
            )

        logger.info("Words that are uppercase are converted to lowercase")
        self.data["reviewText"] = self.data.reviewText.parallel_apply(
            convert_capitalics_to_lower
        )

    def collect_only_english_language_texts(self):
        model = fasttext.load_model(FASTTEXT_LANGUAGE_RECOGNIZER_PATH)
        self.data["languages"] = self.data.reviewText.apply(
            lambda x: model.predict(x.lower(), k=2)[0][0]
        )
        self.data = self.data[self.data["languages"] == "__label__en"].reset_index(
            drop=True
        )
        logger.info("Only English texts selected")

    def remove_html(self):
        self.data["reviewText"] = self.data.reviewText.parallel_apply(html.unescape)
        logger.info("HTML characters dropped")

    def remove_emojis(self):
        def remove_emojis_from_word(word):
            return emoji.demojize(word, delimiters=(" ", ""))

        self.data["reviewText"] = self.data.reviewText.parallel_apply(
            remove_emojis_from_word
        )
        logger.info("Emojis dropped")

    def remove_duplicate_punctuation(self):
        def duplicate_punctuation(text):
            new_text = ""
            start = 0
            for i in re.finditer(f"[{PUNCTUATION}]*([{PUNCTUATION}])", text):
                new_text += text[start : i.start() + 1]
                start = i.end()
            new_text += text[start:]
            if not new_text:
                return text
            return new_text

        self.data["reviewText"] = self.data.reviewText.parallel_apply(
            duplicate_punctuation
        )
        logger.info("Duplicate punctuation removed")

    def remove_three_or_more_consecutive_characters(self):
        pattern = "(?P<g>.)(?P=g){2}"

        def remove_three_or_more_consecutive_characters_for_one_word(text):
            while re.search(pattern, text):
                new_text = ""
                start = 0
                for i in re.finditer(pattern, text, re.M):
                    new_text += text[start : i.start() + 2]
                    start = i.end()
                text = new_text + text[start:]
            return text

        self.data["reviewText"] = self.data.reviewText.parallel_apply(
            remove_three_or_more_consecutive_characters_for_one_word
        )
        logger.info("Removed more than 3 consecutive same characters")

    def remove_duplicated_spaces(self):
        def remove_duplicated_spaces_for_one_word(text):
            while text.find("  ") != -1:
                text = text.replace("  ", " ")
            return text

        self.data["reviewText"] = self.data.reviewText.parallel_apply(
            remove_duplicated_spaces_for_one_word
        )
        logger.info("Duplicate spaces removed")

    def add_space_after_punctuation(self):
        punctuation = re.compile("[!\"#$%&'()*\+,-\.\/:;<=>?@\[\]^_`{|}~]")

        def add_space_after_punctuation_for_one_word(text):
            new_text = ""
            start = 0
            if not punctuation.search(text):
                return text
            for i in punctuation.finditer(text):
                new_text += text[start : i.end()] + " "
                start = i.end()
            new_text += text[start:]
            return new_text

        self.data["reviewText"] = self.data.reviewText.parallel_apply(
            add_space_after_punctuation_for_one_word
        )
        logger.info("Add space after punctuation")

    def convert_contractions(self):
        keyword_processor = KeywordProcessor()
        for word, target_word in CONTRACTION_LIST.items():
            keyword_processor.add_keyword(word, target_word)

        self.data["reviewText"] = self.data.reviewText.apply(
            keyword_processor.replace_keywords
        )
        logger.info("Contraction converted to two words sequences")

    def preprocess(self):
        self.dropna()
        self.convert_capitalics_to_lower_case()
        self.convert_contractions()
        # self.collect_only_english_language_texts()
        self.remove_html()
        self.remove_emojis()
        self.remove_duplicate_punctuation()
        self.remove_three_or_more_consecutive_characters()
        self.add_space_after_punctuation()
        self.remove_duplicated_spaces()
        logger.info("Cleaning process ended")
