import logging
import pickle
from dataclasses import dataclass

import nltk
from flashtext import KeywordProcessor
from nltk.corpus import stopwords
from tqdm import tqdm

from data_preprocessors.processing_functions import (
    check_if_there_are_one_letters_sequences, check_is_in_lemmas,
    correct_word_with_symspell, remove_non_letters, viterbi_segment)

nltk.download("stopwords")
stop_words = list(stopwords.words("english"))

keyword_processor = KeywordProcessor()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


@dataclass
class WordProcessorHistory:
    initial_word: str
    preprocessed_word = None
    final_form = None


class CorrectionHeuristicsExecutor:
    def __init__(self, lemmas, experiment_name="default"):
        self.words_to_process = [
            WordProcessorHistory(initial_word=word) for word in lemmas
        ]
        self.experiment_name = experiment_name

    def preprocessing_step(self, word: WordProcessorHistory, function):
        if word.final_form is None:
            try:
                word.preprocessed_word = function(word.preprocessed_word)
            except TypeError:
                word.preprocessed_word = function(word.initial_word)
            if check_is_in_lemmas(word.preprocessed_word):
                word.final_form = word.preprocessed_word

    def preprocess_word_for_viterbi_segment(self, word: WordProcessorHistory):
        if word.final_form is None:
            word_to_process = (
                word.preprocessed_word if word.preprocessed_word else word.initial_word
            )
            segments = viterbi_segment(word_to_process)[0]
            if check_if_there_are_one_letters_sequences(segments, word_to_process):
                word.final_form = segments
            else:
                segments = viterbi_segment(word_to_process.lower())[0]
                if check_if_there_are_one_letters_sequences(
                    segments, word_to_process.lower()
                ):
                    word.final_form = segments

    def preprocess(self, data):
        for word in self.words_to_process:
            if check_is_in_lemmas(word.initial_word):
                word.final_form = word.initial_word
            self.preprocessing_step(word, remove_non_letters)
            self.preprocessing_step(word, correct_word_with_symspell)
            self.preprocess_word_for_viterbi_segment(word)
        logger.info("Words will be corrected into known words or removed")

        return self.apply_final_forms(data)

    def apply_final_forms(self, data):
        correct_word_map = {
            word.initial_word: word.final_form
            for word in self.words_to_process
            if word.final_form is not None
            and (len(word.initial_word) > 1 and word.initial_word not in stop_words)
        }

        for key in correct_word_map:
            if isinstance(correct_word_map[key], list):
                correct_word_map[key] = " ".join(correct_word_map[key])

        correct_word_map_to_none = {
            word.initial_word: ""
            for word in self.words_to_process
            if word.final_form is None
            and len(word.initial_word) > 1
            and word.initial_word not in stop_words
        }
        with open(f"{self.experiment_name}_excluded_words.pkl", "wb") as f:
            pickle.dump(correct_word_map_to_none, f)

        with open(f"{self.experiment_name}_corrected_words.pkl", "wb") as f:
            pickle.dump(correct_word_map, f)

        for word, target_word in correct_word_map_to_none.items():
            keyword_processor.add_keyword(word, target_word)
        for word, target_word in correct_word_map.items():
            keyword_processor.add_keyword(word, target_word)

        texts = []
        for i in tqdm(data.reviewText):
            texts.append(keyword_processor.replace_keywords(i))
        logger.info("Pipeline ended")
        return texts
