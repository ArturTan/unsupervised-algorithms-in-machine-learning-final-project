import logging
import multiprocessing
import os
import pickle
from dataclasses import dataclass
from glob import glob
from itertools import repeat
from typing import List

import pandas as pd
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
from nltk import SnowballStemmer

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def load_to_spacy_span(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def collect_info(filename, type_info: str):
    data = load_to_spacy_span(filename)
    data = [
        [
            tuple((word.text, word.__getattribute__(type_info)) for word in sentence[0])
            for sentence in sequence
        ]
        for sequence in data
    ]
    return data


def collect_info_about_words(info_type, destination_folder):
    NUM_PROCESSES = 2
    files = sorted(
        glob(f"{destination_folder}/*"),
        key=lambda x: int(x[len(os.path.join(destination_folder, "spacy_info_")) :]),
    )
    info_list = []
    pool = multiprocessing.Pool(NUM_PROCESSES)
    for file_batch in range(0, len(files), NUM_PROCESSES):
        results = pool.starmap(
            collect_info,
            zip(files[file_batch : file_batch + NUM_PROCESSES], repeat(info_type)),
        )
        for result in results:
            info_list.extend(result)
    pool.close()
    return info_list


@dataclass
class Word:
    text: str
    pos: str
    lemma: str
    stem: str
    ner: str


class SpacyAnalyzer:
    def __init__(self, path_to_data: str, chunk_size: int, destination_folder: str):
        self.path_to_data = path_to_data
        self.chunk_size = chunk_size
        self.number_of_chunks = self.calculate_chunk_number()
        self.destination_folder = destination_folder
        os.makedirs(self.destination_folder, exist_ok=True)

    def calculate_chunk_number(self):
        with open(self.path_to_data, "r") as file:
            number_of_chunks = sum(1 for row in file) // self.chunk_size
        return number_of_chunks

    def save_to_spacy_span(self, num: str, data: List):
        with open(f"{self.destination_folder}/spacy_info_{num}", "wb") as f:
            pickle.dump(data, f)

    def extract_spacy_spans(self):
        stemmer = SnowballStemmer(language="english")
        train_data = pd.read_csv(
            self.path_to_data, chunksize=self.chunk_size, index_col=0
        )
        logger.info("Spacy analysis started")
        for chunk_no in tqdm(range(self.number_of_chunks + 1)):
            chunk = next(train_data)
            word_infos_all = []
            for seq in nlp.pipe(chunk["reviewText"], n_process=-1):
                word_infos_all.append(
                    [
                        (
                            [
                                Word(
                                    text=i.text,
                                    lemma=i.lemma_,
                                    pos=i.pos_,
                                    ner=i.ent_type_,
                                    stem=stemmer.stem(i.text),
                                )
                                for i in sent
                            ],
                            sent.text,
                        )
                        for sent in seq.sents
                    ]
                )
            self.save_to_spacy_span(chunk_no, word_infos_all)
        logger.info("Spacy analysis ended")

    def collect_info_about_words(self, info_type):
        return collect_info_about_words(info_type, self.destination_folder)
