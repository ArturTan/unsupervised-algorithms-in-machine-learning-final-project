import re
from collections import Counter

import pkg_resources
from symspellpy import SymSpell, Verbosity

from constant import NON_ALPHA_CHARACTERS_REGEX, WN_LEMMAS

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)


def check_is_in_lemmas(word):
    return word in WN_LEMMAS or word.lower() in WN_LEMMAS


def remove_non_letters(word):
    return NON_ALPHA_CHARACTERS_REGEX.sub("", word)


corrected_words = {}


def correct_word_with_symspell(word):
    if len(word) > 10:
        distance = 2
    else:
        distance = 1
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=distance)
    if suggestions:
        return suggestions[0]._term
    return word


def return_words(text):
    return re.findall("[a-z]+", text.lower())


VITERBI_DICTIONARY = Counter(return_words(open("../data/big.txt").read()))
MAX_WORD_LENGTH = max(map(len, VITERBI_DICTIONARY))
TOTAL = float(sum(VITERBI_DICTIONARY.values()))


def viterbi_segment(text):
    # https://stackoverflow.com/a/481773/6799297
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max(
            (probs[j] * word_prob(text[j:i]), j)
            for j in range(max(0, i - MAX_WORD_LENGTH), i)
        )
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i] : i])
        i = lasts[i]
    words.reverse()
    return words, probs[-1]


def word_prob(word):
    return VITERBI_DICTIONARY[word] / TOTAL


def check_if_there_are_one_letters_sequences(sequence, word):
    return "".join([i for i in sequence if len(i) > 1]) == word


def replace_with_final_forms(word, correct_word_map):
    word = re.sub(
        r"\b(\w+)\b", lambda m: correct_word_map.get(m.group(1), m.group(1), s)
    )
    return word
