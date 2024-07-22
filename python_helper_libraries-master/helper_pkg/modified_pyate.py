
from typing import List
import math
from collections import defaultdict
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import pandas as pd
import numpy as np
spacy.prefer_gpu()


class Modified_PYATE:
    nlp = spacy.load("en_core_web_lg", parser=False, entity=False)
    matcher = Matcher(nlp.vocab)
    MAX_TERM_LENGTH = 4

    term_counter = defaultdict(int)
    noun, adj, prep = (
        {"POS": "NOUN", "IS_PUNCT": False},
        {"POS": "ADJ", "IS_PUNCT": False},
        {"POS": "DET", "IS_PUNCT": False},
    )

    patterns = [
        [adj],
        [{"POS": {"IN": ["ADJ", "NOUN"]}, "OP": "*", "IS_PUNCT": False}, noun],
        [
            {"POS": {"IN": ["ADJ", "NOUN"]}, "OP": "*", "IS_PUNCT": False},
            noun,
            prep,
            {"POS": {"IN": ["ADJ", "NOUN"]}, "OP": "*", "IS_PUNCT": False},
            noun,
        ],
    ]

    for i, pattern in enumerate(patterns):
        matcher.add(
            "term{}".format(i), lambda matcher, doc, i, matches: Modified_PYATE.add_to_counter(matcher, doc, i, matches, Modified_PYATE.term_counter), pattern)

    @staticmethod
    def add_to_counter(matcher, doc, i, matches, term_counter):
        match_id, start, end = matches[i]
        candidate = str(doc[start:end])
        if (
            Modified_PYATE.word_length(candidate)
            <= Modified_PYATE.MAX_TERM_LENGTH
        ):
            #         global term_counter
            term_counter[candidate] += 1

    @staticmethod
    def word_length(string: str):
        return string.count(" ") + 1

    @staticmethod
    def count_terms_from_document(document: str, vocab=None):
        # for single documents
        Modified_PYATE.term_counter = defaultdict(int)

        if vocab is None:

            doc = Modified_PYATE.nlp(
                document.lower(), disable=["parser", "ner"])
            matches = Modified_PYATE.matcher(doc)
            del doc

        return Modified_PYATE.term_counter

    @staticmethod
    def helper_get_subsequences(s: str) -> List[str]:
        sequence = s.split()
        if len(sequence) <= 2:
            return []
        answer = []
        for left in range(len(sequence) + 1):
            for right in range(left + 1, len(sequence) + 1):
                if left == 0 and right == len(sequence):
                    continue
                answer.append(" ".join(sequence[left:right]))
        return answer

    @staticmethod
    def modified_combo_basic(string: str, verbose=False, weights=None, have_single_word=False):
        technical_counts = pd.Series(
            Modified_PYATE.count_terms_from_document(string)).reindex()

        order = sorted(
            list(technical_counts.keys()), key=Modified_PYATE.word_length, reverse=True
        )

        if not have_single_word:
            order = list(
                filter(lambda s: Modified_PYATE.word_length(s.strip()) > 1, order))

        technical_counts = technical_counts[order]

        if len(technical_counts) == 0:
            return pd.Series()

        df = pd.DataFrame(
            {
                "xlogx_score": technical_counts.reset_index()
                .apply(
                    lambda s: math.log(
                        Modified_PYATE.word_length(s["index"])) * s[0],
                    axis=1,
                )
                .values,
                "times_subset": 0,
                "times_superset": 0,
            },
            index=technical_counts.index,
        )

        indices = set(technical_counts.index)
        iterator = tqdm(
            technical_counts.index) if verbose else technical_counts.index

        for index in iterator:
            for substring in Modified_PYATE.helper_get_subsequences(index):
                if substring in indices:
                    df.at[substring, "times_subset"] += 1
                    df.at[index, "times_superset"] += 1

        if weights is None:
            weights = np.array([1, 0.75, 0.1])

        result = df.apply(lambda s: s.values.dot(weights), axis=1)
        result = pd.DataFrame(result)
        result.reset_index(inplace=True, drop=False)
        result['counts'] = technical_counts.reset_index()[0]
        return result
