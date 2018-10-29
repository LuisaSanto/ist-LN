#!/usr/bin/env python3
# Grupo 12 | Ruben Anagua, 78050 | Ana Luisa Santo, 79758

import argparse
from copy import deepcopy
import re

import nltk
from nltk.util import ngrams


def get_sentences(file_path):
    with open(file_path, "r") as input_sents:
        return [line.strip() for line in input_sents.readlines() if len(line.strip()) > 0]


def get_lemmas(file_path):
    with open(file_path, "r") as input_params:
        ambig_word = input_params.readline().strip()
        opts = input_params.readline().strip().split()
        return ambig_word, opts


def get_probability_dicts(file_path):
    def get_number_of_ngrams():
        data_start_index = content.index("\\data\\")
        data_end_index = content.index("\n\n", data_start_index)
        data_content = content[data_start_index:data_end_index]

        try:
            n1grams = int(re.search(r"ngram 1=(\d+)", data_content).group(1))
            n2grams = int(re.search(r"ngram 2=(\d+)", data_content).group(1))
            print("Unigrams: {} | Bigrams: {}".format(n1grams, n2grams))

            return n1grams, n2grams
        except (AttributeError, ValueError):
            raise RuntimeError("Input ARPA file is incorrectly formatted")

    def get_unigram_probability_dict():
        unigram_start_index = content.index("\\1-grams:")
        unigram_end_index = content.index("\n\n", unigram_start_index)
        unigram_content = content[unigram_start_index:unigram_end_index]

        lines = unigram_content.split("\n")[1:]
        probability_dict = {}

        for line in lines:
            split_line = re.split("[ \t]", line.strip())
            probability_dict[split_line[1]] = 10 ** float(split_line[0])

        return probability_dict

    def get_bigram_probability_dict():
        bigram_start_index = content.index("\\2-grams:")
        bigram_end_index = content.index("\n\n", bigram_start_index)
        bigram_content = content[bigram_start_index:bigram_end_index]

        lines = bigram_content.split("\n")[1:]
        probability_dict = {}

        for line in lines:
            split_line = re.split("[ \t]", line.strip())
            probability_dict[tuple(split_line[1:3])] = 10 ** float(split_line[0])

        return probability_dict

    # Function starts here

    with open(file_path, "r") as input_arpa:
        content = input_arpa.read()

        n_unigrams, n_bigrams = get_number_of_ngrams()
        unigram_pdict = get_unigram_probability_dict()
        bigram_pdict = get_bigram_probability_dict()

        if len(unigram_pdict) != n_unigrams or len(bigram_pdict) != n_bigrams:
            raise RuntimeError("Input ARPA file is incorrectly formatted")

        return unigram_pdict, bigram_pdict


def process_sentence(sentence, ambig_word, opts, unigram_probs, bigram_probs, laplace_alpha=0.1):
    def normalize_value(original_value, addition_value, len_vocabulary, total_prob):
        # Applies a variant of the Laplace smoothing strategy
        return total_prob * (original_value + addition_value) / (addition_value * len_vocabulary + total_prob)

    def get_temporary_smoothed_probabilities(new_bigrams):
        # Add new bigrams with prior of 0 to copied dict, compute length of vocabulary
        tmp_bigram_probs = deepcopy(bigram_probs)
        new_bigram_probs = {b: 0 for b in new_bigrams if b not in bigram_probs}
        tmp_bigram_probs.update(new_bigram_probs)
        num_all_bigrams = len(bigram_probs) + len(new_bigram_probs)

        # Compute the probability to add to prior -> laplace_alpha * lowest recorded probability
        # This is (usually) equivalent to assigning a relative frequency of laplace_alpha for new, unknown bigrams
        addition = bigram_probs[min(bigram_probs, key=bigram_probs.get)] * laplace_alpha

        # Add previous variable, maintaining total probability value
        total_probability = sum(tmp_bigram_probs.values())
        tmp_bigram_probs = {b: normalize_value(v, addition, num_all_bigrams, total_probability)
                            for b, v in tmp_bigram_probs.items()}

        return tmp_bigram_probs

    def get_sentence_probability(bigrams, bigram_probs_dict):
        p = 1
        for b in bigrams:
            p *= bigram_probs_dict[b]
        return p

    def print_processed_info(lemma_dict):
        max_bigram_probability = lemma_dict[max(lemma_dict, key=lemma_dict.get)]
        best_lemmas = [l for l, v in lemma_dict.items() if v == max_bigram_probability]  # lemmas with max probability
        best_lemma = max(best_lemmas, key=unigram_dict.get)  # choose the most common lemma if multiple have max value

        print("----------\nSentence: " + sentence)
        print("Suggested lemma: '{}'".format(best_lemma))
        for l in lemmas:
            print("- Probability with lemma '{}': {}".format(l, lemma_dict[l]))

    # Function starts here

    lemma_probability_dict = {}
    for lemma in opts:
        # Replace ambiguous word and tokenize it
        replaced_sent = "<s> " + re.sub(r"\b{}\b".format(ambig_word), lemma, sentence, re.IGNORECASE) + " </s>"
        sent_bigrams = list(ngrams(nltk.word_tokenize(replaced_sent), 2))
        bigram_probs_for_lemma = get_temporary_smoothed_probabilities(sent_bigrams)

        lemma_probability_dict[lemma] = get_sentence_probability(sent_bigrams, bigram_probs_for_lemma)

    print_processed_info(lemma_probability_dict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arpa_path", help="path to the input .arpa file")
    ap.add_argument("param_path", help="path to the file containing the superficial ambiguous"
                                       " form and the respective lemmas")
    ap.add_argument("test_sents_path", help="path to the text file containing test sentences")
    args = ap.parse_args()

    unigram_dict, bigram_dict = get_probability_dicts(args.arpa_path)

    sentences = get_sentences(args.test_sents_path)
    ambiguous_word, lemmas = get_lemmas(args.param_path)

    for s in sentences:
        process_sentence(s, ambiguous_word, lemmas, unigram_dict, bigram_dict, laplace_alpha=0.1)
