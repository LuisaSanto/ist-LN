#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Grupo 12 | Ruben Anagua, 78050 | Ana Luisa Santo, 79758

import argparse
import re


def remove_non_verbs(token_sentence_list):
    return [pair for pair in token_sentence_list if not pair[0].startswith("n-é-verbo")]


def remove_sentences_with_multiple_occurrences(token_sentence_list, word):
    return [pair for pair in token_sentence_list
            if re.findall(r"\b{}\b".format(word), pair[1], re.IGNORECASE) is None
            or len(re.findall(r"\b{}\b".format(word), pair[1], re.IGNORECASE)) < 2]


def remove_sentences_with_errors(token_sentence_list):
    return [pair for pair in token_sentence_list if "#" not in pair[0]]


def remove_sentences_with_doubts(token_sentence_list):
    return [pair for pair in token_sentence_list if "?" not in pair[0]]


def replace_and_remove_tokens(token_sentence_list, word):
    return [re.sub(r"\b{}\b".format(word), token, sentence, re.IGNORECASE) + "\n"
            for token, sentence in token_sentence_list]


def process_list(token_sentence_list, word):
    token_sentence_list = remove_non_verbs(token_sentence_list)
    token_sentence_list = remove_sentences_with_multiple_occurrences(token_sentence_list, word)
    token_sentence_list = remove_sentences_with_errors(token_sentence_list)
    token_sentence_list = remove_sentences_with_doubts(token_sentence_list)
    return replace_and_remove_tokens(token_sentence_list, word)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("word_to_disambiguate", help="Word to disambiguate")
    ap.add_argument("out_path", help="Path to the input .out file")
    ap.add_argument("final_path", help="Path to the output .final file", default=None, nargs="?")
    args = ap.parse_args()

    if args.final_path is None:
        args.final_path = args.out_path[:-4] + ".final"

    with open(args.out_path, "r") as out:
        lines = out.readlines()

        # Get (token, sentence) pairs - ignore empty input lines
        tokens_sentences = [line.strip().split("\t") for line in lines if len(line.strip()) > 0]

        # Raise an error if any line cannot be converted to a (token, sentence) pair
        if not all([len(ts_pair) == 2 for ts_pair in tokens_sentences]):
            raise RuntimeError("Input .out file is incorrectly formatted")

        # Get the processed sentences
        sentences = process_list(tokens_sentences, args.word_to_disambiguate)

        # Write to the output .final file
        with open(args.final_path, "w") as final:
            final.writelines(sentences)
