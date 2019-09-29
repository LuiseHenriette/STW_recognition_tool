#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Thesis: S(peech)T(hought)W(riting)R(epresentation) recognition
author: Luise Schricker

python-3.4

germalemma-0.1.1
pandas-0.21.0
spacy-2.0.12

This file contains methods to postprocess STWR classifications.
"""
import re

from germalemma import GermaLemma
import pandas as pd
import spacy

from preprocessing import QUOTATION_MARKS

# Only load spacy once
NLP = spacy.load('de_core_news_sm')

def postprocess_spans(row, cl=None):
    """
    Method for better span detection as a postprocessing step after STWR classification.

    :param row: Each row consists of a label (format:"direct_speech,2,10") and a text.
    :param cl: label of the positive class instances.
    :return: The updated label
    """
    label = row.values[0]
    # Only do postprocessing for detected instances
    if label == "":
        return label

    text = row.values[1]
    doc = NLP(text)
    tokens = [token for token in doc]
    # Get lemmata with germalemma as spacy is not good at this, only possible for pos tags N, V, ADJ, ADV
    token_lemmata = []
    lemmatizer = GermaLemma()

    for token in tokens:
        if token.pos_ == "VERB":
            token_lemmata.append(lemmatizer.find_lemma(token.text, 'V'))
        elif token.pos_ == "NOUN":
            token_lemmata.append(lemmatizer.find_lemma(token.text, 'N'))
        elif token.pos_ in ["ADJ", "ADV"]:
            token_lemmata.append(lemmatizer.find_lemma(token.text, token.pos_))
        else:
            token_lemmata.append(token.text)

    # Prepare information

    only_opening_quotes = [qu for qu in QUOTATION_MARKS.keys() if qu != QUOTATION_MARKS[qu]]
    only_closing_quotes = [QUOTATION_MARKS[qu] for qu in QUOTATION_MARKS.keys() if qu != QUOTATION_MARKS[qu]]
    # Do not treat apostrophes as possible quotation marks -> too risky
    both_quotes = [qu for qu in QUOTATION_MARKS.keys() if qu == QUOTATION_MARKS[qu] and qu != '\u0027']

    # Find quotation marks that can either be an opening or a closing quote but that don't have the same form as their counter part
    both = [qu for qu in only_opening_quotes if qu in only_closing_quotes]
    only_opening_quotes = [qu for qu in only_opening_quotes if qu not in both]
    only_opening_quotes = [qu for qu in only_opening_quotes if qu not in both]
    both_quotes = both_quotes + both

    # Load reporting word list
    stw_words_all = pd.read_excel("data/stw_words/stw_words_brunner2015.xls")
    # Only use words with penalty value up tp 3
    stw_words_all = stw_words_all[stw_words_all['Penalty'] <= 3]
    # Some words are only usable for reported class
    stw_words = stw_words_all[stw_words_all['Marker'] != 'rep']

    spans = []
    if cl=='direct':

        # Search for quotation marks and try to decide whether they signify quoted STWR. Use conservative heuristics.
        for token in tokens:
            # Mark different candidates for quotation marks
            if token.text in only_opening_quotes:
                token.tag_ = "ONLY_OPENING_QUOTE"
            elif token.text in only_closing_quotes:
                token.tag_ = "ONLY_CLOSING_QUOTE"
            elif token.text in both_quotes:
                token.tag_ = "BOTH_QUOTES"

        stack = []
        for idx, token in enumerate(tokens):
            if token.tag_ == "ONLY_OPENING_QUOTE":
                stack.append((idx, token.text, token.tag_))

            elif token.tag_ in ["ONLY_CLOSING_QUOTE", "BOTH_QUOTES"]:
                # Check whether there is a matching opening quote on the stack
                found = False
                for i in range(len(stack)-1,-1,-1):
                    top = stack[i]
                    if QUOTATION_MARKS[top[1]] == token.text:
                        found = True
                        # Closing quotes are usually preceded by sentence ending punctuation
                        if tokens[idx-1].tag_ == '$.':
                            spans.append((top[0], idx))
                        stack = stack[:i]
                        break
                if not found:
                    # If no opening quotes were found and clear closing quotes are preceded by sentence ending punctuation,
                    # assume everything before is quoted
                    if token.tag_ == "ONLY_CLOSING_QUOTE" and idx > 0 and tokens[idx-1].tag_ == '$.':
                        spans.append((0, idx))
                    # If ambiguous quotation mark is found, decide whether it's opening or closing
                    elif token.tag_ == "BOTH_QUOTES":
                        if idx > 0 and tokens[idx-1].tag_ == '$.':
                            spans.append((0, idx))
                        else:
                            stack.append((idx, token.text, token.tag_))

        # Check for open quotes in the stack
        if len(stack) > 0:
            # Choose first open quote in stack
            # Opening quotes are usually followed by capital letters (except continuing quotations, these are ignored here)
            opening = stack[0]
            if opening[0] < len(tokens)-2:
                if tokens[opening[0]+1].text.istitle():
                    spans.append((opening[0], len(tokens)-1))

        # In case no quotation marks are there, look for colon
        if len(spans) == 0:
            for idx, token in enumerate(tokens):
                if ":" == token.text:
                    spans.append((idx, len(tokens)-1))

    elif cl=='indirect':

        # Following A.B.s directions for annotating indirect representations
        # (Annelen Brunner. Automatische Erkennung von Redewiedergabe: ein Beitrag zur quantitativen Narratologie. Vol. 47. Walter de Gruyter, 2015.)

        # Pattern 1: verbal framing phrase + dependent clause - assume max. one of these patterns per segment
        stw_verb_segment = [tokens[i] for i,lemma in enumerate(token_lemmata) if not lemma.istitle() and any(stw_words["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))))]
        # Only use this pattern if there is a clear candidate
        if len(stw_verb_segment) == 1:
            verb = stw_verb_segment[0]
            dependent_clause = get_children(verb, exception=['sb'])

            start = None
            end = None

            for i, token in enumerate(tokens):
                if token == verb:
                    start = i
                elif token in dependent_clause:
                    if start != None:
                        end = i

            if start != None and end != None:
                spans.append((start, end))

        # Pattern 2: nominal phrase includ. modificators + dependent clause - several of these patterns per segment are possible
        stw_noun_segment = [tokens[i] for i,lemma in enumerate(token_lemmata) if lemma.istitle() and any(stw_words["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))))]

        for noun in stw_noun_segment:
            dependent_clause_modif = get_children(noun, exception=[])
            all_tokens = dependent_clause_modif + [noun]

            start = None
            end = None

            for i, token in enumerate(tokens):
                if token in all_tokens:
                    if start == None:
                        start = i
                    else:
                        end = i

            if start != None and end != None:
                spans.append((start, end))

        # Merge spans
        merged_spans = []
        if len(spans) > 1:
            for i, span in enumerate(spans):
                for other in spans:
                    if other == span:
                        continue
                    else:
                        if span[0] >= other[0] and span[1] <= other[1]:
                            break
                        else:
                            merged_spans.append(span)

            spans = merged_spans

    elif cl == 'free_indirect':
        # Free indirect instances are almost always complete sentences -> leave as is
        pass

    elif cl == 'reported':
        # „Prinzipiell wird bei erzählter Wiedergabe angestrebt, den ganzen Satz oder Satzteil zu markieren, der eine Sprach-, Denk- oder Schreibhandlung wiedergibt.
        # – Wenn es möglich ist, mehrere unterschiedliche sprachliche, schriftliche oder gedankliche Handlungen zu identifizieren, so werden diese jeweils einzeln markiert.
        # – Wenn eine Nominalphrase mit einem Verb verwendet wird, so dass sich im Ganzen eine Sprach-, Denk- oder Schreibhandlung ergibt,
        # sollte – wie bei indirekter Wiedergabe – die ganze Verbalphrase markiert werden (also Pläne entwerfen, nicht nur Pläne).“
        # Following A.B.s directions for annotating reported representations try to annotate the whole clause for reported instances
        # (Annelen Brunner. Automatische Erkennung von Redewiedergabe: ein Beitrag zur quantitativen Narratologie. Vol. 47. Walter de Gruyter, 2015.)

        stw_segment = [tokens[i] for i, lemma in enumerate(token_lemmata) if any(
            stw_words_all["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))))]

        for word in stw_segment:
            dependent_clause = get_children(word, exception=[])
            all_tokens = dependent_clause + [word]

            start = None
            end = None

            for i, token in enumerate(tokens):
                if token in all_tokens:
                    if start == None:
                        start = i
                    else:
                        end = i

            if start != None and end != None:
                spans.append((start, end))
        # Don't merge spans as several different reported instance should be labeled separately following A.B.s directions for annotating reported representations
        # (Annelen Brunner. Automatische Erkennung von Redewiedergabe: ein Beitrag zur quantitativen Narratologie. Vol. 47. Walter de Gruyter, 2015.)

    # Get character based spans
    if len(spans) > 0:
        labels = []
        for span in spans:
            labels.append("{},{},{}".format(cl, tokens[span[0]].idx, (tokens[span[1]].idx + len(tokens[span[1]].text))))
        label = ",".join(labels)

    return label

# Helper Methods
def get_children(verb, exception=[]):
    """
    Method for collecting all children of a given spacy token.
    Exceptions can be given as dependency tags.

    :param verb: The verb which is the root of the dependent clause, expected to be a spacy token object.
    :param exception: Dependency tags that should be excluded.
    :return: a list containing the tokens of the dependent clause.
    """
    clause = []
    for child in verb.children:
        if child.dep_ in exception:
            continue
        elif child.children == []:
            clause.append(child)
        else:
            clause.append(child)
            clause += get_children(child, exception=exception)

    return clause

# Execution
if __name__ == "__main__":
    print(postprocess_spans(["reported_0_21", "Sie war vollkommen überrascht von der Nachricht des Sohnes und schickte ihn wieder weg."], cl='reported'))