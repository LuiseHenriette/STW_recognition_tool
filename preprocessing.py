#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Thesis: S(peech)T(hought)W(riting)R(epresentation) recognition
author: Luise Schricker

python-3.4

nltk-3.2.5
spacy-2.0.12

This file contains methods to preprocess data for the task of extracting STWR.
"""
import os

from nltk import sent_tokenize
import spacy

# Only load spacy once
NLP = spacy.load('de_core_news_sm')
# Possible quotation marks in their UTF-8 encoding, paired as possible opening-closing quote pairs
QUOTATION_MARKS = {
    '\u0022': '\u0022',
    '\u0027': '\u0027',
    '\u201E': '\u201C',
    '\u201A': '\u2018',
    '\u00AB': '\u00BB',
    '\u201C': '\u201D',
    '\u2018': '\u2019',
    '\u00BB': '\u00AB',
    '\u2039': '\u203A',
    '\u203A': '\u2039'
}

def segment_tokenize(text, mode='spacy'):
    """
    Method for tokenizing a given text into segments according to the method described in
    (Annelen Brunner. Automatische Erkennung von Redewiedergabe: ein Beitrag zur quantitativen Narratologie. Vol. 47. Walter de Gruyter, 2015.)

    :param text: The text to be tokenized.
    :param mode: One of 'text', 'spacy'; specifies the type of tokens in the returned list of lists (see below).
    :return: The tokenized text as a list of lists containing spacy annotated tokens or text tokens.
    """
    print("Tokenizing text into segments...")
    segments = []

    # Spacy has a max_length attribute, this might have to be increased
    if len(text) > NLP.max_length:
        NLP.max_length = len(text) + 1

    doc = NLP(text)
    # Exchange quotation marks for special tokens: #OPEN_QUOTE#, #CLOSE_QUOTE#
    doc = annotate_quotes(doc)

    # Use spacys sentence tokenizer
    sents = doc.sents

    # Spacy sentence tokenizer fails for quotation marks -> rectify such cases
    merged = []
    skip = False
    list_sents = [[token for token in sent] for sent in sents]

    for idx, sent in enumerate(list_sents):

        if sent[-1].tag_ == '#OPEN_QUOTE#':

            if idx < len(list_sents)-1:
                merged.append(sent + list_sents[idx+1])
                skip = True

        elif (len(sent) <= 2 and sent[-1].text == "\n") or (len(sent) == 1 and sent[0].tag_ in ['$(', '$.', '#CLOSE_QUOTE#']) :
            merged[-1] = merged[-1] + sent

        elif not skip:
            merged.append(list(sent))

        else:
            skip = False

    list_sents = merged

    cur_segment = []
    has_verb = False
    segment_after = [',', ';', ':']
    segment_after_tag = ['#CLOSE_QUOTE#']
    segment_before = ['und']
    for sent in list_sents:
        # Look for potential segmentation points in the segments:
        # Komma, semicolon, colon, closing quotation marks and the conjunction 'und'
        for token in sent:
            if token.text in segment_after or token.tag_ in segment_after_tag:
                cur_segment.append(token)
                if has_verb:
                    segments.append(cur_segment)

                    # Reset
                    cur_segment = []
                    has_verb = False
                    continue

            elif len(cur_segment) > 0 and cur_segment[-1].text in segment_before:
                if has_verb:
                    segments.append(cur_segment)

                    # Reset
                    cur_segment = []
                    has_verb = False
                cur_segment.append(token)

            else:
                cur_segment.append(token)
                # spacy uses TIGER treebank pos tags
                if token.pos_ in ['VERB', 'AUX']:
                    has_verb = True

        # If only newline is left, append this to the former segment
        if len(cur_segment) == 1 and cur_segment[0].text == "\n" and len(segments) > 0:
            segments[-1] = segments[-1] + cur_segment

        else:
            # End of sentence means end of segment
            if len(cur_segment) > 0:
                segments.append(cur_segment)

        cur_segment = []
        has_verb = False

    if mode == 'text':
        # get original text spans
        segments = [text[seg[0].idx:(seg[-1].idx + len(seg[-1].text))] for seg in segments]

    print("Done.\n")
    return segments


def annotate_quotes(doc):
    """
    Method for annotating quotation marks by exchanging the respective pos tags for special tokens:
    #OPEN_QUOTE#, #CLOSE_QUOTE#.
    CAUTION: this method can't deal with some special cases, e.g. nested quotes with the same opening and
    closing quote characters.

    :param doc: A spacy annotated object containing the text.
    :return: The spacy object with annotations in the .tag_ attribute.
    """

    opening_quotes = QUOTATION_MARKS.keys()
    closing_quotes = QUOTATION_MARKS.values()

    # stack to keep track of opening quotes
    opening_quotes_found = []
    quotes_found = []

    for i,token in enumerate(doc):
        is_closing_quote = False

        # First, check for possible closing quote (case of same character for opening and closing quotes)
        if token.text in closing_quotes and len(opening_quotes_found) > 0:

            # Check if a matching opening quote is at the top of the stack (ignoring apostrophes as these could be
            # erroneously identified as quotation marks)
            top = opening_quotes_found[-1]
            # Ignore apostrophes if no match
            if token.text != '\u0027' and top[0] == '\u0027' and len(opening_quotes_found) > 1:
                for j in range(len(opening_quotes_found)-2, -1, -1):
                    top_alt = opening_quotes_found[j]
                    if QUOTATION_MARKS[top_alt[0]] == token.text:
                        top = top_alt
                        # Delete apostrophes
                        opening_quotes_found = opening_quotes_found[:j+1]
                    if top_alt[0] != '\u0027':
                        break

            if QUOTATION_MARKS[top[0]] == token.text:
                # Again special attention to apostrophes:
                # closing quotes are usually preceded by sentence ending punctuation
                if top[0] == '\u0027' and doc.__getitem__(i-1).tag_ != '$.':
                    continue

                # Add found quotation
                quotes_found.append((top[1], i))
                # Delete opening quote from stack
                opening_quotes_found = opening_quotes_found[:-1]
                is_closing_quote = True

        if token.text in opening_quotes and not is_closing_quote:
            # Special treatment of apostrophe characters - these can be used in other functions than marking quotations
            # and are therefore treated with stricter rules.
            if token.text == '\u0027':
                # Opening quotes are usually followed by capital letters (except continuing quotations, these are ignored
                # in the case of apostrophe characters for marking quotation)
                if i < len(doc)-1 and doc.__getitem__(i+1).text[0].isupper():
                    opening_quotes_found.append((token.text, i))
            else:
                opening_quotes_found.append((token.text, i))

    # Replace found quote pairs in text
    for quote in quotes_found:
        doc.__getitem__(quote[0]).tag_ = "#OPEN_QUOTE#"
        doc.__getitem__(quote[1]).tag_ = "#CLOSE_QUOTE#"

    return doc

def normalize_kolimo(path):
    """
    Method to normalize the characters in the kolimo corpus and write everything into one document.
    This is needed as preprocessing step for training word vectors on kolimo.

    :param path: The path to the corpus directory.
    """
    # Iterate over files in directory
    with open("/".join([path, "corpus/preprocessed.corpus"]), 'w') as output:
        path = path + "/fulltext"
        for dir in os.listdir(path):
            if dir != ".DS_Store":
                for filename in os.listdir(path + "/{}".format(dir)):
                    if filename != ".DS_Store":
                        with open(os.path.join(path, dir, filename), "r", encoding='utf-8') as cur_file:
                            print("Normalizing {}...".format(filename))
                            text = []
                            for line in cur_file:
                                # Replace special characters: ſ, aͤ, oͤ, uͤ, ꝛ, æ, ¬
                                line = line.replace("ſ", "s")
                                line = line.replace("aͤ", "ä")
                                line = line.replace("oͤ", "ö")
                                line = line.replace("uͤ", "ü")
                                line = line.replace("ꝛ", "r")
                                line = line.replace("æ", "ä")
                                line = line.replace("¬", "-")
                                # Make sure punctuation is followed by a whitespace, otherwise the sentence tokenizer does not work properly
                                line = line.replace(".", ". ")
                                line = line.replace(";", "; ")
                                line = line.replace("!", "! ")
                                line = line.replace("?", "? ")
                                # Exceptions
                                line = line.replace("Z. B.", "Z.B.")
                                line = line.replace("z. B.", "z.B.")
                                line = line.replace("S. ", "S.")
                                line = line.rstrip()
                                # Remove quotation marks
                                line = line.replace("\"", "")
                                line = line.replace("„", "")
                                line = line.replace("“", "")
                                line = line.replace("‚", "")
                                line = line.replace("‘", "")
                                line = line.replace("»", "")
                                line = line.replace("«", "")
                                text.append(line)
                            # Eliminate end-of-line hypenation
                            text_new = []
                            skip = False
                            for i, line in enumerate(text):
                                if skip == True:
                                    skip = False
                                elif line.endswith("-") and not line.endswith(" -") and i < (len(text)-1):
                                    text_new.append("".join([line[:-1], text[i+1]]))
                                    skip = True
                                else:
                                    text_new.append(line)
                            text_new = " ".join(text_new)

                            # Spacy has a max_length attribute, this might have to be increased
                            if len(text_new) > NLP.max_length:
                                NLP.max_length = len(text_new)

                            # Tokenize text into sentences and write sentence by sentence to current active file.
                            # Use NLTK as it is faster than spacy for simple tokenization without further analysis
                            sentences = sent_tokenize(text_new, language='german')

                            for sent in sentences:
                                output.write("".join([sent, "\n"]))

    print("\nNormalized data written to {}.".format("/".join([path, "corpus/preprocessed.corpus"])))
    return


# Execution
if __name__ == "__main__":
    test = "Belinde\nBelinde\nBelinde stand in ihrem einsamen Gemach, und schaute hinunter in den blühenden Garten. Lebe wohl, sagte sie mit Thränen, du Schauplatz meines Glücks, lebt wohl, ihr rauschenden Bäume, die ihr mich und den Geliebten oft in eurem dunkeln Schatten vor den spähenden Augen verbarget!"
    print([tok.tag_  for ls in segment_tokenize(test, mode='spacy')for tok in ls])
