#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Thesis: S(peech)T(hought)W(riting)R(epresentation) recognition
author: Luise Schricker

python-3.4

gensim-3.1.0
germalemma-0.1.1
numpy-1.13.3
pandas-0.21.0
scipy-1.0.0
spacy-2.0.12
xlrd-1.2.0

RFTagger (http://www.cis.uni-muenchen.de/~schmid/tools/RFTagger/)

This file contains methods for feature extraction for classifiers for direct, indirect, free_indirect and reported STWR.
"""
import os
import re
from subprocess import check_output, call
import sys

from gensim.models.keyedvectors import KeyedVectors
from germalemma import GermaLemma
import numpy
import pandas as pd
from scipy.spatial import distance
import spacy

from preprocessing import annotate_quotes

# Global variables

# Only load spacy once
NLP = spacy.load('de_core_news_sm')

# Redirect unwanted output to devnull
FNULL = open(os.devnull, 'w')

# Word lists
DEICTIC = ['heute', 'morgen', 'gestern', 'jetzt', 'hier']
MODAL_PART = ['ja', 'nein', 'wohl', 'schon', 'eigentlich', 'sowieso', 'eben']
CONJUNCT = ['dass', 'daß', 'ob', 'wo', 'warum', 'wann', 'wieso', 'weshalb', 'wie', 'wodurch', 'womit', 'worin', 'woraus', 'worauf', 'was']
NEG = ['nein', 'nicht', 'kein']
FACIAL = ['Gesicht', 'Mund', 'Augenbraue', 'Auge', 'Stirn', 'Lippe', 'Nase', 'Nasenflügel']
GESTURE = ['Hand', 'Arm', 'Handfläche', 'Finger', 'Schulter', 'Faust']
VOICE = ['Stimme', 'Ton', 'Tonhöhe', 'Tonfall', 'Stimmlage', 'Atem']
NE_TYPES = ['PER', 'LOC', 'ORG', 'MISC']


### Feature extractors
class SimpleBaseline(object):
    """
    Simple Baseline feature extractor that has only one feature: the character length of segments.
    """

    def __init__(self, sequence_features=False):
        # Same signature is required for Feature extractors, but there are no sequence_features.
        self.sequence_features = sequence_features

        # Number of features
        self.num_features = 1

    def transform(self, text, original_text=None, backlog=[]):
        """
        Method that transforms the given segments into their feature representation.
        Expects dataframe with column ["text"] or list of lists of spacy tokens along with the original text.

        :param text: dataframe with column ["text"] that contains the string segments or list of spacy tokens.
        :param original_text: the original text as string is passed in test mode.
        :param backlog: needed for feature extractor that use sequence features.
        :return: The transformed segments as pandas Series or list, depending on the type of 'text'
        """
        # If the backlog has not been initialized, initialize it
        if len(backlog) == 0:
            backlog = [None for i in range(10)]

        # test mode passes tokenized text
        if type(text)==list:
            transformed = [len(original_text)]

        # train mode passes Dataframe
        else:
            transformed = text["text"].apply(len)

        return transformed, backlog


class STWRFeatureExtractor(object):
    """
    Feature extractor for classifiying STWR.
    """

    def __init__(self, sequence_features=True):
        """
        :param sequence_features: If true, use the sequence features (trained on gold labels).
        """

        # Number of features
        self.num_features = 243
        # Names of features - needed for feature inspection
        self.feature_names = ["perc_pos_NNE", "perc_pos_TRUNC", "perc_pos_APPO", "perc_pos_VVPP", "perc_pos_FM",
                              "perc_pos_KOUI", "perc_pos_ITJ", "perc_pos_PTKANT", "perc_pos_$.", "perc_pos_ADJA",
                              "perc_pos_ADJD", "perc_pos_PTKNEG", "perc_pos_PWS", "perc_pos_PRF", "perc_pos_KOUS",
                              "perc_pos_PDS", "perc_pos_VMINF", "perc_pos_VVIZU", "perc_pos_PPOSS", "perc_pos_VVFIN",
                              "perc_pos_VMFIN", "perc_pos_PROAV", "perc_pos_PRELS", "perc_pos_APPR", "perc_pos_PPOSAT",
                              "perc_pos_APZR", "perc_pos_$,", "perc_pos_PIAT", "perc_pos_VMPP", "perc_pos_NE",
                              "perc_pos__SP", "perc_pos_VAPP", "perc_pos_VAIMP", "perc_pos_CARD", "perc_pos_APPRART",
                              "perc_pos_NN", "perc_pos_KOKOM", "perc_pos_PWAT", "perc_pos_PPER", "perc_pos_XY",
                              "perc_pos_ART", "perc_pos_PWAV", "perc_pos_KON", "perc_pos_PTKA", "perc_pos_VVINF",
                              "perc_pos_$(", "perc_pos_PDAT", "perc_pos_PTKZU", "perc_pos_PRELAT", "perc_pos_PIS",
                              "perc_pos_PTKVZ", "perc_pos_VAINF", "perc_pos_ADV", "perc_pos_VAFIN", "perc_pos_VVIMP",
                              "perc_pos_", "perc_pos_SCONJ", "perc_pos_SYM", "perc_pos_VERB", "perc_pos_X", "perc_pos_EOL",
                              "perc_pos_SPACE", "perc_pos_PUNCT", "perc_pos_ADJ", "perc_pos_ADP", "perc_pos_ADV",
                              "perc_pos_AUX", "perc_pos_CONJ", "perc_pos_CCONJ", "perc_pos_DET", "perc_pos_INTJ",
                              "perc_pos_NOUN", "perc_pos_NUM", "perc_pos_PART", "perc_pos_PRON", "perc_pos_PROPN",
                              "num_ents", "num_PER", "num_LOC", "num_ORG", "num_MISC", "colon", "colon_prev", "comma_end",
                              "perc_emph", "question", "open_quote", "close_quote", "in_quotes", "num_prev_in_quotes",
                              "punct_close_quote", "close_quote_comma", "perc_per1", "perc_per2", "perc_per12", "perc_per3",
                              "only_3_prev_5", "only_1_prev_5", "3_1_prev_5", "has_ind", "has_subj", "no_subj", "no_ind",
                              "has_pres", "has_past", "no_past", "no_pres", "embedded", "wuerden_inf", "wuerden",
                              "has_prep_noun_comp", "has_claus_inf_comp", "subj_cand_speaker", "num_cand_speaker",
                              "prev_subj_cand_speaker", "prev_num_cand_speaker", "has_rep_word_0", "has_rep_word_1",
                              "has_rep_word_2", "has_rep_word_3", "has_rep_word_4", "has_rep_word_5", "has_rep_word_le_1",
                              "has_rep_word_le_2", "has_rep_word_le_3", "has_rep_word_le_4", "has_rep_word_le_5", "has_rep_word_noun",
                              "has_rep_word_verb", "has_spec_rep_word_0", "has_spec_rep_word_1", "has_spec_rep_word_2",
                              "has_spec_rep_word_3", "has_spec_rep_word_4", "has_spec_rep_word_5", "has_spec_rep_word_le_1",
                              "has_spec_rep_word_le_2", "has_spec_rep_word_le_3", "has_spec_rep_word_le_4", "has_spec_rep_word_le_5",
                              "num_rep_word_0", "num_rep_word_1", "num_rep_word_2", "num_rep_word_3", "num_rep_word_4", "num_rep_word_5",
                              "num_rep_word_le_1", "num_rep_word_le_2", "num_rep_word_le_3", "num_rep_word_le_4", "num_rep_word_le_5",
                              "num_rep_word_noun", "num_rep_word_verb", "num_spec_rep_word_0", "num_spec_rep_word_1",
                              "num_spec_rep_word_2", "num_spec_rep_word_3", "num_spec_rep_word_4", "num_spec_rep_word_5",
                              "num_spec_rep_word_le_1", "num_spec_rep_word_le_2", "num_spec_rep_word_le_3", "num_spec_rep_word_le_4",
                              "num_spec_rep_word_le_5", "prev_has_rep_word_0", "prev_has_rep_word_1", "prev_has_rep_word_2",
                              "prev_has_rep_word_3", "prev_has_rep_word_4", "prev_has_rep_word_5", "prev_has_rep_word_le_1",
                              "prev_has_rep_word_le_2", "prev_has_rep_word_le_3", "prev_has_rep_word_le_4", "prev_has_rep_word_le_5",
                              "prev_has_rep_word_noun", "prev_has_rep_word_verb", "prev_has_spec_rep_word_0", "prev_has_spec_rep_word_1",
                              "prev_has_spec_rep_word_2", "prev_has_spec_rep_word_3", "prev_has_spec_rep_word_4", "prev_has_spec_rep_word_5",
                              "prev_has_spec_rep_word_le_1", "prev_has_spec_rep_word_le_2", "prev_has_spec_rep_word_le_3",
                              "prev_has_spec_rep_word_le_4", "prev_has_spec_rep_word_le_5", "prev_num_rep_word_0", "prev_num_rep_word_1",
                              "prev_num_rep_word_2", "prev_num_rep_word_3", "prev_num_rep_word_4", "prev_num_rep_word_5",
                              "prev_num_rep_word_le_1", "prev_num_rep_word_le_2", "prev_num_rep_word_le_3", "prev_num_rep_word_le_4",
                              "prev_num_rep_word_le_5", "prev_num_rep_word_noun", "prev_num_rep_word_verb", "prev_num_spec_rep_word_0",
                              "prev_num_spec_rep_word_1", "prev_num_spec_rep_word_2", "prev_num_spec_rep_word_3", "prev_num_spec_rep_word_4",
                              "prev_num_spec_rep_word_5", "prev_num_spec_rep_word_le_1", "prev_num_spec_rep_word_le_2",
                              "prev_num_spec_rep_word_le_3", "prev_num_spec_rep_word_le_4", "prev_num_spec_rep_word_le_5",
                              "max_sim", "max_sim_rep", "perc_deictic", "spec_conjunct", "perc_modal", "perc_neg",
                              "has_facial", "has_gesture", "has_voice", "repetition", "last_direct", "last_indirect", "last_free_indirect",
                              "last_reported", "last_5_direct", "last_5_indirect", "last_5_free_indirect", "last_5_reported",
                              "last_10_direct", "last_10_indirect", "last_10_free_indirect", "last_10_reported", "num_last_10_reported",
                              "len_tokens", "len_chars", "prev_len_tokens", "prev_len_chars", "sum_len_tokens", "sum_len_chars",
                              "paragraph", "prev_paragraph"]

        # Switch to turn off sequence features
        self.sequence_features = sequence_features
        if not self.sequence_features:
            self.feature_names = self.feature_names[:-21] + self.feature_names[-8:]

        # Get all possible tags
        self.tag_map = sorted(NLP.vocab.morphology.tag_map.keys())
        self.pos_map = sorted(spacy.parts_of_speech.NAMES.values())
        # Set up lemmatizer
        self.lemmatizer = GermaLemma()
        # Set up RFTagger
        call(["make"], cwd="RFTagger/src")
        # Load word vectors
        print("Loading word-vectors. This may take a while ...")
        self.wordvecs = KeyedVectors.load_word2vec_format("data/word_vecs/kolimo.model", binary=True)
        print("Done.\n")

    def transform(self, text, original_text = None, backlog=[]):
        """
        Method that transforms the given segments into their feature representation.
        Expects dataframe with column ["text"] or list of spacy tokens along with the original text or string.

        :param text: dataframe with column ["text"] that contains the string segments or list of spacy tokens.
        :param original_text: the original text as string is passed in test mode.
        :param backlog: For test mode, the backlog stores info and labels of former segments and
                        therefore has to be passed back and forth between classifier and feature extractor.
        :return: The transformed segments as pandas Dataframe or list, depending on the type of 'text'
        """

        # If the backlog has not been initialized, initialize it
        if len(backlog) == 0:
            backlog = ["" for i in range(10)] + [0 for i in range(64)]

        # If spacy tokenization and quote annotation has not been performed, do it now
        if type(text) == list:
            tokens = text

        elif type(text) == pd.DataFrame:
            # Get full text for better results in spacy parsing
            full_text = " ".join(text['text'].values)

            doc = NLP(full_text)
            # Exchange tags for quotation marks for special tokens: #OPEN_QUOTE#, #CLOSE_QUOTE#
            doc = annotate_quotes(doc)
            tokens_full_text = [token for token in doc]

        # Transform individual segments
        if type(text) == list:
            return self.transform_segment(tokens, backlog, original_text)

        else:
            output = pd.DataFrame()
            print("Extracting features...")
            for ind, row in text.iterrows():
                # print progress bar
                sys.stdout.write('\r')
                # the exact output you're looking for:
                sys.stdout.write("[%-20s] %d%%" % ('=' * round(ind/(len(text)/20)), round(ind/(len(text)/100))))
                sys.stdout.flush()

                # Get the tokens corresponding to the segment:
                tokens_text = string_tokenize(row['text'])
                tokens = tokens_full_text[:len(tokens_text)]

                # Check that this is correct
                assert tokens_text[-1] == tokens[-1].text
                tokens_full_text = tokens_full_text[len(tokens_text):]

                transformed, backlog = self.transform_segment(tokens, backlog, row['text'])
                output = output.append(pd.Series(transformed), ignore_index = True)

                # Adapt backlog: backlog stores last ten classifications in the first ten positions
                backlog[0:10] = backlog[1:10] + [row['labels_spans']]

            return output, backlog

    def transform_segment(self, tokens, backlog, original_text):
        """
        Transforms an individual segment of tokens, given the information in the backlog,
        into a feature representation.

        :param tokens: list of spacy tokens
        :param backlog: list containing information about the labels and other features of previous segments
        :param original_text: The original text as string
        :return: the feature representation and the updated backlog
        """

        # --- Preprocessing ---
        transformed = []
        token_strings = [token.text for token in tokens]
        # Get lemmata with germalemma as spacy is not good at this, only possible for pos tags N, V, ADJ, ADV
        token_lemmata = []
        for token in tokens:
            if token.pos_ == "VERB":
                token_lemmata.append(self.lemmatizer.find_lemma(token.text, 'V'))
            elif token.pos_ == "NOUN":
                token_lemmata.append(self.lemmatizer.find_lemma(token.text, 'N'))
            elif token.pos_ in ["ADJ", "ADV"]:
                token_lemmata.append(self.lemmatizer.find_lemma(token.text, token.pos_))
            else:
                token_lemmata.append(token.text)

        # Load reporting word list
        stw_words_orig = pd.read_excel("data/stw_words/stw_words_brunner2015.xls")
        # Some words are only usable for reported class
        stw_words_rep = stw_words_orig[stw_words_orig['Marker'] == 'rep']
        stw_words = stw_words_orig[stw_words_orig['Marker'] != 'rep']

        # Do deeper morphological analysis with RFTagger
        file = open("RFTagger/temp.txt", "w")
        file.write("\n".join(token_strings))
        file.close()
        morph_tagged = check_output(["src/rft-annotate", "lib/german.par", "temp.txt"], cwd="RFTagger", stderr=FNULL).decode(
            "utf-8").split("\n")
        # Split morph tags into attributes
        morph_tagged = [morph_tag.split("\t")[1].split(".") if morph_tag != "" else morph_tag for morph_tag in morph_tagged]

        # --- Pos tag features ---
        tags = [token.tag_ for token in tokens]
        pos = [token.pos_ for token in tokens]
        transformed += [(tags.count(tag)/len(tags)) if tag in tags else 0 for tag in self.tag_map]
        transformed += [(pos.count(p) / len(pos)) if p in pos else 0 for p in self.pos_map]

        # --- NE features ---
        doc = NLP(original_text)
        transformed.append(len(doc.ents))
        for ne_type in NE_TYPES:
            transformed.append(int(len([ent for ent in doc.ents if ent.label_ == ne_type]) > 0))

        # --- Special token features ---
        # Colon in this or in previous segment?
        colon_this = int(":" in token_strings)
        transformed.append(colon_this)
        transformed.append(backlog[10])
        # Comma at the end of this segment means that the next segment is an embedded sentence if it has a verb
        comma_end = int(tags[-1] == '$,')
        transformed.append(comma_end)

        # Percentage of 'emphatic' punctuation marks: ?,!,-,–
        transformed.append((token_strings.count('?') + token_strings.count('!') + token_strings.count('-') + token_strings.count('–'))/len(token_strings))
        # Question?
        transformed.append(int((token_strings.count('?') > 0)))

        # Quotes features
        # Opening Quotes in this segment?
        open_quote = len([tag for tag in tags if tag == "#OPEN_QUOTE#"])
        # Closing Quotes in this segment?
        close_quote = len([tag for tag in tags if tag == "#CLOSE_QUOTE#"])
        # In quotes?
        in_quotes = int(backlog[11] > 0 or open_quote > 0)
        transformed.append(open_quote)
        transformed.append(close_quote)
        transformed.append(in_quotes)
        # How many contiguous prev. segments have been in quotes so far? This is meant to tackle errors bc of missing closing quotes
        # as well as marking sequences of embedded narration
        transformed.append(backlog[49])

        # Special combinations direct - full quoted sentence (sent. ending punct. before closing quotes),
        # comma after closing quotes (prob. frame of direct speech)
        transformed.append(int(len([tag for i, tag in enumerate(tags) if tag == "#CLOSE_QUOTE#" and i > 0 and tags[i-1] == "$."]) > 0))
        transformed.append(int((backlog[12] == 1 and token_strings[0] == ",") or (len([tag for i, tag in enumerate(tags) if tag == "#CLOSE_QUOTE#" and i < len(token_strings)-1 and token_strings[i+1] == ","]) > 0)))

        # --- Morphological Features ---
        # percentage of first and second person pronouns (personal, possessive, reflexive)
        per1 = [morph_tag for morph_tag in morph_tagged if len(morph_tag) > 2 and morph_tag[0] == 'PRO' and
                 morph_tag[1] in ['Pers', 'Pos', 'Refl'] and morph_tag[3] == '1']
        per2 = [morph_tag for morph_tag in morph_tagged if len(morph_tag) > 2 and morph_tag[0] == 'PRO' and
                 morph_tag[1] in ['Pers', 'Pos', 'Refl'] and morph_tag[3] == '2']
        per12 = [morph_tag for morph_tag in morph_tagged if len(morph_tag) > 2 and morph_tag[0] == 'PRO' and
                 morph_tag[1] in ['Pers', 'Pos', 'Refl'] and morph_tag[3] in ['1', '2']]
        transformed.append(len(per1) / len(token_strings))
        # Second person might be a better feature than 1. and 2. together as it is seldom the perspective of a narrative
        transformed.append(len(per2) / len(token_strings))
        transformed.append(len(per12)/len(token_strings))
        # percentage of third person pronouns (personal, possessive, reflexive)
        per3 = [morph_tag for morph_tag in morph_tagged if len(morph_tag) > 2 and morph_tag[0] == 'PRO' and
                 morph_tag[1] in ['Pers', 'Pos', 'Refl'] and morph_tag[3] == '3']
        transformed.append(len(per3) / len(token_strings))

        # Note changes in the usage of person; this might help to distinguish between third and first person perspective narratives
        # Only third person in prev. five segments?
        transformed.append(int(len([b for b in backlog[43:48] if b == '3']) > 0 and len([b for b in backlog[43:48] if b in ['1', '1_3']]) == 0))
        # Only first person in prev. five segments?
        transformed.append(int(len([b for b in backlog[43:48] if b == '1']) > 0 and len([b for b in backlog[43:48] if b in ['3', '1_3']]) == 0))
        # Mixed first and third person in prev. five segments
        transformed.append(int(len([b for b in backlog[43:48] if b == '3_1']) > 0 or (len([b for b in backlog[43:48] if b == '3']) > 0 and len([b for b in backlog[43:48] if b == '1']) > 0)))

        # tempus and modus features
        has_ind = int(len([morph_tag for morph_tag in morph_tagged if len(morph_tag) > 5 and morph_tag[0] == 'VFIN' and
                           morph_tag[5] == 'Ind']) > 0)
        has_subj = int(len([morph_tag for morph_tag in morph_tagged if len(morph_tag) > 5 and morph_tag[0] == 'VFIN' and
                           morph_tag[5] == 'Subj']) > 0)
        no_subj = int(not any([morph_tag[5] == 'Subj' for morph_tag in morph_tagged if len(morph_tag) > 5 and morph_tag[0] == 'VFIN']))
        no_ind = int(not any([morph_tag[5] == 'Ind' for morph_tag in morph_tagged if len(morph_tag) > 5 and morph_tag[0] == 'VFIN']))
        has_pres = int(len([morph_tag for morph_tag in morph_tagged if len(morph_tag) > 5 and morph_tag[0] == 'VFIN' and
                           morph_tag[4] == 'Pres']) > 0)
        has_past = int(len([morph_tag for morph_tag in morph_tagged if len(morph_tag) > 5 and morph_tag[0] == 'VFIN' and
                           morph_tag[4] == 'Past']) > 0)
        no_past = int(not any([morph_tag[4] == 'Past' for morph_tag in morph_tagged if len(morph_tag) > 5 and morph_tag[0] == 'VFIN']))
        no_pres = int(not any([morph_tag[4] == 'Pres' for morph_tag in morph_tagged if len(morph_tag) > 5 and morph_tag[0] == 'VFIN']))
        for feature in [has_ind, has_subj, no_subj, no_ind, has_pres, has_past, no_past, no_pres]:
            transformed.append(feature)

        # --- Grammatical features ---
        # Comma at the end of the prev. segment means that this segment is an embedded sentence if it has a verb
        if backlog[13] and any([tag in ['VFIN', 'VAFIN'] for tag in tags]):
            transformed.append(1)
        else:
            transformed.append(0)
        # A form of verb 'würden' + infinitive can be a pointer towards free indirect
        transformed.append(int(any([lemma == 'würden' for lemma in token_lemmata])
                               and any(
            [(tag in ['VAINF', 'VMINF', 'VVINF', 'VVIZU'] and token_lemmata[i] != 'würden') for i, tag in
             enumerate(tags)])))
        transformed.append(int(any([lemma == 'würden' for lemma in token_lemmata])))

        # Noun/prepositional complements of a rep. word point toward reported STW,
        # sentence/infinitive complements point towards indirect STW
        all_stw_words = [token for i,token in enumerate(tokens) if any(stw_words_orig["Word"].str.contains(r'\b{}\b'.format(re.escape(token_lemmata[i]))))]
        has_prep_noun_comp = int(len([rep_word for rep_word in all_stw_words if len([child for child in rep_word.children if child.pos_ in ['ADP', 'PROPN', 'NOUN'] and child.dep_.startswith('o')]) > 0]) > 0)
        has_claus_inf_comp = int(len([rep_word for rep_word in all_stw_words if len([child for child in rep_word.children if child.dep_ == 'oc']) > 0]) > 0)
        transformed.append(has_prep_noun_comp)
        transformed.append(has_claus_inf_comp)

        # --- Possible speaker features ---
        # Is subject a pronoun, a person NE or a "Person" head noun -> possible speaker
        cand_speakers = [tokens[i] for i,tag in enumerate(tags) if (tag in['PPER', 'PIS', 'PDS'] or (tag in ['NE', 'NNE'] and 'PER' in [ent for ent in doc.ents if tokens[i].idx >= ent.start and tokens[i].idx <= ent.end]))]

        # Check whether any noun phrase has a head that is a synset of "Person" in Germanet
        person = []
        with open('data/person.txt', 'r', encoding='utf-8') as f:
            for l in f:
                person.append(l)

        for np in doc.noun_chunks:
            if np.root.text in person:
                cand_speakers.append(np.root)

        subj_cand_speaker = [token for token in cand_speakers if token.dep_ == 'sb']
        # How many possible speakers/addressees are there in relation to the segment length?
        num_cand_speaker = len(cand_speakers)/len(tokens)
        transformed.append(int(len(subj_cand_speaker) > 0))
        transformed.append(num_cand_speaker)
        # Append prev. segments candidate speaker features
        transformed.append(backlog[38])
        transformed.append(backlog[39])

        # --- Reporting word features ---
        # Appearance of reporting word by penalty
        has_rep_word_0 = int(any([stw_words[stw_words["Penalty"] == 0]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_rep_word_1 = int(any([stw_words[stw_words["Penalty"] == 1]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_rep_word_2 = int(any([stw_words[stw_words["Penalty"] == 2]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_rep_word_3 = int(any([stw_words[stw_words["Penalty"] == 3]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_rep_word_4 = int(any([stw_words[stw_words["Penalty"] == 4]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_rep_word_5 = int(any([stw_words[stw_words["Penalty"] == 5]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))

        # Appearance of reporting word lower or equal a certain penalty
        has_rep_word_le_1 = int(any([stw_words[stw_words["Penalty"] <= 1]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_rep_word_le_2 = int(any([stw_words[stw_words["Penalty"] <= 2]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_rep_word_le_3 = int(any([stw_words[stw_words["Penalty"] <= 3]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_rep_word_le_4 = int(any([stw_words[stw_words["Penalty"] <= 4]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_rep_word_le_5 = int(any([stw_words[stw_words["Penalty"] <= 5]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        # Appearance of noun/verb reporting word -> this might be interesting to differentiate 'reported' from 'direct/'indirect'
        has_rep_word_noun = int(any([(len(stw_words[(stw_words[stw_words["Penalty"] <= 5]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma)))) & (stw_words[stw_words["Penalty"] <= 5]["Word"].str.istitle())]) > 0) for lemma in token_lemmata]))
        has_rep_word_verb = int(any([(len(stw_words[(stw_words[stw_words["Penalty"] <= 5]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma)))) & (stw_words[stw_words["Penalty"] <= 5]["Word"].str.islower())]) > 0) for lemma in token_lemmata]))
        for feature in [has_rep_word_0, has_rep_word_1, has_rep_word_2, has_rep_word_3, has_rep_word_4, has_rep_word_5,
                        has_rep_word_le_1, has_rep_word_le_2, has_rep_word_le_3, has_rep_word_le_4, has_rep_word_le_5,
                        has_rep_word_noun, has_rep_word_verb]:
            transformed.append(feature)

        # Appearance of special reporting words for reported class by penalty
        has_spec_rep_word_0 = int(any([stw_words_rep[stw_words_rep["Penalty"] == 0]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_spec_rep_word_1 = int(any([stw_words_rep[stw_words_rep["Penalty"] == 1]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_spec_rep_word_2 = int(any([stw_words_rep[stw_words_rep["Penalty"] == 2]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_spec_rep_word_3 = int(any([stw_words_rep[stw_words_rep["Penalty"] == 3]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_spec_rep_word_4 = int(any([stw_words_rep[stw_words_rep["Penalty"] == 4]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_spec_rep_word_5 = int(any([stw_words_rep[stw_words_rep["Penalty"] == 5]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))

        # Appearance of special reporting words lower or equal a certain penalty
        has_spec_rep_word_le_1 = int(any([stw_words_rep[stw_words_rep["Penalty"] <= 1]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_spec_rep_word_le_2 = int(any([stw_words_rep[stw_words_rep["Penalty"] <= 2]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_spec_rep_word_le_3 = int(any([stw_words_rep[stw_words_rep["Penalty"] <= 3]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_spec_rep_word_le_4 = int(any([stw_words_rep[stw_words_rep["Penalty"] <= 4]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        has_spec_rep_word_le_5 = int(any([stw_words_rep[stw_words_rep["Penalty"] <= 5]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata]))
        
        for feature in [has_spec_rep_word_0, has_spec_rep_word_1, has_spec_rep_word_2, has_spec_rep_word_3, has_spec_rep_word_4,
                        has_spec_rep_word_5,
                        has_spec_rep_word_le_1, has_spec_rep_word_le_2, has_spec_rep_word_le_3, has_spec_rep_word_le_4,
                        has_spec_rep_word_le_5]:
            transformed.append(feature)

        # Number of reporting word by penalty
        num_rep_word_0 = sum([stw_words[stw_words["Penalty"] == 0]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_rep_word_1 = sum([stw_words[stw_words["Penalty"] == 1]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_rep_word_2 = sum([stw_words[stw_words["Penalty"] == 2]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_rep_word_3 = sum([stw_words[stw_words["Penalty"] == 3]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_rep_word_4 = sum([stw_words[stw_words["Penalty"] == 4]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_rep_word_5 = sum([stw_words[stw_words["Penalty"] == 5]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])

        # Number of reporting word lower or equal a certain penalty
        num_rep_word_le_1 = sum([stw_words[stw_words["Penalty"] <= 1]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_rep_word_le_2 = sum([stw_words[stw_words["Penalty"] <= 2]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_rep_word_le_3 = sum([stw_words[stw_words["Penalty"] <= 3]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_rep_word_le_4 = sum([stw_words[stw_words["Penalty"] <= 4]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_rep_word_le_5 = sum([stw_words[stw_words["Penalty"] <= 5]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        # Number of noun/verb reporting word -> this might be interesting to differentiate 'reported' from 'direct/'indirect'
        num_rep_word_noun = sum([(len(stw_words[(stw_words[stw_words["Penalty"] <= 5]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma)))) & (stw_words[stw_words["Penalty"] <= 5]["Word"].str.istitle())]) > 0) for lemma in token_lemmata])
        num_rep_word_verb = sum([(len(stw_words[(stw_words[stw_words["Penalty"] <= 5]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma)))) & (stw_words[stw_words["Penalty"] <= 5]["Word"].str.islower())]) > 0) for lemma in token_lemmata])
        for feature in [num_rep_word_0, num_rep_word_1, num_rep_word_2, num_rep_word_3, num_rep_word_4,
                        num_rep_word_5,
                        num_rep_word_le_1, num_rep_word_le_2, num_rep_word_le_3, num_rep_word_le_4,
                        num_rep_word_le_5,
                        num_rep_word_noun, num_rep_word_verb]:
            transformed.append(feature)

        # Number of special reporting words for reported class by penalty
        num_spec_rep_word_0 = sum([stw_words_rep[stw_words_rep["Penalty"] == 0]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_spec_rep_word_1 = sum([stw_words_rep[stw_words_rep["Penalty"] == 1]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_spec_rep_word_2 = sum([stw_words_rep[stw_words_rep["Penalty"] == 2]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_spec_rep_word_3 = sum([stw_words_rep[stw_words_rep["Penalty"] == 3]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_spec_rep_word_4 = sum([stw_words_rep[stw_words_rep["Penalty"] == 4]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_spec_rep_word_5 = sum([stw_words_rep[stw_words_rep["Penalty"] == 5]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])

        # Number of special reporting words lower or equal a certain penalty
        num_spec_rep_word_le_1 = sum([stw_words_rep[stw_words_rep["Penalty"] <= 1]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_spec_rep_word_le_2 = sum([stw_words_rep[stw_words_rep["Penalty"] <= 2]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_spec_rep_word_le_3 = sum([stw_words_rep[stw_words_rep["Penalty"] <= 3]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_spec_rep_word_le_4 = sum([stw_words_rep[stw_words_rep["Penalty"] <= 4]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])
        num_spec_rep_word_le_5 = sum([stw_words_rep[stw_words_rep["Penalty"] <= 5]["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma))).any() for lemma in token_lemmata])

        for feature in [num_spec_rep_word_0, num_spec_rep_word_1, num_spec_rep_word_2, num_spec_rep_word_3,
                        num_spec_rep_word_4,
                        num_spec_rep_word_5,
                        num_spec_rep_word_le_1, num_spec_rep_word_le_2, num_spec_rep_word_le_3,
                        num_spec_rep_word_le_4,
                        num_spec_rep_word_le_5]:
            transformed.append(feature)
            
        # Reporting word features prev. segment
        for feature in backlog[14:38]:
            transformed.append(feature)
        for feature in backlog[50:74]:
            transformed.append(feature)

        # Word vectors
        # Get prototypical word vector for reporting words
        proto_rep_vec = numpy.average([self.wordvecs[word] for word in stw_words[stw_words["Penalty"] == 0] if word in self.wordvecs], axis=0)
        # Get prototypical word vector for reported class
        proto_rep_vec_reporting = numpy.average([self.wordvecs[word] for word in stw_words_rep[stw_words_rep["Penalty"] == 0] if word in self.wordvecs], axis=0)
        # Append highest similarity values to proto word vectors within the segment
        max_sim = .0
        max_sim_rep = .0
        for lemma in token_lemmata:
            if lemma in self.wordvecs:
                lemma_vec = self.wordvecs[lemma]
                # cosine similarity = 1 - cosine distance
                sim = 1 - distance.cosine(lemma_vec, proto_rep_vec)
                sim_rep = 1 - distance.cosine(lemma_vec, proto_rep_vec_reporting)

                if sim > max_sim:
                    max_sim = sim
                if sim_rep > max_sim_rep:
                    max_sim_rep = sim_rep

        transformed.append(max_sim)
        transformed.append(max_sim_rep)

        # --- Other word features ---
        # Usage of deictic words can point to character speech - precentage of deictic words
        transformed.append(len([t for t in token_strings if t in DEICTIC])/len(token_strings))
        # Usage of special conjunction at the beginning of the segment can point to indirect
        transformed.append(int(token_strings[0] in CONJUNCT))
        # Usage of modal particles can point towards character speech
        transformed.append(len([t for t in token_strings if t in MODAL_PART])/len(token_strings))
        # Negation?
        transformed.append(len([lemma for lemma in token_lemmata if lemma in NEG])/len(token_strings))

        # Words describing facial expressions, gestures, voice might hint towards STWR
        transformed.append(int(len([lemma for lemma in token_lemmata if lemma in FACIAL]) > 0))
        transformed.append(int(len([lemma for lemma in token_lemmata if lemma in GESTURE]) > 0))
        transformed.append(int(len([lemma for lemma in token_lemmata if lemma in VOICE]) > 0))

        # The repetition of words can hint towards figural speech
        transformed.append(int(any([count >= 2 for count in [token_lemmata.count(el) for el in token_lemmata]])))

        # --- Sequential features ---

        if self.sequence_features:

            # Labels of prev. segment
            labels_last = [l for i,l in enumerate(backlog[9].split(",")) if i%3==0]
            transformed.append(int(any([l.startswith('direct') for l in labels_last])))
            transformed.append(int(any([l.startswith('indirect') for l in labels_last])))
            transformed.append(int(any([l.startswith('free_indirect') for l in labels_last])))
            transformed.append(int(any([l.startswith('reported') for l in labels_last])))
            # Label appears in 5 prev. segments
            labels_last_5 = [fin_l for ls in [[l for i, l in enumerate(label.split(",")) if i % 3 == 0] for label in backlog[5:10]] for fin_l in ls]
            transformed.append(int(any([l.startswith('direct') for l in labels_last_5])))
            transformed.append(int(any([l.startswith('indirect') for l in labels_last_5])))
            transformed.append(int(any([l.startswith('free_indirect') for l in labels_last_5])))
            transformed.append(int(any([l.startswith('reported') for l in labels_last_5])))
            # How many labels for each class and overall within the last 10 segments
            labels_last_10 = [fin_l for ls in [[l for i, l in enumerate(label.split(",")) if i % 3 == 0] for label in backlog[0:10]] for fin_l in ls if fin_l != ""]
            transformed.append(len([l for l in labels_last_10 if l.startswith('direct')]))
            transformed.append(len([l for l in labels_last_10 if l.startswith('indirect')]))
            transformed.append(len([l for l in labels_last_10 if l.startswith('free_indirect')]))
            transformed.append(len([l for l in labels_last_10 if l.startswith('reported')]))
            transformed.append(len(labels_last_10))

        # --- Other features ---
        # Segment and character lengths
        transformed.append(len(token_strings))
        transformed.append(len(original_text))
        # Segment and character lengths of prev. segment
        transformed.append(backlog[40])
        transformed.append(backlog[41])
        # Segment and character lengths of this + prev. segment
        transformed.append(len(token_strings) + backlog[40])
        transformed.append(len(original_text) + backlog[41])
        # Is this segment at the start or end of a paragraph?
        paragraph_end = int("<p>" in original_text)
        transformed.append(paragraph_end)
        transformed.append(backlog[42])

        # --- Update Backlog ---
        # [0:10] encode labels of previous ten segments -> updated elsewhere
        # 10: Colon in prev. segment
        backlog[10] = colon_this
        # 11: How many open quotes
        backlog[11] += open_quote
        if backlog[11] - close_quote >= 0:
            backlog[11] -= close_quote
        else:
            backlog[11] = 0
        # 12: Prev. segment ends with close_quote
        backlog[12] = int(tags[-1] == "#CLOSE_QUOTE#")
        # 13: Comma at the end of this segment
        backlog[13] = comma_end
        # [14:38] reportin word appearance features prev. segment
        for i, feature in enumerate([has_rep_word_0, has_rep_word_1, has_rep_word_2, has_rep_word_3, has_rep_word_4,
                                     has_rep_word_5,
                                     has_rep_word_le_1, has_rep_word_le_2, has_rep_word_le_3, has_rep_word_le_4,
                                     has_rep_word_le_5,
                                     has_rep_word_noun, has_rep_word_verb,
                                     has_spec_rep_word_0, has_spec_rep_word_1, has_spec_rep_word_2, has_spec_rep_word_3,
                                     has_spec_rep_word_4, has_spec_rep_word_5,
                                     has_spec_rep_word_le_1, has_spec_rep_word_le_2, has_spec_rep_word_le_3,
                                     has_spec_rep_word_le_4, has_spec_rep_word_le_5
            ]):
            backlog[14 + i] = feature
        # 38: Candidate speakers as subject
        backlog[38] = int(len(subj_cand_speaker) > 0)
        # 39: Percentage of candidate speakers
        backlog[39] = num_cand_speaker
        # 40, 41: lengths of prev. segment
        backlog[40] = len(token_strings)
        backlog[41] = len(original_text)
        # 42: paragraph end
        backlog[42] = paragraph_end

        # [43:48]: keep track of pronoun person appearances in the 5 prev. segments
        backlog[43:47] = backlog[44:48]
        if per3:
            if per1:
                backlog[48] = '3_1'
            else:
                backlog[48] = '3'
        elif per1:
            backlog[48] = '1'
        else:
            backlog[48] = '-'

        # 49: How many contiguous prev. segments have been in quotes?
        if in_quotes:
            backlog[49] += 1
        else:
            backlog[49] = 0

        # [50:74] reportin word count features prev. segment
        for i, feature in enumerate([num_rep_word_0, num_rep_word_1, num_rep_word_2, num_rep_word_3, num_rep_word_4,
                                     num_rep_word_5,
                                     num_rep_word_le_1, num_rep_word_le_2, num_rep_word_le_3, num_rep_word_le_4,
                                     num_rep_word_le_5,
                                     num_rep_word_noun, num_rep_word_verb,
                                     num_spec_rep_word_0, num_spec_rep_word_1, num_spec_rep_word_2,
                                     num_spec_rep_word_3,
                                     num_spec_rep_word_4, num_spec_rep_word_5,
                                     num_spec_rep_word_le_1, num_spec_rep_word_le_2, num_spec_rep_word_le_3,
                                     num_spec_rep_word_le_4, num_spec_rep_word_le_5
                                     ]):
            backlog[50 + i] = feature

        return transformed, backlog

### Helper methods
def string_tokenize(text, doc=None):
    """
    Method for string tokenizing a text.

    :param text: text
    :param doc: optional parameter, text parsed by spacy
    :return: list of string tokens
    """
    if not doc:
        doc = NLP(text)
    return [token.orth_ for token in doc]
