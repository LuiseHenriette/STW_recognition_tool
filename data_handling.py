#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Thesis: S(peech)T(hought)W(riting)R(epresentation) recognition
author: Luise Schricker

python-3.4

lxml-4.2.5
numpy-1.13.3
pandas-0.21.0

This file contains methods to extract and modify the corpus data presented by Annelen Brunner.
(Annelen Brunner. Automatische Erkennung von Redewiedergabe: ein Beitrag zur quantitativen Narratologie. Vol. 47. Walter de Gruyter, 2015.)
"""

import bisect
import glob
import os
import random
import re

import lxml.etree as ET
import numpy as np
import pandas as pd

from feature_extraction import SimpleBaseline


RANDOM_STATE = 5


### Training data preparation

def get_training_set(path, feature_extractor, label, shuffle=True, original_labels=False):
    """
    Method to extract a training set from the extracted corpus for the given label. Either loads
    saved data-set as feature representations or extracts features and labels from original data.

    :param path: The path to the extracted corpus files.
    :param feature_extractor: Instance of a feature extractor which takes a Dataframe and returns it after transforming the datapoints within it's rows.
    :param label: The label of positive class instances. One of: 'direct', 'indirect', 'free_indirect', 'reported', 'thought', 'speech', 'writing'
    :param shuffle: Flag indicating whether to shuffle the training data. Shuffling is only performed if features are extracted.
    :param original_labels: If set to True, return the original labels and texts as columns of the training data.
    :return: A pandas dataframe containing the training instances, transformed by the feature extractor and the corresponding labels
    """
    _, df_by_text = read_extracted_files(path)

    # Each row consists of a training instance with x features and the label
    df_train = pd.DataFrame()
    extract = False

    # Check if the training data was already transformed to feature representations, don't use precomputed features
    # for the Baseline. In case precomputed features are used and the sequence label features are enabled but not part
    # of the precomputed features, trigger a new feature extraction process.
    path_features = "".join([path, "_features.tsv"])
    if os.path.isfile(path_features) and not isinstance(feature_extractor, SimpleBaseline):

        # In case precomputed features are used and the sequence label features are disabled,
        # delete these features from the data
        if not feature_extractor.sequence_features:
            df_train = pd.read_csv(path_features, index_col=None, header=0, sep='\t')

            if len(df_train.columns) == 243:
                print("Loading precomputed training set. To compute features from corpus data, delete the file {}.".format(
                        path_features))
                df_train = pd.read_csv(path_features, index_col=None,
                                       usecols=(lambda x: int(x) <= 221 or int(x) >= 235), header=0, sep='\t')

            elif (len(df_train.columns) < 243):
                extract = True

        else:
            print("Loading precomputed training set. To compute features from corpus data, delete the file {}.".format(
                path_features))
            df_train = pd.read_csv(path_features, index_col=None, header=0, sep='\t')

    else:
        extract = True

    if extract:
        # Load training data and extract features and labels
        print("Extracting training set...")

        # Feature extraction
        for idx, (name, text) in enumerate(df_by_text):
            print("\nExtracting {} ({}/{}) ...".format(name, idx+1, len(df_by_text)))
            # Replace NaN values with empty strings
            text = text.fillna("")

            # Feature extractor takes whole text instead of sigle datapoints to be able to extract context sensitive features
            # Returned backlog can be discarded
            transformed_text, _ = feature_extractor.transform(text)
            df_train = pd.concat([df_train, transformed_text], axis=0, ignore_index=True)

        # Write data to file if it's not the Baseline
        if not isinstance(feature_extractor, SimpleBaseline):
            df_train.to_csv(path_or_buf=path_features, sep='\t', index=False)

    # Label extraction
    df_labels = pd.DataFrame()

    for idx, (name, text) in enumerate(df_by_text):
        # Replace NaN values with empty strings
        text = text.fillna("")

        # Get labels: positive examples := 1, negative examples := 0
        if label in ['direct', 'indirect', 'free_indirect', 'reported']:
            labels = text["labels_spans"].apply(lambda x: int(any([t.startswith(label) for t in x.split(",")])))
        else:
            # labels speech, thought, writing do not contain each other partly
            labels = text["labels_spans"].apply(lambda x: int(any([label in l for l in x.split(",")])))

        # Add original labels and text for span evaluation if original_labels is True
        if original_labels:
            orig_labels = pd.concat([text["labels_spans"], text["text"]], axis=1,
                                         ignore_index=True)
            labels = pd.concat([orig_labels, labels], axis=1, ignore_index=True)
        df_labels = pd.concat([df_labels, labels], axis=0, ignore_index=True)

    df_train = pd.concat([df_train, df_labels], axis=1, ignore_index=True)

    if shuffle:
        df_train = df_train.sample(random_state=RANDOM_STATE, frac=1).reset_index(drop=True)

    print("Done.\n")

    return df_train

def get_labels_stw(y_orig_labels, clf_class):
    """
    Method for getting the speech, thought & writing labels for the clf_class predictions.

    :param y_orig_labels: The original labels of the instances classified as clf_class.
    :param clf_class: One of direct, indirect, free_indirect and reported.
    :return: The labels for speech, thought & writing.
    """
    labels = {}

    for type in ['speech', 'thought', 'writing']:
        stw_labels = [[l for l in label.split(",") if '_' in l] for label in y_orig_labels]
        labels[type] = [int(any([type in l for l in label if l.startswith(clf_class)])) for label in stw_labels]

    return labels['speech'], labels['thought'], labels['writing']


### Data augmentation methods

def augment_data(X, y, cl=None):
    """
    Method for countering class imbalance by

    :param X: The datapoints transformed into feature vectors.
    :param y: The labels.
    :param cl: The class of the positive samples.
    :return: The augmented dataset plus labels
    """
    # Set random seed
    random.seed(a=RANDOM_STATE)

    df = X.copy()
    df.loc[:,'y'] = y.tolist()

    pos_samples = X[df['y'] == 1]
    neg_samples = X[df['y'] == 0]

    diff = len(neg_samples) - len(pos_samples)
    augment = pos_samples.sample(random_state=RANDOM_STATE, replace=True, frac=(diff/len(pos_samples))).reset_index(drop=True)

    # Pick the correct augmentation method according to the class
    if cl == 'direct':
        method = augment_direct
    elif cl == 'indirect':
        method = augment_indirect
    elif cl == 'free_indirect':
        method = augment_free_indirect
    elif cl == 'reported':
        method = augment_reported

    print("Adding {} samples by augmenting data...".format(diff))
    for _, sample in augment.iterrows():
        X = X.append(pd.Series(method(sample)), ignore_index=True)
        y = np.append(y, 1)

    return X, y


def augment_direct(X):
    """
    Augment sample of direct class

    :param X: Sample as feature vector
    :return: The augmented sample
    """
    # Features that can change for direct class:
    # - in_quotes + has_opening_quote + has_closing_quote features (88, 86, 87)
    # - colon feature (81)
    # - question feature (85)

    features = ['in_quotes', 'colon_feature', 'question_feature']
    selected_features = random.sample(features, random.randint(1, len(features)))

    for feature in selected_features:
        if feature == 'in_quotes':
            if X[88] == 1:
                X[88] = 0
                X[86] = 0
                X[87] = 0
            else:
                X[88] = 1

        elif feature == 'colon_feature':
            if X[81] == 1:
                X[81] == 0
            else:
                X[81] == 1

        elif feature == 'question_feature':
            if X[85] == 1:
                X[85] == 0
            else:
                X[85] == 1
    return X


def augment_indirect(X):
    """
    Augment sample of indirect class

    :param X: Sample as feature vector
    :return: The augmented sample
    """
    # Features that can change for indirect class:
    # - all_subjunctive (102)
    # - has_clause_inf (111)
    # - special_conjunct (165)

    features = ['all_subjunctive', 'has_clause_inf', 'special_conjunct']
    selected_features = random.sample(features, random.randint(1, len(features)))

    for feature in selected_features:
        if feature == 'in_quotes':
            if X[102] == 1:
                # Set subjunctive features to 0
                X[102] = 0
                X[100] = 0
                # Set indicative to 1
                X[99] = 1
                X[101] = 1
            else:
                # Set subjunctive features to 1
                X[102] = 1
                X[100] = 1
                # Set indicative to 0
                X[99] = 0
                X[101] = 0

        elif feature == 'has_clause_inf':
            if X[111] == 1:
                X[111] == 0
            else:
                X[111] == 1

        elif feature == 'special_conjunct':
            if X[165] == 1:
                X[165] == 0
            else:
                X[165] == 1
    return X


def augment_free_indirect(X):
    """
    Augment sample of free_indirect class

    :param X: Sample as feature vector
    :return: The augmented sample
    """
    # Features that can change for free_indirect class:
    # - question feature (85)
    # - würden_inf feature (108)
    # - deictic feature (164)

    features = ['question_feature', 'würden_inf', 'deictic_feature']
    selected_features = random.sample(features, random.randint(1, len(features)))

    for feature in selected_features:
        if feature == 'question_feature':
            if X[85] == 1:
                X[85] == 0
            else:
                X[85] == 1
        elif feature == 'würden_inf':
            if X[108] == 1:
                X[108] == 0
            else:
                X[108] == 1
        elif feature == 'deictic_feature':
            if X[164] == 0:
                # X[185] holds the sentence lengths
                X[164] == (random.randint(1,3)/X[185])
            else:
                X[164] == 0.0
    return X


def augment_reported(X):
    """
    Augment sample of reported class

    :param X: Sample as feature vector
    :return: The augmented sample
    """
    # Features that can change for reported class:
    # - subj_cand_speaker (112)
    # - embedded feature (107)
    # - has_prep_noun_comp (110)

    features = ['subj_cand_speaker', 'embedded_feature', 'has_prep_noun_comp']
    selected_features = random.sample(features, random.randint(1, len(features)))

    for feature in selected_features:
        if feature == 'subj_cand_speaker':
            if X[112] == 1:
                X[112] = 0
            else:
                X[112] = 1

        elif feature == 'embedded_feature':
            if X[107] == 1:
                X[107] == 0
            else:
                X[107] == 1

        elif feature == 'has_prep_noun_comp':
            if X[110] == 1:
                X[110] == 0
            else:
                X[110] == 1
    return X


### Corpus extraction/statistics


def extract_corpus(path):
    """
    Given the path to the corpus file, this method extracts Annelen Brunner's STWR corpus
    in the following format: <Section text> <labels/spans>  <attributes>
    Example: "text <p>"   direct_speech,2,10,indirect_thought,3,6,indirect_speech,3,6  -,ambig_narr,ambig_narr
    The extracted data is saved to tsv files (one per document).

    :param path: The path to the corpus file.
    """
    print("Extracting corpus data ...\n")

    # Makedir for output if it does not already exist
    try:
        os.mkdir("corpusExtracted")
    except OSError:
        print("Output directory already exists. Overwriting files in directory...")

    # Iterate over files in directory
    for filename in os.listdir(path):
        print("Extracting {}...".format(filename))

        # parse xml-file
        tree = ET.parse(os.path.join(path, filename))
        root = tree.getroot()

        dict = {}
        sections = []

        for child in root:
            if (child.tag == "GateDocument"):
                pass

            # Save words with their respective node-ids
            elif (child.tag == "TextWithNodes"):
                for el in child:
                    if el.tail:
                        dict[el.get("id")] = {"text": el.tail,
                                              "labels": [],
                                              "attributes": []}

            # Attribute annotations to words
            elif (child.tag == "AnnotationSet" and child.get("Name") == "RW_Anno"):
                for el in child:
                    if el.get("Type") in ['direct_speech', 'indirect_speech', 'reported_speech', 'free_indirect_speech',
                                        'direct_thought', 'indirect_thought', 'reported_thought', 'free_indirect_thought',
                                        'direct_writing', 'indirect_writing', 'reported_writing', 'free_indirect_writing',
                                        'frame']:
                        start = el.get("StartNode")
                        end = el.get("EndNode")
                        labels = [",".join([el.get("Type"), start, end])]

                        # Get type and attributes of labels, e.g. ambig, narr etc.
                        attributes = []
                        for at in el:
                            for i, att in enumerate(at):

                                # Ambigue labels are listed twice
                                if att.text in ['narr', 'prag', 'metaph', 'border', 'non-fact', 'ambig']:
                                    attributes.append(att.text)

                        for idx in range(int(start),int(end)):
                            if str(idx) in dict:
                                dict[str(idx)]["labels"] += labels
                                dict[str(idx)]["attributes"].append(attributes)

                                # There should be the same number of attribute lists as labels
                                assert len(dict[str(idx)]["attributes"]) == len(dict[str(idx)]["labels"])

            # Get sections and paragraph tags
            elif (child.tag == "AnnotationSet" and child.get("Name") == "PreProc_Anno"):
                paragraphs = []

                for el in child:
                    # Sections
                    if el.get("Type") == "Section":
                        start = int(el.get("StartNode"))
                        end = int(el.get("EndNode"))

                        # Save sections for later
                        sections.append((start, end))

                    # Paragraphs
                    elif el.get("Type") == "p":
                        # Only paragraph endings are interesting
                        paragraphs.append(int(el.get("EndNode")))

        # Sort sections
        sections.sort(key=lambda t: t[0])
        section_ends = [sec[1] for sec in sections]

        # Find the closest section border to append the paragraph ending symbol
        for paragraph in paragraphs:
            closest_section_end = section_ends[bisect.bisect_right(section_ends, paragraph)-1]-1
            if str(closest_section_end) not in dict:
                dict[str(closest_section_end)] = {"text": " <p>",
                                              "labels": [],
                                              "attributes": []}
            else:
                dict[str(closest_section_end)]["text"] += " <p>"

        # Collect text, labels, spans and attributes by section
        df = pd.DataFrame(index=np.arange(0, len(sections)), columns=["text", "labels_spans", "attributes"])

        for i, section in enumerate(sections):

            text = ""
            labels = []
            attributes = []
            # Keep track of labels with global spans in order to not add duplicates
            labels_global = []

            for idx in range(section[0], section[1]):
                if str(idx) in dict:
                    token = dict[str(idx)]
                    text += token["text"]

                    if token["labels"]:
                        for j, label in enumerate(token["labels"]):
                            if label not in labels_global:
                                labels_global.append(label)
                                # Get section-based label spans
                                label_split = label.split(",")
                                start = len(text)-len(token["text"])
                                end = start+(int(label_split[2])-int(label_split[1]))
                                labels.append([label_split[0], str(start), str(end)])
                                if token["attributes"][j]:
                                    attributes.append("_".join(token["attributes"][j]))
                                else:
                                    attributes.append("-")

            # Check for longer labels that are split into sections; for these the end span has to be corrected
            for j,label in enumerate(labels):
                if int(label[2])>len(text):
                    label[2] = str(len(text))
                labels[j] = (",".join(label))

            # Save in dataframe
            df.loc[i] = [text, ",".join(labels), ",".join(attributes)]

        # Write data to file
        df.to_csv(path_or_buf="corpusExtracted/{}.tsv".format(filename.split(".")[0]), sep='\t', index=False)

    print("\nDone. Data written to directory '{}'.\n".format("corpusExtracted"))


def get_corpus_statistics(path):
    """
    Prints statistics for the extracted corpus data.

    :param path: The path to the extracted corpus files.
    """
    # Read data
    df, df_by_text = read_extracted_files(path)

    # Get class dataframes
    direct_df = df[(df['labels_spans'].str.contains("(?:[^in]direct|^direct)", na=False))]
    indirect_df = df[(df['labels_spans'].str.contains("(?:[^free_]indirect|^indirect)", na=False))]
    free_indirect_df = df[(df['labels_spans'].str.contains("free_indirect", na=False))]
    reported_df = df[(df['labels_spans'].str.contains("reported", na=False))]

    # Compute numbers for some classes
    num_direct = len(direct_df)
    num_indirect = len(indirect_df)
    num_free_indirect = len(free_indirect_df)
    num_reported = len(reported_df)

    print("\nStatistics:\n++++++++++++++\n")

    print("Total number of datapoints (segments): {}".format(len(df)))
    print("-------------------------------------------")
    # direct, indirect, free_indirect, reported classes
    print("Total number of datapoints (segments) for direct class: {}".format(num_direct))
    print("Total number of datapoints (segments) for indirect class: {}".format(num_indirect))
    print("Total number of datapoints (segments) for free indirect class: {}".format(num_free_indirect))
    print("Total number of datapoints (segments) for reported class: {}".format(num_reported))
    print("Total number of datapoints (segments) for narration class (including frame class): {}".format(len(df[(df['labels_spans'].isna()) | (df['labels_spans'].str.contains("^[^_]+$", na=False))])))

    # Speech, Thought, Writing classes
    print("\nTotal number of datapoints (segments) for speech class: {}".format(len(df[(df['labels_spans'].str.contains("speech", na=False))])))
    print("Total number of datapoints (segments) for thought class: {}".format(len(df[(df['labels_spans'].str.contains("thought", na=False))])))
    print("Total number of datapoints (segments) for writing class: {}".format(len(df[(df['labels_spans'].str.contains("writing", na=False))])))

    print("\n(Note: Multiple classes per segment are possible, therefore the above numbers don't add up.)")

    # Numbers for individual texts
    print("\nClass distribution for individual texts:")
    print("----------------------------------------")
    for text in df_by_text:
        df_text = text[1]
        print("{}:".format(text[0]))
        # direct, indirect, free_indirect, reported classes
        print("Number of datapoints (segments) for direct class: {}".format(len(df_text[(df_text['labels_spans'].str.contains("(?:[^in]direct|^direct)", na=False))])))
        print("Number of datapoints (segments) for indirect class: {}".format(len(df_text[(df_text['labels_spans'].str.contains("(?:[^free_]indirect|^indirect)", na=False))])))
        print("Number of datapoints (segments) for free indirect class: {}".format(len(df_text[(df_text['labels_spans'].str.contains("free_indirect", na=False))])))
        print("Number of datapoints (segments) for reported class: {}".format(len(df_text[(df_text['labels_spans'].str.contains("reported", na=False))])))
        print("Number of datapoints (segments) for narration class (including frame class): {}\n".format(
        len(df_text[(df_text['labels_spans'].isna()) | (df_text['labels_spans'].str.contains("^[^_]+$", na=False))])))

    print("Border cases ('narr', 'prag', 'metaph', 'border', 'non-fact', 'ambig'):")
    print("-----------------------------------------------------------------------")
    # Percentage of border cases per class
    print("Percentage of datapoints (segments) for direct class marked as border cases: {}%".format(round((len(direct_df) - len(filter_border(direct_df, "direct")))/(len(direct_df)*0.01),2)))
    print("Percentage of datapoints (segments) for indirect class marked as border cases: {}%".format(round((len(indirect_df) - len(filter_border(indirect_df, "indirect")))/(len(indirect_df)*0.01),2)))
    print("Percentage of datapoints (segments) for free indirect class marked as border cases: {}%".format(round((len(free_indirect_df) - len(filter_border(free_indirect_df, "free_indirect")))/(len(free_indirect_df)*0.01),2)))
    print("Percentage of datapoints (segments) for reported class marked as border cases: {}%".format(round((len(reported_df) - len(filter_border(reported_df, "reported")))/(len(reported_df)*0.01),2)))

    print("\nAccuracy of labels:")
    print("---------------------")
    # Multilabels
    print("Percentage of multiple labels for direct class: {}%".format((round(100 - (len(df[(df['labels_spans'].str.contains("^(?!.*indirect)(?!.*reported).*direct.*$", na=False))])/(num_direct*0.01)),2))))
    print("Percentage of multiple labels for indirect class: {}%".format((round(100 - (len(df[(df['labels_spans'].str.contains("^(?!.*free_indirect)(?!.*reported)(?!.*[^in]direct|^direct).*(?:,indirect|^indirect).*$", na=False))])/(num_indirect*0.01)),2))))
    print("Percentage of multiple labels for free indirect class: {}%".format((round(100 - (len(df[(df['labels_spans'].str.contains("^(?!.*(?:,indirect|^indirect))(?!.*reported)(?!.*,direct|^direct).*(?:,free_indirect|^free_indirect).*$", na=False))])/(num_free_indirect*0.01)),2))))
    print("Percentage of multiple labels for reported class: {}%\n".format((round(100 - (len(df[(df['labels_spans'].str.contains("^(?!.*(?:,indirect|^indirect))(?!.*free_indirect)(?!.*,direct|^direct).*(?:,reported|^reported).*$", na=False))])/(num_reported*0.01)),2))))

    # Percentage of Words that do not belong to a label span for labeled datapoints
    # direct class
    direct_instances = direct_df.values
    num_direct_words = sum([len(inst[0].split()) for inst in direct_instances])
    num_direct_words_labeled = sum(list(
        map(lambda segment: sum([len([token for token in mark_labeled_words(segment[0], segment[1], "direct").split() if '$' in token])]),
            direct_instances)))
    print("Percentage of words that are not labeled direct in instances of direct class: {}%".format(round(100 - (num_direct_words_labeled/(num_direct_words*0.01)),2)))

    # indirect class
    indirect_instances = indirect_df.values
    num_indirect_words = sum([len(inst[0].split()) for inst in indirect_instances])
    num_indirect_words_labeled = sum(list(
        map(lambda segment: sum([len([token for token in mark_labeled_words(segment[0], segment[1], "indirect").split() if '$' in token])]),
            indirect_instances)))
    print("Percentage of words that are not labeled indirect in instances of indirect class: {}%".format(
        round(100 - (num_indirect_words_labeled / (num_indirect_words * 0.01)), 2)))

    # free_indirect class
    free_indirect_instances = free_indirect_df.values
    num_free_indirect_words = sum([len(inst[0].split()) for inst in free_indirect_instances])
    num_free_indirect_words_labeled = sum(list(
        map(lambda segment: sum([len([token for token in mark_labeled_words(segment[0], segment[1], "free_indirect").split() if '$' in token])]),
            free_indirect_instances)))
    print("Percentage of words that are not labeled free_indirect in instances of free_indirect class: {}%".format(
        round(100 - (num_free_indirect_words_labeled / (num_free_indirect_words * 0.01)), 2)))

    # reported class
    reported_instances = reported_df.values
    num_reported_words = sum([len(inst[0].split()) for inst in reported_instances])
    num_reported_words_labeled = sum(list(
        map(lambda segment: sum([len([token for token in mark_labeled_words(segment[0], segment[1], "reported").split() if '$' in token])]),
            reported_instances)))
    print("Percentage of words that are not labeled reported in instances of reported class: {}%".format(
        round(100 - (num_reported_words_labeled / (num_reported_words * 0.01)), 2)))


### Helper functions

def read_extracted_files(path):
    """
    Method that reads the extracted corpus files into a single dataframe,
    as well as individual dataframes for each text.

    :param path: The path to the extracted corpus files.
    :return: All datapoints in one dataframe and a list containing individual dataframes for each text.
    """
    # Use regex to get all filenamnes
    files = glob.glob(path + "/*.tsv")

    file_list = []
    df_by_text = []

    # Iterate over files and read as df
    for file in files:
        df = pd.read_csv(file, index_col=None, header=0, sep='\t')
        file_list.append(df)
        # Save dataframe also for the individual texts along with the text names
        df_by_text.append((file.split("/")[1].split(".")[0], df))

    # Concatenate into one dataframe
    df = pd.concat(file_list, axis=0, ignore_index=True)

    return(df, df_by_text)


def mark_labeled_words(text, labels_spans, label, mark='$'):
    """
    Method for marking the labeled words in a text by appending a given mark, given a label-span combination
    in the given format (ex.: "direct_speech,2,10,indirect_thought,3,6,indirect_speech,3,6")
    and a label.

    :param text: text to extract the labeled span from
    :param labels_spans: label-span combination
    :param label: label to extract, can be one of direct, indirect, free_indirect and reported
    :param mark: character to use as mark, default is '$'
    :return: Text with the words within the label marked by the 'mark' token.
    """

    labels_spans_split = labels_spans.split(",")

    spans = []
    for i,token in enumerate(labels_spans_split):
        # Label instance found
        if token.startswith(label):
            start = int(labels_spans_split[i + 1])
            end = int(labels_spans_split[i + 2])
            # Append labeled span of spans for this label instance
            spans.append((start, end))

    # Collapse nested and overlapping labels to not run into problems with spans of already changed parts of strings
    spans_final = []
    delete = []

    for i, token_tup in enumerate(spans):

        if token_tup in delete:
            continue
        start = token_tup[0]
        end = token_tup[1]
        merge = [[start], [end]]
        for j, cur_tup in enumerate(spans[i+1:]):
            # Start or end within other token?
            if (cur_tup[0]) >= start and (cur_tup[0]) <= end or (cur_tup[1]) >= start and (cur_tup[1]) <= end:
                merge[0].append(cur_tup[0])
                merge[1].append(cur_tup[1])

        # Merge the resulting tupels and delete individual spans
        spans_final.append((min(merge[0]),max(merge[1])))
        for span in zip(merge[0], merge[1]):
            delete.append(span)

    text_parts = []
    start = 0

    # Sort final spans
    spans_final = sorted(spans_final, key=lambda x: x[1])
    assert all(spans_final[i][0] <= spans_final[i+1][0] for i in range(len(spans_final)-1))

    for span in sorted(spans_final):
        text_parts.append(text[start:span[0]])
        text_parts.append(re.sub(r'(\b\w+\b)', r'\1{}'.format(mark), text[span[0]:span[1]]))
        start = span[1]

    # Unlabeled instances can be returned as is
    if len(spans_final) == 0:
        return text

    text_parts.append(text[spans_final[-1][1]:])

    return "".join(text_parts)


def filter_border(df, label):
    """
    Method that filters out instances of the label class that are marked as border cases
    (with attributes 'narr', 'prag', 'metaph', 'border', 'non-fact' or 'ambig').

    :param df: the dataframe of instances
    :param label: the label to consider
    :return: the filtered dataframe
    """
    # Check whether there are no instances of label that don't have attributes
    filter_condition = (lambda row: any(lbl_sp.startswith(label) and row["attributes"].split(",")[i//3]=="-" for i,lbl_sp in enumerate(row["labels_spans"].split(","))))
    return(df[df.apply(filter_condition, axis=1)])


# Execution
if __name__ == "__main__":
    extract_corpus("data/corpus/ErzaehltextkorpusMarked/")
    get_corpus_statistics("corpusExtracted")
