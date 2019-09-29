#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Thesis: S(peech)T(hought)W(riting)R(epresentation) recognition
author: Luise Schricker

python-3.4

imbalanced-learn-0.4.3
germalemma-0.1.1
numpy-1.13.3
pandas-0.21.0
sklearn-0.20.2

This file contains classifiers for direct, indirect, free_indirect and reported STWR, as well as speech, thought and writing.
"""
import argparse
from functools import partial
import json
import os
import re
import sys
import warnings

from imblearn.over_sampling import RandomOverSampler, SMOTE
from germalemma import GermaLemma
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from data_handling import get_training_set, mark_labeled_words, augment_data, get_labels_stw
from feature_extraction import SimpleBaseline, STWRFeatureExtractor, NLP
from postprocessing import postprocess_spans
from preprocessing import segment_tokenize
from visualization import visualize_html

# Surpress Warnings
warnings.filterwarnings("ignore")

# Dicitionary for mapping keywords to methods for countering class imbalance
DATA_AUGMENT = {
    'oversampling': RandomOverSampler(random_state=0).fit_resample,
    'SMOTE': SMOTE(random_state=0).fit_resample,
    'augmentation': {
        'direct': partial(augment_data, cl='direct'),
        'indirect': partial(augment_data, cl='indirect'),
        'free_indirect': partial(augment_data, cl='free_indirect'),
        'reported': partial(augment_data, cl='reported')
    }
}

### Classifiers
class STWRecognizer():
    """
    Class that has methods for automatically annotating STWR in narrative text.
    """

    def __init__(self, feature_extractor=SimpleBaseline(), model='random_forest', mode='test', train_path=None, split=0.25, augment_data='oversampling'):
        """
        :param feature_extractor: an instance of the Feature Extractor to be used.
        :param model: type of ML model to be used, should be one of 'random_forest', 'svm', 'neural'.
        :param mode: one of 'test', 'train', 'eval'. train-mode indicates that the ML models should be retrained,
                    eval-mode allows for evaluation, test-mode is for annotating raw text.
        :param train_path: Needed for train- and eval-mode, path to training data.
        :param split: Percentage of data to be used as held out dataset.
        :param augment_data: Keyword for optional method/methods to be used to counter class imbalance when training.
                            One of 'oversampling', 'SMOTE', 'augmentation', None.
        """

        # Check validity of parameters
        if mode not in ['train', 'test', 'eval']:
            print(
                "ERROR: Invalid mode selected. Please select one of 'train', 'test', 'eval'.")
            sys.exit(1)

        if model not in ['random_forest', 'svm', 'neural']:
            print(
                "ERROR: Invalid model selected. Please select one of 'random_forest', 'svm', 'neural'.")
            sys.exit(1)

        if augment_data not in ['oversampling', 'SMOTE', 'augmentation', None]:
            print(
                "ERROR: Invalid keyword for augment_data selected. Please select one of 'oversampling', 'SMOTE', 'augmentation' or leave the parameter out.")
            sys.exit(1)

        # Set up attributes
        self.mode = mode
        self.model = model
        self.Fe = feature_extractor
        if not self.Fe.sequence_features:
            self.sequence_features = False

        self.clfs = {}
        self.classes = ['direct',
                        'indirect',
                        'free_indirect',
                        'reported']

        # Set up training and evaluation parameters
        self.train_path = train_path
        self.split = split
        # Method to use to counter the class imbalance problem when training
        self.augment_data = augment_data

        # Set up random state
        self.random_state = 5

        # Set up model definitions
        self.models = {
            'random_forest': {
                'classifier': RandomForestClassifier,
                'parameter': {
                    'direct': {'random_state': self.random_state,
                              'n_estimators': 50,
                              'max_depth': 30,
                              'min_samples_leaf': 1},
                    'indirect': {'random_state': self.random_state,
                               'n_estimators': 500,
                               'max_depth': 30,
                               'min_samples_leaf': 3},
                    'free_indirect': {'random_state': self.random_state,
                               'n_estimators': 100,
                               'max_depth': 50,
                               'min_samples_leaf': 3},
                    'reported': {'random_state': self.random_state,
                               'n_estimators': 500,
                               'max_depth': 50,
                               'min_samples_leaf': 10}
                }
            },
            'svm': {
                'classifier': SVC,
                'parameter': {
                    'direct': {'random_state': self.random_state,
                              'kernel': 'poly',
                              'degree': 5,
                              'tol': 0.01},
                    'indirect': {'random_state': self.random_state,
                              'kernel': 'poly',
                              'degree': 5,
                              'tol': 0.01},
                    'free_indirect': {'random_state': self.random_state,
                              'kernel': 'poly',
                              'degree': 5,
                              'tol': 0.01},
                    'reported': {'random_state': self.random_state,
                              'kernel': 'poly',
                              'degree': 5,
                              'tol': 0.01}
                }
           },
            'neural': {
                'classifier': MLPClassifier,
                'parameter': {
                    'direct': {'solver':'adam',
                              'early_stopping': True,
                              'tol': 0.01,
                              'hidden_layer_sizes': (1024, 128),
                              'activation': 'tanh',
                              'verbose': True,
                              'random_state': self.random_state},
                    'indirect': {'solver':'adam',
                              'early_stopping': True,
                              'tol': 0.01,
                              'hidden_layer_sizes': (512, 32),
                              'activation': 'relu',
                              'verbose': True,
                              'random_state': self.random_state},
                    'free_indirect': {'solver':'adam',
                              'early_stopping': True,
                              'tol': 0.01,
                              'hidden_layer_sizes': (1024, 32),
                              'activation': 'relu',
                              'verbose': True,
                              'random_state': self.random_state},
                    'reported': {'solver':'adam',
                              'early_stopping': True,
                              'tol': 0.01,
                              'hidden_layer_sizes': (128, 32),
                              'activation': 'logistic',
                              'verbose': True,
                              'random_state': self.random_state}
                }
            }
        }

        # Set up dictionary for majority classes speech, thought, writing
        self.majority_classes = {
            'direct': None,
            'indirect': None,
            'free_indirect': None,
            'reported': None
        }

        # Load reporting word list
        self.stw_words = pd.read_excel("data/stw_words/stw_words_brunner2015.xls")

        # Retrain the models if mode is train
        if mode=='train':
            if self.train_path is None:
                print(
                    "ERROR: Missing path to training data. Please give the path to the training data as parameter 'train_path'.")
                sys.exit(1)
            else:
                for clf_class in self.classes:
                    self.clfs[clf_class] = self.train(clf_class, cross_val=False, augment_data=self.augment_data)

                # Save majority_classes for speech, thought, writing classification
                with open("models/majority_classes.json", "w", encoding='utf-8') as f_out:
                    json.dump(self.majority_classes, f_out, indent=4)

        else:
            if mode == 'eval':
                if self.train_path is None:
                    print(
                        "ERROR: Missing path to training data. Please give the path to the training data as parameter 'train_path'.")
                    sys.exit(1)

            print("Loading models ....")
            for clf_class in self.classes:
                # Load the existing models
                try:
                    self.clfs[clf_class] = joblib.load('models/{}/{}.clf'.format(model, clf_class))
                except IOError:
                    print(
                        "ERROR: No trained models. Please set 'mode' to train.")
                    sys.exit(1)

            # Load majority_class information for speech, thought, writing classification
            with open("models/majority_classes.json", "r", encoding='utf-8') as f:
                self.majority_classes = json.load(f)

            print("Done.\n")


    def train(self, clf_class, cross_val=False, augment_data='oversampling'):
        """
        Train a binary classifier with the given ML technique (model) to be used in classification
        of clf_class instances.

        :param clf_class: label of the positive class instances.
        :param cross_val: if True, print evaluation with stratified 10-fold cross validation.
        :param augment_data: keyword for optional method to counter class imbalance within the training data.
        :return: the trained classifier.
        """
        print("Training the {} classifier for label {}.\n".format(self.model, clf_class))

        # Load training data
        data = get_training_set(self.train_path, self.Fe, label=clf_class, original_labels=True)
        X = data.iloc[:, :-1]
        y = np.ravel(data.iloc[:, -1:])

        # Get the original labels for speech, thought, writing classification
        y_orig = X.iloc[:, -2:-1]
        X = X.iloc[:, :-2]

        # For SVM the features should be scaled for efficiency reasons
        if self.model == "svm":
            scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X)
            X = scaling.transform(X)
            X = pd.DataFrame(X)

        # Use a wrapper for training the classifier, in order to trigger methods for countering class imbalance
        clf = self.models[self.model]['classifier'](**self.models[self.model]['parameter'][clf_class])
        clf_wrapped = CLFWrapper(clf)

        print("\nTraining...")

        # Get method for countering class imbalance if respective parameter is given
        if augment_data in ['oversampling', 'SMOTE']:
            augment_method = DATA_AUGMENT[augment_data]

        elif augment_data == 'augmentation':
            augment_method = DATA_AUGMENT[augment_data][clf_class]

        else:
            augment_method = None

        if cross_val:
            # For cross validation, use stratified train-test split in order to have a hold-out test-set which is NOT used as a dev set in cross validation
            X_train, _, y_train, _ = train_test_split(X, y, test_size=self.split, random_state=self.random_state, stratify = y)

            # Stratified 10-fold cross validation, treatment of imbalanced data sets by oversampling, data augmentation etc. is triggered via the fit_params parameter
            recall = cross_val_score(clf_wrapped, X_train, y_train, cv=10, scoring='recall', fit_params={'augment_method': augment_method})
            precision = cross_val_score(clf_wrapped, X_train, y_train, cv=10, scoring='precision', fit_params={'augment_method': augment_method})
            f1 = cross_val_score(clf_wrapped, X_train, y_train, cv=10, scoring='f1', fit_params={'augment_method': augment_method})

            # Precision, Recall, F1
            print("Scores on training set with 10-fold cross validation for class {}: Precision {}, Recall {}, F1 {}".format(clf_class, precision.mean(), recall.mean(), f1.mean()))

        # After evaluation fit classifier on all available training data
        clf_wrapped.fit(X, y, augment_method=augment_method)

        # Get the trained classifier
        clf = clf_wrapped.clf

        # Check that directory exists
        directory = os.getcwd() + "/models"
        if not os.path.exists(directory):
            os.makedirs(directory)

        print("Saving the trained classifier...")
        # Save the trained classifier
        joblib.dump(clf, 'models/{}/{}.clf'.format(self.model, clf_class))

        # Get the majority class (one of speech, thought, writing) from data for clf_class
        self.majority_classes[clf_class] = self.get_max_type(y_orig, clf_class)

        print("Done.\n")

        return (clf)


    def evaluate(self, span_detection=False, save=False):
        """
        Method for evaluating the classifiers on the held out test sets.

        :param span_detection: If True, use postprocessing method for better span detection.
        :param save: if True the clfs trained for evaluation are saved for further inspection
        """
        # Evaluation is only possible in train or eval mode
        if self.mode not in ['train', 'eval']:
            print("ERROR: Evaluation is only possible in train and eval mode.")
            sys.exit(1)

        majority_classes = {
            'direct': None,
            'indirect': None,
            'free_indirect': None,
            'reported': None
        }

        # Loop through classes and evaluate
        for clf_class in self.classes:

            # Load training data for respective class to extract the train and test set,
            # as well as the original labels and texts to compute the accuracy on word level
            data = get_training_set(self.train_path, self.Fe, label=clf_class, original_labels=True)
            X = data.iloc[:, :-1]
            y = np.ravel(data.iloc[:, -1])
            # Get the correct train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split, random_state=self.random_state, stratify = y)

            # Get the original labels and text for span evaluation after the split
            y_orig_train = X_train.iloc[:, -2:]
            X_train = X_train.iloc[:, :-2]
            y_orig_test = X_test.iloc[:, -2:]
            X_test = X_test.iloc[:, :-2]

            # For SVM the features should be scaled for efficiency reasons
            if self.model == "svm":
                scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
                X_train = scaling.transform(X_train)
                X_train = pd.DataFrame(X_train)
                X_test = scaling.transform(X_test)
                X_test = pd.DataFrame(X_test)

            # Use a wrapper for training the classifier, in order to trigger methods for countering class imbalance
            clf = self.models[self.model]['classifier'](**self.models[self.model]['parameter'][clf_class])
            clf_wrapped = CLFWrapper(clf)

            print("Training the {} classifier for label {}.\n".format(self.model, clf_class))

            # Get method for countering class imbalance if respective parameter is given
            if self.augment_data in ['oversampling', 'SMOTE']:
                augment_method = DATA_AUGMENT[self.augment_data]

            elif self.augment_data == 'augmentation':
                augment_method = DATA_AUGMENT[self.augment_data][clf_class]

            else:
                augment_method = None

            # Fit classifier on training data
            clf_wrapped.fit(X_train, y_train, augment_method=augment_method)

            # Save the classifier if flag indicates this
            if save:
                # Save the trained classifier
                self.clfs[clf_class] = clf
                joblib.dump(clf, 'models/{}/{}.clf'.format(self.model, clf_class))

            y_pred = clf.predict(X_test)

            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            print("Classifier for label {} on test set: Precision {}, Recall {}, F1 {}".format(clf_class, precision, recall, f1))

            majority_classes[clf_class] = self.get_max_type(y_orig_train.iloc[:, -2:-1], clf_class)
            y_pred_stw = [self.annotate_stw(t[0], clf_class, majority_classes=majority_classes) for ind, t in enumerate(y_orig_test.iloc[:, -1:].values.tolist()) if y_pred[ind]]
            y_true_s, y_true_t, y_true_w = get_labels_stw([y[0] for ind, y in enumerate(y_orig_test.iloc[:, -2:-1].values.tolist()) if y_pred[ind]], clf_class)

            for type in ['speech', 'thought', 'writing']:
                if type == 'speech':
                    y_true_stw = y_true_s
                elif type == 'thought':
                    y_true_stw = y_true_t
                else:
                    y_true_stw = y_true_w

                y_pred_type = [int(y == type) for y in y_pred_stw]
                precision = precision_score(y_true_stw, y_pred_type)
                recall = recall_score(y_true_stw, y_pred_type)
                f1 = f1_score(y_true_stw, y_pred_type)
                print("Classification for label {} on predictions for class {} on test set: Precision {}, Recall {}, F1 {} (Count instances: {})".format(type, clf_class, precision,
                                                                                                   recall, f1, sum(y_true_stw)))

            # Evaluate accuracy of span prediction
            y_pred_test = y_orig_test.copy()
            # Get full spans before postprocessing
            y_pred_test.iloc[:, 0] = ["{},0,{}".format(clf_class, str(len(y_orig_test.iloc[i,1]))) if int(y_hat) == 1 else "" for i, y_hat in enumerate(y_pred)]

            # Do span detection if chosen
            if span_detection:
                y_pred_test.iloc[:, 0] = [postprocess_spans(row, cl=clf_class) for _, row in y_pred_test.iterrows()]

            # Mark gold and predicted labeled words with different signs
            marked_text_gold = list(map(lambda segment: mark_labeled_words(segment[1], segment[0], clf_class),
                                        y_orig_test.values))
            marked_text_predicted = list(map(lambda segment: mark_labeled_words(segment[1], segment[0], clf_class, mark='#'),
                                        y_pred_test.values))

            num_words_correctly_marked_total = 0
            num_words_incorrectly_marked_total = 0
            num_words_total = 0
            num_words_correctly_marked_correct_labels = 0
            num_words_incorrectly_marked_correct_labels = 0
            num_words_correct_labels = 0

            for i, gold_segment in enumerate(marked_text_gold):
                tokens_pred = marked_text_predicted[i].split()

                tokens = gold_segment.split()
                num_words_total += len(tokens)
                len_gold = len([token for token in tokens if token.endswith('$')])

                # Correctly identified instances
                if len_gold > 0 and int(y_pred[i]) == 1:
                    num_words_correct_labels += len(tokens)

                for j,token in enumerate(tokens_pred):
                    if token.endswith('#'):
                        if tokens[j].endswith('$'):
                            num_words_correctly_marked_total += 1
                            num_words_correctly_marked_correct_labels += 1
                        else:
                            if len_gold > 0 and int(y_pred[i]) == 1:
                                num_words_incorrectly_marked_correct_labels += 1
                            num_words_incorrectly_marked_total += 1

                    else:
                        if tokens[j].endswith('$'):
                            num_words_incorrectly_marked_total += 1
                            if int(y_pred[i]) == 1:
                                num_words_incorrectly_marked_correct_labels += 1
                        else:
                            num_words_correctly_marked_total += 1
                            if len_gold > 0 and int(y_pred[i]) == 1:
                                num_words_correctly_marked_correct_labels += 1

            print("Word-level accuracy all instances: {}% of total words correctly labeled, {}% of total words incorrectly labeled.".format(
                round((num_words_correctly_marked_total / num_words_total) * 100, 2),
                round((num_words_incorrectly_marked_total / num_words_total) * 100, 2)
            ))

            print("Word-level accuracy within correctly identified instances: {}% of words within labeled instances correctly labeled, {}% of words within labeled instances incorrectly labeled.\n".format(
                round((num_words_correctly_marked_correct_labels / num_words_correct_labels) * 100, 2),
                round((num_words_incorrectly_marked_correct_labels / num_words_correct_labels) * 100, 2)

            ))

        return

    def annotate(self, text, verbose=False, filename_output="text_annotated", postprocess=False, html=False):
        """
        Method for annotating raw text with spans of direct, indirect, free_indirect and reported STWR.

        :param text: the raw text as a string.
        :param verbose: If True, print the annotated text as well as saving it to the output file.
        :param filename_output: Optional filename for the output. If none is given, the output will be written to text_annotated.tsv.
        :param postprocess: If True, do span postprocessing.
        :param html: If True an html file which visualizes the annotation is produced.
        """

        # Tokenize text
        text_tokenized = segment_tokenize(text)

        len_before = 0
        len_characters_before = 0

        # Classify and save in output dataframe
        output = pd.DataFrame(index=np.arange(0, len([token for seg in text_tokenized for token in seg])), columns=["token", "direct", "indirect", "free_indirect", "reported"])

        backlog = []
        labels_html = []

        print("Preprocessing and classifying input... This might take a while.")

        # Predict segment by segment and update backlog
        for ind, segment in enumerate(text_tokenized):

            if ind > 0 and len(text_tokenized[ind-1]) > 0:
                len_characters_before += (segment[0].idx - (text_tokenized[ind-1][-1].idx + len(text_tokenized[ind-1][-1].text)))

            # print progress bar for longer texts
            if len(text_tokenized) > 10:
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('=' * round(ind/(len(text_tokenized)/20)), round(ind/(len(text_tokenized)/100))))
                sys.stdout.flush()

            # Extract features, pass tokenized text
            original_text = text[segment[0].idx:(segment[-1].idx + len(segment[-1].text))]
            text_features, backlog = self.Fe.transform(segment, original_text=original_text, backlog=backlog)

            # Adapt backlog: backlog stores last ten classifications in the first ten positions
            backlog[0:10] = backlog[1:10] + [""]

            for clf_class in self.clfs:

                clf = self.clfs[clf_class]
                pred = clf.predict([text_features])

                label = pred[0]

                start = 0
                end = len(original_text)

                if label == 1:
                    # Adapt backlog; use '#' as standin for spans
                    if backlog[9] == "":
                        backlog[9] = ",".join([clf_class, '#', '#'])
                    else:
                        backlog[9] = ",".join([backlog[9], clf_class, '#', '#'])

                    # Second annotation layer: speech, thought, writing
                    s_t_w = self.annotate_stw(original_text, clf_class, majority_classes=self.majority_classes)

                    # Get full span before postprocessing
                    span_label = "{}_{},0,{}".format(clf_class, s_t_w, str(len(original_text)))

                    if postprocess:
                        span_postprocessed = postprocess_spans([span_label, original_text], cl=clf_class)
                    else:
                        span_postprocessed = span_label

                    start = int(span_postprocessed.split(",")[1]) + len_characters_before
                    end = int(span_postprocessed.split(",")[2]) + len_characters_before

                    labels_html.append("{}_{},{},{}".format(clf_class, s_t_w, start, end))

                # Split up multiple newlines
                segment_cur = []
                for token in segment:
                    if token.text.startswith('\n'):
                        for idx_c, c in enumerate(token.text):
                            # create a dummy object
                            token_new = dummy_token(c, token.idx)
                            segment_cur.append(token_new)

                            if idx_c > 0:
                                output = output.append(pd.Series(name = len(output)), ignore_index=True)
                    else:
                        segment_cur.append((token))
                segment = segment_cur

                for j,token in enumerate(segment):
                    # Replace newlines with <p>
                    if token.text == '\n':
                        output.loc[output.index[len_before + j], 'token'] = '<p>'

                    else:
                        output.loc[output.index[len_before + j], 'token'] = token.text

                    # Encoding via IOB scheme: B-tag - beginning, I-tag - inside, O-tag outside
                    if (token.idx) == start and label==1:
                        label_cur = "B-{}-{}".format(clf_class, s_t_w)
                    elif (token.idx) > start and (token.idx + len(token)) <= end and label == 1:
                        label_cur = "I-{}-{}".format(clf_class, s_t_w)
                    else:
                        label_cur = "O"

                    output.loc[output.index[len_before + j], clf_class] = label_cur
            len_before += len(segment)
            len_characters_before += len(original_text)
        print("\nDone.\n")

        if verbose:
            print(output)
        # Save annotated text in file
        output.to_csv(path_or_buf="{}.tsv".format(filename_output), sep='\t', index=False)

        # Produce the html file
        if html:
            visualize_html(text, labels_html, "{}".format(filename_output))

        return

    def annotate_stw(self, t, clf_class, majority_classes=None):
        """
        Method for annotating a segment with one of the classes speech, thought or writing given
        the STWR classification clf_class.

        :param t: The text of the segment.
        :param clf_class: One of direct, indirect, free_indirect, reported. The predicted class for t.
        :param majority_classes: A dictionary containing the majority classes (one of speech, thought or writing)
                                for each STWR class.
        :return: One of speech, thought or writing; the annotation for t.
        """
        # Get the stored majority classes if no other are given
        if not majority_classes:
            majority_classes = self.majority_classes

        # Direct and free_indirect should always be classified by majority classes as reporting words are more
        # likely to appear outside of segments of these classes.
        if clf_class in ['direct', 'free_indirect']:
            return majority_classes[clf_class]

        # For the other types check for reporting words with unambiguous type else use majority class
        doc = NLP(t)
        # Get lemmata with germalemma as spacy is not good at this
        lemmatizer = GermaLemma()

        lemmata = []
        for token in doc:
            if token.pos_ == "VERB":
                lemmata.append(lemmatizer.find_lemma(token.text, 'V'))

            elif token.pos_ == "NOUN":
                lemmata.append(lemmatizer.find_lemma(token.text, 'N'))

        if len(lemmata) > 0:
            stw_words_t = pd.concat([self.stw_words[self.stw_words["Word"].str.contains(r'\b{}\b'.format(re.escape(lemma)))] for lemma in lemmata], axis=0, ignore_index=True)
        else:
            stw_words_t = []

        if len(stw_words_t) == 1:
            if stw_words_t["Type"][0] in ["speech", "thought", "writing"]:
                return stw_words_t["Type"][0]
            else:
                return majority_classes[clf_class]

        else:
            return majority_classes[clf_class]

    def inspect_features(self):
        """
        Method for inspecting the features of the Random Forest Classifiers by importance.
        """
        features = self.Fe.feature_names

        for clf_class in self.classes:

            clf = self.clfs[clf_class]

            importances = clf.feature_importances_

            print("Features of the Random Forest classifier for class {} by importance:\n*******************\n".format(clf_class))

            for tup in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
                print("Feature: {}, Importance: {}".format(tup[0], tup[1]))

            print("\n")

    def error_analysis(self):
        """
        Method for printing incorrect classifications for manual error inspection.
        """
        # Error Analysis is only possible in train or eval mode
        if self.mode not in ['train', 'eval']:
            print("ERROR: Error analysis is only possible in train and eval mode.")
            sys.exit(1)

        # Loop through classes and print incorrect classifications
        for clf_class in self.classes:
            print("Incorrect classifications for class {}:\n".format(clf_class))

            # Load training data for respective class to extract the test set
            data = get_training_set(self.train_path, self.Fe, label=clf_class, original_labels=True)
            X = data.iloc[:, :-1]
            y = np.ravel(data.iloc[:, -1])
            # Get the correct train-test split
            _, X_test, _, y_test = train_test_split(X, y, test_size=self.split,
                                                                random_state=self.random_state, stratify=y)

            # Get the original labels and text
            y_orig_labels_test = X_test.iloc[:, -2:-1]
            y_orig_text_test = X_test.iloc[:, -1:]
            X_test = X_test.iloc[:, :-2]

            clf = self.clfs[clf_class]
            # Get the predictions
            y_pred = clf.predict(X_test)

            for orig, text, pred in zip(y_orig_labels_test.values, y_orig_text_test.values, y_pred):
                orig = orig[0]
                text = text[0]
                orig_split = orig.split(",")

                if pred != any([label.startswith(clf_class) for label in orig_split]):
                    print("Text: {}, original label: {}, predicted label: {}".format(text, orig, pred))

            print("\n")

    def get_max_type(self, y_orig, clf_class):
        """
        Method for getting the majority classes speech, thought, writing from data for a certain clf_class.

        :param y_orig: The original labels.
        :param clf_class: One of direct, indirect, free_indirect, reported; class for which the majority type
                          is to be extracted.
        :return: The majority type, one of speech, thought, writing.
        """
        # Get the majority classes speech, thought, writing from data for clf_class
        stw_types = [l for ind, row in y_orig.iterrows() for l in row.values[0].split(",") if '_' in l]
        stw_types = [l for l in stw_types if l.startswith(clf_class)]
        stw_types = [l.split("_")[-1] for l in stw_types]

        max_count = 0
        max_type = None
        for stw_type in ['speech', 'thought', 'writing']:
            count = stw_types.count(stw_type)
            if count > max_count:
                max_count = count
                max_type = stw_type

        return max_type

### Helper classes
# Wrapper class to allow for changes of the training data within the fit method -> e.g. during cross validation
class CLFWrapper(BaseEstimator):
    """
    Class that wraps a classifier with a fit method, allowing for changes of the training data within
    the fit method. This is meant to allow for methods to counter class imbalance to ONLY be applied to the training
    sets during cross validation.
    """

    def __init__(self, clf):
        """
        :param clf: Classifier to be wrapped
        """
        self.clf = clf


    def fit(self, X, y, augment_method=None):
        """
        Method wrapping the classifiers fitting method. This allows for an optional method to be given via
        the augment_data parameter for countering class imbalance.

        :param X: Training data
        :param y: Labels
        :param augment_method: optional method to counter class imbalance
        :return: The trained classifier
        """
        # Apply method to counter class imbalance if given
        if augment_method:

            X, y = augment_method(X, y)

            # Assert that dataset is balanced now
            assert len([l for l in y if l == 0]) == len([l for l in y if l==1])

        return self.clf.fit(X, y)


    def predict(self, X):
        """
        Dummy method shadowing the predict method of the classifier to be used during cross validation.

        :param X: The data to be predicted.
        :return: The predictions
        """
        return self.clf.predict(X)

class dummy_token():
    """
    Dummy token class to allow splitting a spacy token into two new ones
    """

    def __init__(self, text, idx):
        """
        :param text: The spacy token's text
        :param idx: The spacy token's idx value
        """
        self.text = text
        self.idx = idx

    def __len__(self):
        return len(self.text)


# Execution
if __name__ == "__main__":

    # Enable command line arguments
    parser = argparse.ArgumentParser(
        description='Annotate a raw text file with direct, indirect, free indirect and reported speech, thought and writing.')
    # Arguments for annotation
    parser.add_argument('path', help='Either the path to the file to be annotated or the path to the training/evaluation data.')
    parser.add_argument('--html', action='store_true', default=False, help='Optional parameter. Indicates that an html visualization of the annotation should be produced.')
    parser.add_argument('--post', action='store_true', default=False, help='Optional parameter. Indicates that the span postprocessing step should be executed for the annotations.')
    parser.add_argument('--output', action='store', help='Optional parameter. Name of the output file (without extension).')

    # Arguments for training mode
    parser.add_argument('--train', action='store_true', default=False, help='Optional parameter. Indicates that the system should be retrained.')
    parser.add_argument('--ml', action='store', help='Optional parameter. Type of ML to be used for training. One of \'random_forest\', \'svm\', \'neural\'.')
    parser.add_argument('--augment', action='store', help='Optional parameter. Type of data augmentation to be used for training. One of \'oversampling\', \'SMOTE\', \'augmentation\'.')
    parser.add_argument('--no_sequ', action='store_true', default=False, help='Optional parameter. Indicates that the system should be retrained without sequential label features.')

    # Arguments for evaluation mode
    parser.add_argument('--eval', action='store_true', default=False, help='Indicates that the system should be evaluated.')

    # Parse arguments
    args = parser.parse_args()

    # Check validity of arguments
    if args.ml and args.ml not in ['random_forest', 'svm', 'neural']:
        print("Invalid choice of parameters. Parameter 'ml' can only take one of the values 'random_forest', 'svm', 'neural'.")
        sys.exit(1)

    if args.augment and args.augment not in ['oversampling', 'SMOTE', 'augmentation']:
        print("Invalid choice of parameters. Parameter 'augment' can only take one of the values 'oversampling', 'SMOTE', 'augmentation'.")
        sys.exit(1)

    if args.ml:
        model = args.ml
    else:
        model = 'random_forest'
    if args.augment:
        augment = args.augment
    else:
        augment = 'oversampling'
    if args.no_sequ:
        sequ = False
    else:
        sequ = True
    if args.post:
        postprocess = True
    else:
        postprocess = False

    # Training
    if args.train:
        stwr_clf = STWRecognizer(mode='train', train_path=args.path, model=model, augment_data=augment, feature_extractor=STWRFeatureExtractor(sequence_features=sequ))

    # Evaluate
    if args.eval:
        stwr_clf = STWRecognizer(mode='eval', train_path=args.path, model=model, augment_data=augment, feature_extractor=STWRFeatureExtractor(sequence_features=sequ))
        stwr_clf.evaluate(span_detection=postprocess)

    # Annotate
    if not args.train and not args.eval:
        stwr_clf = STWRecognizer(mode='test', model=model, feature_extractor=STWRFeatureExtractor(sequence_features=sequ))

        if args.html:
            html = True
        else:
            html = False

        if args.output:
            output = args.output
        else:
            output = "text_annotated"

        # Load the text which is to be annotated
        with open(args.path, "r", encoding='utf-8') as f:
            text = f.read()
            stwr_clf.annotate(text, filename_output=output, postprocess=postprocess, html=html)
