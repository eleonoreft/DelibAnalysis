#!/usr/local/bin/python3
"""Module: DelibAnalysis Classifier

Functions related to the traingin and use of the Random Forest Classifier in
DelibAnalysis.

Package: DelibAnalysis
Version: 2.0
Authors: Eleonore Fournier-Tombs and Curtis Hendricks
Source: https://github.com/eleonoreft/DelibAnalysis
Copyright: Attribution-NonCommercial-ShareAlike CC BY-NC-SA
https://creativecommons.org/licenses/by-nc-sa/4.0/
Contact: eleonore.fournier-tombs@mail.mcgill.ca

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import delib_ana_utils as utils

QUANTITATIVE_FEATURES = ['char_count', 'has_respect', 'has_question',
                         'has_question_parent']

# TODO: Write module to test model by accessing features


def get_feats(data, combo_vec):
    """Get list of indication features

    Arguments:
        data {DataFrame} -- training dataset
        combo_vec {TfidfVectorizer} -- Combo of 2 and 3-grams vectorizers
        q_features {list} -- quantitative features

    Returns:
        [nparray] -- 2d numpy arrary of features
    """

    raw = combo_vec.transform(data["cleaned_comment"])
    feats = raw.toarray()
    for f in QUANTITATIVE_FEATURES:
        feats = utils.append_features(feats, data[f].to_numpy())
    return feats


def f_class_train(feats, data, indicator):
    """Train a random forest classifier model.

    Arguments:
        feats {nparray} -- numpy array of features
        data {dataframe} -- training dataset
        indicator {str} -- indicator to train the model on

    Returns:
        RandomForestClassifier -- trained classifier model
    """

    f_classifier = RandomForestClassifier(n_jobs=-1,
                                          criterion="entropy",
                                          warm_start=True,
                                          bootstrap=True)
    y, _ = pd.factorize(data[indicator])
    return f_classifier.fit(feats, y)


def f_class_predict(data, indic, combo_vec, f_classifier):
    """
    Use a trained RandomForestClassifier to predict the disired field in an
    unlabelled dataset

    Arguments:
        data {DataFrame} -- unlabelled data set
        indic {str} -- name of the indicator to be predicted
        combo_vec {nparray} -- combo vectorizer
        q_features {list} -- quantitative features
        f_classifier {RandomForestClassifier} -- trained random forest
        classifier

    Returns:
        DataFrame -- Labelled data set
    """

    test_inidicator_feats = get_feats(data, combo_vec)

    labels = f_classifier.predict(test_inidicator_feats)
    labelled = data.drop(columns=['Unnamed: 0'])
    labelled[indic] = labels

    return labelled


def f_class_predict_compare(data, combo_vec, f_classifier, indicator):
    """
    Used to test the accuracy of a trained RandomForestClassifier by using a
    test dataset where the target field values are known. The known and
    predicted values can then be compared to measure accuracy of the classifier

    Arguments:
        data {DataFrame} -- test data set
        combo_vec {nparray} -- combo vectorizer
        q_features {list} -- quantitative features
        f_classifier {RandomForestClassifier} -- trained random forest
        classifier
        indicator {str} -- name of the target field/property

    Returns:
        DataFrame -- Dataset with actual and predicted target field included.
    """

    test_inidicator_feats = get_feats(data, combo_vec)

    labelled = f_classifier.predict(test_inidicator_feats)
    compare = pd.DataFrame(data={
        "actual": data[indicator],
        "predicted": labelled
    })

    return compare


def make_vectorizers(data_source, vocab=None):
    """Create vectorizers based on datasource

    Arguments:
        data_source {DataFrame} -- training dataset

    Keyword Arguments:
        vocab {list} -- vocabulary list to be included in the vectorizer
        (default: {None})

    Returns:
        dictionary -- dictionary containing created vectorizers
                      key: vec_word -- tfidf vectorizer
                      key: vec_pos -- positional vectorizer
                      key: vec_combo -- combined vec_word and vec_pos
                      vectorizer
    """

    param_dict = {'use_idf': True, 'analyzer': 'word', 'ngram_range': (1, 2),
                  'max_features': 5000}
    if len(vocab) > 0:
        param_dict['vocabulary'] = vocab
    vec_word = TfidfVectorizer(**param_dict)
    # vec_word = TfidfVectorizer(use_idf=True, analyzer='word',
    #                            ngram_range=(1, 2), max_features=5000,
    #                            vocabulary=vocab)
    vec_word.fit_transform(data_source["cleaned_comment"])
    vec_pos = TfidfVectorizer(use_idf=True,
                              analyzer='word',
                              ngram_range=(1, 3),
                              max_features=5000)
    vec_pos.fit_transform(data_source["pos"])

    vec_combo = FeatureUnion([('tfidf', vec_word), ('pos', vec_pos)])

    return {'vec_word': vec_word, 'vec_pos': vec_pos, 'vec_combo': vec_combo}


def get_top_params(classifier, vec_combo, qty):
    """
    Get an ordered list of the highest rated parameter from most to least
    important.

    Arguments:
        classifier {RandomForestClassifier} -- trained classifier
        vec_combo {nparray} -- combo vectorizer
        feats {list} -- list of quantitative features
        qty {int} -- the number of features to return

    Returns:
        Dataframe -- dataframe containing the order list of features
    """

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    vocab = vec_combo.get_feature_names()
    for i in QUANTITATIVE_FEATURES:
        vocab.append(i)
    feat_ordered_df = pd.DataFrame(data=None,
                                   columns=['Feature name', 'Importance'])
    for f in range(0, qty):
        feat_ordered_df.loc[f + 1] = [vocab[indices[f]],
                                      importances[indices[f]]]
    feat_ordered_df['Feature class'] = feat_ordered_df['Feature name'].apply(
        utils.get_tokenizer)
    feat_ordered_df['Feature name'] = feat_ordered_df['Feature name'].apply(
        utils.strip_tokenizer)
    return feat_ordered_df


def str_class_report(indicator, test_data):
    """
    Generate a precision and recall report on a tested trained random forest
    classifier.

    Arguments:
        indicator {str} -- name of the target indicator
        test_data {DataFrame} -- special test dataframe (generated from
                    function: f_class_predict_compare)

    Returns:
        str -- report on precision and recall of tested random forest
        classifier
    """

    output_str = "Results for " + indicator + "\n"
    output_str += str(
        pd.crosstab(test_data['actual'],
                    test_data['predicted'],
                    rownames=['Actual'],
                    colnames=['Predicted']))

    output_str += "\n*Classification Report:\n" + str(
        classification_report(test_data['actual'], test_data['predicted']))

    return output_str
