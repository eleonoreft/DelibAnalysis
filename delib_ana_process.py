#!/usr/local/bin/python3
"""DelibAnalysis Processes

These functions implements the various processes - Prediction, Model
Generation, Model Generation and Prediction, Testing. Prediction requires the
use of a model that has already been saved. Both processes involving generation
have the option to save els to local storage. Testing process will generate
statistics on a generated model. There is also a helper function that will run
predictions on multiple unlabelled data files in a directory and output the
results in a subdirectoty. The predictions will all be done from the same
stored model.

Package: DelibAnalysis
Version: 2.0
Authors: Eleonore Fournier-Tombs and Curtis Hendricks
Source: https://github.com/eleonoreft/DelibAnalysis
Copyright: Attribution-NonCommercial-ShareAlike CC BY-NC-SA
https://creativecommons.org/licenses/by-nc-sa/4.0/
Contact: eleonore.fournier-tombs@mail.mcgill.ca

"""

import delib_ana_utils as utils
import delib_ana_forest as forest
import delib_ana_modelstore as storage
import os

from sklearn.model_selection import train_test_split


def predict_process(input_unlabelled, indic, model_file_type, model_file_name,
                    verbose=True):
    """Predict the indicator field in a dataset.

    Use a stored model to predict the trained indicator field in an unlabelled
    dataset.

    Arguments:
        input_unlabelled {str} -- filename and location of the labelled
            dataset.
        indic {str} -- the indicator that will be predicted by the model.
        model_file_type {str} -- the method used to store/retrieve the
            model.
        model_file_name {str} -- the name of the file storing the model.

    Keyword Arguments:
        verbose {bool} -- print progress results to standard output
            (default: {True})

    Returns:
        DataFrame -- a Pandas DataFrame containing the dataset with the
            indicator field labelled.
    """

    unlabelled_data = utils.import_unlabelled_data(input_unlabelled)

    if model_file_type == 'joblib':
        model, vecs, _ = storage.joblib_retrieve(model_file_name, verbose=True)
    if model_file_type == 'pickle':
        model, vecs, _ = storage.unpickle(model_file_name, verbose=True)

    if verbose:
        print('Model retrieved from', model_file_name, '.')

    return forest.f_class_predict(unlabelled_data, indic, vecs['vec_combo'],
                                  model)


def generate_process(input_label_data, indic, vocab, tag='', store_name='',
                     store_type='joblib', train_split=0.7, r_state=33,
                     verbose=True):
    """Create a classifier on an indicator an store it to file.

    Arguments:
        input_label_data {str} -- filename and location of the labelled
            dataset.
        indic {str} -- name of the indicator classifer will predict.
        vocab {list} -- vocabulary list related to the indicator.

    Keyword Arguments:
        tag {str} -- overall title for the dataset (default: {''})
        store_name {str} -- prefix to be given to the file name.
            (default: {''})
        store_type {str} -- method to be used to create/retrieve the the model.
            (default: {'joblib'})
        train_split {float} -- the percentage of the labelled dataset to be
            used for training.
            (default: {0.7})
        r_state {int} -- number to initialize train/text creation function.
            (default: {33})
        verbose {bool} -- print descriptive process output to standard output.
            (default: {True})
    """

    labelled_data = utils.import_label_data(input_label_data)
    train, test = train_test_split(labelled_data,
                                   train_size=train_split,
                                   random_state=r_state)
    vecs = forest.make_vectorizers(train, vocab)
    indicator_features = forest.get_feats(train, vecs["vec_combo"])

    forest_classifier = forest.f_class_train(indicator_features, train, indic)

    if verbose:
        print('Classifier on', indic, 'created.')

    store_model(store_type, store_name, forest_classifier, vecs, indic, tag,
                verbose)


def gen_predict_process(input_label_data, input_unlabelled, indic, vocab, tag,
                        train_split=0.7, r_state=33, store=True,
                        store_name='', store_type='joblib', verbose=True):
    """Create a classifier and predict the indicator in an ulabelled dataset

    Arguments:
        input_label_data {str} -- filename of labelled dataset
        input_unlabelled {str} -- filename of unlabelled dataset
        indic {str} -- name of the indicator classifer will predict
        vocab {list} -- vocabulary list related to the indicator
        tag {str} -- name of the datasets being processed

    Keyword Arguments:
        train_split {float} -- the percentage of the labelled data to be used
            for training (default: 0.7)
        r_state {int} -- number to initialize train/text creation function
            (default: 33)
        store {bool} -- True if created classier is to be saved to local
            storage (default: False)
        store_name {str} -- prefix to be given to the file name (default: '')
        store_type {str} -- method to be used to create the storage file
            (default: 'joblib')
        verbose {bool} -- print descriptive process output to standard output.
            (default: True)

    Returns:
        DataFrame -- dataset with the predicted indicator values
    """

    labelled_data = utils.import_label_data(input_label_data)
    train, test = train_test_split(labelled_data,
                                   train_size=train_split,
                                   random_state=r_state)
    vecs = forest.make_vectorizers(train, vocab)
    indicator_features = forest.get_feats(train, vecs["vec_combo"])

    forest_classifier = forest.f_class_train(indicator_features, train, indic)

    if verbose:
        print('Classifier on', indic, 'created.')

    if store:
        store_model(store_type, store_name, forest_classifier, vecs, indic,
                    tag, verbose)

    unlabelled_data = utils.import_unlabelled_data(input_unlabelled)

    return forest.f_class_predict(unlabelled_data, indic, vecs['vec_combo'],
                                  forest_classifier)


def testing_process(input_label_data, indic, vocab, tag, store_name='',
                    store_type='joblib', train_split=0.7, r_state=33,
                    store=False):
    """Special testing process for classifier creation

    Create a classifier and print the results of performance tests to standard
    output.

    Arguments:
        input_label_data {str} -- filename of labelled dataset.
        indic {str} -- name of the indicator classifer will predict.
        vocab {list} -- vocabulary list related to the indicator.
        tag {str} -- overall name of the datasets being processed.
        q_feat {list} -- quantitative features in the dataset.

    Keyword Arguments:
        train_split {float} -- the percentage of the labelled data to be used
            for training.
            (default: 0.7)
        r_state {int} -- number to initialize train/text creation function.
            (default: 33)
        store {bool} -- True if created classier is to be saved to local
            storage.
            (default: False)
        store_name {str} -- prefix to be given to the file name.
            (default: '')
        store_type {str} -- method to be used to create the storage file.
            (default: 'joblib')
    """

    dte_txt = "-" + utils.curr_dte_txt(1)
    tm_txt = "-" + utils.curr_tm_txt()
    feat_out = tag + "-TopFeatures-" + indic + dte_txt + tm_txt + ".csv"
    loc_labelled = tag + "-Labelled-" + indic + dte_txt + tm_txt + ".csv"
    report_out = tag + "-Report-" + indic + dte_txt + tm_txt + ".txt"

    labelled_data = utils.import_label_data(input_label_data)
    train, test = train_test_split(labelled_data, train_size=train_split,
                                   random_state=r_state)
    vecs = forest.make_vectorizers(train, vocab)
    indicator_features = forest.get_feats(train, vecs["vec_combo"])

    forest_classifier = forest.f_class_train(indicator_features, train, indic)

    if store:
        store_model(store_type, store_name, forest_classifier, vecs, indic,
                    tag, True)

    top_parameters = forest.get_top_params(forest_classifier,
                                           vecs['vec_combo'], 1000)

    top_parameters.to_csv(feat_out)
    print('Parameter priority list file created:', feat_out)

    print(top_parameters.head(n=20))
    output = forest.f_class_predict_compare(test, vecs['vec_combo'],
                                            forest_classifier, indic)
    output.to_csv(loc_labelled)
    print('Labelled comparisons file created:', loc_labelled)

    output_str = forest.str_class_report(indic, output)
    with open(report_out, 'w') as f_out:
        f_out.write(output_str)
    print('Testing report file created:', report_out)

    utils.tst_print("Output string", output_str)


def dir_predict_process(dir_path, indic, file_name, file_type,
                        output_dir="results/", master=False, tag='',
                        verbose=True):
    """Predict the indicator field for number of datasets in a directory.

    The filetype for the unlabelled datasets is CSVs with the fields 'Speaker'
    and speech.

    Arguments:
        dir_path {str} -- directory where the data files are stored
        indic {str} -- name of the indicator that will be predicted
        file_name {str} -- name of the file that stores the model

    Keyword Arguments:
        file_type {str} -- type of storage method used to store the model.
                           (default: {'joblib'})
        output_dir {str} -- name of the sub folder to store the results.
            (default: {"results/"})
        master {bool} -- if true, all the results will be stored in one
            DataFrame and returned.
            (default: {False})
        tag {str} -- overall name of the datasets being processed.
            (default: '')
        verbose {bool} -- if true, progress text is shown to default output.
            (default: {False})

    Returns:
        DataFrame -- DataFrame containing the combined results. None if master
                     isn't selected
    """

    if master:
        master_df = forest.pd.DataFrame()
    with os.scandir(dir_path) as f_it:
        for a_file in f_it:
            if a_file.name.endswith('.csv') and a_file.is_file():
                pth = dir_path + a_file.name
                if verbose:
                    print('Processing:', pth + '....')
                new_data = predict_process(pth, indic, file_type,
                                           file_name)
                if tag:
                    result_fname = dir_path + output_dir + tag + '-' + \
                        indic + '_' + a_file.name
                else:
                    result_fname = dir_path + output_dir + indic + '_' \
                        + a_file.name
                new_data.to_csv(result_fname)
                if master:
                    master_df = master_df.append(new_data)
        if verbose:
            print('Outputs are saved in:', dir_path + output_dir)

    if master:
        return master_df
    else:
        return None


def store_model(store_type, store_name, classifier, vecs, indic, tag, verbose):
    if store_type == 'joblib':
        storage.joblib_store(classifier, vecs, indic, tag=tag, name=store_name,
                             verbose=verbose)
    if store_type == 'pickle':
        storage.pickling(classifier, vecs, indic, tag=tag, name=store_name,
                         verbose=verbose)
