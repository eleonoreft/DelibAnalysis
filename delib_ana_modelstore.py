#!/usr/local/bin/python3
"""DelibAnalysis Model Storage

DelibAnalysis functions to store and load the trained Random Forest Classifier
models to enable resue. Vectorizers are also stored and optionally the training
data can also be stored. Two mehtods are implemented: Serialization and
deserialization using Python built-in pickle module, and Sci-kit Learn joblib
module.

Package: DelibAnalysis
Version: 2.0
Authors: Eleonore Fournier-Tombs and Curtis Hendricks
Source: https://github.com/eleonoreft/DelibAnalysis
Copyright: Attribution-NonCommercial-ShareAlike CC BY-NC-SA
https://creativecommons.org/licenses/by-nc-sa/4.0/
Contact: eleonore.fournier-tombs@mail.mcgill.ca

"""

import pickle
from sklearn.externals import joblib
from delib_ana_utils import curr_dte_txt

# TODO: add checks for file overwriting


def pickling(model, vectorizer, indic, train_data=None, tag='', name='',
             verbose=False):
    """Store the model using Python's Pickle module.

    'Pickle' the model and vectorizer, and optionally training data, and stores
    the file to the current directory. Filename will have a suffix of
    "_pickle_model.pkl". Prefix is either name provided or current date
    [YYYY-MM-DD].

    Arguments:
        model {model object} -- model object to be stored
        vectorizer {nparray} -- vectorizers created from the training data.
        indic {str} -- name fo the indicator.

    Keyword Arguments:
        train_data {DataFrame} -- training data used to create vectorizers and
            model. (default: {None})
        name {str} -- prefix to be used for file name. (default: {''})
        verbose {bool} -- print process messages. (default: {False})
    """

    suffix = '-' + indic + '_pickle_model.pkl'
    if name:
        outfile = name + suffix
    else:
        outfile = curr_dte_txt(1) + suffix

    if tag:
        outfile = tag + '-' + outfile

    data_dict = {'model': model, 'vectorizer': vectorizer,
                 'training': train_data}

    with open(outfile, 'wb') as f:
        pickle.dump(data_dict, f)

    if verbose:
        print('Model and vectorizers pickled in file,', outfile, '.')
        if train_data:
            print('Training data pickled in file,', outfile, '.')


def unpickle(pkl_file, verbose=False):
    """Retrieves model saved using Python's Pickle module.

    'Unpickles' the model and optionally other data from file storage and
    returns it.

    Arguments:
        pkl_file {str} -- name of the pickle file with the stored object.

    Keyword Arguments:
        verbose {bool} -- print process messages. (default: {False})

    Returns:
        [tuple] -- three item tuple containg the model object, vectorizers and
            training data is present (None otherwise).
    """

    with open(pkl_file, 'rb') as f:
        data_dict = pickle._load(f)

    if verbose:
        print('Model and vectorizer created from file,', pkl_file, '.')
        if data_dict['training']:
            print('Contains', end=' ')
        else:
            print('Does not contain', end=' ')
        print('training data.')

    return (data_dict['model'], data_dict['vectorizer'], data_dict['training'])


def joblib_store(model, vectorizer, indic, train_data=None, tag='', name='',
                 verbose=False):
    """Store the model using Sci-Kit Learn's JobLib module.

    Stores the model and vectorizers, and optionally the  training data, in the
    current directory using the JobLib module. Filename will have suffix
    "_joblib_model.pkl". Prefix is either the name provided or the current date
    [YYYY-MM-DD].

    Arguments:
        model {model object} -- model to be stored
        vectorizer {nparray} -- vectorizers created from the training data.
        indic {str} -- name fo the indicator.

    Keyword Arguments:
        train_data {DataFrame} -- training data used to create vectorizers and
            model. (default: {None})
        name {str} -- prefix to be used for file name. (default: {''})
        verbose {bool} -- print process messages. (default: {False})
    """

    suffix = '-' + indic + '_joblib_model.pkl'
    if name:
        outfile = name + suffix
    else:
        outfile = curr_dte_txt(1) + suffix

    if tag:
        outfile = tag + '-' + outfile

    data_dict = {'model': model, 'vectorizer': vectorizer,
                 'training': train_data}

    joblib.dump(data_dict, outfile)

    if verbose:
        print('Model and vectorizers added to', outfile, '.')
        if train_data:
            print('Training data added to file', outfile, '.')


def joblib_retrieve(job_file, verbose=False):
    """Retrieves model stored using Sci-kit Learn's Joblib module.

    Unpacks the model and optionally other data from file storage and
    returns it.

    Arguments:
        job_file {string} -- name of the joblib file with the stored object.

    Keyword Arguments:
        verbose {bool} -- print process messages. (default: {False})

    Returns:
        [tuple] -- three item tuple containg the model object, vectorizers and
            training data is present (None otherwise).
    """

    data_dict = joblib.load(job_file)

    if verbose:
        print('Model and vectorizers created from file,', job_file, '.')
        if data_dict['training']:
            print('Contains', end=' ')
        else:
            print('Does not contain', end=' ')
        print('training data.')

    return (data_dict['model'], data_dict['vectorizer'], data_dict['training'])
