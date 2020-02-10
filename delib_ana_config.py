#!/usr/local/bin/python3
"""Module: DelibAnalysis Config File Management

Functions for managing the configuration file used to set the parameters for
the various DelibAnalysis processes and functions.

Package: DelibAnalysis
Version: 2.0
Authors: Eleonore Fournier-Tombs and Curtis Hendricks
Source: https://github.com/eleonoreft/DelibAnalysis
Copyright: Attribution-NonCommercial-ShareAlike CC BY-NC-SA
https://creativecommons.org/licenses/by-nc-sa/4.0/
Contact: eleonore.fournier-tombs@mail.mcgill.ca

"""

import configparser
from delib_ana_utils import INDICATORS

config = configparser.ConfigParser()

ACTIONS = ['predict',  'generate', 'generate_predict', 'batch_predict', 'test']

# TODO: Include file error handling functions


class DelibAnaConfiguration:

    def __init__(self, config_filename):
        self.config_filename = config_filename
        self.config = configparser.ConfigParser()
        self.config.read(self.config_filename)
        self.store_config_to_props()

    def store_config_to_props(self):
        self.action = self.config['general']['action']
        self.indicator = self.config['input']['indicator']
        tag = check_config_key('general', 'tag')
        if tag:
            self.tag = tag
        else:
            self.tag = ''

        if self.action in ['predict', 'generate_predict']:
            self.unlabelled = self.config['input']['unlabelled']
        if self.action in ['predict', 'generate', 'batch_predict']:
            self.store_name = self.config['input']['store_name']
        if self.action in ['generate']:
            str_type = self.config['input']['store_type']
            if str_type == '' or str_type is None:
                self.store_type = None
            else:
                self.store_type = str_type
        if self.action in ['predict', 'batch_predict']:
            self.store_type = self.config['input']['store_type']
        if self.action == 'batch_predict':
            self.unlabelled_dir = self.config['input']['unlabelled_dir']
        if self.action in ['generate', 'generate_predict', 'test']:
            self.labelled = self.config['input']['labelled']
            self.get_vocab()
            train_split = check_config_key('input', 'train_split')
            if train_split:
                self.train_split = float(train_split)
            else:
                self.train_split = None
            random_seed = check_config_key('input', 'random_seed_val')
            if random_seed:
                self.random_seed_val = int(random_seed)
            else:
                self.random_seed_val = None
        if self.action in ['generate_predict', 'test']:
            stored = check_config_key('input', 'stored')
            if stored is not None:
                stored = self.config['input'].getboolean('stored')
                if stored:
                    str_type = self.config['input']['store_type']
                    if str_type == '' or str_type is None:
                        self.store_type = None
                    else:
                        self.store_type = str_type
                    str_name = self.config['input']['store_name']
                    if str_name == '' or str_name is None:
                        self.store_name = None
                    else:
                        self.store_name = str_name
            self.stored = stored

    def get_vocab(self):
        if self.config['input']['vocab'] == '':
            self.vocab = []
        else:
            term_lst = []
            with open(self.config['input']['vocab']) as f:
                for line in f:
                    term_lst.append(line.strip())
            self.vocab = term_lst

    def __repr__(self):
        txt = ''
        for section in self.config:
            if section != 'DEFAULT':
                txt += section.upper() + ':' + '\n'
                for key in self.config[section]:
                    txt += '\t' + key + ' = ' + \
                           self.config[section][key] + '\n'
        return txt

    def __str__(self):
        return self.__repr__()


def test_config_file(config_filename):

    valid = True
    err = 'CONFIG FILE ERROR:'
    warn = 'CONFIG FILE WARNING:'

    config.read(config_filename)
    action = config['general']['action'].lower()
    if action not in ACTIONS:
        print('Invalid action entered. Valid actions:')
        print('\t', ACTIONS)
        return False

    if action == 'predict':
        valid = test_config_predict(err, warn)
    if action == 'generate':
        valid = test_config_generate(err, warn)
    if action == 'generate_predict':
        valid = test_config_generate_predict(err, warn)
    if action == 'batch_predict':
        valid = test_config_batch_predict(err, warn)
    if action == 'test':
        valid = test_config_testing(err, warn)

    if valid:
        print(action.upper(), 'config entries appear valid.')
        print('\tNOTE: File names and directory locations are not tested!')
    else:
        print(action.upper(), 'entries contains one of more errors.')
        print('\tSee previous messages.')

    return valid


def test_config_predict(e_st, w_st):

    valid = True
    e_ed = 'is required for "Predict" process.'
    w_ed = 'may be required for "Predict" process.'

    check_tag(w_st, w_ed)
    if not check_indicator(e_st, e_ed):
        valid = False
    if not check_unlabelled(e_st, e_ed):
        valid = False
    if not check_store_type(e_st, e_ed):
        valid = False
    if not check_store_name(e_st, e_ed):
        valid = False

    return valid


def test_config_generate(e_st, w_st):

    valid = True
    e_ed = 'is required for "Generate" process.'
    w_ed = 'may be required for "Generate" process.'

    check_tag(w_st, w_ed)
    if not check_indicator(e_st, e_ed):
        valid = False
    if not check_labelled(e_st, e_ed):
        valid = False
    check_vocab(w_st, w_ed)
    check_store_type(w_st, w_ed)
    if not check_store_name(e_st, e_ed):
        valid = False

    return valid


def test_config_generate_predict(e_st, w_st):

    valid = True
    e_ed = 'is required for "Generate and Predict" process.'
    w_ed = 'may be required for "Generate and Predict" process.'

    check_tag(w_st, w_ed)
    if not check_indicator(e_st, e_ed):
        valid = False
    if not check_unlabelled(e_st, e_ed):
        valid = False
    if not check_labelled(e_st, e_ed):
        valid = False
    check_vocab(w_st, w_ed)

    if check_stored(w_st, w_ed):
        check_store_type(w_st, w_ed)
        if not check_store_name(e_st, e_ed):
            valid = False

    return valid


def test_config_batch_predict(e_st, w_st):

    valid = True
    e_ed = 'is required for "Batch Predict" process.'
    w_ed = 'may be required for "Batch Predict" process.'

    check_tag(w_st, w_ed)
    if not check_indicator(e_st, e_ed):
        valid = False

    unlabelled_dir = check_config_key('input', 'unlabelled_dir')
    if not unlabelled_dir:
        print(e_st, 'directory name for unlabelled datasets', e_ed)
        valid = False

    if not check_store_type(e_st, e_ed):
        valid = False
    if not check_store_name(e_st, e_ed):
        valid = False

    return valid


def test_config_testing(e_st, w_st):

    valid = True
    e_ed = 'is required for "Testing" process.'
    w_ed = 'may be required for "Testing" process.'

    check_tag(w_st, w_ed)
    if not check_indicator(e_st, e_ed):
        valid = False
    if not check_labelled(e_st, e_ed):
        valid = False
    check_vocab(w_st, w_ed)

    if check_stored(w_st, w_ed):
        check_store_type(w_st, w_ed)
        if not check_store_name(e_st, e_ed):
            valid = False

    return valid


def check_config_key(section, key):
    """
    Input ini file section and key and returns a value if it is not an empty
    string. Returns "None" otherwise.
    """
    try:
        val = config[section][key]
        if val == '':
            val = None
    except KeyError:
        return None
    return val


def check_tag(st, ed):
    tag = check_config_key('general', 'tag')
    if not tag:
        print(st, 'tag name', ed)


def check_indicator(st, ed):
    indic = check_config_key('input', 'indicator')
    if not indic:
        print(st, 'indicator name', ed)
        return False
    elif indic not in INDICATORS:
        print(st, 'invalid indicator. List of valid indicators:', INDICATORS)
        return False
    return True


def check_vocab(st, ed):
    vocab = check_config_key('input', 'vocab')
    if not vocab:
        print(st, 'indicator vocabulary list filename', ed)


def check_labelled(st, ed):
    labelled = check_config_key('input', 'labelled')
    if not labelled:
        print(st, 'labelled dataset filename', ed)
        return False
    return True


def check_unlabelled(st, ed):
    unlabelled = check_config_key('input', 'unlabelled')
    if not unlabelled:
        print(st, 'unlabelled dataset filename', ed)
        return False
    return True


def check_stored(st, ed):
    stored = check_config_key('input', 'stored')
    if not stored:
        print(st, '[stored] true/false flag to store generated model', ed)
        return False
    return True


def check_store_type(st, ed):
    file_type = check_config_key('input', 'store_type')
    if not file_type:
        print(st, 'stored model file type', ed)
        return False
    elif file_type not in ['pickle', 'joblib']:
        print(st, 'invalid stored model file type. Options: ["pickle", \
                "joblib"].')
        return False
    return True


def check_store_name(st, ed):
    file_name = check_config_key('input', 'store_name')
    if not file_name:
        print(st, 'stored model file name', ed)
        return False
    return True


if __name__ == "__main__":

    test_config_file('delib_ana.ini')

    delib_config = DelibAnaConfiguration('delib_ana.ini')
    print(delib_config)
