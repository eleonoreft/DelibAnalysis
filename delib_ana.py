#!/usr/local/bin/python3
""" DelibAnalysis Main

Contains the implementation of the command line argument parser and the main
call to instantiate the configuration file object to manage the key-value items
in the active configuration file.

Package: DelibAnalysis
Version: 2.0
Authors: Eleonore Fournier-Tombs and Curtis Hendricks
Source: https://github.com/eleonoreft/DelibAnalysis
Copyright: Attribution-NonCommercial-ShareAlike CC BY-NC-SA
https://creativecommons.org/licenses/by-nc-sa/4.0/
Contact: eleonore.fournier-tombs@mail.mcgill.ca

"""

import argparse
import delib_ana_process as process
import delib_ana_utils as utils
from delib_ana_config import DelibAnaConfiguration
from delib_ana_config import test_config_file


def main():
    """Check configuration and if file is OK, continue to main application.

    Configuration file will be checked for formatting and requirements,
    warnings and errors will be sent to standard output. If there are errors,
    the main application won't be run.
    """

    # Command line argument parser setup
    parser = argparse.ArgumentParser(
        description="DelibAnalysis command line arguments")
    parser.add_argument('-c', '--config_file', default='delib_ana.ini', help='''
                        Name of configuration file to be used. Default:
                        delib.ana.ini.''')
    args = parser.parse_args()
    config_file = args.config_file

    # Get config info
    config_good = test_config_file(config_file)
    if config_good:
        delib_config = DelibAnaConfiguration(config_file)
        run_process(delib_config)
    else:
        print('Improperly formatted config file\nDelibAnalysis Exiting.')


def run_process(config_obj):
    """Run Delib Analysis process based on configuration file options.

    Arguments:
        config_obj {DelibAnaConfiguration} -- DelibAnalys configuration object.
    """

    ana_process = config_obj.action
    active_indicator = config_obj.indicator
    active_tag = config_obj.tag

    if ana_process in ['predict', 'generate_predict']:
        loc_unlabelled = config_obj.unlabelled
    if ana_process in ['predict', 'generate', 'batch_predict']:
        file_name = config_obj.store_name
        file_type = config_obj.store_type
    if ana_process == 'batch_predict':
        loc_dir_unlabelled = config_obj.unlabelled_dir
    if ana_process in ['generate', 'generate_predict', 'test']:
        loc_labelled_train = config_obj.labelled
        indicator_vocab = config_obj.vocab
        train_split = config_obj.train_split
        r_seed = config_obj.random_seed_val
    if ana_process in ['generate_predict', 'test']:
        if config_obj.stored is not None:
            stored = config_obj.stored
            if stored:
                file_type = config_obj.store_type
                file_name = config_obj.store_name
        else:
            stored = None
            file_type = None
            file_name = None

    param_list = []
    param_dict = {}

    if ana_process == 'predict':
        param_list = utils.add_to_list(loc_unlabelled, active_indicator,
                                       file_type, file_name)
        new_label_dataset = process.predict_process(*param_list)
        f_name = loc_unlabelled.split('/')[-1:][0]
        outfile_name = active_tag + '-' + active_indicator
        outfile_name += f_name
        new_label_dataset.to_csv(outfile_name)
        print("Predict process result saved to file:", outfile_name)
        print(new_label_dataset.head(10))
    elif ana_process == 'generate':
        param_list = utils.add_to_list(loc_labelled_train, active_indicator,
                                       indicator_vocab)
        param_dict = utils.add_to_dict(tag=active_tag, store_name=file_name,
                                       store_type=file_type)
        process.generate_process(*param_list, **param_dict)
    elif ana_process == 'generate_predict':
        param_list = utils.add_to_list(loc_labelled_train, loc_unlabelled,
                                       active_indicator, indicator_vocab,
                                       active_tag)
        param_dict = utils.add_to_dict(store=stored, store_name=file_name,
                                       store_type=file_type,
                                       train_split=train_split, r_state=r_seed)
        new_label_dataset = process.gen_predict_process(*param_list,
                                                        **param_dict)
        f_name = loc_unlabelled.split('/')[-1:][0]
        outfile_name = active_tag + '-' + active_indicator
        outfile_name += f_name
        new_label_dataset.to_csv(outfile_name)
        print("Generate-Predict process result saved to file:", outfile_name)
        print(new_label_dataset.head(10))
    elif ana_process == 'batch_predict':
        param_list = utils.add_to_list(loc_dir_unlabelled, active_indicator,
                                       file_name, file_type)
        param_dict = utils.add_to_dict(tag=active_tag)
        process.dir_predict_process(*param_list, **param_dict)
        # TODO: check if I need to output anything here
        # running list of files to output
    elif ana_process == 'test':
        param_list = utils.add_to_list(loc_labelled_train, active_indicator,
                                       indicator_vocab, active_tag)
        param_dict = utils.add_to_dict(store=stored, store_name=file_name,
                                       store_type=file_type,
                                       train_split=train_split, r_state=r_seed)
        process.testing_process(*param_list, **param_dict)


if __name__ == '__main__':

    main()
