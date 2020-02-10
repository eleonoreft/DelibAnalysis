#!usr/local/bin/python3
"""DelibAnalysis Utilities.

Data cleaning and other utility functions.

Package: DelibAnalysis
Version: 2.0
Authors: Eleonore Fournier-Tombs and Curtis Hendricks
Source: https://github.com/eleonoreft/DelibAnalysis
Copyright: Attribution-NonCommercial-ShareAlike CC BY-NC-SA
https://creativecommons.org/licenses/by-nc-sa/4.0/
Contact: eleonore.fournier-tombs@mail.mcgill.ca

"""

import pandas as pd
import numpy as np
import re

from datetime import date, datetime
from nltk import word_tokenize, pos_tag

# List of indicators that classifier can be trained on
INDICATORS = [
    'narrative', 'question', 'response', 'advocacy', 'public_interest',
    'respect'
]

# Dictionary of character ranges
char_dict = {
    'less_than_1000_chars': (1000, 0),
    'between_1000_and_2000_chars': (2000, 1000),
    'between_2000_and_3000_chars': (3000, 2000),
    'between_3000_and_4000_chars': (4000, 3000),
    'more_than_4000_chars': (1000000, 4000)
}


def comment_to_words(raw_comment):
    """Clean up raw comments and convert to a list of words

    Arguments:
        raw_comment {str} -- raw text containing words

    Returns:
        [list] -- list with each item being the words from the raw_comment
    """

    try:
        letters_only = re.sub("[^a-zA-Z]", " ", raw_comment)
        words = letters_only.lower().split()
        # stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words]  # if not w in stops]
        return (" ".join(meaningful_words))
    except TypeError:
        print(raw_comment)


def append_features(input_matrix, input_feature):
    new_matrix = np.zeros(shape=(input_matrix.shape[0],
                                 input_matrix.shape[1] + 1))
    for i in range(0, len(input_feature)):
        new_matrix[i] = np.append(input_matrix[i], input_feature[i])
    return new_matrix


def add_character_counts(data, chars):
    data['char_count'] = data['cleaned_comment'].apply(lambda x: len(x))
    for k, v in chars.items():
        data[k] = data.char_count.map(lambda x: 1
                                      if (x <= v[0] and x > v[1]) else 0)
    return data


def change_to_binary(value):
    """
    Make a numeric field binary. All 0 values are 0 and all other values are 1

    Arguments:
        value {int} -- input numeric value

    Returns:
        int -- binary version on value
    """

    if value == 0:
        return 0
    else:
        return 1


def get_question(txt):
    """
    Returns true (1) is item contains the text in txt contains and question
    mark, "?"

    Arguments:
        value {str} -- phrase or sentence

    Returns:
        int -- 1 if txt contains a question mark, 0, otherwise
    """
    if "?" in txt:
        return 1
    else:
        return 0


def get_gender(item):
    """Determine gender from text containing names.

    Arguments:
        item {str} -- text contiaing a persons name and title.

    Returns:
        str -- 'F' for female and 'M' otherwise.
    """

    txt = str(item)
    txt = txt.replace(".", "")
    txt = txt.lower()
    if "ms" in txt or "mrs" in txt:
        return "F"
    else:
        return "M"


def get_respect(text):
    """Determines if a text contains respectful sentiment.

    Arguments:
        text {str} -- text to examine

    Returns:
        int -- binary: 1 for respect, 0 for no respect
    """

    respect_vocab = [
        "thank you", "thank you, mr speaker", "thanks", "thank", "good day",
        "good morning", "good afternoon", "welcome", "appreciation",
        "like to recognize", "wish to recognize", "an honour", "my honour",
        "welcoming", "pay tribute", "applause", "give tribute", "appreciate",
        "happy to", "i apologize"
    ]
    respect = 0
    text = str(text)
    text = text.lower()
    for word in respect_vocab:
        if word in text:
            return 1
    return respect


def pos_tokenizer(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    token_array = pos_tag(word_tokenize(text))
    token_pos = [x[1] for x in token_array]
    return ' '.join(token_pos)


def add_column_parent(column):
    """Add a "parent" column for a given column.

    Parent column converts numeric data column to binary format.

    Arguments:
        column {Series} -- Pandas Series datatype, a 'column' from a Pandas
            Dataframe

    Returns:
        Series -- Pandas Series datatype (a DataFrame 'column')
    """
    new_column = [0]
    for i in range(0, len(column) - 1):
        new_column.append(change_to_binary(column[i]))
    return pd.Series(new_column)


def tst_print(descrip, info):
    """Helper function to print output for testing process.

    Arguments:
        descrip {str} -- name.
        info {str} -- detailed information.
    """
    print("---------------------------------")
    print(descrip)
    print("---------------------------------")
    print(info)


def get_tokenizer(text):
    """Get tokenizer type.

    Arguments:
        x {str} -- text to investigate.

    Returns:
        str -- tokenizer type.
    """

    if 'tfidf' in text:
        if len(text.split(' ')) == 2:
            return 'tfidf 2-ngram'
        else:
            return 'tfidf word'
    elif 'pos' in text:
        if len(text.split(' ')) == 2:
            return 'pos 2-ngram'
        else:
            return 'pos single'
    else:
        return text


def strip_tokenizer(field):
    """Remove unwanted text from tokenizer filed

    Arguments:
        field {str} -- tokenizer field to investigate.

    Returns:
        str -- sanitized tokenizer field.
    """

    return field.replace('tfidf__', '').replace('pos__', '')


def curr_dte_txt(date_type):
    """Returns the current date string format.

    Formats:
        1 = 2019-09-16
        2 = September 16, 2019
        3 = 09-16-19
        4 = Sep-16-2019

    Arguments:
        date_type {datetime} -- current date in datetime format

    Returns:
        str -- current date in desired string format.
    """

    if isinstance(date_type, int) and (date_type >= 1 or date_type >= 4):
        today = date.today()
        # YY/mm/dd
        d1 = today.strftime("%Y-%m-%d")
        # Textual month, day and year
        d2 = today.strftime("%B %d, %Y")
        # mm/dd/y
        d3 = today.strftime("%d-%m-%y")
        # Month abbreviation, day and year
        d4 = today.strftime("%b-%d-%Y")

        dates = {}
        dates[1] = d1
        dates[2] = d2
        dates[3] = d3
        dates[4] = d4

        return dates[date_type]
    else:
        return ''


def curr_tm_txt():
    """Returns the current time as text.

    Returns:
        str -- the current time in hours and minutes.
    """

    now = datetime.now()
    return now.strftime('%H:%M')


def import_label_data(file_loc):
    """Import a labelled dataset.

    Dataset should contain all the indicator fields already labelled.

    Arguments:
        file_loc {str} -- location of dataset file.

    Returns:
        DataFrame -- dataset converted to Pandas DataFrame object.
    """

    label_data = pd.read_csv(file_loc)
    label_data["cleaned_comment"] = label_data["speech"].astype(str).apply(
        comment_to_words)
    label_data["speech"] = label_data["speech"].apply(lambda x: x.lower())
    label_data = add_character_counts(label_data, char_dict)
    label_data["has_question"] = label_data["speech"].apply(get_question)
    label_data["has_respect"] = label_data["speech"].apply(get_respect)
    label_data["pos"] = label_data["speech"].apply(pos_tokenizer)
    label_data["interruption"] = label_data["interruption"].apply(
        lambda x: change_to_binary(x))
    label_data["disrespect"] = label_data["disrespect"].apply(
        lambda x: change_to_binary(x))
    label_data["has_question_parent"] = add_column_parent(
        label_data["has_question"])

    return label_data


def import_unlabelled_data(file_loc):
    """ Import an unlabelled dataset.

    Dataset shoud be a two columns csv file with columns "speaker" and
        "speech".

    Arguments:
        file_loc {str} -- Location of input file

    Returns:
        [DataFrame] -- Unlabelled dataset with addiditonal columns required for
        processing
    """

    data = pd.read_csv(file_loc)
    data["cleaned_comment"] = data["speech"].astype(
        str).apply(comment_to_words)
    data = add_character_counts(data, char_dict)
    data["has_respect"] = data["speech"].apply(get_respect)
    data["has_question"] = data["speech"].apply(get_question)
    data["has_question_parent"] = add_column_parent(data["has_question"])
    data["gender"] = data["speaker"].apply(get_gender)
    data["pos"] = data["speech"].apply(pos_tokenizer)
    for i in INDICATORS:
        data[i] = ''

    return data


def add_to_list(*items):
    """Returns all the arguments in a list.

    Returns:
        list -- Each paramter is an item the list.
    """

    lst = []
    for item in items:
        lst.append(item)
    return lst


def add_to_dict(**items):
    """Returns all keyword arguments in a dictionary.

    Returns:
        dict -- Each keyward/value parameter is an item in the dictionary.
    """

    dct = {}
    for key in items:
        if items[key] is not None:
            dct[key] = items[key]
    return dct
