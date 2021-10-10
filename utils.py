import string
import pandas as pd
from os import path, makedirs


def create_output_dir(dataset, output_dir):
    output_path = path.join(output_dir, dataset)
    if not path.isdir(output_path):
        makedirs(output_path)

    return output_path


def get_text(filepath):
    text_file = open(filepath, "r")
    text = text_file.read()
    text_file.close()

    return text


def convert_plus_fill(series, errors='coerce', fill=0):
    series = pd.to_numeric(series, errors=errors)
    if fill is not None:
        series = series.fillna(fill)

    return series


def word_formatter(s):
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s.lower()
