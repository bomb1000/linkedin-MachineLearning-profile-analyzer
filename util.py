import codecs
import json

import pandas as pd
import re

def replace_invalid_field_name_characters(field_name):
    invalid_field_pattern = re.compile('[\\[\\]<,]*')
    return invalid_field_pattern.sub('', field_name)


def read_json_file(file_path):
    with codecs.open(file_path, "r", encoding='utf-8', errors='ignore') as fdata:
        return json.load(fdata)


def read_csv_file_as_df(file_path):
    with codecs.open(file_path, "r", encoding='utf-8', errors='ignore') as fdata:
        return pd.read_csv(fdata, encoding="utf-8")


def build_query_by_feature(feature, input_df_name="input_data"):
    field_names = ["`{}`".format(field_name) for field_name in feature.get_csv_field_names()]
    return 'select {} from {};'.format(", ".join(field_names), input_df_name)
