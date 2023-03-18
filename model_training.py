import os
from collections import Counter

from imblearn.over_sampling import SMOTE, RandomOverSampler
from xgboost import XGBClassifier

from feature_preprocess import read_csv_file_as_df
from util import replace_invalid_field_name_characters

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def handle_imbalanced_data(X, y):

    try:
        sampler = SMOTE(random_state=42)
        return sampler.fit_sample(X, y)
    except Exception:
        sampler = RandomOverSampler(random_state=0)
        return sampler.fit_sample(X, y)


if __name__ == '__main__':
    data_path = "data/model_input.csv"
    model_file_name_pattern = "data/xgb_{}.model"
    input_data = read_csv_file_as_df(data_path)
    input_data = input_data.rename(columns=replace_invalid_field_name_characters)

    label_types = [
        'CV',
        'Tool',
        'NLP'
    ]

    for label_type in label_types:
        print("Training XGBoost model with label name: {}".format(label_type))
        model_file_name = model_file_name_pattern.format(label_type.lower())
        # Split label and data
        y = input_data[label_type]
        X = input_data.drop('CV', 1).drop('Tool', 1).drop('NLP', 1)

        # Handle imbalanced data set
        print(sorted(Counter(y).items()))
        X, y = handle_imbalanced_data(X, y)
        print(sorted(Counter(y).items()))

        clf = XGBClassifier()
        clf.fit(X, y)
        clf.save_model(model_file_name)
        print("Training is complected, model is saved to {}".format(model_file_name))
