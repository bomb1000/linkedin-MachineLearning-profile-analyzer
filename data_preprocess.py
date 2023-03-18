from __future__ import print_function

import re

import nltk
import pandas as pd
import pandasql as pdsql
# from googletrans import Translator
from google_trans_new import google_translator
from sklearn.base import BaseEstimator, TransformerMixin
import time
from feature.feature import experience_company_feature, experience_date_feature, experience_name_feature, \
    education_feature, position_feature
from util import read_csv_file_as_df, build_query_by_feature


class TranslatorWrapper:

    def __init__(self):
        self.translator = google_translator()

    def translate(self, text, dest='en'):
        try:
            return self.translator.translate(text, lang_tgt=dest)
        except Exception:
            time.sleep(5)
            return self.translate(text, dest=dest)


class CompanyFeaturePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.translator = TranslatorWrapper()
        self.replaced_word_list_pattern = re.compile('(Permanent|Full-time|Internship|Part-time)')
        self.blank_space_pattern = re.compile('\\s+')

    def fit(self, X, y=None, **fit_params):
        print("Fitting X by CompanyFeaturePreprocessor")
        return self

    def transform(self, X, **transform_params):
        print("Transforming X by CompanyFeaturePreprocessor")
        df = X.fillna("none")
        df_transformed = df.applymap(self.preprocess_value)
        return df_transformed

    def preprocess_value(self, origin_value):
        text = self.replaced_word_list_pattern.sub('', origin_value)
        text = self.blank_space_pattern.sub(' ', text).strip()
        return self.translator.translate(text, dest='en').lower()


class NameFeaturePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()
        self.translator = TranslatorWrapper()
        self.replaced_word_list_pattern = re.compile('[^0-9a-zA-Z\\s]*')
        self.blank_space_pattern = re.compile('\\s+')

    def fit(self, X, y=None, **fit_params):
        print("Fitting X by NameFeaturePreprocessor")
        return self

    def transform(self, X, **transform_params):
        print("Transforming X by NameFeaturePreprocessor")
        df = X.fillna("none")
        return df.applymap(self.preprocess_value)

    def preprocess_value(self, origin_value):
        text = str(origin_value)
        translated_text = self.translator.translate(text, dest='en').lower()
        translated_text = self.replaced_word_list_pattern.sub('', translated_text)
        preprocessed_name = self.blank_space_pattern.sub(' ', translated_text).strip()
        name_tokens = preprocessed_name.split(' ')
        return ','.join(name_tokens)


class PositionFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.translator = TranslatorWrapper()
        self.blank_space_pattern = re.compile('\\s+')
        self.replaced_word_list_pattern = re.compile('[^0-9a-zA-Z\\s\']+')

    def fit(self, X, y=None, **fit_params):
        print("Fitting X by PositionFeaturePreprocessor")
        return self

    def transform(self, X, **transform_params):
        print("Transforming X by PositionFeaturePreprocessor")
        df = X.fillna("none")
        return df.applymap(self.preprocess_value)

    def preprocess_value(self, origin_value):
        translated_text = self.translator.translate(origin_value, dest='en').lower()
        translated_text = self.replaced_word_list_pattern.sub(' ', translated_text)
        translated_text = self.blank_space_pattern.sub(' ', translated_text).strip()
        return ','.join(nltk.word_tokenize(translated_text))


class EducationFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.translator = TranslatorWrapper()
        self.replaced_word_list_pattern = re.compile('[^0-9a-zA-Z\\s]*')
        self.blank_space_pattern = re.compile('\\s+')

    def fit(self, X, y=None, **fit_params):
        print("Fitting X by EducationFeaturePreprocessor")

        return self

    def transform(self, X, **transform_params):
        print("Transforming X by EducationFeaturePreprocessor")
        df = X.fillna("none")
        return df.applymap(self.preprocess_value)

    def preprocess_value(self, origin_value):
        text = str(origin_value)
        text = self.blank_space_pattern.sub(' ', text).strip()
        return self.translator.translate(text, dest='en').lower()


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformer_field_mapping = [
            (
                CompanyFeaturePreprocessor(),
                experience_company_feature
            ),
            (
                None,
                experience_date_feature
            ),
            (
                NameFeaturePreprocessor(),
                experience_name_feature
            ),
            (
                EducationFeaturePreprocessor(),
                education_feature
            ),
            (
                PositionFeaturePreprocessor(),
                position_feature
            )
        ]

    def transform(self, X, **transform_params):
        input_df = X
        df_list = []
        for mapping in self.transformer_field_mapping:
            feature = mapping[1]
            print("Transforming {} feature".format(feature.field_name))
            query = build_query_by_feature(feature, input_df_name="X")
            sub_df = pdsql.sqldf(query, locals())
            transformer = mapping[0]
            if transformer is not None:
                sub_feature_df = transformer.transform(sub_df)
            else:
                sub_feature_df = sub_df
            df_list.append(sub_feature_df)
            print()

        return pd.concat(df_list, axis=1)


if __name__ == '__main__':
    data_path = "data/data_with_labels.csv"
    preprocessed_data_path = "data/preprocessed_data.csv"
    input_data = read_csv_file_as_df(data_path)

    data_preprocessor = DataPreprocessor()
    preprocessed_data_df = data_preprocessor.transform(input_data)
    preprocessed_data_df.to_csv(preprocessed_data_path, index=False)
    print("Preprocessed file has been saved to {}.".format(preprocessed_data_path))
