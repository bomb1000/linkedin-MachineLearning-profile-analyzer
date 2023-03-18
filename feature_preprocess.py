from __future__ import print_function

import joblib
import pandas as pd
import pandasql as pdsql
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from feature.feature import experience_company_feature, experience_date_feature, experience_name_feature, \
    education_feature, position_feature
from feature.preprocess.process_date import generate_on_job_status_df
from util import read_csv_file_as_df, build_query_by_feature


def add_prefix_to_column(prefix, df):
    df.columns = list(map(lambda column_name: prefix + "/" + column_name, df.columns))


class DateRangeFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        print("Fitting X by DateRangeFeatureTransformer")
        return self

    def transform(self, X, **transform_params):
        print("Transforming X by DateRangeFeatureTransformer")
        on_job_status_df = generate_on_job_status_df(X)
        return on_job_status_df


class TypeOneHotFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None, **fit_params):
        print("Fitting X by TypeOneHotFeatureTransformer")
        total_data = []
        for i, row in X.iterrows():
            column_size = len(X.columns)
            row_data = []
            for column_index in range(0, column_size):
                value = str(row[column_index])
                row_data.append(value)
            total_data.append(row_data)

        self.mlb.fit(total_data)
        return self

    def transform(self, X, **transform_params):
        print("Transforming X by TypeOneHotFeatureTransformer")
        df_list = []
        classes = self.mlb.classes_

        for column_index, column_name in enumerate(X.columns):
            mlb_input = []
            for index, row in X.iterrows():
                value = [row[column_index]]
                mlb_input.append(value)

            data_df = pd.DataFrame(self.mlb.transform(mlb_input), columns=classes)
            add_prefix_to_column(column_name, data_df)
            df_list.append(data_df)
        return pd.concat(df_list, axis=1)


class TokensOneHotFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None, **fit_params):
        """
        Form the feature vector space base on df like [['data,engineer','ai,engineer'], [..., ...]]

        :param df: df like [['data,engineer','ai,engineer'], [..., ...]]
        :return: instance itself
        """
        print("Fitting X by TokensOneHotFeatureTransformer")
        total_data = []
        for i, row in X.iterrows():
            column_size = len(X.columns)
            row_data = []
            for column_index in range(0, column_size):
                # should be in the form of: ai,engineer
                tokens = str(row[column_index]).split(',')
                row_data.extend(tokens)
            total_data.append(row_data)

        # After the preprocessing, the df becomes list of list as the input format of mlb
        # e.g. [['data','engineer','ai','engineer'], [..., ...]]
        self.mlb.fit(total_data)
        return self

    def transform(self, X, **transform_params):
        """
        Form the feature value base on mlb vector space from the fit() result

        :param df: df like [['data,engineer','ai,engineer'], [..., ...]]
        :return: feature value df like [[0, 0, ...], [1, 0, ...]]
        """
        print("Transforming X by TokensOneHotFeatureTransformer")
        df_list = []
        classes = self.mlb.classes_

        for column_index, column_name in enumerate(X.columns):
            mlb_input = []
            for index, row in X.iterrows():
                tokens = str(row[column_index]).split(',')
                mlb_input.append(tokens)

            data_df = pd.DataFrame(self.mlb.transform(mlb_input), columns=classes)
            add_prefix_to_column(column_name, data_df)
            df_list.append(data_df)
        return pd.concat(df_list, axis=1)


def save_object(obj, file_name):
    print("Save object to {}".format(file_name))
    joblib.dump(obj, file_name, compress=1)
    # pickle.dump(obj, open(file_name, "wb"))


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, company_type_one_hot_encoder, name_tokens_one_hot_encoder, education_type_one_hot_encoder,
                 position_token_one_hot_encoder):
        self.company_type_one_hot_encoder = company_type_one_hot_encoder
        self.name_tokens_one_hot_encoder = name_tokens_one_hot_encoder
        self.education_type_one_hot_encoder = education_type_one_hot_encoder
        self.position_token_one_hot_encoder = position_token_one_hot_encoder

        self.transformer_field_mapping = [
            (
                Pipeline(steps=[
                    ('one_hot_encoder', self.company_type_one_hot_encoder)
                ]),
                experience_company_feature
            ),
            (
                DateRangeFeatureTransformer(),
                experience_date_feature
            ),
            (
                Pipeline(steps=[
                    ('one_hot_encoder', self.name_tokens_one_hot_encoder)
                ]),

                experience_name_feature
            ),
            (
                Pipeline(steps=[
                    ('one_hot_encoder', self.education_type_one_hot_encoder)
                ]),
                education_feature
            ),
            (
                Pipeline(steps=[
                    ('one_hot_encoder', self.position_token_one_hot_encoder)
                ]),
                position_feature
            )
        ]

    def fit(self, X, y=None, **fit_params):
        for mapping in self.transformer_field_mapping:
            feature = mapping[1]
            print("Fitting {} feature".format(feature.field_name))
            query = build_query_by_feature(feature, input_df_name="X")
            sub_df = pdsql.sqldf(query, locals())
            transformer = mapping[0]
            transformer.fit(sub_df)
            print()
        return self

    def transform(self, X, **transform_params):
        input_df = X
        df_list = []
        for mapping in self.transformer_field_mapping:
            feature = mapping[1]
            print("Transforming {} feature".format(feature.field_name))
            query = build_query_by_feature(feature, input_df_name="X")
            sub_df = pdsql.sqldf(query, locals())
            transformer = mapping[0]
            sub_feature_df = transformer.transform(sub_df)
            df_list.append(sub_feature_df)
            print()

        return pd.concat(df_list, axis=1)


if __name__ == '__main__':
    data_path = "data/data_with_labels.csv"
    model_input_file_path = "data/model_input.csv"
    preprocessed_data_path = "data/preprocessed_data.csv"
    input_data = read_csv_file_as_df(data_path)

    preprocessed_data_df = read_csv_file_as_df(preprocessed_data_path)

    company_type_one_hot_encoder = TypeOneHotFeatureTransformer()
    name_tokens_one_hot_encoder = TokensOneHotFeatureTransformer()
    education_type_one_hot_encoder = TypeOneHotFeatureTransformer()
    position_token_one_hot_encoder = TokensOneHotFeatureTransformer()

    feature_preprocessor = FeaturePreprocessor(
        company_type_one_hot_encoder,
        name_tokens_one_hot_encoder,
        education_type_one_hot_encoder,
        position_token_one_hot_encoder,
    )

    feature_preprocessed = feature_preprocessor.fit_transform(preprocessed_data_df)
    save_object(obj=company_type_one_hot_encoder, file_name="data/companyTypeOneHotEncoder.pickle")
    save_object(obj=name_tokens_one_hot_encoder, file_name="data/nameTokensOneHotEncoder.pickle")
    save_object(obj=education_type_one_hot_encoder, file_name="data/educationTypeOneHotEncoder.pickle")
    save_object(obj=position_token_one_hot_encoder, file_name="data/positionTokenOneHotEncoder.pickle")

    # append label
    label_query = 'select NLP, CV, Tool from input_data;'
    label_df = pdsql.sqldf(label_query, locals())

    # Combine data with label
    feature_with_label_df = pd.concat([label_df, feature_preprocessed], axis=1)
    feature_with_label_df.to_csv(model_input_file_path, index=False)
    print("Model input file has been saved to {}.".format(model_input_file_path))
