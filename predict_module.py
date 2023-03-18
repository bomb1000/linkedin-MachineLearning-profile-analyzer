import os

from feature.feature import features
from feature_preprocess import *
from util import read_json_file
from util import replace_invalid_field_name_characters


def get_personal_profile_df(profile_url):
    command = r"node ./crawler/profileCrawler.js {}".format(profile_url)
    os.system(command)
    profile_json = read_json_file("data/profile.json")
    profile_data = flatten_json(profile_json)
    profile_data_transformed = fetch_required_fields_and_rename(profile_data)
    return pd.DataFrame(profile_data_transformed, index=[0])


def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '/')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '/')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


def fetch_required_fields_and_rename(profile_data):
    mappings = {}

    for feature in features:
        mappings.update(feature.get_json_to_csv_field_mapping())
    result = {}

    for key, value in mappings.items():
        result[value] = profile_data.get(key, None)
    return result





if __name__ == '__main__':
    # https://www.linkedin.com/in/chungkaihsieh/
    # nltk.download('punkt')
    # data_preprocessor = DataPreprocessor()
    # feature_preprocessor = load_feature_preprocessor()
    # cv_clf, nlp_clf, tool_clf = load_xgb_models()

    should_continue = True
    while should_continue:
        profile_url = input("Enter profile url: ")
        if profile_url == "exit":
            should_continue = False
            break
        profile_df = get_personal_profile_df(profile_url)

        preprocessed_data_df = data_preprocessor.transform(profile_df)
        feature_df = feature_preprocessor.transform(preprocessed_data_df)
        feature_df = feature_df.rename(columns=replace_invalid_field_name_characters)
        cv_y_pred = cv_clf.predict(feature_df)[0]
        nlp_y_pred = nlp_clf.predict(feature_df)[0]
        tool_y_pred = tool_clf.predict(feature_df)[0]

        score = "CV : {}, NLP : {}, TOOL : {}".format(cv_y_pred, nlp_y_pred, tool_y_pred)

        # print("CV Score: {}".format(cv_y_pred))
        # print("NLP Score: {}".format(nlp_y_pred))
        # print("Tool Score: {}".format(tool_y_pred))
