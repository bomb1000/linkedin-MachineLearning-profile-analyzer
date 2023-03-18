import tkinter as tk
from tkinter import ttk

from xgboost import XGBClassifier

from data_preprocess import *
from predict_module import *
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
clf = XGBClassifier()


def load_xgb_model(model_file_name):
    clf = XGBClassifier()
    clf.load_model(model_file_name)
    return clf


def load_object(file_name):
    print("Load object from {}".format(file_name))
    return joblib.load(file_name)


def load_feature_preprocessor():
    company_type_one_hot_encoder = load_object(file_name="data/companyTypeOneHotEncoder.pickle")
    name_tokens_one_hot_encoder = load_object(file_name="data/nameTokensOneHotEncoder.pickle")
    education_type_one_hot_encoder = load_object(file_name="data/educationTypeOneHotEncoder.pickle")
    position_token_one_hot_encoder = load_object(file_name="data/positionTokenOneHotEncoder.pickle")

    return FeaturePreprocessor(
        company_type_one_hot_encoder,
        name_tokens_one_hot_encoder,
        education_type_one_hot_encoder,
        position_token_one_hot_encoder,
    )


def load_xgb_models():
    cv_clf_path = "data/xgb_cv.model"
    nlp_clf_path = "data/xgb_nlp.model"
    tool_clf_path = "data/xgb_tool.model"
    print("Load model from {}".format(cv_clf_path))
    print("Load model from {}".format(nlp_clf_path))
    print("Load model from {}".format(tool_clf_path))
    cv_clf = load_xgb_model(cv_clf_path)
    nlp_clf = load_xgb_model(nlp_clf_path)
    tool_clf = load_xgb_model(tool_clf_path)
    return cv_clf, nlp_clf, tool_clf


data_preprocessor = DataPreprocessor()
feature_preprocessor = load_feature_preprocessor()
cv_clf, nlp_clf, tool_clf = load_xgb_models()


def predict_scores(profile_url):
    # 14.644576072692871 62.7%
    profile_df = get_personal_profile_df(profile_url)
    # my_progress['value'] += 60
    # window.update_idletasks()
    # 7.758749961853027 32.6%
    preprocessed_data_df = data_preprocessor.transform(profile_df)
    # my_progress['value'] += 30
    # window.update_idletasks()
    # 0.3488481044769287 1.5%
    feature_df = feature_preprocessor.transform(preprocessed_data_df)
    # 0.07547593116760254 0.3%
    feature_df = feature_df.rename(columns=replace_invalid_field_name_characters)
    # 0.5326921939849854 2.2%
    cv_y_pred = cv_clf.predict(feature_df)[0]
    nlp_y_pred = nlp_clf.predict(feature_df)[0]
    tool_y_pred = tool_clf.predict(feature_df)[0]
    # my_progress['value'] += 10
    # window.update_idletasks()
    # my_progress.stop()

    return cv_y_pred, nlp_y_pred, tool_y_pred


window = tk.Tk()
window.title('MLer App')
window.geometry('800x600')
window.configure(background='cyan')

# my_progress = ttk.Progressbar(window, orient='horizontal', length=300, mode='determinate')
# my_progress.pack(pady=20)
def print_scores():
    url = str(url_entry.get())

    try:
        cv_y_pred, nlp_y_pred, tool_y_pred = predict_scores(url)

        talent_level = {0 : '不懂', 1 : '新手', 2 : '熟手', 3 : '高手', }
        scores = "CV : {}, NLP : {}, TOOL : {}".format(talent_level[cv_y_pred], talent_level[nlp_y_pred], talent_level[tool_y_pred])

        result_label.configure(text=scores)
    except Exception:
        error_message = "發生錯誤"
        result_label.configure(text=error_message)


header_label = tk.Label(window, text='ML 人才分析')
header_label.pack()

url_frame = tk.Frame(window)
url_frame.pack(side=tk.TOP)
url_label = tk.Label(url_frame, text='LinkedIn履歷URL')
url_label.pack(side=tk.LEFT)
url_entry = tk.Entry(url_frame)
url_entry.pack(side=tk.LEFT)


# weight_frame = tk.Frame(window)
# weight_frame.pack(side=tk.TOP)
# weight_label = tk.Label(weight_frame, text='體重（kg）')
# weight_label.pack(side=tk.LEFT)
# weight_entry = tk.Entry(weight_frame)
# weight_entry.pack(side=tk.LEFT)

result_label = tk.Label(window)
result_label.pack()

calculate_btn = tk.Button(window, text='馬上分析', command=print_scores)
calculate_btn.pack()

window.mainloop()
