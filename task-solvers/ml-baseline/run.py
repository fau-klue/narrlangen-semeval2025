from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer



import pandas as pd
import numpy as np


def generate_baseline_matrix(out_path: str,
                             post_ids: str,
                             list_of_results: list[float],
                             labels_list: list[str],
                             sent_ids: list[int] = None
                             ):
    for i, _ in enumerate(list_of_results):
        list_of_results[i].insert(0, post_ids[i])
        if sent_ids is not None:
            list_of_results[i].insert(1, sent_ids[i])

    with open(out_path, "w", encoding="utf-8") as out_f:
        if sent_ids is not None:
            out_f.write("id\tsent_id\tlabel_id\tscore\n")
            for line in list_of_results:
                for i, _ in enumerate(labels_list):
                    out_f.write(f"{line[0]}\t{line[1]}\t{labels_list[i]}\t{line[i + 2]}\n")
        else:
            out_f.write("id\tlabel_id\tscore\n")
            for line in list_of_results:
                for i, _ in enumerate(labels_list):
                    out_f.write(f"{line[0]}\t{labels_list[i]}\t{line[i + 1]}\n")


def prepare_data(path_to_file: str, delimiter: str = ",", text_column: str = "text", filtered_columns: list[str] = None, drop_zeros_labels: bool=False):
    dataframe = pd.read_csv(path_to_file, delimiter=delimiter)
    remaining_columns = []
    if drop_zeros_labels:
        only_zeros_labels = dataframe.columns[dataframe.eq(0).all()]
        dataframe.drop(columns=only_zeros_labels, inplace=True)
        remaining_columns = [column for column in dataframe.columns.tolist() if column not in filtered_columns + only_zeros_labels.tolist()]
    labels = [column for column in dataframe.columns.tolist() if column not in filtered_columns]
    texts = dataframe[text_column].to_list()
    return np.array(texts), np.array(dataframe[labels]), remaining_columns


def keep_true_label_probability(list_of_probabilities: list[list[float]]):
    """ """
    probabilities = []
    label_proba = []
    for i, _ in enumerate(list_of_probabilities):
        for j, _ in enumerate(list_of_probabilities[0]):
            label_proba.append(float(list_of_probabilities[i][j][1]))
        probabilities.append(label_proba)
        label_proba = []
    return probabilities

def get_top_predictions(labels_probabilities, top_k=3):
    top_k_predictions = []
    for array in labels_probabilities:
        top_k_predictions.append(np.argsort(array)[::-1][:top_k].tolist())
    return top_k_predictions


if __name__ == "__main__":
    use_descriptions_in_training = False
    is_combined_labels = True

    language = "PT"
    # Path to training files
    TRAIN_FILE = f"../../task_data/subtask2_hf/{language}_fine-grained/train.csv"
    DEV_FILE = f"../../task_data/subtask2_hf/{language}_fine-grained/dev.csv"
    TEST_FILE = None


    OUT_PATH_BASELINE1 = f"./results/lr_baseline_{language}.txt" # ("../results/combined-labels/ml-baseline"
                          # "/ml_baseline_lr_combined_labels_w_description_validation_scores.tsv")
    OUT_PATH_BASELINE2 = f"./results/svm_baseline_{language}.txt" # ("../results/combined-labels/ml-baseline"
                         # "/ml_baseline_svm_combined_labels_w_description_validation_scores.tsv")

    # Create labels to ids mapping


    # Prepare data for baseline model
    # Get post ids from test data for score matrix
    # test_post_ids = pd.read_csv(DEV_FILE, delimiter=",")["id"].to_list()

    print("Prepare training data...")
    train_text, train_labels, labels = prepare_data(TRAIN_FILE, filtered_columns=["file", "text"], drop_zeros_labels=True)
    # labels = [column for column in pd.read_csv(TRAIN_FILE, delimiter=",").columns.tolist() if column not in
    #          ["file", "text"]
    #          ]
    labels_ids = {label: i for i, label in enumerate(labels)}
    ids_labels = {i: label for i, label in enumerate(labels)}
    labels_list = list(labels_ids.keys())
    if use_descriptions_in_training:
        DESCRIPTION_FILE = "" # TODO: Use description if required
        desc_text, desc_labels,_ = prepare_data(DESCRIPTION_FILE, filtered_columns=["file", "text"])
        train_text = np.append(train_text, desc_text)
        train_labels = np.append(train_labels, desc_labels, axis=0)

    print("Prepare dev/test data...")
    dev_files = pd.read_csv(DEV_FILE, delimiter=",")["file"].tolist()
    dev_text, dev_labels,_ = prepare_data(DEV_FILE, filtered_columns=["file", "text"])

    # test_files = pd.read_csv(TEST_FILE, delimiter=",")["file"].tolist()
    # test_text, test_labels = prepare_data(TEST_FILE, labels=labels_ids.keys())

    # Baselines
    # Logistic Regression
    print("Run Baseline Training...")
    lr_pipeline = Pipeline([("tf_idf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
                            ("clf", MultiOutputClassifier(LogisticRegression(class_weight="balanced")))
                            ]
                           )
    lr_pipeline.fit(train_text, train_labels)

    # Support Vector Machine
    svm_pipeline = Pipeline([("tf_idf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
                             ("clf", MultiOutputClassifier(SVC(probability=True, class_weight="balanced")))
                             ]
                            )
    svm_pipeline.fit(train_text, train_labels)

    # Evaluate models performance on test set and retrieve predicted probabilities from pipeline. Since
    # sklearn-multiouput returns an m x n x 2 array, where m is number of labels and n is number of instances,
    # we need to normalise them for further evaluation
    lr_labels_proba = keep_true_label_probability(lr_pipeline.predict_proba(dev_text))
    svm_labels_proba = keep_true_label_probability(svm_pipeline.predict_proba(dev_text))

    # Get top predictions based on probabilities
    lr_top_predictions = get_top_predictions(lr_labels_proba)
    svm_top_predictions = get_top_predictions(svm_labels_proba)


    with open(OUT_PATH_BASELINE1, "w", encoding="utf-8") as lr_out:
        with open(OUT_PATH_BASELINE2, "w", encoding="utf-8") as svm_out:
            for i,file_id in enumerate(dev_files):
                lr_out.write(f"{file_id}\tOther\t{';'.join([ids_labels[label_id] for label_id in lr_top_predictions[i]])}\n")
                svm_out.write(f"{file_id}\tOther\t{';'.join([ids_labels[label_id] for label_id in svm_top_predictions[i]])}\n")