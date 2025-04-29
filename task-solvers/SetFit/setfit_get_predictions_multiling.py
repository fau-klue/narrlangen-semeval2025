from datasets import load_dataset
from setfit import SetFitModel
from pandas import DataFrame, Series
import regex as re


# Single label feature
def encode_labels(record) -> dict[str, list[list[int]]]:
    return {"labels": [record[feature] for feature in features]}


if __name__ == "__main__":
    # Load our prepared datasets (we're only interested in the test data, however)

    path_data = ["../../task_data/subtask2_hf/EN_fine-grained/",
                 "../../task_data/subtask2_hf/BG_fine-grained/",
                 "../../task_data/subtask2_hf/HI_fine-grained/",
                 "../../task_data/subtask2_hf/PT_fine-grained/",
                 "../../task_data/subtask2_hf/RU_fine-grained/"]

    path_model = "../../models/setfit/setfit_simple_multiling/"

    # Local model
    model = SetFitModel.from_pretrained(path_model, local_files_only=True)

    for path in path_data:
        lang_set = re.search(r"([A-Z]{2}_fine-grained)", path).group(1)
        path_prediction_prob = "results/setfit_simple_multiling_{lang_set}_dev_probs.tsv.gz".format(lang_set=lang_set)
        path_prediction_cat = "results/setfit_simple_multiling_{lang_set}_dev_cats.tsv.gz".format(lang_set=lang_set)
        path_prediction_cat_target = "results/setfit_simple_multiling_{lang_set}_dev_cats_target.txt".format(lang_set=lang_set)

        dataset = load_dataset(path, download_mode='force_redownload')

        # Labels to be predicted
        features = dataset["train"].column_names
        for col in ["file", "text"]:
            features.remove(col)

        dataset = dataset.map(encode_labels)

        # Run inference
        colnames = features.copy()
        colnames.append("file")

        pred_prob = model.predict_proba(dataset["validation"]["text"])
        prediction_prob = DataFrame(pred_prob.tolist())

        prediction_prob["file"] = dataset["validation"]["file"]
        prediction_prob = prediction_prob.set_axis(colnames, axis=1)
        prediction_prob.set_index("file").to_csv(path_prediction_prob, sep="\t", compression="gzip")

        pred_cat = model.predict(dataset["validation"]["text"])
        prediction_cat = DataFrame(pred_cat.tolist())

        prediction_cat["file"] = dataset["validation"]["file"]
        prediction_cat = prediction_cat.set_axis(colnames, axis=1)
        prediction_cat.set_index("file").to_csv(path_prediction_cat, sep="\t", compression="gzip")

        fine = [[f for f, p in zip(features, ps) if p] for ps in pred_cat]
        fine = [";".join(cat).replace("", "Other") if not cat else ";".join(cat) for cat in fine]
        coarse = ["Other" if cat == "Other" else ";".join(re.findall("((?:URW|CC):.+?):", cat)) for cat in fine]
        prediction_cat_target = DataFrame({"file": prediction_cat["file"], "coarsegrained": coarse, "finegrained": fine})
        prediction_cat_target.set_index("file").to_csv(path_prediction_cat_target, sep="\t", header=False)