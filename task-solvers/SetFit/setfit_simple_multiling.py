from datasets import load_dataset, concatenate_datasets
from setfit import SetFitModel, Trainer
from sklearn.metrics import classification_report
from typing import Any
from optuna import Trial


# Model-specific hyperparameters
def model_init(params: dict[str, Any]) -> SetFitModel:
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    model_id = params.get("model_id", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    model_params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return SetFitModel.from_pretrained(model_id, multi_target_strategy="one-vs-rest", **model_params)


# Training hyperparameters
def hp_space(trial: Trial) -> dict[str, float | int | str]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 3),  # 1, 5
        "batch_size": trial.suggest_categorical("batch_size", [16]),  # [4, 8, 16, 32]
        "num_iterations": trial.suggest_categorical("num_iterations", [20]),  # [5, 10, 20, 40]  # if not used, sampling_strategy defaults to "oversampling", see https://huggingface.co/docs/setfit/conceptual_guides/sampling_strategies
        "seed": trial.suggest_int("seed", 1, 42),
        #"max_iter": trial.suggest_int("max_iter", 50, 300),
        #"solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
        "model_id": trial.suggest_categorical(
            "model_id",
            [
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                # "sentence-transformers/distiluse-base-multilingual-cased-v1"
            ],
        ),
    }


def encode_labels(record) -> dict[str, list[list[int]]]:
    return {"labels": [record[feature] for feature in features]}


if __name__ == '__main__':

    path_dataset1 = "../../task_data/subtask2_hf/EN_fine-grained/"
    path_dataset2 = "../../task_data/subtask2_hf/BG_fine-grained/"
    path_dataset3 = "../../task_data/subtask2_hf/HI_fine-grained/"
    path_dataset4 = "../../task_data/subtask2_hf/PT_fine-grained/"
    path_dataset5 = "../../task_data/subtask2_hf/RU_fine-grained/"
    path_model = "../../models/setfit/setfit_simple_multiling"

    # Load our prepared datasets
    dataset1 = load_dataset(path_dataset1, download_mode='force_redownload')
    dataset2 = load_dataset(path_dataset2, download_mode='force_redownload')
    dataset3 = load_dataset(path_dataset3, download_mode='force_redownload')
    dataset4 = load_dataset(path_dataset4, download_mode='force_redownload')
    dataset5 = load_dataset(path_dataset5, download_mode='force_redownload')

    # Labels to be predicted
    features = dataset1["train"].column_names
    for col in ["file", "text"]:
        features.remove(col)

    # Single label feature
    dataset1 = dataset1.map(encode_labels)
    dataset2 = dataset2.map(encode_labels)
    dataset3 = dataset3.map(encode_labels)
    dataset4 = dataset4.map(encode_labels)
    dataset5 = dataset5.map(encode_labels)

    # Define final training data (concatenation or sample)
    train_dataset = concatenate_datasets([dataset1["train"], dataset2["train"], dataset3["train"], dataset4["train"], dataset5["train"]])
    eval_dataset = concatenate_datasets([dataset1["validation"], dataset2["validation"], dataset3["validation"], dataset4["validation"], dataset5["validation"]])

    trainer = Trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_init=model_init,
        metric="f1",
        metric_kwargs={"average": "micro"},
        column_mapping={"text": "text", "labels": "label"} # Map dataset columns to text/label expected by trainer
    )

    # Find the best model (optuna needs to be installed)
    best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=1) # 10 or even 100

    # Train the best model and evaluate
    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    trainer.train()
    metrics = trainer.evaluate(eval_dataset)
    print(metrics)

    preds = trainer.model.predict(eval_dataset["text"])
    preds_labels = [[f for f, p in zip(features, ps) if p] for ps in preds] # in text form

    print(classification_report(eval_dataset["labels"], preds.tolist()))

    # Push model to the Hub
    # trainer.push_to_hub("AndreasBlombach/setfit_schwurpert_train_desc_optimised")

    # Save locally
    trainer.model.save_pretrained(path_model)
