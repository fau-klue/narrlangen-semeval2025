#!/usr/bin/env python3

import argparse
from typing import List

import torch
import evaluate
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer, RobertaConfig, XLMRobertaConfig
from hierarchical_model import (RoBERTaForMultiLabelSequenceClassification,
                                XLMRoBERTaForMultiLabelSequenceClassification,
                                DataCollatorForMultiLabelClassification)

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse_dev", action="append", help="Path to dev dataset with coarse labels")
    parser.add_argument("--fine_dev", action="append",  help="Path to dev dataset with fine labels")
    parser.add_argument("--multi-lingual", action="store_true")
    parser.add_argument("--use_description_from", default=None, help="Path to description file as additional training data")
    parser.add_argument("--model_name_or_path", help="Path to transformer model")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of batches before updating models parameters")
    parser.add_argument("--epoch", type=int, default=5, help="Number of epochs")
    parser.add_argument("--fine_tuned_model", default="multi_label_model", help="Path for saving trained model")
    parser.add_argument("--confidence_threshold", default=0.5, type=float, help="Confidence threshold for multi labels predictions")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--predict", action="store_true", help="Make predictions")
    return parser.parse_args()


class CoarseFineDatasetsLoader:
    def __init__(self, path_to_coarse_dataset:str|List[str], path_to_fine_dataset:str|List[str]):
        self.loaded_coarse_dataset = load_dataset("csv", data_files=path_to_coarse_dataset)
        self.loaded_fine_dataset = load_dataset("csv", data_files=path_to_fine_dataset)

    def __call__(self):
        coarse_fine_mapped_dataset = dict(file=[], text=[], coarse_labels=[], fine_labels=[])
        coarse_grained_data = self._create_dataset(self.loaded_coarse_dataset)
        fine_grained_data = self._create_dataset(self.loaded_fine_dataset)
        for i,_ in enumerate(coarse_grained_data["file"]):
            assert coarse_grained_data["file"][i] == fine_grained_data["file"][i], f"File doesn't match (coarse_grained_data['file'][i], fine_grained_data['file'][i])"
            coarse_fine_mapped_dataset["file"].append(coarse_grained_data["file"][i])
            coarse_fine_mapped_dataset["text"].append(coarse_grained_data["text"][i])
            coarse_fine_mapped_dataset["coarse_labels"].append(coarse_grained_data["labels"][i])
            coarse_fine_mapped_dataset["fine_labels"].append(fine_grained_data["labels"][i])
        return Dataset.from_dict(coarse_fine_mapped_dataset)

    def coarse_labels_ids_mappings(self):
        labels2ids, ids2labels = self.labels_ids_mappings(self.loaded_coarse_dataset)
        return labels2ids, ids2labels

    def fine_labels_ids_mappings(self):
        labels2ids, ids2labels  = self.labels_ids_mappings(self.loaded_fine_dataset)
        return labels2ids, ids2labels

    @staticmethod
    def _create_dataset(inp_dataset):
        dataset = dict(file=[], text=[], labels=[])
        for i,_ in enumerate(inp_dataset["train"]):
            dataset["file"].append(inp_dataset["train"][i]["file"])
            dataset["text"].append(inp_dataset["train"][i]["text"])
            dataset["labels"].append([inp_dataset["train"][i][key] for key in inp_dataset["train"][i].keys() if key not in ["file", "text"]])
        return dataset

    @staticmethod
    def labels_ids_mappings(inp_dataset):
        """ Create labels ids mapping for later uses."""
        labels2ids = {}
        ids2labels = {}
        list_of_labels = [label for label in inp_dataset["train"].features.keys() if label not in ["file", "text"]]
        for i, label in enumerate(list_of_labels):
            labels2ids.update({label: i})
            ids2labels.update({i: label})
        return labels2ids, ids2labels

    @staticmethod
    def _extract_domain(input_string, alternative="Other"):
        domain = None
        if "CC:" not in input_string and "URW:" not in input_string:
            domain = alternative
        else:
            if "CC:" in input_string:
                domain = "Climate Change"
            if "URW:" in input_string:
                domain = "Ukraine-Russia War"
        return domain

class Tokenization:
    def __init__(self, model_name_or_path, max_length: int=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.max_length = max_length

    def __call__(self, features: Dataset):
        tokenized_features = features.map(self.tokenize,
                                          batched=True)
        return tokenized_features

    def tokenize(self, examples):
        tokenized_inputs = dict(file=[],
                                input_ids=[],
                                attention_mask=[],
                                coarse_labels=[],
                                fine_labels=[]
                         )
        tokenized = self.tokenizer(examples["text"],
                                   max_length=self.max_length,
                                   truncation=True,
                                   padding=True,
                                   return_tensors="pt"
                                )
        for i,_ in enumerate(tokenized["input_ids"]):
            tokenized_inputs["file"].append(examples["file"][i])
            tokenized_inputs["input_ids"].append(tokenized["input_ids"][i])
            tokenized_inputs["attention_mask"].append(tokenized["attention_mask"][i])
            tokenized_inputs["coarse_labels"].append(examples["coarse_labels"][i])
            tokenized_inputs["fine_labels"].append(examples["fine_labels"][i])
        return Dataset.from_dict(tokenized_inputs)

class LossLabelsWeights:
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, normalised=False):
        if not normalised:
            return self.get_weights()
        else:
            return self.get_weights() / self.get_weights().sum()

    def get_weights(self):
        return self._get_labels_frequencies() / self._get_number_of_samples()

    def _get_number_of_samples(self):
        return len(self.labels)

    def _get_labels_frequencies(self):
        return torch.tensor(self.labels, dtype=torch.float64).sum(dim=0)

def predict(model, dataset, ids2coarse, ids2fine, confidence_threshold=0.5):
    input_ids = torch.tensor(dataset["input_ids"], dtype=torch.int64)
    attention_mask = torch.tensor(dataset["attention_mask"], dtype=torch.int64)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
    coarse_predictions = labels_decoder(dataset["file"], (sigmoid(np.array(logits["coarse_logits"])) > confidence_threshold), ids2coarse)
    fine_predictions = labels_decoder(dataset["file"], (sigmoid(np.array(logits["fine_logits"])) > confidence_threshold), ids2fine)
    return coarse_predictions, fine_predictions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_metrics(p):
    metrics = evaluate.load("f1")
    predictions, references = p
    fine_preds = sigmoid(predictions[1]).astype(int).reshape(-1)
    fine_labels = references[1].astype(int).reshape(-1)
    metrics_fine_labels = metrics.compute(predictions=fine_preds, references=fine_labels)
    return metrics_fine_labels

def labels_decoder(file_ids, predictions, ids_to_labels_map):
    sparse_array = np.where(predictions)
    file_pos = sparse_array[0]
    label_pos = sparse_array[1]
    current_file = None
    decoded_labels = []
    labels = []
    for i, _ in enumerate(file_ids):
        if i not in file_pos:
            labels.append([ids_to_labels_map[len(ids_to_labels_map)-1]])
        else:
            file_pos_ids = np.where(file_pos == i)[0].tolist()
            for pos_id in file_pos_ids:
                decoded_labels.append(ids_to_labels_map[label_pos[pos_id]])
            labels.append(decoded_labels)
            decoded_labels = []
    return labels


if __name__=="__main__":
    args = arguments()
    # args.multi_lingual
    # args.coarse_train = "../../task_data/subtask2_hf/EN_coarse-grained/train.csv"
    # args.fine_train = "../../task_data/subtask2_hf/EN_fine-grained/train.csv"
    # args.coarse_dev = "../../task_data/subtask2_hf/EN_coarse-grained/dev.csv"
    # args.fine_dev = "../../task_data/subtask2_hf/EN_fine-grained/dev.csv"
    # args.model_name_or_path = "FacebookAI/roberta-base"
    # args.learning_rate = 3e-5
    # args.batch_size = 8
    # args.gradient_accumulation_steps = 1
    # args.epoch = 5
    # args.train = True
    # args.predict = True

    PATH_COARSE_DEV = args.coarse_dev
    PATH_FINE_DEV = args.fine_dev
    MODEL_NAME_OR_PATH = args.model_name_or_path
    base_model_name = MODEL_NAME_OR_PATH.split("/")[1]

    # Data
    if args.multi_lingual:
        train_dataset_loader = CoarseFineDatasetsLoader(path_to_coarse_dataset=["../../task_data/subtask2_hf/EN_coarse-grained/train.csv",
                                                                                "../../task_data/subtask2_hf/BG_coarse-grained/train.csv",
                                                                                "../../task_data/subtask2_hf/PT_coarse-grained/train.csv",
                                                                                "../../task_data/subtask2_hf/HI_coarse-grained/train.csv",
                                                                                "../../task_data/subtask2_hf/RU_coarse-grained/train.csv"
                                                                              ],
                                                        path_to_fine_dataset=["../../task_data/subtask2_hf/EN_fine-grained/train.csv",
                                                                              "../../task_data/subtask2_hf/BG_fine-grained/train.csv",
                                                                              "../../task_data/subtask2_hf/PT_fine-grained/train.csv",
                                                                              "../../task_data/subtask2_hf/HI_fine-grained/train.csv",
                                                                              "../../task_data/subtask2_hf/RU_fine-grained/train.csv"]
                                                        )
    else:
        train_dataset_loader = CoarseFineDatasetsLoader(path_to_coarse_dataset="../../task_data/subtask2_hf/EN_coarse-grained/train.csv", path_to_fine_dataset="../../task_data/subtask2_hf/EN_fine-grained/train.csv")
    dev_dataset_loader  = CoarseFineDatasetsLoader(path_to_coarse_dataset=PATH_COARSE_DEV, path_to_fine_dataset=PATH_FINE_DEV)

    train_coarse_dataset = train_dataset_loader.loaded_coarse_dataset
    coarselabels2ids, ids2coarselabels = train_dataset_loader.coarse_labels_ids_mappings()
    finelabels2ids, ids2finelabels = train_dataset_loader.fine_labels_ids_mappings()
    num_coarse_labels = len(coarselabels2ids)
    num_fine_labels = len(finelabels2ids)

    # Set config
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=MODEL_NAME_OR_PATH)
    config.num_coarse_labels =  num_coarse_labels
    config.num_fine_labels = num_fine_labels
    config.coarselabels2ids = coarselabels2ids
    config.ids2coarselabels = ids2coarselabels
    config.finelabels2ids = finelabels2ids
    config.ids2finelabels = ids2finelabels

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME_OR_PATH)
    data_collator = DataCollatorForMultiLabelClassification(tokenizer=tokenizer)

    tokenization = Tokenization(MODEL_NAME_OR_PATH)
    tokenizer = tokenization.tokenizer

    # Run training
    if args.train:
        train_dataset = train_dataset_loader()
        tokenized_train = tokenization.tokenize(train_dataset)

        # Get train labels weights
        coarse_labels_weights = LossLabelsWeights(tokenized_train["coarse_labels"])(normalised=True)
        fine_labels_weights = LossLabelsWeights(tokenized_train["fine_labels"])(normalised=True)

        # Initialise training arguments and trainer
        if "xlm" in base_model_name.lower():
            model = XLMRoBERTaForMultiLabelSequenceClassification.from_pretrained(pretrained_model_name_or_path=MODEL_NAME_OR_PATH,
                                                                              config=config,
                                                                              coarse_labels_weights=coarse_labels_weights,
                                                                              fine_labels_weights=fine_labels_weights
                                                                              )
        else:
            model = RoBERTaForMultiLabelSequenceClassification.from_pretrained(pretrained_model_name_or_path=MODEL_NAME_OR_PATH,
                                                                           config=config,
                                                                           coarse_labels_weights=coarse_labels_weights,
                                                                           fine_labels_weights=fine_labels_weights
                                                                           )
        training_args = TrainingArguments(
            output_dir=args.fine_tuned_model,
            learning_rate= args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            do_train=True,
            do_eval=True,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            eval_strategy="no",
            save_strategy="no",
            push_to_hub=False,
            overwrite_output_dir=True
            )
        hf_trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
            )
        print(training_args)
        train_results = hf_trainer.train()
        hf_trainer.save_metrics(split="train", metrics=train_results.metrics)
        hf_trainer.save_model()
        hf_trainer.save_state()
    
       # preds = hf_trainer.predict(tokenized_dev)
    # Make predictions
    if args.predict:
        if args.multi_lingual:
            config = XLMRobertaConfig.from_pretrained(args.fine_tuned_model, output_hidden_states=True)
            model = XLMRoBERTaForMultiLabelSequenceClassification.from_pretrained(args.fine_tuned_model, config=config)
        else:
            config = RobertaConfig.from_pretrained(args.fine_tuned_model, output_hidden_states=True)
            model = RoBERTaForMultiLabelSequenceClassification.from_pretrained(args.fine_tuned_model, config=config)
        dev_dataset = dev_dataset_loader()
        tokenizer = Tokenization(args.fine_tuned_model).tokenizer
        tokenized_dev = tokenization.tokenize(dev_dataset)
        coarse_predictions, fine_predictions = predict(model, tokenized_dev, ids2coarselabels, ids2finelabels, confidence_threshold=args.confidence_threshold)

        assert len(coarse_predictions) == len(dev_dataset["file"]), f"{len(coarse_predictions)}, {len(dev_dataset['file'])}"
        assert len(fine_predictions) == len(dev_dataset["file"]), f"{len(fine_predictions)}, {len(dev_dataset['file'])}"
        # write predictions to tsv file
        if args.use_description_from is None:
            with open(f"./predictions_subtask_2_{base_model_name}.txt", "w", encoding="utf-8") as pred_file:
                for i, filename in enumerate(dev_dataset["file"]):
                    pred_file.write(f"{filename}\t{";".join(coarse_predictions[i])}\t{";".join(fine_predictions[i])}\n")
        else:
            with open(f"./predictions_subtask_2_w_description_{base_model_name}.txt", "w", encoding="utf-8") as pred_file:
                for i, filename in enumerate(dev_dataset["file"]):
                    pred_file.write(f"{filename}\t{";".join(coarse_predictions[i])}\t{";".join(fine_predictions[i])}\n")
