# -*- coding: utf-8 -*-
"""
Script for running BERT-based baselines experiments with optuna optimization.

Copyright (c) 2023 Idiap Research Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import torch
import optuna
import datasets
import argparse
import itertools
import numpy as np

from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from main import load_dataset

SEED = 42
TARGET_METRIC = "f1-score"  # as in classification_report
EVALUATION_STEPS = 10
PATH_RESULTS = "output"

# Hyperparameter search
HP_N_TRIALS = 80
HP_RANGE_LR = [1e-6, 1e-1]
HP_RANGE_EPOCHS = [1, 10]
OPTUNA_STORAGE = "sqlite:///db_baselines.sqlite3"

DATASETS = ["AVEC_16_data", "AVEC_19_data", "AVEC_19_data-dev"]
MODELS = ["bert-base-cased", "bert-base-uncased", "bert-large-cased", "bert-large-uncased", "roberta-base", "roberta-large"]
OPTIONS = list(itertools.product(DATASETS, MODELS, (True, False)))
OPTIONS_N = len(OPTIONS)


def get_task_info(option, only_description=False):
    dataset, model, finetune = option

    batch_size = 8 if "large" in model else 16
    study_name = f"({dataset}){model}[{'fine-tuned' if finetune else 'pre-trined'}]"
    path_results = os.path.join(PATH_RESULTS, study_name)

    dataset_str = f"{dataset[:7]} (eval on {'test' if dataset == 'AVEC_19_data' else 'dev'}set)"
    option_description = f"Dataset={dataset_str}; Model: {model}; Finetuned={finetune};"

    if only_description:
        return option_description

    return dataset, model, finetune, batch_size, path_results, study_name, option_description


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--task-id", help="The task id to be run", default=-1, required=False, type=int)
args = parser.parse_args()

if args.task_id < 0:
    print(f"\nThere's a total of {OPTIONS_N} task options. List by task id:")
    for ix, op in enumerate(OPTIONS):
        print(f"  {ix + 1}.", get_task_info(op, only_description=True))
    print()
    exit()
elif args.task_id is not None:
    if 0 < args.task_id <= OPTIONS_N:
        task_ix = args.task_id - 1
    else:
        raise ValueError(f"argument --task-id must be a valid task id (i.e. integer between 1 and {OPTIONS_N})")
elif os.getenv('SGE_TASK_ID') is not None:
    task_ix = int(os.getenv('SGE_TASK_ID')) - 1
    print(f"(JOB with index {task_ix} will run task {id})")
else:
    print("(No task id was provided, using the first one as default)")
    task_ix = 0

DATASET, MODEL, FINETUNE, BATCH_SIZE, PATH_RESULTS, STUDY_NAME, description = get_task_info(OPTIONS[task_ix])

print(f"\nSelected option to run is {task_ix + 1}.")
print("  Task description:", description)
print(f"  Optuna study name: '{STUDY_NAME}'\n")

PATH_TRAIN = f"data/{DATASET}/train_all_data.txt"
PATH_DEV = f"data/{DATASET}/test_all_data.txt"
PATH_TEST = ''

if DATASET.endswith("-dev"):
    PATH_TRAIN = f"data/{DATASET[:-4]}/train_all_data.txt"
    PATH_DEV = f"data/{DATASET[:-4]}/devel_all_data.txt"
    PATH_TEST = f"data/{DATASET[:-4]}/test_all_data.txt"

training_arguments = {
    "output_dir": PATH_RESULTS,
    "learning_rate": None,
    "num_train_epochs": None,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": TARGET_METRIC,
    "save_total_limit": 1,
    "eval_steps": EVALUATION_STEPS,
    "seed": SEED
}

np.random.RandomState(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    DEVICE = torch.device('cpu')
print("Device:", DEVICE)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class BalancedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
        loss = loss_fct(logits.view(-1, CLASS_WEIGHTS.shape[0]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class CustomMetric(datasets.Metric):
    def __init__(self, metric: str) -> None:
        super(CustomMetric, self).__init__()
        self._metric = metric

    def _info(self):
        return datasets.MetricInfo(
            description='',
            citation='',
            inputs_description='',
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html"],
        )

    def _compute(self, predictions, references, labels=None, pos_label=1, average="macro", sample_weight=None):
        report = classification_report(references, predictions, output_dict=True)
        if self._metric == 'accuracy':
            value = report[self._metric]
        else:
            value = report[f"{average} avg"][self._metric]
        return {self._metric: value}


def tokenize(x_dataset):
    return tokenizer(x_dataset, truncation=True, padding=True)


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels))
    print("Model loaded:", MODEL)
    print("Total number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    for name, param in model.named_parameters():
        if 'classifier' in name:
            print(f"Size of classification linear layer ('{name}'):", param.shape)

    if not FINETUNE:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro')


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate",
                                             HP_RANGE_LR[0], HP_RANGE_LR[1],
                                             log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs",
                                              HP_RANGE_EPOCHS[0], HP_RANGE_EPOCHS[1],
                                              log=True),
    }


def compute_objective(metric):
    # Instead of returning the current value, since there's an issue with optuna (https://github.com/optuna/optuna/issues/2575)
    # that reports the last value of a trial instead of the best intermediate value, I'll always report the best value.
    # So instead of simply:
    # return metric[f"eval_{TARGET_METRIC}"]
    # I'll do, as a workaround (and I'll disable Pruning too):
    global best_metric

    current_metric = metric[f"eval_{TARGET_METRIC}"]
    if (TARGET_METRIC == "loss" and current_metric < best_metric) or \
       (TARGET_METRIC != "loss" and current_metric > best_metric):
        best_metric = current_metric

    return best_metric


metric = CustomMetric(TARGET_METRIC)
best_metric = float("inf") if TARGET_METRIC == "loss" else float("-inf")  # https://github.com/optuna/optuna/issues/2575

print("Loading dataset...")
X_train, y_train, ix2label, label2ix = load_dataset(PATH_TRAIN)
X_dev, y_dev, _, _ = load_dataset(PATH_DEV, label2ix=label2ix)
if PATH_TEST:
    X_test, y_test, _, _ = load_dataset(PATH_TEST, label2ix=label2ix)
labels = [ix2label[ix] for ix in range(len(ix2label))]


CLASS_WEIGHTS = torch.tensor(compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train),
                                                  y=np.array(y_train)), dtype=torch.float)
CLASS_WEIGHTS = CLASS_WEIGHTS.to(DEVICE)

print("Dataset loaded: ")
print(f"    - Training set: {len(y_train)} instances")
print(f"    - Evaluation set: {len(y_dev)} instances")
if PATH_TEST:
    print(f"    - Test set: {len(y_test)} instances")

print("Loading tokenizer and tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
train_dataset = MyDataset(tokenize(X_train), y_train)
dev_dataset = MyDataset(tokenize(X_dev), y_dev)
if PATH_TEST:
    test_dataset = MyDataset(tokenize(X_test), y_test)

trainer = BalancedTrainer(
    model_init=model_init,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    args=TrainingArguments(**training_arguments),
)

best_trial = trainer.hyperparameter_search(
    hp_space=optuna_hp_space,
    compute_objective=compute_objective,
    n_trials=HP_N_TRIALS,
    direction="minimize" if TARGET_METRIC == "loss" else "maximize",
    backend="optuna",
    storage=OPTUNA_STORAGE,
    study_name=STUDY_NAME,
    load_if_exists=True,
    pruner=optuna.pruners.NopPruner()
)

print(best_trial)

study = optuna.create_study(study_name=STUDY_NAME,
                            storage=OPTUNA_STORAGE,
                            load_if_exists=True)
study.set_user_attr("model", MODEL)
study.set_user_attr("dataset", DATASET)
study.set_user_attr("fine tune", FINETUNE)
study.set_user_attr(f"best {TARGET_METRIC}", study.best_value)

for arg in best_trial.hyperparameters:
    training_arguments[arg] = best_trial.hyperparameters[arg]

del trainer

training_arguments["save_steps"] = EVALUATION_STEPS
trainer = BalancedTrainer(
    model_init=model_init,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    args=TrainingArguments(**training_arguments),
)

print("Training...")
trainer.train()

print("EVAL")
val_preds_logits, val_labels, _ = trainer.predict(dev_dataset)
val_preds = np.argmax(val_preds_logits, axis=-1)
print(f"Results for {STUDY_NAME}")
print(classification_report(val_labels, val_preds, target_names=labels, digits=3))

if PATH_TEST:
    print("TEST SET")
    test_preds_logits, test_labels, _ = trainer.predict(test_dataset)
    test_preds = np.argmax(test_preds_logits, axis=-1)
    print(f"Results for {STUDY_NAME}")
    print(classification_report(test_labels, test_preds, target_names=labels, digits=3))
