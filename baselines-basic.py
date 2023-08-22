# -*- coding: utf-8 -*-
"""
Script for running simple baselines experiments with optuna optimization.

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
import optuna
import warnings
import argparse
import itertools

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif


SEED = 17
PATH_OUTPUT = 'output'
TARGET_METRIC = "f1-score"
FEATURE_VECTORIZER_ARGS = {"stop_words": "english"}  # TfidfVectorizer args

# Hyperparameter search
HP_N_TRIALS = 100
HP_RANGE_LR = [1e-7, 1e-3]
HP_RANGE_STEPS = [50, 10000]
OPTUNA_STORAGE = "sqlite:///db_baselines.sqlite3"
MODELS = {"LR": LogisticRegression, "SVM": SVC}

DATASETS = ["AVEC_16_data", "AVEC_19_data", "AVEC_19_data-dev"]
FEATURES = [0, -1, 100, 250, 500, 1000, 1500]  # 0 no feature selection; -1 auto. feature selection; n top-n feature selection
MODELS_NAMES = sorted(list(MODELS.keys()))
OPTIONS = list(itertools.product(DATASETS, MODELS_NAMES, FEATURES))
OPTIONS_N = len(OPTIONS)


def get_task_info(option, only_description=False):
    dataset, model, feature_size = option
    if feature_size == 0:
        feature_selection = False
    else:
        feature_selection = True
    study_name = f"({dataset}){model}[features{feature_size}]"

    feature_str = 'all' if not feature_selection else ('auto' if feature_size == -1 else f'top-{feature_size}')
    dataset_str = f"{dataset[:7]} (eval on {'test' if dataset == 'AVEC_19_data' else 'dev'}set)"
    option_description = f"Dataset={dataset_str}; Model: {model}; Features={feature_str};"

    if only_description:
        return option_description

    return dataset, model, feature_size, feature_selection, study_name, option_description


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

DATASET, MODEL, FEATURE_SIZE, FEATURE_SELECTION, STUDY_NAME, description = get_task_info(OPTIONS[task_ix])

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

warnings.filterwarnings('ignore')


def load_dataset(path, label2ix=None):
    with open(path, 'r') as f:
        f_lines = f.read().split('\n')
    data = [(line.split('\t')[0], line.split('\t')[1]) for line in f_lines if line]
    document_labels = [label for label, _ in data]
    documents = [doc for _, doc in data]
    labels = sorted(list(set(document_labels)))

    ix2label = {}
    if label2ix is None:
        label2ix = {}
        for label in labels:
            label2ix[label] = len(label2ix)
            ix2label[label2ix[label]] = label

    y = [label2ix[lbl] for lbl in document_labels]

    return documents, y, ix2label, label2ix


def train_and_eval(c, penalty, class_weight, print_results=False):
    FEATURE_VECTORIZER_ARGS["vocabulary"] = None
    doc_vectorizer = TfidfVectorizer(**FEATURE_VECTORIZER_ARGS)

    if FEATURE_SELECTION:
        doc_vectorizer.fit(X_train)
        vocab = doc_vectorizer.get_feature_names_out()
        print("Original vocab size:", len(vocab))
        if FEATURE_SIZE == -1:
            estimator = LogisticRegression(class_weight='balanced', dual=False,
                                           fit_intercept=True, penalty='none',
                                           solver='newton-cg', random_state=SEED, n_jobs=-1)
            selector = SelectFromModel(estimator=estimator).fit(doc_vectorizer.transform(X_train), y_train)
        else:
            selector = SelectKBest(f_classif, k=FEATURE_SIZE).fit(doc_vectorizer.transform(X_train), y_train)
        support = selector.get_support()
        vocab = [vocab[i] for i in range(len(vocab)) if support[i]]
        print("Vocab size after feature selection:", len(vocab))

        FEATURE_VECTORIZER_ARGS["vocabulary"] = vocab
        doc_vectorizer = TfidfVectorizer(**FEATURE_VECTORIZER_ARGS)

    if MODEL == "LR":
        clf = MODELS[MODEL](C=c, class_weight=class_weight, penalty=penalty, random_state=SEED, n_jobs=-1)
    else:
        clf = MODELS[MODEL](C=c, class_weight=class_weight, random_state=SEED)
    clf.fit(doc_vectorizer.fit_transform(X_train), y_train)
    y_pred = clf.predict(doc_vectorizer.transform(X_dev)).tolist()

    result = classification_report(y_dev, y_pred, target_names=labels, zero_division=0, output_dict=not print_results)

    if print_results:
        print(result)
    else:
        return result[TARGET_METRIC] if TARGET_METRIC == 'accuracy' else result["macro avg"][TARGET_METRIC]


if __name__ == "__main__":
    os.makedirs(PATH_OUTPUT, exist_ok=True)

    X_train, y_train, ix2label, label2ix = load_dataset(PATH_TRAIN)
    X_dev, y_dev, _, _ = load_dataset(PATH_DEV, label2ix=label2ix)

    n_labels = len(ix2label)
    labels = [ix2label[ix] for ix in range(n_labels)]

    best_metric = float("-inf")  # https://github.com/optuna/optuna/issues/2575

    def objective(trial):
        global best_metric

        penalty = trial.suggest_categorical("penalty", ['l2', 'none']) if MODEL == "LR" else None

        current_metric = train_and_eval(
            trial.suggest_float("C", 0.001, 10, log=True),
            penalty,
            trial.suggest_categorical("class_weight", ['balanced', None])
        )
        if current_metric > best_metric:
            best_metric = current_metric

        return best_metric

    study = optuna.create_study(
        storage=OPTUNA_STORAGE,
        pruner=optuna.pruners.NopPruner(),
        study_name=STUDY_NAME,
        direction="maximize",
        load_if_exists=True)

    study.optimize(objective, n_trials=HP_N_TRIALS)

    train_and_eval(
        study.best_params["C"],
        study.best_params["penalty"] if "penalty" in study.best_params else 0,
        study.best_params["class_weight"],
        print_results=True
    )

    study.set_user_attr("model", MODEL.upper())
    study.set_user_attr("dataset", DATASET)
    study.set_user_attr("features selection", "all" if not FEATURE_SELECTION else ("auto" if FEATURE_SIZE == -1 else f"top-{FEATURE_SIZE}"))
    study.set_user_attr(f"best {TARGET_METRIC}", study.best_value)
