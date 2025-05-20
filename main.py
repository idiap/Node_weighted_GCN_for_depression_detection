# -*- coding: utf-8 -*-
"""
Script for running the InducT-GCN experiments with optuna optimization.

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
import json
import pickle
import random
import optuna
import warnings
import argparse
import itertools
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest, f_classif

from collections import defaultdict, Counter
from datetime import datetime as dt
from math import log
from tqdm import tqdm

warnings.filterwarnings('ignore')


SEED = 17
SAVE_MODEL_AS_FINAL = True
PATH_OUTPUT = 'output'
PATH_OUTPUT_MODEL = 'model'

EMBEDDING_DIM = 64
DROPOUT = .5
MIN_FREQ = 1  # Min word/token frequency
WINDOW_SIZE = 3  # for PMI computation
ACTIVATION_FUNC = nn.ReLU()

USE_WEIGHTED_PMI = False  # weight PMI by co-occurrence distance
USE_P_TOK2TOK_WEIGHT = False  # weight token-to-token edges by P(wi->wj) where wi and wj are tokens (words, char ngrams, etc.)
USE_WORDPIECE = False

TARGET_METRIC = "f1-score"

VALIDATION_PERIOD = 1
NO_IMPROVEMENT_LIMIT = 2500
CLASS_WEIGHT = TARGET_METRIC != "accuracy"
FEATURE_VECTORIZER_ARGS = {"stop_words": "english"}  # TfidfVectorizer args
STEMMER = None  # e.g. SnowballStemmer('english')

# Hyperparameter search
HP_N_TRIALS = 100
HP_RANGE_LR = [1e-7, 1e-3]
HP_RANGE_STEPS = [50, 10000]
OPTUNA_STORAGE = "sqlite:///db_inductgcn.sqlite3"

DATASETS = ["AVEC_16_data", "AVEC_19_data-dev", "AVEC_19_data"]
FEATURES = [0, -1, 100, 250, 500, 1000, 1500]
OPTIONS = list(itertools.product(DATASETS, FEATURES, (True, False)))
OPTIONS_N = len(OPTIONS)


def get_task_info(option, only_description=False):
    dataset, feature_size, use_pagerank = option
    if feature_size == 0:
        feature_selection = False
    else:
        feature_selection = True
    study_name = f"({dataset})induct-gcn[{'pagerank' if use_pagerank else 'original'}-features{feature_size}]"

    feature_str = 'all' if not feature_selection else ('auto' if feature_size == -1 else f'top-{feature_size}')
    dataset_str = f"{dataset[:7]} (eval on {'test' if dataset == 'AVEC_19_data' else 'dev'}set)"
    option_description = f"Dataset={dataset_str}; Features={feature_str}; Pagerank={use_pagerank};"

    if only_description:
        return option_description

    return dataset, feature_size, use_pagerank, feature_selection, study_name, option_description


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--task-id", help="The task id to be run", default=-1, required=False, type=int)
args = parser.parse_args()


class InducTGCN(nn.Module):
    """InducT-GCN model class definition"""
    def __init__(self, embedding_dim, labels, dropout, vectorizer):
        super(InducTGCN, self).__init__()

        self.Conv_0 = None
        self.Conv_0_Test = None
        self.A_norm = None
        self.A_B = None
        self.H_0_wB = None
        self.H_1_words = None

        self.classes_ = labels

        self._doc_vectorizer = vectorizer
        self._tokenizer = self._doc_vectorizer.build_analyzer()
        self._vocab = self._doc_vectorizer.vocabulary_

        self.get_H_1 = nn.Sequential(
            nn.Linear(len(self._vocab), embedding_dim, bias=False),
            ACTIVATION_FUNC,
            nn.Dropout(dropout, inplace=False)
        )
        self.node_emb2out = nn.Linear(embedding_dim, len(labels), bias=False)

    def build_graph(self, train_documents, window_size=20, verbose=True):
        if verbose:
            print(f"[InducTGCN {dt.now()}] Building vocabulary...")
        self._window_size = window_size

        # TF-IDF
        if verbose:
            print(f"[InducTGCN {dt.now()}] Computing TF-IDF values...")
        H_0_docs = torch.tensor(self._doc_vectorizer.transform(train_documents).todense(), dtype=torch.float)
        vocab_size = len(self._vocab)
        if verbose:
            print(f"[InducTGCN {dt.now()}] Vocabulary size is {vocab_size}")

        if USE_P_TOK2TOK_WEIGHT:
            if verbose:
                print(f"[InducTGCN {dt.now()}] Computing token sequence frequencies...")
            seq_fr = defaultdict(lambda: defaultdict(int))
            # Tried using the same A matrix to store the values but it was actually around 10 times slower than defaultdict
            for doc in tqdm([self._tokenizer(doc) for doc in train_documents], disable=not verbose):
                for ix in range(len(doc) - 1):
                    seq_fr[doc[ix]][doc[ix + 1]] += 1
        else:
            if verbose:
                print(f"[InducTGCN {dt.now()}] Computing frequencies for PMI values...")
            total_windows = 0
            tok2win_fr = defaultdict(int)
            cotok2win_fr = defaultdict(lambda: defaultdict(int))
            # Tried using the same A matrix to store the values but it was actually around 10 times slower...
            for doc in tqdm([self._tokenizer(doc) for doc in train_documents], disable=not verbose):
                doc = [w for w in doc if w in self._vocab]  # only keep words that are part of the vocab
                finish_sliding = False
                ix = 0
                while not finish_sliding:
                    window = set(doc[ix: ix + window_size])
                    for token_i in window:
                        tok2win_fr[token_i] += 1
                        for token_j in window:
                            if token_i != token_j:
                                cotok2win_fr[token_i][token_j] += 1
                    finish_sliding = ix + window_size >= len(doc)
                    total_windows += 1
                    ix += 1

        nodes_n = vocab_size + len(train_documents)
        if verbose:
            print(f"[InducTGCN {dt.now()}] Creaing matrix A ({nodes_n} nodes)...")
        A = torch.zeros((nodes_n, nodes_n), dtype=torch.float)

        if USE_P_TOK2TOK_WEIGHT:
            # Aij = P(i->j) if i and j are words
            token2ix = self._vocab
            for token_i in tqdm(seq_fr):
                if token_i in token2ix:
                    i = token2ix[token_i]
                    for token_j in seq_fr[token_i]:
                        if token_j in token2ix:
                            j = token2ix[token_j]
                            A[i, j] = seq_fr[token_i][token_j]
            del seq_fr
            A /= A.sum(dim=1).view((-1, 1))
            torch.nan_to_num(A, neginf=0, out=A)
        else:
            # # Aij = PMI(i, j) if i and j are words
            token2ix = self._vocab
            for token_i in tok2win_fr:
                if token_i in token2ix:
                    i = token2ix[token_i]
                    A[i, i] = tok2win_fr[token_i]
                    for token_j in cotok2win_fr[token_i]:
                        if token_j in token2ix:
                            j = token2ix[token_j]
                            A[i, j] = cotok2win_fr[token_i][token_j]
            del tok2win_fr, cotok2win_fr

            invalid_coocurr = A == 0  # invalid co-occurences
            torch.log(A, out=A)
            torch.nan_to_num(A, neginf=0, out=A)
            tok_win_fr = A[range(vocab_size), range(vocab_size)]
            torch.sub(A[:vocab_size, :vocab_size], tok_win_fr, out=A[:vocab_size, :vocab_size])
            torch.sub(A[:vocab_size, :vocab_size], tok_win_fr.reshape(-1, 1), out=A[:vocab_size, :vocab_size])
            A[:vocab_size, :vocab_size] += log(total_windows)
            A[invalid_coocurr] = 0
            A[A < 0] = 0

        # Aij = Tf-Idf(i, j) if i word and j document
        A[:vocab_size, vocab_size:] = H_0_docs.T

        # Aii = 1
        A[range(nodes_n), range(nodes_n)] = 1

        if USE_WEIGHTED_PMI:
            if verbose:
                print(f"[InducTGCN {dt.now()}] Computing co-ocurrence distances...")
            cotok2distance = defaultdict(lambda: defaultdict(int))
            cotok2count = defaultdict(lambda: defaultdict(int))
            # Tried using the same A matrix to store the values but it was actually around 10 times slower... (than using defaultdict)
            for doc in tqdm([self._tokenizer(doc) for doc in train_documents], disable=not verbose):
                finish_sliding = False
                ix = 0
                while not finish_sliding:
                    window = doc[ix: ix + window_size]
                    for token_i_ix, token_i in enumerate(window):
                        for token_j_ix, token_j in enumerate(window[token_i_ix + 1:]):
                            if token_i != token_j:
                                cotok2distance[token_i][token_j] += token_j_ix
                                cotok2distance[token_j][token_i] += token_j_ix
                                cotok2count[token_i][token_j] += 1
                                cotok2count[token_j][token_i] += 1
                    finish_sliding = ix + window_size >= len(doc)
                    ix += 1
            for token_i in tqdm(cotok2distance, desc="weighting PMI values", disable=not verbose):
                if token_i in token2ix:
                    i = token2ix[token_i]
                    for token_j in cotok2distance[token_i]:
                        if token_j in token2ix:
                            j = token2ix[token_j]
                            distance_weight = (1 - ((cotok2distance[token_i][token_j] / cotok2count[token_i][token_j]) / window_size)) ** .1
                            A[i, j] *= distance_weight
            del cotok2count, cotok2distance

        if verbose:
            print(f"[InducTGCN {dt.now()}] Normalizing matrix A...")
        D_smooth_inverse = torch.inverse(torch.eye(A.shape[0], dtype=torch.float) * A.sum(dim=1).T).sqrt()
        torch.chain_matmul(D_smooth_inverse, A, D_smooth_inverse, out=A)

        if USE_PAGERANK:
            if verbose:
                print(f"[InducTGCN {dt.now()}] Computing vertex values by PageRank algorithm...")
            # A[range(nodes_n), range(nodes_n)] = torch.tensor(page_rank(A), dtype=torch.float)
            A[range(vocab_size), range(vocab_size)] = torch.tensor(page_rank(A[:vocab_size, :vocab_size]), dtype=torch.float)

        H_0 = torch.cat([torch.eye(vocab_size, dtype=torch.float), H_0_docs])
        self.A_norm = torch.tensor(A, device=DEVICE).detach()
        self.Conv_0 = torch.mm(A, H_0).to(DEVICE).detach()

        del H_0_docs, D_smooth_inverse, A, H_0

        if verbose:
            print(f"[InducTGCN {dt.now()}] Graph creation finished successfully")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        original_dict = super().state_dict(destination, prefix, keep_vars)
        original_dict[prefix + 'H_1_words'] = self.H_1_words
        # These two only needed for ploting learned embeddings:
        original_dict[prefix + 'Conv_0'] = self.Conv_0
        original_dict[prefix + 'A_norm'] = self.A_norm
        return original_dict

    def load_state_dict(self, state_dict, strict=True):
        self.H_1_words = state_dict.pop('H_1_words')
        # These two only needed for ploting learned embeddings:
        self.Conv_0 = state_dict.pop('Conv_0')
        self.A_norm = state_dict.pop('A_norm')
        super().load_state_dict(state_dict, strict)

    def cross_entropy_loss_on_document_nodes(self, y, class_weight=None):
        vocab_size = len(self._vocab)

        # modules = [module for k, module in self.get_H_1._modules.items()]
        # H_1 = checkpoint_sequential(modules, 2, self.Conv_0)
        H_1 = self.get_H_1(self.Conv_0)

        if self.H_1_words is None:
            self.H_1_words = H_1[:vocab_size, :].cpu()  # test done in CPU to save GPU memory
        else:
            self.H_1_words[:] = H_1[:vocab_size, :].cpu()  # test done in CPU to save GPU memory

        z_logits = self.node_emb2out(torch.mm(self.A_norm, H_1))[vocab_size:, :]

        return nn.CrossEntropyLoss(weight=class_weight)(z_logits, y)

    def forward(self, test_documents):
        vocab_size = len(self._vocab)
        batch_size = len(test_documents)
        self.eval()
        with torch.no_grad():
            if self.Conv_0_Test is None:
                # TF-IDF
                try:
                    H_0_wB = torch.concat([torch.eye(vocab_size, dtype=torch.float),
                                           torch.tensor(self._doc_vectorizer.transform(test_documents).todense(), dtype=torch.float)])
                except AttributeError:
                    H_0_wB = torch.concat([torch.eye(vocab_size, dtype=torch.float),
                                           torch.tensor(self._doc_vectorizer.transform(test_documents), dtype=torch.float)])

                A_B = torch.zeros((batch_size, vocab_size + batch_size), dtype=torch.float, device="cpu")
                A_B[:, :vocab_size] = H_0_wB[vocab_size:, :]
                A_B[:, vocab_size:] = torch.eye(batch_size, dtype=torch.float)

                self.A_B = A_B.detach()
                self.Conv_0_Test = torch.mm(self.A_B, H_0_wB).cpu().detach()

            H_1_B = self.get_H_1.cpu()(self.Conv_0_Test)

            if self.H_1_words.is_cuda:
                self.H_1_words = self.H_1_words.cpu()

            H_1_wB = torch.concat([self.H_1_words[:vocab_size, :], H_1_B])
            z_logits = self.node_emb2out.cpu()(torch.mm(self.A_B, H_1_wB))

            self.get_H_1.to(DEVICE)
            self.node_emb2out.to(DEVICE)

            return nn.Softmax()(z_logits)

    def save(self, test_documents, path):
        vocab_size = len(self._vocab)
        batch_size = len(test_documents)

        H_0_wB = torch.concat([torch.eye(vocab_size, dtype=torch.float),
                               torch.tensor(self._doc_vectorizer.transform(test_documents).todense(), dtype=torch.float)])
        A_B = torch.zeros((batch_size, vocab_size + batch_size), dtype=torch.float, device="cpu")
        A_B[:, :vocab_size] = H_0_wB[vocab_size:, :]
        A_B[:, vocab_size:] = torch.eye(batch_size, dtype=torch.float)

        self.A_B = A_B

        torch.save({'model_state_dict': self.state_dict(),
                    'A_dev': self.A_B,
                    'classes_': self.classes_,
                    'embedding_dim': self.node_emb2out.weight.shape[1]}, path)

    def predict_proba(self, doc_or_list):
        self.eval()
        if type(doc_or_list) != list:
            with torch.no_grad():
                vocab_size = len(self._vocab)
                batch_size = self.A_B.shape[1] - vocab_size

                doc_vec = torch.tensor(self._doc_vectorizer.transform([doc_or_list]).todense(), dtype=torch.float)

                if self.H_0_wB is None:
                    self.A_B = self.A_B.to(DEVICE)
                    self.H_1_words = self.H_1_words.to(DEVICE)
                    self.get_H_1 = self.get_H_1.to(DEVICE)
                    self.node_emb2out = self.node_emb2out.to(DEVICE)

                    self.H_0_wB = torch.concat([torch.eye(vocab_size, dtype=torch.float, device=DEVICE), self.A_B[:, :vocab_size]])
                    embedding_dim = self.node_emb2out.weight.shape[1]
                    self.H_1_wB = torch.concat([self.H_1_words[:vocab_size, :], torch.zeros((batch_size, embedding_dim), device=DEVICE)])

                self.A_B[0, :vocab_size] = doc_vec
                self.H_0_wB[vocab_size, :] = doc_vec

                H_1_B = self.get_H_1(torch.mm(self.A_B, self.H_0_wB))
                self.H_1_wB[vocab_size:, :] = H_1_B
                z_logits = self.node_emb2out(torch.mm(self.A_B, self.H_1_wB))[0]  # only the given doc logit

                return np.array(nn.Softmax()(z_logits).cpu())
        else:
            return [self.predict_proba(doc) for doc in doc_or_list]

    def predict(self, doc_or_list):
        if type(doc_or_list) != list:
            return self.classes_[np.argmax(self.predict_proba(doc_or_list))]
        else:
            return [self.predict(doc) for doc in doc_or_list]

    def predict_batch(self, test_documents):
        self.Conv_0_Test = None
        y_proba = self.forward(test_documents)
        y_pred_ix = torch.argmax(y_proba, axis=-1).tolist()
        return [self.classes_[ix] for ix in y_pred_ix]


def page_rank(x, df=0.85, max_iter=100, bias=None):
    # Initialize
    A = normalize(sp.csr_matrix(x, dtype=float), axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1, 1)

    # Check bias
    if bias is None:
        bias = (1 - df) * np.ones(A.shape[0]).reshape(-1, 1)
    else:
        bias = bias.reshape(-1, 1)
        bias = A.shape[0] * bias / bias.sum()
        assert bias.shape[0] == A.shape[0]
        bias = (1 - df) * bias

    # Iteration
    for _ in range(max_iter):
        R = df * (A * R) + bias

    return R.T[0]


def load_induct_gcn_model(path_model, path_vectorizer, device=None):
    with open(path_vectorizer, 'rb') as f:
        doc_vectorizer = pickle.load(f)
    state_dict = torch.load(path_model, map_location=device)

    model = InducTGCN(state_dict['embedding_dim'], state_dict['classes_'], 0, doc_vectorizer)

    model.load_state_dict(state_dict['model_state_dict'])
    model.A_B = state_dict['A_dev']
    model.classes_ = state_dict['classes_']
    model.to(device)

    return model


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

    y = torch.tensor([label2ix[lbl] for lbl in document_labels], dtype=torch.long)

    if STEMMER:
        documents = [" ".join([STEMMER.stem(i) for i in word_tokenize(doc)]) for doc in documents]
    return documents, y, ix2label, label2ix


def scale(values, max):
    return [v * max for v in values]


# Functions for saving and loading model parameters and metrics.
def save_checkpoint(path, model, target_metric):
    torch.save({'model_state_dict': model.state_dict(),
                'target_metric': target_metric}, path)


def load_checkpoint(path, model):
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['target_metric']


def save_metrics(path, train_loss_list, steps_list, target_metric_list):
    state_dict = {'train_loss_list': train_loss_list,
                  'target_metric_list': target_metric_list,
                  'steps_list': steps_list}
    torch.save(state_dict, path)


def load_metrics(path):
    state_dict = torch.load(path, map_location=DEVICE)
    return state_dict['steps_list'], state_dict['train_loss_list'], state_dict['target_metric_list']


def train(model,
          y_train,
          x_valid,
          y_valid,
          optimizer,
          class_weight,
          num_epochs=5,
          valid_period=5,
          output_path=PATH_OUTPUT,
          final=False):

    model.to(DEVICE)

    no_improvement_counter = 0
    best_target_metric = 0
    best_results = None
    train_loss_list = []
    target_metric_list = []
    steps_list = []
    for epoch in range(num_epochs):
        if str(DEVICE).startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        model.train()
        model.zero_grad()

        loss = model.cross_entropy_loss_on_document_nodes(y_train, class_weight=class_weight)

        loss.backward()
        optimizer.step()

        if epoch % valid_period == 0:
            model.eval()
            with torch.no_grad():
                eval_results = evaluate(model, x_valid, y_valid, None, show_result=False)
                if TARGET_METRIC == 'accuracy':
                    target_metric = eval_results[TARGET_METRIC]
                else:
                    target_metric = eval_results["macro avg"][TARGET_METRIC]

            target_metric_list.append(target_metric)
            steps_list.append(epoch)
            train_loss_list.append(float(loss.item()))

            if final:
                save_metrics(os.path.join(output_path, 'induct-gcn-metric.pkl'), train_loss_list, steps_list, target_metric_list)
                metric_value = f", {TARGET_METRIC}: {target_metric:.4}"
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train loss: {train_loss_list[-1]:.4}{metric_value}")

            if target_metric > best_target_metric:
                print(f"\tBest model found ({TARGET_METRIC}: {target_metric:.4})")
                best_target_metric = target_metric
                best_results = eval_results
                no_improvement_counter = 0
                if final:
                    save_checkpoint(os.path.join(output_path, 'induct-gcn-model.pkl'), model, target_metric)
            no_improvement_counter += 1

        if no_improvement_counter > NO_IMPROVEMENT_LIMIT:
            print(f"Early stoping since no improvement in the last {NO_IMPROVEMENT_LIMIT} steps...")
            break

    return best_target_metric, best_results


def evaluate(model, x_dev, y_dev, labels, show_result=True):
    model.eval()
    with torch.no_grad():
        output = model(x_dev)
        y_pred = torch.argmax(output, axis=-1).tolist()

    if show_result:
        print('\nClassification Report:')
        report = classification_report(y_dev, y_pred, target_names=labels, digits=4)
        print(report)
        ConfusionMatrixDisplay.from_predictions(y_dev.tolist(), y_pred, display_labels=labels)
        plt.savefig(os.path.join(PATH_OUTPUT, f'CM-inducT-GCN({EMBEDDING_DIM})S{SEED}.png'))
        plt.show()
    else:
        return classification_report(y_dev, y_pred,
                                     target_names=labels, digits=4,
                                     output_dict=True)


def train_inductgcn(learning_rate, num_steps, save_model=False):
    random.seed(SEED)
    np.random.RandomState(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # seed all gpus

    FEATURE_VECTORIZER_ARGS["vocabulary"] = None
    doc_vectorizer = TfidfVectorizer(**FEATURE_VECTORIZER_ARGS)
    if MIN_FREQ > 1:
        tokenizer = doc_vectorizer.build_analyzer()
        counter = Counter()
        for doc in X_train:
            counter.update(tokenizer(doc))
        FEATURE_VECTORIZER_ARGS["vocabulary"] = [token for token in counter if counter[token] >= MIN_FREQ]
        doc_vectorizer = TfidfVectorizer(**FEATURE_VECTORIZER_ARGS)
    if FEATURE_SELECTION:
        doc_vectorizer.fit(X_train)
        vocab = doc_vectorizer.get_feature_names_out()
        print("Original vocab size:", len(vocab))
        if FEATURE_SIZE == -1:
            estimator = LogisticRegression(class_weight='balanced', dual=False, fit_intercept=True, penalty='none', solver='newton-cg', random_state=SEED, n_jobs=-1)
            selector = SelectFromModel(estimator=estimator).fit(doc_vectorizer.transform(X_train), y_train_list)
        else:
            selector = SelectKBest(f_classif, k=FEATURE_SIZE).fit(doc_vectorizer.transform(X_train), y_train_list)
        support = selector.get_support()
        vocab = [vocab[i] for i in range(len(vocab)) if support[i]]
        print("Vocab size after feature selection:", len(vocab))

        FEATURE_VECTORIZER_ARGS["vocabulary"] = vocab
        doc_vectorizer = TfidfVectorizer(**FEATURE_VECTORIZER_ARGS)

    doc_vectorizer.fit(X_train)  # the vectorizer must be fitted

    print("======================= Creating graph =================================")
    model = InducTGCN(EMBEDDING_DIM, labels, DROPOUT, doc_vectorizer)
    model.build_graph(X_train, window_size=WINDOW_SIZE, verbose=True)

    print("Total number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print("======================= Start training =================================")

    try:
        best_value, best_results = train(model, y_train, X_dev, y_dev, optimizer, class_weight, num_epochs=num_steps, valid_period=VALIDATION_PERIOD, final=save_model)
    except KeyboardInterrupt:
        print("[Keyboard Interrupt]")

    if save_model:
        print("=======================   Evaluation   =================================")

        load_checkpoint(os.path.join(PATH_OUTPUT, 'induct-gcn-model.pkl'), model)

        with open(f"data/{DATASET}_{FEATURE_SIZE}words.txt", "w") as writer:
            H_1 = model.get_H_1(model.Conv_0)
            z_logits = model.node_emb2out(torch.mm(model.A_norm, H_1))
            ix2word = {}
            for word in doc_vectorizer.vocabulary_:
                ix2word[doc_vectorizer.vocabulary_[word]] = word
            for ix in ix2word:
                label_ix = z_logits[ix].argmax().item()
                label = ix2label[label_ix]
                writer.write(f"{ix2word[ix]},{label},{nn.Softmax()(z_logits[ix])[label_ix]}\n")

        evaluate(model, X_dev, y_dev, labels, show_result=True)

        if PATH_TEST:
            print("==================   Evaluation (test set)  ============================")
            print("Test set")
            model.Conv_0_Test = None
            evaluate(model, X_test, y_test, labels, show_result=True)

        if SAVE_MODEL_AS_FINAL:
            print("Saving final model to disk...")
            model.save(X_dev, os.path.join(PATH_OUTPUT_MODEL, f'model_inductgcn[{DATASET}_{FEATURE_SIZE}].pkl'))
            with open(os.path.join(PATH_OUTPUT_MODEL, f'vtzer_inductgcn[{DATASET}_{FEATURE_SIZE}].pkl'), 'wb') as fout:
                pickle.dump(doc_vectorizer, fout)

    return best_value, best_results


if __name__ == "__main__":

    if os.getenv('SGE_TASK_ID') is not None:  # SGE
        task_ix = int(os.getenv('SGE_TASK_ID')) - 1
        print(f"(JOB with index {task_ix} will run task {id})")
    elif os.getenv('SLURM_ARRAY_TASK_ID') is not None:  # Slurm
        task_ix = int(os.getenv('SLURM_ARRAY_TASK_ID')) - int(os.getenv('SLURM_ARRAY_TASK_MIN'))
        print(f"(JOB with index {task_ix} will run task {id})")
    elif args.task_id < 0:
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
    else:
        print("(No task id was provided, using the first one as default)")
        task_ix = 0

    DATASET, FEATURE_SIZE, USE_PAGERANK, FEATURE_SELECTION, STUDY_NAME, description = get_task_info(OPTIONS[task_ix])

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

    if USE_WORDPIECE:
        STEMMER = None
        from transformers import AutoTokenizer
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        def wordpiece_tokenizer(text):
            return bert_tokenizer.convert_ids_to_tokens(bert_tokenizer(text)['input_ids'])[1:-1]

        FEATURE_VECTORIZER_ARGS = {"tokenizer": wordpiece_tokenizer, "stop_words": stopwords.words('english')}

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        DEVICE = torch.device('cpu')
    print("Device:", DEVICE)

    os.makedirs(PATH_OUTPUT, exist_ok=True)
    os.makedirs(PATH_OUTPUT_MODEL, exist_ok=True)

    X_train, y_train, ix2label, label2ix = load_dataset(PATH_TRAIN)
    X_dev, y_dev, _, _ = load_dataset(PATH_DEV, label2ix=label2ix)
    if PATH_TEST:
        X_test, y_test, _, _ = load_dataset(PATH_TEST, label2ix=label2ix)

    y_train_list = y_train.tolist()
    y_train = y_train.to(DEVICE)
    n_labels = len(ix2label)
    labels = [ix2label[ix] for ix in range(n_labels)]
    class_weight = torch.tensor(compute_class_weight(class_weight='balanced', classes=list(range(len(ix2label))), y=y_train_list),
                                dtype=torch.float).to(DEVICE) if CLASS_WEIGHT else None

    best_metric = float("inf") if TARGET_METRIC == "loss" else float("-inf")  # https://github.com/optuna/optuna/issues/2575
    best_results = None

    def objective(trial):
        global best_metric, best_results

        lr = trial.suggest_float("learning_rate", HP_RANGE_LR[0], HP_RANGE_LR[1], log=True)
        num_steps = trial.suggest_int("num_steps", HP_RANGE_STEPS[0], HP_RANGE_STEPS[1], log=True)

        current_metric, current_best_results = train_inductgcn(lr, num_steps, False)
        if (TARGET_METRIC == "loss" and current_metric < best_metric) or \
           (TARGET_METRIC != "loss" and current_metric > best_metric):
            best_metric = current_metric
            best_results = current_best_results

        return best_metric

    study = optuna.create_study(
        storage=OPTUNA_STORAGE,
        pruner=optuna.pruners.NopPruner(),
        study_name=STUDY_NAME,
        direction="minimize" if TARGET_METRIC == "loss" else "maximize",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=HP_N_TRIALS)

    study.set_user_attr("model", "Induct-GCN" + (" + PageRank" if USE_PAGERANK else ''))
    study.set_user_attr("dataset", DATASET)
    study.set_user_attr("features selection", "all" if not FEATURE_SELECTION else ("auto" if FEATURE_SIZE == -1 else f"top-{FEATURE_SIZE}"))
    study.set_user_attr(f"best {TARGET_METRIC}", study.best_value)

    print("BEST RESULTS:")
    print(json.dumps(best_results, indent=2))
