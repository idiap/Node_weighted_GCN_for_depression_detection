# -*- coding: utf-8 -*-
"""
Script to generate the graph plots shown in the paper as well as the
interactive plots.

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
import sys
import torch
import plotly
import networkx as nx
import torch.nn as nn
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from umap import UMAP
from matplotlib.patches import Rectangle
from main import load_dataset, load_induct_gcn_model


# Usage: plots_generator.py [AVEC_16|AVEC_19] [N]

DATASET = sys.argv[1] if len(sys.argv) > 1 else "AVEC_16"  # "AVEC_19"
FEATURE_SIZE = sys.argv[2] if len(sys.argv) > 2 else 250
PLOT_DEV_DOCS = False
PLOT_ZOOM_REGION = [[7, 8.6], [4.8, 5.8]]
PATH_OUTPUT = "plots/"
COLOR_POS_WORD = "#596d8e"
COLOR_POS_DOC = "#7188B0"
COLOR_POS_DOC_TEST = "#37b8db"
COLOR_NEG_WORD = "#b28769"
COLOR_NEG_DOC = "#CD9E7B"
COLOR_NEG_DOC_TEST = "#eb8438"
COLOR_EDGE_WORD = "#ebc9fc"
COLOR_EDGE_DOC = "#dddddd"

PATH_DATASET = f"data/{DATASET}_data/"
PATH_MODEL_INDUCT_GCN = f"model/model_inductgcn[{DATASET}_data_{FEATURE_SIZE}].pkl"
PATH_VTZER_INDUCT_GCN = f"model/vtzer_inductgcn[{DATASET}_data_{FEATURE_SIZE}].pkl"
if len(os.listdir(PATH_DATASET)) > 2:
    PATH_TRAIN = f"{PATH_DATASET}train_all_data.txt"
    PATH_DEV = f"{PATH_DATASET}devel_all_data.txt"
    PATH_TEST = f"{PATH_DATASET}test_all_data.txt"
else:
    PATH_TRAIN = f"{PATH_DATASET}train_all_data.txt"
    PATH_DEV = f"{PATH_DATASET}test_all_data.txt"
    PATH_TEST = ""
OUTPUT_PLOT_FILENAME = f"{DATASET}_plot_umap"


_, y_train, ix2label, label2ix = load_dataset(PATH_TRAIN)
X_dev, y_dev, _, _ = load_dataset(PATH_DEV, label2ix=label2ix)
y_train = y_train.tolist()

gcn = load_induct_gcn_model(PATH_MODEL_INDUCT_GCN, PATH_VTZER_INDUCT_GCN, "cpu")
gcn.eval()

H_1 = gcn.get_H_1(gcn.Conv_0)
z_logits = gcn.node_emb2out(torch.mm(gcn.A_norm, H_1))

ix2word = dict((gcn._vocab[word], word) for word in gcn._vocab)
vocab_size = len(gcn._vocab)
n_nodes = H_1.shape[0]

A_norm = gcn.A_norm.numpy()
pagerank = A_norm[range(A_norm.shape[0]), range(A_norm.shape[1])]
pagerank /= pagerank.max()
A_norm[range(A_norm.shape[0]), range(A_norm.shape[1])] = 0  # removing loops before normalizing
A_norm[:vocab_size, :vocab_size] /= A_norm[:vocab_size, :vocab_size].max()  # normalizing PMI values
A_norm[:vocab_size, vocab_size:] /= A_norm[:vocab_size, vocab_size:].max()  # normalizing TF-IDF values
A_norm[range(A_norm.shape[0]), range(A_norm.shape[1])] = pagerank

words = {"positive": [], "negative": []}
docs = {"positive": [], "negative": []}
for ix in range(n_nodes):
    label_ix = z_logits[ix].argmax().item()
    label = ix2label[label_ix]
    if ix < vocab_size:
        word = ix2word[ix]
        words[label].append((word, nn.Softmax()(z_logits[ix])[label_ix]))
    else:
        ix = ix - vocab_size
        doc = f"doc_{ix}_{ix2label[y_train[ix]]}"
        docs[label].append((doc, nn.Softmax()(z_logits[ix])[label_ix]))

for label in words:
    words[label].sort(key=lambda e: -e[1])
    print(f"=== {label.upper()} ===\n")
    for ix, wv in enumerate(words[label]):
        word, value = wv
        print(f"{ix}. {word} ({value:2%})")

# Dev/test set documents
X_dev_len = len(X_dev)
try:
    H_0_wB = torch.concat([torch.eye(vocab_size, dtype=torch.float),
                           torch.tensor(gcn._doc_vectorizer.transform(X_dev).todense(), dtype=torch.float)])
except AttributeError:
    H_0_wB = torch.concat([torch.eye(vocab_size, dtype=torch.float),
                           torch.tensor(gcn._doc_vectorizer.transform(X_dev), dtype=torch.float)])

A_B = torch.zeros((X_dev_len, vocab_size + X_dev_len), dtype=torch.float, device="cpu")
A_B[:, :vocab_size] = H_0_wB[vocab_size:, :]
A_B[:, vocab_size:] = torch.eye(X_dev_len, dtype=torch.float)

gcn.A_B = A_B.detach()
gcn.Conv_0_Test = torch.mm(gcn.A_B, H_0_wB).cpu().detach()

H_1_B = gcn.get_H_1.cpu()(gcn.Conv_0_Test)
gcn.H_1_words = gcn.H_1_words.cpu()

H_1_wB = torch.concat([gcn.H_1_words[:vocab_size, :], H_1_B])
z_logits_test = gcn.node_emb2out.cpu()(torch.mm(gcn.A_B, H_1_wB))

# GRAPH PLOT
features = torch.concat([H_1, H_1_wB[vocab_size:]]).detach().numpy()
if PLOT_DEV_DOCS:
    probs = nn.Softmax()(torch.concat([z_logits, z_logits_test]).detach())
else:
    probs = nn.Softmax()(z_logits.detach())
    y_dev = []
probs = [probs[ix][label2ix["positive"]].item() for ix in range(probs.shape[0])]
docs_mask = ["word" if ix < vocab_size else ("document" if ix < n_nodes else "document_test") for ix in range(len(probs))]
legends = ["word(D)" if p > .5 else "word(C)" for p in probs[:vocab_size]] + ["document(D)" if y else "document(C)" for y in y_train] + ["document_test(D)" if y else "document_test(C)" for y in y_dev]
color_map = {"word(D)": COLOR_POS_WORD, "word(C)": COLOR_NEG_WORD,
             "document(D)": COLOR_POS_DOC, "document(C)": COLOR_NEG_DOC,
             "document_test(D)": COLOR_POS_DOC_TEST, "document_test(C)": COLOR_NEG_DOC_TEST}
symbol_map_2d = {"word": "triangle-up",
                 "document": "circle"}
symbol_map_3d = {"word": "diamond",
                 "document": "circle"}

size = pagerank.tolist()[:vocab_size] + [0.25] * (len(y_train) + len(y_dev))
labels = [f"{ix2word[ix]}" for ix in range(vocab_size)] + [None] * (len(y_train) + len(y_dev))

umap_2d = UMAP(random_state=0)
umap_2d.fit(features[:n_nodes])
projections_umap = umap_2d.transform(features)[:None if PLOT_DEV_DOCS else n_nodes]

G = nx.Graph()
for node_i_ix, row in enumerate(A_norm):
    for node_j_ix, value in enumerate(row):
        if node_i_ix != node_j_ix and value > .0:
            G.add_edge(node_i_ix, node_j_ix)

pos = projections_umap
nodes_words_pos = [n_id for n_id, n_type in enumerate(docs_mask) if n_type == "word" and "(D)" in legends[n_id]]
nodes_words_neg = [n_id for n_id, n_type in enumerate(docs_mask) if n_type == "word" and "(D)" not in legends[n_id]]
nodes_words = nodes_words_pos + nodes_words_neg
nodes_docs_pos = [n_id for n_id, n_type in enumerate(docs_mask) if n_type == "document" and "(D)" in legends[n_id]]
nodes_docs_neg = [n_id for n_id, n_type in enumerate(docs_mask) if n_type == "document" and "(D)" not in legends[n_id]]
nodes_docs_pos_test = [n_id for n_id, n_type in enumerate(docs_mask) if n_type == "document_test" and "(D)" in legends[n_id]]
nodes_docs_neg_test = [n_id for n_id, n_type in enumerate(docs_mask) if n_type == "document_test" and "(D)" not in legends[n_id]]
nodes_docs = nodes_docs_pos + nodes_docs_neg
edges_words = [(n_s, n_t) for n_s, n_t in G.edges() if n_s in nodes_words and n_t in nodes_words]
edges_docs = [(n_s, n_t) for n_s, n_t in G.edges() if n_s in nodes_docs or n_t in nodes_docs]

nx.draw_networkx_nodes(G, pos, nodelist=nodes_words_pos, node_shape='^',
                       node_size=[20 * A_norm[n_id, n_id] for n_id in nodes_words_pos],
                       margins=0, label="word(D)", node_color=COLOR_POS_WORD)
nx.draw_networkx_nodes(G, pos, nodelist=nodes_words_neg, node_shape='^',
                       node_size=[20 * A_norm[n_id, n_id] for n_id in nodes_words_neg],
                       margins=0, label="word(C)", node_color=COLOR_NEG_WORD)
nx.draw_networkx_nodes(G, pos, nodelist=nodes_docs_pos, node_shape='o', node_size=5,
                       margins=0, label="document(D)", node_color=COLOR_POS_DOC)
nx.draw_networkx_nodes(G, pos, nodelist=nodes_docs_neg, node_shape='o', node_size=5,
                       margins=0, label="document(C)", node_color=COLOR_NEG_DOC)
nx.draw_networkx_nodes(G, pos, nodelist=nodes_docs_pos_test, node_shape='o', node_size=10, linewidths=1, edgecolors='black',
                       margins=0, label="document_test(D)", node_color=COLOR_POS_DOC_TEST)
nx.draw_networkx_nodes(G, pos, nodelist=nodes_docs_neg_test, node_shape='o', node_size=10, linewidths=1, edgecolors='black',
                       margins=0, label="document_test(C)", node_color=COLOR_NEG_DOC_TEST)
nx.draw_networkx_edges(G, pos, edgelist=edges_words, edge_color=COLOR_EDGE_WORD,
                       width=[A_norm[s, t] for s, t in edges_words], alpha=.1)
nx.draw_networkx_edges(G, pos, edgelist=edges_docs, edge_color=COLOR_EDGE_DOC,
                       width=[A_norm[s, t] for s, t in edges_docs], alpha=.1)

plt.legend()
# plt.xlabel("1st dimension of UMAP projection")
# plt.ylabel("2nd dimension of UMAP projection")
ax = plt.gca()

ax.add_patch(Rectangle((PLOT_ZOOM_REGION[0][0], PLOT_ZOOM_REGION[1][0]),
                       PLOT_ZOOM_REGION[0][1] - PLOT_ZOOM_REGION[0][0],
                       PLOT_ZOOM_REGION[1][1] - PLOT_ZOOM_REGION[1][0],
                       color='grey', alpha=.1))
ax.tick_params(
    axis="both",
    which="both",
    bottom=True,
    left=True,
    labelbottom=True,
    labelleft=True,
)
plt.tight_layout()
plt.savefig(PATH_OUTPUT + f'{OUTPUT_PLOT_FILENAME}[{vocab_size}paper].png', dpi=300)
plt.show()

# Zoom in on given region
ax.get_legend().remove()
nx.draw_networkx_labels(G, pos, {ix: word for ix, word in enumerate(labels) if ix < vocab_size},
                        font_size=8, font_color="black",
                        verticalalignment="bottom")
plt.xlim(PLOT_ZOOM_REGION[0])
plt.ylim(PLOT_ZOOM_REGION[1])
plt.tight_layout()
plt.savefig(PATH_OUTPUT + f'{OUTPUT_PLOT_FILENAME}[{vocab_size}paper-zoom].png', dpi=300)
plt.show()

fig_umap = px.scatter(
    projections_umap, x=0, y=1,
    title=f"UMAP - {DATASET}",
    color=legends,
    color_discrete_map=color_map,
    size=size,
    text=labels,
    symbol=docs_mask,
    symbol_map=symbol_map_2d
)
fig_umap.update_layout(dict(plot_bgcolor='white'))
plotly.offline.plot(fig_umap, filename=PATH_OUTPUT + f'{OUTPUT_PLOT_FILENAME}[{vocab_size}word_embeddings].html')

# fig_umap_probs = px.scatter(
#     projections_umap, x=0, y=1,
#     title=f"UMAP (probs) - {DATASET}",
#     color=probs,
#     size=size,
#     text=labels,
#     symbol=docs_mask,
#     symbol_map=symbol_map_2d
# )
# fig_umap_probs.update_layout(dict(plot_bgcolor='white'))
# plotly.offline.plot(fig_umap_probs, filename=PATH_OUTPUT + f'{OUTPUT_PLOT_FILENAME}_probs[{vocab_size}word_embeddings].html')

umap_3d = UMAP(n_components=3, random_state=0)
umap_3d.fit(features[:n_nodes])
projections_umap_3d = umap_3d.transform(features)[:None if PLOT_DEV_DOCS else n_nodes]

fig_umap_3d = px.scatter_3d(
    projections_umap_3d, x=0, y=1, z=2,
    title=f"UMAP 3D - {DATASET}",
    color=legends,
    color_discrete_map=color_map,
    size=size,
    text=labels,
    symbol=docs_mask,
    symbol_map=symbol_map_3d
)
fig_umap_3d.update_layout(scene=dict(
    xaxis=dict(backgroundcolor="white",
               gridcolor="white",
               showbackground=True,
               zerolinecolor="white"),
    yaxis=dict(backgroundcolor="white",
               gridcolor="white",
               showbackground=True,
               zerolinecolor="white"),
    zaxis=dict(backgroundcolor="white",
               gridcolor="white",
               showbackground=True,
               zerolinecolor="white")
))
plotly.offline.plot(fig_umap_3d, filename=PATH_OUTPUT + f'{OUTPUT_PLOT_FILENAME}_3d[{vocab_size}word_embeddings].html')

for node_i_ix, row in enumerate(A_norm):
    x_i, y_i, z_i = projections_umap_3d[node_i_ix]
    x_2d_i, y_2d_i = projections_umap[node_i_ix]

    for node_j_ix, value in enumerate(row):
        if node_i_ix != node_j_ix and value > .0:
            x_j, y_j, z_j = projections_umap_3d[node_j_ix]
            x_2d_j, y_2d_j = projections_umap[node_j_ix]

            fig_umap_3d.add_trace(go.Scatter3d(
                x=[x_i, x_j],
                y=[y_i, y_j],
                z=[z_i, z_j],
                showlegend=False,
                mode="lines",
                opacity=value,
                line=dict(
                    color=COLOR_EDGE_WORD if node_j_ix < vocab_size else COLOR_EDGE_DOC,
                    width=value * 3,
                )))

            fig_umap.add_trace(go.Scatter(
                x=[x_2d_i, x_2d_j],
                y=[y_2d_i, y_2d_j],
                showlegend=False,
                mode="lines",
                opacity=value,
                line=dict(
                    color=COLOR_EDGE_WORD if node_j_ix < vocab_size else COLOR_EDGE_DOC,
                    width=value * 3,
                )))

fig_umap.data = fig_umap.data[::-1]
fig_umap_3d.data = fig_umap_3d.data[::-1]

plotly.offline.plot(fig_umap, filename=PATH_OUTPUT + f'{OUTPUT_PLOT_FILENAME}[graph_{vocab_size}words].html')
plotly.offline.plot(fig_umap_3d, filename=PATH_OUTPUT + f'{OUTPUT_PLOT_FILENAME}_3d[graph_{vocab_size}words].html')
