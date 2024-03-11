# Node-weighted Graph Convolutional Network for Depression Detection in Transcribed Clinical Interviews

### Authors: 

#### *Sergio Burdisso, Esaú Villatoro-Tello, Srikanth Madikeri, Petr Motlicek*

###### Paper accepted at [INTERSPEECH 2023 Conference](https://interspeech2023.org/) ([ARXIV Version](https://arxiv.org/abs/2307.00920)).


## Abstract

> We propose a simple approach for weighting self-connecting edges in a Graph Convolutional Network (GCN) and show its impact on depression detection from transcribed clinical interviews. To this end, we use a GCN for modeling non-consecutive and long-distance semantics to classify the transcriptions into depressed or control subjects. The proposed method aims to mitigate the limiting assumptions of locality and the equal importance of self-connections vs. edges to neighboring nodes in GCNs, while preserving attractive features such as low computational cost, data agnostic, and interpretability capabilities. We perform an exhaustive evaluation in two benchmark datasets. Results show that our approach consistently outperforms the vanilla GCN model as well as previously reported results, achieving an F1=0.84 on both datasets. Finally, a qualitative analysis illustrates the interpretability capabilities of the proposed approach and its alignment with previous findings in psychology.

---
## :computer: How to use?

**IMPORTANT:** Be aware that for running the experiments you need to download the corresponding datasets and for generating the LIWC plots also the LIWC dictionary. Read the README files in [`data/`](data/) and [`plots/LIWC_plot`](plots/LIWC_plot) for more info.

First, make sure your environment and dependencies are all set up:

```bash
$ conda env create -f conda_env.yaml
$ conda activate gcndepression
(gcndepression)$ pip install -r requirements.txt
```

The script `main.py` contains the implementation of the InducT-GCN and the experimentation. To relpicate paper experiments, call the script first with no arguments to get the list of all possible configuration to run, for example:

```bash
$ python main.py

There's a total of 42 task options. List by task id:
  1. Dataset=AVEC_16 (eval on devset); Features=all; Pagerank=True;
  2. Dataset=AVEC_16 (eval on devset); Features=all; Pagerank=False;
  3. Dataset=AVEC_16 (eval on devset); Features=auto; Pagerank=True;
  4. Dataset=AVEC_16 (eval on devset); Features=auto; Pagerank=False;
  5. Dataset=AVEC_16 (eval on devset); Features=top-100; Pagerank=True;
  6. Dataset=AVEC_16 (eval on devset); Features=top-100; Pagerank=False;
  ...
```

Then, select one option and call the script passing the index to the `-i` argument. For instance, suppose we want to run the experiment number 3 above, then we call main script simply as follows:

```bash
$ python main.py -i 3
```

**TIP:** you can use this indices to call multiple instances in parallel, each with different `-i` value.

For replicating baselines experiments, you can follow the exact same process but with the scripts `baselines-basic.py` and `baselines-bert.py` for replicating, respectively, the baselines with simple/classic models and with bert-based models. For instance:

```bash
$ python baselines-bert.py

There's a total of 36 task options. List by task id:
  1. Dataset=AVEC_16 (eval on devset); Model: bert-base-cased; Finetuned=True;
  2. Dataset=AVEC_16 (eval on devset); Model: bert-base-cased; Finetuned=False;
  3. Dataset=AVEC_16 (eval on devset); Model: bert-base-uncased; Finetuned=True;
  4. Dataset=AVEC_16 (eval on devset); Model: bert-base-uncased; Finetuned=False;
  5. Dataset=AVEC_16 (eval on devset); Model: bert-large-cased; Finetuned=True;
  6. Dataset=AVEC_16 (eval on devset); Model: bert-large-cased; Finetuned=False;
  ...
```

Then, let's say I want to replicate option 6:

```bash
$ python baselines-bert.py -i 6
```
---
### Plots

Plots shown in the paper as well as interactive plots are located in the `plots` folder. The interactive plots are 4 for each dataset (AVEC_16 and AVEC_19):
  - `AVEC_XX_data_plot_umap_3d[graph_250words].html` - 3D plot with the learned graph and embeddings.
  - `AVEC_XX_data_plot_umap_3d[250word_embeddings].html` - 3D plot with the learned embeddings.
  - `AVEC_XX_data_plot_umap[graph_250words].html` - 2D plot with the learned graph and embeddings.
  - `AVEC_XX_data_plot_umap[250word_embeddings].html` - 2D plot with the learned embeddings.
Where XX is either 16 or 19.

To re-generated the plots you can use the script:

```bash
$ python plots_generator.py
```

To generate the plot with distribution of LIWC categories learned by the model you can run the script located in the `plots/LIWC_plot` as follow:

```bash
$ cd plots/LIWC_plot
$ python plot_generator.py
```

---
### Optuna Results Reported in the paper

This repo also contains the original optuna sqlite database files (`db_baselines.sqlite3`, `db_inductgcn.sqlite3`) with results reported in the paper after optimization process, to read then simply call the dashboard with either of the files as follows:

```bash
$ optuna-dashboard sqlite:///db_inductgcn.sqlite3
```
