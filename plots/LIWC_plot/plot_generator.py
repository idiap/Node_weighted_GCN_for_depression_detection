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

@author: Esaú Villatoro-Tello (esau.villatoro@idiap.ch)
"""
import liwc
import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


avec_19_word_list = 'AVEC_19_data_250words.txt'
avec_16_word_list = 'AVEC_16_data_250words.txt'
dictionary = 'LIWC2007_English131104.dic'  # Download you're own

def read_dictionary(fname):
    parse, category_names = liwc.load_token_parser(fname)
    return parse, category_names


def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)


def load_word_list(fname):
    df = pd.read_csv(fname, sep=',', names=['word','category','prob'])
    return df

def normalize_counter(counter_obj):
    total = sum(counter_obj.values(), 0.0)
    for key in counter_obj:
        counter_obj[key] /= total
    return counter_obj

def normalize_dataframe(df):
    total_pos = sum(df['[D]epress'].to_list())
    total_neg = sum(df['[C]ontrol'].to_list())
    print(df)
    for index,row in df.iterrows():
        df.loc[index,'[D]epress'] = int(row['[D]epress'])/total_pos
        df.loc[index,'[C]ontrol'] = int(row['[C]ontrol'])/total_neg
    print(df)
    return df

def plot_word_categories(c1, c2, c3, c4):
    k1 = c1.keys()
    k2 = c2.keys()
    k3 = c3.keys()
    k4 = c4.keys()

    keys = k1 & k2 & k3 & k4
    words=[]
    freq1=[] #DAIC - POS
    freq2=[] #EDAIC - POS
    freq3=[] #DAIC - NEG
    freq4=[] #EAIC - NEG
    for k in keys:
        words.append(k)
        freq1.append(c1[k])
        freq2.append(c2[k])
        freq3.append(c3[k])
        freq4.append(c4[k])


    df = pd.DataFrame(list(zip(words, freq1, freq2, freq3, freq4)), columns =['LIWC Categories', 'DAIC-WOZ [D]', 'EDAIC [D]','DAIC-WOZ [C]', 'EDAIC [C]'])
    print(df)
    df = df.sort_values(by=['DAIC-WOZ [D]', 'EDAIC [D]'], ascending=False)
    #set seaborn plotting aesthetics
    sns.set(style='white')

    #create grouped bar chart
    fig, ax1 = plt.subplots(figsize=(10, 10))
    tidy = df.melt(id_vars='LIWC Categories', var_name = 'Partition', value_name='Freq(%)') #.rename(columns=str.title)
    print (tidy)
    sns.barplot(x='LIWC Categories', y='Freq(%)', hue='Partition', data=tidy, ax=ax1)
    sns.despine(fig)
    plt.show()

def plot_accumulated_word_categories(c1, c2, c3, c4):
    k1 = c1.keys()
    k2 = c2.keys()
    k3 = c3.keys()
    k4 = c4.keys()

    keys = k1 & k2 & k3 & k4
    words=[]
    freq1=[] #DAIC + EDAIC - POS
    freq2=[] #DAIC + EDAIC - NEG
    for k in keys:
        words.append(k)
        freq1.append(c1[k]+c2[k])
        freq2.append(c3[k]+c4[k])


    df = pd.DataFrame(list(zip(words, freq1, freq2)), columns =['LIWC Categories', '[D]epress', '[C]ontrol'])
    print(df)
    df = df.sort_values(by=['[D]epress', '[C]ontrol'], ascending=False)
    df = normalize_dataframe(df)
    #set seaborn plotting aesthetics
    sns.set(style='white')
    #increase font size of all elements
    sns.set(font_scale=1.5)


    #create grouped bar chart
    fig, ax1 = plt.subplots(figsize=(14, 9))
    tidy = df.melt(id_vars='LIWC Categories', var_name = 'Category', value_name='Freq (%)') #.rename(columns=str.title)
    print (tidy)
    ax = sns.barplot(x='LIWC Categories', y='Freq (%)', hue='Category', data=tidy)#, ax=ax1)
    # Define some hatches
    hatches = ['x', 'o']

    # Loop over the bars
    # Loop over the bars
    for bars, hatch in zip(ax.containers, hatches):
        # Set a different hatch for each group of bars
        for bar in bars:
            bar.set_hatch(hatch)
    ax.legend(title='Class')
    sns.despine(fig)
    # df.plot(x="word", y=["DAIC-WOZ [D]", "EDAIC [D]", "DAIC-WOZ [C]", "EDAIC [C]"], kind="bar")
    plt.title("Main LIWC categories present in the GCN")
    plt.xlabel("LIWC categories")
    # plt.ylabel("Freq (%)")
    plt.show()


if __name__ == "__main__":
    print(f'Working with file: {avec_19_word_list}')
    print(f'Working with file: {avec_16_word_list}')
    print(f'Loading LIWC dictionary: {dictionary}')
    
    parse, category_names = read_dictionary(dictionary)

    avec_19_words = load_word_list(avec_19_word_list)
    avec_16_words = load_word_list(avec_16_word_list)

    #print(df_words.head()) 
    positive_words19 = avec_19_words.loc[avec_19_words['category']=='positive'].word.to_list()
    negative_words19 = avec_19_words.loc[avec_19_words['category']=='negative'].word.to_list()
    pos_counts19 = Counter(category for token in positive_words19 for category in parse(token))
    #pos_counts19 = normalize_counter(pos_counts19)
    neg_counts19 = Counter(category for token in negative_words19 for category in parse(token))
    #neg_counts19 = normalize_counter(neg_counts19)
    print(pos_counts19)
    print(neg_counts19)
    print('\n')

    positive_words16 = avec_16_words.loc[avec_16_words['category']=='positive'].word.to_list()
    negative_words16 = avec_16_words.loc[avec_16_words['category']=='negative'].word.to_list()
    pos_counts16 = Counter(category for token in positive_words16 for category in parse(token))
    #pos_counts16 = normalize_counter(pos_counts16)
    neg_counts16 = Counter(category for token in negative_words16 for category in parse(token))
    #neg_counts16 = normalize_counter(neg_counts16)
    print(pos_counts16)
    print(neg_counts16)

    plot_accumulated_word_categories(pos_counts16,pos_counts19, neg_counts16, neg_counts19)
    print(':)')

