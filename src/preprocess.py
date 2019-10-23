import re
import os
import pandas as pd

from sklearn import preprocessing
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.scripts.utils import make_embedding

if(os.path.exists("./data/data_train.txt")):
    src = "./data/data_train.txt"
else:
    src = "../data/data_train.txt"
with open(src) as f:
    content = f.readlines()

def process(state):
    print("Preprocessing...")
    regex = re.compile(r"^\S*")
    result = regex.search(content[0])

    # %% tach label
    label = []
    for i in range(0, len(content)):
        topic = regex.search(content[i])
        label.append(topic[0])
        content[i] = content[i].replace(topic[0], "")
    # %% bien label sang dang encoder
    le = preprocessing.LabelEncoder()
    le.fit(label)
    total_label = le.transform(label)
    # %%
    df = pd.DataFrame(content, columns=["Content"])
    df["Label"] = total_label
    # %
    labels = df["Label"]
    data = df["Content"]

    if (state == 'train'):
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=1)
    else:
        x_train, x_test, y_train, y_test = data

    X_train, X_test = tokenizer(x_train, x_test)
    return X_train, X_test, y_train, y_test

def tokenizer(x_train, x_test):
    # %% Tokenizer
    X_data = []
    for x in x_train:
        essay = word_tokenize(x, format="text")
        X_data.append(essay)
    X_test = []
    for x in x_test:
        essay = word_tokenize(x, format="text")
        X_test.append(essay)
    X_data, X_test =vectorizer(X_data, X_test)
    # X_data, X_test = vectorizerWord2Vec(X_data, X_test)
    print("Tokenize...")
    return X_data, X_test

def vectorizer(X_data, X_test):
    # %% TF-IDF
    tfidf_vect_ngram = TfidfVectorizer(
        analyzer="word", max_features=500000, ngram_range=(1, 3)
    )
    tfidf_vect_ngram.fit(X_data)
    X_data_tfidf_ngram = tfidf_vect_ngram.transform(X_data)
    X_test_tfidf_ngram = tfidf_vect_ngram.transform(X_test)
    print("Vectorizer...")
    # print(X_data_tfidf_ngram[0])
    return X_data_tfidf_ngram, X_test_tfidf_ngram

def vectorizerWord2Vec(X_data, X_test):
    embedding_path = "./embeddings/baomoi.model.bin"
    X_data_embed_size, X_data_word_map, X_dataembedding_mat = make_embedding(X_data, embedding_path, 500000)
    X_test_embed_size, X_test_word_map, X_test_embedding_mat_X = make_embedding(X_test, embedding_path, 500000)
    print("Word2Vec...")
    print(X_test_embed_size)
    print(X_test_word_map)
    print(X_test_embedding_mat_X)
