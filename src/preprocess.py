import re
import os
import pandas as pd

from sklearn import preprocessing
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.scripts.utils import make_embedding

def process(state):
    print("Preprocessing...")
    if (state == 'train'):
        content_src = "./data/train/content.csv" if os.path.exists("./data/train/content.csv") else "../../data/train/content.csv"
        label_src = "./data/train/label.csv" if os.path.exists("./data/train/label.csv") else "../../data/train/label.csv"
        if(os.stat(content_src).st_size == 0 or os.stat(label_src).st_size == 0):
            print("Reading data...")
            src = "./data/train/data_train.txt" if os.path.exists("./data/train/data_train.txt") else "../data/train/data_train.txt"
            with open(src) as f:
                content = f.readlines()
            print("Reading done!")
            regex = re.compile(r"^\S*")

            # %% tach label
            print("Get Label...")
            label = []
            for i in range(0, len(content)):
                topic = regex.search(content[i])
                label.append(topic[0])
                content[i] = content[i].replace(topic[0], "")
            # %% bien label sang dang encoder
            print("Encode Label...")
            le = preprocessing.LabelEncoder()
            le.fit(label)
            total_label = le.transform(label)
            # %%
            content = tokenizer(content)
            df = pd.DataFrame(content, columns=["Content"])
            df["Label"] = total_label
            df["Content"].to_csv(content_src, sep='\t', encoding='utf-8', header=False, index=False)
            df["Label"].to_csv(label_src, sep='\t', encoding='utf-8', header=True, index=False)
        else:
            print("Reading data in CSV...")
            df = pd.DataFrame()
            content = []
            import csv
            with open(content_src, 'r') as csvfile:  # this will close the file automatically.
                reader = csv.reader(csvfile)
                for row in reader:
                    row = ''.join(str(e) for e in row)
                    content.append(row)
            # %
            df["Content"] = content
            df["Label"] = pd.read_csv(label_src, squeeze=True)
            print("Reading done!")
        # %%
        print(".......")
        labels = df["Label"]
        data = df["Content"]
        # %%
        print("Generating data test...")
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=1)
        X_train, X_test = vectorizer_train(x_train, x_test)
        #X_train, X_test = vectorizerWord2Vec(x_train, x_test)
        return X_train, X_test, y_train, y_test, x_test
    else:
        print("test")
        # content_src = "./data/test/content.csv"
        # if (os.stat("./data/test/content.csv").st_size == 0):
        #     print("Reading data...")
        #     if (os.path.exists("./data/test/data_train.txt")):
        #         src = "./data/test/data_test.txt"
        #     else:
        #         src = "../data/test/data_test.txt"
        #     with open(src) as f:
        #         content = f.readlines()
        #     print("Reading done!")
        #     # %%
        #     content = tokenizer(content)
        #     df = pd.DataFrame(content, columns=["Content"])
        #     df["Content"].to_csv(content_src, sep='\t', encoding='utf-8', header=False, index=False)
        # else:
        #     print("Reading data in CSV...")
        #
        #     df = pd.DataFrame()
        #     data = []
        #     import csv
        #     with open(content_src, 'r') as csvfile:  # this will close the file automatically.
        #         reader = csv.reader(csvfile)
        #         for row in reader:
        #             row = ''.join(str(e) for e in row)
        #             data.append(row)
        #     # %
        #     print("Reading done!")
        # # %%
        # print(".......")
        # # %%
        # print("Generating data test...")
        # x_test = data
        # X_test = vectorizer_test(x_test)
        # return X_test


def tokenizer(x_data):
    print("Tokenizer...")
    # %% Tokenizer
    X_data = []
    for x in x_data:
        essay = word_tokenize(x, format="text")
        X_data.append(essay)
    return X_data

def vectorizer_train(X_data, X_test):
    print("Vectorizer...")
    # %% TF-IDF
    tfidf_vect_ngram = TfidfVectorizer(
        analyzer="word", max_features=500000, ngram_range=(1, 3)
    )
    tfidf_vect_ngram.fit(X_data)
    X_data_tfidf_ngram = tfidf_vect_ngram.transform(X_data)
    X_test_tfidf_ngram = tfidf_vect_ngram.transform(X_test)
    return X_data_tfidf_ngram, X_test_tfidf_ngram

# def vectorizer_test(X_data):
#     print("Vectorizer...")
#     # %% TF-IDF
#     tfidf_vect_ngram = TfidfVectorizer(
#         analyzer="word", max_features=500000, ngram_range=(1, 3)
#     )
#     tfidf_vect_ngram.fit(X_data)
#     X_data_tfidf_ngram = tfidf_vect_ngram.transform(X_data)
#     return X_data_tfidf_ngram

def vectorizerWord2Vec(X_data, X_test):
    embedding_path = "./embeddings/smallFasttext.vi.vec"
    X_data_embed_size, X_data_word_map, X_dataembedding_mat = make_embedding(X_data, embedding_path, 500000)
    X_test_embed_size, X_test_word_map, X_test_embedding_mat_X = make_embedding(X_test, embedding_path, 500000)
    print("Word2Vec...")
    print(X_test_embed_size)
    print(X_test_word_map)
    print(X_test_embedding_mat_X)
    return X_dataembedding_mat, X_test_embedding_mat_X

