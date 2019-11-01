# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from keras.layers import Dropout, Dense
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
import pandas as pd
import re
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from pyvi.ViTokenizer import ViTokenizer
from pyvi.ViPosTagger import ViPosTagger
from sklearn.feature_extraction.text import CountVectorizer

import pickle

from src.model.DNN_model import DNNModel

SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''

# with open('src/testing/data.txt', encoding="utf8") as f:
with open('topic_detection_train.v1.0.txt', encoding="utf8") as f:
    content = f.readlines()
regex = re.compile(r'^\S*')
# result = regex.search(content[0])
# print(result)
# print(content[0])
# print(ViTokenizer.tokenize(content[0]))

content = content[:100]
# content = content[:1000]
# content = content[1000:2000]
print(len(content))

label = []
for i in range(0, len(content)):
    topic = regex.search(content[i])
    label.append(topic[0])
    temp_str = content[i].replace(topic[0], '')
    temp_str = ' '.join([x.strip(SPECIAL_CHARACTER).lower() for x in temp_str.split()])
    temp_str = ViTokenizer.tokenize(temp_str)
    content[i] = temp_str
    # content[i] = content[i].replace(topic[0], '')
    # content[i] = ViTokenizer.tokenize(content[i])

# print(label)
encoder = LabelEncoder()
label = encoder.fit_transform(label)
# print(label)

# đổ dữ liệu vào data frame
df = pd.DataFrame(content, columns=['sentence'])
df['label'] = label
# print(df)
labels = df['label'].values
# print(labels)
sentences = df['sentence'].values
# print(sentences)
# print(df.iloc)


# chia tập dữ liệu thành bộ train và bộ test - để có dữ liệu kiểm tra mô hình chạy được không
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.20, random_state=1)

# x_train = x_train.values.reshape(-1, 1)

# count_vectorizer = CountVectorizer()
# count_vectorizer.fit(x_train)
# X_train = count_vectorizer.transform(x_train)
# X_test = count_vectorizer.transform(x_test)

tfidf_vectornizer = TfidfVectorizer(analyzer="word", max_features=50000)
tfidf_vectornizer.fit(x_train)
X_train = tfidf_vectornizer.transform(x_train)
X_test = tfidf_vectornizer.transform(x_test)

# from keras.preprocessing.text import Tokenizer
#
# tokenizer = Tokenizer(num_words=100000)
# tokenizer.fit_on_texts(x_train)
#
# X_train = tokenizer.texts_to_sequences(x_train)
# X_test = tokenizer.texts_to_sequences(x_test)
#
# vocab_size = len(tokenizer.word_index) + 1
#
#
# from keras.preprocessing.sequence import pad_sequences
#
# maxlen = 400
#
# X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
# X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


input_dim = X_train.shape[1]  # Number of features
# print(X_train.shape)


def train(train_model):
    history = train_model.fit(X_train, y_train,
                              epochs=20,
                              verbose=1,
                              validation_data=(X_test, y_test),
                              batch_size=5)

    loss, accuracy = train_model.evaluate(X_train, y_train, verbose=1)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = train_model.evaluate(X_test, y_test, verbose=1)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    # filename = "DNN_model_training_1000"
    # pickle.dump(train_model, open(filename, 'wb'))


def check():
    # file_path = "src/testing/DNN_model_training_1000"
    file_path = "DNN_model_training_1000"
    custom_model = pickle.load(open(file_path, 'rb'))

    # loss, accuracy = custom_model.evaluate(X_train, y_train, verbose=1)
    # print("Training Accuracy: {:.4f}".format(accuracy))
    # loss, accuracy = custom_model.evaluate(X_test, y_test, verbose=1)
    # print("Testing Accuracy:  {:.4f}".format(accuracy))

    # predictions = custom_model.predict(X_train)
    # print(predictions)
    report = classification_report(y_test, custom_model.predict(X_test))
    print(report)


model = DNNModel(input_dim).get_model()


# check()

train(model)