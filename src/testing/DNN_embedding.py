# -*- coding: utf-8 -*-
from keras import Input, Model
from sklearn.datasets import fetch_20newsgroups
from keras.layers import Dropout, Dense, Embedding, LSTM
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
import pandas as pd
import re
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from pyvi.ViTokenizer import ViTokenizer
from pyvi.ViPosTagger import ViPosTagger
from sklearn.feature_extraction.text import CountVectorizer

import pickle

from src.model.DNN_model import DNNModel

SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''


def dump_to_file(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def load_obj_from_file(path):
    obj = pickle.load(open(path, 'rb'))
    return obj


# with open('src/testing/data.txt', encoding="utf8") as f:
with open('topic_detection_train.v1.0.txt', encoding="utf8") as f:
    content = f.readlines()
regex = re.compile(r'^\S*')
# result = regex.search(content[0])
# print(result)
# print(content[0])
# print(ViTokenizer.tokenize(content[0]))

content = content[:1000]
# content = content[:1000]
# content = content[2000:3000]
print(len(content))

label = []
for i in range(0, len(content)):
    topic = regex.search(content[i])
    label.append(topic[0])
    temp_str = content[i].replace(topic[0], '')
    temp_str = ' '.join([x.strip(SPECIAL_CHARACTER).lower() for x in temp_str.split()])
    temp_str = ViTokenizer.tokenize(temp_str)
    content[i] = temp_str

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


tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x_train)

X_train = tokenizer.texts_to_sequences(x_train)
X_test = tokenizer.texts_to_sequences(x_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 400

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


input_dim = X_train.shape[1]  # Number of features
# print(X_train.shape)

embedding_matrix = np.zeros((vocab_size, 100))
deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(6, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])


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

    filename = "DNN_model_embedding_training_1000"
    dump_to_file(train_model, filename)


def check():
    # file_path = "src/testing/DNN_model_training_1000"
    file_path = "DNN_model_embedding_training_1000"
    custom_model = load_obj_from_file(file_path)

    loss, accuracy = custom_model.evaluate(X_train, y_train, verbose=1)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = custom_model.evaluate(X_test, y_test, verbose=1)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    # predictions = custom_model.predict(X_train)
    # print(predictions)
    # report = classification_report(y_test, custom_model.predict(X_test))
    # print(report)


model = DNNModel(input_dim).get_model()


# check()

train(model)
