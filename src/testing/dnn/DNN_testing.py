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

import keras

from src.model.DNN_model import DNNModel

SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''

test_path = "resources/topic_detection_test.v1.0.txt"
model_path = "src/testing/dnn/DNN_model_training_16000"
result_file = "src/testing/dnn/dnn_result.txt"
labels_encoder_path = "src/testing/dnn/label_encoder.pk"


def load_obj_from_file(path):
    obj = pickle.load(open(path, 'rb'))
    return obj


# with open('src/testing/data.txt', encoding="utf8") as f:
with open(test_path, encoding="utf8") as f:
    content = f.readlines()

# content = content[:100]
print(len(content))

for i in range(0, len(content)):
    temp_str = content[i]
    temp_str = ' '.join([x.strip(SPECIAL_CHARACTER).lower() for x in temp_str.split()])
    temp_str = ViTokenizer.tokenize(temp_str)
    content[i] = temp_str

# đổ dữ liệu vào data frame
df = pd.DataFrame(content, columns=['sentence'])

sentences = df['sentence'].values
# print(sentences)
# print(df.iloc)

tfidf_vectornizer = load_obj_from_file("tfidf_full_vocab.pk")

X_test = tfidf_vectornizer.transform(sentences)

custom_model = load_obj_from_file(model_path)

# predicts = custom_model.evaluate(X_train, y_train, verbose=1)
predicts = custom_model.predict(X_test, verbose=1, use_multiprocessing=True)
labels_result = predicts.argmax(axis=-1)
labels_encoder = load_obj_from_file(labels_encoder_path)
labels_result = labels_encoder.inverse_transform(labels_result)

with open(result_file, 'w') as f:
    for item in labels_result:
        f.write("%s\n" % item)

print(labels_result)

# predictions = custom_model.predict(X_train)
# print(predictions)
# report = classification_report(y_test, custom_model.predict(X_test))
# print(report)
